import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image as PILImage
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from utils.quantize.lookupFree import LFQ
from utils.hypformer_multimodality import HypFormer
from utils.improved_model import Encoder
from utils.manifolds.hyp_layer import Optimizer
import traceback
from huggingface_hub import hf_hub_download

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronize CUDA execution for easier debugging


# ===================== Special Token Constants ======================
IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"

# ===================== Argument Parsing ======================
parser = argparse.ArgumentParser(description="Train MultiModal HypFormer")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--image_size', type=int, default=224, help='Resize images to this size')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--vocab_size', type=int, default=2**14, help='Text vocabulary size')
parser.add_argument('--image_vocab_size', type=int, default=2**16, help='Image vocabulary size')
parser.add_argument('--hidden_channels', type=int, default=768, help='Hidden channels for HypFormer')
parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
parser.add_argument('--num_layers', type=int, default=10, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--hyp_lr', type=float, default=2e-5, help='Hyperbolic learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Euclidean optimizer type')
parser.add_argument('--hyp_optimizer_type', type=str, default='radam', help='Hyperbolic optimizer type')
parser.add_argument('--hyp_weight_decay', type=float, default=1e-4, help='Hyperbolic weight decay')
parser.add_argument('--img_loss_weight', type=float, default=1.0, help='Image loss weight')
args = parser.parse_args()

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# ===================== Data Processing Utilities ======================
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Text tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ===================== Image Encoder and Quantizer ======================
# CNN encoder configuration
ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 16,
    "ch_mult": (1, 2, 2, 4),
    "resolution": args.image_size,
    "double_z": False,
}

cnn_encoder = Encoder(**ddconfig).to(device).eval()

# LFQ quantizer
lfq = LFQ(codebook_size=args.image_vocab_size, dim=16).to(device)
ckpt_path = hf_hub_download(repo_id="TencentARC/Open-MAGVIT2-Tokenizer-262144-Video", filename="video_128_262144.ckpt")
ckpt = torch.load(ckpt_path, map_location=device)
if "state_dict" in ckpt:
    lfq.load_state_dict(ckpt["state_dict"], strict=False)
elif "codebook" in ckpt:
    lfq.register_buffer("codebook", ckpt["codebook"])
lfq.eval()

# ===================== Dataset Definition ======================
print("Loading COCO dataset using COCO API...")

try:
    from pycocotools.coco import COCO
    from PIL import Image
    
    # Set COCO dataset path — adjust based on your local download location
    data_dir = 'coco_dataset'
    train_annotations = os.path.join(data_dir, 'annotations/captions_train2017.json')
    val_annotations = os.path.join(data_dir, 'annotations/captions_val2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')
    val_image_dir = os.path.join(data_dir, 'val2017')
    
    # Check if dataset files exist
    if not os.path.exists(train_annotations) or not os.path.exists(train_image_dir):
        raise FileNotFoundError(f"COCO dataset files not found. Please make sure the dataset is downloaded into {data_dir}")
    
    # Load COCO API
    print("Loading COCO training annotations...")
    train_coco = COCO(train_annotations)
    print("Loading COCO validation annotations...")
    val_coco = COCO(val_annotations)
    
    # Get all image IDs
    train_ids = list(train_coco.imgs.keys())
    val_ids = list(val_coco.imgs.keys())
    
    # Split validation set into validation & test sets
    random.seed(args.seed)
    random.shuffle(val_ids)
    val_split = int(len(val_ids) * 0.5)
    new_val_ids = val_ids[:val_split]
    test_ids = val_ids[val_split:]
    
    print(f"COCO dataset loaded successfully! "
          f"Training set: {len(train_ids)} images, "
          f"Validation set: {len(new_val_ids)} images, "
          f"Test set: {len(test_ids)} images")
    
    # Create COCO Dataset class
    class COCODataset(Dataset):
        def __init__(self, coco, img_ids, img_dir, transform=None, max_length=128):
            self.coco = coco
            self.img_ids = img_ids
            self.img_dir = img_dir
            self.transform = transform
            self.max_length = max_length
            
        def __len__(self):
            return len(self.img_ids)
        
        def __getitem__(self, idx):
            # Get image ID and image path
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                # Create a blank fallback image
                image = Image.new('RGB', (args.image_size, args.image_size), color='black')
            
            # Get image captions
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Randomly select one caption
            if anns and 'caption' in anns[0]:
                caption = random.choice([ann['caption'] for ann in anns])
            else:
                caption = "No caption available"
            
            # Apply image transformations
            if self.transform:
                image = self.transform(image)
            
            # Encode caption text
            encoding = tokenizer(
                caption,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                "text_ids": encoding['input_ids'].squeeze(0),
                "image": image,
                "caption": caption,
                "image_id": img_id
            }
    
    # Create datasets
    train_dataset = COCODataset(train_coco, train_ids, train_image_dir, transform=transform)
    val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, transform=transform)
    test_dataset = COCODataset(val_coco, test_ids, val_image_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,  # You can increase this if your system supports it
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    print(f"Data loaders created successfully! Batch size: {args.batch_size}")

except Exception as e:
    print(f"Failed to load COCO dataset: {e}")
    traceback.print_exc()  # Print detailed error information
    
    # Fallback to synthetic dataset if COCO fails
    print("Unable to load COCO dataset, creating synthetic dataset for testing...")
    
    # ... Keep your original synthetic dataset code here ...

# ===================== Model Definition ======================
class Args:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = "hyp"
        self.device = device
        self.add_positional_encoding = True
        self.attention_type = "full"
        self.power_k = 2
        self.trans_heads_concat = False

hyp_args = Args()


class MultimodalHypFormer(nn.Module):
    def __init__(self, text_vocab_size, image_vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        
        # Add special tokens and extend vocabulary
        special_tokens = {"additional_special_tokens": [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")

        # Initialize text and image embeddings
        self.text_embedding = nn.Embedding(len(tokenizer), embed_dim)  # Use updated vocabulary size
        self.image_embedding = nn.Embedding(image_vocab_size, embed_dim)
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0 = text, 1 = image

        # Initialize embeddings for new special tokens
        if num_added > 0:
            with torch.no_grad():
                input_embeddings = self.text_embedding.weight.data
                input_embeddings_avg = input_embeddings[:-num_added].mean(dim=0, keepdim=True)
                input_embeddings[-num_added:] = input_embeddings_avg

        # Special token IDs
        self.img_start_token_id = len(tokenizer) - 2  # Image start token is second last in vocab
        self.img_end_token_id = len(tokenizer) - 1    # Image end token is last in vocab
        
        # Core HypFormer model
        self.hypformer = HypFormer(
            in_channels=embed_dim,
            hidden_channels=hidden_dim,
            trans_num_layers=num_layers,
            trans_num_heads=num_heads,
            trans_dropout=dropout,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            out_channels=None,  # Not used in multimodal mode
            multimodal=True,  # Enable multimodal processing
            text_vocab_size=len(tokenizer),
            image_vocab_size=image_vocab_size,
            args=hyp_args
        )
        
                # Image token predictor (for image generation tasks)
        #self.img_predictor = nn.Linear(embed_dim, image_vocab_size)
        #self.text_output_head = nn.Linear(hidden_dim, len(tokenizer))
        
        # Modality type records
        self.register_buffer("text_token_type", torch.zeros(1, dtype=torch.long))
        self.register_buffer("image_token_type", torch.ones(1, dtype=torch.long))
        print(f"Special token IDs: Image start = {self.img_start_token_id}, Image end = {self.img_end_token_id}")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
         

    def generate_image_from_text(self, text):
        """Generate an image from a text description"""
        # Encode the input text
        encoding = tokenizer(
                text, 
                return_tensors="pt"
            ).to(next(self.parameters()).device)
            
        # Initialize image tokens (can be random or start with a special token)
        batch_size = encoding["input_ids"].shape[0]
        image_shape = (batch_size, 256)  # Adjust according to your requirements
        image_tokens = torch.zeros(image_shape, dtype=torch.long, device=next(self.parameters()).device)
            
        # Use the model to generate image tokens
        with torch.no_grad():
            text_logits, img_logits, _, _ = self(encoding["input_ids"], image_tokens)
            if img_logits is not None:
                # Take the most probable tokens from the logits
                pred_tokens = torch.argmax(img_logits, dim=-1)
                    
                # Use LFQ and CNN decoder to convert image tokens back into an image
                # You will need to implement this based on your specific setup
                image_tokens = pred_tokens.reshape(batch_size, -1)
                # Implement the token-to-image conversion process here
                    
                return image_tokens, img_logits
        
        return None, None

    def process_batch(self, text_ids, image_tokens):
        """VILA-style processing: Embed first, then concatenate"""
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        # Store all pre-processed embeddings, masks, and token types
        all_embeddings = []
        all_masks = []
        all_types = []  # For modality type embeddings
        #all_positions = []  # Optional: Position IDs for each sample

        for i in range(batch_size):
            curr_text = text_ids[i]
            curr_img_tokens = image_tokens[i]
            
            # 1. Embed text tokens directly (safe)
            text_embeddings = self.text_embedding(curr_text)
            text_mask = torch.ones(text_embeddings.size(0), dtype=torch.bool, device=device)
            text_types = torch.zeros(text_embeddings.size(0), dtype=torch.long, device=device)
            
            # 2. Embed the <img> start token
            img_start_embed = self.text_embedding(torch.tensor([self.img_start_token_id], device=device))
            img_start_mask = torch.ones(1, dtype=torch.bool, device=device)
            img_start_type = torch.zeros(1, dtype=torch.long, device=device)
            
            # 3. Embed image tokens (use image embedding layer)
            img_embeddings = self.image_embedding(curr_img_tokens)
            img_mask = torch.ones(img_embeddings.size(0), dtype=torch.bool, device=device)
            img_types = torch.ones(img_embeddings.size(0), dtype=torch.long, device=device)
            
            # 4. Embed the </img> end token
            img_end_embed = self.text_embedding(torch.tensor([self.img_end_token_id], device=device))
            img_end_mask = torch.ones(1, dtype=torch.bool, device=device)
            img_end_type = torch.zeros(1, dtype=torch.long, device=device)
            
            # Concatenate everything: text + <img> + image tokens + </img>
            sample_embeds = torch.cat([text_embeddings, img_start_embed, img_embeddings, img_end_embed], dim=0)
            sample_mask = torch.cat([text_mask, img_start_mask, img_mask, img_end_mask], dim=0)
            sample_types = torch.cat([text_types, img_start_type, img_types, img_end_type], dim=0)

            text_len = text_embeddings.size(0)
            img_len = img_embeddings.size(0)
            #positions = torch.arange(text_len + img_len + 2, device=device)  # +2 because of <img> and </img> tokens
            
            all_embeddings.append(sample_embeds)
            all_masks.append(sample_mask)
            all_types.append(sample_types)
            #all_positions.append(positions)
        
        # Pad sequences to the same length across the batch
        max_len = max([emb.size(0) for emb in all_embeddings])
        padded_embeddings = []
        padded_masks = []
        padded_types = []
        #padded_positions = []
        
        for emb, mask, types in zip(all_embeddings, all_masks, all_types):
            # Calculate required padding length
            pad_len = max_len - emb.size(0)
            
            if pad_len > 0:
                # Create padding embeddings
                padding_emb = torch.zeros((pad_len, emb.size(1)), device=device)
                padded_emb = torch.cat([emb, padding_emb], dim=0)
                
                # Pad masks and types
                padding_mask = torch.zeros(pad_len, dtype=torch.bool, device=device)
                padded_mask = torch.cat([mask, padding_mask], dim=0)
                
                padding_type = torch.zeros(pad_len, dtype=torch.long, device=device)
                padded_type = torch.cat([types, padding_type], dim=0)

                #padding_pos = torch.zeros(pad_len, dtype=torch.long, device=device)
                #padded_pos = torch.cat([pos, padding_pos], dim=0)
            else:
                padded_emb = emb
                padded_mask = mask
                padded_type = types
                #padded_pos = pos
            
            padded_embeddings.append(padded_emb)
            padded_masks.append(padded_mask)
            padded_types.append(padded_type)
            #padded_positions.append(padded_pos)
        
               # Stack into batch tensors
        return torch.stack(padded_embeddings), torch.stack(padded_types), torch.stack(padded_masks) #torch.stack(padded_positions)
    
    def forward(self, text_ids, image_tokens, training=True):
        """VILA-style forward propagation"""
        # Use special token IDs to locate the image region
        embeddings, token_types, attention_mask = self.process_batch(text_ids, image_tokens)
        
        # Add type embeddings
        type_embeds = self.type_embedding(token_types)
        embeddings = embeddings + type_embeds

        if training and embeddings.size(0) > 1:  # Only when batch size > 1 in training mode
            seq_lengths = attention_mask.sum(dim=1)
            if seq_lengths.min() != seq_lengths.max():  # Sequence lengths differ
                # Sort by sequence length
                sorted_lengths, sorted_indices = torch.sort(seq_lengths, descending=True)
                embeddings = embeddings[sorted_indices]
                attention_mask = attention_mask[sorted_indices]
                token_types = token_types[sorted_indices]
                #position_ids = position_ids[sorted_indices]
                
                # Record the original order
                inverse_indices = torch.argsort(sorted_indices)
                sorted = True
            else:
                sorted = False
        else:
            sorted = False
        
        # Pass through HypFormer model
        text_logits, img_logits, text_mask, img_mask = self.hypformer(
            embeddings, 
            attention_mask=attention_mask,
            token_types=token_types  # Pass modality types
        )
        
        if True:
        #if batch_idx == 0 and epoch == 0:  # Debug only the first batch in the first epoch
            print("\n[DEBUG] —— First batch diagnostic log ——")
            print(f"Text logits shape: {text_logits.shape}")
            print(f"Text target shape: {target_text.shape}")
            print(f"Text mask shape: {text_mask.shape}")

            # ✅ Text prediction diagnostics
            if text_mask.any():
                text_indices = torch.where(text_mask)
                valid_row = text_indices[0] < target_text.shape[0]
                valid_col = text_indices[1] < target_text.shape[1]
                valid_indices = valid_row & valid_col
                valid_row_indices = text_indices[0][valid_indices]
                valid_col_indices = text_indices[1][valid_indices]

                if valid_row_indices.numel() > 0:
                    print(f"[DEBUG] Valid text predictions: {valid_row_indices.numel()}")
                    sample_preds = torch.argmax(text_logits[valid_row_indices, valid_col_indices], dim=-1)
                    sample_targets = target_text[valid_row_indices, valid_col_indices]
                    print(f"[DEBUG] First 10 text predicted tokens: {sample_preds[:10]}")
                    print(f"[DEBUG] First 10 actual text target tokens: {sample_targets[:10]}")
                    unique_ids, counts = torch.unique(sample_targets, return_counts=True)
                    print(f"[DEBUG] Text target token distribution: {dict(zip(unique_ids.tolist(), counts.tolist()))}")  
                else:
                    print("[WARNING] ⚠️ No valid text prediction tokens (masking issue suspected)")

            # ✅ Image prediction diagnostics
            if img_logits is not None:
                print(f"Image logits shape: {img_logits.shape}")
                print(f"Target image shape: {target_img.reshape(-1).shape}")
                if img_logits.size(0) > 0:
                    image_preds = torch.argmax(img_logits, dim=-1)
                    image_targets = target_img.reshape(-1)[:img_logits.size(0)]

                    print(f"[DEBUG] Image logits range: mean={img_logits.mean().item():.4f}, std={img_logits.std().item():.4f}")
                    print(f"[DEBUG] First 10 predicted image tokens: {image_preds[:10]}")
                    print(f"[DEBUG] First 10 actual image target tokens: {image_targets[:10]}")
                    unique_ids, counts = torch.unique(image_targets, return_counts=True)
                    image_token_dist = dict(zip(unique_ids.tolist(), counts.tolist()))
                    print(f"[DEBUG] Image target token distribution: {image_token_dist}")
        
        
        # Restore original order if sorting was performed
        if sorted:
            text_logits = text_logits[inverse_indices]
            text_mask = text_mask[inverse_indices]
            # Note: img_logits may be None or require special handling if it's not None
        
        # Return logits and masks processed by HypFormer
        return text_logits, img_logits, text_mask, img_mask
            
# Create the model
model = MultimodalHypFormer(
    text_vocab_size=tokenizer.vocab_size,
    image_vocab_size=args.image_vocab_size,
    embed_dim=args.embed_dim,
    hidden_dim=args.hidden_channels,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout
).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    model = nn.DataParallel(model)
    # When using DataParallel, ensure the main device is GPU:0
    device = torch.device('cuda:0')
    
# Define optimizer
optimizer = Optimizer(model, args)
'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer,
    T_0=3,  # Epochs until first restart
    T_mult=2,  # Period multiplier after each restart
    eta_min=1e-6  # Minimum learning rate
)
'''
text_loss_fn = nn.CrossEntropyLoss()
image_loss_fn = nn.CrossEntropyLoss()


# ===================== Training Loop ======================
for epoch in range(args.epochs):
    start_time = time.time()
    model.train()
    
    total_loss = 0
    text_tokens_count = 0
    img_tokens_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        text_ids = batch["text_ids"].to(device)
        images = batch["image"].to(device)
        
        # Convert images into tokens
        with torch.no_grad():
            features = cnn_encoder(images)
            _, _, image_tokens = lfq(features)
            image_tokens = image_tokens.view(images.size(0), -1).long()
        
        # Adjust sequences for next-token prediction
        input_text = text_ids[:, :-1]  # Remove last token
        target_text = text_ids[:, 1:]  # Targets shifted by one position
        
        # Do the same for image tokens
        input_img = image_tokens[:, :-1]
        target_img = image_tokens[:, 1:]
        
        # Forward pass
        text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
        
        # Compute losses
        text_loss = torch.tensor(0.0, device=device, requires_grad=True)
        img_loss = torch.tensor(0.0, device=device, requires_grad=True)
        #print("\n==== Shape Information ====")
        #print(f"input_text: {input_text.shape}, target_text: {target_text.shape}")
        #print(f"input_img: {input_img.shape}, target_img: {target_img.shape}")
        #print(f"text_logits: {text_logits.shape}, text_mask: {text_mask.shape}")
        #print(f"img_logits shape: {img_logits.shape if img_logits is not None else 'None'}")
        #print(f"img_mask: {img_mask.shape}")
        
        # Text loss
        if text_mask.any():
          text_indices = torch.where(text_mask)
          
          # Filter out indices beyond the bounds of target_text
          valid_row = text_indices[0] < target_text.shape[0]  # Row indices must be valid
          valid_col = text_indices[1] < target_text.shape[1]  # Column indices must be valid
          valid_indices = valid_row & valid_col
          
          # Get predictions and targets using valid indices
          valid_row_indices = text_indices[0][valid_indices]
          valid_col_indices = text_indices[1][valid_indices]
          
          text_preds = text_logits[valid_row_indices, valid_col_indices]
          text_targets = target_text[valid_row_indices, valid_col_indices]
          
          text_loss = text_loss_fn(text_preds, text_targets)
          text_tokens_count += text_targets.numel()
        
        # Image loss
        if img_mask.any() and img_logits is not None:
            img_targets = target_img.reshape(-1)[:img_logits.size(0)]  # Ensure dimensions match
            img_loss = image_loss_fn(img_logits, img_targets)
            img_tokens_count += img_targets.numel()
        
        base_alpha = args.img_loss_weight  # Base weight 2.0
        max_alpha = 4.0  # Maximum weight
        alpha = base_alpha + (max_alpha - base_alpha) * min(1.0, epoch / 8)  # Gradually increase during first 8 epochs

        loss = text_loss + img_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #scheduler.step()
        

        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            text_loss=f"{text_loss.item():.4f}" if text_tokens_count > 0 else "N/A",
            img_loss=f"{img_loss.item():.4f}" if img_tokens_count > 0 else "N/A"
        )
    
    # Calculate average training loss
    avg_loss = total_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text_ids = batch["text_ids"].to(device)
            images = batch["image"].to(device)
            
            # Convert images into tokens
            features = cnn_encoder(images)
            _, _, image_tokens = lfq(features)
            image_tokens = image_tokens.view(images.size(0), -1).long()
            
            # Adjust sequences for next-token prediction
            input_text = text_ids[:, :-1]
            target_text = text_ids[:, 1:]
            
            input_img = image_tokens[:, :-1]
            target_img = image_tokens[:, 1:]
            
            # Forward pass
            text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
            
            # Compute losses
            v_text_loss = 0
            v_img_loss = 0
            
            # Text loss
            if text_mask.any():
                text_indices = torch.where(text_mask)
                
                # Filter out indices beyond the range of target_text
                valid_row = text_indices[0] < target_text.shape[0]
                valid_col = text_indices[1] < target_text.shape[1]
                valid_indices = valid_row & valid_col
                
                # Use valid indices to obtain predictions and targets
                valid_row_indices = text_indices[0][valid_indices]
                valid_col_indices = text_indices[1][valid_indices]
                
                text_preds = text_logits[valid_row_indices, valid_col_indices]
                text_targets = target_text[valid_row_indices, valid_col_indices]
                
                v_text_loss = text_loss_fn(text_preds, text_targets)
                      
            # Image loss
            if img_mask.any() and img_logits is not None:
                img_targets = target_img.reshape(-1)[:img_logits.size(0)]
                v_img_loss = image_loss_fn(img_logits, img_targets)
            
            # Total validation loss
            v_loss = v_text_loss + alpha * v_img_loss
            val_loss += v_loss.item()
    
    val_avg_loss = val_loss / len(val_loader)

    epoch_time = time.time() - start_time
    #scheduler.step(val_avg_loss)
    # Periodically generate and save sample images
    '''
    if (epoch + 1) % 5 == 0:
        os.makedirs("generated_samples", exist_ok=True)
        
        # Generate sample images
        sample_prompts = [
            "a beautiful sunset over the ocean",
            "a dog playing in the park", 
            "a red car driving down the road"
        ]
        
        print("\nGenerating sample images...")
        for i, prompt in enumerate(sample_prompts):
            try:
                generated_image, _ = model.generate_image_from_text(prompt)
                if generated_image is not None:
                    # Convert to PIL image and save
                    from torchvision.utils import save_image
                    save_image(
                        generated_image, 
                        f"generated_samples/epoch_{epoch+1}_sample_{i}.png",
                        normalize=True
                    )
                    print(f"  Saved image for prompt: '{prompt}'")
            except Exception as e:
                print(f"  Failed to generate image for prompt '{prompt}': {e}")
      '''

    print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
          f"Train Loss: {avg_loss:.4f}(Text:{text_loss:.4f},Image:{img_loss:.4f}), "
          f"Val Loss: {val_avg_loss:.4f}(Text:{v_text_loss:.4f},Image:{v_img_loss:.4f})", flush=True)

# ===================== Testing Phase ======================
model.eval()
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        text_ids = batch["text_ids"].to(device)
        images = batch["image"].to(device)
        
        # Convert images into tokens
        features = cnn_encoder(images)
        _, _, image_tokens = lfq(features)
        image_tokens = image_tokens.view(images.size(0), -1).long()
        
        # Adjust sequences for next-token prediction
        input_text = text_ids[:, :-1]
        target_text = text_ids[:, 1:]
        
        input_img = image_tokens[:, :-1]
        target_img = image_tokens[:, 1:]
        
        # Forward pass
        text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
        
        # Compute test loss
        t_text_loss = 0
        t_img_loss = 0
        
        if text_mask.any():
            text_indices = torch.where(text_mask)
                
            # Filter out indices beyond the range of target_text
            valid_row = text_indices[0] < target_text.shape[0]
            valid_col = text_indices[1] < target_text.shape[1]
            valid_indices = valid_row & valid_col
                
            # Use valid indices to get predictions and targets
            valid_row_indices = text_indices[0][valid_indices]
            valid_col_indices = text_indices[1][valid_indices]
                
            text_preds = text_logits[valid_row_indices, valid_col_indices]
            text_targets = target_text[valid_row_indices, valid_col_indices]
            t_text_loss = text_loss_fn(text_preds, text_targets)
        
        if img_mask.any() and img_logits is not None:
            img_targets = target_img.reshape(-1)[:img_logits.size(0)]
            t_img_loss = image_loss_fn(img_logits, img_targets)
        
        t_loss = t_text_loss + alpha * t_img_loss
        test_loss += t_loss.item()

test_avg_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_avg_loss:.4f}", flush=True)

# ===================== Save Model ======================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'args': args,
}, 'multimodal_hypformer.pth')

print("Model saved successfully!")
