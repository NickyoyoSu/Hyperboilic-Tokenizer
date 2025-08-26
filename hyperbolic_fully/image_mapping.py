import sys
import os
import time
import math
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlamaForCausalLM, AutoModelForVision2Seq, MllamaForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def get_parser():
    parser = argparse.ArgumentParser(description="Train a multimodal vision-language model with Hyperbolic mapping")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for the dataset to load")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--image_size", type=int, default=336, help="Size of input images")
    return parser

args = get_parser().parse_args()

USE_HYPERBOLIC = args.use_hyperbolic
USE_LORA = args.use_lora
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BASE_LR = args.learning_rate
MAX_LENGTH = args.max_length
DEVICE = args.device
MAX_SAMPLES = args.max_samples
IMAGE_SIZE = args.image_size

# ===================== Load Llama Vision Model =====================
hf_token = "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"
model_name = "meta-llama/Llama-3.2-11B-Vision"  # 11B Vision model instead of 1B text-only

device = torch.device(DEVICE)
processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

# Use the correct model class to load Llama-3.2-Vision
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

# ===== Apply LoRA Adaptation or Freeze Original Model =====
if USE_LORA:
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
else:
    # Freeze original model parameters
    for param in model.parameters():
        param.requires_grad = False

# ====================== Load Stable Diffusion for Image Generation ======================
# Since Llama Vision cannot generate images, we use a separate model for image generation.
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    scheduler=EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
    )
).to(device)

# ============== Custom Multimodal Mapping Head =================
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.vocab_size = base_model.config.vocab_size
        self.hidden_size = base_model.config.hidden_size
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Multimodal-specific adapter to handle multimodal features
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # First linear layer + normalization
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # Optional second layer
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # Hyperbolic classifier: num_features = hidden_size + 1 (extra time component)
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # Euclidean classifier
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # Pass through multimodal adapter first
        x = self.multimodal_adapter(last_hidden_states)
        
        # Standard processing
        x = self.linear1(x)
        x = self.norm1(x)
        
        if self.num_layers == 2:
            x = torch.relu(x)
            x = self.linear2(x)
            x = self.norm2(x)
        
        if self.use_hyperbolic:
            # Add time component, then map to hyperbolic space
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# ====================== Load COCO Dataset ======================
print("Loading COCO dataset using COCO API...")

try:
    import os
    from pycocotools.coco import COCO
    import numpy as np
    from PIL import Image
    import random
    
    # Set COCO dataset path - adjust based on your actual location
    data_dir = 'coco_dataset'
    train_annotations = os.path.join(data_dir, 'annotations/captions_train2017.json')
    val_annotations = os.path.join(data_dir, 'annotations/captions_val2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')
    val_image_dir = os.path.join(data_dir, 'val2017')
    
    # Check if dataset files exist
    if not os.path.exists(train_annotations) or not os.path.exists(train_image_dir):
        raise FileNotFoundError(f"COCO dataset files not found. Please ensure the dataset is downloaded to {data_dir}")
    
    # Load COCO API
    print("Loading COCO training annotations...")
    train_coco = COCO(train_annotations)
    print("Loading COCO validation annotations...")
    val_coco = COCO(val_annotations)
    
    # Get all image IDs
    train_ids = list(train_coco.imgs.keys())
    val_ids = list(val_coco.imgs.keys())
    
    # For test set, split from validation set
    random.seed(args.seed)
    random.shuffle(val_ids)
    val_split = int(len(val_ids) * 0.5)
    new_val_ids = val_ids[:val_split]
    test_ids = val_ids[val_split:]
    
    print(f"COCO dataset loaded successfully! Train: {len(train_ids)} images, Val: {len(new_val_ids)} images, Test: {len(test_ids)} images")
    
    # Create COCO dataset class
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
            # Get image ID and path
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                # Create a blank image as fallback
                image = Image.new('RGB', (args.image_size, args.image_size), color='black')
            
            # Get captions
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Randomly pick one caption
            if anns and 'caption' in anns[0]:
                caption = random.choice([ann['caption'] for ann in anns])
            else:
                caption = "No description"
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Encode text
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
    
    # Build datasets
    train_dataset = COCODataset(train_coco, train_ids, train_image_dir, transform=transform)
    val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, transform=transform)
    test_dataset = COCODataset(val_coco, test_ids, val_image_dir, transform=transform)
    
    # Build data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4, 
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
    traceback.print_exc()  
    print("Unable to load COCO dataset, creating synthetic dataset for testing instead...")

# ==================== Data Batching ====================
def get_batches(pairs, batch_size):
    """Divide dataset into batches"""
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i+batch_size]

def compute_lm_loss(logits, labels):
    """Compute language modeling loss"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

# ====================== Training ======================
def train_model():
    global best_loss
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0

        # Training loop
        for batch_pairs in tqdm(get_batches(train_pairs, BATCH_SIZE), total=len(train_pairs)//BATCH_SIZE, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            # Process multimodal inputs
            inputs = prepare_multimodal_inputs(batch_pairs, processor, MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                # Use custom mapping head
                logits = custom_lm_head(outputs.hidden_states[-1], learnable_curvature)
                loss = compute_lm_loss(logits, labels)
            
            # Backward pass
            loss.backward()
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + 
                list(custom_lm_head.parameters()) + 
                [learnable_curvature], 
                max_norm=1.0
            )
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Keep curvature parameter within a valid range
            with torch.no_grad():
                learnable_curvature.clamp_(1e-3, 1e1)
                
            total_loss += loss.item() * len(batch_pairs)
            count += len(batch_pairs)

        # Calculate and log metrics
        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        val_loss = evaluate_model(val_pairs, phase="Validation")
        save_model(epoch, val_loss, math.exp(val_loss) if val_loss < 20 else float("inf"))

# ====================== Evaluation ======================
def evaluate_model(pairs, phase="Val"):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    
    with torch.no_grad():
        for batch_pairs in tqdm(get_batches(pairs, BATCH_SIZE), total=len(pairs)//BATCH_SIZE, desc=f"Evaluating {phase}"):
            # Process inputs
            inputs = prepare_multimodal_inputs(batch_pairs, processor, MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            logits = custom_lm_head(hidden_states, learnable_curvature)
            
            # Compute loss
            loss_val = compute_lm_loss(logits, labels)
            total_loss += loss_val.item() * len(batch_pairs)
            count += len(batch_pairs)
            
    avg_loss = total_loss / count if count > 0 else 9999.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print(f"{phase} - loss={avg_loss:.4f}, PPL={ppl:.2f}")
    return avg_loss

# ====================== Image-to-Text ======================
def generate_text_from_image(image_path, max_len=100):
    """Input an image, output generated description"""
    model.eval()
    custom_lm_head.eval()
    
    # Load image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Process image input
    inputs = processor(images=image, text="Describe this image:", return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode output
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return generated_text

# ====================== Text-to-Image ======================
def generate_image_from_text(text_prompt, output_path=None):
    """Input a text prompt, generate an image"""
    # Use Stable Diffusion to generate image
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image = image_gen_model(text_prompt, guidance_scale=7.5).images[0]
    
    # Save image if path provided
    if output_path:
        image.save(output_path)
    
    return image

# ====================== Main ======================
if __name__ == "__main__":
    print(f"USE_HYPERBOLIC = {USE_HYPERBOLIC}")
    print(f"USE_LORA = {USE_LORA}")
    
    # Train model
    train_model()
    
    # Evaluate on test set
    print("\n=== Testing ===")
    evaluate_model(test_pairs, phase="Test")
    
    # Demo: Image-to-text generation
    print("\n=== Image-to-Text Generation ===")
    sample_image = test_pairs[0]["image"]
    sample_image_path = "sample_image.jpg"
    sample_image.save(sample_image_path)
    generated_description = generate_text_from_image(sample_image_path)
    print(f"Generated description: {generated_description}")
    
    # Demo: Text-to-image generation
    print("\n=== Text-to-Image Generation ===")
    sample_text_prompt = "A beautiful landscape with mountains and a lake at sunset"
    generated_image = generate_image_from_text(sample_text_prompt, "generated_image.jpg")
    print(f"Image generated and saved as generated_image.jpg")
