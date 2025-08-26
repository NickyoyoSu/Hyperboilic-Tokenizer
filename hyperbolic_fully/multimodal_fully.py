#### python
# filepath: /Users/nick/Documents/yale/科研/代码/hpyllama/hyperbolic_fully/multimodal_fully.py

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

##############################################################################
# Import: CNN Encoder / Decoder, HypFormer, Hyperbolic Optimizer, LFQ Quantizer
##############################################################################
from utils.improved_model import Encoder, Decoder       # CNN encoder & decoder for images
from utils.hypformer_backup import HypFormer            # Transformer in hyperbolic space
from utils.manifolds.hyp_layer import Optimizer         # Custom hyperbolic optimizer
from utils.quantize.lookupFree import LFQ               # Codebook-based quantizer

##############################################################################
# Import: Hugging Face for text processing
##############################################################################
from datasets import load_dataset
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    """
    Simplified multimodal dataset example: contains both images and text.
    - image_folder: torchvision's ImageFolder or similar dataset
    - text_dataset: Hugging Face text dataset
    - tokenizer: text tokenizer
    - transform: image preprocessing
    """
    def __init__(self, image_folder, text_dataset, tokenizer, transform=None, max_len=64):
        super().__init__()
        self.image_folder = image_folder
        self.text_dataset = text_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return min(len(self.image_folder), len(self.text_dataset))

    def __getitem__(self, idx):
        # Load image sample
        img, _ = self.image_folder[idx]
        if self.transform:
            img = self.transform(img)

        # Load text sample
        text_data = self.text_dataset[idx]["text"]

        # Do not directly tokenize here, leave it to collate_fn or outside
        return img, text_data


class MultiModalModel(nn.Module):
    """
    Includes two branches: text & image:
    1) Image branch: CNN Encoder + LFQ quantization -> tokenized_x -> embedding -> HypFormer
    2) Text branch: tokenizer output (input_ids) -> embedding -> HypFormer
    3) The HypFormer output can be passed to the CNN Decoder to generate images ("text→image").
    """
    def __init__(self, args, text_vocab_size):
        super().__init__()
        self.device = torch.device(args.device)

        # ------------------ Image branch: encoder + quantization ------------------
        self.cnn_encoder = Encoder(
            ch=64, out_ch=3, in_channels=3, num_res_blocks=2,
            z_channels=18, ch_mult=(1, 2, 2, 4),
            resolution=args.image_size, double_z=False
        )
        self.lfq = LFQ(codebook_size=args.vocab_size, dim=16)

        # ------------------ Text branch: Embedding ------------------
        # Ensure the text embedding dimension matches the encoder's z_channels
        self.text_embedding_dim = 18
        self.text_embedding = nn.Embedding(num_embeddings=text_vocab_size, embedding_dim=self.text_embedding_dim)

        # ------------------ HypFormer: backbone model ------------------
        # Used to process multimodal features in hyperbolic space
        self.hypformer = HypFormer(
            in_channels=self.text_embedding_dim,      
            hidden_channels=args.hidden_channels,
            out_channels=args.vocab_size,
            trans_num_layers=args.num_layers,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            args=None
        )

        # ------------------ Image decoder: text-to-image ------------------
        # Decode HypFormer output into images
        self.img_decoder = Decoder(
            ch=64, out_ch=3, in_channels=18,   # Matches HypFormer output dimension
            num_res_blocks=2, resolution=args.image_size
        )

    @torch.no_grad()
    def forward_image_branch(self, images: torch.Tensor):
        """
        Image -> Encoder -> LFQ -> tokenize -> text_embedding -> HypFormer
        Returns: (batch_size, seq_len, vocab_size), and tokenized_x
        """
        features = self.cnn_encoder(images)                      # (b, 18, H, W)
        _, _, tokenized_x_raw = self.lfq(features)               # Quantize
        tokenized_x = tokenized_x_raw.view(images.size(0), -1).long()
        # Embedding
        img_embed = self.text_embedding(tokenized_x)             # (b, seq_len, 18)
        # Forward to HypFormer
        output = self.hypformer(img_embed)                       # (b, seq_len, vocab_size)
        return output, tokenized_x

    def forward_text_branch(self, input_ids: torch.Tensor):
        """
        Text -> embedding -> HypFormer
        Returns: (batch_size, seq_len, vocab_size)
        """
        txt_embed = self.text_embedding(input_ids)               # (b, seq_len, 18)
        output = self.hypformer(txt_embed)                       # (b, seq_len, vocab_size)
        return output

    def generate_image_from_text(self, input_ids: torch.Tensor):
        """
        Text -> HypFormer -> reshape -> Decoder -> generate image
        """
        # 1) Text embeddings
        txt_embed = self.text_embedding(input_ids)               # (b, seq_len, 18)
        # 2) HypFormer output sequence
        seq_out = self.hypformer(txt_embed)                      # (b, seq_len, vocab_size)
        # Use mean pooling for simplicity; real cases may need attention pooling or concatenation
        global_repr = seq_out.mean(dim=1)                        # (b, vocab_size)
        
        # 3) Map global_repr to 18 dims, reshape -> (b, 18, H, W)
        hidden_dim = 18
        linear_map = nn.Linear(seq_out.shape[-1], hidden_dim).to(self.device)
        global_repr_18 = linear_map(global_repr)                 # (b, 18)
        # Reshape
        b_size = global_repr_18.shape[0]
        global_repr_4d = global_repr_18.view(b_size, hidden_dim, 8, 8)

        # 4) Decode into image
        generated_img = self.img_decoder(global_repr_4d)         # (b, 3, H, W)
        return generated_img


def collate_fn(batch_list):
    """
    Collate function for DataLoader:
    - Pack images into a single tensor
    - Keep texts as a list
    """
    imgs, texts = zip(*batch_list)
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(texts)


def main():
    parser = argparse.ArgumentParser(description="MultiModal HypFormer at extremes!")
    # Training configs
    parser.add_argument('--data_dir', type=str, default="/path/to/tiny-imagenet-200")
    parser.add_argument('--text_dataset_name', type=str, default="wikitext")
    parser.add_argument('--text_dataset_config', type=str, default="wikitext-2-raw-v1")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=2**18)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # -------------------- Image dataset --------------------
    from torchvision import transforms, datasets
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "train"), transform=transform)

    # -------------------- Text dataset --------------------
    raw_datasets = load_dataset(args.text_dataset_name, args.text_dataset_config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Ensure text_dataset matches image_dataset size
    text_train_data = raw_datasets["train"].select(range(len(image_dataset)))

    # -------------------- Multimodal dataset & Dataloader --------------------
    multimodal_data = MultiModalDataset(
        image_folder=image_dataset,
        text_dataset=text_train_data,
        tokenizer=tokenizer,
        transform=None
    )
    train_loader = DataLoader(multimodal_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # -------------------- Model initialization --------------------
    model = MultiModalModel(args, tokenizer.vocab_size).to(device)
    optimizer = Optimizer(model.hypformer, args)
    loss_fn = nn.CrossEntropyLoss()

    # -------------------- Training example --------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch_img, batch_text in train_loader:
            batch_img = batch_img.to(device)
            # Tokenize text
            tokenized_text = tokenizer(batch_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(device)  # (b, seq_len)

            # Forward: image branch
            with torch.no_grad():
                img_output, img_tokens = model.forward_image_branch(batch_img)
            # Forward: text branch
            txt_output = model.forward_text_branch(input_ids)

            # Use image branch as example loss computation
            b, seq_len, v_size = img_output.shape
            img_output_flat = img_output.view(b * seq_len, v_size)
            # Shift by 1 for target alignment
            target = img_tokens[:, 1:1+seq_len].reshape(b * seq_len)

            loss = loss_fn(img_output_flat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    print("Multimodal training finished!")

    # -------------------- Text-to-Image inference example --------------------
    model.eval()
    sample_text = ["A red cat jumping on the moon"]  # Input any English description
    with torch.no_grad():
        tokenized_text = tokenizer(sample_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        gen_img = model.generate_image_from_text(input_ids)
        print("Generated image tensor shape:", gen_img.shape)
        # You can convert gen_img to PIL or save it to visualize

if __name__ == "__main__":
    main()
