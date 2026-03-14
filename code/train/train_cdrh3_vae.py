import os
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config

@dataclass
class Config:
    csv_path: str = "heavy_with_cdrh3.csv"
    seq_col: str = "CDRH3"
    max_seq_len: int = 30

    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5

    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.1

    beta_kl: float = 0.1
    kl_anneal_epochs: int = 10
    length_loss_weight: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "vae_cdrh3_pretrain_varlen.pt"


cfg = Config()

# Vocabulary

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

# Dataset

class CDRH3Dataset(Dataset):
    def __init__(self, csv_path: str, seq_col: str, max_seq_len: int):
        self.df = pd.read_csv(csv_path)
        if seq_col not in self.df.columns:
            raise ValueError(f"Column '{seq_col}' not found in {csv_path}")

        raw = self.df[seq_col].dropna().astype(str).tolist()
        self.max_seq_len = max_seq_len
        self.samples = []

        for s in raw:
            s = s.strip().upper()
            if len(s) == 0:
                continue
            self.samples.append(s)

        if len(self.samples) == 0:
            raise ValueError("No valid sequences found.")

        print(f"Loaded {len(self.samples)} sequences before length normalization")

    def __len__(self):
        return len(self.samples)

    def encode_seq(self, seq: str):
        true_len = min(len(seq), self.max_seq_len)
        seq = seq[:self.max_seq_len]
        ids = [stoi.get(ch, UNK_IDX) for ch in seq]
        if len(ids) < self.max_seq_len:
            ids += [PAD_IDX] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(true_len, dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x, length = self.encode_seq(seq)
        return x, length

# Residual Block

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out

# Encoder

class CNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

        flat_dim = hidden_dim * max_seq_len
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def encode_feature(self, x):
        emb = self.embedding(x)       # [B, L, E]
        emb = emb.transpose(1, 2)     # [B, E, L]
        h = self.input_proj(emb)      # [B, H, L]
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)  # [B, H*L]
        return h

    def forward(self, x):
        h = self.encode_feature(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder

class CNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim * max_seq_len)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, z):
        h = self.fc(z)  # [B, H*L]
        h = h.view(z.size(0), self.hidden_dim, self.max_seq_len)  # [B, H, L]
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)          # [B, V, L]
        logits = logits.transpose(1, 2)       # [B, L, V]
        return logits

# VAE

class CNNVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            max_seq_len=cfg.max_seq_len,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        self.decoder = CNNDecoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            vocab_size=VOCAB_SIZE,
            max_seq_len=cfg.max_seq_len,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        # length prediction from latent z
        self.length_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, cfg.max_seq_len),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        len_logits = self.length_head(z)
        return logits, mu, logvar, len_logits, z

# Loss

def vae_loss(logits, targets, true_lengths, mu, logvar, len_logits, beta=1.0, len_weight=0.2):
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_IDX,
        reduction="mean",
    )

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()

    len_targets = true_lengths - 1
    length_loss = F.cross_entropy(len_logits, len_targets, reduction="mean")

    total = recon_loss + beta * kl + len_weight * length_loss
    return total, recon_loss, kl, length_loss


def get_beta(epoch, max_beta, anneal_epochs):
    if anneal_epochs <= 0:
        return max_beta
    return max_beta * min(1.0, epoch / anneal_epochs)

# Train / Eval

def train_one_epoch(model, loader, optimizer, device, beta, len_weight):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_len = 0.0

    for x, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits, mu, logvar, len_logits, z = model(x)
        loss, recon, kl, len_loss = vae_loss(
            logits, x, lengths, mu, logvar, len_logits, beta=beta, len_weight=len_weight
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_len += len_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_len / n


@torch.no_grad()
def evaluate(model, loader, device, beta, len_weight):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_len = 0.0

    for x, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)

        logits, mu, logvar, len_logits, z = model(x)
        loss, recon, kl, len_loss = vae_loss(
            logits, x, lengths, mu, logvar, len_logits, beta=beta, len_weight=len_weight
        )

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_len += len_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_len / n

# Utility

def decode_tokens(token_ids):
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


@torch.no_grad()
def reconstruct_examples(model, dataset, device, num_examples=5):
    model.eval()
    print("\nSample reconstructions:")
    for i in range(min(num_examples, len(dataset))):
        x, true_len = dataset[i]
        x = x.unsqueeze(0).to(device)

        logits, _, _, len_logits, _ = model(x)
        pred_tokens = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        pred_len = int(len_logits.argmax(dim=-1).item()) + 1

        inp_tokens = x.squeeze(0).cpu().tolist()
        inp_seq = decode_tokens(inp_tokens[:true_len.item()])
        pred_seq = decode_tokens(pred_tokens[:pred_len])

        print(f"input(len={true_len.item():2d}): {inp_seq}")
        print(f"recon(len={pred_len:2d}): {pred_seq}")
        print("-" * 60)
# Main

def main():
    dataset = CDRH3Dataset(cfg.csv_path, cfg.seq_col, cfg.max_seq_len)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = CNNVAE(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        beta = get_beta(epoch, cfg.beta_kl, cfg.kl_anneal_epochs)

        train_loss, train_recon, train_kl, train_len = train_one_epoch(
            model, train_loader, optimizer, cfg.device, beta, cfg.length_loss_weight
        )
        val_loss, val_recon, val_kl, val_len = evaluate(
            model, val_loader, cfg.device, beta, cfg.length_loss_weight
        )

        print(
            f"Epoch {epoch:02d} | beta={beta:.4f} | "
            f"train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} len={train_len:.4f} | "
            f"val_loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} len={val_len:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "stoi": stoi,
                    "itos": itos,
                },
                cfg.save_path,
            )
            print(f"Saved best model to {cfg.save_path}")

    reconstruct_examples(model, dataset, cfg.device, num_examples=5)


if __name__ == "__main__":
    main()
