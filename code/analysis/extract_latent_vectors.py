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
    csv_path: str = "CoV-AbDab.csv"
    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    max_cdr3_len: int = 30
    batch_size: int = 256

    finetune_ckpt: str = "conditional_cvae_finetune.pt"
    scratch_ckpt: str = "conditional_cvae_scratch.pt"

    finetune_out: str = "latent_finetune.csv"
    scratch_out: str = "latent_scratch.csv"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

class AntigenCDR3Dataset(Dataset):
    def __init__(self, csv_path, antigen_col, cdr3_col, max_cdr3_len):
        self.df = pd.read_csv(csv_path)

        if antigen_col not in self.df.columns:
            raise ValueError(f"Column '{antigen_col}' not found in {csv_path}")
        if cdr3_col not in self.df.columns:
            raise ValueError(f"Column '{cdr3_col}' not found in {csv_path}")

        df = self.df[[antigen_col, cdr3_col]].dropna().copy()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()
        df = df[(df[antigen_col].str.len() > 0) & (df[cdr3_col].str.len() > 0)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid rows after filtering.")

        self.max_cdr3_len = max_cdr3_len
        self.samples = list(zip(df[antigen_col].tolist(), df[cdr3_col].tolist()))
        self.max_antigen_len = max(len(a) for a, _ in self.samples)

        print(f"Loaded {len(self.samples)} paired samples")
        print(f"Max antigen length = {self.max_antigen_len}")
        print(f"Max cdr3 length cap = {self.max_cdr3_len}")

    def __len__(self):
        return len(self.samples)

    def encode_seq(self, seq, fixed_len):
        true_len = min(len(seq), fixed_len)
        seq = seq[:fixed_len]
        ids = [stoi.get(ch, UNK_IDX) for ch in seq]
        if len(ids) < fixed_len:
            ids += [PAD_IDX] * (fixed_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(true_len, dtype=torch.long)

    def __getitem__(self, idx):
        antigen, cdr3 = self.samples[idx]
        a, a_len = self.encode_seq(antigen, self.max_antigen_len)
        x, x_len = self.encode_seq(cdr3, self.max_cdr3_len)
        a_mask = (a != PAD_IDX).long()

        return {
            "idx": idx,
            "antigen": antigen,
            "cdr3": cdr3,
            "x": x,
            "x_len": x_len,
            "a": a,
            "a_mask": a_mask,
            "a_len": a_len,
        }


def collate_fn(batch):
    out = {}
    out["idx"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    out["antigen"] = [b["antigen"] for b in batch]
    out["cdr3"] = [b["cdr3"] for b in batch]
    out["x"] = torch.stack([b["x"] for b in batch], dim=0)
    out["x_len"] = torch.stack([b["x_len"] for b in batch], dim=0)
    out["a"] = torch.stack([b["a"] for b in batch], dim=0)
    out["a_mask"] = torch.stack([b["a_mask"] for b in batch], dim=0)
    out["a_len"] = torch.stack([b["a_len"] for b in batch], dim=0)
    return out

# Model blocks

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

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

    def encode_feature(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)
        return h

    def forward(self, x):
        h = self.encode_feature(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


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
        h = self.fc(z)
        h = h.view(z.size(0), self.hidden_dim, self.max_seq_len)
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)
        logits = logits.transpose(1, 2)
        return logits


class AntigenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, a, a_mask):
        emb = self.embedding(a)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)

        mask = a_mask.unsqueeze(1).float()
        h = h * mask
        pooled = h.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return pooled


class ConditionalCNNVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.decoder = CNNDecoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=VOCAB_SIZE,
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.antigen_encoder = AntigenEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["antigen_hidden_dim"],
            num_layers=config["antigen_num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        seq_feat_dim = config["hidden_dim"] * config["max_cdr3_len"]
        ant_feat_dim = config["antigen_hidden_dim"]

        self.posterior_mu = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.posterior_logvar = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"] + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.length_head = nn.Sequential(
            nn.Linear(config["latent_dim"], config["latent_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["latent_dim"], config["max_cdr3_len"]),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def extract_latents(self, x, a, a_mask):
        hx = self.encoder.encode_feature(x)
        ha = self.antigen_encoder(a, a_mask)

        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = self.posterior_mu(q_input)
        logvar_q = self.posterior_logvar(q_input)
        z = self.reparameterize(mu_q, logvar_q)
        z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))

        return mu_q, z_cond

# Extraction

@torch.no_grad()
def extract_to_csv(model, loader, device, out_csv, tag):
    model.eval()
    rows = []

    for batch in loader:
        x = batch["x"].to(device)
        a = batch["a"].to(device)
        a_mask = batch["a_mask"].to(device)

        mu_q, z_cond = model.extract_latents(x, a, a_mask)

        mu_q = mu_q.cpu()
        z_cond = z_cond.cpu()

        for i in range(x.size(0)):
            row = {
                "idx": int(batch["idx"][i].item()),
                "model": tag,
                "antigen": batch["antigen"][i],
                "cdr3": batch["cdr3"][i],
                "cdr3_len": int(batch["x_len"][i].item()),
                "antigen_len": int(batch["a_len"][i].item()),
            }

            for j in range(mu_q.size(1)):
                row[f"mu_q_{j}"] = float(mu_q[i, j].item())

            for j in range(z_cond.size(1)):
                row[f"z_cond_{j}"] = float(z_cond[i, j].item())

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = ConditionalCNNVAE(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def main():
    dataset = AntigenCDR3Dataset(
        cfg.csv_path,
        cfg.antigen_col,
        cfg.cdr3_col,
        cfg.max_cdr3_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    if os.path.exists(cfg.finetune_ckpt):
        print(f"\nLoading finetune model: {cfg.finetune_ckpt}")
        model_ft, ckpt_ft = load_model(cfg.finetune_ckpt, cfg.device)
        extract_to_csv(model_ft, loader, cfg.device, cfg.finetune_out, tag="finetune")
    else:
        print(f"Skip finetune: checkpoint not found -> {cfg.finetune_ckpt}")

    if os.path.exists(cfg.scratch_ckpt):
        print(f"\nLoading scratch model: {cfg.scratch_ckpt}")
        model_sc, ckpt_sc = load_model(cfg.scratch_ckpt, cfg.device)
        extract_to_csv(model_sc, loader, cfg.device, cfg.scratch_out, tag="scratch")
    else:
        print(f"Skip scratch: checkpoint not found -> {cfg.scratch_ckpt}")


if __name__ == "__main__":
    main()

