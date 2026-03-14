import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)

from transformers import AutoTokenizer, AutoModel

# Config

@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    heavy_col: str = "Heavy"
    antigen_col: str = "antigen"
    label_col: str = "Label"
    target_col: str = "Target"

    model_name: str = "facebook/esm2_t6_8M_UR50D"

    max_heavy_len: int = 256
    max_antigen_len: int = 512

    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5

    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1

    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_esm2_cross_attention_targetsplit.pt"


cfg = Config()

# Utils

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {}
    try:
        metrics["AUC"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["AUC"] = float("nan")

    metrics["F1"] = f1_score(labels, preds, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(labels, preds)
    metrics["Accuracy"] = accuracy_score(labels, preds)
    return metrics


def contains_any(text: str, keywords):
    t = str(text).upper()
    return any(k.upper() in t for k in keywords)

# Dataset

class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records
        if len(self.records) == 0:
            raise ValueError("Empty dataset.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        heavy, antigen, label, target = self.records[idx]
        return {
            "heavy": heavy,
            "antigen": antigen,
            "label": float(label),
            "target": target,
        }


def load_and_split_dataset(cfg: Config):
    df = pd.read_csv(cfg.csv_path)

    required_cols = [cfg.heavy_col, cfg.antigen_col, cfg.label_col, cfg.target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg.csv_path}")

    df = df[required_cols].dropna().copy()

    df[cfg.heavy_col] = df[cfg.heavy_col].astype(str).str.strip().str.upper()
    df[cfg.antigen_col] = df[cfg.antigen_col].astype(str).str.strip().str.upper()
    df[cfg.target_col] = df[cfg.target_col].astype(str).str.strip()
    df[cfg.label_col] = pd.to_numeric(df[cfg.label_col], errors="coerce")
    df = df.dropna().copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    df = df[
        (df[cfg.heavy_col].str.len() > 0) &
        (df[cfg.antigen_col].str.len() > 0)
    ].reset_index(drop=True)

    # train/val: WT / Beta / Alpha
    trainval_mask = df[cfg.target_col].apply(
        lambda x: contains_any(x, ["WT", "BETA", "ALPHA"])
    )

    # test: Delta
    test_mask = df[cfg.target_col].apply(
        lambda x: contains_any(x, ["DELTA"])
    )

    trainval_df = df[trainval_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    if len(trainval_df) == 0:
        raise ValueError("No training/validation samples found for WT/Beta/Alpha.")
    if len(test_df) == 0:
        raise ValueError("No test samples found for Delta.")

    # shuffle trainval and split into train/val
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(trainval_df))
    rng.shuffle(indices)

    val_size = max(1, int(len(trainval_df) * cfg.val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Train/Val candidate samples (WT/Beta/Alpha): {len(trainval_df)}")
    print(f"Test samples (Delta): {len(test_df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    def to_records(sub_df):
        return list(zip(
            sub_df[cfg.heavy_col].tolist(),
            sub_df[cfg.antigen_col].tolist(),
            sub_df[cfg.label_col].tolist(),
            sub_df[cfg.target_col].tolist(),
        ))

    train_set = PairDataset(to_records(train_df))
    val_set = PairDataset(to_records(val_df))
    test_set = PairDataset(to_records(test_df))

    return train_set, val_set, test_set

# Collator

class PairCollator:
    def __init__(self, tokenizer, max_heavy_len, max_antigen_len):
        self.tokenizer = tokenizer
        self.max_heavy_len = max_heavy_len
        self.max_antigen_len = max_antigen_len

    @staticmethod
    def add_spaces(seq: str) -> str:
        return " ".join(list(seq))

    def __call__(self, batch):
        heavy_texts = [self.add_spaces(item["heavy"]) for item in batch]
        antigen_texts = [self.add_spaces(item["antigen"]) for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

        heavy_inputs = self.tokenizer(
            heavy_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_heavy_len,
        )
        antigen_inputs = self.tokenizer(
            antigen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_antigen_len,
        )

        return {
            "heavy_input_ids": heavy_inputs["input_ids"],
            "heavy_attention_mask": heavy_inputs["attention_mask"],
            "antigen_input_ids": antigen_inputs["input_ids"],
            "antigen_attention_mask": antigen_inputs["attention_mask"],
            "labels": labels,
        }

# Cross Attention Block

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        out = self.norm(query + self.dropout(out))
        return out, attn_weights

# Model

class ESM2BidirectionalCrossAttentionClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)

        # freeze ESM2
        for p in self.esm.parameters():
            p.requires_grad = False

        esm_dim = self.esm.config.hidden_size

        self.ab_to_ag = CrossAttentionBlock(
            dim=esm_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ag_to_ab = CrossAttentionBlock(
            dim=esm_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ab_proj = nn.Linear(esm_dim, hidden_dim)
        self.ag_proj = nn.Linear(esm_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):
        with torch.no_grad():
            heavy_emb = self.encode(heavy_input_ids, heavy_attention_mask)
            antigen_emb = self.encode(antigen_input_ids, antigen_attention_mask)

        antigen_key_padding_mask = (antigen_attention_mask == 0)
        heavy_key_padding_mask = (heavy_attention_mask == 0)

        heavy_ctx, heavy_to_antigen_attn = self.ab_to_ag(
            query=heavy_emb,
            key_value=antigen_emb,
            key_padding_mask=antigen_key_padding_mask,
        )

        antigen_ctx, antigen_to_heavy_attn = self.ag_to_ab(
            query=antigen_emb,
            key_value=heavy_emb,
            key_padding_mask=heavy_key_padding_mask,
        )

        heavy_vec = masked_mean(self.ab_proj(heavy_ctx), heavy_attention_mask)
        antigen_vec = masked_mean(self.ag_proj(antigen_ctx), antigen_attention_mask)

        pair_feat = torch.cat([
            heavy_vec,
            antigen_vec,
            torch.abs(heavy_vec - antigen_vec),
            heavy_vec * antigen_vec,
        ], dim=-1)

        logits = self.classifier(pair_feat).squeeze(-1)
        return logits, heavy_to_antigen_attn, antigen_to_heavy_attn

# Train / Eval

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits, _, _ = model(
                heavy_input_ids=heavy_input_ids,
                heavy_attention_mask=heavy_attention_mask,
                antigen_input_ids=antigen_input_ids,
                antigen_attention_mask=antigen_attention_mask,
            )

            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics

# Main

def main():
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_set, val_set, test_set = load_and_split_dataset(cfg)

    collator = PairCollator(
        tokenizer=tokenizer,
        max_heavy_len=cfg.max_heavy_len,
        max_antigen_len=cfg.max_antigen_len,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=cfg.model_name,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # pos_weight only from training data
    train_labels = [int(item["label"]) for item in train_set]
    pos = sum(train_labels)
    neg = len(train_labels) - pos

    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=cfg.device)
        print(f"Using train pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, cfg.device, optimizer=optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, cfg.device, optimizer=None
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_AUC={train_metrics['AUC']:.4f} "
            f"train_F1={train_metrics['F1']:.4f} "
            f"train_MCC={train_metrics['MCC']:.4f} "
            f"train_ACC={train_metrics['Accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_AUC={val_metrics['AUC']:.4f} "
            f"val_F1={val_metrics['F1']:.4f} "
            f"val_MCC={val_metrics['MCC']:.4f} "
            f"val_ACC={val_metrics['Accuracy']:.4f}"
        )

        if not np.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "best_val_auc": best_val_auc,
                },
                cfg.save_path,
            )
            print(f"Saved best model to {cfg.save_path}")

    # Final test on Delta

    print("\nLoading best model for final Delta test...")
    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics = run_epoch(
        model, test_loader, criterion, cfg.device, optimizer=None
    )

    print("\n===== Final Test on Delta =====")
    print(
        f"test_loss={test_loss:.4f} "
        f"test_AUC={test_metrics['AUC']:.4f} "
        f"test_F1={test_metrics['F1']:.4f} "
        f"test_MCC={test_metrics['MCC']:.4f} "
        f"test_ACC={test_metrics['Accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
