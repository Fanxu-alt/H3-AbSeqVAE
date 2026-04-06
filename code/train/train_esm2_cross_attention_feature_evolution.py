import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from transformers import AutoTokenizer, AutoModel

@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    heavy_col: str = "Heavy"
    antigen_col: str = "antigen"
    label_col: str = "Label"

    model_name: str = "facebook/esm2_t33_650M_UR50D"

    max_heavy_len: int = 256
    max_antigen_len: int = 512

    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5

    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1

    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0

    # early stopping
    early_stop_patience: int = 8
    min_delta: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_esm2_cross_attention.pt"

    vis_batch_size: int = 16
    max_vis_samples: int = 400
    feature_dir: str = "feature_evolution"
    frame_dir: str = "feature_evolution/frames"
    feature_npy_dir: str = "feature_evolution/features"
    gif_path: str = "feature_evolution/feature_evolution.gif"


cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

class PairDataset(Dataset):
    def __init__(self, csv_path, heavy_col, antigen_col, label_col):
        df = pd.read_csv(csv_path)

        for col in [heavy_col, antigen_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {csv_path}")

        df = df[[heavy_col, antigen_col, label_col]].dropna().copy()

        df[heavy_col] = df[heavy_col].astype(str).str.strip().str.upper()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna().copy()
        df[label_col] = df[label_col].astype(int)

        df = df[
            (df[heavy_col].str.len() > 0) &
            (df[antigen_col].str.len() > 0)
        ].reset_index(drop=True)

        self.samples = list(
            zip(
                df[heavy_col].tolist(),
                df[antigen_col].tolist(),
                df[label_col].tolist()
            )
        )

        if len(self.samples) == 0:
            raise ValueError("No valid samples after filtering.")

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        heavy, antigen, label = self.samples[idx]
        return {
            "heavy": heavy,
            "antigen": antigen,
            "label": float(label),
        }


class SubsetWithOriginalIndex(Dataset):

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

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

class ESM2BidirectionalCrossAttentionClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)

        # Freeze ESM2
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

    def extract_pair_feature(
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

        return pair_feat, heavy_to_antigen_attn, antigen_to_heavy_attn

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):
        pair_feat, heavy_to_antigen_attn, antigen_to_heavy_attn = self.extract_pair_feature(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        logits = self.classifier(pair_feat).squeeze(-1)
        return logits, heavy_to_antigen_attn, antigen_to_heavy_attn, pair_feat

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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []
    all_probs = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _, _, pair_feat = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        probs = torch.sigmoid(logits)

        all_features.append(pair_feat.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    return features, labels, probs


def save_epoch_features(epoch, features, labels, probs, out_dir):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, f"epoch_{epoch:03d}_features.npy"), features)
    np.save(os.path.join(out_dir, f"epoch_{epoch:03d}_labels.npy"), labels)
    np.save(os.path.join(out_dir, f"epoch_{epoch:03d}_probs.npy"), probs)

    df = pd.DataFrame(features)
    df["label"] = labels.astype(int)
    df["prob"] = probs
    df.to_csv(os.path.join(out_dir, f"epoch_{epoch:03d}_features.csv"), index=False)


def load_all_epoch_features(feature_dir, num_epochs):
    all_epoch_features = []
    labels_ref = None

    for epoch in range(1, num_epochs + 1):
        feat_path = os.path.join(feature_dir, f"epoch_{epoch:03d}_features.npy")
        label_path = os.path.join(feature_dir, f"epoch_{epoch:03d}_labels.npy")

        feats = np.load(feat_path)
        labels = np.load(label_path)

        if labels_ref is None:
            labels_ref = labels
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(
                    "Visualization labels changed across epochs. "
                    "Make sure the same visualization subset is used every epoch."
                )

        all_epoch_features.append(feats)

    return all_epoch_features, labels_ref


def plot_epoch_frame(epoch, coords, labels, save_path):
    plt.figure(figsize=(7, 6))

    pos = labels == 1
    neg = labels == 0

    plt.scatter(coords[neg, 0], coords[neg, 1], s=20, alpha=0.7, label="Negative")
    plt.scatter(coords[pos, 0], coords[pos, 1], s=20, alpha=0.7, label="Positive")

    plt.title(f"Feature Evolution - Epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_evolution_gif(frame_dir, num_epochs, gif_path, duration=0.8):
    images = []
    for epoch in range(1, num_epochs + 1):
        frame_path = os.path.join(frame_dir, f"epoch_{epoch:03d}.png")
        images.append(imageio.imread(frame_path))
    imageio.mimsave(gif_path, images, duration=duration)


def create_feature_evolution_visualization(cfg: Config, num_epochs: int):
    print("\nCreating feature evolution visualization...")

    all_epoch_features, labels = load_all_epoch_features(cfg.feature_npy_dir, num_epochs)

    # Fit one PCA on all epochs together for consistent coordinates
    stacked = np.concatenate(all_epoch_features, axis=0)
    pca = PCA(n_components=2, random_state=cfg.seed)
    stacked_2d = pca.fit_transform(stacked)

    n_samples = all_epoch_features[0].shape[0]

    ensure_dir(cfg.frame_dir)

    for epoch in range(1, num_epochs + 1):
        start = (epoch - 1) * n_samples
        end = epoch * n_samples
        coords = stacked_2d[start:end]

        frame_path = os.path.join(cfg.frame_dir, f"epoch_{epoch:03d}.png")
        plot_epoch_frame(epoch, coords, labels, frame_path)

    make_evolution_gif(cfg.frame_dir, num_epochs, cfg.gif_path)

    print(f"Saved frames to: {cfg.frame_dir}")
    print(f"Saved GIF to: {cfg.gif_path}")

def main():
    set_seed(cfg.seed)

    ensure_dir(cfg.feature_dir)
    ensure_dir(cfg.frame_dir)
    ensure_dir(cfg.feature_npy_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset = PairDataset(
        csv_path=cfg.csv_path,
        heavy_col=cfg.heavy_col,
        antigen_col=cfg.antigen_col,
        label_col=cfg.label_col,
    )

    val_size = max(1, int(len(dataset) * cfg.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    vis_count = min(cfg.max_vis_samples, len(val_set))
    vis_indices = list(range(vis_count))
    vis_set = SubsetWithOriginalIndex(
        val_set.dataset,
        [val_set.indices[i] for i in vis_indices]
    )

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

    vis_loader = DataLoader(
        vis_set,
        batch_size=cfg.vis_batch_size,
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

    labels_all = [int(item["label"]) for item in dataset]
    pos = sum(labels_all)
    neg = len(labels_all) - pos

    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=cfg.device)
        print(f"Using pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch_ran = 0

    for epoch in range(1, cfg.epochs + 1):
        last_epoch_ran = epoch

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, cfg.device
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

        if val_metrics["AUC"] > best_val_auc + cfg.min_delta:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                cfg.save_path,
            )
            print(
                f"Saved best model to {cfg.save_path} "
                f"(best_epoch={best_epoch:02d}, best_val_AUC={best_val_auc:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No val_AUC improvement for {epochs_without_improvement} epoch(s). "
                f"Current best: epoch {best_epoch:02d}, AUC={best_val_auc:.4f}"
            )
        features, vis_labels, vis_probs = extract_features(model, vis_loader, cfg.device)
        save_epoch_features(
            epoch=epoch,
            features=features,
            labels=vis_labels,
            probs=vis_probs,
            out_dir=cfg.feature_npy_dir,
        )
        print(f"Saved visualization features for epoch {epoch:02d}")

        # early stopping
        if epochs_without_improvement >= cfg.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch:02d}. "
                f"Best epoch = {best_epoch:02d}, best val_AUC = {best_val_auc:.4f}"
            )
            break

    print(
        f"\nTraining finished. Last epoch run = {last_epoch_ran:02d}, "
        f"best epoch = {best_epoch:02d}, best val_AUC = {best_val_auc:.4f}"
    )

    create_feature_evolution_visualization(cfg, last_epoch_ran)


if __name__ == "__main__":
    main()
