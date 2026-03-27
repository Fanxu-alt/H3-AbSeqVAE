import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# Config

@dataclass
class Config:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    max_heavy_len: int = 256
    max_antigen_len: int = 512
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

# Utils

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def add_spaces(seq: str) -> str:
    return " ".join(list(seq.strip().upper()))


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def find_valid_token_span(input_ids, tokenizer):
    """
      [CLS] A B C ... [EOS] [PAD] [PAD]
    """
    special_ids = set(tokenizer.all_special_ids)
    ids = input_ids.tolist()
    valid_idx = [i for i, tid in enumerate(ids) if tid not in special_ids]
    return valid_idx


def idx_to_residue_labels(seq: str):
    return [f"{aa}{i+1}" for i, aa in enumerate(seq)]

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
            average_attn_weights=False,  # [B, H, Q, K]
        )
        out = self.norm(query + self.dropout(out))
        return out, attn_weights

# Model

class ESM2BidirectionalCrossAttentionClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)

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

# Attention analysis

def build_inputs(tokenizer, heavy_seq, antigen_seq, max_heavy_len, max_antigen_len):
    heavy_inputs = tokenizer(
        add_spaces(heavy_seq),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_heavy_len,
    )
    antigen_inputs = tokenizer(
        add_spaces(antigen_seq),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_antigen_len,
    )
    return heavy_inputs, antigen_inputs


@torch.no_grad()
def analyze_attention(
    model,
    tokenizer,
    heavy_seq,
    antigen_seq,
    max_heavy_len=256,
    max_antigen_len=512,
    device="cpu",
):
    model.eval()

    heavy_inputs, antigen_inputs = build_inputs(
        tokenizer, heavy_seq, antigen_seq, max_heavy_len, max_antigen_len
    )

    heavy_inputs = to_device(heavy_inputs, device)
    antigen_inputs = to_device(antigen_inputs, device)

    logits, h2a_attn, a2h_attn = model(
        heavy_input_ids=heavy_inputs["input_ids"],
        heavy_attention_mask=heavy_inputs["attention_mask"],
        antigen_input_ids=antigen_inputs["input_ids"],
        antigen_attention_mask=antigen_inputs["attention_mask"],
    )

    prob = torch.sigmoid(logits).item()

    # shape: [B, H, Q, K]
    h2a_attn = h2a_attn[0].detach().cpu()  # [H, Qh, Ka]
    a2h_attn = a2h_attn[0].detach().cpu()  # [H, Qa, Kh]

    #  head
    h2a_mean = h2a_attn.mean(dim=0)  # [Qh, Ka]
    a2h_mean = a2h_attn.mean(dim=0)  # [Qa, Kh]

    heavy_valid_idx = find_valid_token_span(heavy_inputs["input_ids"][0].cpu(), tokenizer)
    antigen_valid_idx = find_valid_token_span(antigen_inputs["input_ids"][0].cpu(), tokenizer)

    h2a_clean = h2a_mean[np.ix_(heavy_valid_idx, antigen_valid_idx)].numpy()
    a2h_clean = a2h_mean[np.ix_(antigen_valid_idx, heavy_valid_idx)].numpy()

    heavy_labels = idx_to_residue_labels(heavy_seq[:len(heavy_valid_idx)])
    antigen_labels = idx_to_residue_labels(antigen_seq[:len(antigen_valid_idx)])

    # residue importance
    # heavy->antigen
    heavy_importance_from_h2a = h2a_clean.mean(axis=1)   # [H]
    antigen_importance_from_h2a = h2a_clean.mean(axis=0) # [A]

    # antigen->heavy
    antigen_importance_from_a2h = a2h_clean.mean(axis=1) # [A]
    heavy_importance_from_a2h = a2h_clean.mean(axis=0)   # [H]

    heavy_importance = (heavy_importance_from_h2a + heavy_importance_from_a2h) / 2.0
    antigen_importance = (antigen_importance_from_h2a + antigen_importance_from_a2h) / 2.0

    return {
        "prob": prob,
        "heavy_labels": heavy_labels,
        "antigen_labels": antigen_labels,
        "h2a_attn": h2a_clean,
        "a2h_attn": a2h_clean,
        "heavy_importance": heavy_importance,
        "antigen_importance": antigen_importance,
        "h2a_heads": h2a_attn.numpy(),
        "a2h_heads": a2h_attn.numpy(),
    }


def print_top_residues(labels, scores, top_k=15, title="Top residues"):
    order = np.argsort(scores)[::-1][:top_k]
    print(f"\n{title}")
    print("-" * len(title))
    for rank, idx in enumerate(order, 1):
        print(f"{rank:02d}. {labels[idx]:>6s}  score={scores[idx]:.6f}")


def plot_attention_heatmap(attn_matrix, row_labels, col_labels, title, save_path=None):
    plt.figure(figsize=(max(10, len(col_labels) * 0.18), max(6, len(row_labels) * 0.18)))
    plt.imshow(attn_matrix, aspect="auto")
    plt.colorbar(label="Attention")
    plt.title(title)
    plt.xlabel("Antigen residues")
    plt.ylabel("Heavy residues")

    step_x = max(1, len(col_labels) // 40)
    step_y = max(1, len(row_labels) // 40)

    plt.xticks(
        ticks=np.arange(0, len(col_labels), step_x),
        labels=[col_labels[i] for i in range(0, len(col_labels), step_x)],
        rotation=90,
        fontsize=8,
    )
    plt.yticks(
        ticks=np.arange(0, len(row_labels), step_y),
        labels=[row_labels[i] for i in range(0, len(row_labels), step_y)],
        fontsize=8,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_residue_importance(labels, scores, title, save_path=None):
    plt.figure(figsize=(max(12, len(labels) * 0.16), 4))
    x = np.arange(len(labels))
    plt.bar(x, scores)
    plt.title(title)
    plt.xlabel("Residue")
    plt.ylabel("Importance")

    step = max(1, len(labels) // 40)
    plt.xticks(
        ticks=np.arange(0, len(labels), step),
        labels=[labels[i] for i in range(0, len(labels), step)],
        rotation=90,
        fontsize=8,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_submatrix(attn_matrix, row_labels, col_labels, top_row_k=20, top_col_k=30, title="", save_path=None):
    row_scores = attn_matrix.mean(axis=1)
    col_scores = attn_matrix.mean(axis=0)

    top_rows = np.argsort(row_scores)[::-1][:top_row_k]
    top_cols = np.argsort(col_scores)[::-1][:top_col_k]

    top_rows = np.sort(top_rows)
    top_cols = np.sort(top_cols)

    sub = attn_matrix[np.ix_(top_rows, top_cols)]
    sub_row_labels = [row_labels[i] for i in top_rows]
    sub_col_labels = [col_labels[i] for i in top_cols]

    plt.figure(figsize=(max(8, len(sub_col_labels) * 0.35), max(5, len(sub_row_labels) * 0.35)))
    plt.imshow(sub, aspect="auto")
    plt.colorbar(label="Attention")
    plt.title(title)
    plt.xlabel("Top antigen residues")
    plt.ylabel("Top heavy residues")
    plt.xticks(np.arange(len(sub_col_labels)), sub_col_labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(sub_row_labels)), sub_row_labels, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Main demo

def main():
    set_seed(42)

    checkpoint_path = "best_esm2_cross_attention.pt"

    heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGITVSSNYMTWVRQAPGKGLEWVSVIYSGGSTFYADSVRGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLEMAGAFDIWGQGTMVTVSS"
    antigen_seq = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=cfg.model_name,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if "best_val_auc" in ckpt:
        print(f"best_val_auc = {ckpt['best_val_auc']:.4f}")

    result = analyze_attention(
        model=model,
        tokenizer=model.tokenizer,
        heavy_seq=heavy_seq,
        antigen_seq=antigen_seq,
        max_heavy_len=cfg.max_heavy_len,
        max_antigen_len=cfg.max_antigen_len,
        device=cfg.device,
    )

    print(f"\nPredicted binding probability: {result['prob']:.6f}")

    print_top_residues(
        result["heavy_labels"],
        result["heavy_importance"],
        top_k=15,
        title="Top heavy residues by bidirectional attention importance",
    )

    print_top_residues(
        result["antigen_labels"],
        result["antigen_importance"],
        top_k=20,
        title="Top antigen residues by bidirectional attention importance",
    )

    os.makedirs("attention_outputs", exist_ok=True)

    plot_attention_heatmap(
        result["h2a_attn"],
        result["heavy_labels"],
        result["antigen_labels"],
        title="Heavy -> Antigen Cross-Attention",
        save_path="attention_outputs/heavy_to_antigen_heatmap.png",
    )

    plot_attention_heatmap(
        result["a2h_attn"].T,  
        result["heavy_labels"],
        result["antigen_labels"],
        title="Antigen -> Heavy Cross-Attention (transposed for alignment)",
        save_path="attention_outputs/antigen_to_heavy_heatmap_transposed.png",
    )

    plot_residue_importance(
        result["heavy_labels"],
        result["heavy_importance"],
        title="Heavy residue importance",
        save_path="attention_outputs/heavy_importance.png",
    )

    plot_residue_importance(
        result["antigen_labels"],
        result["antigen_importance"],
        title="Antigen residue importance",
        save_path="attention_outputs/antigen_importance.png",
    )

    plot_top_submatrix(
        result["h2a_attn"],
        result["heavy_labels"],
        result["antigen_labels"],
        top_row_k=20,
        top_col_k=30,
        title="Top focused submatrix: Heavy -> Antigen",
        save_path="attention_outputs/heavy_to_antigen_top_submatrix.png",
    )

    print("\nSaved figures to: attention_outputs/")


if __name__ == "__main__":
    main()
