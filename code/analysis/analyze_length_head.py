import os
import math
from dataclasses import dataclass
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config

@dataclass
class AnalysisConfig:
    ckpt_path: str = "conditional_cvae_finetune.pt"
    csv_path: str = "CoV-AbDab.csv"
    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    batch_size: int = 256
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_unique_antigens_to_report: int = 20

    # latent traversal
    traversal_num_antigens: int = 5
    traversal_dims_to_show: int = 12
    traversal_delta_values: tuple = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    # Output directory
    out_dir: str = "length_head_analysis"


cfg = AnalysisConfig()

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
    def __init__(self, csv_path, antigen_col, cdr3_col, max_cdr3_len, max_antigen_len=None):
        self.df = pd.read_csv(csv_path)

        if antigen_col not in self.df.columns:
            raise ValueError(f"Column '{antigen_col}' not found in {csv_path}")
        if cdr3_col not in self.df.columns:
            raise ValueError(f"Column '{cdr3_col}' not found in {csv_path}")

        self.antigen_col = antigen_col
        self.cdr3_col = cdr3_col
        self.max_cdr3_len = max_cdr3_len

        df = self.df[[antigen_col, cdr3_col]].dropna().copy()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()

        df = df[(df[antigen_col].str.len() > 0) & (df[cdr3_col].str.len() > 0)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No valid rows after filtering.")

        self.samples = list(zip(df[antigen_col].tolist(), df[cdr3_col].tolist()))

        inferred_max_antigen_len = max(len(a) for a, _ in self.samples)
        self.max_antigen_len = inferred_max_antigen_len if max_antigen_len is None else max_antigen_len

        print(f"Loaded {len(self.samples)} paired samples")
        print(f"Max antigen length used = {self.max_antigen_len}")
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
            "x": x,
            "x_len": x_len,
            "a": a,
            "a_mask": a_mask,
            "a_len": a_len,
            "antigen_seq": antigen,
            "cdr3_seq": cdr3,
        }


# Blocks

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

# Model

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

    def forward(self, x, a, a_mask):
        hx = self.encoder.encode_feature(x)
        ha = self.antigen_encoder(a, a_mask)

        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = self.posterior_mu(q_input)
        logvar_q = self.posterior_logvar(q_input)

        mu_p = self.prior_mu(ha)
        logvar_p = self.prior_logvar(ha)

        z = self.reparameterize(mu_q, logvar_q)

        z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))
        logits = self.decoder(z_cond)
        len_logits = self.length_head(z_cond)

        return logits, mu_q, logvar_q, mu_p, logvar_p, len_logits, z_cond

    @torch.no_grad()
    def get_length_logits_from_posterior_mean(self, x, a, a_mask):
        """
        Deterministic analysis using posterior mean：
            x,a -> q(z|x,a) of mu_q
            z_cond = f([mu_q, ha])
            length_head(z_cond)
        
        """
        hx = self.encoder.encode_feature(x)
        ha = self.antigen_encoder(a, a_mask)

        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = self.posterior_mu(q_input)
        logvar_q = self.posterior_logvar(q_input)

        z_cond = self.decoder_input(torch.cat([mu_q, ha], dim=-1))
        len_logits = self.length_head(z_cond)
        return len_logits, mu_q, logvar_q, ha, z_cond

    @torch.no_grad()
    def get_length_logits_from_prior_mean(self, a, a_mask):
        """
        Deterministic analysis based solely on antigen conditions：
            a -> p(z|a) of mu_p
            z_cond = f([mu_p, ha])
            length_head(z_cond)
     
        """
        ha = self.antigen_encoder(a, a_mask)
        mu_p = self.prior_mu(ha)
        logvar_p = self.prior_logvar(ha)

        z_cond = self.decoder_input(torch.cat([mu_p, ha], dim=-1))
        len_logits = self.length_head(z_cond)
        return len_logits, mu_p, logvar_p, ha, z_cond

# Utilities

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def topk_length_probs(len_probs, k=5):
    """
    len_probs: [Lmax]
    Return the (length, prob) values ​​of the top-k integers.
    """
    vals, idx = torch.topk(len_probs, k=min(k, len_probs.numel()))
    results = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        results.append((i + 1, float(v)))
    return results


def build_confusion_matrix(y_true, y_pred, max_len):
    cm = np.zeros((max_len, max_len), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 1 <= t <= max_len and 1 <= p <= max_len:
            cm[t - 1, p - 1] += 1
    return cm


def format_confusion_matrix(cm):
    lines = []
    n = cm.shape[0]
    header = ["T\\P"] + [str(i) for i in range(1, n + 1)]
    lines.append("\t".join(header))
    for i in range(n):
        row = [str(i + 1)] + [str(int(x)) for x in cm[i]]
        lines.append("\t".join(row))
    return "\n".join(lines)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(((x - x_mean) * (y - y_mean)).mean() / (x_std * y_std))

# 1) Actual length vs. predicted length

@torch.no_grad()
def analyze_true_vs_pred_length(model, loader, device, max_cdr3_len, out_dir):
    model.eval()

    y_true = []
    y_pred = []
    all_probs = []
    rows = []

    for batch in loader:
        x = batch["x"].to(device)
        x_len = batch["x_len"].to(device)
        a = batch["a"].to(device)
        a_mask = batch["a_mask"].to(device)

        # Deterministic analysis using posterior mean
        len_logits, mu_q, logvar_q, ha, z_cond = model.get_length_logits_from_posterior_mean(x, a, a_mask)
        len_probs = F.softmax(len_logits, dim=-1)
        pred_len = torch.argmax(len_probs, dim=-1) + 1

        for i in range(x.size(0)):
            t = int(x_len[i].item())
            p = int(pred_len[i].item())
            probs_i = len_probs[i].detach().cpu()

            y_true.append(t)
            y_pred.append(p)
            all_probs.append(probs_i.numpy())

            rows.append({
                "antigen_seq": batch["antigen_seq"][i],
                "cdr3_seq": batch["cdr3_seq"][i],
                "true_len": t,
                "pred_len": p,
                "abs_error": abs(t - p),
                "top5_pred_lengths": str(topk_length_probs(probs_i, k=5)),
            })

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
    acc = float(np.mean(y_true_np == y_pred_np))
    acc_pm1 = float(np.mean(np.abs(y_true_np - y_pred_np) <= 1))
    corr = pearson_corr(y_true_np, y_pred_np)

    cm = build_confusion_matrix(y_true, y_pred, max_cdr3_len)

    summary = []
    summary.append("=== True length vs Predicted length ===")
    summary.append(f"Num samples: {len(y_true)}")
    summary.append(f"MAE: {mae:.4f}")
    summary.append(f"Exact accuracy: {acc:.4f}")
    summary.append(f"±1 accuracy: {acc_pm1:.4f}")
    summary.append(f"Pearson correlation: {corr:.4f}")
    summary.append("")
    summary.append("Confusion matrix (rows=true length, cols=pred length):")
    summary.append(format_confusion_matrix(cm))
    summary_text = "\n".join(summary)

    print(summary_text)

    save_text(os.path.join(out_dir, "01_true_vs_pred_summary.txt"), summary_text)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "01_true_vs_pred_details.csv"), index=False)
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "01_confusion_matrix.csv"), index=False)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "all_probs": all_probs,
        "summary_text": summary_text,
    }

# 2) Length distribution under different antigen conditions

@torch.no_grad()
def analyze_antigen_conditioned_length_distribution(model, dataset, device, max_cdr3_len, out_dir, max_unique_antigens_to_report=None):
    model.eval()

    antigen_to_indices = defaultdict(list)
    for idx, (antigen_seq, cdr3_seq) in enumerate(dataset.samples):
        antigen_to_indices[antigen_seq].append(idx)

    unique_antigens = list(antigen_to_indices.keys())
    if max_unique_antigens_to_report is not None:
        unique_antigens = unique_antigens[:max_unique_antigens_to_report]

    rows = []
    report_lines = []
    report_lines.append("=== Antigen-conditioned predicted length distributions ===")
    report_lines.append(f"Num unique antigens analyzed: {len(unique_antigens)}")
    report_lines.append("")

    for antigen_seq in unique_antigens:
        indices = antigen_to_indices[antigen_seq]

        sample = dataset[indices[0]]
        a = sample["a"].unsqueeze(0).to(device)
        a_mask = sample["a_mask"].unsqueeze(0).to(device)

        len_logits, mu_p, logvar_p, ha, z_cond = model.get_length_logits_from_prior_mean(a, a_mask)
        len_probs = F.softmax(len_logits, dim=-1).squeeze(0).cpu()

        pred_len = int(torch.argmax(len_probs).item()) + 1
        top5 = topk_length_probs(len_probs, k=5)

        # True length distribution
        true_lengths = [min(len(dataset.samples[i][1]), max_cdr3_len) for i in indices]
        true_counter = Counter(true_lengths)

        report_lines.append("-" * 100)
        report_lines.append(f"Antigen length: {len(antigen_seq)}")
        report_lines.append(f"Num paired CDR3 samples in dataset: {len(indices)}")
        report_lines.append(f"Predicted top length = {pred_len}")
        report_lines.append(f"Top5 predicted length probs = {top5}")
        report_lines.append(f"Observed true length counts = {dict(sorted(true_counter.items()))}")
        report_lines.append(f"Antigen sequence = {antigen_seq}")

        row = {
            "antigen_seq": antigen_seq,
            "antigen_len": len(antigen_seq),
            "num_paired_cdr3": len(indices),
            "pred_top_len": pred_len,
            "true_length_counts": str(dict(sorted(true_counter.items()))),
        }
        for L in range(1, max_cdr3_len + 1):
            row[f"pred_prob_len_{L}"] = float(len_probs[L - 1].item())
        rows.append(row)

    report_text = "\n".join(report_lines)
    print(report_text)

    save_text(os.path.join(out_dir, "02_antigen_conditioned_length_report.txt"), report_text)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "02_antigen_conditioned_length_probs.csv"), index=False)

    return {
        "rows": rows,
        "report_text": report_text,
    }

# 3) latent traversal

@torch.no_grad()
def analyze_latent_traversal_for_length(
    model,
    dataset,
    device,
    latent_dim,
    max_cdr3_len,
    out_dir,
    traversal_num_antigens=5,
    traversal_dims_to_show=12,
    traversal_delta_values=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
):
    model.eval()

    unique_antigens = []
    seen = set()
    for antigen_seq, _ in dataset.samples:
        if antigen_seq not in seen:
            seen.add(antigen_seq)
            unique_antigens.append(antigen_seq)

    unique_antigens = unique_antigens[:traversal_num_antigens]
    dims_to_show = min(traversal_dims_to_show, latent_dim)

    summary_rows = []
    report_lines = []
    report_lines.append("=== Latent traversal for length_head ===")
    report_lines.append(f"Num antigens analyzed: {len(unique_antigens)}")
    report_lines.append(f"Num latent dims analyzed per antigen: {dims_to_show}")
    report_lines.append(f"Traversal deltas: {list(traversal_delta_values)}")
    report_lines.append("")

    for antigen_idx, antigen_seq in enumerate(unique_antigens, 1):
        
        for i, (a_seq, _) in enumerate(dataset.samples):
            if a_seq == antigen_seq:
                sample = dataset[i]
                break

        a = sample["a"].unsqueeze(0).to(device)
        a_mask = sample["a_mask"].unsqueeze(0).to(device)

        len_logits_base, mu_p, logvar_p, ha, z_cond_base = model.get_length_logits_from_prior_mean(a, a_mask)
        base_probs = F.softmax(len_logits_base, dim=-1).squeeze(0)
        base_pred_len = int(torch.argmax(base_probs).item()) + 1

        report_lines.append("=" * 120)
        report_lines.append(f"[Antigen {antigen_idx}] length={len(antigen_seq)}")
        report_lines.append(f"Base predicted length = {base_pred_len}")
        report_lines.append(f"Base top5 = {topk_length_probs(base_probs.cpu(), k=5)}")
        report_lines.append(f"Antigen sequence = {antigen_seq}")
        report_lines.append("")

        for dim in range(dims_to_show):
            dim_results = []
            pred_lens = []

            for delta in traversal_delta_values:
                z_mod = mu_p.clone()
                z_mod[:, dim] += delta

                z_cond = model.decoder_input(torch.cat([z_mod, ha], dim=-1))
                len_logits = model.length_head(z_cond)
                len_probs = F.softmax(len_logits, dim=-1).squeeze(0)
                pred_len = int(torch.argmax(len_probs).item()) + 1

                pred_lens.append(pred_len)
                dim_results.append({
                    "delta": float(delta),
                    "pred_len": pred_len,
                    "top3": topk_length_probs(len_probs.cpu(), k=3),
                })

            length_range = max(pred_lens) - min(pred_lens)
            trend = "flat"
            if pred_lens[-1] > pred_lens[0]:
                trend = "toward_longer"
            elif pred_lens[-1] < pred_lens[0]:
                trend = "toward_shorter"

            report_lines.append(f"Dim {dim:02d} | range={length_range} | trend={trend}")
            for item in dim_results:
                report_lines.append(
                    f"  delta={item['delta']:>4.1f} -> pred_len={item['pred_len']:>2d}, top3={item['top3']}"
                )

            for item in dim_results:
                summary_rows.append({
                    "antigen_index": antigen_idx,
                    "antigen_seq": antigen_seq,
                    "latent_dim": dim,
                    "delta": item["delta"],
                    "pred_len": item["pred_len"],
                    "base_pred_len": base_pred_len,
                    "length_range_for_dim": length_range,
                    "trend": trend,
                })

            report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    save_text(os.path.join(out_dir, "03_latent_traversal_report.txt"), report_text)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "03_latent_traversal_summary.csv"), index=False)

    return {
        "summary_rows": summary_rows,
        "report_text": report_text,
    }

def summarize_length_sensitive_latent_dims(summary_rows, out_dir):

    if len(summary_rows) == 0:
        return

    df = pd.DataFrame(summary_rows)

    key_df = df.drop_duplicates(subset=["antigen_index", "latent_dim"])[
        ["antigen_index", "latent_dim", "length_range_for_dim", "trend"]
    ]

    agg = key_df.groupby("latent_dim").agg(
        mean_length_range=("length_range_for_dim", "mean"),
        max_length_range=("length_range_for_dim", "max"),
        num_antigens=("antigen_index", "nunique"),
    ).reset_index()

    longer_counts = key_df[key_df["trend"] == "toward_longer"].groupby("latent_dim").size()
    shorter_counts = key_df[key_df["trend"] == "toward_shorter"].groupby("latent_dim").size()
    flat_counts = key_df[key_df["trend"] == "flat"].groupby("latent_dim").size()

    agg["toward_longer_count"] = agg["latent_dim"].map(longer_counts).fillna(0).astype(int)
    agg["toward_shorter_count"] = agg["latent_dim"].map(shorter_counts).fillna(0).astype(int)
    agg["flat_count"] = agg["latent_dim"].map(flat_counts).fillna(0).astype(int)

    agg = agg.sort_values(["mean_length_range", "max_length_range"], ascending=False)

    report_lines = []
    report_lines.append("=== Summary of latent dimensions affecting length ===")
    report_lines.append(agg.to_string(index=False))

    report_text = "\n".join(report_lines)
    print(report_text)

    save_text(os.path.join(out_dir, "04_latent_dim_summary.txt"), report_text)
    agg.to_csv(os.path.join(out_dir, "04_latent_dim_summary.csv"), index=False)

# Main

def main():
    ensure_dir(cfg.out_dir)

    print(f"Loading checkpoint from: {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)

    model_cfg = ckpt["config"]
    max_antigen_len = ckpt["max_antigen_len"]

    print("Checkpoint config:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    print(f"  max_antigen_len(from ckpt): {max_antigen_len}")

    dataset = AntigenCDR3Dataset(
        csv_path=cfg.csv_path,
        antigen_col=cfg.antigen_col,
        cdr3_col=cfg.cdr3_col,
        max_cdr3_len=model_cfg["max_cdr3_len"],
        max_antigen_len=max_antigen_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = ConditionalCNNVAE(model_cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("\n[1/4] Analyze true length vs predicted length...")
    result_1 = analyze_true_vs_pred_length(
        model=model,
        loader=loader,
        device=cfg.device,
        max_cdr3_len=model_cfg["max_cdr3_len"],
        out_dir=cfg.out_dir,
    )

    print("\n[2/4] Analyze antigen-conditioned length distribution...")
    result_2 = analyze_antigen_conditioned_length_distribution(
        model=model,
        dataset=dataset,
        device=cfg.device,
        max_cdr3_len=model_cfg["max_cdr3_len"],
        out_dir=cfg.out_dir,
        max_unique_antigens_to_report=cfg.max_unique_antigens_to_report,
    )

    print("\n[3/4] Analyze latent traversal for length...")
    result_3 = analyze_latent_traversal_for_length(
        model=model,
        dataset=dataset,
        device=cfg.device,
        latent_dim=model_cfg["latent_dim"],
        max_cdr3_len=model_cfg["max_cdr3_len"],
        out_dir=cfg.out_dir,
        traversal_num_antigens=cfg.traversal_num_antigens,
        traversal_dims_to_show=cfg.traversal_dims_to_show,
        traversal_delta_values=cfg.traversal_delta_values,
    )

    print("\n[4/4] Summarize latent dimensions that affect length...")
    summarize_length_sensitive_latent_dims(
        summary_rows=result_3["summary_rows"],
        out_dir=cfg.out_dir,
    )

    print("\nDone.")
    print(f"All outputs saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()

