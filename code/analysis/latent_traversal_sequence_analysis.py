import os
import csv
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# User config

CKPT_PATH = "conditional_cvae_finetune.pt"

TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Traversal settings
DELTA_VALUES = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
SAMPLE_MODE = "argmax"   # "argmax" or "sample"
TEMPERATURE = 1.0
MIN_LEN = 8

# Sampling repeats per traversal point

NUM_SAMPLES_PER_POINT = 1

MOTIFS = [
    "AR",
    "GG",
    "YY",
    "FDY",
    "GMD",
    "RG",
    "YW",
    "WG",
]

# Output file
RESULTS_CSV = "latent_traversal_results.csv"
SUMMARY_CSV = "latent_traversal_summary.csv"

# Vocabulary

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

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

# Utils

def encode_seq(seq: str, fixed_len: int) -> torch.Tensor:
    seq = seq[:fixed_len]
    ids = [stoi.get(ch, UNK_IDX) for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_IDX] * (fixed_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode_tokens(token_ids: List[int]) -> str:
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


def aa_composition(seq: str) -> Dict[str, float]:

    L = max(len(seq), 1)
    counter = Counter(seq)

    # Single amino acid frequency
    comp = {f"aa_{aa}": counter.get(aa, 0) / L for aa in AMINO_ACIDS}


    groups = {
        "hydrophobic": set("AVILMFWY"),
        "polar": set("STNQCY"),
        "positive": set("KRH"),
        "negative": set("DE"),
        "glycine": set("G"),
        "proline": set("P"),
        "aromatic": set("FWY"),
        "small": set("AGSTC"),
    }

    for name, aa_set in groups.items():
        count = sum(counter.get(aa, 0) for aa in aa_set)
        comp[f"group_{name}"] = count / L

    return comp


def motif_stats(seq: str, motifs: List[str]) -> Dict[str, int]:
    stats = {}
    for motif in motifs:
        count = 0
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            count += 1
            start = idx + 1
        stats[f"motif_{motif}_count"] = count
        stats[f"motif_{motif}_present"] = int(count > 0)
    return stats


@torch.no_grad()
def get_antigen_condition(
    model: ConditionalCNNVAE,
    antigen_seq: str,
    antigen_max_len: int,
    device: str,
):
    a = encode_seq(antigen_seq, antigen_max_len).unsqueeze(0).to(device)
    a_mask = (a != PAD_IDX).long()
    ha = model.antigen_encoder(a, a_mask)
    mu_p = model.prior_mu(ha)
    logvar_p = model.prior_logvar(ha)
    return a, a_mask, ha, mu_p, logvar_p


@torch.no_grad()
def decode_from_given_z(
    model: ConditionalCNNVAE,
    z: torch.Tensor,
    ha: torch.Tensor,
    max_cdr3_len: int,
    min_len: int = 8,
    temperature: float = 1.0,
    sample_mode: str = "argmax",
) -> Tuple[str, int, torch.Tensor, torch.Tensor]:
    """
Decode from the given z and ha, and return:
    - seq
    - pred_len
    - token_logits [1, L, V]
    - len_logits [1, max_len]
    """
    z_cond = model.decoder_input(torch.cat([z, ha], dim=-1))
    token_logits = model.decoder(z_cond)
    len_logits = model.length_head(z_cond)

    if temperature != 1.0:
        token_logits = token_logits / temperature

    pred_len = torch.argmax(len_logits, dim=-1) + 1
    pred_len = int(torch.clamp(pred_len, min=min_len, max=max_cdr3_len).item())

    if sample_mode == "sample":
        probs = torch.softmax(token_logits, dim=-1)  # [1, L, V]
        preds = torch.multinomial(
            probs.reshape(-1, probs.size(-1)), num_samples=1
        ).view(1, max_cdr3_len)
    else:
        preds = token_logits.argmax(dim=-1)

    token_ids = preds[0, :pred_len].cpu().tolist()
    seq = decode_tokens(token_ids)
    return seq, pred_len, token_logits, len_logits


def summarize_records(records: List[Dict], motifs: List[str]) -> List[Dict]:

    grouped = {}
    for r in records:
        key = (r["latent_dim"], r["delta"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for (dim_idx, delta), rows in grouped.items():
        out = {
            "latent_dim": dim_idx,
            "delta": delta,
            "n": len(rows),
            "mean_length": sum(r["pred_len"] for r in rows) / len(rows),
            "mean_group_aromatic": sum(r["group_aromatic"] for r in rows) / len(rows),
            "mean_group_hydrophobic": sum(r["group_hydrophobic"] for r in rows) / len(rows),
            "mean_group_positive": sum(r["group_positive"] for r in rows) / len(rows),
            "mean_group_negative": sum(r["group_negative"] for r in rows) / len(rows),
            "mean_group_glycine": sum(r["group_glycine"] for r in rows) / len(rows),
            "mean_group_proline": sum(r["group_proline"] for r in rows) / len(rows),
        }

        for motif in motifs:
            present_key = f"motif_{motif}_present"
            count_key = f"motif_{motif}_count"
            out[f"frac_{present_key}"] = sum(r[present_key] for r in rows) / len(rows)
            out[f"mean_{count_key}"] = sum(r[count_key] for r in rows) / len(rows)

        summary_rows.append(out)

    summary_rows = sorted(summary_rows, key=lambda x: (x["latent_dim"], x["delta"]))
    return summary_rows


def save_csv(path: str, rows: List[Dict]):
    if not rows:
        print(f"[WARN] No rows to save: {path}")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_changes(summary_rows: List[Dict], top_k: int = 10):

    if not summary_rows:
        return

    def best_by_metric(metric_name, reverse=True):
        valid = [r for r in summary_rows if metric_name in r]
        valid = sorted(valid, key=lambda x: x[metric_name], reverse=reverse)
        return valid[:top_k]

    print("\n" + "=" * 80)
    print("Top traversal points by mean_length")
    print("=" * 80)
    for r in best_by_metric("mean_length", reverse=True):
        print(
            f"dim={r['latent_dim']:02d}, delta={r['delta']:>4}, "
            f"mean_length={r['mean_length']:.3f}"
        )

    print("\n" + "=" * 80)
    print("Top traversal points by mean_group_aromatic")
    print("=" * 80)
    for r in best_by_metric("mean_group_aromatic", reverse=True):
        print(
            f"dim={r['latent_dim']:02d}, delta={r['delta']:>4}, "
            f"aromatic={r['mean_group_aromatic']:.3f}, length={r['mean_length']:.3f}"
        )

    print("\n" + "=" * 80)
    print("Top traversal points by mean_group_positive")
    print("=" * 80)
    for r in best_by_metric("mean_group_positive", reverse=True):
        print(
            f"dim={r['latent_dim']:02d}, delta={r['delta']:>4}, "
            f"positive={r['mean_group_positive']:.3f}, length={r['mean_length']:.3f}"
        )

# Main traversal

@torch.no_grad()
def run_latent_traversal():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]
    antigen_max_len = ckpt["max_antigen_len"]

    latent_dim = config["latent_dim"]
    max_cdr3_len = config["max_cdr3_len"]

    model = ConditionalCNNVAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    _, _, ha, mu_p, logvar_p = get_antigen_condition(
        model=model,
        antigen_seq=TARGET_ANTIGEN,
        antigen_max_len=antigen_max_len,
        device=DEVICE,
    )

    # baseline: z = mu_p(a)
    z_base = mu_p.clone()  # [1, latent_dim]

    records = []

    for sample_idx in range(NUM_SAMPLES_PER_POINT):
        seq, pred_len, _, _ = decode_from_given_z(
            model=model,
            z=z_base,
            ha=ha,
            max_cdr3_len=max_cdr3_len,
            min_len=MIN_LEN,
            temperature=TEMPERATURE,
            sample_mode=SAMPLE_MODE,
        )

        row = {
            "latent_dim": -1,
            "delta": 0.0,
            "sample_idx": sample_idx,
            "pred_len": pred_len,
            "sequence": seq,
        }
        row.update(aa_composition(seq))
        row.update(motif_stats(seq, MOTIFS))
        records.append(row)

    print("=" * 80)
    print("Baseline generation from z = mu_p(a)")
    print("=" * 80)
    for r in records[:NUM_SAMPLES_PER_POINT]:
        print(f"baseline\tlen={r['pred_len']:2d}\t{r['sequence']}")

    # Perform traversal on each latent dimension
    print("\n" + "=" * 80)
    print(f"Running latent traversal over {latent_dim} dimensions ...")
    print("=" * 80)

    for dim_idx in range(latent_dim):
        for delta in DELTA_VALUES:
            for sample_idx in range(NUM_SAMPLES_PER_POINT):
                z_mod = z_base.clone()
                z_mod[0, dim_idx] += delta

                seq, pred_len, _, _ = decode_from_given_z(
                    model=model,
                    z=z_mod,
                    ha=ha,
                    max_cdr3_len=max_cdr3_len,
                    min_len=MIN_LEN,
                    temperature=TEMPERATURE,
                    sample_mode=SAMPLE_MODE,
                )

                row = {
                    "latent_dim": dim_idx,
                    "delta": float(delta),
                    "sample_idx": sample_idx,
                    "pred_len": pred_len,
                    "sequence": seq,
                }
                row.update(aa_composition(seq))
                row.update(motif_stats(seq, MOTIFS))
                records.append(row)

            last = records[-1]
            print(
                f"dim={dim_idx:02d}, delta={delta:>4} -> "
                f"len={last['pred_len']:2d}, seq={last['sequence']}"
            )

    summary_rows = summarize_records(records, MOTIFS)

    save_csv(RESULTS_CSV, records)
    save_csv(SUMMARY_CSV, summary_rows)

    print("\nSaved detailed results to:", RESULTS_CSV)
    print("Saved summary results to:", SUMMARY_CSV)

    print_top_changes(summary_rows, top_k=10)

    print("\n" + "=" * 80)
    print("Per-dimension contrast: delta=+3 vs delta=-3")
    print("=" * 80)

    summary_map = {(r["latent_dim"], r["delta"]): r for r in summary_rows}
    for dim_idx in range(latent_dim):
        r_neg = summary_map.get((dim_idx, -3.0))
        r_pos = summary_map.get((dim_idx, 3.0))
        if r_neg is None or r_pos is None:
            continue

        d_len = r_pos["mean_length"] - r_neg["mean_length"]
        d_aromatic = r_pos["mean_group_aromatic"] - r_neg["mean_group_aromatic"]
        d_positive = r_pos["mean_group_positive"] - r_neg["mean_group_positive"]
        d_gly = r_pos["mean_group_glycine"] - r_neg["mean_group_glycine"]

        print(
            f"dim={dim_idx:02d} | "
            f"Δlen={d_len:+.3f} | "
            f"Δaromatic={d_aromatic:+.3f} | "
            f"Δpositive={d_positive:+.3f} | "
            f"Δglycine={d_gly:+.3f}"
        )


if __name__ == "__main__":
    run_latent_traversal()
