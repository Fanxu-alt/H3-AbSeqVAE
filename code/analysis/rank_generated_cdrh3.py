import re
import torch
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Config

CKPT_PATH = "best_esm2_cross_attention.pt"
GENERATED_FILE = "generated_cdrh3_from_antigenfinetune.txt"
OUTPUT_CSV = "ranked_cdrh3_candidates.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

TEMPLATE_HEAVY = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTVSDNYMSWVRQAPGKGLQWVSVIYSGGNTYYADFVKGRFNITRDDSKNMLYLQMNSLRREDTAVYYCVRDRRIVGYYFGLDVWGQGTTVTVFS"
)

# The original CDRH3 that needs to be replaced
TEMPLATE_CDRH3 = "VRDRRIVGYYFGLDV"

# Target Antigen
TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

# Utils

def masked_mean(x, mask):
    # x: [B, L, D]
    # mask: [B, L]
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def add_spaces(seq: str) -> str:
    return " ".join(list(seq))


def build_heavy_from_cdrh3(template_heavy: str, template_cdrh3: str, new_cdrh3: str) -> str:
    if template_cdrh3 not in template_heavy:
        raise ValueError(f"Template CDRH3 '{template_cdrh3}' not found in TEMPLATE_HEAVY")
    return template_heavy.replace(template_cdrh3, new_cdrh3, 1)


def read_generated_cdrh3(path: str):

    seqs = []
    pattern = re.compile(r"^\s*\d+\s+len=(\d+)\s+([A-Z]+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                pred_len = int(m.group(1))
                seq = m.group(2)
                seqs.append((pred_len, seq))

    return seqs


def filter_cdrh3(cdrh3_items, min_len=8, max_len=40):
    """
   Filtering rules:

- Deduplication

- Remove characters containing "X"

- Length limit

- Only keep standard uppercase amino acid characters
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seen = set()
    kept = []

    for pred_len, seq in cdrh3_items:
        if seq in seen:
            continue
        if "X" in seq:
            continue
        if not (min_len <= len(seq) <= max_len):
            continue
        if any(ch not in valid_aa for ch in seq):
            continue

        seen.add(seq)
        kept.append({
            "pred_len": pred_len,
            "cdrh3": seq,
            "cdrh3_len": len(seq),
        })

    return kept

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
            key_padding_mask=key_padding_mask,  # True means ignore
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
        return outputs.last_hidden_state  # [B, L, D]

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

# Scoring

@torch.no_grad()
def score_heavy_antigen_pairs(
    model,
    tokenizer,
    heavy_list,
    antigen_seq,
    max_heavy_len,
    max_antigen_len,
    batch_size=8,
    device="cpu",
):
    model.eval()
    all_scores = []

    for i in range(0, len(heavy_list), batch_size):
        batch_heavy = heavy_list[i:i + batch_size]
        cur_bs = len(batch_heavy)

        heavy_inputs = tokenizer(
            [add_spaces(seq) for seq in batch_heavy],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_heavy_len,
        )

        antigen_inputs = tokenizer(
            [add_spaces(antigen_seq)] * cur_bs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_antigen_len,
        )

        heavy_input_ids = heavy_inputs["input_ids"].to(device)
        heavy_attention_mask = heavy_inputs["attention_mask"].to(device)
        antigen_input_ids = antigen_inputs["input_ids"].to(device)
        antigen_attention_mask = antigen_inputs["attention_mask"].to(device)

        logits, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        all_scores.extend(probs)

    return all_scores

# Main

def main():
    print(f"Using device: {DEVICE}")

    if TEMPLATE_CDRH3 not in TEMPLATE_HEAVY:
        raise ValueError("TEMPLATE_CDRH3 is not in TEMPLATE_HEAVY, please check.")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]

    print("Loading model...")
    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    print("Reading generated CDRH3...")
    raw_items = read_generated_cdrh3(GENERATED_FILE)
    print(f"Found {len(raw_items)} raw generated sequences")

    filtered_items = filter_cdrh3(raw_items, min_len=8, max_len=40)
    print(f"Kept {len(filtered_items)} unique valid CDRH3 sequences after filtering")

    if len(filtered_items) == 0:
        raise ValueError("No CDRH3 sequences were found after filtering.")

    print("Building full heavy sequences...")
    full_heavy_list = []
    cdrh3_list = []
    cdrh3_len_list = []
    pred_len_list = []

    for item in filtered_items:
        cdrh3 = item["cdrh3"]
        full_heavy = build_heavy_from_cdrh3(
            template_heavy=TEMPLATE_HEAVY,
            template_cdrh3=TEMPLATE_CDRH3,
            new_cdrh3=cdrh3,
        )

        pred_len_list.append(item["pred_len"])
        cdrh3_len_list.append(item["cdrh3_len"])
        cdrh3_list.append(cdrh3)
        full_heavy_list.append(full_heavy)

    print("Scoring...")
    scores = score_heavy_antigen_pairs(
        model=model,
        tokenizer=tokenizer,
        heavy_list=full_heavy_list,
        antigen_seq=TARGET_ANTIGEN,
        max_heavy_len=config["max_heavy_len"],
        max_antigen_len=config["max_antigen_len"],
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    df = pd.DataFrame({
        "cdrh3": cdrh3_list,
        "pred_len_from_generator": pred_len_list,
        "actual_cdrh3_len": cdrh3_len_list,
        "full_heavy": full_heavy_list,
        "antigen": [TARGET_ANTIGEN] * len(cdrh3_list),
        "binding_score": scores,
    })

    df = df.sort_values("binding_score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nTop 20 candidates:")
    print(df[["cdrh3", "actual_cdrh3_len", "binding_score"]].head(20).to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
