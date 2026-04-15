import re
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer

from binder_model import BinderConfig, ESM2BidirectionalCrossAttentionClassifier


class AntibodyBinder:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_dict = ckpt.get("config", {})

        self.cfg = BinderConfig(**{
            k: v for k, v in cfg_dict.items() if k in BinderConfig.__annotations__
        })

        self.model = ESM2BidirectionalCrossAttentionClassifier(
            model_name=self.cfg.model_name,
            hidden_dim=self.cfg.hidden_dim,
            num_heads=self.cfg.num_heads,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.best_val_auc = ckpt.get("best_val_auc", None)

    def clean_seq(self, seq: str) -> str:
        seq = str(seq).strip().upper()
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)

    @staticmethod
    def add_spaces(seq: str) -> str:
        return " ".join(list(seq))

    def tokenize_pair(self, heavy_seq: str, antigen_seq: str):
        heavy_seq = self.clean_seq(heavy_seq)
        antigen_seq = self.clean_seq(antigen_seq)

        heavy_enc = self.tokenizer(
            self.add_spaces(heavy_seq),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_heavy_len,
        )

        antigen_enc = self.tokenizer(
            self.add_spaces(antigen_seq),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_antigen_len,
        )

        return {
            "heavy_input_ids": heavy_enc["input_ids"].to(self.device),
            "heavy_attention_mask": heavy_enc["attention_mask"].to(self.device),
            "antigen_input_ids": antigen_enc["input_ids"].to(self.device),
            "antigen_attention_mask": antigen_enc["attention_mask"].to(self.device),
        }

    @torch.no_grad()
    def predict(self, heavy_seq: str, antigen_seq: str) -> dict:
        batch = self.tokenize_pair(heavy_seq, antigen_seq)

        logits, heavy_to_antigen_attn, antigen_to_heavy_attn = self.model(
            heavy_input_ids=batch["heavy_input_ids"],
            heavy_attention_mask=batch["heavy_attention_mask"],
            antigen_input_ids=batch["antigen_input_ids"],
            antigen_attention_mask=batch["antigen_attention_mask"],
        )

        prob = torch.sigmoid(logits)[0].item()

        return {
            "heavy": self.clean_seq(heavy_seq),
            "antigen": self.clean_seq(antigen_seq),
            "logit": float(logits[0].item()),
            "binding_probability": float(prob),
        }

    @torch.no_grad()
    def predict_batch(self, df: pd.DataFrame, heavy_col: str = "heavy", antigen_col: str = "antigen") -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            out = self.predict(
                heavy_seq=row[heavy_col],
                antigen_seq=row[antigen_col],
            )
            rows.append(out)
        return pd.DataFrame(rows)
