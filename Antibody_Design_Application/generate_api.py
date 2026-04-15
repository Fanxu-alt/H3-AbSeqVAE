import re
from pathlib import Path
from typing import List

import pandas as pd
import torch

from generator_model import (
    Config,
    ConditionalCNNVAE,
    ITOS_DEFAULT,
    STOI_DEFAULT,
    PAD_TOKEN,
)


class AntibodyGenerator:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        cfg_dict = ckpt.get("config", {})
        self.cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__annotations__})

        self.stoi = ckpt.get("stoi", STOI_DEFAULT)
        self.itos = ckpt.get("itos", ITOS_DEFAULT)
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.max_antigen_len = ckpt.get("max_antigen_len", 512)

        self.model = ConditionalCNNVAE(
            cfg=self.cfg,
            vocab_size=len(self.itos),
            pad_idx=self.pad_idx,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def clean_seq(self, seq: str) -> str:
        seq = str(seq).strip().upper()
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)

    def encode_antigen(self, antigen: str):
        antigen = self.clean_seq(antigen)
        antigen = antigen[: self.max_antigen_len]

        ids = [self.stoi.get(ch, self.stoi["X"]) for ch in antigen]
        if len(ids) < self.max_antigen_len:
            ids += [self.pad_idx] * (self.max_antigen_len - len(ids))

        a = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        a_mask = (a != self.pad_idx).long()
        return a, a_mask

    def decode_tokens(self, token_ids: List[int]) -> str:
        chars = []
        for idx in token_ids:
            ch = self.itos[idx]
            if ch == "<PAD>":
                continue
            chars.append(ch)
        return "".join(chars)

    @torch.no_grad()
    def generate(
        self,
        antigen: str,
        num_samples: int = 32,
        min_len: int = 5,
        sample_mode: str = "sample",
        temperature: float = 1.0,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        a, a_mask = self.encode_antigen(antigen)

        results = self.model.generate_from_antigen(
            a=a,
            a_mask=a_mask,
            num_samples=num_samples,
            min_len=min_len,
            sample_mode=sample_mode,
            temperature=temperature,
        )

        rows = []
        for preds, pred_len in results:
            seq = self.decode_tokens(preds[0].cpu().tolist()[: int(pred_len.item())])
            rows.append({
                "antigen": self.clean_seq(antigen),
                "cdrh3": seq,
                "pred_len": len(seq),
                "sample_mode": sample_mode,
                "temperature": temperature,
            })

        df = pd.DataFrame(rows)

        if deduplicate:
            df = df.drop_duplicates(subset=["cdrh3"]).reset_index(drop=True)

        return df


if __name__ == "__main__":
    gen = AntibodyGenerator("models/conditional_cvae_finetune.pt")
    df = gen.generate(
        antigen="RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
        num_samples=20,
        temperature=0.8,
    )
    print(df.head())
