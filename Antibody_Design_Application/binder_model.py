from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


@dataclass
class BinderConfig:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    max_heavy_len: int = 256
    max_antigen_len: int = 512
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


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

    def get_input_embeddings(self):
        return self.esm.get_input_embeddings()

    def encode_from_ids(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def encode_from_embeds(self, inputs_embeds, attention_mask):
        outputs = self.esm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def _forward_from_hidden(
        self,
        heavy_emb,
        heavy_attention_mask,
        antigen_emb,
        antigen_attention_mask,
    ):
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

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):
        with torch.no_grad():
            heavy_emb = self.encode_from_ids(heavy_input_ids, heavy_attention_mask)
            antigen_emb = self.encode_from_ids(antigen_input_ids, antigen_attention_mask)

        return self._forward_from_hidden(
            heavy_emb=heavy_emb,
            heavy_attention_mask=heavy_attention_mask,
            antigen_emb=antigen_emb,
            antigen_attention_mask=antigen_attention_mask,
        )

    def forward_from_embeds(
        self,
        heavy_inputs_embeds,
        heavy_attention_mask,
        antigen_inputs_embeds,
        antigen_attention_mask,
    ):
        heavy_emb = self.encode_from_embeds(heavy_inputs_embeds, heavy_attention_mask)
        antigen_emb = self.encode_from_embeds(antigen_inputs_embeds, antigen_attention_mask)

        return self._forward_from_hidden(
            heavy_emb=heavy_emb,
            heavy_attention_mask=heavy_attention_mask,
            antigen_emb=antigen_emb,
            antigen_attention_mask=antigen_attention_mask,
        )
