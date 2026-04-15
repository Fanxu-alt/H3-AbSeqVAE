import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]
ITOS_DEFAULT = AMINO_ACIDS + SPECIAL_TOKENS
STOI_DEFAULT = {ch: i for i, ch in enumerate(ITOS_DEFAULT)}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "X"


@dataclass
class Config:
    max_cdr3_len: int = 30
    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    antigen_hidden_dim: int = 128
    fusion_dim: int = 128
    antigen_num_layers: int = 3
    num_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.1


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
        pad_idx: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
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
        pad_idx: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
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
    def __init__(self, cfg: Config, vocab_size: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_cdr3_len = cfg.max_cdr3_len

        self.encoder = CNNEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            max_seq_len=cfg.max_cdr3_len,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        self.decoder = CNNDecoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            vocab_size=vocab_size,
            max_seq_len=cfg.max_cdr3_len,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        self.antigen_encoder = AntigenEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.antigen_hidden_dim,
            num_layers=cfg.antigen_num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        seq_feat_dim = cfg.hidden_dim * cfg.max_cdr3_len
        ant_feat_dim = cfg.antigen_hidden_dim

        self.posterior_mu = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )
        self.posterior_logvar = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(cfg.latent_dim + ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )

        self.length_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, cfg.max_cdr3_len),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def generate_from_antigen(self, a, a_mask, num_samples=1, min_len=5, sample_mode="sample", temperature=1.0):
        ha = self.antigen_encoder(a, a_mask)
        mu_p = self.prior_mu(ha)
        logvar_p = self.prior_logvar(ha)

        results = []
        for _ in range(num_samples):
            z = self.reparameterize(mu_p, logvar_p)
            z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))
            logits = self.decoder(z_cond)
            len_logits = self.length_head(z_cond)

            if temperature != 1.0:
                logits = logits / temperature

            pred_len = torch.argmax(len_logits, dim=-1) + 1
            pred_len = torch.clamp(pred_len, min=min_len, max=logits.size(1))

            if sample_mode == "sample":
                probs = torch.softmax(logits, dim=-1)
                preds = torch.multinomial(
                    probs.reshape(-1, probs.size(-1)), num_samples=1
                ).view(logits.size(0), logits.size(1))
            else:
                preds = logits.argmax(dim=-1)

            results.append((preds, pred_len))
        return results
