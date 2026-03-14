import os
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config

@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    max_cdr3_len: int = 30
    pretrain_ckpt: str = "vae_cdrh3_pretrain_varlen.pt"

    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5

    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    antigen_hidden_dim: int = 128
    fusion_dim: int = 128
    antigen_num_layers: int = 3
    num_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.1

    beta_kl: float = 0.1
    kl_anneal_epochs: int = 10
    length_loss_weight: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "conditional_cvae_scratch.pt"
    #save_path: str = "conditional_cvae_finetune.pt"


cfg = Config()

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
    def __init__(self, csv_path, antigen_col, cdr3_col, max_cdr3_len):
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
        self.max_antigen_len = max(len(a) for a, _ in self.samples)

        print(f"Loaded {len(self.samples)} paired samples")
        print(f"Max antigen length = {self.max_antigen_len}")
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
        return x, x_len, a, a_mask, a_len

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

# Pretrained-compatible CDR3 encoder / decoder

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

        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

    def encode_feature(self, x):
        emb = self.embedding(x)       # [B, L, E]
        emb = emb.transpose(1, 2)     # [B, E, L]
        h = self.input_proj(emb)      # [B, H, L]
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)  # [B, H*L]
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
        logits = self.output_proj(h)     # [B, V, L]
        logits = logits.transpose(1, 2)  # [B, L, V]
        return logits

# Antigen encoder

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
        emb = self.embedding(a)          # [B, La, E]
        emb = emb.transpose(1, 2)        # [B, E, La]
        h = self.input_proj(emb)         # [B, H, La]
        for block in self.blocks:
            h = block(h)

        mask = a_mask.unsqueeze(1).float()   # [B,1,La]
        h = h * mask
        pooled = h.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return pooled                        # [B,H]

# Conditional CVAE

class ConditionalCNNVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
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
            vocab_size=VOCAB_SIZE,
            max_seq_len=cfg.max_cdr3_len,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        self.antigen_encoder = AntigenEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.antigen_hidden_dim,
            num_layers=cfg.antigen_num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

        seq_feat_dim = cfg.hidden_dim * cfg.max_cdr3_len
        ant_feat_dim = cfg.antigen_hidden_dim

        # posterior q(z|x,a)
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

        # prior p(z|a)
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

        # fuse z + antigen embedding -> conditional latent for decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(cfg.latent_dim + ant_feat_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
        )

        # length from conditional latent
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

# Load pretrained backbone

def load_pretrained_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    enc_state = {}
    dec_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder."):]] = v
        elif k.startswith("decoder."):
            dec_state[k[len("decoder."):]] = v

    missing_e, unexpected_e = model.encoder.load_state_dict(enc_state, strict=False)
    missing_d, unexpected_d = model.decoder.load_state_dict(dec_state, strict=False)

    print("Loaded pretrained encoder/decoder")
    print("Encoder missing:", missing_e)
    print("Encoder unexpected:", unexpected_e)
    print("Decoder missing:", missing_d)
    print("Decoder unexpected:", unexpected_d)
    
# Loss

def conditional_vae_loss(
    logits,
    targets,
    true_lengths,
    mu_q,
    logvar_q,
    mu_p,
    logvar_p,
    len_logits,
    beta=1.0,
    len_weight=0.2,
):
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_IDX,
        reduction="mean",
    )

    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1,
        dim=1
    )
    kl = kl.mean()

    len_targets = true_lengths - 1
    len_loss = F.cross_entropy(len_logits, len_targets, reduction="mean")

    total = recon + beta * kl + len_weight * len_loss
    return total, recon, kl, len_loss


def get_beta(epoch, max_beta, anneal_epochs):
    if anneal_epochs <= 0:
        return max_beta
    return max_beta * min(1.0, epoch / anneal_epochs)

# Utils

def decode_tokens(token_ids):
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)

# Train / Eval

def train_one_epoch(model, loader, optimizer, device, beta, len_weight):
    model.train()
    total_loss = total_recon = total_kl = total_len = 0.0

    for x, x_len, a, a_mask, a_len in loader:
        x = x.to(device)
        x_len = x_len.to(device)
        a = a.to(device)
        a_mask = a_mask.to(device)

        optimizer.zero_grad()
        logits, mu_q, logvar_q, mu_p, logvar_p, len_logits, z_cond = model(x, a, a_mask)

        loss, recon, kl, len_loss = conditional_vae_loss(
            logits, x, x_len, mu_q, logvar_q, mu_p, logvar_p, len_logits,
            beta=beta, len_weight=len_weight
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_len += len_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_len / n


@torch.no_grad()
def evaluate(model, loader, device, beta, len_weight):
    model.eval()
    total_loss = total_recon = total_kl = total_len = 0.0

    for x, x_len, a, a_mask, a_len in loader:
        x = x.to(device)
        x_len = x_len.to(device)
        a = a.to(device)
        a_mask = a_mask.to(device)

        logits, mu_q, logvar_q, mu_p, logvar_p, len_logits, z_cond = model(x, a, a_mask)

        loss, recon, kl, len_loss = conditional_vae_loss(
            logits, x, x_len, mu_q, logvar_q, mu_p, logvar_p, len_logits,
            beta=beta, len_weight=len_weight
        )

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_len += len_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_len / n


@torch.no_grad()
def show_reconstructions(model, dataset, device, num_examples=5):
    model.eval()
    print("\nConditional reconstructions:")
    for i in range(min(num_examples, len(dataset))):
        x, x_len, a, a_mask, a_len = dataset[i]

        x = x.unsqueeze(0).to(device)
        a = a.unsqueeze(0).to(device)
        a_mask = a_mask.unsqueeze(0).to(device)

        logits, _, _, _, _, len_logits, _ = model(x, a, a_mask)
        pred_tokens = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        pred_len = int(len_logits.argmax(dim=-1).item()) + 1

        antigen_seq = decode_tokens(a.squeeze(0).cpu().tolist()[:a_len.item()])
        input_seq = decode_tokens(x.squeeze(0).cpu().tolist()[:x_len.item()])
        recon_seq = decode_tokens(pred_tokens[:pred_len])

        print(f"antigen(len={a_len.item():2d}): {antigen_seq}")
        print(f"input  (len={x_len.item():2d}): {input_seq}")
        print(f"recon  (len={pred_len:2d}): {recon_seq}")
        print("-" * 80)

# Main

def main():
    dataset = AntigenCDR3Dataset(
        cfg.csv_path,
        cfg.antigen_col,
        cfg.cdr3_col,
        cfg.max_cdr3_len,
    )

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ConditionalCNNVAE(cfg).to(cfg.device)
#    load_pretrained_backbone(model, cfg.pretrain_ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        beta = get_beta(epoch, cfg.beta_kl, cfg.kl_anneal_epochs)

        train_loss, train_recon, train_kl, train_len = train_one_epoch(
            model, train_loader, optimizer, cfg.device, beta, cfg.length_loss_weight
        )
        val_loss, val_recon, val_kl, val_len = evaluate(
            model, val_loader, cfg.device, beta, cfg.length_loss_weight
        )

        print(
            f"Epoch {epoch:02d} | beta={beta:.4f} | "
            f"train_loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} len={train_len:.4f} | "
            f"val_loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} len={val_len:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "stoi": stoi,
                    "itos": itos,
                    "max_antigen_len": dataset.max_antigen_len,
                },
                cfg.save_path,
            )
            print(f"Saved best conditional model to {cfg.save_path}")

    show_reconstructions(model, dataset, cfg.device, num_examples=5)


if __name__ == "__main__":
    main()

