import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Config

CKPT_PATH = "conditional_cvae_finetune.pt"
CSV_PATH = "CoV-AbDab.csv"
ANTIGEN_COL = "antigen"
CDR3_COL = "cdr3"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        df = self.df[[antigen_col, cdr3_col]].dropna().copy()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()

        df = df[
            (df[antigen_col].str.len() > 0) &
            (df[cdr3_col].str.len() > 0)
        ].reset_index(drop=True)

        self.samples = list(zip(df[antigen_col].tolist(), df[cdr3_col].tolist()))
        self.max_antigen_len = max(len(a) for a, _ in self.samples)
        self.max_cdr3_len = max_cdr3_len

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

        z = self.reparameterize(mu_q, logvar_q)
        z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))

        logits = self.decoder(z_cond)
        len_logits = self.length_head(z_cond)

        return logits, len_logits

# Decode / SRR

def decode_tokens(token_ids):
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


@torch.no_grad()
def evaluate_srr(model, loader, device):
    model.eval()

    total_matches = 0
    total_positions = 0
    sample_srrs = []
    exact_match_count = 0
    total_samples = 0

    for x, x_len, a, a_mask, a_len in loader:
        x = x.to(device)
        x_len = x_len.to(device)
        a = a.to(device)
        a_mask = a_mask.to(device)

        logits, len_logits = model(x, a, a_mask)

        pred_tokens = logits.argmax(dim=-1)            # [B, L]
        pred_lens = torch.argmax(len_logits, dim=-1) + 1

        B, L = pred_tokens.shape

        # Positions outside the predicted length are marked as PAD, indicating "no further generation".
        for i in range(B):
            pred_len_i = int(pred_lens[i].item())
            if pred_len_i < L:
                pred_tokens[i, pred_len_i:] = PAD_IDX

        for i in range(B):
            true_len = int(x_len[i].item())
            true_tokens = x[i, :true_len]
            pred_seq_tokens = pred_tokens[i, :true_len]

            matches = (true_tokens == pred_seq_tokens).sum().item()
            srr_i = matches / true_len

            total_matches += matches
            total_positions += true_len
            sample_srrs.append(srr_i)
            total_samples += 1

            # exact match: They are identical within the actual length range, and the predicted length is also equal to the actual length.
            pred_len_i = int(pred_lens[i].item())
            if pred_len_i == true_len and matches == true_len:
                exact_match_count += 1

    srr_micro = total_matches / total_positions
    srr_macro = sum(sample_srrs) / len(sample_srrs)
    exact_match_rate = exact_match_count / total_samples

    return {
        "SRR_micro": srr_micro,
        "SRR_macro": srr_macro,
        "ExactMatchRate": exact_match_rate,
        "TotalSamples": total_samples,
        "TotalPositions": total_positions,
    }


@torch.no_grad()
def show_examples(model, dataset, device, num_examples=5):
    model.eval()
    print("\nExamples:")
    for i in range(min(num_examples, len(dataset))):
        x, x_len, a, a_mask, a_len = dataset[i]

        x = x.unsqueeze(0).to(device)
        a = a.unsqueeze(0).to(device)
        a_mask = a_mask.unsqueeze(0).to(device)

        logits, len_logits = model(x, a, a_mask)
        pred_tokens = logits.argmax(dim=-1).squeeze(0).cpu()
        pred_len = int(torch.argmax(len_logits, dim=-1).item()) + 1

        true_len = int(x_len.item())
        true_seq = decode_tokens(x.squeeze(0).cpu().tolist()[:true_len])
        pred_seq = decode_tokens(pred_tokens.tolist()[:pred_len])

        # SRR for this sample
        compare_len = true_len
        pred_compare = pred_tokens.tolist()[:compare_len]
        # If the prediction length is insufficient, it will be considered a mismatch.
        if len(pred_compare) < compare_len:
            pred_compare += [PAD_IDX] * (compare_len - len(pred_compare))
        true_compare = x.squeeze(0).cpu().tolist()[:compare_len]

        matches = sum(int(t == p) for t, p in zip(true_compare, pred_compare))
        srr_i = matches / true_len

        print(f"true(len={true_len:2d}): {true_seq}")
        print(f"pred(len={pred_len:2d}): {pred_seq}")
        print(f"SRR={srr_i:.4f}")
        print("-" * 70)

# Main

def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]

    model = ConditionalCNNVAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dataset = AntigenCDR3Dataset(
        csv_path=CSV_PATH,
        antigen_col=ANTIGEN_COL,
        cdr3_col=CDR3_COL,
        max_cdr3_len=config["max_cdr3_len"],
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    metrics = evaluate_srr(model, loader, DEVICE)

    print("\n===== SRR Results =====")
    print(f"TotalSamples   : {metrics['TotalSamples']}")
    print(f"TotalPositions : {metrics['TotalPositions']}")
    print(f"SRR_micro      : {metrics['SRR_micro']:.4f}")
    print(f"SRR_macro      : {metrics['SRR_macro']:.4f}")
    print(f"ExactMatchRate : {metrics['ExactMatchRate']:.4f}")

    show_examples(model, dataset, DEVICE, num_examples=5)


if __name__ == "__main__":
    main()

