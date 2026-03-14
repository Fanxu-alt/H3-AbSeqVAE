import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

# Residual Block

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

# Encoder

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

# Decoder

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
        h = self.fc(z)  # [B, H*L]
        h = h.view(z.size(0), self.hidden_dim, self.max_seq_len)  # [B, H, L]
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)          # [B, V, L]
        logits = logits.transpose(1, 2)       # [B, L, V]
        return logits

# VAE

class CNNVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            max_seq_len=config["max_seq_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )
        self.decoder = CNNDecoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=VOCAB_SIZE,
            max_seq_len=config["max_seq_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )
        self.length_head = nn.Sequential(
            nn.Linear(config["latent_dim"], config["latent_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["latent_dim"], config["max_seq_len"]),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def decode_tokens(token_ids):
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


@torch.no_grad()
def generate_sequences(
    model,
    latent_dim,
    max_seq_len,
    num_samples,
    device,
    temperature=1.0,
    sample_mode="sample",
    min_len=5,
):
    model.eval()

    z = torch.randn(num_samples, latent_dim, device=device)

    logits = model.decoder(z)          # [B, L, V]
    len_logits = model.length_head(z)  # [B, L]

    if temperature != 1.0:
        logits = logits / temperature

    pred_lens = torch.argmax(len_logits, dim=-1) + 1
    pred_lens = torch.clamp(pred_lens, min=min_len, max=max_seq_len)

    if sample_mode == "sample":
        probs = torch.softmax(logits, dim=-1)
        preds = torch.multinomial(
            probs.reshape(-1, probs.size(-1)), num_samples=1
        ).view(num_samples, max_seq_len)
    else:
        preds = logits.argmax(dim=-1)

    seqs = []
    lengths = []

    for i in range(num_samples):
        L = int(pred_lens[i].item())
        token_ids = preds[i][:L].cpu().tolist()
        seq = decode_tokens(token_ids)
        seqs.append(seq)
        lengths.append(len(seq))

    return seqs, lengths


def main():
    ckpt_path = "vae_cdrh3_pretrain_varlen.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_samples = 10
    temperature = 1.0
    sample_mode = "sample"   # "sample" or "argmax"
    min_len = 5

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = CNNVAE(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seqs, lengths = generate_sequences(
        model=model,
        latent_dim=config["latent_dim"],
        max_seq_len=config["max_seq_len"],
        num_samples=num_samples,
        device=device,
        temperature=temperature,
        sample_mode=sample_mode,
        min_len=min_len,
    )

    print(f"Generated {len(seqs)} sequences:\n")
    for i, (seq, L) in enumerate(zip(seqs, lengths), 1):
        print(f"{i:02d}\tlen={L}\t{seq}")


if __name__ == "__main__":
    main()

