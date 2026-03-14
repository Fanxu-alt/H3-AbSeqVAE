import re
import pandas as pd
import matplotlib.pyplot as plt

# python -m pip install pandas matplotlib logomaker

import logomaker


TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

GENERATED_FILE = "generated_cdrh3_from_antigenfinetune.txt"
NATURAL_FILE = "CoV-AbDab.csv"

ANTIGEN_COL = "antigen"
CDR3_COL = "cdr3"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def read_generated_cdrh3(txt_path):
    seqs = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^\d+\s+len=\d+\s+([A-Z]+)$", line)
            if m:
                seqs.append(m.group(1))
    return seqs


def read_natural_cdrh3(csv_path, antigen_target, antigen_col="antigen", cdr3_col="cdr3"):
    df = pd.read_csv(csv_path)

    if antigen_col not in df.columns:
        raise ValueError(f"Column '{antigen_col}' not found in {csv_path}")
    if cdr3_col not in df.columns:
        raise ValueError(f"Column '{cdr3_col}' not found in {csv_path}")

    df = df[[antigen_col, cdr3_col]].dropna().copy()
    df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
    df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()

    df = df[df[antigen_col] == antigen_target.upper()].copy()
    seqs = df[cdr3_col].tolist()
    return seqs


def make_frequency_matrix(seqs, amino_acids=AMINO_ACIDS):
    if len(seqs) == 0:
        raise ValueError("No sequences found.")

    max_len = max(len(s) for s in seqs)
    rows = []

    for pos in range(max_len):
        counts = {aa: 0 for aa in amino_acids}
        valid = 0

        for s in seqs:
            if pos < len(s):
                aa = s[pos]
                if aa in counts:
                    counts[aa] += 1
                    valid += 1

        if valid == 0:
            freqs = {aa: 0.0 for aa in amino_acids}
        else:
            freqs = {aa: counts[aa] / valid for aa in amino_acids}

        rows.append(freqs)

    return pd.DataFrame(rows)


def plot_logo(seqs, title, out_png):
    freq_df = make_frequency_matrix(seqs)

    plt.figure(figsize=(max(10, freq_df.shape[0] * 0.35), 4))
    logo = logomaker.Logo(freq_df, shade_below=.5, fade_below=.5)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_title(title, fontsize=14)
    logo.ax.set_xlabel("CDRH3 position")
    logo.ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compare_logo(generated_seqs, natural_seqs, out_png):
    gen_df = make_frequency_matrix(generated_seqs)
    nat_df = make_frequency_matrix(natural_seqs)

    max_len = max(gen_df.shape[0], nat_df.shape[0])

    gen_df = gen_df.reindex(range(max_len), fill_value=0.0)
    nat_df = nat_df.reindex(range(max_len), fill_value=0.0)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(12, max_len * 0.35), 8),
        sharex=True
    )

    logo1 = logomaker.Logo(gen_df, ax=axes[0], shade_below=.5, fade_below=.5)
    logo1.style_spines(visible=False)
    logo1.style_spines(spines=['left', 'bottom'], visible=True)
    axes[0].set_title(f"Artificial antibody CDRH3 (n={len(generated_seqs)})", fontsize=13)
    axes[0].set_ylabel("Frequency")

    logo2 = logomaker.Logo(nat_df, ax=axes[1], shade_below=.5, fade_below=.5)
    logo2.style_spines(visible=False)
    logo2.style_spines(spines=['left', 'bottom'], visible=True)
    axes[1].set_title(f"Natural antibody CDRH3 (n={len(natural_seqs)})", fontsize=13)
    axes[1].set_xlabel("CDRH3 position")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    generated_seqs = read_generated_cdrh3(GENERATED_FILE)
    natural_seqs = read_natural_cdrh3(
        NATURAL_FILE,
        TARGET_ANTIGEN,
        antigen_col=ANTIGEN_COL,
        cdr3_col=CDR3_COL
    )

    print(f"Generated sequences: {len(generated_seqs)}")
    print(f"Natural sequences: {len(natural_seqs)}")

    if len(generated_seqs) == 0:
        raise ValueError("No generated CDRH3 sequences found.")
    if len(natural_seqs) == 0:
        raise ValueError("No natural CDRH3 sequences found for the target antigen.")

    plot_logo(
        generated_seqs,
        title="Artificial antibody CDRH3 sequence logo",
        out_png="logo_generated.png"
    )

    plot_logo(
        natural_seqs,
        title="Natural antibody CDRH3 sequence logo",
        out_png="logo_natural.png"
    )

    plot_compare_logo(
        generated_seqs,
        natural_seqs,
        out_png="logo_compare.png"
    )

    print("Saved:")
    print("  logo_generated.png")
    print("  logo_natural.png")
    print("  logo_compare.png")


if __name__ == "__main__":
    main()
