import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import pairwise2


GENERATED_FILE = "generated_cdrh3_from_antigenfinetune.txt"
NATURAL_FILE = "CoV-AbDab.csv"

TARGET_ANTIGEN = (
"RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

MAX_GENERATED = 30
MAX_NATURAL = 30

def read_generated():

    seqs = []
    pattern = re.compile(r"^\d+\s+len=\d+\s+([A-Z]+)")

    with open(GENERATED_FILE) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                seqs.append(m.group(1))

    seqs = list(set(seqs))

    return seqs[:MAX_GENERATED]

def read_natural():

    df = pd.read_csv(NATURAL_FILE)

    df = df[df["antigen"] == TARGET_ANTIGEN]

    seqs = df["cdr3"].dropna().unique().tolist()

    return seqs[:MAX_NATURAL]

def seq_identity(a, b):

    align = pairwise2.align.globalxx(a, b, one_alignment_only=True)[0]

    matches = align.score
    length = max(len(a), len(b))

    return matches / length

def main():

    generated = read_generated()
    natural = read_natural()

    print("Generated:", len(generated))
    print("Natural:", len(natural))

    matrix = np.zeros((len(generated), len(natural)))

    for i, g in enumerate(generated):
        for j, n in enumerate(natural):

            matrix[i, j] = seq_identity(g, n)

    df = pd.DataFrame(
        matrix,
        index=generated,
        columns=natural
    )

    plt.figure(figsize=(12,8))

    sns.heatmap(
        df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Sequence Identity"}
    )

    plt.xlabel("Natural CDRH3")
    plt.ylabel("Generated CDRH3")
    plt.title("CDRH3 Sequence Alignment Similarity")

    plt.tight_layout()

    plt.savefig(
        "cdrh3_alignment_heatmap.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()
