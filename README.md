# Sequence Driven Closed Loop Design of Antigen-Specific Antibodies with Interpretable Binding Prediction

<p align="center">
  <img src="data/raw/fig.png" width="700">
</p>

The framework combines:

- CDRH3 VAE pretraining
- Conditional generation from antigen sequences
- ESM-2 cross-attention binding prediction

## Hardware Requirements

The experiments in this repository were conducted on a Linux server with the following hardware:

- GPU: NVIDIA A100
- CPU: ≥8 cores
- RAM: ≥32 GB
- Storage: ≥20 GB 

The code can run on a single GPU.

### Estimated GPU Memory Usage

| Model | GPU Memory |
|------|------------|
CDRH3 VAE | ~4–6 GB |
Conditional CDRH3 VAE | ~6–8 GB |
ESM-2 Cross-Attention Binding Model | ~10–14 GB |

### Minimum Requirements

The code can also run on smaller GPUs (e.g., RTX 2080) by reducing:

- 'batch_size'
- 'max_heavy_len'
- 'max_antigen_len'

## System Requirements

### Operating System

The code has been tested on:

- Ubuntu 20.04
- Linux-based HPC environments

### Python

Python version:

- Python ≥ 3.9

### Python Dependencies

Major Python libraries used in this project:

- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- ANARCI (for CDRH3 extraction)

## Installation

To install the required packages for running the code, use the following command:

## Pretrained Models

The checkpoint provided in this repository:
```bash
checkpoints/best_esm2_cross_attention.pt
```
was trained using the small ESM2 model (esm2_t12_35M_UR50D) for demonstration and reproducibility.

Due to file size limitations, checkpoints trained with larger ESM2 models (esm2_t33_650M_UR50D) are hosted on Google Drive:

Download larger pretrained models (best_esm2_cross_attention_regression_fixed_antigen.pt, best_esm2_cross_attention_regression.pt, best_esm2_cross_attention.pt): 

https://drive.google.com/file/d/14ZK1tzs6QaPVj8i74B2Rzhb3JpxOE25r/view?usp=drive_link, https://drive.google.com/file/d/1ZZQzJYHQ37Zc1KjwqAsiiYMyB8yyORGY/view?usp=drive_link, https://drive.google.com/file/d/1SdkpORkcsUErk5c2iiNBYlkyTVKrbPLN/view?usp=drive_link.

After downloading, place the checkpoints in the `checkpoints/` directory.


## How to Train and Use H3-AbSeqVAE

This repository provides scripts for training models, generating antibody CDRH3 sequences, and performing downstream analysis.

##  Train the Models

To pretrain the variational autoencoder on CDRH3 sequences:

```bash
python code/train/train_cdrh3_vae.py
```

To train the conditional VAE for antigen-conditioned CDRH3 generation:

```bash
python code/train/train_conditional_cvae.py
```

To train the antibody–antigen binding prediction model based on ESM2 and cross-attention:

```bash
python code/train/train_esm2_cross_attention.py
```

##  Generate CDRH3 Sequences from Antigens

After training the conditional VAE, candidate CDRH3 sequences can be generated using:

```bash
python code/train/generate_cdrh3_from_antigen.py
```

Generated sequences and predicted binding scores will be saved to:

```bash
data/processed/
```

##  Run Analysis

Various analysis scripts are provided to study the learned latent space and generated sequences.

Extract Latent Representations
```bash
python code/analysis/extract_latent_vectors.py
```
Latent Space Visualization
```bash
python code/analysis/compare_latent_pca.py
```
Length Prediction Analysis
```bash
python code/analysis/analyze_length_head.py
```
CDRH3 Similarity Heatmap
```bash
python code/analysis/cdrh3_similarity_heatmap.py
```

Output Files

Generated results and intermediate outputs are stored in:

```bash
data/processed/
```

### Contact

If you have any questions about this repository, please contact:

**[Fanxu Meng](mailto:f.meng@vu.nl)**
