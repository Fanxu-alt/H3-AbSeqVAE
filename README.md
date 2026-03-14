# An interpretable deep learning framework for antigen-guided antibody generation and binding prediction

<p align="center">
  <img src="data/raw/fig1.png" width="700">
</p>

The framework combines:

- CDRH3 VAE pretraining
- Conditional generation from antigen sequences
- ESM2 cross-attention binding prediction

## Hardware Requirements

The experiments in this repository were conducted on a Linux server with the following hardware:

- GPU: NVIDIA A100
- CPU: ≥8 cores
- RAM: ≥32 GB
- Storage: ≥20 GB free disk space

The code can run on a single GPU.

### Estimated GPU Memory Usage

| Model | GPU Memory |
|------|------------|
CDRH3 VAE | ~4–6 GB |
Conditional CDRH3 VAE | ~6–8 GB |
ESM2 Cross-Attention Binding Model | ~10–14 GB |

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
