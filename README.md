# Antigen-specific antibody design from sequence via coupled generation and interaction prediction

<p align="center">
  <img src="data/raw/fig.png" width="700">
</p>

The framework combines:

- CDRH3 VAE pretraining
- Conditional generation from antigen sequences
- ESM-2 cross-attention binding prediction
- Developability-aware ranking
- Interactive Gradio Web App

## Framework Architecture
Antigen Sequence → [ Conditional CVAE ] → Generated CDRH3 candidates → [ ESM2 Cross-Attention Model ] → Binding Scores → [ Developability Ranking ] → Final Antibody Candidates

## Hardware Requirements
### Recommended
- GPU: NVIDIA A100
- CPU: ≥8 cores
- RAM: ≥32 GB
- Storage: ≥20 GB

### System Requirements
- OS: Ubuntu 20.04 / Linux / macOS
- Python ≥ 3.9

### Dependencies
- torch
- Transformers ≥ 4.30
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- gradio==4.44.1
- gradio_client==1.3.0
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

## Dataset: covid_human_heavy_cdr3_aa_unique_len4_30.txt

This file contains a non-redundant collection of human SARS-CoV-2-associated heavy-chain CDRH3 amino acid sequences curated from the Observed Antibody Space (OAS) database.

### Description
- **Processing steps**: 
  1. removed empty entries
  2. removed sequences containing non-canonical amino acid characters
  3. removed duplicate sequences globally across all files
  4. retained only sequences with lengths between **4 and 30 amino acids**
     
Download: https://drive.google.com/file/d/1n46ld31QrC9oYlZVsR7JZsoOgX_TFupc/view?usp=drive_link.

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

##  Web Application
Launch:

```bash
Antibody_Design_Application/train/python app_gradio.py
```

Open:

```bash
http://127.0.0.1:7860
```

### Contact

If you have any questions about this repository, please contact:

**[Fanxu Meng](mailto:f.meng@vu.nl)**
