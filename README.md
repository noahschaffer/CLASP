# CLASP: Contrastive Language-Audio-Spectrogram Pretraining

## Environment Setup

Create a conda environment with Python 3.12, FFmpeg 6, and uv:

```bash
conda create -n clasp python=3.12 ffmpeg=6 uv
conda activate clasp
```

Install dependencies with uv:

```bash
uv pip sync requirements.txt
```

## Generating paired spectrogram/caption data

### Audio Flamingo Captioning

### CLASP HF dataset

### CLASP DataLoader

## Fine-tuning CLIP on CLASP data