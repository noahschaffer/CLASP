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
To generate captions with AudioFlamingo, run the script 
```bash
python clasp/data/build_dataset.py
```
Important arguments:
- `--balanced_dir`
  Location of the AudioSet balanced dataset
- `--eval_dir`
  Location of the AudioSet eval dataset
- `--balanced_out`
  Location to write the caption json for AudioSet balanced
- `--eval_out`
  Location to write the caption json for AudioSet eval
### CLASP HF dataset
To generate a HuggingFace Dataset containing Spectrogram/Caption pairs, run 
```bash
python clasp/data/build_dataset.py
```
Important arguments:
- `--balanced_dir`
  Location of the AudioSet balanced dataset
- `--eval_dir`
  Location of the AudioSet eval dataset
- `--balanced_captions`
  Locationn of the caption json for AudioSet balanced
- `--eval_captions`
  Location of the caption json for AudioSet eval
- `--balanced_csv`
  Location of balanced_train_segments.csv. This is provided when AudioSet is downloaded
- `--eval_csv`
  Location of eval_segments.csv. This is provided when AudioSet is downloaded
- `--hub_repo`
  HuggingFace Repo to write the CLASP dataset
### CLASP DataLoader

## Fine-tuning CLIP on CLASP data
