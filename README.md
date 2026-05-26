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

## Download the Dataset

This project uses the processed Hugging Face dataset
`noahschaffer/clasp-audioset-subset`.

Download the `train` and `eval` dataset splits:

```bash
python scripts/download_clasp_dataset.py
```

This creates:

- `data/hf_cache/`: Hugging Face download cache
- `data/clasp-audioset-subset/`: local dataset copy

The dataset rows contain:

- `image`: spectrogram image
- `caption`: text caption
- `label`: AudioSet label IDs
- `ytid`: YouTube ID
- `start`: clip start time

Verify the download:

```bash
python -c "from datasets import load_from_disk; ds = load_from_disk('data/clasp-audioset-subset'); print(ds)"
```

If you only want the Hugging Face cache and not the extra local copy, run:

```bash
python scripts/download_clasp_dataset.py --no_save_to_disk
```

## Fine-Tune CLIP

After downloading the dataset, run LoRA fine-tuning:

```bash
python clasp/train/train.py
```

For a quick smoke test:

```bash
python clasp/train/train.py --subsample 8 --epochs 1 --batch_size 2 --num_workers 0
```

The training script saves the fine-tuned model to `clasp-finetuned/` by default.

### How to Tell Training Is Working

When training starts, you should see:

- `Loaded ... records`: the dataset loaded successfully
- `Trainable parameters: ...`: only a small fraction of CLIP should be trainable
- `batch_loss` and `avg_loss`: contrastive training loss values
- `epoch_avg_loss`: average loss for the full epoch

Example log lines:

```text
Trainable parameters: 1,234,567 / 150,000,000 (0.82%)
epoch 1/10 step 10/32 batch_loss 3.1021 avg_loss 3.2844
epoch 1/10 complete epoch_avg_loss 3.0419 (first epoch)
epoch 2/10 complete epoch_avg_loss 2.7315 (down 0.3104 from previous epoch)
```

Lower loss is better. Step-level `batch_loss` can bounce around, so judge progress
mainly by `epoch_avg_loss`. A random contrastive baseline is roughly
`ln(batch_size)`, so with the default batch size of `32` it is about `3.47`.

The script also writes loss logs to:

```text
clasp-finetuned/train_metrics.csv
```

If the loss becomes `nan`, `inf`, or the epoch average never moves after several
epochs, stop the run and check the learning rate, batch size, and dataset loading.
