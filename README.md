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

If you saved a local dataset copy with `python scripts/download_clasp_dataset.py`, point training at it with:

```bash
python clasp/train/train.py --dataset_dir data/clasp-audioset-subset
```

For a quick smoke test:

```bash
python clasp/train/train.py --subsample 8 --epochs 1 --batch_size 2 --num_workers 0
```

To log loss curves to Weights & Biases:

```bash
python clasp/train/train.py --wandb --wandb_project clasp
```

If you want to avoid uploading metrics while testing, use offline mode:

```bash
python clasp/train/train.py --wandb --wandb_mode offline
```

The training script saves the fine-tuned model to `clasp-finetuned/` by default.

### How to Tell Training Is Working

When training starts, you should see:

- `Loaded ... records`: the dataset loaded successfully
- `Trainable parameters: ...`: only a small fraction of CLIP should be trainable
- `batch_loss` and `avg_loss`: contrastive training loss values
- `epoch_avg_loss`: average loss for the full epoch
- `pre-train eval global_step 0 ...`: true eval baseline before any updates
- `eval_avg_loss`: average loss on the `eval` split after each epoch
- `i2t_r1` and `t2i_r1`: retrieval recall@1 on the eval split

Example log lines:

```text
Trainable parameters: 1,234,567 / 150,000,000 (0.82%)
pre-train eval global_step 0 eval_avg_loss 3.2148 i2t_r1 0.041 t2i_r1 0.038
epoch 1/10 step 10/32 batch_loss 3.1021 avg_loss 3.2844
epoch 1/10 complete epoch_avg_loss 3.0419 eval_avg_loss 2.9981 i2t_r1 0.182 t2i_r1 0.176 (first epoch)
epoch 2/10 complete epoch_avg_loss 2.7315 eval_avg_loss 2.7024 i2t_r1 0.241 t2i_r1 0.233 (down 0.3104 from previous epoch)
```

Lower loss is better. Step-level `batch_loss` can bounce around, so judge progress
mainly by `eval_avg_loss` plus retrieval metrics like `i2t_r1` and `t2i_r1`. A random contrastive baseline is roughly
`ln(batch_size)`, so with the default batch size of `32` it is about `3.47`.

Training evaluates on the `eval` split by default at the end of each epoch. To disable eval entirely:

```bash
python clasp/train/train.py --eval_split none
```

To keep eval loss but skip retrieval metrics:

```bash
python clasp/train/train.py --no-eval_retrieval
```

The script also writes loss logs to:

```text
clasp-finetuned/train_metrics.csv
```

With `--wandb`, it also logs:

- `train/batch_loss`
- `train/running_avg_loss`
- `train/epoch_avg_loss`
- `train/learning_rate`
- `eval/epoch_avg_loss`
- `eval/i2t_r1`, `eval/i2t_r5`, `eval/i2t_r10`
- `eval/t2i_r1`, `eval/t2i_r5`, `eval/t2i_r10`

If the loss becomes `nan`, `inf`, or the epoch average never moves after several
epochs, stop the run and check the learning rate, batch size, and dataset loading.
