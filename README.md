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

## Evaluation

The evaluation script runs on the public Hugging Face dataset
`noahschaffer/clasp-audioset-subset`, which contains the 100-example `eval`
split used for quick verification. You do not need to download raw AudioSet
audio to run evaluation.

Evaluation reports:

- spectrogram-to-text retrieval: `R@1`, `R@5`, `R@10`, median rank, mean rank
- text-to-spectrogram retrieval: `R@1`, `R@5`, `R@10`, median rank, mean rank
- zero-shot AudioSet multi-label classification: mean average precision (`mAP`)

### Install evaluation dependencies

If you used the conda/uv setup above, the main dependencies should already be
installed. The evaluation script also needs `peft` and `accelerate` when loading
a LoRA fine-tuned adapter:

```bash
python -m pip install peft accelerate
```

If you are using a local `.venv`, make sure the command uses that environment's
Python:

```bash
./.venv/bin/python3 -m pip install peft accelerate
./.venv/bin/python3 -c "import torch, transformers, datasets, peft, accelerate; print('eval deps ok')"
```

### Verify the evaluation CLI

```bash
python clasp/eval/eval.py --help
```

For a local `.venv`:

```bash
./.venv/bin/python3 clasp/eval/eval.py --help
```

### Run the frozen CLIP baseline

This checks that model loading, dataset loading, retrieval evaluation, and
classification evaluation all work end to end.

```bash
python clasp/eval/eval.py \
  --hub_repo noahschaffer/clasp-audioset-subset \
  --split eval \
  --batch_size 16 \
  --output_json results/clasp_baseline_eval.json
```

On a CPU-only laptop, add `--device cpu` and optionally use a temporary cache:

```bash
./.venv/bin/python3 clasp/eval/eval.py \
  --hub_repo noahschaffer/clasp-audioset-subset \
  --split eval \
  --batch_size 16 \
  --cache_dir /private/tmp/clasp_hf_cache \
  --device cpu \
  --output_json /private/tmp/clasp_baseline_eval.json
```

A successful run should print lines like:

```text
Loaded 100 records from noahschaffer/clasp-audioset-subset (eval)
--- Retrieval ---
Spectrogram -> Text | R@1: ...
Text -> Spectrogram | R@1: ...
--- Classification ---
mAP: ...
```

### Evaluate a fine-tuned LoRA adapter

After fine-tuning, the training script should save an adapter directory such as
`clasp-finetuned/`. Evaluate it by adding `--lora_path`:

```bash
python clasp/eval/eval.py \
  --lora_path clasp-finetuned \
  --hub_repo noahschaffer/clasp-audioset-subset \
  --split eval \
  --batch_size 32 \
  --output_json results/clasp_lora_eval.json
```

In a CUDA environment, the script automatically uses CUDA when available. Use
the same command without `--device cpu` for full GPU evaluation.

### Notes on external code and checkpoints

Evaluation uses Hugging Face Transformers for CLIP model loading, the LAION
checkpoint `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, Hugging Face Datasets for
`noahschaffer/clasp-audioset-subset`, and PEFT when loading LoRA adapters. The
retrieval and multi-label AudioSet mAP evaluation logic in `clasp/eval/eval.py`
is project code.
This project uses the processed Hugging Face dataset
`noahschaffer/clasp-audioset-subset`.

