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
