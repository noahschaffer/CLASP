import os
import json
import argparse
import numpy as np
import matplotlib.cm as cm

import torch
import torchaudio
import torchaudio.transforms as T

from pathlib import Path
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image, Sequence

# ---------------------------------------------------------------------------
# Spectrogram config
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
IMG_SIZE = 224

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
amplitude_to_db = T.AmplitudeToDB(top_db=80)

def audio_to_spectrogram(audio_path, img_size=IMG_SIZE):
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono, shape (1, T)

    mel = mel_transform(waveform)
    mel_db = amplitude_to_db(mel).squeeze().numpy()  # (n_mels, time)

    # Normalize to [0, 1] and flip so low frequencies are at the bottom
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_norm = np.flipud(mel_norm)

    # Apply viridis colormap and convert to uint8 RGB
    colored = (cm.viridis(mel_norm)[:, :, :3] * 255).astype(np.uint8)

    return PILImage.fromarray(colored).resize((img_size, img_size), PILImage.BILINEAR)

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_records(audio_dir, captions_json, labels_csv=None):
    """
    Assembles records from:
      - audio_dir: directory of downloaded .wav files (ytid_start.wav)
      - captions_json: output of caption_audioset.py
      - labels_csv: optional path to balanced_train_segments.csv or eval_segments.csv
                    for ground-truth AudioSet labels
    """
    with open(captions_json) as f:
        captions = json.load(f)

    # Build label lookup from CSV if provided
    label_lookup = {}
    if labels_csv is not None:
        import csv
        with open(labels_csv) as f:
            lines = [l for l in f if not l.startswith('#')]
        reader = csv.DictReader(
            lines,
            fieldnames=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
            skipinitialspace=True
        )
        for row in reader:
            if row['YTID'] == 'YTID':
                continue
            ytid = row['YTID'].strip()
            start = int(float(row['start_seconds'].strip()))
            key = f"{ytid}_{start}"
            labels = [l.strip().strip('"') for l in row['positive_labels'].split(',')]
            label_lookup[key] = labels

    records = []
    for audio_path in sorted(Path(audio_dir).glob("*.wav")):
        stem = audio_path.stem  # e.g. "--PJHxphWEs_30"
        caption = captions.get(stem)
        if caption is None:
            continue  # skip failed captions

        parts = stem.rsplit("_", 1)
        ytid = parts[0]
        start = float(parts[1]) if len(parts) == 2 else 0.0

        records.append({
            "audio_path": str(audio_path),
            "caption": caption,
            "labels": label_lookup.get(stem, []),
            "ytid": ytid,
            "start": start,
        })

    return records


def build_dataset(records):
    def generate():
        for i, r in enumerate(records):
            try:
                image = audio_to_spectrogram(r["audio_path"])
            except Exception as e:
                print(f"Skipping {r['ytid']}: {e}", flush=True)
                continue

            yield {
                "image": image,
                "caption": r["caption"],
                "label": r["labels"],
                "ytid": r["ytid"],
                "start": r["start"],
            }

            if i % 500 == 0:
                print(f"[{i}/{len(records)}] processed", flush=True)

    features = Features({
        "image": Image(),
        "caption": Value("string"),
        "label": Sequence(Value("string")),
        "ytid": Value("string"),
        "start": Value("float32"),
    })

    return Dataset.from_generator(generate, features=features)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_dir", default="./audioset_balanced")
    parser.add_argument("--eval_dir", default="./audioset_eval")
    parser.add_argument("--balanced_captions", default="captions_balanced.json")
    parser.add_argument("--eval_captions", default="captions_eval.json")
    parser.add_argument("--balanced_csv", default="balanced_train_segments.csv")
    parser.add_argument("--eval_csv", default="eval_segments.csv")
    parser.add_argument("--hub_repo", required=True,
                        help="HuggingFace repo to push to, e.g. username/clasp-audioset")
    parser.add_argument("--subsample_train", type=int, default=None,
                        help="Only process first N records (for testing)")
    parser.add_argument("--subsample_eval", type=int, default=None,
                        help="Only process first N records (for testing)")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    print("Building balanced split...", flush=True)
    balanced_records = build_records(args.balanced_dir, args.balanced_captions, args.balanced_csv)
    if args.subsample_train:
        balanced_records = balanced_records[:args.subsample_train]
    balanced_ds = build_dataset(balanced_records)
    print(f"Balanced split: {len(balanced_ds)} examples", flush=True)

    if not args.skip_eval:
        print("Building eval split...", flush=True)
        eval_records = build_records(args.eval_dir, args.eval_captions, args.eval_csv)
        if args.subsample_eval:
            eval_records = eval_records[:args.subsample_eval]
        eval_ds = build_dataset(eval_records)
        print(f"Eval split: {len(eval_ds)} examples", flush=True)

        from datasets import DatasetDict
        ds = DatasetDict({"train": balanced_ds, "eval": eval_ds})
    else:
        ds = balanced_ds

    print(f"Pushing to Hub: {args.hub_repo}", flush=True)
    ds.push_to_hub(args.hub_repo, private=True)
    print("Done.", flush=True)