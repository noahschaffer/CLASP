"""Evaluate CLASP retrieval and AudioSet multi-label classification.

This script is intentionally close to ``clasp/train/train.py``:

* it uses the same Hugging Face CLIP checkpoint and ``CLIPProcessor``;
* it reads the same CLASP Hugging Face dataset through ``CLASPDataset``;
* it can evaluate either frozen CLIP or a LoRA adapter produced by training;
* it reports the two evaluation families described in the final report:
  cross-modal retrieval and zero-shot AudioSet classification.

The CLIP model/checkpoint code comes from Hugging Face Transformers. LoRA
loading, when ``--lora_path`` is provided, uses Hugging Face PEFT. The dataset
schema is the project dataset at noahschaffer/clasp-audioset-subset.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

# ``clasp/data/clasp_datset.py`` is the dataset file in this repository. The
# fallback keeps the evaluator compatible if the filename is later corrected.
CLASP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CLASP_ROOT))
try:
    from data.clasp_datset import CLASPDataset
except ModuleNotFoundError:
    from data.clasp_dataset import CLASPDataset


DEFAULT_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_HUB_REPO = "noahschaffer/clasp-audioset-subset"
DEFAULT_CLASS_LABELS_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen CLIP or a CLASP LoRA adapter on retrieval and "
            "AudioSet-style multi-label classification."
        )
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help="Base CLIP checkpoint used by training and evaluation.",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        help=(
            "Optional path to a PEFT LoRA adapter, e.g. clasp-finetuned. "
            "If omitted, the frozen base CLIP model is evaluated."
        ),
    )
    parser.add_argument(
        "--hub_repo",
        default=DEFAULT_HUB_REPO,
        help="Hugging Face dataset repo containing image/caption/label rows.",
    )
    parser.add_argument(
        "--split",
        default="eval",
        help="Dataset split to evaluate. The project subset uses train/eval.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional first-N slice for quick smoke tests.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory for models and dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for maximum portability on laptops.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device, e.g. cuda, mps, or cpu. Defaults to auto.",
    )
    parser.add_argument(
        "--class_labels_csv",
        default=None,
        help=(
            "Optional local AudioSet class_labels_indices.csv. If omitted, "
            "the script tries the official AudioSet URL and then falls back "
            "to labels observed in the evaluated split."
        ),
    )
    parser.add_argument(
        "--class_labels_url",
        default=DEFAULT_CLASS_LABELS_URL,
        help="Official AudioSet class-label CSV URL used when no CSV is given.",
    )
    parser.add_argument(
        "--label_prompt_template",
        default="the sound of {}",
        help="Prompt template used to embed AudioSet class names.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write all metrics as JSON.",
    )
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_clasp_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields while preserving variable-length label lists.

    PyTorch's default collate function expects nested Python lists to have the
    same length. AudioSet examples are multi-label, so ``label`` has variable
    length and needs to stay as a list of lists.
    """

    return {
        "pixel_values": torch.stack([r["pixel_values"] for r in rows]),
        "input_ids": torch.stack([r["input_ids"] for r in rows]),
        "attention_mask": torch.stack([r["attention_mask"] for r in rows]),
        "label": [r["label"] for r in rows],
        "ytid": [r["ytid"] for r in rows],
        "start": [r["start"] for r in rows],
    }


def load_model_and_processor(
    model_id: str,
    lora_path: str | None,
    cache_dir: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, CLIPProcessor]:
    print(f"Loading CLIP checkpoint: {model_id}", flush=True)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    base_model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)

    if lora_path is None:
        print("Evaluating frozen base CLIP weights.", flush=True)
        model: torch.nn.Module = base_model
    else:
        print(f"Loading PEFT LoRA adapter: {lora_path}", flush=True)
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The --lora_path option requires the peft package. Install the "
                "same environment used for training before evaluating adapters."
            ) from exc
        model = PeftModel.from_pretrained(base_model, lora_path)

    model.to(device)
    model.eval()
    return model, processor


def bootstrap_retrieval(sim_matrix: np.ndarray, ks: tuple[int, ...], n_bootstrap: int = 100, ci: int = 95) -> dict:
    n = sim_matrix.shape[0]
    results = {f"R@{k}": [] for k in ks}

    for _ in trange(n_bootstrap, desc="Bootstrap retrieval"):
        idx = np.random.choice(n, size=n, replace=True)
        sim_resampled = sim_matrix[idx]
        ranked = sim_resampled.argsort(axis=-1)[:, ::-1]
        for k in ks:
            correct = (ranked[:, :k] == idx.reshape(-1, 1)).any(axis=1)
            results[f"R@{k}"].append(correct.mean())

    alpha = (100 - ci) / 2
    return {
        k: {
            "mean": float(np.mean(v)),
            "lower": float(np.percentile(v, alpha)),
            "upper": float(np.percentile(v, 100 - alpha)),
        }
        for k, v in results.items()
    }


def bootstrap_map(scores: np.ndarray, targets: np.ndarray, n_bootstrap: int = 100, ci: int = 95) -> dict:
    n, n_classes = scores.shape
    maps = []

    # Precompute sort order once
    order = np.argsort(-scores, axis=0)  # (N, n_classes)
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    sorted_targets = np.take_along_axis(targets, order, axis=0)
    ranks = np.arange(1, n + 1, dtype=np.float32).reshape(-1, 1)

    for _ in trange(n_bootstrap, desc="Bootstrap mAP"):
        idx = np.random.choice(n, size=n, replace=True)

        # Resample and re-sort only the resampled rows
        s = sorted_scores[idx]
        t = sorted_targets[idx]

        # Re-sort resampled rows
        resample_order = np.argsort(-s, axis=0)
        t = np.take_along_axis(t, resample_order, axis=0)

        tp_cumsum = np.cumsum(t, axis=0)
        precision_at_hits = (tp_cumsum / ranks) * t
        positives = t.sum(axis=0)

        with np.errstate(invalid="ignore"):
            ap_per_class = np.where(
                positives > 0,
                precision_at_hits.sum(axis=0) / positives,
                np.nan
            )
        maps.append(float(np.nanmean(ap_per_class)))

    alpha = (100 - ci) / 2
    return {
        "mean": float(np.mean(maps)),
        "lower": float(np.percentile(maps, alpha)),
        "upper": float(np.percentile(maps, 100 - alpha)),
    }

def feature_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying CLIP model, preserving active PEFT adapters."""

    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def feature_tensor(features: Any) -> torch.Tensor:
    """Extract the projected feature tensor across Transformers versions.

    Some Transformers releases return the projected tensor directly from
    ``get_image_features``/``get_text_features``. Newer releases return a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` has been replaced by
    the projected CLIP embedding. Supporting both keeps this evaluator usable in
    the pinned course environment and in newer local environments.
    """

    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Could not extract feature tensor from {type(features)!r}")


@torch.no_grad()
def encode_label_prompts(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    label_names: list[str],
    prompt_template: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Embed all AudioSet class prompts once for zero-shot classification."""

    clip_model = feature_model(model)
    embeds: list[torch.Tensor] = []

    for start in range(0, len(label_names), batch_size):
        names = label_names[start : start + batch_size]
        prompts = [prompt_template.format(name.lower()) for name in names]
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_features = feature_tensor(clip_model.get_text_features(**inputs))
        embeds.append(F.normalize(text_features, dim=-1).cpu())

    return torch.cat(embeds, dim=0).to(device)


@torch.no_grad()
def encode_eval_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    """Encode spectrograms and captions in dataset order.

    Retrieval assumes row i's spectrogram is paired with row i's caption, so the
    dataloader must not shuffle. Classification reuses the image embeddings.
    """

    clip_model = feature_model(model)
    image_embeds: list[torch.Tensor] = []
    text_embeds: list[torch.Tensor] = []
    label_rows: list[list[str]] = []

    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_features = feature_tensor(
            clip_model.get_image_features(pixel_values=pixel_values)
        )
        text_features = feature_tensor(
            clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        image_embeds.append(F.normalize(image_features, dim=-1).cpu())
        text_embeds.append(F.normalize(text_features, dim=-1).cpu())
        label_rows.extend(clean_label_list(labels) for labels in batch["label"])

        if step == 1 or step % 10 == 0 or step == len(loader):
            print(f"Encoded batch {step}/{len(loader)}", flush=True)

    return torch.cat(image_embeds), torch.cat(text_embeds), label_rows


def clean_label_list(labels: Any) -> list[str]:
    """Normalize a dataset label field into a list of AudioSet MID strings."""

    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [labels]
    return [str(label).strip().strip('"') for label in labels if str(label).strip()]


def retrieval_metrics(similarity: torch.Tensor, ks: tuple[int, ...]) -> dict[str, float]:
    """Compute CLIP-style Recall@K and rank statistics for paired rows."""

    n_queries = similarity.shape[0]
    ranked = similarity.argsort(dim=1, descending=True)
    target = torch.arange(n_queries).unsqueeze(1)
    matches = ranked.eq(target)
    ranks = matches.float().argmax(dim=1) + 1

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = matches[:, :k].any(dim=1).float().mean().item()
    metrics["median_rank"] = ranks.float().median().item()
    metrics["mean_rank"] = ranks.float().mean().item()
    return metrics


def evaluate_retrieval(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    """Evaluate spectrogram-caption retrieval in both directions."""

    similarity = image_embeds @ text_embeds.T
    return {
        "spectrogram_to_text": retrieval_metrics(similarity, ks),
        "text_to_spectrogram": retrieval_metrics(similarity.T, ks),
    }


def parse_class_label_rows(handle: Any) -> list[dict[str, str]]:
    """Parse AudioSet class_labels_indices.csv into normalized dictionaries."""

    reader = csv.DictReader(handle)
    rows: list[dict[str, str]] = []
    for row in reader:
        mid = (row.get("mid") or "").strip()
        display_name = (row.get("display_name") or mid).strip()
        if mid:
            rows.append({"mid": mid, "display_name": display_name})
    return rows


def observed_label_rows(label_rows: list[list[str]]) -> list[dict[str, str]]:
    """Fallback label table when the official AudioSet class CSV is unavailable."""

    observed = sorted({mid for labels in label_rows for mid in labels})
    return [{"mid": mid, "display_name": mid} for mid in observed]


def load_class_labels(
    class_labels_csv: str | None,
    class_labels_url: str | None,
    label_rows: list[list[str]],
) -> tuple[list[dict[str, str]], str]:
    """Load 527 AudioSet labels, falling back to observed labels if needed."""

    if class_labels_csv is not None:
        path = Path(class_labels_csv)
        with path.open(newline="") as handle:
            rows = parse_class_label_rows(handle)
        return rows, str(path)

    if class_labels_url:
        try:
            with urllib.request.urlopen(class_labels_url, timeout=30) as response:
                text = response.read().decode("utf-8")
            rows = parse_class_label_rows(io.StringIO(text))
            return rows, class_labels_url
        except Exception as exc:  # noqa: BLE001 - print reason and use fallback.
            print(
                "Could not load official AudioSet class labels; falling back "
                f"to labels observed in this split. Reason: {exc}",
                flush=True,
            )

    rows = observed_label_rows(label_rows)
    return rows, "observed labels in evaluated split"


def average_precision_binary(target: np.ndarray, score: np.ndarray) -> float | None:
    """Compute average precision for one binary class without sklearn.

    AudioSet classification is multi-label. We score each class independently
    and then average AP across classes that have at least one positive example
    in the evaluated split.
    """

    positives = int(target.sum())
    if positives == 0:
        return None

    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    tp_cumsum = np.cumsum(sorted_target)
    ranks = np.arange(1, len(sorted_target) + 1)
    precision_at_hits = (tp_cumsum / ranks) * sorted_target
    return float(precision_at_hits.sum() / positives)


def evaluate_classification(
    image_embeds: torch.Tensor,
    label_embeds: torch.Tensor,
    label_rows: list[list[str]],
    class_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute zero-shot multi-label classification mAP over AudioSet labels."""

    mid_to_idx = {row["mid"]: idx for idx, row in enumerate(class_rows)}
    targets = np.zeros((len(label_rows), len(class_rows)), dtype=np.float32)

    for row_idx, labels in enumerate(label_rows):
        for mid in labels:
            class_idx = mid_to_idx.get(mid)
            if class_idx is not None:
                targets[row_idx, class_idx] = 1.0

    scores = (image_embeds @ label_embeds.cpu().T).numpy()
    class_ap: list[float] = []
    evaluated_mids: list[str] = []

    for class_idx, class_row in enumerate(class_rows):
        ap = average_precision_binary(targets[:, class_idx], scores[:, class_idx])
        if ap is not None:
            class_ap.append(ap)
            evaluated_mids.append(class_row["mid"])

    return {
        "mAP": float(np.mean(class_ap)) if class_ap else float("nan"),
        "n_classes_total": len(class_rows),
        "n_classes_evaluated": len(class_ap),
        "n_examples": len(label_rows),
        "evaluated_mids": evaluated_mids,
    }


def print_retrieval_block(title: str, metrics: dict[str, float]) -> None:
    print(
        f"{title} | "
        f"R@1: {metrics['R@1']:.3f}  "
        f"R@5: {metrics['R@5']:.3f}  "
        f"R@10: {metrics['R@10']:.3f}  "
        f"MedR: {metrics['median_rank']:.1f}  "
        f"MeanR: {metrics['mean_rank']:.1f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
        device=device,
    )

    print(
        f"Loading dataset {args.hub_repo} split={args.split} "
        f"subsample={args.subsample}",
        flush=True,
    )
    dataset = CLASPDataset(
        args.hub_repo,
        processor,
        split=args.split,
        subsample=args.subsample,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_clasp_batch,
    )

    print(f"Encoding {len(dataset)} examples...", flush=True)
    image_embeds, text_embeds, label_rows = encode_eval_split(model, loader, device)

    # Similarity matrix used for both retrieval and bootstrap
    sim_matrix = (image_embeds @ text_embeds.T).numpy()

    # Retrieval
    print("Evaluating retrieval...", flush=True)
    retrieval = evaluate_retrieval(image_embeds, text_embeds)

    print("Bootstrap CIs for retrieval...", flush=True)
    retrieval_ci_s2t = bootstrap_retrieval(sim_matrix, ks=(1, 5, 10))
    retrieval_ci_t2s = bootstrap_retrieval(sim_matrix.T, ks=(1, 5, 10))

    # Classification
    class_rows, class_label_source = load_class_labels(
        args.class_labels_csv,
        args.class_labels_url,
        label_rows,
    )
    label_names = [row["display_name"] for row in class_rows]
    print(
        f"Embedding {len(label_names)} class prompts from {class_label_source}...",
        flush=True,
    )
    label_embeds = encode_label_prompts(
        model,
        processor,
        label_names,
        args.label_prompt_template,
        device,
        args.batch_size,
    )

    print("Evaluating zero-shot multi-label classification...", flush=True)
    classification = evaluate_classification(
        image_embeds,
        label_embeds,
        label_rows,
        class_rows,
    )

    # Scores and targets for bootstrap mAP
    mid_to_idx = {row["mid"]: idx for idx, row in enumerate(class_rows)}
    targets = np.zeros((len(label_rows), len(class_rows)), dtype=np.float32)
    for row_idx, labels in enumerate(label_rows):
        for mid in labels:
            class_idx = mid_to_idx.get(mid)
            if class_idx is not None:
                targets[row_idx, class_idx] = 1.0
    scores = (image_embeds @ label_embeds.cpu().T).numpy()

    print("Bootstrap CIs for mAP...", flush=True)
    map_ci = bootstrap_map(scores, targets)

    # Print results
    print("\n--- Retrieval ---", flush=True)
    for direction, dir_key, ci in [
        ("Spectrogram -> Text", "spectrogram_to_text", retrieval_ci_s2t),
        ("Text -> Spectrogram", "text_to_spectrogram", retrieval_ci_t2s),
    ]:
        m = retrieval[dir_key]
        print(f"{direction}", flush=True)
        for k in (1, 5, 10):
            c = ci[f"R@{k}"]
            print(f"  R@{k}: {m[f'R@{k}']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}]", flush=True)
        print(f"  MedR: {m['median_rank']:.1f}  MeanR: {m['mean_rank']:.1f}", flush=True)

    print("\n--- Classification ---", flush=True)
    print(
        f"mAP: {classification['mAP']:.4f} "
        f"[{map_ci['lower']:.4f}, {map_ci['upper']:.4f}] "
        f"(over {classification['n_classes_evaluated']} classes)",
        flush=True,
    )

    metrics = {
        "model_id": args.model_id,
        "lora_path": args.lora_path,
        "hub_repo": args.hub_repo,
        "split": args.split,
        "n_examples": len(dataset),
        "class_label_source": class_label_source,
        "retrieval": retrieval,
        "classification": classification,
        "retrieval_ci": {
            "spectrogram_to_text": retrieval_ci_s2t,
            "text_to_spectrogram": retrieval_ci_t2s,
        },
        "map_ci": map_ci,
    }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {output_path}", flush=True)


if __name__ == "__main__":
    main()