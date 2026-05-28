import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import laion_clap

from datasets import load_dataset
from sklearn.metrics import average_precision_score
from tqdm import trange

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub_repo", default="noahschaffer/clasp-audioset")
    parser.add_argument("--cache_dir", default="/dartfs/rc/lab/S/SinghN/shared/hf_cache")
    parser.add_argument("--audio_dir", required=True,
                        help="Directory containing downloaded .wav files")
    parser.add_argument("--class_labels_csv", default="class_labels_indices.csv")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bootstrap", type=int, default=100,
                        help="Number of bootstrap samples for confidence intervals")
    parser.add_argument("--ci", type=int, default=95,
                        help="Confidence interval percentage (default: 95)")
    parser.add_argument("--output_json", default=None,
                        help="Optional path to write all metrics as JSON")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Recall @ K
# ---------------------------------------------------------------------------

def recall_at_k(sim_matrix, ks):
    n = sim_matrix.shape[0]
    ranked = sim_matrix.argsort(dim=-1, descending=True)
    results = {}
    for k in ks:
        correct = (
            ranked[:, :k] == torch.arange(n, device=ranked.device).unsqueeze(1)
        ).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item()
    return results

# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_retrieval(sim_matrix: np.ndarray, ks, n_bootstrap: int = 1000, ci: int = 95) -> dict:
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


def average_precision_binary(target: np.ndarray, score: np.ndarray):
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    tp_cumsum = np.cumsum(sorted_target)
    ranks = np.arange(1, len(sorted_target) + 1)
    precision_at_hits = (tp_cumsum / ranks) * sorted_target
    return float(precision_at_hits.sum() / positives)


def bootstrap_map(scores: np.ndarray, targets: np.ndarray, n_bootstrap: int = 1000, ci: int = 95) -> dict:
    n, n_classes = scores.shape
    maps = []

    # Precompute sort order once
    order = np.argsort(-scores, axis=0)
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    sorted_targets = np.take_along_axis(targets, order, axis=0)
    ranks = np.arange(1, n + 1, dtype=np.float32).reshape(-1, 1)

    for _ in trange(n_bootstrap, desc="Bootstrap mAP"):
        idx = np.random.choice(n, size=n, replace=True)
        s = sorted_scores[idx]
        t = sorted_targets[idx]

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

def encode_retrieval(model, audio_paths, captions, batch_size):
    all_audio_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            batch_caps = captions[i:i+batch_size]

            audio_embeds = model.get_audio_embedding_from_filelist(
                x=batch_paths, use_tensor=True
            )
            text_embeds = model.get_text_embedding(batch_caps, use_tensor=True)

            all_audio_embeds.append(F.normalize(audio_embeds, dim=-1).cpu())
            all_text_embeds.append(F.normalize(text_embeds, dim=-1).cpu())

            torch.cuda.empty_cache()

            if i % 500 == 0:
                print(f"[Retrieval] {i}/{len(audio_paths)}", flush=True)

    audio_embeds = torch.cat(all_audio_embeds)
    text_embeds = torch.cat(all_text_embeds)
    return audio_embeds, text_embeds


def encode_classification(model, audio_paths, labels, class_labels_csv, batch_size):
    df = pd.read_csv(class_labels_csv)
    mid_to_idx = {mid: i for i, mid in enumerate(df["mid"])}
    n_classes = len(df)

    print("Building label embeddings...", flush=True)
    with torch.no_grad():
        prompts = [f"the sound of {name.lower()}" for name in df["display_name"]]
        label_embeds = F.normalize(
            model.get_text_embedding(prompts, use_tensor=True), dim=-1
        ).cpu()

    all_scores = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            audio_embeds = F.normalize(
                model.get_audio_embedding_from_filelist(x=batch_paths, use_tensor=True),
                dim=-1,
            ).cpu()

            sim = (audio_embeds @ label_embeds.T).numpy()
            all_scores.append(sim)

            torch.cuda.empty_cache()

            batch_targets = np.zeros((len(batch_paths), n_classes))
            for j, clip_labels in enumerate(batch_labels):
                for mid in clip_labels:
                    mid = mid.strip().strip('"')
                    if mid in mid_to_idx:
                        batch_targets[j, mid_to_idx[mid]] = 1
            all_targets.append(batch_targets)

            if i % 500 == 0:
                print(f"[Classification] {i}/{len(audio_paths)}", flush=True)

    scores = np.vstack(all_scores)
    targets = np.vstack(all_targets)
    return scores, targets, n_classes


def evaluate_classification(scores, targets, n_classes):
    class_ap = []
    for c in range(n_classes):
        if targets[:, c].sum() > 0:
            ap = average_precision_score(targets[:, c], scores[:, c])
            class_ap.append(ap)
    return {
        "mAP": float(np.mean(class_ap)),
        "n_classes_evaluated": len(class_ap),
    }

def evaluate_retrieval(audio_embeds, text_embeds):
    sim = audio_embeds @ text_embeds.T
    return {
        "audio_to_text": recall_at_k(sim, [1, 5, 10]),
        "text_to_audio": recall_at_k(sim.T, [1, 5, 10]),
    }, sim.cpu().numpy()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load dataset
    print(f"Loading dataset {args.hub_repo} ({args.split})...", flush=True)
    ds = load_dataset(args.hub_repo, split=args.split, cache_dir=args.cache_dir)
    if args.subsample is not None:
        ds = ds.select(range(args.subsample))
    print(f"Loaded {len(ds)} records", flush=True)

    # Reconstruct audio paths from ytid and start
    audio_paths = []
    captions = []
    labels = []
    missing = 0

    for r in ds:
        path = os.path.join(args.audio_dir, f"{r['ytid']}_{int(r['start'])}.wav")
        if not os.path.exists(path):
            missing += 1
            continue
        audio_paths.append(path)
        captions.append(r["caption"])
        labels.append(r["label"])

    print(f"Found {len(audio_paths)} audio files ({missing} missing)", flush=True)

    # Load CLAP model
    print("Loading LAION-CLAP...", flush=True)
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()
    print("CLAP loaded.", flush=True)

    # Retrieval
    print("\nRunning retrieval eval...", flush=True)
    audio_embeds, text_embeds = encode_retrieval(model, audio_paths, captions, args.batch_size)
    retrieval, sim_matrix = evaluate_retrieval(audio_embeds, text_embeds)

    print("Bootstrap CIs for retrieval...", flush=True)
    retrieval_ci_a2t = bootstrap_retrieval(sim_matrix, ks=[1, 5, 10],
                                           n_bootstrap=args.n_bootstrap, ci=args.ci)
    retrieval_ci_t2a = bootstrap_retrieval(sim_matrix.T, ks=[1, 5, 10],
                                           n_bootstrap=args.n_bootstrap, ci=args.ci)

    # Classification
    print("\nRunning classification eval...", flush=True)
    scores, targets, n_classes = encode_classification(
        model, audio_paths, labels, args.class_labels_csv, args.batch_size
    )
    classification = evaluate_classification(scores, targets, n_classes)

    print("Bootstrap CIs for mAP...", flush=True)
    map_ci = bootstrap_map(scores, targets, n_bootstrap=args.n_bootstrap, ci=args.ci)

    # Print results
    print("\n--- Retrieval ---", flush=True)
    for direction, metrics, ci in [
        ("Audio → Text", retrieval["audio_to_text"], retrieval_ci_a2t),
        ("Text → Audio", retrieval["text_to_audio"], retrieval_ci_t2a),
    ]:
        print(f"{direction}", flush=True)
        for k in (1, 5, 10):
            c = ci[f"R@{k}"]
            print(f"  R@{k}: {metrics[f'R@{k}']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}]", flush=True)

    print("\n--- Classification ---", flush=True)
    print(
        f"mAP: {classification['mAP']:.4f} "
        f"[{map_ci['lower']:.4f}, {map_ci['upper']:.4f}] "
        f"(over {classification['n_classes_evaluated']} classes)",
        flush=True,
    )

    metrics = {
        "hub_repo": args.hub_repo,
        "split": args.split,
        "n_examples": len(audio_paths),
        "retrieval": retrieval,
        "retrieval_ci": {
            "audio_to_text": retrieval_ci_a2t,
            "text_to_audio": retrieval_ci_t2a,
        },
        "classification": classification,
        "map_ci": map_ci,
    }

    if args.output_json is not None:
        output_path = args.output_json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote metrics to {output_path}", flush=True)

if __name__ == "__main__":
    main()
