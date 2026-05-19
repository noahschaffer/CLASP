import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel
from sklearn.metrics import average_precision_score

from clasp_dataset import CLASPDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA adapter. If None, evaluates base model.")
    parser.add_argument("--hub_repo", default="noahschaffer/clasp-audioset")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--class_labels_csv", default="class_labels_indices.csv")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Retrieval eval
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

def evaluate_retrieval(all_image_embeds, all_text_embeds):
    image_embeds = torch.cat(all_image_embeds)
    text_embeds = torch.cat(all_text_embeds)
    sim = image_embeds @ text_embeds.T

    return {
        "image_to_text": recall_at_k(sim, [1, 5, 10]),
        "text_to_image": recall_at_k(sim.T, [1, 5, 10]),
    }

# ---------------------------------------------------------------------------
# Classification eval
# ---------------------------------------------------------------------------

def build_label_embeddings(model, processor, df, device):
    prompts = [f"the sound of {name.lower()}" for name in df["display_name"]]
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        text_embeds = F.normalize(model.get_text_features(**inputs), dim=-1)
    return text_embeds

def evaluate_classification(all_scores, all_targets, n_classes):
    scores = np.vstack(all_scores)   # (N, n_classes)
    targets = np.vstack(all_targets) # (N, n_classes)

    class_ap = []
    for c in range(n_classes):
        if targets[:, c].sum() > 0:
            ap = average_precision_score(targets[:, c], scores[:, c])
            class_ap.append(ap)

    return {
        "mAP": np.mean(class_ap),
        "n_classes_evaluated": len(class_ap),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load model
    print(f"Loading model {args.model_id}...", flush=True)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    base_model = CLIPModel.from_pretrained(args.model_id).to(device)

    if args.lora_path is not None:
        print(f"Loading LoRA adapter from {args.lora_path}...", flush=True)
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        print("Evaluating base model (no LoRA).", flush=True)
        model = base_model

    model.eval()

    # Load dataset
    ds = CLASPDataset(
        args.hub_repo,
        processor,
        split=args.split,
        subsample=args.subsample,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load AudioSet class labels
    df = pd.read_csv(args.class_labels_csv)
    mid_to_idx = {mid: i for i, mid in enumerate(df["mid"])}
    n_classes = len(df)

    # Pre-compute label embeddings for classification
    print("Building label embeddings...", flush=True)
    label_embeds = build_label_embeddings(model, processor, df, device)

    # Eval loop
    all_image_embeds = []
    all_text_embeds = []
    all_scores = []
    all_targets = []

    print(f"Evaluating {len(ds)} clips...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]  # list of list of MID strings

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Retrieval
            all_image_embeds.append(F.normalize(outputs.image_embeds, dim=-1))
            all_text_embeds.append(F.normalize(outputs.text_embeds, dim=-1))

            # Classification
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            sim = (image_embeds @ label_embeds.T).cpu().numpy()
            all_scores.append(sim)

            batch_targets = np.zeros((len(labels), n_classes))
            for j, clip_labels in enumerate(labels):
                for mid in clip_labels:
                    mid = mid.strip().strip('"')
                    if mid in mid_to_idx:
                        batch_targets[j, mid_to_idx[mid]] = 1
            all_targets.append(batch_targets)

            if i % 10 == 0:
                print(f"[{i}/{len(loader)}]", flush=True)

    # Retrieval results
    retrieval = evaluate_retrieval(all_image_embeds, all_text_embeds)
    i2t = retrieval["image_to_text"]
    t2i = retrieval["text_to_image"]

    # Classification results
    classification = evaluate_classification(all_scores, all_targets, n_classes)

    # Print results
    print("\n--- Retrieval ---", flush=True)
    print(f"Image → Text | R@1: {i2t['R@1']:.3f}  R@5: {i2t['R@5']:.3f}  R@10: {i2t['R@10']:.3f}", flush=True)
    print(f"Text → Image | R@1: {t2i['R@1']:.3f}  R@5: {t2i['R@5']:.3f}  R@10: {t2i['R@10']:.3f}", flush=True)

    print("\n--- Classification ---", flush=True)
    print(f"mAP: {classification['mAP']:.4f}  (over {classification['n_classes_evaluated']} classes)", flush=True)

if __name__ == "__main__":
    main()