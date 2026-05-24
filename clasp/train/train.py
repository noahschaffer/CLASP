import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.clasp_dataset import (  # noqa: E402
    CLASPDataset,
    DEFAULT_HUB_REPO,
    collate_clasp_batch,
)


MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_CACHE_DIR = "data/hf_cache"
DEFAULT_DATASET_DIR = "data/clasp-audioset-subset"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on CLASP data with LoRA.")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--hub_repo", default=DEFAULT_HUB_REPO)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", default="clasp-finetuned")
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="CSV file for training losses. Defaults to <output_dir>/train_metrics.csv.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", nargs="+", default=["q_proj", "v_proj"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_clip_backbone(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def build_lora_model(args, device):
    model = CLIPModel.from_pretrained(args.model_id)

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    clip_model = get_clip_backbone(model)

    for param in clip_model.visual_projection.parameters():
        param.requires_grad = True
    for param in clip_model.text_projection.parameters():
        param.requires_grad = True
    clip_model.logit_scale.requires_grad = True

    bad_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and "lora_" not in name
        and "visual_projection" not in name
        and "text_projection" not in name
        and "logit_scale" not in name
    ]
    if bad_trainable:
        raise RuntimeError(
            "Unexpected trainable parameters outside LoRA/projections/logit_scale: "
            + ", ".join(bad_trainable[:20])
        )

    return model.to(device)


def contrastive_loss(logits_per_image):
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_image.T, labels)
    return (image_loss + text_loss) / 2


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = build_lora_model(args, device)

    total_params, trainable_params = count_parameters(model)
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)",
        flush=True,
    )

    dataset_dir = args.dataset_dir if Path(args.dataset_dir).exists() else None
    if dataset_dir is None:
        print(
            f"Local dataset not found at {args.dataset_dir}; loading from Hugging Face.",
            flush=True,
        )

    train_ds = CLASPDataset(
        hub_repo=args.hub_repo,
        processor=processor,
        split=args.split,
        subsample=args.subsample,
        cache_dir=args.cache_dir,
        dataset_dir=dataset_dir,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clasp_batch,
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "train_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Logging losses to {metrics_path}. Lower loss is better; "
        f"a random batch-size-{args.batch_size} baseline is about {np.log(args.batch_size):.3f}.",
        flush=True,
    )

    metrics_fields = [
        "event",
        "epoch",
        "step",
        "batch_loss",
        "running_avg_loss",
        "epoch_avg_loss",
        "learning_rate",
    ]
    previous_epoch_loss = None

    with metrics_path.open("w", newline="") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        metrics_writer.writeheader()

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = contrastive_loss(outputs.logits_per_image)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}: {loss.item()}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()

                batch_loss = loss.item()
                running_loss += batch_loss
                running_avg_loss = running_loss / step

                if step % args.log_every == 0 or step == len(train_loader):
                    print(
                        f"epoch {epoch + 1}/{args.epochs} "
                        f"step {step}/{len(train_loader)} "
                        f"batch_loss {batch_loss:.4f} "
                        f"avg_loss {running_avg_loss:.4f}",
                        flush=True,
                    )
                    metrics_writer.writerow(
                        {
                            "event": "step",
                            "epoch": epoch + 1,
                            "step": step,
                            "batch_loss": batch_loss,
                            "running_avg_loss": running_avg_loss,
                            "epoch_avg_loss": "",
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    metrics_file.flush()

            epoch_loss = running_loss / len(train_loader)
            if previous_epoch_loss is None:
                change_text = "first epoch"
            else:
                delta = epoch_loss - previous_epoch_loss
                direction = "down" if delta < 0 else "up"
                change_text = f"{direction} {abs(delta):.4f} from previous epoch"

            print(
                f"epoch {epoch + 1}/{args.epochs} complete "
                f"epoch_avg_loss {epoch_loss:.4f} ({change_text})",
                flush=True,
            )
            metrics_writer.writerow(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "step": len(train_loader),
                    "batch_loss": "",
                    "running_avg_loss": "",
                    "epoch_avg_loss": epoch_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            metrics_file.flush()
            previous_epoch_loss = epoch_loss

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved fine-tuned CLIP model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
