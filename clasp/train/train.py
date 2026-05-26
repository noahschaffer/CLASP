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


def non_negative_int(value):
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed_value


def optional_string(value):
    return None if value in (None, "", "none", "None") else value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on CLASP data with LoRA.")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--hub_repo", default=DEFAULT_HUB_REPO)
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Optional path to a local Hugging Face dataset saved with save_to_disk().",
    )
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
    parser.add_argument(
        "--eval_split",
        type=optional_string,
        default="eval",
        help="Optional split to evaluate at the end of each epoch. Set to none to disable.",
    )
    parser.add_argument(
        "--eval_retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute retrieval metrics on the eval split at the end of each epoch.",
    )
    parser.add_argument("--eval_subsample", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--log_every",
        type=non_negative_int,
        default=10,
        help="Log step metrics every N batches. Set to 0 to disable step-level logging.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_targets", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for training losses.",
    )
    parser.add_argument("--wandb_project", default="clasp")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases mode when --wandb is enabled.",
    )
    return parser.parse_args(argv)


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


def evaluate_model(model, data_loader, device, compute_retrieval=False):
    if len(data_loader) == 0:
        raise ValueError(
            "Eval loader is empty. Check the eval split, subsample, and batch size before running eval."
        )

    model.eval()
    total_loss = 0.0
    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in data_loader:
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
                raise RuntimeError(f"Non-finite eval loss: {loss.item()}")
            total_loss += loss.item()

            if compute_retrieval:
                all_image_embeds.append(F.normalize(outputs.image_embeds, dim=-1))
                all_text_embeds.append(F.normalize(outputs.text_embeds, dim=-1))

    metrics = {"eval_avg_loss": total_loss / len(data_loader)}
    if compute_retrieval:
        image_embeds = torch.cat(all_image_embeds)
        text_embeds = torch.cat(all_text_embeds)
        sim = image_embeds @ text_embeds.T
        image_to_text = recall_at_k(sim, [1, 5, 10])
        text_to_image = recall_at_k(sim.T, [1, 5, 10])
        metrics.update(
            {
                "eval_i2t_r1": image_to_text["R@1"],
                "eval_i2t_r5": image_to_text["R@5"],
                "eval_i2t_r10": image_to_text["R@10"],
                "eval_t2i_r1": text_to_image["R@1"],
                "eval_t2i_r5": text_to_image["R@5"],
                "eval_t2i_r10": text_to_image["R@10"],
            }
        )

    return metrics


def init_wandb(args, output_dir, device, total_params, trainable_params, steps_per_epoch):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install the 'wandb' package to use --wandb logging.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config={
            **vars(args),
            "device": str(device),
            "steps_per_epoch": steps_per_epoch,
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
    )
    wandb.define_metric("train/global_step")
    for metric_name in (
        "train/batch_loss",
        "train/running_avg_loss",
        "train/epoch_avg_loss",
        "train/learning_rate",
        "train/epoch",
        "eval/epoch_avg_loss",
        "eval/i2t_r1",
        "eval/i2t_r5",
        "eval/i2t_r10",
        "eval/t2i_r1",
        "eval/t2i_r5",
        "eval/t2i_r10",
    ):
        wandb.define_metric(metric_name, step_metric="train/global_step")
    return run


def train_model(
    model,
    train_loader,
    eval_loader,
    optimizer,
    args,
    device,
    metrics_path,
    wandb_run,
):
    metrics_fields = [
        "event",
        "epoch",
        "step",
        "batch_loss",
        "running_avg_loss",
        "epoch_avg_loss",
        "eval_avg_loss",
        "eval_i2t_r1",
        "eval_i2t_r5",
        "eval_i2t_r10",
        "eval_t2i_r1",
        "eval_t2i_r5",
        "eval_t2i_r10",
        "learning_rate",
    ]
    previous_epoch_loss = None

    with metrics_path.open("w", newline="") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        metrics_writer.writeheader()

        if eval_loader is not None:
            baseline_eval_metrics = evaluate_model(
                model,
                eval_loader,
                device,
                compute_retrieval=args.eval_retrieval,
            )
            baseline_summary = (
                f"pre-train eval global_step 0 "
                f"eval_avg_loss {baseline_eval_metrics['eval_avg_loss']:.4f}"
            )
            if args.eval_retrieval:
                baseline_summary += (
                    f" i2t_r1 {baseline_eval_metrics['eval_i2t_r1']:.3f}"
                    f" t2i_r1 {baseline_eval_metrics['eval_t2i_r1']:.3f}"
                )
            print(baseline_summary, flush=True)
            metrics_writer.writerow(
                {
                    "event": "eval",
                    "epoch": 0,
                    "step": 0,
                    "batch_loss": "",
                    "running_avg_loss": "",
                    "epoch_avg_loss": "",
                    "eval_avg_loss": baseline_eval_metrics.get("eval_avg_loss", ""),
                    "eval_i2t_r1": baseline_eval_metrics.get("eval_i2t_r1", ""),
                    "eval_i2t_r5": baseline_eval_metrics.get("eval_i2t_r5", ""),
                    "eval_i2t_r10": baseline_eval_metrics.get("eval_i2t_r10", ""),
                    "eval_t2i_r1": baseline_eval_metrics.get("eval_t2i_r1", ""),
                    "eval_t2i_r5": baseline_eval_metrics.get("eval_t2i_r5", ""),
                    "eval_t2i_r10": baseline_eval_metrics.get("eval_t2i_r10", ""),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/global_step": 0,
                        "train/epoch": 0,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "eval/epoch_avg_loss": baseline_eval_metrics["eval_avg_loss"],
                        **(
                            {
                                "eval/i2t_r1": baseline_eval_metrics["eval_i2t_r1"],
                                "eval/i2t_r5": baseline_eval_metrics["eval_i2t_r5"],
                                "eval/i2t_r10": baseline_eval_metrics["eval_i2t_r10"],
                                "eval/t2i_r1": baseline_eval_metrics["eval_t2i_r1"],
                                "eval/t2i_r5": baseline_eval_metrics["eval_t2i_r5"],
                                "eval/t2i_r10": baseline_eval_metrics["eval_t2i_r10"],
                            }
                            if args.eval_retrieval
                            else {}
                        ),
                    }
                )
            model.train()

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
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch + 1}, step {step}: {loss.item()}"
                    )

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
                global_step = epoch * len(train_loader) + step
                learning_rate = optimizer.param_groups[0]["lr"]

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/global_step": global_step,
                            "train/epoch": epoch + 1,
                            "train/batch_loss": batch_loss,
                            "train/running_avg_loss": running_avg_loss,
                            "train/learning_rate": learning_rate,
                        }
                    )

                should_log_step = args.log_every > 0 and step % args.log_every == 0
                if should_log_step or (args.log_every > 0 and step == len(train_loader)):
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
                            "eval_avg_loss": "",
                            "eval_i2t_r1": "",
                            "eval_i2t_r5": "",
                            "eval_i2t_r10": "",
                            "eval_t2i_r1": "",
                            "eval_t2i_r5": "",
                            "eval_t2i_r10": "",
                            "learning_rate": learning_rate,
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

            eval_metrics = {}
            if eval_loader is not None:
                eval_metrics = evaluate_model(
                    model,
                    eval_loader,
                    device,
                    compute_retrieval=args.eval_retrieval,
                )
                model.train()

            epoch_summary = (
                f"epoch {epoch + 1}/{args.epochs} complete "
                f"epoch_avg_loss {epoch_loss:.4f}"
            )
            if eval_metrics:
                epoch_summary += f" eval_avg_loss {eval_metrics['eval_avg_loss']:.4f}"
                if args.eval_retrieval:
                    epoch_summary += (
                        f" i2t_r1 {eval_metrics['eval_i2t_r1']:.3f}"
                        f" t2i_r1 {eval_metrics['eval_t2i_r1']:.3f}"
                    )
            epoch_summary += f" ({change_text})"
            print(epoch_summary, flush=True)
            metrics_writer.writerow(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "step": len(train_loader),
                    "batch_loss": "",
                    "running_avg_loss": "",
                    "epoch_avg_loss": epoch_loss,
                    "eval_avg_loss": eval_metrics.get("eval_avg_loss", ""),
                    "eval_i2t_r1": eval_metrics.get("eval_i2t_r1", ""),
                    "eval_i2t_r5": eval_metrics.get("eval_i2t_r5", ""),
                    "eval_i2t_r10": eval_metrics.get("eval_i2t_r10", ""),
                    "eval_t2i_r1": eval_metrics.get("eval_t2i_r1", ""),
                    "eval_t2i_r5": eval_metrics.get("eval_t2i_r5", ""),
                    "eval_t2i_r10": eval_metrics.get("eval_t2i_r10", ""),
                    "learning_rate": learning_rate,
                }
            )
            metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/global_step": (epoch + 1) * len(train_loader),
                        "train/epoch": epoch + 1,
                        "train/epoch_avg_loss": epoch_loss,
                        "train/learning_rate": learning_rate,
                        **(
                            {"eval/epoch_avg_loss": eval_metrics["eval_avg_loss"]}
                            if "eval_avg_loss" in eval_metrics
                            else {}
                        ),
                        **(
                            {
                                "eval/i2t_r1": eval_metrics["eval_i2t_r1"],
                                "eval/i2t_r5": eval_metrics["eval_i2t_r5"],
                                "eval/i2t_r10": eval_metrics["eval_i2t_r10"],
                                "eval/t2i_r1": eval_metrics["eval_t2i_r1"],
                                "eval/t2i_r5": eval_metrics["eval_t2i_r5"],
                                "eval/t2i_r10": eval_metrics["eval_t2i_r10"],
                            }
                            if args.eval_retrieval and eval_metrics
                            else {}
                        ),
                    }
                )
            previous_epoch_loss = epoch_loss


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

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    if dataset_dir is not None and not dataset_dir.exists():
        print(
            f"Local dataset not found at {args.dataset_dir}; loading from Hugging Face.",
            flush=True,
        )
        dataset_dir = None

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
    eval_loader = None
    if args.eval_split is not None:
        eval_ds = CLASPDataset(
            hub_repo=args.hub_repo,
            processor=processor,
            split=args.eval_split,
            subsample=args.eval_subsample,
            cache_dir=args.cache_dir,
            dataset_dir=dataset_dir,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
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
    if args.wandb:
        print(
            f"Logging training metrics to Weights & Biases project '{args.wandb_project}' "
            f"in {args.wandb_mode} mode.",
            flush=True,
        )
    wandb_run = init_wandb(
        args,
        output_dir=output_dir,
        device=device,
        total_params=total_params,
        trainable_params=trainable_params,
        steps_per_epoch=len(train_loader),
    )

    try:
        train_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved fine-tuned CLIP model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
