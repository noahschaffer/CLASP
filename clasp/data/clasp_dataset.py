from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset
import torch

DEFAULT_HUB_REPO = "noahschaffer/clasp-audioset-subset"


def _normalize_cache_dir(cache_dir):
    return None if cache_dir in (None, "") else cache_dir


def _load_split_from_disk(dataset_dir, split):
    dataset = load_from_disk(str(dataset_dir))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not found in local dataset at {dataset_dir}. "
                f"Available splits: {list(dataset.keys())}"
            )
        return dataset[split]
    if split != "train":
        raise ValueError(
            f"Local dataset at {dataset_dir} is not a DatasetDict; only split='train' is supported."
        )
    return dataset


class CLASPDataset(Dataset):
    def __init__(
        self,
        hub_repo=DEFAULT_HUB_REPO,
        processor=None,
        split="train",
        subsample=None,
        cache_dir=None,
        dataset_dir=None,
    ):
        if processor is None:
            raise ValueError("CLASPDataset requires a CLIPProcessor instance.")

        self.processor = processor
        dataset_dir = Path(dataset_dir) if dataset_dir else None
        if dataset_dir is not None and dataset_dir.exists():
            self.data = _load_split_from_disk(dataset_dir, split)
            source = str(dataset_dir)
        else:
            self.data = load_dataset(
                hub_repo,
                split=split,
                cache_dir=_normalize_cache_dir(cache_dir),
            )
            source = hub_repo

        if subsample is not None:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        print(f"Loaded {len(self.data)} records from {source} ({split})", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        inputs = self.processor(
            text=r["caption"],
            images=r["image"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "ytid": r["ytid"],
            "start": r["start"],
            "label": r["label"],
        }


def collate_clasp_batch(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "ytid": [item["ytid"] for item in batch],
        "start": torch.tensor([item["start"] for item in batch], dtype=torch.float32),
        "label": [item["label"] for item in batch],
    }
