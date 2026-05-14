from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor

class CLASPDataset(Dataset):
    def __init__(self, hub_repo, processor, split="train", subsample=None):
        self.processor = processor
        self.data = load_dataset(hub_repo, split=split)
        if subsample is not None:
            self.data = self.data.select(range(subsample))
        print(f"Loaded {len(self.data)} records from {hub_repo} ({split})", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        inputs = self.processor(
            text=r["caption"],
            images=r["image"],  # HF Image() feature returns a PIL image directly
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (3, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }