from datasets import Dataset, Features, Value, Image, Sequence
import pandas as pd
from PIL import Image as PILImage

def build_dataset(records):
    # records is a list of dicts with keys: image_path, caption, labels, ytid, start
    def generate():
        for r in records:
            yield {
                "image": PILImage.open(r["image_path"]).convert("RGB"),
                "caption": r["caption"],
                "label": r["labels"],  # list of strings
                "ytid": r["ytid"],
                "start": r["start"],
            }

    features = Features({
        "image": Image(),
        "caption": Value("string"),
        "label": Sequence(Value("string")),
        "ytid": Value("string"),
        "start": Value("float32"),
    })

    ds = Dataset.from_generator(generate, features=features)
    return ds

ds = build_dataset(records)
ds.push_to_hub("your-username/audioset-balanced-spectrograms", private=True)