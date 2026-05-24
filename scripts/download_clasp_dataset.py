import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


DEFAULT_HUB_REPO = "noahschaffer/clasp-audioset-subset"
DEFAULT_CACHE_DIR = "data/hf_cache"
DEFAULT_OUTPUT_DIR = "data/clasp-audioset-subset"


def _optional_path(value):
    return None if value in (None, "", "none", "None") else value


def download_splits(hub_repo, splits, cache_dir):
    downloaded = {}
    for split in splits:
        print(f"Downloading {hub_repo} split={split}...", flush=True)
        ds = load_dataset(hub_repo, split=split, cache_dir=_optional_path(cache_dir))
        downloaded[split] = ds
        print(
            f"Loaded {split}: {len(ds)} rows | columns: {', '.join(ds.column_names)}",
            flush=True,
        )
    return DatasetDict(downloaded)


def main():
    parser = argparse.ArgumentParser(
        description="Download the CLASP AudioSet subset from Hugging Face."
    )
    parser.add_argument("--hub_repo", default=DEFAULT_HUB_REPO)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument(
        "--no_save_to_disk",
        action="store_true",
        help="Only populate the Hugging Face cache; do not write a local DatasetDict copy.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    ds = download_splits(args.hub_repo, args.splits, args.cache_dir)

    if not args.no_save_to_disk:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))
        print(f"Saved local dataset copy to {output_dir}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
