import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import shutil

YTDLP_PATH = shutil.which("yt-dlp")

def download_clip(row, output_dir):

    time.sleep(random.uniform(0.5, 2.0))

    try:
        ytid = row['YTID'].strip()
        start = float(row['start_seconds'].strip())
        end = float(row['end_seconds'].strip())
    except KeyError as e:
        print(f"KeyError: {e}, row keys: {list(row.keys())}", flush=True)
        return "unknown", "error"

    out_path = os.path.join(output_dir, f"{ytid}_{int(start)}.wav")
    if os.path.exists(out_path):
        return ytid, "skipped"

    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_path = os.path.join(output_dir, f"{ytid}_tmp.%(ext)s")

    print(f"Downloading {ytid}...", flush=True)
    dl_cmd = [
        YTDLP_PATH, "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-color",
        "-o", tmp_path,
        url
    ]
    result = subprocess.run(dl_cmd, timeout=60, env=os.environ.copy())
    if result.returncode != 0:
        return ytid, "unavailable"

    tmp_wav = os.path.join(output_dir, f"{ytid}_tmp.wav")
    trim_cmd = [
        "ffmpeg", "-ss", str(start), "-to", str(end),
        "-i", tmp_wav, "-ar", "22050", "-ac", "1",
        out_path, "-y", "-loglevel", "error"
    ]
    subprocess.run(trim_cmd)
    os.remove(tmp_wav)
    return ytid, "ok"

def download_audioset(csv_path, output_dir, workers=8):
    os.makedirs(output_dir, exist_ok=True)
    print(f"yt-dlp path: {YTDLP_PATH}", flush=True)
    print(f"Opening {csv_path}...", flush=True)
    with open(csv_path) as f:
        lines = [l for l in f if not l.startswith('#')]

    # Strip whitespace from lines and use explicit fieldnames
    reader = csv.DictReader(
        lines,
        fieldnames=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
        skipinitialspace=True
    )
    rows = [row for row in reader if row['YTID'] != 'YTID']  # skip header row if present
    print(f"Read {len(lines)} lines", flush=True)
    print(f"Parsed {len(rows)} rows", flush=True)
    print(f"CSV columns: {list(rows[0].keys())}", flush=True)
    print(f"First row: {rows[0]}", flush=True)
    print(f"Launching {workers} workers...", flush=True)

    results = {"ok": 0, "unavailable": 0, "skipped": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_clip, r, output_dir): r for r in rows}
        for i, future in enumerate(as_completed(futures)):
            ytid, status = future.result()
            results[status] += 1
            if i % 500 == 0:
                print(f"[{i}/{len(rows)}] {results}", flush=True)

    print(f"Done: {results}", flush=True)

if __name__ == "__main__":
    download_audioset("balanced_train_segments.csv", "./audioset_balanced")
    download_audioset("eval_segments.csv", "./audioset_eval")