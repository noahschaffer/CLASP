import os
import json
import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

CAPTION_PROMPT = (
    "Describe all the sounds present in this audio clip in detail. "
    "For each sound, describe what is producing it and how it sounds. "
    "Be specific and descriptive."
)

def load_model(model_id):
    print(f"Loading model {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.", flush=True)
    return model, processor

def generate_caption(audio_path, model, processor):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(dim=0)  # mono

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": waveform.numpy()},
            {"type": "text", "text": CAPTION_PROMPT}
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    return processor.decode(output[0], skip_special_tokens=True)

def caption_clips(audio_dir, output_json, model, processor, subsample=None):
    audio_files = sorted(Path(audio_dir).glob("*.wav"))

    if subsample is not None:
        audio_files = audio_files[:subsample]
        print(f"Subsampling to {subsample} clips", flush=True)

    if os.path.exists(output_json):
        with open(output_json) as f:
            captions = json.load(f)
    else:
        captions = {}

    pending = [f for f in audio_files if f.stem not in captions]
    print(f"Captioning {len(pending)} clips ({len(captions)} already done)", flush=True)

    for i, audio_path in enumerate(pending):
        try:
            caption = generate_caption(str(audio_path), model, processor)
            captions[audio_path.stem] = caption
        except Exception as e:
            print(f"Failed {audio_path.stem}: {e}", flush=True)
            captions[audio_path.stem] = None

        # Save incrementally every 100 clips
        if i % 100 == 0:
            with open(output_json, "w") as f:
                json.dump(captions, f, indent=2)
            print(f"[{i}/{len(pending)}] saved checkpoint", flush=True)

    # Final save
    with open(output_json, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"Done. {sum(v is not None for v in captions.values())} captions generated.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/audio-flamingo-3-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Only caption the first N clips (for testing)")
    parser.add_argument("--balanced_dir", default="./audioset_balanced")
    parser.add_argument("--eval_dir", default="./audioset_eval")
    parser.add_argument("--balanced_out", default="captions_balanced.json")
    parser.add_argument("--eval_out", default="captions_eval.json")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only caption the balanced set")
    args = parser.parse_args()

    model, processor = load_model(args.model)

    caption_clips(args.balanced_dir, args.balanced_out, model, processor,
                  subsample=args.subsample)

    if not args.skip_eval:
        caption_clips(args.eval_dir, args.eval_out, model, processor,
                      subsample=args.subsample)