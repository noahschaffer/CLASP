import os
import json
import argparse
from pathlib import Path

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

CAPTION_PROMPT = (
    "Describe the sounds in this audio clip in 1-2 sentences. "
    "Be specific about what is producing each sound. "
    "Do not include timestamps."
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

def generate_captions_batch(audio_paths, model, processor):
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "audio", "path": str(p)},
                {"type": "text", "text": CAPTION_PROMPT},
            ]
        }]
        for p in audio_paths
    ]

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    return processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

def caption_clips(audio_dir, output_json, model, processor, batch_size=8, subsample=None):
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

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        try:
            results = generate_captions_batch(batch, model, processor)
            for audio_path, caption in zip(batch, results):
                captions[audio_path.stem] = caption
        except Exception as e:
            print(f"Batch {i} failed: {e}", flush=True)
            for audio_path in batch:
                captions[audio_path.stem] = None

        if i % 100 == 0:
            with open(output_json, "w") as f:
                json.dump(captions, f, indent=2)
            print(f"[{i}/{len(pending)}] saved checkpoint", flush=True)

    with open(output_json, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"Done. {sum(v is not None for v in captions.values())} captions generated.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/audio-flamingo-3-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of clips to process per batch")
    parser.add_argument("--subsample_train", type=int, default=None,
                        help="Only caption the first N clips (for testing)")
    parser.add_argument("--subsample_eval", type=int, default=None,
                        help="Only caption the first N clips (for testing)")
    parser.add_argument("--balanced_dir", default="./audioset_balanced")
    parser.add_argument("--eval_dir", default="./audioset_eval")
    parser.add_argument("--balanced_out", default="captions_balanced.json")
    parser.add_argument("--eval_out", default="captions_eval.json")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only caption the balanced set")
    args = parser.parse_args()

    print("Loading AudioFlamingo Model", flush=True)
    model, processor = load_model(args.model)

    print("AF loaded, captioning clips", flush=True)
    caption_clips(args.balanced_dir, args.balanced_out, model, processor,
                  batch_size=args.batch_size, subsample=args.subsample_train)

    if not args.skip_eval:
        caption_clips(args.eval_dir, args.eval_out, model, processor,
                      batch_size=args.batch_size, subsample=args.subsample_eval)