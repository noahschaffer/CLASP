import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig

from clasp.dataset.clasp_dataset import CLASPDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUB_REPO = "noahschaffer/clasp-audioset"
CACHE_DIR = ""
MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.01

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "vision_model.encoder.layers.*.self_attn.q_proj",
        "vision_model.encoder.layers.*.self_attn.v_proj",
        "text_model.encoder.layers.*.self_attn.q_proj",
        "text_model.encoder.layers.*.self_attn.v_proj",
    ],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)

for param in model.visual_projection.parameters():
    param.requires_grad = True
for param in model.text_projection.parameters():
    param.requires_grad = True
model.logit_scale.requires_grad = True

model = model.to(DEVICE)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_ds = CLASPDataset(HUB_REPO, processor, split="train", cache_dir=CACHE_DIR)
eval_ds = CLASPDataset(HUB_REPO, processor, split="eval", cache_dir=CACHE_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# Loss + optimizer
# ---------------------------------------------------------------------------

def contrastive_loss(logits):
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def recall_at_k(sim_matrix, ks):
    n = sim_matrix.shape[0]
    ranked = sim_matrix.argsort(dim=-1, descending=True)
    results = {}
    for k in ks:
        correct = (ranked[:, :k] == torch.arange(n, device=ranked.device).unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item()
    return results

def evaluate(loader):
    model.eval()
    total_loss = 0
    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = contrastive_loss(outputs.logits_per_image)
            total_loss += loss.item()

            all_image_embeds.append(F.normalize(outputs.image_embeds, dim=-1))
            all_text_embeds.append(F.normalize(outputs.text_embeds, dim=-1))

    image_embeds = torch.cat(all_image_embeds)
    text_embeds = torch.cat(all_text_embeds)
    sim = image_embeds @ text_embeds.T

    retrieval = {
        "image_to_text": recall_at_k(sim, [1, 5, 10]),
        "text_to_image": recall_at_k(sim.T, [1, 5, 10]),
    }

    return total_loss / len(loader), retrieval

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = contrastive_loss(outputs.logits_per_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    eval_loss, retrieval = evaluate(eval_loader)

    i2t = retrieval["image_to_text"]
    t2i = retrieval["text_to_image"]

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"train loss: {train_loss:.4f} | eval loss: {eval_loss:.4f} | "
        f"I→T R@1: {i2t['R@1']:.3f} R@5: {i2t['R@5']:.3f} | "
        f"T→I R@1: {t2i['R@1']:.3f} R@5: {t2i['R@5']:.3f}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

model.save_pretrained("clasp-finetuned")
processor.save_pretrained("clasp-finetuned")
print("Saved to clasp-finetuned/", flush=True)