"""
ROCStories Fine-tuning (3-GPU DDP) + SelfCheckGPT Evaluation

Phase 1: Fine-tune GPT-2 XL with accelerate DDP on 3 GPUs
         - BF16 mixed precision (gradients in BF16)
         - 8-bit Adam for optimizer states
         - Gradient checkpointing → activation memory ~1-2GB per GPU
         - Per-GPU batch=16, 3 GPUs → effective batch = 16×3×4 = 192
         - Expected VRAM per GPU: ~13GB / 24GB

Phase 2: SelfCheckGPT (NLI) evaluation
         - Generation on GPU 0, NLI scoring on GPU 1

Run with:
    accelerate launch --num_processes=3 --mixed_precision=bf16 train_ddp.py
"""

import os, csv, random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TRAIN_FILES = [
    "dataset/train/ROCStories__spring2016 - ROCStories_spring2016.csv",
    "dataset/train/ROCStories_winter2017 - ROCStories_winter2017.csv",
]
VAL_FILES = [
    "dataset/val/cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
    "dataset/val/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
]

MODEL_NAME    = "gpt2-xl"
OUTPUT_DIR    = "finetuned_gpt2xl_rocstories"
GEN_GPU       = "cuda:0"
NLI_GPU       = "cuda:1"
TRAIN_SAMPLES = 98000
VAL_SAMPLES   = 300
NUM_EPOCHS    = 1
BATCH_SIZE    = 16          # per GPU; 3 GPUs → 48 per step
GRAD_ACCUM    = 4           # effective batch = 48×4 = 192
MAX_LEN       = 512
LR            = 3e-5
NUM_SC_SAMP   = 5
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def load_rocstories(files):
    stories = []
    for path in files:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                ctx = " ".join(row[f'sentence{i}'] for i in range(1, 5))
                stories.append((ctx, row['sentence5']))
    return stories


class ROCStoriesDataset(Dataset):
    def __init__(self, stories, tokenizer, max_len):
        self.data = []
        sep = " <|endoftext|> "
        for ctx, end in stories:
            text = ctx + sep + end + " <|endoftext|>"
            enc = tokenizer(text, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            lbls = ids.clone(); lbls[mask == 0] = -100
            self.data.append((ids, mask, lbls))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ─────────────────────────────────────────────
# Phase 1 — 3-GPU DDP Fine-tuning
# ─────────────────────────────────────────────

def train(accelerator, model, tokenizer, stories):
    num_gpus = accelerator.num_processes
    eff_bs = BATCH_SIZE * GRAD_ACCUM * num_gpus
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Phase 1: Fine-tuning {MODEL_NAME} — DDP on {num_gpus} GPUs")
        print(f"  Per-GPU batch: {BATCH_SIZE}  GradAccum: {GRAD_ACCUM}")
        print(f"  Effective batch: {eff_bs}")
        print(f"  MaxLen: {MAX_LEN}  dtype: bf16")
        print(f"  Optimizer: 8-bit AdamW")
        print(f"{'='*60}")

    dataset = ROCStoriesDataset(stories, tokenizer, MAX_LEN)
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=SEED,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)

    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                    disable=not accelerator.is_main_process)

        for step, (ids, mask, labels) in enumerate(pbar):
            with accelerator.accumulate(model):
                out  = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += out.loss.item()

            if step % 100 == 0 and accelerator.is_main_process:
                vram = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix(loss=f"{out.loss.item():.4f}", vram=f"{vram:.1f}G")

        avg_loss = total_loss / len(loader)
        if accelerator.is_main_process:
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    if accelerator.is_main_process:
        print("Saving model ...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR, legacy_format=False)
        print(f"Saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
# Phase 2 — SelfCheckGPT Evaluation (main process only)
# ─────────────────────────────────────────────

def load_val(files, n):
    rows = []
    for p in files:
        with open(p, newline='', encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    random.shuffle(rows)
    return rows[:n]


def generate(model, tokenizer, prompt, n_samples, max_new=80, device="cuda:0"):
    enc  = tokenizer(prompt, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[1]
    eos  = tokenizer.eos_token_id
    with torch.no_grad():
        g = model.generate(enc["input_ids"], max_new_tokens=max_new,
                           do_sample=False, pad_token_id=eos)
    main = tokenizer.decode(g[0][plen:], skip_special_tokens=True).strip()
    samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            s = model.generate(enc["input_ids"], max_new_tokens=max_new,
                               do_sample=True, temperature=1.0, top_p=0.95,
                               pad_token_id=eos)
        samples.append(tokenizer.decode(s[0][plen:], skip_special_tokens=True).strip())
    return main, samples


def evaluate_selfcheck(gen_model, tokenizer, val_rows):
    print(f"\n{'='*60}")
    print(f"Phase 2: SelfCheckGPT (NLI) — {len(val_rows)} examples")
    print(f"  Generation : {GEN_GPU}")
    print(f"  NLI scoring: {NLI_GPU}")
    print(f"{'='*60}")

    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
    nli = SelfCheckNLI(device=torch.device(NLI_GPU))

    gen_model.eval()
    correct, results = 0, []

    for idx, row in enumerate(tqdm(val_rows, desc="SelfCheck eval")):
        ctx = " ".join(row[f'InputSentence{i}'] for i in range(1, 5))
        q1  = row['RandomFifthSentenceQuiz1']
        q2  = row['RandomFifthSentenceQuiz2']
        ans = int(row['AnswerRightEnding'])

        main_text, sampled = generate(gen_model, tokenizer,
                                      ctx + " <|endoftext|> ", NUM_SC_SAMP,
                                      device=GEN_GPU)
        s1 = nli.predict([q1], sampled)[0]
        s2 = nli.predict([q2], sampled)[0]

        pred = 1 if s1 <= s2 else 2
        ok   = pred == ans
        if ok: correct += 1
        results.append(dict(
            context=ctx, quiz1=q1, quiz2=q2,
            correct_answer=ans, predicted=pred,
            score_q1=float(s1), score_q2=float(s2),
            main_generated=main_text, correct=ok,
        ))
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(val_rows)}] accuracy: {correct/(idx+1):.4f}")

    return correct / len(val_rows), results


def print_examples(results, n=5):
    print(f"\n{'='*60}\nSample Predictions\n{'='*60}")
    for r in results[:n]:
        print(f"\nContext:   {r['context'][:110]}...")
        print(f"Quiz 1:    {r['quiz1']}")
        print(f"Quiz 2:    {r['quiz2']}")
        print(f"Generated: {r['main_generated'][:100]}")
        tick = '✓' if r['correct'] else '✗'
        print(f"NLI  Q1:{r['score_q1']:.3f}  Q2:{r['score_q2']:.3f}  "
              f"→ pred:{r['predicted']}  ans:{r['correct_answer']}  {tick}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )

    if accelerator.is_main_process:
        for i in range(torch.cuda.device_count()):
            gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gb:.1f}GB)")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model_ok = (os.path.exists(os.path.join(OUTPUT_DIR, "model.safetensors")) or
                os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin")))

    if not model_ok:
        if accelerator.is_main_process:
            print(f"\nLoading {MODEL_NAME} in BF16 ...")
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        if accelerator.is_main_process:
            param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params "
                  f"({param_gb:.1f}GB in BF16)")

        stories = load_rocstories(TRAIN_FILES)
        random.shuffle(stories)
        stories = stories[:TRAIN_SAMPLES]

        if accelerator.is_main_process:
            print(f"  Training examples: {len(stories)}")

        train(accelerator, model, tokenizer, stories)

        del model
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    else:
        if accelerator.is_main_process:
            print(f"Found fine-tuned model at {OUTPUT_DIR}, skipping training.")

    # Phase 2 — main process only
    if accelerator.is_main_process:
        print(f"\nLoading fine-tuned model for inference on {GEN_GPU} ...")
        gen_model = GPT2LMHeadModel.from_pretrained(
            OUTPUT_DIR, torch_dtype=torch.bfloat16
        ).to(GEN_GPU)
        gen_model.eval()

        val_rows = load_val(VAL_FILES, VAL_SAMPLES)
        acc, results = evaluate_selfcheck(gen_model, tokenizer, val_rows)
        print_examples(results)

        print(f"\n{'='*60}")
        print(f"Story Cloze Accuracy (SelfCheckGPT NLI): {acc:.4f}  "
              f"({int(acc*len(results))}/{len(results)})")
        print(f"Random baseline:  0.5000")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
