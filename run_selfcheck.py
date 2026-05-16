"""
SelfCheckGPT Evaluation — Phase 2 only
Loads fine-tuned GPT-2 XL from checkpoint and evaluates on Story Cloze val set.
"""

import csv, random
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

OUTPUT_DIR   = "finetuned_gpt2xl_rocstories"
MODEL_NAME   = "gpt2-xl"
VAL_FILES    = [
    "dataset/val/cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
    "dataset/val/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
]
VAL_SAMPLES      = 300
NUM_SC_SAMP      = 5
SEED             = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_val(files, n):
    rows = []
    for p in files:
        with open(p, newline='', encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    random.shuffle(rows)
    return rows[:n]


def generate(model, tokenizer, prompt, n_samples, max_new=80):
    enc  = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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


def main():
    print(f"Loading tokenizer from {MODEL_NAME} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading fine-tuned model from {OUTPUT_DIR} ...")
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    print("\nLoading SelfCheckGPT NLI ...")
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
    nli = SelfCheckNLI(device=DEVICE)

    print(f"\nLoading {VAL_SAMPLES} validation examples ...")
    val_rows = load_val(VAL_FILES, VAL_SAMPLES)

    print(f"\n{'='*60}")
    print(f"SelfCheckGPT (NLI) Evaluation — {VAL_SAMPLES} examples")
    print(f"{'='*60}")

    correct, results = 0, []
    for idx, row in enumerate(tqdm(val_rows, desc="Evaluating")):
        ctx = " ".join(row[f'InputSentence{i}'] for i in range(1, 5))
        q1  = row['RandomFifthSentenceQuiz1']
        q2  = row['RandomFifthSentenceQuiz2']
        ans = int(row['AnswerRightEnding'])

        main_text, sampled = generate(model, tokenizer,
                                      ctx + " <|endoftext|> ", NUM_SC_SAMP)
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

    acc = correct / len(val_rows)

    # 전체 결과 CSV 저장
    import csv as csv_mod
    csv_path = "eval_results_raw.csv"
    fieldnames = ["idx", "context", "quiz1", "quiz2", "correct_answer",
                  "predicted", "score_q1", "score_q2", "main_generated", "correct"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({"idx": i + 1, **r})
    print(f"\n[저장] 전체 결과: {csv_path}  ({len(results)}행)")

    print(f"\n{'='*60}\nSample Predictions\n{'='*60}")
    for r in results[:5]:
        print(f"\nContext:   {r['context'][:110]}...")
        print(f"Quiz 1:    {r['quiz1']}")
        print(f"Quiz 2:    {r['quiz2']}")
        print(f"Generated: {r['main_generated'][:100]}")
        print(f"NLI score — Q1: {r['score_q1']:.4f}  Q2: {r['score_q2']:.4f}  "
              f"→ predicted: {r['predicted']}  correct: {r['correct_answer']}  "
              f"{'✓' if r['correct'] else '✗'}")

    print(f"\n{'='*60}")
    print(f"Story Cloze Accuracy (SelfCheckGPT NLI): {acc:.4f}  "
          f"({correct}/{len(val_rows)})")
    print(f"Random baseline:  0.5000")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
