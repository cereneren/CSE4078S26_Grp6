"""
preprocess_02_semanticembed.py — STAGE 2: embed the (Soru + Cevap) PAIR of every cleaned
row from stage 01, so stage 03 can cluster on semantic similarity.

Runs on data/train_clean.jsonl (stage-01 output) — NOT the raw dataset. The embedding is
of the combined "Soru: ... Cevap: ..." text, one vector per pair.

Usage:
  python -m src.preprocess_02_semanticembed --mode embed
  python -m src.preprocess_02_semanticembed --mode embed --model Qwen/Qwen3-Embedding-0.6B --batch_size 64

Output:
  data/emb/pair_emb.npy  + data/emb/items.jsonl

Requires: pip install -U sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

CLEAN_PATH = "data/train_clean.jsonl"
EMB_DIR = "data/emb"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


def load_clean() -> tuple[list[str], list[str]]:
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(
            f"{CLEAN_PATH} yok. Önce stage 01: python -m src.preprocess_01_regexfilter"
        )
    qs, ans = [], []
    with open(CLEAN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qs.append(str(d.get("soru", "")).strip())
            ans.append(str(d.get("cevap", "")).strip())
    return qs, ans


def do_embed(args) -> None:
    from sentence_transformers import SentenceTransformer

    qs, ans = load_clean()
    texts = [f"Soru: {q}\nCevap: {a}" for q, a in zip(qs, ans)]
    print(f"{len(texts)} (soru+cevap) çifti. Model: {args.model}  (bf16)")
    # Without torch_dtype, sentence-transformers loads fp32 -> a 4B model is ~16GB and
    # won't fit a 12GB GPU (offloads, crawls). bf16 halves it (~8GB) -> fits.
    model = SentenceTransformer(
        args.model, trust_remote_code=True, model_kwargs={"torch_dtype": "bfloat16"}
    )
    emb = model.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )

    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(os.path.join(EMB_DIR, "pair_emb.npy"), emb.astype(np.float32))
    with open(os.path.join(EMB_DIR, "items.jsonl"), "w", encoding="utf-8") as f:
        for q, a in zip(qs, ans):
            f.write(json.dumps({"soru": q, "cevap": a}, ensure_ascii=False) + "\n")

    print(f"\nKaydedildi: {EMB_DIR}/  (pair_emb.npy, items.jsonl)")
    print("Sonraki: python -m src.preprocess_03_clustercap")


def main():
    p = argparse.ArgumentParser(description="Stage 2: embed (Soru+Cevap) pairs.")
    p.add_argument("--mode", choices=["embed"], default="embed")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    do_embed(args)


if __name__ == "__main__":
    main()
