"""
preprocess_01_regexfilter.py — STAGE 1: fast, deterministic cleaning of the TRAIN split.
The official TEST split is NEVER touched.

All thresholds are CLI parameters (nothing hardcoded) — runs in ~15s, tune freely.

Steps (train only):
  0. Drop rows whose answer has fewer than --min_a_words words (default 7),
     or whose question has fewer than --min_q_words words (default 1 = only empties).
  1. Normalize Soru/Cevap (Unicode NFC + strip). Dedup keys additionally lowercase and
     drop punctuation/whitespace.
  2. Exact-pair dedup: keep only ONE row per normalized (soru, cevap) pair.
  3. Same-question cap: keep each normalized soru at most --max_q_freq times (default 2).
  4. Same-answer  cap: keep each normalized cevap at most --max_a_freq times (default 5).

Output:
  data/train_clean.jsonl  — cleaned {"soru","cevap"} pairs (input for stage 02)

Usage:
  python -m src.preprocess_01_regexfilter
  python -m src.preprocess_01_regexfilter --min_a_words 7 --max_q_freq 2 --max_a_freq 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

from datasets import load_dataset

DATASET_NAME = "Renicames/turkish-law-chatbot"


def _get(row: dict, keys: list[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return v
    return ""


def _norm_text(s: str) -> str:
    """Output normalization: Unicode NFC + strip (keeps case for readable training text)."""
    return unicodedata.normalize("NFC", str(s)).strip()


def _key(s: str) -> str:
    """Dedup key: NFC + lowercase + drop everything that is not a letter/digit."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"[^\w]+", "", s, flags=re.UNICODE)


def _wc(s: str) -> int:
    return len(s.split())


def build(min_q_words: int, min_a_words: int, max_q_freq: int, max_a_freq: int):
    ds = load_dataset(DATASET_NAME)["train"]
    total = len(ds)

    # 0 + 1: extract, normalize, drop short
    rows, dropped_short = [], 0
    for r in ds:
        q = _norm_text(_get(r, ["Soru", "soru", "instruction", "question"]))
        a = _norm_text(_get(r, ["Cevap", "cevap", "output", "answer"]))
        if _wc(q) < min_q_words or _wc(a) < min_a_words:
            dropped_short += 1
            continue
        rows.append((q, a))

    # 2: exact-pair dedup on normalized (soru, cevap)
    seen_pair, after_pair, dropped_pair = set(), [], 0
    for q, a in rows:
        pk = (_key(q), _key(a))
        if pk in seen_pair:
            dropped_pair += 1
            continue
        seen_pair.add(pk)
        after_pair.append((q, a))

    # 3: same-question cap
    qcount, after_q, dropped_q = Counter(), [], 0
    for q, a in after_pair:
        qk = _key(q)
        if qcount[qk] >= max_q_freq:
            dropped_q += 1
            continue
        qcount[qk] += 1
        after_q.append((q, a))

    # 4: same-answer cap
    acount, final, dropped_a = Counter(), [], 0
    for q, a in after_q:
        ak = _key(a)
        if acount[ak] >= max_a_freq:
            dropped_a += 1
            continue
        acount[ak] += 1
        final.append((q, a))

    stats = {
        "total": total,
        "dropped_short": dropped_short,
        "dropped_pair": dropped_pair,
        "dropped_qcap": dropped_q,
        "dropped_acap": dropped_a,
        "final": len(final),
    }
    return final, stats


def main():
    p = argparse.ArgumentParser(description="Stage 1: deterministic train cleaning (parametric).")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--min_q_words", type=int, default=1,
                   help="Drop questions with fewer than this many words (default 1 = empties).")
    p.add_argument("--min_a_words", type=int, default=7,
                   help="Drop answers with fewer than this many words (default 7).")
    p.add_argument("--max_q_freq", type=int, default=2,
                   help="Keep each normalized question at most this many times (default 2).")
    p.add_argument("--max_a_freq", type=int, default=5,
                   help="Keep each normalized answer at most this many times (default 5).")
    args = p.parse_args()

    final, st = build(args.min_q_words, args.min_a_words, args.max_q_freq, args.max_a_freq)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "train_clean.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for q, a in final:
            f.write(json.dumps({"soru": q, "cevap": a}, ensure_ascii=False) + "\n")

    print("\n=== STAGE 01 — regex/exact/cap (TEST'e dokunulmadi) ===")
    print(f"params: min_q_words={args.min_q_words} min_a_words={args.min_a_words} "
          f"max_q_freq={args.max_q_freq} max_a_freq={args.max_a_freq}")
    print(f"Ham train                         : {st['total']}")
    print(f"0. Kisa/bos atilan (<{args.min_a_words} kelime cevap) : {st['dropped_short']}")
    print(f"2. Exact-pair dedup atilan        : {st['dropped_pair']}")
    print(f"3. Same-question cap (>{args.max_q_freq}) atilan  : {st['dropped_qcap']}")
    print(f"4. Same-answer cap (>{args.max_a_freq}) atilan    : {st['dropped_acap']}")
    print(f"Temiz toplam                      : {st['final']}")
    print(f"\nKaydedildi: {path}")
    print("Sonraki: python -m src.preprocess_02_semanticembed --mode embed")


if __name__ == "__main__":
    main()
