"""
error_analysis.py — reproducible before/after error-pattern analysis (no LLM, fixed rules).

Reads the base and fine-tuned full-test inference JSONL files and reports, for each, the
rate of objective error categories, plus writes a few example pairs for the report.

Categories (all deterministic / scriptable):
  - avg answer length (words)                 -> verbosity
  - article-number accuracy                   -> does the answer cite the same "Madde N"
                                                 as the question/reference (when both do)
  - polarity accuracy (POS/NEG, keyword)      -> correct Evet/Hayır conclusion vs reference
  - repetition / degeneration rate            -> answer has a 3-gram repeated >= 4 times

Usage:
  python -m src.error_analysis \
      --base outputs/Qwen_Qwen3-4B-Instruct-2507_a_base_full_inference.jsonl \
      --finetuned outputs/Qwen_Qwen3-4B-Instruct-2507_finetuned_full_inference.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

POS = ("uygundur", "yapılabilir", "edilebilir", "alınabilir", "mümkündür",
       "geçerlidir", "hakkı vardır", "hakkına sahiptir", "serbesttir", "caizdir")
NEG = ("aykırıdır", "yapılamaz", "edilemez", "alınamaz", "mümkün değil",
       "geçersiz", "yasaktır", "reddedil", "hakkı yoktur", "sahip değil")


def polarity(text: str) -> str:
    t = text.lower().strip()
    if t.startswith("evet"):
        return "POS"
    if t.startswith("hayır"):
        return "NEG"
    p = any(m in t for m in POS)
    n = any(m in t for m in NEG)
    return "POS" if (p and not n) else "NEG" if (n and not p) else "UNK"


def articles(text: str) -> set[str]:
    return set(re.findall(r"madde\s*(\d+)", text.lower()))


def max_repeat(text: str, n: int = 3) -> int:
    w = text.lower().split()
    if len(w) < n:
        return 0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return max(Counter(grams).values()) if grams else 0


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    words, art_match, art_total, pol_match, pol_total, rep = [], 0, 0, 0, 0, 0
    for r in rows:
        q = r.get("question", "")
        ref = r.get("reference_answer", "")
        gen = r.get("generated_answer", "")
        words.append(len(gen.split()))
        ref_art = articles(q) or articles(ref)
        gen_art = articles(gen)
        if ref_art and gen_art:
            art_total += 1
            art_match += 1 if (ref_art & gen_art) else 0
        rp, gp = polarity(ref), polarity(gen)
        if rp != "UNK" and gp != "UNK":
            pol_total += 1
            pol_match += 1 if rp == gp else 0
        if max_repeat(gen) >= 4:
            rep += 1
    return {
        "n": n,
        "avg_words": statistics.mean(words),
        "article_acc": art_match / art_total if art_total else 0.0,
        "article_total": art_total,
        "polarity_acc": pol_match / pol_total if pol_total else 0.0,
        "polarity_total": pol_total,
        "repetition_rate": rep / n if n else 0.0,
    }


def write_examples(base, ft, path, k=4):
    lines = []

    def dump(i, tag):
        lines.append(f"### {tag}")
        lines.append(f"SORU: {base[i]['question']}")
        lines.append(f"REF : {base[i]['reference_answer']}")
        lines.append(f"BASE: {base[i]['generated_answer']}")
        lines.append(f"FT  : {ft[i]['generated_answer']}")
        lines.append("")

    used = 0
    for i in range(min(len(base), len(ft))):
        if max_repeat(base[i]['generated_answer']) >= 5 and max_repeat(ft[i]['generated_answer']) < 4:
            dump(i, "DEGENERASYON duzeltildi"); used += 1
            break
    for i in range(min(len(base), len(ft))):
        rp = polarity(base[i]['reference_answer'])
        if rp != "UNK" and polarity(base[i]['generated_answer']) != rp and polarity(ft[i]['generated_answer']) == rp:
            dump(i, "POLARITE duzeltildi"); used += 1
            break
    for i in range(min(len(base), len(ft))):
        if len(base[i]['generated_answer'].split()) > 80 and len(ft[i]['generated_answer'].split()) < 30:
            dump(i, "VERBOSITY (base uzun -> ft kisa)"); used += 1
            break

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return used


def main():
    p = argparse.ArgumentParser(description="Before/after error-pattern analysis.")
    p.add_argument("--base", required=True, help="base model full-test inference JSONL")
    p.add_argument("--finetuned", required=True, help="fine-tuned full-test inference JSONL")
    p.add_argument("--examples_out", default="outputs/error_analysis_examples.txt")
    args = p.parse_args()

    base = load(args.base)
    ft = load(args.finetuned)
    b, f = analyze(base), analyze(ft)

    print(f"\n{'metrik':<28}{'BASE':>10}{'FINE-TUNED':>14}")
    print(f"{'ort. cevap kelime':<28}{b['avg_words']:>10.1f}{f['avg_words']:>14.1f}")
    print(f"{'madde-no isabet %':<28}{100*b['article_acc']:>10.1f}{100*f['article_acc']:>14.1f}"
          f"   (n={b['article_total']}/{f['article_total']})")
    print(f"{'polarite isabet %':<28}{100*b['polarity_acc']:>10.1f}{100*f['polarity_acc']:>14.1f}"
          f"   (n={b['polarity_total']}/{f['polarity_total']})")
    print(f"{'tekrar/degenerasyon %':<28}{100*b['repetition_rate']:>10.1f}{100*f['repetition_rate']:>14.1f}")

    n = write_examples(base, ft, args.examples_out)
    print(f"\n{n} örnek çift yazıldı: {args.examples_out}")


if __name__ == "__main__":
    main()
