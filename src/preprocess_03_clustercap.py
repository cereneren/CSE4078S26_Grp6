"""
preprocess_03_clustercap.py — STAGE 3: semantic clustering + polarity-aware cap, then the
final train/val split. Runs on the stage-02 PAIR embeddings (data/emb/), nothing raw.

Steps:
  5. Cluster: union-find over (Soru+Cevap) pair cosine similarity >= --threshold (0.97).
  6. Polarity: label each row POS / NEG / UNK from the ANSWER text (keyword rule, no LLM).
  7. Semantic cap: within each (cluster, polarity) group keep at most --cap rows (default 6),
     drop the rest. Splitting by polarity means opposite Evet/Hayır answers are capped
     SEPARATELY (never collapsed into one).
  8. train/val split (from train only).
  9. Write data/train_sft.jsonl + data/val_sft.jsonl in the Qwen chat format.

Review (default — nothing written except the review file):
  python -m src.preprocess_03_clustercap                 # cap=6, threshold=0.97
  python -m src.preprocess_03_clustercap --cap 6 --threshold 0.97

Apply (writes the final training set):
  python -m src.preprocess_03_clustercap --apply

Nothing is hardcoded: --threshold and --cap are parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

EMB_DIR = "data/emb"
REVIEW_PATH = "data/cluster_review.txt"
SYSTEM_PROMPT = (
    "Sen bir Türk hukuk asistanısın. "
    "Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
)

_POS_MARKERS = (
    "uygundur", "yapılabilir", "edilebilir", "alınabilir", "açılabilir",
    "mümkündür", "geçerlidir", "hakkı vardır", "hakkına sahiptir",
    "serbesttir", "caizdir",
)
_NEG_MARKERS = (
    "aykırıdır", "yapılamaz", "edilemez", "alınamaz", "açılamaz",
    "mümkün değil", "geçersiz", "yasaktır", "reddedil", "hakkı yoktur",
    "sahip değil", "caiz değil",
)


def polarity(answer: str) -> str:
    """POS / NEG / UNK from the answer text (pure keyword rule, no LLM)."""
    t = answer.lower().strip()
    if t.startswith("evet"):
        return "POS"
    if t.startswith("hayır"):
        return "NEG"
    pos = any(m in t for m in _POS_MARKERS)
    neg = any(m in t for m in _NEG_MARKERS)
    if pos and not neg:
        return "POS"
    if neg and not pos:
        return "NEG"
    return "UNK"


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)


def load_pairs():
    emb = np.load(os.path.join(EMB_DIR, "pair_emb.npy"))
    items = []
    with open(os.path.join(EMB_DIR, "items.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append({"soru": "?", "cevap": "?"})
    n = min(len(items), emb.shape[0])
    emb, items = emb[:n], items[:n]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, items


def cluster(emb, thr: float, chunk: int = 1000):
    n = len(emb)
    uf = UnionFind(n)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sims = emb[start:end] @ emb.T
        for li in range(end - start):
            i = start + li
            for j in (np.where(sims[li][i + 1:] >= thr)[0] + (i + 1)):
                uf.union(i, int(j))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    return list(comp.values())


def survivors_and_review(clusters, items, pols, cap: int):
    """Keep <=cap per (cluster, polarity) group. Returns (kept_indices, dropped, review_rows)."""
    keep, dropped = [], 0
    review = []  # clusters that actually drop something
    for c in clusters:
        groups = defaultdict(list)
        for i in c:
            groups[pols[i]].append(i)
        cluster_drops = any(len(idxs) > cap for idxs in groups.values())
        for pol, idxs in groups.items():
            keep.extend(idxs[:cap])
            dropped += max(0, len(idxs) - cap)
        if cluster_drops:
            review.append(c)
    return sorted(keep), dropped, review


def write_review(path, review_clusters, items, pols, cap):
    review_clusters = sorted(review_clusters, key=lambda c: -len(c))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(review_clusters)} kume (cap={cap} asilan polarite-grubu var)\n")
        f.write("Her (kume + polarite) grubunda ilk {0} TUT, fazlasi AT.\n\n".format(cap))
        for ci, c in enumerate(review_clusters, 1):
            groups = defaultdict(list)
            for i in c:
                groups[pols[i]].append(i)
            pc = Counter(pols[i] for i in c)
            f.write(f"===== KUME {ci} | {len(c)} uye | POS:{pc['POS']} NEG:{pc['NEG']} UNK:{pc['UNK']} =====\n")
            for pol, idxs in groups.items():
                for k, idx in enumerate(idxs):
                    mark = "TUT" if k < cap else "AT "
                    f.write(f"[{mark}|{pol}] S: {items[idx]['soru']}\n")
                    f.write(f"           C: {items[idx]['cevap']}\n")
            f.write("\n")


def to_record(q: str, a: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
        "soru": q, "cevap": a,
    }


def write_final(keep_idx, items, out_dir, val_ratio, seed):
    pairs = [(items[i]["soru"], items[i]["cevap"]) for i in keep_idx]
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val, train = shuffled[:n_val], shuffled[n_val:]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_sft.jsonl"), "w", encoding="utf-8") as f:
        for q, a in train:
            f.write(json.dumps(to_record(q, a), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "val_sft.jsonl"), "w", encoding="utf-8") as f:
        for q, a in val:
            f.write(json.dumps(to_record(q, a), ensure_ascii=False) + "\n")
    return len(train), len(val)


def main():
    p = argparse.ArgumentParser(description="Stage 3: semantic cluster + polarity cap (review/apply).")
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--cap", type=int, default=6)
    p.add_argument("--apply", action="store_true",
                   help="Write the final train_sft/val_sft (default: review only).")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    emb, items = load_pairs()
    pols = [polarity(it["cevap"]) for it in items]
    print(f"{len(items)} çift. eşik={args.threshold} cap={args.cap} (cluster+polarity başına)")

    clusters = cluster(emb, args.threshold)
    keep, dropped, review = survivors_and_review(clusters, items, pols, args.cap)
    write_review(REVIEW_PATH, review, items, pols, args.cap)

    print(f"\nKüme sayısı            : {len(clusters)}")
    print(f"Cap aşan küme          : {len(review)}")
    print(f"Atılacak satır         : {dropped}")
    print(f"Kalan (uygulanırsa)    : {len(keep)}")
    print(f"İnceleme dosyası       : {REVIEW_PATH}  ([TUT|POS]/[AT|NEG] etiketli)")

    if args.apply:
        n_tr, n_val = write_final(keep, items, args.out_dir, args.val_ratio, args.seed)
        print(f"\n--apply: yazildi -> train {n_tr} | val {n_val}")
        print(f"  {args.out_dir}/train_sft.jsonl")
        print(f"  {args.out_dir}/val_sft.jsonl")
        print("Sonraki: python -m src.train_v2 --epochs 1")
    else:
        print("\n(review modu — hiçbir şey silinmedi. Uygulamak için --apply ekle.)")


if __name__ == "__main__":
    main()
