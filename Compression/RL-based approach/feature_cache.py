from __future__ import annotations

import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import math

BASES = ["A", "C", "G", "T"]
DINUCS = [a + b for a in BASES for b in BASES]

def shannon_entropy_from_counts(counts: dict[str, int], total: int) -> float:
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def kmer_counts(seq: str, k: int) -> Counter[str]:
    if len(seq) < k:
        return Counter()
    return Counter(seq[i:i+k] for i in range(len(seq) - k + 1))


def repeat_ratio(seq: str, k: int = 4) -> float:
    kmers = kmer_counts(seq, k)
    total = sum(kmers.values())
    if total == 0:
        return 0.0
    repeated = sum(v for v in kmers.values() if v > 1)
    return repeated / total


def longest_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def homopolymer_frequency(seq: str, min_run: int = 3) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    covered = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and seq[j] == seq[i]:
            j += 1
        if j - i >= min_run:
            covered += (j - i)
        i = j
    return covered / n


def lz_complexity_proxy(seq: str) -> float:
    if not seq:
        return 0.0
    seen: set[str] = set()
    i = 0
    phrases = 0
    while i < len(seq):
        j = i + 1
        while j <= len(seq) and seq[i:j] in seen:
            j += 1
        seen.add(seq[i:j])
        phrases += 1
        i = j
    return phrases / max(1, len(seq))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def extract_window_features(seq: str) -> list[float]:
    n = len(seq)
    if n == 0:
        return [0.0] * 28

    mono = Counter(seq)
    di = kmer_counts(seq, 2)

    base_freqs = [mono.get(b, 0) / n for b in BASES]
    gc = (mono.get("G", 0) + mono.get("C", 0)) / n
    entropy = shannon_entropy_from_counts(mono, n) / 2.0   # normalize

    di_total = max(1, n - 1)
    di_freqs = [di.get(d, 0) / di_total for d in DINUCS]

    uniq3_ratio = len(kmer_counts(seq, 3)) / max(1, n - 2)
    uniq4_ratio = len(kmer_counts(seq, 4)) / max(1, n - 3)
    rep_ratio = repeat_ratio(seq, k=4)
    longest_run = longest_homopolymer_run(seq) / n
    lz_proxy = lz_complexity_proxy(seq)
    homo_freq = homopolymer_frequency(seq)

    return (base_freqs + [gc, entropy] + di_freqs + [uniq3_ratio, uniq4_ratio, rep_ratio, longest_run, lz_proxy, homo_freq])


def build_window_feature_cache(current_nucleotides : str | Path, window_size: int = 1000) -> list[list[float]]:
    windows = [current_nucleotides[i:i + window_size] for i in range(0, len(current_nucleotides), window_size)]
    window_features = [extract_window_features(w) for w in windows]

    # print(f"Num windows: {len(window_features)}")
    return window_features


def load_window_feature_cache(cache_path: str | Path) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build feature cache for RL segmentation")
    ap.add_argument("input", type=str)
    # ap.add_argument("output", type=str)
    ap.add_argument("--window-size", type=int, default=1000)
    args = ap.parse_args()

    window_features = build_window_feature_cache(   # matrice me windows, dhe features per cdo window
        current_nucleotides=args.input,
        # cache_path=args.output_cache,
        window_size=args.window_size
    )