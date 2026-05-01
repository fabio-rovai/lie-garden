"""
Combined detector benchmark: does holonomy add value on top of the probe?

Builds multi-message scenarios from each dataset, then compares three
classifiers on the SAME train/test split with the SAME pipeline:

  1. probe-only          — logistic on [max probe score over conversation]
  2. holonomy-only       — logistic on holonomy features (algebra log + invariants)
  3. combined            — logistic on [probe_max, probe_mean, holonomy_features]

If (3) > (1) by a statistically significant margin (paired bootstrap),
holonomy provides incremental F1 over the probe — directly addressing
the AISI maintainer's "probe alone matches" critique.

All three classifiers see the SAME scenarios, SAME train/test split,
SAME embedding pipeline. The only difference is the feature set.

Usage::

    python3 bench_combined.py            # all four datasets
    python3 bench_combined.py --dataset tensortrust  # single dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import expm, logm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Embedding (Model2Vec or SHA-fallback)
# ---------------------------------------------------------------------------

_MODEL = None
_EMB_DIM = 512


def encode(texts: list[str]) -> np.ndarray:
    global _MODEL, _EMB_DIM
    try:
        from model2vec import StaticModel
        if _MODEL is None:
            _MODEL = StaticModel.from_pretrained(
                "minishlab/potion-base-32M",
                # No revision pin in the original; we leave consistent with
                # holonomy_honest.py to keep results comparable.
            )
        embs = np.array(_MODEL.encode(texts))
        _EMB_DIM = embs.shape[1]
        return embs
    except ImportError:
        print("WARN: model2vec unavailable, using SHA-fallback (results not comparable)", file=sys.stderr)
        vecs = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array([int.from_bytes(h[i:i + 2], "little") / 65535.0
                            for i in range(0, min(len(h), 2 * _EMB_DIM), 2)])
            if len(vec) < _EMB_DIM:
                vec = np.resize(vec, _EMB_DIM)
            vecs.append(vec)
        return np.array(vecs)


# ---------------------------------------------------------------------------
# Lie-algebra mapping and holonomy features
# ---------------------------------------------------------------------------

def vec_to_skew(v: np.ndarray, n: int) -> np.ndarray:
    """Project a feature vector onto the antisymmetric n×n algebra so(n)."""
    A = np.zeros((n, n))
    k = 0
    target = n * (n - 1) // 2
    if len(v) < target:
        v = np.resize(v, target)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = v[k]
            A[j, i] = -v[k]
            k += 1
    return A


_LOGM_FALLBACKS = {"count": 0, "total": 0}


def safe_logm(R: np.ndarray) -> np.ndarray:
    _LOGM_FALLBACKS["total"] += 1
    try:
        L = logm(R)
        return np.real((L - L.T) / 2.0)
    except Exception:
        _LOGM_FALLBACKS["count"] += 1
        return np.zeros_like(R)


def skew_to_vec(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.array([A[i, j] for i in range(n) for j in range(i + 1, n)])


def holonomy_features(vecs: np.ndarray, n: int = 10, scale: float = 0.1) -> np.ndarray:
    """Apply step-wise group composition; return log-matrix of accumulated state."""
    state = np.eye(n)
    for v in vecs:
        skew = vec_to_skew(v, n) * scale
        try:
            R = expm(skew)
        except Exception:
            R = np.eye(n)
        state = state @ R
    return skew_to_vec(safe_logm(state))


def raw_embedding_features(vecs: np.ndarray, n_features: int) -> np.ndarray:
    """Mean of raw embeddings (truncated/projected to n_features)."""
    avg = vecs.mean(axis=0)
    if len(avg) >= n_features:
        return avg[:n_features]
    return np.resize(avg, n_features)


def probe_features(vecs: np.ndarray, probe_dir: np.ndarray) -> np.ndarray:
    """Probe statistics over the conversation:
       [max_score, mean_score, last_score, max_minus_mean].
    """
    scores = vecs @ probe_dir
    return np.array([
        float(scores.max()),
        float(scores.mean()),
        float(scores[-1]) if len(scores) > 0 else 0.0,
        float(scores.max() - scores.mean()),
    ])


# ---------------------------------------------------------------------------
# Scenario construction (multi-message conversations from single-message data)
# ---------------------------------------------------------------------------

def make_scenarios(
    b_vecs: np.ndarray,
    h_vecs: np.ndarray,
    n_per_type: int = 200,
    conv_len: int = 4,
    attack_pos: int | None = 2,
    seed: int = 42,
) -> list[dict]:
    """Build N benign and N attack scenarios.

    Benign:  conv_len benign messages drawn from b_vecs.
    Attack:  conv_len-1 benign messages + one attack at attack_pos
             (default position 2 of the conversation).
    """
    rng = np.random.RandomState(seed)
    nb, nh = len(b_vecs), len(h_vecs)
    scenarios = []

    if attack_pos is None:
        attack_pos = max(0, conv_len // 2)

    for _ in range(n_per_type):
        # Benign
        b_idx = rng.choice(nb, conv_len, replace=True)
        scenarios.append({"vecs": b_vecs[b_idx], "label": 0})

        # Attack
        b_idx = rng.choice(nb, conv_len - 1, replace=True)
        h_idx = rng.choice(nh)
        vecs = np.zeros((conv_len, b_vecs.shape[1]))
        bi = 0
        for pos in range(conv_len):
            if pos == attack_pos:
                vecs[pos] = h_vecs[h_idx]
            else:
                vecs[pos] = b_vecs[b_idx[bi]]
                bi += 1
        scenarios.append({"vecs": vecs, "label": 1})

    return scenarios


# ---------------------------------------------------------------------------
# Classifier evaluation with proper splits
# ---------------------------------------------------------------------------

@dataclass
class ClfResult:
    f1: float
    precision: float
    recall: float
    n_train: int
    n_test: int


def evaluate_classifier_disjoint(
    train_scenarios: list[dict],
    test_scenarios: list[dict],
    feature_fn: Callable[[dict], np.ndarray],
) -> tuple[ClfResult, np.ndarray, np.ndarray]:
    """Train on train_scenarios, evaluate on test_scenarios.

    The caller is responsible for ensuring the two scenario lists draw from
    DISJOINT vector pools — no single encoded text vector should be able to
    appear in any train scenario AND any test scenario. This rules out the
    weak leakage that would exist if scenarios were stratified-split from
    a shared pool of vectors with replacement.
    """
    train_feats = np.array([feature_fn(s) for s in train_scenarios])
    test_feats = np.array([feature_fn(s) for s in test_scenarios])
    train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=1e6, neginf=-1e6)
    test_feats = np.nan_to_num(test_feats, nan=0.0, posinf=1e6, neginf=-1e6)
    train_labels = np.array([s["label"] for s in train_scenarios])
    test_labels = np.array([s["label"] for s in test_scenarios])

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.fit_transform(train_feats), train_labels)
    preds = clf.predict(sc.transform(test_feats))
    return (
        ClfResult(
            f1=f1_score(test_labels, preds, zero_division=0),
            precision=precision_score(test_labels, preds, zero_division=0),
            recall=recall_score(test_labels, preds, zero_division=0),
            n_train=len(train_scenarios),
            n_test=len(test_scenarios),
        ),
        preds,
        test_labels,
    )


def paired_bootstrap_delta(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    truth: np.ndarray,
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """95% bootstrap CI on F1(b) - F1(a). Positive means b > a."""
    rng = np.random.RandomState(seed)
    n = len(truth)
    deltas = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        f1_a = f1_score(truth[idx], preds_a[idx], zero_division=0)
        f1_b = f1_score(truth[idx], preds_b[idx], zero_division=0)
        deltas.append(f1_b - f1_a)
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_neuralchemy() -> tuple[list[str], list[str]]:
    p = "/tmp/neuralchemy_prompts.jsonl"
    if not os.path.exists(p):
        return [], []
    b, h = [], []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            (b if d["label"] == 0 else h).append(d["text"])
    return b, h


def load_tensortrust() -> tuple[list[str], list[str]]:
    p = "/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    if not os.path.exists(p):
        return [], []
    b, h = [], []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            b.append(d["pre_prompt"])
            h.append(d["attack"])
    return b, h


def load_deepset() -> tuple[list[str], list[str]]:
    p = "/tmp/deepset_prompts.jsonl"
    if not os.path.exists(p):
        return [], []
    b, h = [], []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            (b if d["label"] == 0 else h).append(d["text"])
    return b, h


def load_injecagent() -> tuple[list[str], list[str]]:
    """InjecAgent: benign user_cases + attack cases."""
    base = "/tmp/InjecAgent/data"
    if not os.path.exists(base):
        return [], []
    b = []
    with open(f"{base}/user_cases.jsonl") as f:
        for line in f:
            d = json.loads(line)
            b.append(d.get("User Instruction", ""))
    h = []
    for fname in ("attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"):
        with open(f"{base}/{fname}") as f:
            for line in f:
                d = json.loads(line)
                h.append(d.get("Attacker Instruction", ""))
    # Augment with full test_cases_*.json (much larger)
    import json as _json
    for tcf in ("test_cases_dh_base.json", "test_cases_ds_base.json"):
        path = f"{base}/{tcf}"
        if os.path.exists(path):
            with open(path) as f:
                cases = _json.load(f)
            for c in cases:
                instr = c.get("Attacker Instruction", "") or c.get("Attacker Instruction Achievement", "")
                if instr:
                    h.append(instr)
    return b, h


DATASETS: dict[str, Callable[[], tuple[list[str], list[str]]]] = {
    "neuralchemy": load_neuralchemy,
    "tensortrust": load_tensortrust,
    "deepset": load_deepset,
    "injecagent": load_injecagent,
}


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_dataset(
    name: str,
    b_texts: list[str],
    h_texts: list[str],
    so_n: int = 10,
    conv_len: int = 4,
    attack_pos: int = 2,
    n_per_type: int = 250,
    test_size: float = 0.4,
    seed: int = 42,
) -> dict:
    if not b_texts or not h_texts:
        print(f"\n  [{name}] skipped: empty pool ({len(b_texts)} benign / {len(h_texts)} harmful)")
        return {"name": name, "skipped": True}

    n_b, n_h = len(b_texts), len(h_texts)
    print(f"\n  [{name}] {n_b} benign / {n_h} harmful texts")
    print(f"  [{name}] encoding...")
    t0 = time.time()
    b_vecs = encode(b_texts)
    h_vecs = encode(h_texts)
    print(f"  [{name}] encoded in {time.time() - t0:.1f}s, dim={b_vecs.shape[1]}")

    # Need enough texts in BOTH the probe-training split and the eval pool.
    # Refuse to run if either side would end up empty.
    MIN_PER_SPLIT = 30
    if n_b < 2 * MIN_PER_SPLIT or n_h < 2 * MIN_PER_SPLIT:
        print(f"  [{name}] skipped: pool too small "
              f"({n_b} benign / {n_h} harmful, need ≥{2 * MIN_PER_SPLIT} each)")
        return {"name": name, "skipped": True,
                "reason": f"pool too small (n_b={n_b}, n_h={n_h})"}

    # Train probe direction on a held-out portion of the raw text pool —
    # never seen by the scenario generator that produces eval data.
    rng = np.random.RandomState(0)
    b_perm = rng.permutation(n_b)
    h_perm = rng.permutation(n_h)
    n_b_probe = max(MIN_PER_SPLIT, min(n_b // 4, n_b - MIN_PER_SPLIT))
    n_h_probe = max(MIN_PER_SPLIT, min(n_h // 4, n_h - MIN_PER_SPLIT))
    b_probe = b_vecs[b_perm[:n_b_probe]]
    h_probe = h_vecs[h_perm[:n_h_probe]]
    b_pool = b_vecs[b_perm[n_b_probe:]]
    h_pool = h_vecs[h_perm[n_h_probe:]]

    diff = h_probe.mean(0) - b_probe.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    print(f"  [{name}] probe trained on {n_b_probe}+{n_h_probe} held-out texts")
    print(f"  [{name}] eval pool: {len(b_pool)} benign + {len(h_pool)} harmful")

    # Strict split: divide the eval pool itself into disjoint train/test halves
    # BEFORE building scenarios. With this split, no single encoded vector can
    # appear in any train-scenario AND any test-scenario. Scenarios sample
    # with-replacement WITHIN their own half, but the two halves never mix.
    rng_split = np.random.RandomState(0)
    n_b_pool = len(b_pool)
    n_h_pool = len(h_pool)
    b_split = int((1.0 - test_size) * n_b_pool)
    h_split = int((1.0 - test_size) * n_h_pool)
    b_pool_perm = rng_split.permutation(n_b_pool)
    h_pool_perm = rng_split.permutation(n_h_pool)
    b_train_pool = b_pool[b_pool_perm[:b_split]]
    b_test_pool = b_pool[b_pool_perm[b_split:]]
    h_train_pool = h_pool[h_pool_perm[:h_split]]
    h_test_pool = h_pool[h_pool_perm[h_split:]]
    print(f"  [{name}] strict split: train pool {len(b_train_pool)}+{len(h_train_pool)}, "
          f"test pool {len(b_test_pool)}+{len(h_test_pool)} (no vector shared)")

    n_train = int(n_per_type * (1.0 - test_size))
    n_test = n_per_type - n_train
    train_scenarios = make_scenarios(
        b_train_pool, h_train_pool,
        n_per_type=n_train,
        conv_len=conv_len,
        attack_pos=attack_pos,
        seed=seed,
    )
    test_scenarios = make_scenarios(
        b_test_pool, h_test_pool,
        n_per_type=n_test,
        conv_len=conv_len,
        attack_pos=attack_pos,
        seed=seed + 57,
    )
    print(f"  [{name}] built {len(train_scenarios)} train + {len(test_scenarios)} test scenarios "
          f"(length {conv_len}, attack at pos {attack_pos})")

    # Feature functions
    n_holo = so_n * (so_n - 1) // 2

    def f_probe(s):
        return probe_features(s["vecs"], probe_dir)

    def f_holo(s):
        return holonomy_features(s["vecs"], n=so_n, scale=0.1)

    def f_combined(s):
        return np.concatenate([
            probe_features(s["vecs"], probe_dir),
            holonomy_features(s["vecs"], n=so_n, scale=0.1),
        ])

    print(f"  [{name}] training probe-only / holonomy-only / combined classifiers...")
    res_p, preds_p, truth_p = evaluate_classifier_disjoint(train_scenarios, test_scenarios, f_probe)
    res_h, preds_h, truth_h = evaluate_classifier_disjoint(train_scenarios, test_scenarios, f_holo)
    res_c, preds_c, truth_c = evaluate_classifier_disjoint(train_scenarios, test_scenarios, f_combined)

    # Sanity check: same test split → identical truth arrays
    assert np.array_equal(truth_p, truth_h)
    assert np.array_equal(truth_h, truth_c)

    # Bootstrap deltas
    delta_h_vs_p = paired_bootstrap_delta(preds_p, preds_h, truth_p, n_iter=2000, seed=42)
    delta_c_vs_p = paired_bootstrap_delta(preds_p, preds_c, truth_p, n_iter=2000, seed=42)
    delta_c_vs_h = paired_bootstrap_delta(preds_h, preds_c, truth_p, n_iter=2000, seed=42)

    print(f"\n  [{name}] RESULTS (test set: {res_p.n_test} scenarios)")
    print(f"    probe-only      F1={res_p.f1:.3f} P={res_p.precision:.3f} R={res_p.recall:.3f}")
    print(f"    holonomy-only   F1={res_h.f1:.3f} P={res_h.precision:.3f} R={res_h.recall:.3f}")
    print(f"    combined        F1={res_c.f1:.3f} P={res_c.precision:.3f} R={res_c.recall:.3f}")
    print(f"    Δ holo - probe   = {delta_h_vs_p[0]:+.3f}  CI95 [{delta_h_vs_p[1]:+.3f}, {delta_h_vs_p[2]:+.3f}]")
    print(f"    Δ comb - probe   = {delta_c_vs_p[0]:+.3f}  CI95 [{delta_c_vs_p[1]:+.3f}, {delta_c_vs_p[2]:+.3f}]")
    print(f"    Δ comb - holo    = {delta_c_vs_h[0]:+.3f}  CI95 [{delta_c_vs_h[1]:+.3f}, {delta_c_vs_h[2]:+.3f}]")

    return {
        "name": name,
        "n_b_pool": int(len(b_pool)),
        "n_h_pool": int(len(h_pool)),
        "n_test": res_p.n_test,
        "probe": {"f1": res_p.f1, "precision": res_p.precision, "recall": res_p.recall},
        "holonomy": {"f1": res_h.f1, "precision": res_h.precision, "recall": res_h.recall},
        "combined": {"f1": res_c.f1, "precision": res_c.precision, "recall": res_c.recall},
        "delta_holo_vs_probe": {"mean": delta_h_vs_p[0], "ci_lo": delta_h_vs_p[1], "ci_hi": delta_h_vs_p[2]},
        "delta_comb_vs_probe": {"mean": delta_c_vs_p[0], "ci_lo": delta_c_vs_p[1], "ci_hi": delta_c_vs_p[2]},
        "delta_comb_vs_holo": {"mean": delta_c_vs_h[0], "ci_lo": delta_c_vs_h[1], "ci_hi": delta_c_vs_h[2]},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--so-n", type=int, default=10)
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--attack-pos", type=int, default=2)
    parser.add_argument("--n-per-type", type=int, default=250)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for scenario generation and train/test split.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write JSON results.")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  COMBINED DETECTOR BENCHMARK")
    print(f"  conv_len={args.conv_len}, attack_pos={args.attack_pos}, "
          f"SO({args.so_n}), n_per_type={args.n_per_type}, test_size={args.test_size}")
    print("=" * 78)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    all_results = []

    t0 = time.time()
    for name in targets:
        b, h = DATASETS[name]()
        r = run_dataset(
            name, b, h,
            so_n=args.so_n,
            conv_len=args.conv_len,
            attack_pos=args.attack_pos,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
        )
        all_results.append(r)

    total = _LOGM_FALLBACKS["total"]
    fb = _LOGM_FALLBACKS["count"]
    fb_rate = (fb / total) if total else 0.0

    print()
    print("=" * 78)
    print(f"  SUMMARY  (total wall: {time.time() - t0:.1f}s, "
          f"logm-fallbacks: {fb}/{total} = {fb_rate:.2%})")
    print("=" * 78)
    print(f"  {'Dataset':<14} {'Probe':>7} {'Holo':>7} {'Comb':>7} "
          f"{'Δ H-P':>9} {'Δ C-P':>9} {'Δ C-H':>9}")
    print("  " + "-" * 76)
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['name']:<14} (skipped)")
            continue
        dhp = r["delta_holo_vs_probe"]
        dcp = r["delta_comb_vs_probe"]
        dch = r["delta_comb_vs_holo"]
        print(f"  {r['name']:<14} {r['probe']['f1']:>7.3f} {r['holonomy']['f1']:>7.3f} "
              f"{r['combined']['f1']:>7.3f} "
              f"{dhp['mean']:>+9.3f} {dcp['mean']:>+9.3f} {dch['mean']:>+9.3f}")
    print("  " + "-" * 76)
    print("  Δ X-Y = mean F1(X) - F1(Y) over 2000 paired bootstraps; CI95 in per-dataset detail above.")
    print("  Headline question: is Δ comb - probe > 0 with the CI excluding 0?")

    if fb_rate > 0.05:
        print(f"\n  WARN: logm fallback rate {fb_rate:.1%} exceeds 5%; results may be unreliable")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "logm_fallbacks": {"count": fb, "total": total, "rate": fb_rate},
                "results": all_results,
            }, f, indent=2)
        print(f"\n  Wrote results to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
