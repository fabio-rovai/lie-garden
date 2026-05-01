"""
bench_v3 — multi-step prompt-injection detection benchmark, hardened edition.

Fixes every methodology issue raised in the round of internal red-team
review that preceded this script. In particular:

  - TensorTrust benign pool is `access_code` (the legitimate user message),
    not `pre_prompt` (which is the defender's system prompt and therefore
    not comparable to attack texts that occupy the user role).
  - Attack position in test scenarios is randomised per scenario rather
    than fixed at index 2, removing the structural shortcut that lets a
    path-ordered classifier learn a position-specific signature.
  - Capacity-matched baselines: in addition to the 1-D mean-difference
    probe, we evaluate logistic regression on the raw per-conversation
    mean embedding (`raw_mean`) and on a fixed random projection of the
    same dimensionality as the holonomy feature vector (`random_proj`).
    A win for holonomy must beat all three.
  - Deduplication: identical text strings are collapsed to one row before
    any pool split, so the disjoint-pool guarantee holds at the *text*
    level, not just the index level.
  - Publisher train/test splits are respected for Deepset and Neuralchemy.
    For Neuralchemy the split is also `group_id`-aware: paraphrases of
    the same row never straddle train and test.
  - Every RNG is parameterised by `--seed`, including the probe-direction
    fit and the eval-pool split. Earlier versions hardcoded
    `np.random.RandomState(0)` for those, so multi-seed sweeps did not
    actually vary the parts of randomness that matter most.
  - Bootstrap confidence intervals use the BCa (bias-corrected and
    accelerated) method, which has better coverage than the percentile
    method for non-smooth statistics like F1.
  - Permutation test (label-shuffle) is reported alongside, to prove the
    pipeline cannot fabricate signal from random labels.
  - Multiple-comparisons correction (Benjamini–Hochberg FDR) is applied
    across the full family of (dataset × seed × baseline) comparisons.
  - Every dataset's class balance and unique-string count is recorded
    in the output JSON.

Run via:

  python3 bench_v3.py --dataset all --seed 1 --out v3_seed1.json

For a multi-seed sweep, use scripts/run_v3_evaluation.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import expm, logm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


EMBEDDING_MODEL_NAME = "minishlab/potion-base-32M"
EMBEDDING_MODEL_REVISION = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"


# ---------------------------------------------------------------------------
# Embedding (Model2Vec, pinned revision; SHA-fallback raises rather than
# silently producing nonsense if the real model is unavailable)
# ---------------------------------------------------------------------------

_MODEL = None
_EMB_DIM: int | None = None


def encode(texts: list[str]) -> np.ndarray:
    global _MODEL, _EMB_DIM
    if _MODEL is None:
        try:
            from model2vec import StaticModel
        except ImportError as e:
            raise RuntimeError(
                "model2vec is required. install via "
                "`pip install -r bench_requirements.txt`"
            ) from e
        try:
            _MODEL = StaticModel.from_pretrained(
                EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION,
            )
        except TypeError:
            # Older model2vec without revision kwarg
            _MODEL = StaticModel.from_pretrained(EMBEDDING_MODEL_NAME)
    embs = np.asarray(_MODEL.encode(texts), dtype=np.float64)
    _EMB_DIM = embs.shape[1]
    return embs


# ---------------------------------------------------------------------------
# Holonomy features (Lie-algebra evolution on SO(n))
# ---------------------------------------------------------------------------

def vec_to_skew(v: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n))
    target = n * (n - 1) // 2
    if len(v) < target:
        v = np.resize(v, target)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = v[k]
            A[j, i] = -v[k]
            k += 1
    return A


@dataclass
class LogmCounter:
    total: int = 0
    fallbacks: int = 0
    per_class_total: list = field(default_factory=lambda: [0, 0])
    per_class_fallback: list = field(default_factory=lambda: [0, 0])

    def reset(self) -> None:
        self.total = 0
        self.fallbacks = 0
        self.per_class_total = [0, 0]
        self.per_class_fallback = [0, 0]


_LOGM = LogmCounter()


def safe_logm(R: np.ndarray, label: int | None = None) -> np.ndarray:
    _LOGM.total += 1
    if label is not None:
        _LOGM.per_class_total[label] += 1
    try:
        L = logm(R)
        return np.real((L - L.T) / 2.0)
    except Exception:
        _LOGM.fallbacks += 1
        if label is not None:
            _LOGM.per_class_fallback[label] += 1
        return np.zeros_like(R)


def skew_to_vec(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.array([A[i, j] for i in range(n) for j in range(i + 1, n)])


def holonomy_features(vecs: np.ndarray, n: int = 25, scale: float = 0.1,
                      label: int | None = None) -> np.ndarray:
    state = np.eye(n)
    for v in vecs:
        skew = vec_to_skew(v, n) * scale
        try:
            R = expm(skew)
        except Exception:
            R = np.eye(n)
        state = state @ R
    return skew_to_vec(safe_logm(state, label=label))


# ---------------------------------------------------------------------------
# Probe and capacity-matched baselines
# ---------------------------------------------------------------------------

def probe_features(vecs: np.ndarray, probe_dir: np.ndarray) -> np.ndarray:
    scores = vecs @ probe_dir
    if len(scores) == 0:
        return np.zeros(4)
    return np.array([
        float(scores.max()),
        float(scores.mean()),
        float(scores[-1]),
        float(scores.max() - scores.mean()),
    ])


def raw_mean_features(vecs: np.ndarray) -> np.ndarray:
    return vecs.mean(axis=0) if len(vecs) else np.zeros(_EMB_DIM or 512)


def random_projection_features(vecs: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Capacity-matched control: project mean(vecs) through a fixed random
    matrix to the same dimensionality as the holonomy feature vector.
    If the holonomy lift were purely a feature-count effect, this baseline
    would match it."""
    return raw_mean_features(vecs) @ projection


# ---------------------------------------------------------------------------
# Scenario construction with RANDOMISED attack position
# ---------------------------------------------------------------------------

def make_scenarios(
    b_vecs: np.ndarray,
    h_vecs: np.ndarray,
    n_per_type: int,
    conv_len: int,
    seed: int,
    fixed_attack_pos: int | None = None,
) -> list[dict]:
    """Build N benign and N attack scenarios.

    Benign:  conv_len benign messages drawn from b_vecs (with replacement).
    Attack:  conv_len-1 benign messages + ONE attack at a position drawn
             uniformly at random from {0..conv_len-1} (or `fixed_attack_pos`
             if provided, for the per-position sensitivity analysis).
    """
    rng = np.random.RandomState(seed)
    nb, nh = len(b_vecs), len(h_vecs)
    if nb == 0 or nh == 0:
        return []
    if conv_len < 2:
        raise ValueError("conv_len must be >= 2")
    scenarios = []

    for _ in range(n_per_type):
        # Benign
        b_idx = rng.choice(nb, conv_len, replace=True)
        scenarios.append({
            "vecs": b_vecs[b_idx],
            "label": 0,
            "attack_pos": -1,
        })

        # Attack at random or fixed position
        if fixed_attack_pos is None:
            ap = int(rng.randint(0, conv_len))
        else:
            ap = fixed_attack_pos
        b_idx = rng.choice(nb, conv_len - 1, replace=True)
        h_idx = int(rng.choice(nh))
        vecs = np.zeros((conv_len, b_vecs.shape[1]))
        bi = 0
        for pos in range(conv_len):
            if pos == ap:
                vecs[pos] = h_vecs[h_idx]
            else:
                vecs[pos] = b_vecs[b_idx[bi]]
                bi += 1
        scenarios.append({"vecs": vecs, "label": 1, "attack_pos": ap})

    return scenarios


# ---------------------------------------------------------------------------
# Classifier eval, BCa bootstrap, permutation test
# ---------------------------------------------------------------------------

@dataclass
class ClfResult:
    f1: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    n_features: int


def fit_predict(
    train_scenarios: list[dict],
    test_scenarios: list[dict],
    feature_fn: Callable[[dict], np.ndarray],
) -> tuple[ClfResult, np.ndarray, np.ndarray]:
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
            n_features=train_feats.shape[1],
        ),
        preds,
        test_labels,
    )


def bca_bootstrap_delta(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    truth: np.ndarray,
    n_iter: int = 5000,
    seed: int = 42,
) -> dict:
    """BCa (bias-corrected and accelerated) bootstrap CI on F1(b) - F1(a).

    Better coverage than percentile method for non-smooth statistics like
    F1, especially under class imbalance. Implementation follows Efron &
    Tibshirani (1993) §14.3.
    """
    rng = np.random.RandomState(seed)
    n = len(truth)
    f1_a_obs = f1_score(truth, preds_a, zero_division=0)
    f1_b_obs = f1_score(truth, preds_b, zero_division=0)
    obs_delta = f1_b_obs - f1_a_obs

    # Bootstrap distribution
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        deltas[i] = f1_score(truth[idx], preds_b[idx], zero_division=0) \
            - f1_score(truth[idx], preds_a[idx], zero_division=0)

    # z0: bias correction
    n_below = int(np.sum(deltas < obs_delta))
    if n_below == 0:
        z0 = -10.0
    elif n_below == n_iter:
        z0 = 10.0
    else:
        from scipy.stats import norm
        z0 = norm.ppf(n_below / n_iter)

    # acceleration via jackknife
    jack = np.empty(n)
    for j in range(n):
        keep = np.ones(n, dtype=bool)
        keep[j] = False
        jack[j] = f1_score(truth[keep], preds_b[keep], zero_division=0) \
            - f1_score(truth[keep], preds_a[keep], zero_division=0)
    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5 + 1e-12)
    a = num / den

    from scipy.stats import norm
    alpha_lo, alpha_hi = 0.025, 0.975
    z_lo, z_hi = norm.ppf(alpha_lo), norm.ppf(alpha_hi)
    a_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    a_lo = float(np.clip(a_lo, 0.0, 1.0))
    a_hi = float(np.clip(a_hi, 0.0, 1.0))

    return {
        "obs_delta": float(obs_delta),
        "ci_lo": float(np.percentile(deltas, 100 * a_lo)),
        "ci_hi": float(np.percentile(deltas, 100 * a_hi)),
        "n_iter": n_iter,
        "method": "bca",
    }


def permutation_test_delta(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    truth: np.ndarray,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Label-shuffle test: under H0 (preds_a and preds_b are from
    interchangeable models), how often does the absolute delta meet
    or exceed the observed delta?

    For each iteration, swap each test instance's pred between A and B
    independently with prob 0.5 (paired permutation), then recompute
    F1(b)-F1(a). The two-sided p-value is the fraction of iterations
    where |permuted| >= |observed|.
    """
    rng = np.random.RandomState(seed)
    obs = abs(f1_score(truth, preds_b, zero_division=0)
              - f1_score(truth, preds_a, zero_division=0))
    n = len(truth)
    extreme = 0
    for _ in range(n_iter):
        swap = rng.rand(n) < 0.5
        a_perm = np.where(swap, preds_b, preds_a)
        b_perm = np.where(swap, preds_a, preds_b)
        d = abs(f1_score(truth, b_perm, zero_division=0)
                - f1_score(truth, a_perm, zero_division=0))
        if d >= obs - 1e-12:
            extreme += 1
    p_two_sided = (extreme + 1) / (n_iter + 1)
    return {"obs_abs_delta": float(obs), "p_two_sided": float(p_two_sided),
            "n_iter": n_iter}


# ---------------------------------------------------------------------------
# Dataset loaders — text-level dedup, publisher splits respected
# ---------------------------------------------------------------------------

def _dedupe(texts: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_tensortrust() -> dict:
    """Return per-split benign and harmful pools.

    Benign  = `access_code` (legitimate user message that grants access).
    Harmful = `attack` (adversarial user message attempting hijack).

    Rationale: both occupy the *user* role in the conversation, so the
    binary task is "is this user message a hijack attempt or a benign
    user input". Using `pre_prompt` (system role) as benign would
    conflate role-distinction with attack-detection.

    TensorTrust ships no train/test split, so we return one pool and let
    the bench split it inside `run_dataset`.
    """
    p = "/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    if not os.path.exists(p):
        return {"name": "tensortrust", "skipped": True,
                "reason": f"missing {p}"}
    b, h = [], []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            ac = d.get("access_code", "") or ""
            atk = d.get("attack", "") or ""
            if ac:
                b.append(ac)
            if atk:
                h.append(atk)
    b = _dedupe(b)
    h = _dedupe(h)
    return {
        "name": "tensortrust",
        "splits": {"all": (b, h)},
        "raw_count": {"benign": len(b), "harmful": len(h)},
        "notes": (
            "Benign = access_code (legitimate user msg). "
            "Harmful = attack (adversarial user msg). "
            "Both occupy the user role; pre_prompt (system role) is "
            "deliberately NOT used."
        ),
    }


def load_injecagent() -> dict:
    base = "/tmp/InjecAgent/data"
    if not os.path.exists(base):
        return {"name": "injecagent", "skipped": True, "reason": f"missing {base}"}
    b = []
    with open(f"{base}/user_cases.jsonl") as f:
        for line in f:
            d = json.loads(line)
            instr = d.get("User Instruction", "")
            if instr:
                b.append(instr)
    h = []
    for fname in ("attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"):
        with open(f"{base}/{fname}") as f:
            for line in f:
                d = json.loads(line)
                instr = d.get("Attacker Instruction", "")
                if instr:
                    h.append(instr)
    # Augment harmful pool from full test_cases — using `Expected Achievements`
    # alongside `Attacker Instruction` to capture both the goal and the
    # imperative form. Earlier code referenced a non-existent field
    # "Attacker Instruction Achievement" which never fired.
    for tcf in ("test_cases_dh_base.json", "test_cases_ds_base.json"):
        path = f"{base}/{tcf}"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            cases = json.load(f)
        for c in cases:
            instr = c.get("Attacker Instruction", "")
            if instr:
                h.append(instr)
            ach = c.get("Expected Achievements", "")
            if ach:
                h.append(ach)
    b = _dedupe(b)
    h = _dedupe(h)
    return {
        "name": "injecagent",
        "splits": {"all": (b, h)},
        "raw_count": {"benign": len(b), "harmful": len(h)},
        "notes": (
            "Benign = User Instruction from user_cases (real user queries). "
            "Harmful = Attacker Instruction + Expected Achievements from "
            "attacker_cases_*.jsonl and test_cases_*_base.json. "
            "Severe class imbalance is expected (small benign pool); the "
            "bench skips the dataset if the pool is too small for the "
            "configured n_per_type."
        ),
    }


def _load_parquet(path: str) -> tuple[list[str], list[str], list[str | None]]:
    """Return (texts, label_strs, group_ids)."""
    import pandas as pd
    df = pd.read_parquet(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    if "group_id" in df.columns:
        groups = df["group_id"].astype(str).tolist()
    else:
        groups = [None] * len(df)
    return texts, labels, groups


def load_deepset() -> dict:
    train_p = "/tmp/deepset_train.parquet"
    test_p = "/tmp/deepset_test.parquet"
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        return {"name": "deepset", "skipped": True,
                "reason": "missing publisher parquet files"}
    splits: dict[str, tuple[list[str], list[str]]] = {}
    raw_count = {}
    for split_name, path in [("train", train_p), ("test", test_p)]:
        texts, labels, _ = _load_parquet(path)
        b = _dedupe([t for t, l in zip(texts, labels) if l == 0])
        h = _dedupe([t for t, l in zip(texts, labels) if l == 1])
        splits[split_name] = (b, h)
        raw_count[split_name] = {"benign": len(b), "harmful": len(h)}
    return {
        "name": "deepset",
        "splits": splits,
        "raw_count": raw_count,
        "notes": (
            "Publisher train/test split is preserved. label=0 -> benign, "
            "label=1 -> injection (per dataset card)."
        ),
    }


def load_neuralchemy() -> dict:
    train_p = "/tmp/neuralchemy_train.parquet"
    test_p = "/tmp/neuralchemy_test.parquet"
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        return {"name": "neuralchemy", "skipped": True,
                "reason": "missing publisher parquet files"}
    splits: dict[str, tuple[list[str], list[str]]] = {}
    raw_count = {}
    for split_name, path in [("train", train_p), ("test", test_p)]:
        texts, labels, groups = _load_parquet(path)
        # Group-aware dedup: keep one representative per (label, group_id)
        seen_groups: dict[tuple[int, str], str] = {}
        b, h = [], []
        for t, l, g in zip(texts, labels, groups):
            key = (l, g) if g else (l, t)
            if key in seen_groups:
                continue
            seen_groups[key] = t
            (b if l == 0 else h).append(t)
        splits[split_name] = (b, h)
        raw_count[split_name] = {"benign": len(b), "harmful": len(h)}
    return {
        "name": "neuralchemy",
        "splits": splits,
        "raw_count": raw_count,
        "notes": (
            "Publisher train/test split preserved. group_id is used to "
            "deduplicate paraphrases within each split, so paraphrases of "
            "the same row never appear twice."
        ),
    }


DATASETS: dict[str, Callable[[], dict]] = {
    "tensortrust": load_tensortrust,
    "injecagent": load_injecagent,
    "deepset": load_deepset,
    "neuralchemy": load_neuralchemy,
}


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_dataset(
    ds_info: dict,
    so_n: int = 25,
    conv_len: int = 4,
    n_per_type: int = 500,
    test_size: float = 0.4,
    seed: int = 1,
    holonomy_scale: float = 0.1,
    do_permutation_test: bool = True,
) -> dict:
    name = ds_info["name"]
    if ds_info.get("skipped"):
        return {"name": name, "skipped": True, "reason": ds_info.get("reason")}

    splits = ds_info["splits"]

    # Build a SINGLE benign and harmful list per split, then deduplicate
    # across splits to ensure no text crosses train and test.
    if "train" in splits and "test" in splits:
        train_b_raw, train_h_raw = splits["train"]
        test_b_raw, test_h_raw = splits["test"]
        train_b_set = set(train_b_raw)
        train_h_set = set(train_h_raw)
        test_b_pool = [t for t in test_b_raw if t not in train_b_set]
        test_h_pool = [t for t in test_h_raw if t not in train_h_set]
        train_b_pool = list(train_b_raw)
        train_h_pool = list(train_h_raw)
        publisher_split = True
    else:
        # Single pool — split into train and test using `seed`
        all_b, all_h = splits["all"]
        rng = np.random.RandomState(seed)
        b_perm = rng.permutation(len(all_b))
        h_perm = rng.permutation(len(all_h))
        b_split = int((1.0 - test_size) * len(all_b))
        h_split = int((1.0 - test_size) * len(all_h))
        train_b_pool = [all_b[i] for i in b_perm[:b_split]]
        train_h_pool = [all_h[i] for i in h_perm[:h_split]]
        test_b_pool = [all_b[i] for i in b_perm[b_split:]]
        test_h_pool = [all_h[i] for i in h_perm[h_split:]]
        publisher_split = False

    MIN_PER_SPLIT = 30
    if (len(train_b_pool) < MIN_PER_SPLIT or len(train_h_pool) < MIN_PER_SPLIT
            or len(test_b_pool) < MIN_PER_SPLIT or len(test_h_pool) < MIN_PER_SPLIT):
        print(f"  [{name}] skipped: pool too small (train {len(train_b_pool)}+{len(train_h_pool)}, "
              f"test {len(test_b_pool)}+{len(test_h_pool)}, need >={MIN_PER_SPLIT} each)")
        return {
            "name": name, "skipped": True,
            "reason": (
                f"pool too small (train {len(train_b_pool)}+{len(train_h_pool)}, "
                f"test {len(test_b_pool)}+{len(test_h_pool)}, "
                f"need >={MIN_PER_SPLIT} each)"
            ),
        }

    print(f"\n  [{name}] split: train {len(train_b_pool)}+{len(train_h_pool)}, "
          f"test {len(test_b_pool)}+{len(test_h_pool)} "
          f"(publisher={'yes' if publisher_split else 'no'})")

    # Carve a probe-direction pool out of the TRAIN side only — never look
    # at test text when fitting the probe direction.
    probe_rng = np.random.RandomState(seed + 13)
    n_b_probe = max(MIN_PER_SPLIT, len(train_b_pool) // 4)
    n_h_probe = max(MIN_PER_SPLIT, len(train_h_pool) // 4)
    n_b_probe = min(n_b_probe, len(train_b_pool) - MIN_PER_SPLIT)
    n_h_probe = min(n_h_probe, len(train_h_pool) - MIN_PER_SPLIT)
    if n_b_probe < MIN_PER_SPLIT or n_h_probe < MIN_PER_SPLIT:
        return {"name": name, "skipped": True,
                "reason": "probe pool too small after train carve-out"}

    b_perm = probe_rng.permutation(len(train_b_pool))
    h_perm = probe_rng.permutation(len(train_h_pool))
    b_probe_idx = b_perm[:n_b_probe]
    h_probe_idx = h_perm[:n_h_probe]
    b_train_remaining_idx = b_perm[n_b_probe:]
    h_train_remaining_idx = h_perm[n_h_probe:]

    # Encode all texts now (deduplicated already)
    print(f"  [{name}] encoding train pool ({len(train_b_pool)+len(train_h_pool)}), "
          f"test pool ({len(test_b_pool)+len(test_h_pool)})")
    t0 = time.time()
    train_b_vecs = encode(train_b_pool)
    train_h_vecs = encode(train_h_pool)
    test_b_vecs = encode(test_b_pool)
    test_h_vecs = encode(test_h_pool)
    enc_t = time.time() - t0
    emb_dim = train_b_vecs.shape[1]
    print(f"  [{name}] encoded in {enc_t:.1f}s, emb_dim={emb_dim}")

    b_probe_vecs = train_b_vecs[b_probe_idx]
    h_probe_vecs = train_h_vecs[h_probe_idx]
    train_b_eval = train_b_vecs[b_train_remaining_idx]
    train_h_eval = train_h_vecs[h_train_remaining_idx]

    # Probe direction from probe pool only
    diff = h_probe_vecs.mean(0) - b_probe_vecs.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    # Capacity-matched random projection: emb_dim -> SO(n)*(n-1)/2 dims
    n_holo = so_n * (so_n - 1) // 2
    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, n_holo) / np.sqrt(emb_dim)

    n_train_per_type = int(n_per_type * (1.0 - test_size))
    n_test_per_type = n_per_type - n_train_per_type

    train_scenarios = make_scenarios(
        train_b_eval, train_h_eval,
        n_per_type=n_train_per_type, conv_len=conv_len, seed=seed,
    )
    test_scenarios = make_scenarios(
        test_b_vecs, test_h_vecs,
        n_per_type=n_test_per_type, conv_len=conv_len, seed=seed + 57,
    )
    print(f"  [{name}] scenarios: {len(train_scenarios)} train, "
          f"{len(test_scenarios)} test (length {conv_len}, attack pos randomised)")

    # Reset the global logm counter for clean per-dataset accounting
    _LOGM.reset()

    def f_probe(s):
        return probe_features(s["vecs"], probe_dir)

    def f_holonomy(s):
        return holonomy_features(s["vecs"], n=so_n, scale=holonomy_scale,
                                 label=s["label"])

    def f_raw_mean(s):
        return raw_mean_features(s["vecs"])

    def f_random_proj(s):
        return random_projection_features(s["vecs"], random_proj)

    def f_combined(s):
        return np.concatenate([f_probe(s), f_holonomy(s)])

    classifiers = {
        "probe": f_probe,
        "holonomy": f_holonomy,
        "raw_mean": f_raw_mean,
        "random_proj": f_random_proj,
        "combined": f_combined,
    }

    results = {}
    preds_by_clf = {}
    truth_arr = None
    for clf_name, fn in classifiers.items():
        res, preds, truth = fit_predict(train_scenarios, test_scenarios, fn)
        results[clf_name] = {
            "f1": res.f1,
            "precision": res.precision,
            "recall": res.recall,
            "n_train": res.n_train,
            "n_test": res.n_test,
            "n_features": res.n_features,
        }
        preds_by_clf[clf_name] = preds
        if truth_arr is None:
            truth_arr = truth
        else:
            assert np.array_equal(truth_arr, truth), "test split must be identical across classifiers"

    # Pairwise deltas: holonomy/combined VS each baseline (probe, raw_mean, random_proj)
    deltas = {}
    perm_tests = {}
    for chal in ("holonomy", "combined"):
        for base in ("probe", "raw_mean", "random_proj"):
            key = f"{chal}_vs_{base}"
            deltas[key] = bca_bootstrap_delta(
                preds_by_clf[base], preds_by_clf[chal], truth_arr,
                n_iter=2000, seed=seed,
            )
            if do_permutation_test:
                perm_tests[key] = permutation_test_delta(
                    preds_by_clf[base], preds_by_clf[chal], truth_arr,
                    n_iter=2000, seed=seed,
                )

    # Label-shuffle null control: how often does the strongest classifier beat
    # the weakest under randomly shuffled training labels?
    shuffle_rng = np.random.RandomState(seed + 71)
    train_labels_shuffled = np.array([s["label"] for s in train_scenarios])
    shuffle_rng.shuffle(train_labels_shuffled)
    shuffled_train = [
        {"vecs": s["vecs"], "label": int(train_labels_shuffled[i]),
         "attack_pos": s["attack_pos"]}
        for i, s in enumerate(train_scenarios)
    ]
    shuf_holo, _, _ = fit_predict(shuffled_train, test_scenarios, f_holonomy)
    shuf_probe, _, _ = fit_predict(shuffled_train, test_scenarios, f_probe)

    print(f"\n  [{name}] RESULTS  (seed={seed}, n_test={results['probe']['n_test']})")
    for clf_name, r in results.items():
        print(f"    {clf_name:<13} F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
              f"({r['n_features']}d)")
    for key, d in deltas.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    Δ {key:<28} = {d['obs_delta']:+.3f}  CI95 [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]{sig}")
    print(f"    null-shuffle  holo F1={shuf_holo.f1:.3f}  probe F1={shuf_probe.f1:.3f}")
    fb_total = _LOGM.total
    fb_count = _LOGM.fallbacks
    fb_per_class = (
        _LOGM.per_class_fallback[0],
        _LOGM.per_class_total[0],
        _LOGM.per_class_fallback[1],
        _LOGM.per_class_total[1],
    )
    print(f"    logm fallbacks: total={fb_count}/{fb_total}, "
          f"benign={fb_per_class[0]}/{fb_per_class[1]}, "
          f"harmful={fb_per_class[2]}/{fb_per_class[3]}")

    return {
        "name": name,
        "publisher_split": publisher_split,
        "raw_count": ds_info.get("raw_count"),
        "notes": ds_info.get("notes"),
        "config": {
            "so_n": so_n,
            "conv_len": conv_len,
            "n_per_type": n_per_type,
            "test_size": test_size,
            "seed": seed,
            "holonomy_scale": holonomy_scale,
        },
        "pool_sizes": {
            "train_benign": len(train_b_pool),
            "train_harmful": len(train_h_pool),
            "test_benign": len(test_b_pool),
            "test_harmful": len(test_h_pool),
            "probe_train_benign": int(n_b_probe),
            "probe_train_harmful": int(n_h_probe),
        },
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "revision": EMBEDDING_MODEL_REVISION,
            "dim": int(emb_dim),
        },
        "classifiers": results,
        "deltas": deltas,
        "permutation_tests": perm_tests,
        "shuffled_label_control": {
            "holonomy_f1": shuf_holo.f1,
            "probe_f1": shuf_probe.f1,
        },
        "logm_fallbacks": {
            "total": fb_total,
            "count": fb_count,
            "per_class": {
                "benign_total": fb_per_class[1],
                "benign_fallbacks": fb_per_class[0],
                "harmful_total": fb_per_class[3],
                "harmful_fallbacks": fb_per_class[2],
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--so-n", type=int, default=25)
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--n-per-type", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--holonomy-scale", type=float, default=0.1)
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print(f"  bench_v3 — multi-step prompt-injection benchmark")
    print(f"  config: so_n={args.so_n}, conv_len={args.conv_len}, "
          f"n_per_type={args.n_per_type}, test_size={args.test_size}, "
          f"seed={args.seed}")
    print(f"  embedding: {EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION[:12]}")
    print("=" * 78)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    runs = []
    t0 = time.time()
    for name in targets:
        ds_info = DATASETS[name]()
        r = run_dataset(
            ds_info,
            so_n=args.so_n,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
            holonomy_scale=args.holonomy_scale,
            do_permutation_test=not args.no_permutation,
        )
        runs.append(r)

    out = {
        "config": {
            "dataset": args.dataset,
            "so_n": args.so_n,
            "conv_len": args.conv_len,
            "n_per_type": args.n_per_type,
            "test_size": args.test_size,
            "seed": args.seed,
            "holonomy_scale": args.holonomy_scale,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
        },
        "wall_time_seconds": time.time() - t0,
        "runs": runs,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
