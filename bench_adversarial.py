"""
bench_adversarial — does holonomy degrade gracefully under adversarial
                    perturbation of attack texts?

Motivation: bench_v3 shows that on clean inputs, holonomy is parity with
a 512-d raw mean embedding baseline. The honest open question is whether
holonomy's *aggregation over many algebra coordinates* makes it more
robust to small perturbations than a directional probe or a raw mean.

Hypothesis (to be tested, not assumed):
  - A directional probe scores along ONE harmful direction. A perturbation
    that pushes the embedding off that direction breaks detection.
  - Holonomy aggregates contributions across n*(n-1)/2 algebra
    coordinates. A perturbation has to break many of them simultaneously.
  - Therefore holonomy F1 should degrade slower than probe F1 as
    perturbation strength increases.

Construction:
  1. Train probe + holonomy + raw_mean classifiers on clean Neuralchemy
     train split (the dataset where v3 shows reasonable F1s).
  2. Take the test split, apply perturbations of increasing strength to
     the attack texts only:
       - "char": insert a random character every k characters
       - "swap": randomly swap adjacent characters
       - "truncate": keep only the first p% of the attack
  3. Measure F1 at each perturbation level.
  4. Compare the degradation rate per classifier.

Caveat: this benchmark uses simple character-level perturbations. A real
adversarial setup would use trained perturbation models (e.g. TextAttack)
that find embedding-aware adversarial examples. This benchmark is
necessary-but-not-sufficient; if holonomy doesn't help here, it almost
certainly doesn't help under stronger adversaries either.

Usage:
  python3 bench_adversarial.py --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_v3 import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    encode,
    holonomy_features,
    probe_features,
    raw_mean_features,
    random_projection_features,
    fit_predict,
    bca_bootstrap_delta,
    _LOGM,
    _dedupe,
    load_neuralchemy,
)


# ---------------------------------------------------------------------------
# Perturbation strategies
# ---------------------------------------------------------------------------

def perturb_char_insert(text: str, k: int, seed: int) -> str:
    if k <= 0 or len(text) == 0:
        return text
    rng = np.random.RandomState(seed)
    chars = "qwertyuiopasdfghjklzxcvbnm"
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if (i + 1) % k == 0:
            out.append(chars[int(rng.randint(0, len(chars)))])
    return "".join(out)


def perturb_swap_chars(text: str, n_swaps: int, seed: int) -> str:
    if n_swaps <= 0 or len(text) < 2:
        return text
    rng = np.random.RandomState(seed)
    arr = list(text)
    for _ in range(n_swaps):
        i = int(rng.randint(0, len(arr) - 1))
        arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return "".join(arr)


def perturb_truncate(text: str, keep_frac: float, seed: int = 0) -> str:
    if keep_frac >= 1.0:
        return text
    keep = max(20, int(len(text) * keep_frac))
    return text[:keep]


PERTURBATIONS = {
    "char_insert_k20": (lambda t, s: perturb_char_insert(t, 20, s),
                        "insert random char every 20"),
    "char_insert_k10": (lambda t, s: perturb_char_insert(t, 10, s),
                        "insert random char every 10"),
    "char_insert_k5":  (lambda t, s: perturb_char_insert(t, 5, s),
                        "insert random char every 5"),
    "swap_5":          (lambda t, s: perturb_swap_chars(t, 5, s),
                        "5 adjacent-char swaps"),
    "swap_20":         (lambda t, s: perturb_swap_chars(t, 20, s),
                        "20 adjacent-char swaps"),
    "swap_50":         (lambda t, s: perturb_swap_chars(t, 50, s),
                        "50 adjacent-char swaps"),
    "truncate_75":     (lambda t, s: perturb_truncate(t, 0.75, s),
                        "keep first 75% only"),
    "truncate_50":     (lambda t, s: perturb_truncate(t, 0.50, s),
                        "keep first 50% only"),
    "truncate_25":     (lambda t, s: perturb_truncate(t, 0.25, s),
                        "keep first 25% only"),
}


# ---------------------------------------------------------------------------
# Scenario builder (single-message — we are testing per-message robustness)
# ---------------------------------------------------------------------------

def make_single_message_scenarios(
    b_vecs: np.ndarray, h_vecs: np.ndarray,
    n_per_type: int, conv_len: int, seed: int,
) -> list[dict]:
    """Build conv_len-length scenarios with a single attack at a random
    position (the bench_v3 default setting, randomised attack_pos)."""
    rng = np.random.RandomState(seed)
    nb, nh = len(b_vecs), len(h_vecs)
    scenarios = []
    if nb == 0 or nh == 0:
        return []
    for _ in range(n_per_type):
        b_idx = rng.choice(nb, conv_len, replace=True)
        scenarios.append({"vecs": b_vecs[b_idx], "label": 0})

        ap = int(rng.randint(0, conv_len))
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
        scenarios.append({"vecs": vecs, "label": 1, "attack_idx": h_idx})
    return scenarios


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_adversarial(
    n_per_type: int = 400,
    conv_len: int = 4,
    test_size: float = 0.4,
    seed: int = 1,
    so_n: int = 25,
    holonomy_scale: float = 0.1,
) -> dict:
    ds = load_neuralchemy()
    if ds.get("skipped"):
        return {"skipped": True, "reason": ds.get("reason")}
    train_b_raw, train_h_raw = ds["splits"]["train"]
    test_b_raw, test_h_raw = ds["splits"]["test"]

    # Probe-train sub-pool from train side only
    rng = np.random.RandomState(seed)
    n_b_probe = max(50, len(train_b_raw) // 4)
    n_h_probe = max(50, len(train_h_raw) // 4)
    bperm = rng.permutation(len(train_b_raw))
    hperm = rng.permutation(len(train_h_raw))
    probe_b_texts = [train_b_raw[i] for i in bperm[:n_b_probe]]
    probe_h_texts = [train_h_raw[i] for i in hperm[:n_h_probe]]
    train_b_texts = [train_b_raw[i] for i in bperm[n_b_probe:]]
    train_h_texts = [train_h_raw[i] for i in hperm[n_h_probe:]]
    print(f"  pool: train {len(train_b_texts)}+{len(train_h_texts)}, "
          f"test {len(test_b_raw)}+{len(test_h_raw)}, "
          f"probe {n_b_probe}+{n_h_probe}")

    # Encode clean texts
    print(f"  encoding clean texts...")
    t0 = time.time()
    probe_b_v = encode(probe_b_texts)
    probe_h_v = encode(probe_h_texts)
    train_b_v = encode(train_b_texts)
    train_h_v = encode(train_h_texts)
    test_b_v = encode(test_b_raw)
    test_h_v = encode(test_h_raw)
    print(f"  encoded in {time.time()-t0:.1f}s, emb_dim={train_b_v.shape[1]}")
    emb_dim = train_b_v.shape[1]

    # Probe direction
    diff = probe_h_v.mean(0) - probe_b_v.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    n_holo = so_n * (so_n - 1) // 2
    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, n_holo) / np.sqrt(emb_dim)

    # Build train scenarios from clean train pool
    n_train_per = int(n_per_type * (1.0 - test_size))
    n_test_per = n_per_type - n_train_per
    train_scenarios = make_single_message_scenarios(
        train_b_v, train_h_v,
        n_per_type=n_train_per, conv_len=conv_len, seed=seed,
    )

    # Reusable feature functions
    def make_classifiers():
        return {
            "probe": lambda s: probe_features(s["vecs"], probe_dir),
            "holonomy": lambda s: holonomy_features(s["vecs"], n=so_n,
                                                    scale=holonomy_scale,
                                                    label=s["label"]),
            "raw_mean": lambda s: raw_mean_features(s["vecs"]),
            "random_proj": lambda s: random_projection_features(s["vecs"], random_proj),
        }

    # Pretrain each classifier ONCE on clean train scenarios; reuse for all
    # perturbation levels. This isolates the effect of perturbing the test
    # inputs — we are measuring degradation of a fixed model under
    # increasingly perturbed attacks.
    classifiers = make_classifiers()
    train_features = {}
    scalers = {}
    clfs = {}
    for name, fn in classifiers.items():
        feats = np.array([fn(s) for s in train_scenarios])
        feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
        labels = np.array([s["label"] for s in train_scenarios])
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.fit_transform(feats), labels)
        scalers[name] = sc
        clfs[name] = clf
        train_features[name] = feats.shape[1]

    print(f"  trained {len(classifiers)} classifiers on {len(train_scenarios)} train scenarios")

    def test_at_level(level_name: str, perturb_fn) -> dict:
        # Perturb every test attack text; benigns stay clean
        seed_p = seed + hash(level_name) % 10000
        if perturb_fn is None:
            perturbed_h_texts = list(test_h_raw)
        else:
            perturbed_h_texts = [perturb_fn(t, seed_p + i) for i, t in enumerate(test_h_raw)]
        # Re-encode perturbed attacks
        perturbed_h_v = encode(perturbed_h_texts)
        test_scenarios = make_single_message_scenarios(
            test_b_v, perturbed_h_v,
            n_per_type=n_test_per, conv_len=conv_len, seed=seed + 57,
        )
        labels = np.array([s["label"] for s in test_scenarios])
        out = {"name": level_name, "n_test": len(test_scenarios)}
        preds_by_clf = {}
        for name, fn in classifiers.items():
            feats = np.array([fn(s) for s in test_scenarios])
            feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            preds = clfs[name].predict(scalers[name].transform(feats))
            preds_by_clf[name] = preds
            out[name] = {
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
            }
        # Pairwise deltas: holonomy vs probe / raw_mean / random_proj
        out["deltas"] = {}
        for base in ("probe", "raw_mean", "random_proj"):
            key = f"holonomy_vs_{base}"
            out["deltas"][key] = bca_bootstrap_delta(
                preds_by_clf[base], preds_by_clf["holonomy"], labels,
                n_iter=2000, seed=seed,
            )
        return out

    print(f"\n  running test at clean baseline + {len(PERTURBATIONS)} perturbation levels...")
    levels = {"clean": test_at_level("clean", None)}
    for lname, (fn, _desc) in PERTURBATIONS.items():
        levels[lname] = test_at_level(lname, fn)

    print(f"\n  --- F1 across perturbations ---")
    print(f"  {'level':<18} {'probe':>8} {'holonomy':>10} {'raw_mean':>10} {'rand_proj':>10}")
    for lname, lv in levels.items():
        if lv.get("skipped"):
            continue
        print(f"  {lname:<18} "
              f"{lv['probe']['f1']:>8.3f} "
              f"{lv['holonomy']['f1']:>10.3f} "
              f"{lv['raw_mean']['f1']:>10.3f} "
              f"{lv['random_proj']['f1']:>10.3f}")

    # Degradation = clean_F1 - perturbed_F1 (positive = worse under perturbation)
    print(f"\n  --- Degradation (clean F1 - perturbed F1) ---")
    print(f"  {'level':<18} {'probe':>8} {'holonomy':>10} {'raw_mean':>10} {'rand_proj':>10}")
    clean_f1 = {n: levels["clean"][n]["f1"] for n in ("probe", "holonomy", "raw_mean", "random_proj")}
    for lname, lv in levels.items():
        if lname == "clean" or lv.get("skipped"):
            continue
        deg = {n: clean_f1[n] - lv[n]["f1"] for n in ("probe", "holonomy", "raw_mean", "random_proj")}
        print(f"  {lname:<18} "
              f"{deg['probe']:>+8.3f} "
              f"{deg['holonomy']:>+10.3f} "
              f"{deg['raw_mean']:>+10.3f} "
              f"{deg['random_proj']:>+10.3f}")

    print(f"\n  --- Δ holonomy vs others (perturbed) ---")
    print(f"  {'level':<18} {'Δ holo-probe':>16} {'Δ holo-raw_mean':>20} {'Δ holo-rand_proj':>20}")
    for lname, lv in levels.items():
        if lv.get("skipped"):
            continue
        d = lv["deltas"]
        print(f"  {lname:<18} "
              f"{d['holonomy_vs_probe']['obs_delta']:>+16.3f} "
              f"{d['holonomy_vs_raw_mean']['obs_delta']:>+20.3f} "
              f"{d['holonomy_vs_random_proj']['obs_delta']:>+20.3f}")

    return {
        "config": {
            "dataset": "neuralchemy",
            "n_per_type": n_per_type,
            "conv_len": conv_len,
            "test_size": test_size,
            "seed": seed,
            "so_n": so_n,
            "holonomy_scale": holonomy_scale,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
        },
        "levels": levels,
        "train_feature_dims": train_features,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-type", type=int, default=400)
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--so-n", type=int, default=25)
    parser.add_argument("--holonomy-scale", type=float, default=0.1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_adversarial — robustness to character-level perturbations")
    print(f"  conv_len={args.conv_len}, n_per_type={args.n_per_type}, "
          f"so_n={args.so_n}, seed={args.seed}")
    print("=" * 78)

    t0 = time.time()
    r = run_adversarial(
        n_per_type=args.n_per_type,
        conv_len=args.conv_len,
        test_size=args.test_size,
        seed=args.seed,
        so_n=args.so_n,
        holonomy_scale=args.holonomy_scale,
    )
    print(f"\n  wall: {time.time()-t0:.1f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, default=str))
        print(f"  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
