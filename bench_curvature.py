"""
bench_curvature — explicit non-Abelian features that raw_mean cannot reproduce.

Why this benchmark exists
--------------------------
bench_v3 found holonomy at parity (not better) than a 512-d raw mean
embedding. The mathematical reason is the Baker–Campbell–Hausdorff
formula:

    log( exp(A1) · exp(A2) · ... · exp(An) )
        = Σᵢ Aᵢ + ½ Σᵢ<ⱼ [Aᵢ, Aⱼ] + higher_order

The first term is exactly the per-step algebra coordinate sum — same
information raw_mean has access to. With the conventional
holonomy_scale = 0.1 used elsewhere, the commutator term [Aᵢ, Aⱼ] is
multiplied by 0.005 and is dominated by the linear sum at the
classifier's decision boundary. So "holonomy ≈ raw_mean" is BCH-1st-
order coincidence, not a fundamental limit.

This benchmark extracts the non-Abelian information *directly*, without
relying on BCH:

  • commutator_features:   stack [Aᵢ, Aⱼ] for all i < j
                           — zero in any Abelian group
                           — anti-symmetric in i,j (path-ordered)
                           — functions of pairs of messages, which a
                             permutation-invariant aggregator cannot
                             capture by construction
  • wilson_loop_features:  for each 2-step "plaquette"
                           Pᵢ = U_i U_{i+1} U_i† U_{i+1}†,
                           features include tr(P), tr(P²), tr(P^k) — the
                           gauge-invariant Wilson loop observables on a
                           1D lattice. These are the discrete-time
                           non-Abelian curvature observables.

Both feature sets are mathematically motivated by the Lie-group
structure (Yang–Mills lattice gauge theory in 1+1D) and are genuinely
not reproducible by raw_mean.

Capacity-matched comparison: each new feature set is built to roughly
match raw_mean's 512-d capacity by stacking enough commutators or
plaquettes.
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
from scipy.linalg import expm
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
    permutation_test_delta,
    vec_to_skew,
    skew_to_vec,
    make_scenarios,
    _LOGM,
    _dedupe,
    DATASETS,
)


# ---------------------------------------------------------------------------
# Curvature features
# ---------------------------------------------------------------------------

def commutator_features(vecs: np.ndarray, n: int = 10, scale: float = 0.5,
                        max_pairs: int | None = None) -> np.ndarray:
    """For each pair (i < j) of step algebra elements compute [Aᵢ, Aⱼ]
    and serialise the antisymmetric part to a vector.

    With conv_len = T, there are T*(T-1)/2 pairs. Each commutator is an
    n×n antisymmetric matrix, contributing n*(n-1)/2 values. To stay
    capacity-matched (target ~300-500 dims) we cap at max_pairs.

    `scale` is the per-step algebra scaling. Larger scale exposes the
    bracket information; we use 0.5 by default (vs 0.1 for the BCH
    approximation in bench_v3) so [Aᵢ, Aⱼ] is not dwarfed by Aᵢ + Aⱼ.
    """
    T = len(vecs)
    if T < 2:
        return np.zeros(0)
    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    feats = []
    skews = [vec_to_skew(v, n) * scale for v in vecs]
    for i, j in pairs:
        comm = skews[i] @ skews[j] - skews[j] @ skews[i]
        feats.append(skew_to_vec(comm))
    return np.concatenate(feats) if feats else np.zeros(0)


def wilson_loop_features(vecs: np.ndarray, n: int = 10, scale: float = 0.5,
                         loop_size: int = 2, n_powers: int = 3) -> np.ndarray:
    """Discrete Wilson-loop observables on a 1D lattice.

    For each window of `loop_size` consecutive link variables
        U_i = exp(scale · skew(v_i))
    form the loop W_i = U_i · U_{i+1} · U_i† · U_{i+1}†  (2-link plaquette
    in 1+1D — the smallest non-trivial closed loop).

    For each loop W, record:
        tr(W^k) for k = 1..n_powers
        Frobenius norm of (W - I)
        tr(W) - tr(W^T) (signed orientation feature)

    Wilson loops are gauge-invariant: any conjugation U_i → g U_i g^{-1}
    leaves tr(W) unchanged. They are zero in any Abelian group.
    """
    T = len(vecs)
    feats = []
    if T < loop_size + 1:
        return np.zeros(0)
    Us = [expm(vec_to_skew(v, n) * scale) for v in vecs]
    for i in range(T - loop_size):
        W = Us[i]
        for k in range(1, loop_size):
            W = W @ Us[i + k]
        W = W @ Us[i].T
        for k in range(1, loop_size):
            W = W @ Us[i + k].T
        # Per-loop observables
        loop_feats = []
        Wk = np.eye(n)
        for _ in range(n_powers):
            Wk = Wk @ W
            loop_feats.append(float(np.real(np.trace(Wk))))
        loop_feats.append(float(np.linalg.norm(W - np.eye(n), "fro")))
        loop_feats.append(float(np.real(np.trace(W) - np.trace(W.T))))
        feats.extend(loop_feats)
    return np.asarray(feats)


def commutator_plus_linear(vecs: np.ndarray, n: int = 10, scale: float = 0.5,
                           max_pairs: int | None = None) -> np.ndarray:
    """Concatenate the linear term Σᵢ Aᵢ (skew vector) and the explicit
    commutator features. This is the *full* second-order BCH expansion
    extracted directly, not approximated through expm/logm.
    """
    T = len(vecs)
    if T == 0:
        return np.zeros(0)
    skews = [vec_to_skew(v, n) * scale for v in vecs]
    linear = sum(skews)
    linear_vec = skew_to_vec(linear)
    comm = commutator_features(vecs, n=n, scale=scale, max_pairs=max_pairs)
    return np.concatenate([linear_vec, comm])


def _feature_dim_for(name: str, conv_len: int, n: int, max_pairs: int | None,
                     loop_size: int, n_powers: int) -> int:
    """Compute output dimension of each feature function (for sanity)."""
    so_dim = n * (n - 1) // 2
    pairs = conv_len * (conv_len - 1) // 2
    if max_pairs is not None:
        pairs = min(pairs, max_pairs)
    if name == "commutator":
        return so_dim * pairs
    if name == "commutator_plus_linear":
        return so_dim * (pairs + 1)
    if name == "wilson":
        n_loops = conv_len - loop_size
        return n_loops * (n_powers + 2)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Main experiment loop (multi-seed, multi-dataset)
# ---------------------------------------------------------------------------

def run_dataset_curvature(
    ds_info: dict,
    so_n: int,
    conv_len: int,
    n_per_type: int,
    test_size: float,
    seed: int,
    holonomy_scale: float,
    curvature_scale: float,
    max_pairs: int | None,
) -> dict:
    name = ds_info["name"]
    if ds_info.get("skipped"):
        return {"name": name, "skipped": True}
    splits = ds_info["splits"]

    # Reuse the bench_v3 split machinery for parity with prior results
    if "train" in splits and "test" in splits:
        train_b_raw, train_h_raw = splits["train"]
        test_b_raw, test_h_raw = splits["test"]
        train_b_set = set(train_b_raw)
        train_h_set = set(train_h_raw)
        test_b_pool = [t for t in test_b_raw if t not in train_b_set]
        test_h_pool = [t for t in test_h_raw if t not in train_h_set]
        train_b_pool = list(train_b_raw)
        train_h_pool = list(train_h_raw)
    else:
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

    if min(len(train_b_pool), len(train_h_pool), len(test_b_pool), len(test_h_pool)) < 30:
        return {"name": name, "skipped": True, "reason": "pool too small"}

    # Probe-direction sub-pool from train side only
    probe_rng = np.random.RandomState(seed + 13)
    n_b_probe = max(30, len(train_b_pool) // 4)
    n_h_probe = max(30, len(train_h_pool) // 4)
    n_b_probe = min(n_b_probe, len(train_b_pool) - 30)
    n_h_probe = min(n_h_probe, len(train_h_pool) - 30)
    b_perm = probe_rng.permutation(len(train_b_pool))
    h_perm = probe_rng.permutation(len(train_h_pool))
    probe_b_texts = [train_b_pool[i] for i in b_perm[:n_b_probe]]
    probe_h_texts = [train_h_pool[i] for i in h_perm[:n_h_probe]]
    train_b_remaining = [train_b_pool[i] for i in b_perm[n_b_probe:]]
    train_h_remaining = [train_h_pool[i] for i in h_perm[n_h_probe:]]

    print(f"  [{name}] encoding...")
    t0 = time.time()
    probe_b_v = encode(probe_b_texts)
    probe_h_v = encode(probe_h_texts)
    train_b_v = encode(train_b_remaining)
    train_h_v = encode(train_h_remaining)
    test_b_v = encode(test_b_pool)
    test_h_v = encode(test_h_pool)
    print(f"  [{name}] encoded in {time.time()-t0:.1f}s, emb_dim={train_b_v.shape[1]}")
    emb_dim = train_b_v.shape[1]

    diff = probe_h_v.mean(0) - probe_b_v.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    n_holo = so_n * (so_n - 1) // 2
    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, n_holo) / np.sqrt(emb_dim)

    n_train_per = int(n_per_type * (1.0 - test_size))
    n_test_per = n_per_type - n_train_per
    train_scenarios = make_scenarios(
        train_b_v, train_h_v,
        n_per_type=n_train_per, conv_len=conv_len, seed=seed,
    )
    test_scenarios = make_scenarios(
        test_b_v, test_h_v,
        n_per_type=n_test_per, conv_len=conv_len, seed=seed + 57,
    )

    _LOGM.reset()

    classifiers: dict[str, Callable[[dict], np.ndarray]] = {
        "probe": lambda s: probe_features(s["vecs"], probe_dir),
        "holonomy": lambda s: holonomy_features(s["vecs"], n=so_n,
                                                scale=holonomy_scale,
                                                label=s["label"]),
        "raw_mean": lambda s: raw_mean_features(s["vecs"]),
        "random_proj": lambda s: random_projection_features(s["vecs"], random_proj),
        "commutator": lambda s: commutator_features(
            s["vecs"], n=so_n, scale=curvature_scale, max_pairs=max_pairs),
        "commutator_plus_linear": lambda s: commutator_plus_linear(
            s["vecs"], n=so_n, scale=curvature_scale, max_pairs=max_pairs),
        "wilson": lambda s: wilson_loop_features(
            s["vecs"], n=so_n, scale=curvature_scale, loop_size=2, n_powers=3),
        # Strict-superset test: raw_mean concatenated with commutator
        # features. If commutators add ANY incremental signal over
        # raw_mean, this should beat raw_mean alone.
        "raw_mean_plus_commutator": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            commutator_features(s["vecs"], n=so_n, scale=curvature_scale,
                                max_pairs=max_pairs),
        ]),
    }

    results = {}
    preds = {}
    truth_arr = None
    for clf_name, fn in classifiers.items():
        res, p, t = fit_predict(train_scenarios, test_scenarios, fn)
        results[clf_name] = {
            "f1": res.f1, "precision": res.precision, "recall": res.recall,
            "n_features": res.n_features,
        }
        preds[clf_name] = p
        if truth_arr is None:
            truth_arr = t
        else:
            assert np.array_equal(truth_arr, t)

    # Pairwise comparisons (curvature-family vs the baselines)
    deltas = {}
    perm_p = {}
    for chal in ("commutator", "commutator_plus_linear", "wilson",
                 "raw_mean_plus_commutator"):
        for base in ("probe", "raw_mean", "random_proj", "holonomy"):
            key = f"{chal}_vs_{base}"
            deltas[key] = bca_bootstrap_delta(
                preds[base], preds[chal], truth_arr, n_iter=2000, seed=seed,
            )
            perm_p[key] = permutation_test_delta(
                preds[base], preds[chal], truth_arr, n_iter=2000, seed=seed,
            )

    print(f"\n  [{name}] RESULTS  (seed={seed}, n_test={results['probe']['n_test'] if 'n_test' in results['probe'] else len(test_scenarios)})")
    for cname, r in results.items():
        print(f"    {cname:<24} F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  ({r['n_features']}d)")
    for k, d in deltas.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    Δ {k:<40} = {d['obs_delta']:+.3f}  CI95 [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]{sig}")

    return {
        "name": name,
        "config": {
            "so_n": so_n, "conv_len": conv_len,
            "n_per_type": n_per_type, "test_size": test_size, "seed": seed,
            "holonomy_scale": holonomy_scale,
            "curvature_scale": curvature_scale,
            "max_pairs": max_pairs,
        },
        "classifiers": results,
        "deltas": deltas,
        "permutation_tests": perm_p,
        "logm_fallbacks": {
            "total": _LOGM.total, "fallbacks": _LOGM.fallbacks,
            "rate": _LOGM.fallbacks / max(1, _LOGM.total),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*",
                        default=["tensortrust", "neuralchemy", "deepset"])
    parser.add_argument("--so-n", type=int, default=10,
                        help="SO(n) for curvature features. Smaller than v3 "
                             "default since pairwise commutators amplify "
                             "feature count.")
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--n-per-type", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--holonomy-scale", type=float, default=0.1)
    parser.add_argument("--curvature-scale", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap on number of commutator pairs (default: all).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_curvature — explicit non-Abelian features")
    print(f"  SO({args.so_n}), conv_len={args.conv_len}, "
          f"n_per_type={args.n_per_type}, seed={args.seed}, "
          f"curvature_scale={args.curvature_scale}")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for name in args.datasets:
        ds_info = DATASETS[name]()
        r = run_dataset_curvature(
            ds_info,
            so_n=args.so_n,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
            holonomy_scale=args.holonomy_scale,
            curvature_scale=args.curvature_scale,
            max_pairs=args.max_pairs,
        )
        runs.append(r)

    print(f"\n  total wall: {time.time()-t0:.1f}s")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": vars(args), "runs": runs,
        }, indent=2, default=str))
        print(f"  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
