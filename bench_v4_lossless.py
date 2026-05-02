"""
bench_v4_lossless — five new mathematical approaches to multi-step detection.

The previous family of geometric features (holonomy, commutator, Wilson)
all routed embeddings through `vec_to_skew(v, n)` which kept only the
first n*(n-1)/2 entries of v. For SO(10) and a 512-d embedding that is
45 of 512 dims = 91% of embedding information thrown away. raw_mean
keeps all 512.

This bench tries five alternative mathematical structures, every one of
which is *information-preserving* with respect to the embedding:

  (A) **SO(33) holonomy** — for 512-d embeddings, n=33 gives 528 algebra
      dimensions ≥ 512, so vec_to_skew preserves ALL the embedding.

  (B) **Path signatures** (Chen 1957, Lyons 1998) — the universal
      approximator for path-dependent functionals. Level-1 is the
      net displacement; level-2 contains the Lévy area
      A^{ij} = ½∮(X_i dX_j − X_j dX_i) which is the iterated integral
      capturing non-commutative path information. We project to k=20
      then compute level-1 (20d) + level-2 (400d) = 420 features.
      The projection is FIXED (no information leakage between
      classifiers) and full-rank, so the embedding is nearly
      preserved up to JL distortion.

  (C) **Persistent homology** of the trajectory point cloud. Compute
      pairwise distances between message embeddings, build a
      Vietoris–Rips filtration, summarise as persistence-curve
      statistics (Betti-0 / Betti-1 lifetimes). These are
      topological invariants — invariant under continuous deformations
      of the embedding but sensitive to discrete cluster structure
      that raw_mean cannot see.

  (D) **Wasserstein distance** between the trajectory's empirical
      distribution and a benign-trajectory prior. Distributional
      distance, not aggregational. Computed via Sinkhorn divergence
      on the conv_len point cloud.

  (E) **Spectral features** — eigenvalues of the conv_len × conv_len
      Gram matrix `G_ij = <X_i, X_j>` and of the normalised graph
      Laplacian. Captures self-similarity / temporal-correlation
      structure that vanishes under naive averaging.

Each feature set is run against the v3 baselines (probe, raw_mean,
random_proj). The headline question: does ANY of (A)-(E) provide a
statistically significant lift over raw_mean?

Run: python3 bench_v4_lossless.py --seed 1
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
from scipy.linalg import expm, logm
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
    vec_to_skew,
    skew_to_vec,
    fit_predict,
    bca_bootstrap_delta,
    permutation_test_delta,
    make_scenarios,
    _LOGM,
    DATASETS,
)


# ---------------------------------------------------------------------------
# (B) Path signatures
# ---------------------------------------------------------------------------

def path_signature_features(vecs: np.ndarray, projection: np.ndarray,
                            level: int = 2) -> np.ndarray:
    """Truncated Chen path signature in a projected space.

    For T points X_0,...,X_{T-1} in R^d (after projection), compute:
      S^1 = X_{T-1} - X_0                                     (k dims)
      S^2_ij = Σ_{t=0}^{T-2} (X_t^i + X_{t+1}^i)/2 * dX_t^j   (k*k dims)
              where dX_t = X_{t+1} - X_t

    The trapezoidal rule for the iterated integral. The antisymmetric
    part 1/2(S^2 - S^2.T) is exactly the Lévy area; the symmetric part
    is 1/2 * (X_{T-1} ⊗ X_{T-1} - X_0 ⊗ X_0). We return the FULL S^2
    (so the classifier can learn what to use).

    For level 3 we add the iterated triple-integral (k^3 dims) — only
    enable when explicitly requested because the dimensionality blows
    up fast.
    """
    Y = vecs @ projection  # shape (T, k)
    T, k = Y.shape
    feats = []
    if T < 1:
        return np.zeros(k + (k * k if level >= 2 else 0))

    # Level 1: net displacement
    S1 = Y[-1] - Y[0]
    feats.append(S1)

    if level >= 2 and T >= 2:
        S2 = np.zeros((k, k))
        for t in range(T - 1):
            mid = 0.5 * (Y[t] + Y[t + 1])
            dX = Y[t + 1] - Y[t]
            S2 += np.outer(mid, dX)
        feats.append(S2.flatten())

    if level >= 3 and T >= 3:
        S3 = np.zeros((k, k, k))
        for t in range(T - 1):
            for u in range(t):
                mid_u = 0.5 * (Y[u] + Y[u + 1])
                dX_u = Y[u + 1] - Y[u]
                for v in range(u):
                    mid_v = 0.5 * (Y[v] + Y[v + 1])
                    S3 += np.einsum("i,j,k->ijk", mid_v, mid_u, Y[t + 1] - Y[t])
        feats.append(S3.flatten())

    return np.concatenate(feats)


def levy_area_only(vecs: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Pure antisymmetric Lévy area — the genuinely non-Abelian part of
    the path signature. Returns k(k-1)/2 dims (upper triangle only)."""
    Y = vecs @ projection
    T, k = Y.shape
    A = np.zeros((k, k))
    if T < 2:
        return np.zeros(k * (k - 1) // 2)
    for t in range(T - 1):
        # Lévy area increment: ½(X_t ⊗ dX_t - dX_t ⊗ X_t) (= antisym)
        Xt = Y[t]
        dX = Y[t + 1] - Y[t]
        A += 0.5 * (np.outer(Xt, dX) - np.outer(dX, Xt))
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            out.append(A[i, j])
    return np.asarray(out)


# ---------------------------------------------------------------------------
# (C) Persistent homology features (lightweight)
# ---------------------------------------------------------------------------

def persistent_homology_features(vecs: np.ndarray) -> np.ndarray:
    """Vietoris-Rips persistence summary statistics for the trajectory
    point cloud — without a TDA library, we compute simplified
    invariants that capture the same intuition.

    For T points, returns:
      - sorted upper-triangle pairwise distances (T*(T-1)/2 dims)
        (equivalent to the persistence diagram of H_0 up to Morse
        equivalence)
      - distance-matrix spectrum (T eigenvalues)
      - max-min ratio (a single proxy for cluster vs uniform structure)
    """
    T = len(vecs)
    feats = []
    # Pairwise distances
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            D[i, j] = np.linalg.norm(vecs[i] - vecs[j])
    upper = D[np.triu_indices(T, k=1)]
    sorted_dists = np.sort(upper)
    feats.append(sorted_dists)
    # Distance-matrix spectrum (T eigenvalues)
    eigs = np.linalg.eigvalsh(D)
    feats.append(np.sort(eigs))
    # Cluster proxy
    if upper.size and upper.min() > 1e-9:
        feats.append(np.array([upper.max() / upper.min()]))
    else:
        feats.append(np.array([0.0]))
    return np.concatenate(feats)


# ---------------------------------------------------------------------------
# (D) Wasserstein-to-benign prior
# ---------------------------------------------------------------------------

def make_wasserstein_features(benign_prior_pts: np.ndarray, n_proj: int,
                               proj_seed: int):
    """Returns a closure that for each scenario projects to n_proj dims
    and computes 1D Wasserstein distance to the benign prior in each
    of the n_proj projected coordinates. 1D Wasserstein is just sorted
    diff — fast and well-defined.
    """
    rng = np.random.RandomState(proj_seed)
    d = benign_prior_pts.shape[1]
    proj = rng.randn(d, n_proj) / np.sqrt(d)
    benign_proj = benign_prior_pts @ proj  # (N_benign, n_proj)
    # Pre-sort benign coords per dim
    benign_sorted = np.sort(benign_proj, axis=0)

    def feats(vecs: np.ndarray) -> np.ndarray:
        S = vecs @ proj  # (T, n_proj)
        out = np.zeros(n_proj)
        T = len(vecs)
        for d_idx in range(n_proj):
            scen_sorted = np.sort(S[:, d_idx])
            # Resample benign to T points by quantile interpolation
            benign_q = np.interp(
                np.linspace(0, 1, T),
                np.linspace(0, 1, len(benign_sorted)),
                benign_sorted[:, d_idx],
            )
            out[d_idx] = float(np.mean(np.abs(scen_sorted - benign_q)))
        return out
    return feats


# ---------------------------------------------------------------------------
# (E) Spectral features
# ---------------------------------------------------------------------------

def spectral_features(vecs: np.ndarray) -> np.ndarray:
    """Eigenvalues of the conv_len × conv_len Gram matrix and of its
    normalised Laplacian. Returns 2T values plus a few summary moments.
    """
    T = len(vecs)
    G = vecs @ vecs.T  # (T, T)
    # Symmetrise numerically
    G = 0.5 * (G + G.T)
    g_eigs = np.linalg.eigvalsh(G)
    # Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}
    W = np.exp(-G / (np.median(np.abs(G)) + 1e-12))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    deg = np.where(deg > 0, deg, 1.0)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(T) - Dinv @ W @ Dinv
    L = 0.5 * (L + L.T)
    l_eigs = np.linalg.eigvalsh(L)
    return np.concatenate([
        np.sort(g_eigs),
        np.sort(l_eigs),
        np.array([float(np.mean(g_eigs)),
                  float(np.std(g_eigs) + 1e-12),
                  float(np.mean(l_eigs)),
                  float(np.std(l_eigs) + 1e-12)]),
    ])


# ---------------------------------------------------------------------------
# Multi-method experiment driver
# ---------------------------------------------------------------------------

def run_dataset_v4(
    ds_info: dict,
    conv_len: int,
    n_per_type: int,
    test_size: float,
    seed: int,
    sig_proj_dim: int,
    levy_proj_dim: int,
    wasserstein_proj_dim: int,
) -> dict:
    name = ds_info["name"]
    if ds_info.get("skipped"):
        return {"name": name, "skipped": True}
    splits = ds_info["splits"]

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

    if min(len(train_b_pool), len(train_h_pool),
           len(test_b_pool), len(test_h_pool)) < 30:
        return {"name": name, "skipped": True, "reason": "pool too small"}

    probe_rng = np.random.RandomState(seed + 13)
    n_b_probe = max(30, len(train_b_pool) // 4)
    n_h_probe = max(30, len(train_h_pool) // 4)
    n_b_probe = min(n_b_probe, len(train_b_pool) - 30)
    n_h_probe = min(n_h_probe, len(train_h_pool) - 30)
    bperm = probe_rng.permutation(len(train_b_pool))
    hperm = probe_rng.permutation(len(train_h_pool))
    probe_b_texts = [train_b_pool[i] for i in bperm[:n_b_probe]]
    probe_h_texts = [train_h_pool[i] for i in hperm[:n_h_probe]]
    train_b_remaining = [train_b_pool[i] for i in bperm[n_b_probe:]]
    train_h_remaining = [train_h_pool[i] for i in hperm[n_h_probe:]]

    print(f"  [{name}] encoding...")
    t0 = time.time()
    probe_b_v = encode(probe_b_texts)
    probe_h_v = encode(probe_h_texts)
    train_b_v = encode(train_b_remaining)
    train_h_v = encode(train_h_remaining)
    test_b_v = encode(test_b_pool)
    test_h_v = encode(test_h_pool)
    emb_dim = train_b_v.shape[1]
    print(f"  [{name}] encoded in {time.time()-t0:.1f}s, emb_dim={emb_dim}")

    diff = probe_h_v.mean(0) - probe_b_v.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    # Random projections (fixed per seed) for path signatures and
    # Wasserstein. NEVER seen by training texts directly — built from
    # a deterministic seed.
    proj_rng_sig = np.random.RandomState(seed + 31)
    proj_sig = proj_rng_sig.randn(emb_dim, sig_proj_dim) / np.sqrt(emb_dim)

    proj_rng_levy = np.random.RandomState(seed + 32)
    proj_levy = proj_rng_levy.randn(emb_dim, levy_proj_dim) / np.sqrt(emb_dim)

    # Capacity-matched random_proj baseline (matches level-1 + level-2 sig dim)
    sig_total_dim = sig_proj_dim + sig_proj_dim * sig_proj_dim
    proj_rng_rand = np.random.RandomState(seed + 33)
    proj_random_capmatch = proj_rng_rand.randn(emb_dim, sig_total_dim) \
        / np.sqrt(emb_dim)

    # Wasserstein needs a benign prior — sample from train_b_v
    n_benign_prior = min(200, len(train_b_v))
    benign_prior_idx = np.random.RandomState(seed + 34).choice(
        len(train_b_v), n_benign_prior, replace=False
    )
    benign_prior_pts = train_b_v[benign_prior_idx]
    wasserstein_fn = make_wasserstein_features(
        benign_prior_pts, n_proj=wasserstein_proj_dim, proj_seed=seed + 35
    )

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
        # baselines
        "probe": lambda s: probe_features(s["vecs"], probe_dir),
        "raw_mean": lambda s: raw_mean_features(s["vecs"]),
        "random_proj_capmatch": lambda s: s["vecs"].mean(axis=0) @ proj_random_capmatch,
        # (A) lossless holonomy
        "holonomy_so33": lambda s: holonomy_features(
            s["vecs"], n=33, scale=0.1, label=s["label"]),
        # (B) path signatures
        "path_sig_lvl2": lambda s: path_signature_features(
            s["vecs"], proj_sig, level=2),
        "levy_area": lambda s: levy_area_only(s["vecs"], proj_levy),
        # (C) persistent homology
        "persistence": lambda s: persistent_homology_features(s["vecs"]),
        # (D) Wasserstein
        "wasserstein": lambda s: wasserstein_fn(s["vecs"]),
        # (E) spectral
        "spectral": lambda s: spectral_features(s["vecs"]),
        # (F) raw_mean ∪ best-of-each (strict superset to test
        # incremental signal)
        "raw_mean_plus_path_sig": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            path_signature_features(s["vecs"], proj_sig, level=2),
        ]),
        "raw_mean_plus_levy": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            levy_area_only(s["vecs"], proj_levy),
        ]),
        "raw_mean_plus_persistence": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            persistent_homology_features(s["vecs"]),
        ]),
        "raw_mean_plus_wasserstein": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            wasserstein_fn(s["vecs"]),
        ]),
        "raw_mean_plus_spectral": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            spectral_features(s["vecs"]),
        ]),
    }

    results = {}
    preds = {}
    truth_arr = None
    for clf_name, fn in classifiers.items():
        try:
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
        except Exception as e:
            print(f"    {clf_name:<26} FAILED: {type(e).__name__}: {e}")
            results[clf_name] = {"failed": str(e)}

    # The headline question for each (challenger): does it beat raw_mean?
    deltas_vs_raw_mean = {}
    for chal_name in classifiers:
        if "failed" in results[chal_name] or chal_name == "raw_mean":
            continue
        deltas_vs_raw_mean[chal_name] = bca_bootstrap_delta(
            preds["raw_mean"], preds[chal_name], truth_arr,
            n_iter=2000, seed=seed,
        )

    print(f"\n  [{name}] RESULTS  (seed={seed})")
    for cname, r in results.items():
        if "failed" in r:
            continue
        print(f"    {cname:<26} F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  ({r['n_features']}d)")
    print(f"\n  [{name}] Δ vs raw_mean (positive = beats raw_mean):")
    for cname, d in deltas_vs_raw_mean.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    {cname:<26} {d['obs_delta']:>+8.3f}  CI95 [{d['ci_lo']:>+7.3f}, {d['ci_hi']:>+7.3f}]{sig}")

    return {
        "name": name,
        "config": {
            "conv_len": conv_len, "n_per_type": n_per_type,
            "test_size": test_size, "seed": seed,
            "sig_proj_dim": sig_proj_dim,
            "levy_proj_dim": levy_proj_dim,
            "wasserstein_proj_dim": wasserstein_proj_dim,
        },
        "classifiers": results,
        "deltas_vs_raw_mean": deltas_vs_raw_mean,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*",
                        default=["tensortrust", "neuralchemy", "deepset"])
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--n-per-type", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sig-proj-dim", type=int, default=20,
                        help="Project embeddings to this dim before path-sig")
    parser.add_argument("--levy-proj-dim", type=int, default=30,
                        help="Project embeddings to this dim before Lévy area")
    parser.add_argument("--wasserstein-proj-dim", type=int, default=64)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_v4_lossless — five new mathematical approaches")
    print(f"  conv_len={args.conv_len}, n_per_type={args.n_per_type}, seed={args.seed}")
    print(f"  sig_proj_dim={args.sig_proj_dim}, levy_proj_dim={args.levy_proj_dim}, "
          f"wasserstein_proj_dim={args.wasserstein_proj_dim}")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for name in args.datasets:
        ds_info = DATASETS[name]()
        r = run_dataset_v4(
            ds_info,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
            sig_proj_dim=args.sig_proj_dim,
            levy_proj_dim=args.levy_proj_dim,
            wasserstein_proj_dim=args.wasserstein_proj_dim,
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
