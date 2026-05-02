"""
bench_v6_proper — corrected implementations of the multi-step features.

Earlier benchmarks had implementation flaws that made them fail for
reasons unrelated to the underlying mathematics:

  - vec_to_skew truncated 512-d embeddings to the first 45 dims for
    SO(10), discarding 91% of the signal by INDEX rather than by
    a content-preserving projection.
  - "persistent homology" was sorted pairwise distances, not the actual
    H_0 persistence diagram (which for a finite point cloud is the
    multiset of MST edge weights).
  - Spectral / rhythm / frequency-band features ran on conv_len=4
    sequences, where the FFT has only 2 bins and entropy is bounded
    above by log 2 ≈ 0.69 — degenerate by construction.
  - Wasserstein was 1-D-projected per random direction, not the
    full Sinkhorn OT in embedding space.
  - Path signatures used random projection rather than PCA; random
    directions destroy the discriminative subspace.
  - CHC was trained on scenario MEANS (high variance from averaging
    4 random benign messages), not on individual benign embeddings.

This bench corrects all six and runs them at conv_len=20 (long enough
for spectral methods to be meaningful). Small-sample by design:
n_per_type=100, single seed, single dataset.

Methods (each properly implemented):

  • probe                — 4-D directional probe (baseline)
  • raw_mean             — 512-D mean-pooled embedding (baseline)
  • holonomy_proper      — random Gaussian projection 512→45 then SO(10)
                           holonomy. No truncation.
  • h0_persistence       — Real MST-based H_0 persistence diagram
                           (T-1 sorted edge weights). The actual
                           topological invariant.
  • path_sig_pca         — Path signature level-1 (k dims) + level-2
                           (k² dims) in a PCA-projected subspace
                           fitted on benign training. The Lévy area
                           lives in the directions that actually carry
                           signal.
  • sinkhorn_ot          — Sinkhorn OT distance between the scenario
                           point-cloud and a benign anchor cloud,
                           computed in 50-D PCA subspace for tractability.
                           Single scalar per scenario.
  • spectral_proper      — FFT power spectrum of step-magnitudes plus
                           band-energy ratios (now meaningful with 19
                           inter-step deltas).
  • chc_on_individuals   — High-D CHC trained on individual benign
                           message embeddings (~5000 on Neuralchemy,
                           ~600 on TensorTrust). Tighter envelope,
                           anomaly detection over messages not means.
                           Aggregated over the T messages per scenario.

The headline test for each: does (raw_mean ∪ method) beat raw_mean
alone? If yes, the method captures incremental signal; if no, the
method's information is redundant with mean-pooling.
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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_v3 import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    encode,
    probe_features,
    raw_mean_features,
    fit_predict,
    bca_bootstrap_delta,
    permutation_test_delta,
    make_scenarios,
    DATASETS,
)


# ---------------------------------------------------------------------------
# Proper holonomy: Gaussian projection (not truncation)
# ---------------------------------------------------------------------------

def make_skew_projector(emb_dim: int, n: int, seed: int):
    """Return a (emb_dim, n*(n-1)/2) Gaussian projection matrix that
    maps 512-d embeddings to skew-symmetric algebra coords without
    discarding info by index."""
    rng = np.random.RandomState(seed)
    target = n * (n - 1) // 2
    P = rng.randn(emb_dim, target) / np.sqrt(emb_dim)
    return P


def vec_to_skew_via_projection(v: np.ndarray, n: int, projection: np.ndarray) -> np.ndarray:
    """Project full embedding onto skew-coord subspace via random
    Gaussian map, then unpack to antisymmetric matrix."""
    coeffs = v @ projection  # (n*(n-1)/2,)
    A = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = coeffs[k]
            A[j, i] = -coeffs[k]
            k += 1
    return A


def skew_to_vec(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.array([A[i, j] for i in range(n) for j in range(i + 1, n)])


def holonomy_proper(vecs: np.ndarray, n: int, scale: float, projection: np.ndarray) -> np.ndarray:
    state = np.eye(n)
    for v in vecs:
        skew = vec_to_skew_via_projection(v, n, projection) * scale
        try:
            R = expm(skew)
        except Exception:
            R = np.eye(n)
        state = state @ R
    try:
        L = logm(state)
        L = np.real((L - L.T) / 2.0)
    except Exception:
        L = np.zeros_like(state)
    return skew_to_vec(L)


# ---------------------------------------------------------------------------
# Real H_0 persistence via MST
# ---------------------------------------------------------------------------

def h0_persistence(vecs: np.ndarray) -> np.ndarray:
    """For T points in R^d, the H_0 persistence diagram in the
    Vietoris-Rips filtration is the multiset of MST edge weights
    (T-1 deaths; the single H_0 generator that survives to infinity
    is the connected component).

    Returns: sorted T-1 MST edge weights + summary stats.
    """
    T = len(vecs)
    if T < 2:
        return np.zeros(0)
    D = squareform(pdist(vecs))
    mst = minimum_spanning_tree(D).toarray()
    weights = mst[mst > 0]
    weights_sorted = np.sort(weights)
    # Pad/truncate to fixed length T-1 (should always be T-1 for
    # connected complete graph, but be safe)
    target = T - 1
    if len(weights_sorted) < target:
        out = np.zeros(target)
        out[:len(weights_sorted)] = weights_sorted
    else:
        out = weights_sorted[:target]
    # Summary stats
    summary = np.array([
        float(out.mean()),
        float(out.std() + 1e-12),
        float(out.max()),
        float(out.min()),
        float(out.sum()),  # total MST weight = persistence-1 norm
    ])
    return np.concatenate([out, summary])


# ---------------------------------------------------------------------------
# Path signatures with PCA projection
# ---------------------------------------------------------------------------

def path_sig_pca(vecs: np.ndarray, pca: PCA) -> np.ndarray:
    """Level-1 + level-2 path signature in a PCA-projected subspace.

    Level-1: net displacement (k dims).
    Level-2: trapezoidal-rule iterated integral S^{ij} = (1/2) sum_t
             (Y_t^i + Y_{t+1}^i) * (Y_{t+1}^j - Y_t^j). The Lévy
             area = antisymmetric part = (1/2)(S^{ij} - S^{ji}).
             We return the FULL S^2 (k² dims) so the classifier
             sees both the symmetric quadratic part and the Lévy area.
    """
    Y = pca.transform(vecs)  # (T, k)
    T, k = Y.shape
    if T < 1:
        return np.zeros(k + k * k)
    S1 = Y[-1] - Y[0]
    S2 = np.zeros((k, k))
    for t in range(T - 1):
        mid = 0.5 * (Y[t] + Y[t + 1])
        dX = Y[t + 1] - Y[t]
        S2 += np.outer(mid, dX)
    return np.concatenate([S1, S2.flatten()])


# ---------------------------------------------------------------------------
# Sinkhorn OT
# ---------------------------------------------------------------------------

def sinkhorn_distance(X: np.ndarray, Y: np.ndarray, eps: float = 0.1,
                      n_iter: int = 50) -> float:
    """Entropic-regularised OT distance between empirical
    distributions over X (n points) and Y (m points) with uniform
    weights. Returns <P, C> where P is the optimal transport plan."""
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    K = np.exp(-C / eps)
    u = np.ones(n)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-12)
        u = a / (K @ v + 1e-12)
    P = u[:, None] * K * v[None, :]
    return float((P * C).sum())


def make_sinkhorn_fn(benign_anchor_pca: np.ndarray, pca: PCA,
                     eps: float = 0.5, n_iter: int = 30):
    def fn(vecs: np.ndarray) -> np.ndarray:
        Y = pca.transform(vecs)
        d = sinkhorn_distance(Y, benign_anchor_pca, eps=eps, n_iter=n_iter)
        return np.array([d])  # 1-D feature
    return fn


# ---------------------------------------------------------------------------
# Spectral on long step-magnitude time series
# ---------------------------------------------------------------------------

def spectral_proper(vecs: np.ndarray) -> np.ndarray:
    """Spectral features computed on the step-magnitude time series.
    Requires conv_len >= 10 to be meaningful (≥9 deltas → ≥5 freq bins).
    Returns power-spectrum bins + band-energy ratios + summary stats.
    """
    T = len(vecs)
    if T < 4:
        return np.zeros(20)
    deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    P = np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2
    n_bins = len(P)
    # Pad/truncate to fixed length 12 bins (assumes conv_len ~ 20-24)
    target_bins = 12
    if n_bins < target_bins:
        Pn = np.zeros(target_bins)
        Pn[:n_bins] = P
    else:
        Pn = P[:target_bins]
    # Normalise
    if Pn.sum() > 0:
        Pn_norm = Pn / Pn.sum()
    else:
        Pn_norm = Pn
    # Band-energy ratios (low/mid/high)
    third = max(1, target_bins // 3)
    low = Pn_norm[:third].sum()
    mid = Pn_norm[third:2 * third].sum()
    high = Pn_norm[2 * third:].sum()
    bands = np.array([low, mid, high,
                      high / (low + 1e-12),
                      mid / (low + 1e-12)])
    # Summary
    if Pn.sum() > 0:
        centroid = float((np.arange(target_bins) * Pn).sum() / Pn.sum())
    else:
        centroid = 0.0
    Pp = Pn[Pn > 0]
    flat = float(np.exp(np.mean(np.log(Pp + 1e-12))) /
                 (Pn.mean() + 1e-12)) if len(Pp) else 0.0
    summary = np.array([
        centroid, flat,
        float(deltas.mean()),
        float(deltas.std() + 1e-12),
        float(deltas.max()),
    ])
    # Concatenate: 12 power-spectrum bins + 5 band ratios + 5 summary = 22
    return np.concatenate([Pn_norm, bands, summary])


# ---------------------------------------------------------------------------
# CHC on individual benign embeddings (proper training)
# ---------------------------------------------------------------------------

class HighDimCHCIndividuals:
    """High-D CHC fitted on individual message embeddings (not scenario
    means). Tighter envelope; aggregated over T messages per scenario."""

    def __init__(self, n_directions: int = 200, seed: int = 0):
        self.n_directions = n_directions
        self.seed = seed
        self.directions: np.ndarray | None = None
        self.benign_min: np.ndarray | None = None
        self.benign_max: np.ndarray | None = None
        self.benign_mean: np.ndarray | None = None
        self.benign_std: np.ndarray | None = None
        self.benign_sorted: np.ndarray | None = None

    def fit(self, individual_benign: np.ndarray) -> "HighDimCHCIndividuals":
        N, d = individual_benign.shape
        rng = np.random.RandomState(self.seed)
        D = rng.randn(d, self.n_directions)
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
        self.directions = D
        proj = individual_benign @ D
        self.benign_min = proj.min(axis=0)
        self.benign_max = proj.max(axis=0)
        self.benign_mean = proj.mean(axis=0)
        self.benign_std = proj.std(axis=0) + 1e-12
        self.benign_sorted = np.sort(proj, axis=0)
        return self

    def aggregate_features(self, vecs: np.ndarray) -> np.ndarray:
        """Per-message overshoot/zscore/percentile, aggregated across the
        T messages of a scenario via max + mean."""
        T, d = vecs.shape
        proj = vecs @ self.directions  # (T, K)
        # Overshoot per (msg, dir)
        below = np.maximum(0.0, self.benign_min - proj)
        above = np.maximum(0.0, proj - self.benign_max)
        overshoot = below + above  # (T, K)
        # Z-score per (msg, dir)
        z = (proj - self.benign_mean) / self.benign_std
        # Percentile per (msg, dir) — vectorised searchsorted
        N = self.benign_sorted.shape[0]
        K = self.benign_sorted.shape[1]
        ranks = np.zeros((T, K))
        for k in range(K):
            ranks[:, k] = np.searchsorted(self.benign_sorted[:, k], proj[:, k]) / N
        # Aggregate over messages: max + mean per direction
        out = np.concatenate([
            overshoot.max(axis=0),  # K dims
            overshoot.mean(axis=0),  # K dims
            np.abs(z).max(axis=0),  # K dims
            ranks.max(axis=0),       # K dims
            ranks.min(axis=0),       # K dims (deviation below benign)
        ])
        return out


# ---------------------------------------------------------------------------
# Per-dataset experiment driver
# ---------------------------------------------------------------------------

def run_v6(
    ds_info: dict,
    conv_len: int,
    n_per_type: int,
    test_size: float,
    seed: int,
    holo_n: int,
    holo_scale: float,
    pca_dim: int,
    sinkhorn_eps: float,
    chc_directions: int,
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

    # Probe direction
    diff = probe_h_v.mean(0) - probe_b_v.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    # Fit PCA on benign training (for path-sig and Sinkhorn projections)
    pca = PCA(n_components=pca_dim).fit(train_b_v)
    print(f"  [{name}] PCA-{pca_dim} fitted: explained variance = "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # Skew projector for proper holonomy
    skew_proj = make_skew_projector(emb_dim, holo_n, seed=seed + 21)

    # Benign anchor for Sinkhorn (sample ~200 individual benign embeddings, in PCA space)
    n_anchor = min(200, len(train_b_v))
    anchor_idx = np.random.RandomState(seed + 41).choice(
        len(train_b_v), n_anchor, replace=False
    )
    anchor_pca = pca.transform(train_b_v[anchor_idx])
    sinkhorn_fn = make_sinkhorn_fn(anchor_pca, pca, eps=sinkhorn_eps, n_iter=30)

    # CHC on individual benign embeddings (training set, NOT scenario means)
    chc = HighDimCHCIndividuals(n_directions=chc_directions, seed=seed + 31)
    chc.fit(train_b_v)
    print(f"  [{name}] CHC fitted on {len(train_b_v)} individual benign messages")

    # Build scenarios
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
    print(f"  [{name}] scenarios: {len(train_scenarios)} train, "
          f"{len(test_scenarios)} test (conv_len={conv_len})")

    classifiers: dict[str, Callable[[dict], np.ndarray]] = {
        "probe": lambda s: probe_features(s["vecs"], probe_dir),
        "raw_mean": lambda s: raw_mean_features(s["vecs"]),
        "holonomy_proper": lambda s: holonomy_proper(
            s["vecs"], n=holo_n, scale=holo_scale, projection=skew_proj),
        "h0_persistence": lambda s: h0_persistence(s["vecs"]),
        "path_sig_pca": lambda s: path_sig_pca(s["vecs"], pca),
        "sinkhorn_ot": sinkhorn_fn,
        "spectral_proper": lambda s: spectral_proper(s["vecs"]),
        "chc_individuals": lambda s: chc.aggregate_features(s["vecs"]),
        # Strict-superset tests
        "raw_mean_plus_holonomy": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            holonomy_proper(s["vecs"], n=holo_n, scale=holo_scale,
                            projection=skew_proj),
        ]),
        "raw_mean_plus_h0": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            h0_persistence(s["vecs"]),
        ]),
        "raw_mean_plus_path_sig": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            path_sig_pca(s["vecs"], pca),
        ]),
        "raw_mean_plus_sinkhorn": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            sinkhorn_fn(s["vecs"]),
        ]),
        "raw_mean_plus_spectral": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            spectral_proper(s["vecs"]),
        ]),
        "raw_mean_plus_chc": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            chc.aggregate_features(s["vecs"]),
        ]),
    }

    # Wrap sinkhorn since make_sinkhorn_fn returns a fn taking vecs not s
    real_classifiers = {}
    for name_, fn in classifiers.items():
        if name_ == "sinkhorn_ot":
            real_classifiers[name_] = lambda s: sinkhorn_fn(s["vecs"])
        else:
            real_classifiers[name_] = fn

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    truth_arr = None
    for clf_name, fn in real_classifiers.items():
        try:
            res, p, t = fit_predict(train_scenarios, test_scenarios, fn)
            results[clf_name] = {
                "f1": res.f1, "precision": res.precision, "recall": res.recall,
                "n_features": res.n_features,
            }
            preds[clf_name] = p
            if truth_arr is None:
                truth_arr = t
        except Exception as e:
            results[clf_name] = {"failed": f"{type(e).__name__}: {e}"}

    deltas = {}
    for cname in real_classifiers:
        if cname == "raw_mean" or "failed" in results[cname]:
            continue
        deltas[cname] = bca_bootstrap_delta(
            preds["raw_mean"], preds[cname], truth_arr,
            n_iter=2000, seed=seed,
        )

    print(f"\n  [{name}] RESULTS  (seed={seed})")
    for cname, r in results.items():
        if "failed" in r:
            print(f"    {cname:<26} FAILED: {r['failed']}")
            continue
        print(f"    {cname:<26} F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  ({r['n_features']}d)")
    print(f"\n  [{name}] Δ vs raw_mean (positive = beats raw_mean):")
    for cname, d in deltas.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    {cname:<26} {d['obs_delta']:>+8.3f}  CI95 [{d['ci_lo']:>+7.3f}, {d['ci_hi']:>+7.3f}]{sig}")

    return {
        "name": name,
        "config": {
            "conv_len": conv_len, "n_per_type": n_per_type,
            "test_size": test_size, "seed": seed,
            "holo_n": holo_n, "holo_scale": holo_scale,
            "pca_dim": pca_dim, "sinkhorn_eps": sinkhorn_eps,
            "chc_directions": chc_directions,
        },
        "classifiers": results,
        "deltas_vs_raw_mean": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*",
                        default=["tensortrust"])
    parser.add_argument("--conv-len", type=int, default=20)
    parser.add_argument("--n-per-type", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--holo-n", type=int, default=10)
    parser.add_argument("--holo-scale", type=float, default=0.1)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--sinkhorn-eps", type=float, default=0.5)
    parser.add_argument("--chc-directions", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_v6_proper — corrected implementations of multi-step features")
    print(f"  conv_len={args.conv_len}, n_per_type={args.n_per_type}, "
          f"seed={args.seed}, pca_dim={args.pca_dim}, "
          f"holo SO({args.holo_n})")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for name in args.datasets:
        ds_info = DATASETS[name]()
        r = run_v6(
            ds_info,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
            holo_n=args.holo_n,
            holo_scale=args.holo_scale,
            pca_dim=args.pca_dim,
            sinkhorn_eps=args.sinkhorn_eps,
            chc_directions=args.chc_directions,
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
