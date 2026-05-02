"""
bench_deviation — features that are well-defined on lossy embeddings.

Motivation
----------
The previous Fourier-of-sequence experiments (bench_v5) revealed that the
discriminative signal lives almost entirely in the DC component of the
DFT (= raw_mean). Higher frequencies of the embedding sequence carry
near-zero incremental discriminative power.

The user's insight: classical Fourier transforms presuppose
**reconstructibility** — the basis is complete and the inverse exists.
For sentence embeddings the original signal (text) cannot be
reconstructed from the embedding regardless of basis, so Fourier-style
features inherit that lossiness without exploiting it.

The fix is to drop the reconstruction requirement and use features that
are well-defined on lossy projections — measures of how a path
**deviates** rather than measures that try to reconstruct the path.

Three families:

  (M) **Magnitude features.** Per-step embedding norms, step-to-step
      delta magnitudes ||X_t − X_{t-1}||, total path length, max step,
      step-magnitude coefficient of variation. ∼10 dims.

  (R) **Rhythm features.** Consecutive-step magnitude ratios
      ||dX_t|| / ||dX_{t-1}|| (well-defined; no inverse), spectral
      entropy of the step-magnitude time series, lag-1 autocorrelation,
      and ratio of local-to-global magnitude. ∼10 dims.

  (B) **Frequency-band-ratio features.** Don't try to invert the
      spectrum. Instead bin the per-step magnitude spectrum into
      low/mid/high bands and report energy *ratios* between bands.
      Spectral centroid, rolloff, and flatness — all scalars derived
      from the spectrum without reconstruction. ∼10 dims.

  (D) **Deviation-from-benign-baseline features.** Ratios of path
      statistics to the same statistics estimated on benign training
      conversations. E.g. (current_path_length / mean_benign_path_length).
      No reconstruction. ∼10 dims.

Total: ∼30-40 dims. The test is the strict-superset comparison:
does **raw_mean ∪ deviation** beat **raw_mean** alone?

Note on the "rhythm" interpretation: the user's framing is that an
attack changes the conversation's *rhythm* of magnitude/frequency
deviations from a benign baseline. These features quantify that
deviation directly without trying to reconstruct the underlying signal.
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
    probe_features,
    raw_mean_features,
    fit_predict,
    bca_bootstrap_delta,
    permutation_test_delta,
    make_scenarios,
    DATASETS,
)


# ---------------------------------------------------------------------------
# (M) Magnitude features
# ---------------------------------------------------------------------------

def magnitude_features(vecs: np.ndarray) -> np.ndarray:
    """Per-step embedding norms and step-to-step delta magnitudes.

    All scalars, all well-defined under any lossy embedding map."""
    T = len(vecs)
    if T == 0:
        return np.zeros(8)
    norms = np.linalg.norm(vecs, axis=1)  # (T,)
    if T >= 2:
        deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    else:
        deltas = np.zeros(0)
    feats = []
    feats.append(norms.mean())
    feats.append(norms.std() + 1e-12)
    feats.append(norms.max())
    feats.append(norms.min())
    if len(deltas):
        feats.append(deltas.sum())  # total path length
        feats.append(deltas.mean())
        feats.append(deltas.std() + 1e-12)
        feats.append(deltas.max())
    else:
        feats.extend([0.0, 0.0, 1e-12, 0.0])
    return np.asarray(feats)


# ---------------------------------------------------------------------------
# (R) Rhythm features
# ---------------------------------------------------------------------------

def _spectral_entropy(x: np.ndarray) -> float:
    """Shannon entropy of the (positive) power spectrum of x — measures
    'rhythm' (low entropy = periodic; high entropy = noisy)."""
    if len(x) < 2:
        return 0.0
    F = np.abs(np.fft.rfft(x - x.mean()))
    p = F / (F.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    x = x - x.mean()
    num = float(np.sum(x[:-lag] * x[lag:]))
    den = float(np.sum(x * x) + 1e-12)
    return num / den


def rhythm_features(vecs: np.ndarray) -> np.ndarray:
    """Ratio-based and entropy-based rhythm descriptors."""
    T = len(vecs)
    if T < 2:
        return np.zeros(8)
    deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    feats = []
    # Consecutive-step magnitude ratios
    if len(deltas) >= 2:
        ratios = deltas[1:] / (deltas[:-1] + 1e-12)
        feats.append(float(ratios.mean()))
        feats.append(float(ratios.std() + 1e-12))
        feats.append(float(ratios.max()))
        feats.append(float(ratios.min()))
    else:
        feats.extend([1.0, 1e-12, 1.0, 1.0])
    # Spectral entropy of step-magnitude time series
    feats.append(_spectral_entropy(deltas))
    # Autocorrelation of step magnitudes at lag 1
    feats.append(_autocorr(deltas, lag=1))
    # Local-to-global magnitude ratio (max-step / mean-step)
    feats.append(float(deltas.max() / (deltas.mean() + 1e-12)))
    # Coefficient of variation
    feats.append(float(deltas.std() / (deltas.mean() + 1e-12)))
    return np.asarray(feats)


# ---------------------------------------------------------------------------
# (B) Frequency-band-ratio features
# ---------------------------------------------------------------------------

def freq_band_features(vecs: np.ndarray) -> np.ndarray:
    """Spectral descriptors that don't require reconstruction.

    Computes the power spectrum of the step-magnitude time series and
    summarises it via band-energy ratios, centroid, rolloff, flatness.
    """
    T = len(vecs)
    if T < 3:
        return np.zeros(8)
    deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    if len(deltas) < 2:
        return np.zeros(8)
    # Power spectrum of step-magnitudes
    P = np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2
    if P.sum() < 1e-20:
        return np.zeros(8)
    n = len(P)
    # Bands: low/mid/high — third-by-third
    third = max(1, n // 3)
    low = P[:third].sum()
    mid = P[third:2 * third].sum()
    high = P[2 * third:].sum()
    total = low + mid + high + 1e-12
    feats = []
    feats.append(low / total)
    feats.append(mid / total)
    feats.append(high / total)
    feats.append(high / (low + 1e-12))   # high/low ratio
    feats.append(mid / (low + 1e-12))    # mid/low ratio
    # Spectral centroid (weighted average frequency index)
    freqs = np.arange(n)
    feats.append(float((freqs * P).sum() / (P.sum() + 1e-12)))
    # Spectral rolloff (freq below which 85% of energy lies)
    cum = np.cumsum(P)
    rolloff_idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
    feats.append(rolloff_idx / max(1, n - 1))
    # Spectral flatness (geometric mean / arithmetic mean)
    P_pos = P[P > 0]
    if len(P_pos) > 0:
        flat = float(np.exp(np.mean(np.log(P_pos))) / (np.mean(P_pos) + 1e-12))
    else:
        flat = 0.0
    feats.append(flat)
    return np.asarray(feats)


# ---------------------------------------------------------------------------
# (D) Deviation-from-benign-baseline features
# ---------------------------------------------------------------------------

class BenignDeviationStats:
    """Pre-compute mean/std of various path statistics over the benign
    training set, then express each test scenario's stats as ratios /
    z-scores relative to the benign baseline.

    Distribution-free, no inverse map required.
    """

    def __init__(self):
        self.stats: dict = {}

    def fit(self, benign_vecs_per_scenario: list[np.ndarray]) -> "BenignDeviationStats":
        from collections import defaultdict
        accum = defaultdict(list)
        for vecs in benign_vecs_per_scenario:
            T = len(vecs)
            if T < 2:
                continue
            norms = np.linalg.norm(vecs, axis=1)
            deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
            accum["mean_norm"].append(norms.mean())
            accum["max_norm"].append(norms.max())
            accum["mean_delta"].append(deltas.mean())
            accum["max_delta"].append(deltas.max())
            accum["path_length"].append(deltas.sum())
            accum["delta_std"].append(deltas.std())
            accum["spectral_entropy"].append(_spectral_entropy(deltas))
            accum["centroid"].append(
                float((np.arange(len(deltas)) *
                       (np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2)).sum())
                / (float((np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2).sum()) + 1e-12)
            )
        for k, v in accum.items():
            arr = np.asarray(v)
            self.stats[k] = (arr.mean(), arr.std() + 1e-12)
        return self

    def features(self, vecs: np.ndarray) -> np.ndarray:
        T = len(vecs)
        if T < 2 or not self.stats:
            return np.zeros(len(self.stats) * 2 if self.stats else 16)
        norms = np.linalg.norm(vecs, axis=1)
        deltas = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
        cur = {
            "mean_norm": norms.mean(),
            "max_norm": norms.max(),
            "mean_delta": deltas.mean(),
            "max_delta": deltas.max(),
            "path_length": deltas.sum(),
            "delta_std": deltas.std(),
            "spectral_entropy": _spectral_entropy(deltas),
            "centroid": float(
                (np.arange(len(deltas)) *
                 (np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2)).sum()
            ) / (float((np.abs(np.fft.rfft(deltas - deltas.mean())) ** 2).sum()) + 1e-12),
        }
        feats = []
        for k, v in cur.items():
            mu, sigma = self.stats.get(k, (0.0, 1.0))
            feats.append(v / (mu + 1e-12))         # ratio
            feats.append((v - mu) / (sigma + 1e-12))  # z-score
        return np.asarray(feats)


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_dataset_deviation(
    ds_info: dict,
    conv_len: int,
    n_per_type: int,
    test_size: float,
    seed: int,
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

    # Fit benign baseline on TRAIN benign scenarios only
    benign_train_vecs = [s["vecs"] for s in train_scenarios if s["label"] == 0]
    dev_stats = BenignDeviationStats().fit(benign_train_vecs)
    print(f"  [{name}] benign baseline fitted on {len(benign_train_vecs)} scenarios")

    def all_deviation_features(s):
        return np.concatenate([
            magnitude_features(s["vecs"]),
            rhythm_features(s["vecs"]),
            freq_band_features(s["vecs"]),
            dev_stats.features(s["vecs"]),
        ])

    classifiers: dict[str, Callable[[dict], np.ndarray]] = {
        "probe": lambda s: probe_features(s["vecs"], probe_dir),
        "raw_mean": lambda s: raw_mean_features(s["vecs"]),
        # Individual deviation families
        "magnitude": lambda s: magnitude_features(s["vecs"]),
        "rhythm": lambda s: rhythm_features(s["vecs"]),
        "freq_band": lambda s: freq_band_features(s["vecs"]),
        "benign_deviation": lambda s: dev_stats.features(s["vecs"]),
        # All deviation families combined
        "all_deviation": all_deviation_features,
        # Strict-superset tests: does deviation add to raw_mean?
        "raw_mean_plus_magnitude": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]), magnitude_features(s["vecs"])]),
        "raw_mean_plus_rhythm": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]), rhythm_features(s["vecs"])]),
        "raw_mean_plus_freq_band": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]), freq_band_features(s["vecs"])]),
        "raw_mean_plus_benign_deviation": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]), dev_stats.features(s["vecs"])]),
        "raw_mean_plus_all_deviation": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]), all_deviation_features(s)]),
    }

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
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
        except Exception as e:
            results[clf_name] = {"failed": f"{type(e).__name__}: {e}"}

    deltas = {}
    for cname in classifiers:
        if cname == "raw_mean" or "failed" in results[cname]:
            continue
        deltas[cname] = bca_bootstrap_delta(
            preds["raw_mean"], preds[cname], truth_arr,
            n_iter=2000, seed=seed,
        )

    print(f"\n  [{name}] RESULTS  (seed={seed})")
    for cname, r in results.items():
        if "failed" in r:
            print(f"    {cname:<32} FAILED: {r['failed']}")
            continue
        print(f"    {cname:<32} F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  ({r['n_features']}d)")
    print(f"\n  [{name}] Δ vs raw_mean (positive = beats raw_mean):")
    for cname, d in deltas.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    {cname:<32} {d['obs_delta']:>+8.3f}  CI95 [{d['ci_lo']:>+7.3f}, {d['ci_hi']:>+7.3f}]{sig}")

    return {
        "name": name,
        "config": {
            "conv_len": conv_len, "n_per_type": n_per_type,
            "test_size": test_size, "seed": seed,
        },
        "classifiers": results,
        "deltas_vs_raw_mean": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*",
                        default=["tensortrust", "neuralchemy", "deepset"])
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--n-per-type", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_deviation — non-reconstructive features")
    print(f"  (magnitude / rhythm / frequency-band-ratio / benign-deviation)")
    print(f"  conv_len={args.conv_len}, n_per_type={args.n_per_type}, seed={args.seed}")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for name in args.datasets:
        ds_info = DATASETS[name]()
        r = run_dataset_deviation(
            ds_info,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
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
