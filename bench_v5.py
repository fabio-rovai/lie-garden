"""
bench_v5 — Fourier of the sequence + high-dim Convex Hull Calipers.

Two new mathematical frames informed by:
  (i)   the user's observation that this is a Fourier-transform problem;
  (ii)  unpublished work on Convex Hull Caliper Classifiers (CHC) — see
        /Users/fabio/projects/convex_hull_classifier.py.

(F) **Sequence-DFT features.**
    For a conversation X_1, ..., X_T in R^d, the discrete Fourier
    transform along the time axis is
        F_k = Σ_t X_t · exp(-2πikt/T).
    The DC component F_0 = T · raw_mean. The other components carry
    exactly the information raw_mean throws away. Three features sets:

      - "concat"        — all T·d entries (= same info as full DFT, just
                          unrotated). Sanity test: if this doesn't beat
                          raw_mean, no Fourier-of-sequence ever will.
      - "dft_full"      — Re/Im parts of all DFT components, T·d real
                          dims. Unitary equivalent to concat but in a
                          basis where DC is the first block.
      - "dft_no_dc"     — Re/Im parts of F_1..F_{T-1} only (drops the
                          raw_mean component). If this matches raw_mean,
                          half the conversation's information is in DC;
                          if it beats raw_mean the structure is in the
                          higher frequencies.

(C) **Convex Hull Calipers (CHC) — high-dim generalisation.**
    Lift the user's 2D CHC to arbitrary embedding dimension. Replace
    "rotating direction θ" with K random unit directions on S^{d-1}.
    For each direction u_k, record from training benign data:
      - [min_b u_k·X, max_b u_k·X]   — the hull envelope along u_k
      - mean_b, std_b                — distribution centre/spread
    For a test scenario, compute mean(scenario) and project onto each
    u_k:
      - "chc_overshoot"  — overshoot beyond the benign envelope
                           (the user's caliper distance, generalised)
      - "chc_zscore"     — (proj − mean_b) / std_b  per direction
      - "chc_pct"        — empirical CDF position relative to benign
                           projections (rank-based, distribution-free)

    This is a **confinement violation** feature — it asks "did this
    conversation's mean step outside the benign hull along direction
    u_k". Connects directly with Lie Garden's confinement thesis
    (benign hull = the confinement region in embedding space).

We compare to baselines (probe, raw_mean, random_proj_capmatch),
report Δ vs raw_mean with BCa CIs, and seed-permutation-test
hardened from bench_v3.
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
    permutation_test_delta,
    make_scenarios,
    DATASETS,
    _LOGM,
)


# ---------------------------------------------------------------------------
# (F) Sequence DFT features
# ---------------------------------------------------------------------------

def concat_features(vecs: np.ndarray) -> np.ndarray:
    """Flatten the (T, d) matrix to a (T·d,) vector. By construction, a
    classifier on this is at least as expressive as one on raw_mean."""
    return vecs.flatten()


def dft_features(vecs: np.ndarray, drop_dc: bool = False) -> np.ndarray:
    """Real and imaginary parts of the time-axis DFT.

    F[k, j] = Σ_t X_t[j] · exp(-2πikt/T)

    Returns Re(F[k,:]) for k=0 and (Re, Im) for k=1..T-1 stacked.
    For real inputs F[k] = conj(F[T-k]); we keep only k=0..T//2 to avoid
    redundancy.
    """
    T, d = vecs.shape
    F = np.fft.fft(vecs, axis=0)  # (T, d) complex
    # Keep k = 0, 1, ..., T//2 (Hermitian symmetry handles the rest)
    keep_k = T // 2 + 1
    out_blocks = []
    if not drop_dc:
        out_blocks.append(F[0].real)
    for k in range(1, keep_k):
        out_blocks.append(F[k].real)
        out_blocks.append(F[k].imag)
    return np.concatenate(out_blocks)


def dft_per_frequency_feature_fns(T: int) -> dict:
    """Return one feature function per frequency component.

    This lets us measure WHICH frequency carries the discriminative
    signal (the user's hypothesis: high-frequency bursts at injection
    points)."""
    keep_k = T // 2 + 1
    fns = {}
    for k in range(keep_k):
        def make_fn(k_val=k):
            def fn(s):
                T_ = len(s["vecs"])
                F = np.fft.fft(s["vecs"], axis=0)
                if k_val == 0 or k_val * 2 == T_:  # purely real
                    return F[k_val].real
                return np.concatenate([F[k_val].real, F[k_val].imag])
            return fn
        fns[f"dft_freq_{k}"] = make_fn()
    return fns


# ---------------------------------------------------------------------------
# (C) Convex Hull Calipers — high-dim generalisation
# ---------------------------------------------------------------------------

class HighDimCHC:
    """Convex hull caliper profile in R^d, fitted from training benign
    examples.

    For K random unit directions u_1, ..., u_K on S^{d-1}, store the
    benign projection statistics. At test time, compute three feature
    vectors per scenario summarising hull-violation along each direction.

    Generalises the user's 2D rotating-calipers PoC to arbitrary
    embedding dimension by replacing the angle parameterisation with
    a random fixed projection set.
    """

    def __init__(self, n_directions: int = 200, seed: int = 0):
        self.n_directions = n_directions
        self.seed = seed
        self.directions: np.ndarray | None = None
        self.benign_min: np.ndarray | None = None
        self.benign_max: np.ndarray | None = None
        self.benign_mean: np.ndarray | None = None
        self.benign_std: np.ndarray | None = None
        self.benign_sorted: np.ndarray | None = None  # for percentile

    def fit(self, benign_means: np.ndarray) -> "HighDimCHC":
        """`benign_means` is (N_benign, d) — one mean-embedding per
        benign training conversation."""
        N, d = benign_means.shape
        rng = np.random.RandomState(self.seed)
        # Random unit directions on S^{d-1}
        D = rng.randn(d, self.n_directions)
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
        self.directions = D  # (d, K)

        # Project benign means onto each direction
        proj = benign_means @ D  # (N, K)
        self.benign_min = proj.min(axis=0)
        self.benign_max = proj.max(axis=0)
        self.benign_mean = proj.mean(axis=0)
        self.benign_std = proj.std(axis=0) + 1e-12
        # Sorted projections per direction for percentile-rank features
        self.benign_sorted = np.sort(proj, axis=0)
        return self

    def overshoot(self, mean_emb: np.ndarray) -> np.ndarray:
        """Per-direction overshoot beyond benign envelope. 0 means
        in-hull; positive means outside."""
        proj = mean_emb @ self.directions  # (K,)
        below = np.maximum(0.0, self.benign_min - proj)
        above = np.maximum(0.0, proj - self.benign_max)
        return below + above

    def zscore(self, mean_emb: np.ndarray) -> np.ndarray:
        proj = mean_emb @ self.directions
        return (proj - self.benign_mean) / self.benign_std

    def percentile_rank(self, mean_emb: np.ndarray) -> np.ndarray:
        """Empirical CDF of the test projection within the benign sorted
        projections. 0 = below all benign, 1 = above all benign,
        0.5 = median benign. Distribution-free."""
        proj = mean_emb @ self.directions  # (K,)
        K = self.benign_sorted.shape[1]
        ranks = np.empty(K)
        for k in range(K):
            ranks[k] = np.searchsorted(self.benign_sorted[:, k], proj[k]) \
                / max(1, len(self.benign_sorted))
        return ranks


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_dataset_v5(
    ds_info: dict,
    conv_len: int,
    n_per_type: int,
    test_size: float,
    seed: int,
    chc_directions: int,
    do_per_freq: bool,
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

    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, conv_len * emb_dim) / np.sqrt(emb_dim)

    # Build benign training scenarios first; their MEAN embeddings are
    # the points that fit the convex hull / calipers.
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

    # CHC fit: hull defined by the BENIGN training-scenario means.
    benign_train_means = np.array([
        s["vecs"].mean(axis=0) for s in train_scenarios if s["label"] == 0
    ])
    chc = HighDimCHC(n_directions=chc_directions, seed=seed + 41)
    chc.fit(benign_train_means)

    print(f"  [{name}] CHC fitted on {len(benign_train_means)} benign means, "
          f"{chc_directions} directions")

    _LOGM.reset()

    classifiers: dict[str, Callable[[dict], np.ndarray]] = {
        # baselines
        "probe": lambda s: probe_features(s["vecs"], probe_dir),
        "raw_mean": lambda s: raw_mean_features(s["vecs"]),
        "random_proj_capmatch": lambda s: raw_mean_features(s["vecs"]) @ random_proj,
        # (F) Fourier-of-sequence
        "concat": lambda s: concat_features(s["vecs"]),
        "dft_full": lambda s: dft_features(s["vecs"], drop_dc=False),
        "dft_no_dc": lambda s: dft_features(s["vecs"], drop_dc=True),
        # (C) Convex Hull Calipers
        "chc_overshoot": lambda s: chc.overshoot(s["vecs"].mean(axis=0)),
        "chc_zscore": lambda s: chc.zscore(s["vecs"].mean(axis=0)),
        "chc_pct": lambda s: chc.percentile_rank(s["vecs"].mean(axis=0)),
        # (C+) CHC strict-superset: raw_mean ∪ chc
        "raw_mean_plus_chc_overshoot": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            chc.overshoot(s["vecs"].mean(axis=0)),
        ]),
        "raw_mean_plus_chc_pct": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            chc.percentile_rank(s["vecs"].mean(axis=0)),
        ]),
        # (F+) DFT strict-superset: raw_mean ∪ dft_no_dc
        "raw_mean_plus_dft_no_dc": lambda s: np.concatenate([
            raw_mean_features(s["vecs"]),
            dft_features(s["vecs"], drop_dc=True),
        ]),
    }

    # Per-frequency ablation (one feature fn per DFT bin)
    if do_per_freq:
        classifiers.update(dft_per_frequency_feature_fns(conv_len))

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

    # Δ vs raw_mean
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
            "conv_len": conv_len,
            "n_per_type": n_per_type,
            "test_size": test_size,
            "seed": seed,
            "chc_directions": chc_directions,
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
    parser.add_argument("--chc-directions", type=int, default=200)
    parser.add_argument("--do-per-freq", action="store_true",
                        help="Add per-DFT-frequency ablation classifiers")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_v5 — Fourier of the sequence + Convex Hull Calipers (high-D)")
    print(f"  conv_len={args.conv_len}, n_per_type={args.n_per_type}, "
          f"seed={args.seed}, chc_directions={args.chc_directions}")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for name in args.datasets:
        ds_info = DATASETS[name]()
        r = run_dataset_v5(
            ds_info,
            conv_len=args.conv_len,
            n_per_type=args.n_per_type,
            test_size=args.test_size,
            seed=args.seed,
            chc_directions=args.chc_directions,
            do_per_freq=args.do_per_freq,
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
