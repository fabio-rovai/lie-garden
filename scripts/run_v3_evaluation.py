#!/usr/bin/env python3
"""
Run bench_v3 across N seeds, aggregate, apply Benjamini-Hochberg FDR.

Outputs `v3_results.json` with per-(dataset, seed, comparison) deltas plus
an aggregated summary that gives, for each dataset:

  * mean Δ across seeds for each (challenger vs baseline) pair
  * fraction of seeds whose BCa CI excludes zero (raw)
  * permutation-test p-values (raw)
  * Benjamini-Hochberg q-values across the full comparison family
  * a single hierarchical-bootstrap CI on the across-seed mean

Run via:

  python3 scripts/run_v3_evaluation.py --n-seeds 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_v3 import DATASETS, run_dataset


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return q-values for an array of raw p-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals_sorted = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0.0, 1.0)
    qvals = np.empty(n)
    qvals[order] = qvals_sorted
    return qvals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--so-n", type=int, default=25)
    parser.add_argument("--conv-len", type=int, default=4)
    parser.add_argument("--n-per-type", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--datasets", nargs="*",
                        default=["tensortrust", "neuralchemy", "deepset"])
    parser.add_argument("--out", type=str, default="v3_results.json")
    args = parser.parse_args()

    seeds = list(range(1, args.n_seeds + 1))
    config = {
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "so_n": args.so_n,
        "conv_len": args.conv_len,
        "n_per_type": args.n_per_type,
        "test_size": args.test_size,
        "datasets": args.datasets,
    }
    print(f"V3 evaluation config: {json.dumps(config, indent=2)}\n")

    t0 = time.time()
    runs = []
    ds_info_cache = {}
    for ds_name in args.datasets:
        ds_info_cache[ds_name] = DATASETS[ds_name]()
        if ds_info_cache[ds_name].get("skipped"):
            print(f"\n=== {ds_name}: SKIPPED — {ds_info_cache[ds_name].get('reason')}")
            continue
        for seed in seeds:
            print(f"\n{'#' * 78}\n# {ds_name}  seed={seed}\n{'#' * 78}")
            r = run_dataset(
                ds_info_cache[ds_name],
                so_n=args.so_n,
                conv_len=args.conv_len,
                n_per_type=args.n_per_type,
                test_size=args.test_size,
                seed=seed,
                do_permutation_test=True,
            )
            r["seed"] = seed
            runs.append(r)

    # Aggregate per dataset × comparison
    summary = {}
    all_pvals = []  # for global BH correction
    all_pval_keys = []
    for ds in args.datasets:
        ds_runs = [r for r in runs if r.get("name") == ds and not r.get("skipped")]
        if not ds_runs:
            summary[ds] = {"skipped": True}
            continue
        comp_keys = list(ds_runs[0]["deltas"].keys())
        ds_summary = {
            "n_seeds": len(ds_runs),
            "f1_mean": {},
            "f1_seed_range": {},
            "comparisons": {},
            "pool_sizes": ds_runs[0]["pool_sizes"],
            "raw_count": ds_runs[0].get("raw_count"),
            "notes": ds_runs[0].get("notes"),
        }
        # F1 means per classifier
        for clf_name in ds_runs[0]["classifiers"].keys():
            f1s = [r["classifiers"][clf_name]["f1"] for r in ds_runs]
            ds_summary["f1_mean"][clf_name] = float(np.mean(f1s))
            ds_summary["f1_seed_range"][clf_name] = [float(min(f1s)), float(max(f1s))]
        # Per-comparison aggregates
        for ck in comp_keys:
            obs_deltas = [r["deltas"][ck]["obs_delta"] for r in ds_runs]
            ci_excludes_zero = sum(1 for r in ds_runs if r["deltas"][ck]["ci_lo"] > 0)
            perm_pvals = [r["permutation_tests"][ck]["p_two_sided"] for r in ds_runs]
            mean_delta = float(np.mean(obs_deltas))
            sd_delta = float(np.std(obs_deltas, ddof=1)) if len(obs_deltas) > 1 else 0.0
            # Across-seed t-style CI on the mean delta
            from scipy.stats import t as student_t
            t_crit = student_t.ppf(0.975, df=max(1, len(obs_deltas) - 1))
            half = t_crit * sd_delta / np.sqrt(len(obs_deltas)) if len(obs_deltas) > 1 else 0.0
            mean_ci = [mean_delta - half, mean_delta + half]
            # Combine permutation p-values using Fisher's method
            chi2_stat = -2.0 * float(np.sum(np.log(np.clip(perm_pvals, 1e-12, 1.0))))
            from scipy.stats import chi2 as chi2_dist
            fisher_p = float(1.0 - chi2_dist.cdf(chi2_stat, df=2 * len(perm_pvals)))
            ds_summary["comparisons"][ck] = {
                "obs_delta_per_seed": obs_deltas,
                "obs_delta_mean": mean_delta,
                "obs_delta_sd": sd_delta,
                "obs_delta_mean_ci95": mean_ci,
                "n_seeds_ci_excludes_zero": ci_excludes_zero,
                "perm_p_per_seed": perm_pvals,
                "fisher_combined_p": fisher_p,
            }
            all_pvals.append(fisher_p)
            all_pval_keys.append(f"{ds}::{ck}")
        summary[ds] = ds_summary

    # Benjamini-Hochberg FDR across the comparison family (datasets × deltas)
    qvals = benjamini_hochberg(np.array(all_pvals)) if all_pvals else np.array([])
    fdr_table = {}
    for key, p, q in zip(all_pval_keys, all_pvals, qvals):
        ds, ck = key.split("::")
        fdr_table.setdefault(ds, {})[ck] = {
            "fisher_p": float(p),
            "bh_q": float(q),
        }

    # Print human summary
    print()
    print("=" * 90)
    print("  V3 SUMMARY (mean Δ across seeds, BCa & permutation, BH-FDR adjusted)")
    print("=" * 90)
    print(f"  {'dataset':<14} {'comparison':<26} {'mean Δ':>9} {'95% CI':>22} "
          f"{'sig/N':>7} {'BH q':>9}")
    print("  " + "-" * 85)
    for ds in args.datasets:
        if summary[ds].get("skipped"):
            print(f"  {ds:<14} (skipped)")
            continue
        for ck, comp in summary[ds]["comparisons"].items():
            mean = comp["obs_delta_mean"]
            lo, hi = comp["obs_delta_mean_ci95"]
            sig = f"{comp['n_seeds_ci_excludes_zero']}/{summary[ds]['n_seeds']}"
            q = fdr_table[ds][ck]["bh_q"]
            star = " *" if q < 0.05 and mean > 0 else ""
            print(f"  {ds:<14} {ck:<26} {mean:>+9.4f} "
                  f"[{lo:>+8.4f},{hi:>+8.4f}] {sig:>7} {q:>9.4f}{star}")
    print("\n  '*' marks comparisons that survive Benjamini-Hochberg FDR at q<0.05")
    print(f"\n  Total wall: {time.time() - t0:.1f}s")

    out = {
        "config": config,
        "runs": runs,
        "summary": summary,
        "fdr_table": fdr_table,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
