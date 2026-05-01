#!/usr/bin/env python3
"""
Version-2 evaluation: multi-seed, strict pool-level split, multi-dataset.

Runs the bench_combined experiment for each (dataset, seed) combination
and collects F1 + paired bootstrap deltas into a single JSON. Used as the
factual basis for the "version n2" issue answering the AISI/Jay-Bailey
critique.

Usage::

    python3 scripts/run_v2_evaluation.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_combined import DATASETS, run_dataset


def main() -> int:
    config = {
        "so_n": 25,
        "conv_len": 4,
        "attack_pos": 2,
        "n_per_type": 500,
        "test_size": 0.4,
        "seeds": [1, 7, 13, 42, 99],
        "datasets": ["neuralchemy", "tensortrust", "deepset"],
    }

    print(f"V2 evaluation config: {json.dumps(config, indent=2)}")
    t0 = time.time()
    all_runs = []
    for ds in config["datasets"]:
        b, h = DATASETS[ds]()
        if not b or not h:
            all_runs.append({"dataset": ds, "skipped": True, "reason": "empty pool"})
            continue
        for seed in config["seeds"]:
            print(f"\n{'#' * 78}\n# {ds}  seed={seed}\n{'#' * 78}")
            r = run_dataset(
                ds, b, h,
                so_n=config["so_n"],
                conv_len=config["conv_len"],
                attack_pos=config["attack_pos"],
                n_per_type=config["n_per_type"],
                test_size=config["test_size"],
                seed=seed,
            )
            r["seed"] = seed
            all_runs.append(r)

    # Aggregate per dataset
    summary = {}
    for ds in config["datasets"]:
        ds_runs = [r for r in all_runs if r.get("name") == ds and not r.get("skipped")]
        if not ds_runs:
            summary[ds] = {"skipped": True}
            continue
        probe_f1 = [r["probe"]["f1"] for r in ds_runs]
        holo_f1 = [r["holonomy"]["f1"] for r in ds_runs]
        comb_f1 = [r["combined"]["f1"] for r in ds_runs]
        d_hp = [r["delta_holo_vs_probe"]["mean"] for r in ds_runs]
        d_cp = [r["delta_comb_vs_probe"]["mean"] for r in ds_runs]
        ci_excl_zero_hp = sum(1 for r in ds_runs if r["delta_holo_vs_probe"]["ci_lo"] > 0)
        ci_excl_zero_cp = sum(1 for r in ds_runs if r["delta_comb_vs_probe"]["ci_lo"] > 0)
        summary[ds] = {
            "n_seeds": len(ds_runs),
            "probe_f1_mean": sum(probe_f1) / len(probe_f1),
            "probe_f1_min": min(probe_f1),
            "probe_f1_max": max(probe_f1),
            "holonomy_f1_mean": sum(holo_f1) / len(holo_f1),
            "holonomy_f1_min": min(holo_f1),
            "holonomy_f1_max": max(holo_f1),
            "combined_f1_mean": sum(comb_f1) / len(comb_f1),
            "combined_f1_min": min(comb_f1),
            "combined_f1_max": max(comb_f1),
            "delta_holo_vs_probe_mean": sum(d_hp) / len(d_hp),
            "delta_comb_vs_probe_mean": sum(d_cp) / len(d_cp),
            "seeds_with_holo_ci_excluding_zero": ci_excl_zero_hp,
            "seeds_with_comb_ci_excluding_zero": ci_excl_zero_cp,
        }

    print()
    print("=" * 78)
    print("  V2 EVALUATION SUMMARY")
    print("=" * 78)
    print(f"  {'Dataset':<14} {'Probe F1':>14} {'Holo F1':>14} {'Comb F1':>14} "
          f"{'Δ H-P mean':>11} {'Δ C-P mean':>11} {'sig':>5}")
    print("  " + "-" * 76)
    for ds, s in summary.items():
        if s.get("skipped"):
            print(f"  {ds:<14} (skipped)")
            continue
        sig = f"{s['seeds_with_comb_ci_excluding_zero']}/{s['n_seeds']}"
        probe_str = f"{s['probe_f1_mean']:.3f}±{(s['probe_f1_max']-s['probe_f1_min'])/2:.3f}"
        holo_str = f"{s['holonomy_f1_mean']:.3f}±{(s['holonomy_f1_max']-s['holonomy_f1_min'])/2:.3f}"
        comb_str = f"{s['combined_f1_mean']:.3f}±{(s['combined_f1_max']-s['combined_f1_min'])/2:.3f}"
        print(f"  {ds:<14} {probe_str:>14} {holo_str:>14} {comb_str:>14} "
              f"{s['delta_holo_vs_probe_mean']:>+11.3f} {s['delta_comb_vs_probe_mean']:>+11.3f} {sig:>5}")
    print(f"\n  Total wall time: {time.time() - t0:.1f}s")

    out = {
        "config": config,
        "runs": all_runs,
        "summary": summary,
    }
    out_path = Path(__file__).resolve().parents[1] / "v2_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
