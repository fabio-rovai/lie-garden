"""
bench_permutation — does the SAME multi-set of messages in different ORDERS
                    produce different detection outcomes?

Motivation: bench_v3 and bench_split_attacks showed that on multi-message
prompt-injection scenarios, holonomy does not consistently beat a 512-d
raw mean embedding baseline. The reason is that raw_mean has plenty of
capacity to capture the bag-of-features signal — order is irrelevant.

This benchmark constructs the regime where raw_mean *cannot* win by
construction: scenarios where benign and attack contain exactly the same
set of messages, only the ORDER differs. raw_mean is permutation-
invariant — its features are identical across reorderings, so it cannot
discriminate by definition. Holonomy, being path-ordered, can.

Construction (3 classes, all using the same M attack chunks + benign cover):
  Class 0 (canonical attack):  chunks at positions [p_1, ..., p_M]
                               in their original sentence order
  Class 1 (shuffled):          same chunks, but permuted to a *different*
                               canonical order (deterministic mapping)
  Class 2 (benign):            no attack chunks, all benign messages

We treat (class 0, class 2) as the train labels and ask:
  - When evaluated on (class 1, class 2), does holonomy generalise its
    "attack" decision boundary to the shuffled version, or treat it as
    benign?
  - For raw_mean, by construction class 0 and class 1 have IDENTICAL
    features, so its decision must be identical to the train decision.
    F1 is therefore the same as on the canonical test set — but that's
    *not* a virtue: it means the model has no notion of order.

The interesting metric is "shuffle-detection rate":
  fraction of class-1 (shuffled) scenarios scored DIFFERENTLY from their
  canonical class-0 counterparts.

Caveats baked in by design:
  - This is a *capability* benchmark, not a deployment-realistic F1
    benchmark. We are measuring "is the classifier order-aware at all".
  - A classifier that says "all permutations of attack chunks are
    attacks" is operationally correct for prompt-injection detection
    (because attack chunks are still attack-like). The benchmark
    therefore reports BOTH F1 (operational metric) AND
    "shuffle-distinguishability" (capability metric).

Usage:
  python3 bench_permutation.py --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    bca_bootstrap_delta,
    fit_predict,
    _LOGM,
    _dedupe,
)
from bench_split_attacks import split_by_sentence  # noqa: E402


def _load_attacks_chunked(n_chunks: int = 3, min_chunk_chars: int = 20) -> list[list[str]]:
    """Return TensorTrust attacks split into >= 2 sentence chunks each."""
    p = "/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    if not os.path.exists(p):
        return []
    attacks = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            atk = d.get("attack", "") or ""
            if atk:
                attacks.append(atk)
    attacks = _dedupe(attacks)
    chunked = []
    for atk in attacks:
        parts = split_by_sentence(atk, n_chunks, min_chunk_chars=min_chunk_chars)
        if len(parts) >= 2:
            chunked.append(parts)
    return chunked


def _load_benign_natural() -> list[str]:
    out: list[str] = []
    nch = "/tmp/neuralchemy_train.parquet"
    if os.path.exists(nch):
        import pandas as pd
        df = pd.read_parquet(nch)
        out.extend(df.loc[df["label"] == 0, "text"].astype(str).tolist())
    ds = "/tmp/deepset_train.parquet"
    if os.path.exists(ds):
        import pandas as pd
        df = pd.read_parquet(ds)
        out.extend(df.loc[df["label"] == 0, "text"].astype(str).tolist())
    return _dedupe(out)


def _shuffle_perm(n: int, rng: np.random.RandomState) -> list[int]:
    """Return a permutation of [0..n) that is NOT the identity."""
    while True:
        perm = list(rng.permutation(n))
        if perm != list(range(n)):
            return perm


def make_perm_scenarios(
    benign_vecs: np.ndarray,
    chunked_attack_vecs: list[list[np.ndarray]],
    n_per_type: int,
    conv_len: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return three lists of scenarios:
        - canonical: attack chunks in original order
        - shuffled : same chunks, non-identity permutation
        - benign   : no attack chunks
    """
    rng = np.random.RandomState(seed)
    nb = len(benign_vecs)
    if nb == 0 or not chunked_attack_vecs:
        return [], [], []
    canonical: list[dict] = []
    shuffled: list[dict] = []
    benign: list[dict] = []
    emb_dim = benign_vecs.shape[1]

    for _ in range(n_per_type):
        # Pick an attack with at least 2 chunks
        a_idx = int(rng.randint(0, len(chunked_attack_vecs)))
        chunks = chunked_attack_vecs[a_idx]
        K = min(len(chunks), conv_len)
        positions = sorted(rng.choice(conv_len, K, replace=False).tolist())
        b_idx = rng.choice(nb, conv_len - K, replace=True)

        # canonical
        vecs_c = np.zeros((conv_len, emb_dim))
        bi = 0
        for i, pos in enumerate(range(conv_len)):
            if pos in positions:
                ci = positions.index(pos)
                vecs_c[pos] = chunks[ci]
            else:
                vecs_c[pos] = benign_vecs[b_idx[bi]]
                bi += 1
        canonical.append({"vecs": vecs_c, "label": 1, "kind": "canonical"})

        # shuffled: same chunks, different order at the same positions
        vecs_s = np.zeros((conv_len, emb_dim))
        perm = _shuffle_perm(K, rng)
        bi = 0
        for pos in range(conv_len):
            if pos in positions:
                ci_orig = positions.index(pos)
                vecs_s[pos] = chunks[perm[ci_orig]]
            else:
                vecs_s[pos] = benign_vecs[b_idx[bi]]
                bi += 1
        shuffled.append({"vecs": vecs_s, "label": 1, "kind": "shuffled"})

        # benign
        b_only_idx = rng.choice(nb, conv_len, replace=True)
        benign.append({
            "vecs": benign_vecs[b_only_idx],
            "label": 0,
            "kind": "benign",
        })

    return canonical, shuffled, benign


def _verify_permutation_invariance(canonical: list[dict],
                                   shuffled: list[dict],
                                   feature_fn: Callable[[dict], np.ndarray],
                                   tol: float = 1e-9) -> bool:
    """Check whether feature_fn produces identical features for canonical
    and shuffled scenarios. raw_mean MUST satisfy this (mean is permutation-
    invariant); holonomy MUST NOT (path matters)."""
    diffs = []
    for c, s in zip(canonical[: min(20, len(canonical))], shuffled):
        fc = feature_fn(c)
        fs = feature_fn(s)
        diffs.append(float(np.max(np.abs(fc - fs))))
    return max(diffs) < tol


def run_permutation_experiment(
    n_chunks: int = 3,
    conv_len: int = 5,
    n_per_type: int = 400,
    test_size: float = 0.4,
    seed: int = 1,
    so_n: int = 25,
    holonomy_scale: float = 0.1,
) -> dict:
    print(f"\n  loading TensorTrust attacks split into {n_chunks} chunks...")
    chunked = _load_attacks_chunked(n_chunks=n_chunks)
    if not chunked:
        return {"skipped": True, "reason": "TensorTrust unavailable"}
    benign_texts = _load_benign_natural()
    if not benign_texts:
        return {"skipped": True, "reason": "no natural benign source"}
    print(f"  benign pool: {len(benign_texts)}, chunked attacks: {len(chunked)}")

    rng = np.random.RandomState(seed)
    bp = rng.permutation(len(benign_texts))
    ap = rng.permutation(len(chunked))
    b_split = int((1.0 - test_size) * len(benign_texts))
    a_split = int((1.0 - test_size) * len(chunked))
    train_benign = [benign_texts[i] for i in bp[:b_split]]
    test_benign = [benign_texts[i] for i in bp[b_split:]]
    train_chunked = [chunked[i] for i in ap[:a_split]]
    test_chunked = [chunked[i] for i in ap[a_split:]]
    print(f"  pool split: train benign {len(train_benign)} + chunked attacks {len(train_chunked)}; "
          f"test benign {len(test_benign)} + chunked attacks {len(test_chunked)}")

    if min(len(train_benign), len(train_chunked), len(test_benign), len(test_chunked)) < 30:
        return {"skipped": True, "reason": "pool too small after split"}

    # Probe-direction sub-pool from the train side only.
    prng = np.random.RandomState(seed + 13)
    n_b_probe = max(30, len(train_benign) // 4)
    n_a_probe = max(30, len(train_chunked) // 4)
    n_b_probe = min(n_b_probe, len(train_benign) - 30)
    n_a_probe = min(n_a_probe, len(train_chunked) - 30)
    bperm = prng.permutation(len(train_benign))
    aperm = prng.permutation(len(train_chunked))
    probe_benign_texts = [train_benign[i] for i in bperm[:n_b_probe]]
    probe_attack_texts = [
        " ".join(train_chunked[i]) for i in aperm[:n_a_probe]
    ]
    train_benign_remaining = [train_benign[i] for i in bperm[n_b_probe:]]
    train_chunked_remaining = [train_chunked[i] for i in aperm[n_a_probe:]]

    print(f"  encoding probe pool ({len(probe_benign_texts)+len(probe_attack_texts)})")
    t0 = time.time()
    probe_b = encode(probe_benign_texts)
    probe_a = encode(probe_attack_texts)
    train_b_v = encode(train_benign_remaining)
    test_b_v = encode(test_benign)

    flat_train = [c for lst in train_chunked_remaining for c in lst]
    flat_test = [c for lst in test_chunked for c in lst]
    flat_train_v = encode(flat_train) if flat_train else np.zeros((0, 512))
    flat_test_v = encode(flat_test) if flat_test else np.zeros((0, 512))
    enc_t = time.time() - t0
    emb_dim = train_b_v.shape[1]
    print(f"  encoded in {enc_t:.1f}s, emb_dim={emb_dim}")

    def reassemble(lst, flat):
        out, i = [], 0
        for sub in lst:
            out.append([flat[i + j] for j in range(len(sub))])
            i += len(sub)
        return out

    train_attack_v = reassemble(train_chunked_remaining, flat_train_v)
    test_attack_v = reassemble(test_chunked, flat_test_v)

    # Probe direction
    diff = probe_a.mean(0) - probe_b.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    # Random projection
    n_holo = so_n * (so_n - 1) // 2
    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, n_holo) / np.sqrt(emb_dim)

    n_train_per = int(n_per_type * (1.0 - test_size))
    n_test_per = n_per_type - n_train_per

    train_can, train_shuf, train_ben = make_perm_scenarios(
        train_b_v, train_attack_v,
        n_per_type=n_train_per, conv_len=conv_len, seed=seed,
    )
    test_can, test_shuf, test_ben = make_perm_scenarios(
        test_b_v, test_attack_v,
        n_per_type=n_test_per, conv_len=conv_len, seed=seed + 57,
    )
    print(f"  scenarios: train (canonical {len(train_can)}, shuffled {len(train_shuf)}, "
          f"benign {len(train_ben)}); test (canonical {len(test_can)}, shuffled {len(test_shuf)}, "
          f"benign {len(test_ben)})")

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

    classifiers = {
        "probe": f_probe,
        "holonomy": f_holonomy,
        "raw_mean": f_raw_mean,
        "random_proj": f_random_proj,
    }

    # Sanity-check: raw_mean MUST be permutation-invariant; holonomy MUST NOT.
    invariance = {}
    for name, fn in classifiers.items():
        invariance[name] = _verify_permutation_invariance(test_can, test_shuf, fn)
    print(f"  permutation invariance check (canonical vs shuffled features identical?):")
    for name, inv in invariance.items():
        marker = "INV" if inv else "VAR"
        print(f"    {name:<13} {marker}")
    assert invariance["raw_mean"], (
        "raw_mean MUST be permutation-invariant; if this fails the experiment is broken"
    )
    assert not invariance["holonomy"], (
        "holonomy MUST be permutation-variant; if this fails check vec_to_skew"
    )

    # Train on (canonical, benign) — these are the labels we have ground truth for.
    train_scenarios = train_can + train_ben
    rng2 = np.random.RandomState(seed + 71)
    rng2.shuffle(train_scenarios)

    # Test on each of the three groupings:
    #   A. canonical vs benign  (operational F1; same setting as bench_v3)
    #   B. shuffled  vs benign  (does the model treat shuffled chunks as attacks?)
    #   C. canonical vs shuffled  (does the model distinguish canonical from shuffled?)
    def evaluate_set(name: str, test_set: list[dict]) -> dict:
        out = {"name": name, "n": len(test_set)}
        labels = np.array([s["label"] for s in test_set])
        preds_by_clf = {}
        truth = None
        for clf_name, fn in classifiers.items():
            res, preds, t = fit_predict(train_scenarios, test_set, fn)
            out[clf_name] = {
                "f1": res.f1, "precision": res.precision, "recall": res.recall,
                "n_features": res.n_features,
            }
            preds_by_clf[clf_name] = preds
            if truth is None:
                truth = t
        # Pairwise deltas
        out["deltas"] = {}
        for chal in ("holonomy",):
            for base in ("probe", "raw_mean", "random_proj"):
                key = f"{chal}_vs_{base}"
                out["deltas"][key] = bca_bootstrap_delta(
                    preds_by_clf[base], preds_by_clf[chal], truth,
                    n_iter=2000, seed=seed,
                )
        return out

    set_A = evaluate_set("canonical_vs_benign", test_can + test_ben)
    set_B = evaluate_set(
        "shuffled_vs_benign",
        [{**s, "label": 1} for s in test_shuf] + test_ben,
    )

    # Set C: how often does the model assign DIFFERENT labels to a
    # canonical scenario and its matched shuffled twin?
    same_clf_changes = {}
    for clf_name, fn in classifiers.items():
        # Train classifier on (canonical, benign) and predict on canonical+shuffled
        res_can, preds_can, _ = fit_predict(train_scenarios, test_can, fn)
        res_shuf, preds_shuf, _ = fit_predict(train_scenarios, test_shuf, fn)
        diffs = int(np.sum(preds_can != preds_shuf))
        same_clf_changes[clf_name] = {
            "n_pairs": len(preds_can),
            "n_predictions_changed": diffs,
            "fraction_changed": diffs / max(1, len(preds_can)),
            "canonical_recall_attack": float(np.mean(preds_can == 1)),
            "shuffled_recall_attack": float(np.mean(preds_shuf == 1)),
        }

    print(f"\n  RESULTS")
    print(f"  --- Set A: canonical vs benign (operational) ---")
    for clf_name in classifiers:
        r = set_A[clf_name]
        print(f"    {clf_name:<13} F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}")
    print(f"  --- Set B: shuffled vs benign (does model still flag shuffled?) ---")
    for clf_name in classifiers:
        r = set_B[clf_name]
        print(f"    {clf_name:<13} F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}")
    print(f"  --- Set C: canonical vs shuffled distinguishability ---")
    print(f"  {'classifier':<13} {'pred-changed':>14} {'pred-attack on can':>20} {'pred-attack on shuf':>22}")
    for clf_name in classifiers:
        r = same_clf_changes[clf_name]
        print(f"    {clf_name:<13} {r['fraction_changed']:>14.1%} "
              f"{r['canonical_recall_attack']:>20.1%} "
              f"{r['shuffled_recall_attack']:>22.1%}")
    if same_clf_changes["raw_mean"]["fraction_changed"] > 1e-9:
        print(f"    !! raw_mean changed predictions across permutations — implementation bug")
    print(f"\n  KEY FINDING:")
    holo_changes = same_clf_changes["holonomy"]["fraction_changed"]
    rm_changes = same_clf_changes["raw_mean"]["fraction_changed"]
    if holo_changes > rm_changes:
        print(f"    Holonomy changes its prediction on {holo_changes:.1%} of canonical→shuffled pairs;")
        print(f"    raw_mean changes on {rm_changes:.1%} (must be 0 by construction).")
        print(f"    => Holonomy uses order information; raw_mean cannot.")
    else:
        print(f"    Holonomy is also permutation-stable in practice ({holo_changes:.1%} changes).")
        print(f"    => The order signal is small enough that the trained logistic ignores it.")

    return {
        "config": {
            "n_chunks": n_chunks,
            "conv_len": conv_len,
            "n_per_type": n_per_type,
            "test_size": test_size,
            "seed": seed,
            "so_n": so_n,
            "holonomy_scale": holonomy_scale,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
        },
        "invariance_check": invariance,
        "set_A_canonical_vs_benign": set_A,
        "set_B_shuffled_vs_benign": set_B,
        "set_C_canonical_vs_shuffled_distinguishability": same_clf_changes,
        "logm_fallbacks": {
            "total": _LOGM.total,
            "fallbacks": _LOGM.fallbacks,
            "rate": _LOGM.fallbacks / max(1, _LOGM.total),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, default=3)
    parser.add_argument("--conv-len", type=int, default=5)
    parser.add_argument("--n-per-type", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--so-n", type=int, default=25)
    parser.add_argument("--holonomy-scale", type=float, default=0.1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_permutation — order-sensitivity benchmark")
    print(f"  n_chunks={args.n_chunks}, conv_len={args.conv_len}, seed={args.seed}")
    print("=" * 78)

    t0 = time.time()
    r = run_permutation_experiment(
        n_chunks=args.n_chunks,
        conv_len=args.conv_len,
        n_per_type=args.n_per_type,
        test_size=args.test_size,
        seed=args.seed,
        so_n=args.so_n,
        holonomy_scale=args.holonomy_scale,
    )
    print(f"\n  wall: {time.time() - t0:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, default=str))
        print(f"  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
