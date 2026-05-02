"""
bench_split_attacks — does holonomy detect attacks SPLIT across messages?

This benchmark tests the regime where multi-step holonomy is *theoretically
supposed to* help: attacks divided across multiple messages so no single
message carries the full attack signal, while the trajectory through them
accumulates evidence the per-step probe cannot see.

Construction:
  - Take TensorTrust attack texts.
  - Split each attack into K chunks at sentence boundaries (or length-based
    chunks if too few sentences).
  - Build a multi-message scenario: a conv_len-message conversation where
    K consecutive positions hold the K attack chunks and the remaining
    (conv_len - K) positions are benign.
  - The benign baseline scenario is conv_len benign messages, no attack.

Hypothesis:
  - probe-MAX over a scenario sees the chunks individually. A short chunk
    (single sentence) carries less attack-direction signal than the full
    attack text. If chunks are below threshold, probe misses.
  - holonomy accumulates the path through ALL K chunks — even if no single
    chunk triggers a per-step alarm, the cumulative geometry should retain
    evidence.

If holonomy beats probe here, the multi-step framing is justified for
*split / camouflaged* attacks specifically. If it doesn't, the multi-step
framing is permanently dropped.

Inherits from bench_v3:
  - text-level dedup before split
  - probe direction trained on full attacks vs benign user messages
  - capacity-matched baselines (probe, raw_mean, random_proj, combined)
  - BCa bootstrap CIs and label-shuffle null control
  - all RNGs parameterised by --seed

Usage::

    python3 bench_split_attacks.py --seed 1 --n-chunks 3 --conv-len 5

For multi-seed sweeps, see scripts/run_split_attacks_evaluation.py.
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

# Reuse the v3 utilities
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
    permutation_test_delta,
    fit_predict,
    _LOGM,
    _dedupe,
)


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------

def split_by_sentence(text: str, n_chunks: int, min_chunk_chars: int = 20) -> list[str]:
    """Split `text` into roughly `n_chunks` chunks, preserving sentence
    boundaries when possible. Falls back to length-based partitioning when
    sentences are too few or too short.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= min_chunk_chars]
    if len(parts) >= n_chunks:
        # Greedy roughly-equal grouping
        target_per_group = len(parts) / n_chunks
        groups: list[list[str]] = [[] for _ in range(n_chunks)]
        for i, p in enumerate(parts):
            g = min(int(i / target_per_group), n_chunks - 1)
            groups[g].append(p)
        chunks = [" ".join(g) for g in groups if g]
        if len(chunks) == n_chunks:
            return chunks
    # Fallback: length-based chunking
    L = len(text)
    if L < n_chunks * min_chunk_chars:
        return []
    cuts = [int(round(L * i / n_chunks)) for i in range(n_chunks + 1)]
    chunks = [text[cuts[i]:cuts[i + 1]].strip() for i in range(n_chunks)]
    return [c for c in chunks if len(c) >= min_chunk_chars]


# ---------------------------------------------------------------------------
# Scenario construction with split attacks
# ---------------------------------------------------------------------------

def make_split_scenarios(
    b_vecs: np.ndarray,
    chunked_attack_vecs: list[list[np.ndarray]],
    n_per_type: int,
    conv_len: int,
    seed: int,
    *,
    consecutive: bool = False,
) -> list[dict]:
    """Build benign and split-attack scenarios.

    benign:     conv_len benign messages drawn from b_vecs.
    attack:     One attack from chunked_attack_vecs is selected; its K chunks
                are placed at K positions in the conversation; the remaining
                conv_len - K positions are filled with benign messages.

    `consecutive=True` puts the K chunks at K consecutive positions; otherwise
    the K positions are sampled uniformly without replacement from
    {0, ..., conv_len-1}.
    """
    rng = np.random.RandomState(seed)
    nb = len(b_vecs)
    if nb == 0 or not chunked_attack_vecs:
        return []
    scenarios = []
    emb_dim = b_vecs.shape[1]

    for _ in range(n_per_type):
        # Benign
        b_idx = rng.choice(nb, conv_len, replace=True)
        scenarios.append({
            "vecs": b_vecs[b_idx],
            "label": 0,
            "n_chunks": 0,
            "chunk_positions": [],
        })

        # Attack: pick a chunked attack
        a_idx = int(rng.randint(0, len(chunked_attack_vecs)))
        chunks = chunked_attack_vecs[a_idx]
        K = min(len(chunks), conv_len)
        # Choose K positions
        if consecutive:
            start = int(rng.randint(0, conv_len - K + 1))
            positions = list(range(start, start + K))
        else:
            positions = sorted(rng.choice(conv_len, K, replace=False).tolist())
        b_idx = rng.choice(nb, conv_len - K, replace=True)
        vecs = np.zeros((conv_len, emb_dim))
        bi = 0
        chunk_iter = iter(range(K))
        for pos in range(conv_len):
            if pos in positions:
                ci = next(chunk_iter)
                vecs[pos] = chunks[ci]
            else:
                vecs[pos] = b_vecs[b_idx[bi]]
                bi += 1
        scenarios.append({
            "vecs": vecs,
            "label": 1,
            "n_chunks": K,
            "chunk_positions": positions,
        })

    return scenarios


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def load_tensortrust_split(n_chunks: int, min_chunk_chars: int = 20,
                            benign_source: str = "natural",
                            attack_source: str = "tensortrust") -> dict:
    """Load attacks from `attack_source` and a benign pool, then split each attack.

    `attack_source`:
      - "tensortrust"   — TT attacks (gibberish-heavy, jailbreak-style)
      - "deepset"       — Deepset injection rows (more natural-language)
      - "neuralchemy"   — Neuralchemy malicious rows (varied attack types)

    `benign_source`:
      - "access_code"   — TensorTrust access_codes (short, distinctive tokens).
                          Trivially separable from attack-chunks by length.
      - "natural"       — natural-language benign messages from Neuralchemy
                          and InjecAgent (richer, harder benign pool).
    """
    attacks: list[str] = []
    if attack_source == "tensortrust":
        p = "/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
        if not os.path.exists(p):
            return {"skipped": True, "reason": f"missing {p}"}
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                atk = d.get("attack", "") or ""
                if atk:
                    attacks.append(atk)
    elif attack_source == "deepset":
        ds_train = "/tmp/deepset_train.parquet"
        ds_test = "/tmp/deepset_test.parquet"
        if not (os.path.exists(ds_train) and os.path.exists(ds_test)):
            return {"skipped": True, "reason": "missing Deepset parquets"}
        import pandas as pd
        for path in (ds_train, ds_test):
            df = pd.read_parquet(path)
            attacks.extend(df.loc[df["label"] == 1, "text"].astype(str).tolist())
    elif attack_source == "neuralchemy":
        nch_train = "/tmp/neuralchemy_train.parquet"
        nch_test = "/tmp/neuralchemy_test.parquet"
        if not (os.path.exists(nch_train) and os.path.exists(nch_test)):
            return {"skipped": True, "reason": "missing Neuralchemy parquets"}
        import pandas as pd
        for path in (nch_train, nch_test):
            df = pd.read_parquet(path)
            attacks.extend(df.loc[df["label"] == 1, "text"].astype(str).tolist())
    else:
        raise ValueError(f"unknown attack_source: {attack_source}")
    attacks = _dedupe(attacks)

    benign: list[str] = []
    if benign_source == "access_code":
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                ac = d.get("access_code", "") or ""
                if ac:
                    benign.append(ac)
    elif benign_source == "natural":
        # Pool natural-language benign messages from multiple sources so the
        # comparison is "real user query vs adversarial chunk", not "short
        # token vs long fragment".
        nch = "/tmp/neuralchemy_train.parquet"
        if os.path.exists(nch):
            try:
                import pandas as pd
                df = pd.read_parquet(nch)
                benign.extend(df.loc[df["label"] == 0, "text"].astype(str).tolist())
            except Exception as e:
                print(f"  WARN: failed to load Neuralchemy benign: {e}")
        ia_uc = "/tmp/InjecAgent/data/user_cases.jsonl"
        if os.path.exists(ia_uc):
            with open(ia_uc) as f:
                for line in f:
                    d = json.loads(line)
                    instr = d.get("User Instruction", "")
                    if instr:
                        benign.append(instr)
        ds_train = "/tmp/deepset_train.parquet"
        if os.path.exists(ds_train):
            try:
                import pandas as pd
                df = pd.read_parquet(ds_train)
                benign.extend(df.loc[df["label"] == 0, "text"].astype(str).tolist())
            except Exception as e:
                print(f"  WARN: failed to load Deepset benign: {e}")
    else:
        raise ValueError(f"unknown benign_source: {benign_source}")

    benign = _dedupe(benign)

    # Split attacks
    chunked = []
    n_skipped = 0
    chunk_lengths = []
    for atk in attacks:
        parts = split_by_sentence(atk, n_chunks, min_chunk_chars=min_chunk_chars)
        if len(parts) < 2:  # need at least 2 chunks for "split" attack
            n_skipped += 1
            continue
        chunked.append(parts)
        chunk_lengths.extend(len(p) for p in parts)

    return {
        "benign_texts": benign,
        "chunked_attacks": chunked,
        "stats": {
            "attack_source": attack_source,
            "benign_source": benign_source,
            "n_benign": len(benign),
            "n_attacks_total": len(attacks),
            "n_attacks_split": len(chunked),
            "n_attacks_skipped": n_skipped,
            "chunk_length_min": min(chunk_lengths) if chunk_lengths else 0,
            "chunk_length_median": int(np.median(chunk_lengths)) if chunk_lengths else 0,
            "chunk_length_max": max(chunk_lengths) if chunk_lengths else 0,
            "n_chunks_target": n_chunks,
        },
    }


def run_split_attack_experiment(
    n_chunks: int = 3,
    conv_len: int = 5,
    n_per_type: int = 500,
    test_size: float = 0.4,
    seed: int = 1,
    so_n: int = 25,
    holonomy_scale: float = 0.1,
    consecutive: bool = False,
    benign_source: str = "natural",
    attack_source: str = "tensortrust",
) -> dict:
    print(f"\n  loading {attack_source} attacks split into {n_chunks} chunks "
          f"(benign source: {benign_source})...")
    data = load_tensortrust_split(n_chunks=n_chunks, benign_source=benign_source,
                                   attack_source=attack_source)
    if data.get("skipped"):
        return data
    stats = data["stats"]
    print(f"  benign: {stats['n_benign']} unique user messages")
    print(f"  attacks: {stats['n_attacks_total']} total, "
          f"{stats['n_attacks_split']} successfully split (skipped {stats['n_attacks_skipped']})")
    print(f"  chunk lengths: min={stats['chunk_length_min']}, "
          f"median={stats['chunk_length_median']}, max={stats['chunk_length_max']}")

    benign_texts = data["benign_texts"]
    chunked_attacks = data["chunked_attacks"]

    # Split-aware pool split: BENIGN texts split into train/test halves;
    # ATTACKS split into train/test halves. Probe direction is fit on a
    # disjoint sub-pool of TRAIN-side texts only.
    split_rng = np.random.RandomState(seed)
    n_b = len(benign_texts)
    n_a = len(chunked_attacks)

    if n_b < 80 or n_a < 60:
        return {"skipped": True,
                "reason": f"pool too small ({n_b} benign, {n_a} chunked attacks)"}

    b_perm = split_rng.permutation(n_b)
    a_perm = split_rng.permutation(n_a)
    b_split = int((1.0 - test_size) * n_b)
    a_split = int((1.0 - test_size) * n_a)
    train_benign = [benign_texts[i] for i in b_perm[:b_split]]
    test_benign = [benign_texts[i] for i in b_perm[b_split:]]
    train_attacks_chunks = [chunked_attacks[i] for i in a_perm[:a_split]]
    test_attacks_chunks = [chunked_attacks[i] for i in a_perm[a_split:]]
    print(f"  pool split: train {len(train_benign)} benign + {len(train_attacks_chunks)} attacks; "
          f"test {len(test_benign)} benign + {len(test_attacks_chunks)} attacks")

    # Probe-direction sub-pool from train side
    probe_rng = np.random.RandomState(seed + 13)
    n_b_probe = max(30, len(train_benign) // 4)
    n_a_probe = max(30, len(train_attacks_chunks) // 4)
    n_b_probe = min(n_b_probe, len(train_benign) - 30)
    n_a_probe = min(n_a_probe, len(train_attacks_chunks) - 30)
    if n_b_probe < 30 or n_a_probe < 30:
        return {"skipped": True, "reason": "probe pool too small"}

    b_p = probe_rng.permutation(len(train_benign))
    a_p = probe_rng.permutation(len(train_attacks_chunks))
    probe_benign_texts = [train_benign[i] for i in b_p[:n_b_probe]]
    probe_attack_texts_full = [
        " ".join(train_attacks_chunks[i])  # use the full reassembled attack text
        for i in a_p[:n_a_probe]
    ]
    train_benign_remaining = [train_benign[i] for i in b_p[n_b_probe:]]
    train_attacks_remaining = [train_attacks_chunks[i] for i in a_p[n_a_probe:]]

    print(f"  encoding probe pool ({len(probe_benign_texts)+len(probe_attack_texts_full)}) "
          f"+ train pool ({len(train_benign_remaining)}+{sum(len(c) for c in train_attacks_remaining)} chunks) "
          f"+ test pool ({len(test_benign)}+{sum(len(c) for c in test_attacks_chunks)} chunks)")
    t0 = time.time()
    probe_b_vecs = encode(probe_benign_texts)
    probe_a_vecs = encode(probe_attack_texts_full)
    train_b_vecs = encode(train_benign_remaining)
    test_b_vecs = encode(test_benign)
    # Encode all chunks with a single call per pool for efficiency
    train_attack_chunk_lists = train_attacks_remaining
    test_attack_chunk_lists = test_attacks_chunks
    flat_train_chunks = [c for lst in train_attack_chunk_lists for c in lst]
    flat_test_chunks = [c for lst in test_attack_chunk_lists for c in lst]
    flat_train_chunk_vecs = encode(flat_train_chunks) if flat_train_chunks else np.zeros((0, 512))
    flat_test_chunk_vecs = encode(flat_test_chunks) if flat_test_chunks else np.zeros((0, 512))
    enc_t = time.time() - t0
    emb_dim = train_b_vecs.shape[1]
    print(f"  encoded in {enc_t:.1f}s, emb_dim={emb_dim}")

    # Reassemble per-attack chunk lists
    def reassemble(chunk_lists, flat_vecs):
        out = []
        i = 0
        for lst in chunk_lists:
            k = len(lst)
            out.append([flat_vecs[i + j] for j in range(k)])
            i += k
        return out
    train_chunked_attack_vecs = reassemble(train_attack_chunk_lists, flat_train_chunk_vecs)
    test_chunked_attack_vecs = reassemble(test_attack_chunk_lists, flat_test_chunk_vecs)

    # Probe direction from full attack vs benign user messages
    diff = probe_a_vecs.mean(0) - probe_b_vecs.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)

    # Capacity-matched random projection
    n_holo = so_n * (so_n - 1) // 2
    proj_rng = np.random.RandomState(seed + 31)
    random_proj = proj_rng.randn(emb_dim, n_holo) / np.sqrt(emb_dim)

    n_train_per_type = int(n_per_type * (1.0 - test_size))
    n_test_per_type = n_per_type - n_train_per_type

    train_scenarios = make_split_scenarios(
        train_b_vecs, train_chunked_attack_vecs,
        n_per_type=n_train_per_type, conv_len=conv_len, seed=seed,
        consecutive=consecutive,
    )
    test_scenarios = make_split_scenarios(
        test_b_vecs, test_chunked_attack_vecs,
        n_per_type=n_test_per_type, conv_len=conv_len, seed=seed + 57,
        consecutive=consecutive,
    )
    print(f"  scenarios: {len(train_scenarios)} train, {len(test_scenarios)} test "
          f"(conv_len {conv_len}, K up to {n_chunks}, "
          f"placement={'consecutive' if consecutive else 'scattered'})")

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
            assert np.array_equal(truth_arr, truth)

    deltas = {}
    perm_tests = {}
    for chal in ("holonomy", "combined"):
        for base in ("probe", "raw_mean", "random_proj"):
            key = f"{chal}_vs_{base}"
            deltas[key] = bca_bootstrap_delta(
                preds_by_clf[base], preds_by_clf[chal], truth_arr,
                n_iter=2000, seed=seed,
            )
            perm_tests[key] = permutation_test_delta(
                preds_by_clf[base], preds_by_clf[chal], truth_arr,
                n_iter=2000, seed=seed,
            )

    print(f"\n  RESULTS  (split-attack, conv_len={conv_len}, K={n_chunks}, "
          f"consecutive={consecutive}, seed={seed})")
    for clf_name, r in results.items():
        print(f"    {clf_name:<13} F1={r['f1']:.3f}  P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  ({r['n_features']}d)")
    for key, d in deltas.items():
        sig = " *" if d["ci_lo"] > 0 else ("  " if d["ci_hi"] > 0 else " -")
        print(f"    Δ {key:<28} = {d['obs_delta']:+.3f}  "
              f"CI95 [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]{sig}")

    return {
        "config": {
            "n_chunks": n_chunks,
            "conv_len": conv_len,
            "n_per_type": n_per_type,
            "test_size": test_size,
            "seed": seed,
            "so_n": so_n,
            "holonomy_scale": holonomy_scale,
            "consecutive": consecutive,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_revision": EMBEDDING_MODEL_REVISION,
        },
        "stats": stats,
        "pool_sizes": {
            "train_benign": len(train_benign_remaining),
            "train_attacks": len(train_attacks_remaining),
            "test_benign": len(test_benign),
            "test_attacks": len(test_attacks_chunks),
            "probe_benign": n_b_probe,
            "probe_attacks_full": n_a_probe,
        },
        "classifiers": results,
        "deltas": deltas,
        "permutation_tests": perm_tests,
        "logm_fallbacks": {
            "total": _LOGM.total,
            "count": _LOGM.fallbacks,
            "per_class": {
                "benign_total": _LOGM.per_class_total[0],
                "benign_fallbacks": _LOGM.per_class_fallback[0],
                "harmful_total": _LOGM.per_class_total[1],
                "harmful_fallbacks": _LOGM.per_class_fallback[1],
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, default=3,
                        help="Target number of chunks per attack")
    parser.add_argument("--conv-len", type=int, default=5,
                        help="Total messages in the multi-step scenario")
    parser.add_argument("--n-per-type", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--so-n", type=int, default=25)
    parser.add_argument("--holonomy-scale", type=float, default=0.1)
    parser.add_argument("--consecutive", action="store_true",
                        help="Place chunks at consecutive positions "
                             "(instead of scattered)")
    parser.add_argument("--benign-source", choices=["natural", "access_code"],
                        default="natural",
                        help="Source of benign messages: 'natural' "
                             "(Neuralchemy/InjecAgent/Deepset benign rows) "
                             "or 'access_code' (TT access_codes, short tokens)")
    parser.add_argument("--attack-source",
                        choices=["tensortrust", "deepset", "neuralchemy"],
                        default="tensortrust",
                        help="Dataset to draw attacks from")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  bench_split_attacks — split / camouflaged attack benchmark")
    print(f"  n_chunks={args.n_chunks}, conv_len={args.conv_len}, "
          f"placement={'consecutive' if args.consecutive else 'scattered'}, "
          f"seed={args.seed}")
    print("=" * 78)

    t0 = time.time()
    r = run_split_attack_experiment(
        n_chunks=args.n_chunks,
        conv_len=args.conv_len,
        n_per_type=args.n_per_type,
        test_size=args.test_size,
        seed=args.seed,
        so_n=args.so_n,
        holonomy_scale=args.holonomy_scale,
        consecutive=args.consecutive,
        benign_source=args.benign_source,
        attack_source=args.attack_source,
    )
    print(f"\n  wall time: {time.time() - t0:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, default=str))
        print(f"  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
