"""
Holonomy HONEST evaluation: No data leakage, no misleading claims
=================================================================

Fixes EVERY methodological issue:

1. SAMPLE-LEVEL SPLIT: train/test split on underlying texts, not scenarios.
   No embedding vector appears in both train and test.

2. NO FAKE ERASURE: removed -h_vec trick (never occurs in real text).
   Recovery scenarios use only real benign messages after attack.

3. HONEST BASELINES: compare same-capacity classifiers on same data.

4. STATISTICAL TESTS: report confidence intervals, not just point estimates.

The question we're honestly answering:
  "Does running embeddings through SO(n) → expm → product → logm
   produce features that a classifier can use BETTER than features
   extracted directly from the raw embeddings?"
"""
import json
import hashlib
import sys
import time

import numpy as np
from scipy.linalg import expm, logm
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_MODEL = None

def encode(texts: list[str]) -> np.ndarray:
    global _MODEL
    try:
        from model2vec import StaticModel
        if _MODEL is None:
            _MODEL = StaticModel.from_pretrained("minishlab/potion-base-32M")
        return np.array(_MODEL.encode(texts))
    except ImportError:
        vecs = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array([int.from_bytes(h[i:i+2], "little") / 65535.0
                           for i in range(0, min(len(h), 512), 2)])
            if len(vec) < 256:
                vec = np.resize(vec, 256)
            vecs.append(vec)
        return np.array(vecs)


# ---------------------------------------------------------------------------
# Lie algebra ops
# ---------------------------------------------------------------------------

def vec_to_skew(coeffs, n):
    A = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(coeffs):
                A[i, j] = coeffs[idx]
                A[j, i] = -coeffs[idx]
            idx += 1
    return A


def skew_to_vec(A):
    n = A.shape[0]
    return np.array([A[i, j] for i in range(n) for j in range(i + 1, n)])


_LOGM_FALLBACKS = {"count": 0, "total": 0}


def safe_logm(R):
    _LOGM_FALLBACKS["total"] += 1
    try:
        L = logm(R)
        return np.real((L - L.T) / 2.0)
    except Exception as e:
        _LOGM_FALLBACKS["count"] += 1
        # Returning zeros here makes feature extraction continue, but the
        # zero vector is indistinguishable from a benign step. Track and
        # warn so a run dominated by fallbacks isn't reported as a
        # successful experiment.
        if _LOGM_FALLBACKS["count"] <= 5 or _LOGM_FALLBACKS["count"] % 100 == 0:
            print(
                f"WARN: safe_logm fallback #{_LOGM_FALLBACKS['count']}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        return np.zeros_like(R)


def assert_logm_fallbacks_acceptable(threshold: float = 0.05):
    total = _LOGM_FALLBACKS["total"]
    fallbacks = _LOGM_FALLBACKS["count"]
    if total == 0:
        return
    rate = fallbacks / total
    print(
        f"\nlogm fallbacks: {fallbacks}/{total} ({rate:.1%})",
        file=sys.stderr,
    )
    if rate > threshold:
        raise RuntimeError(
            f"safe_logm hit the zero-fallback path on {rate:.1%} of calls "
            f"(>{threshold:.0%}); features for those steps are zero vectors "
            "and any reported F1 is unreliable. Investigate before quoting "
            "these numbers."
        )


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def holonomy_features(vecs, n=10, scale=0.1):
    """
    Group-based features: embed sequence into SO(n), extract from trajectory.
    """
    n_coeffs = n * (n - 1) // 2
    I = np.eye(n)

    state = I.copy()
    skews = []
    scar_norms = []
    step_norms = []

    for v in vecs:
        coeffs = v[:n_coeffs] * scale
        A = vec_to_skew(coeffs, n)
        skews.append(A)
        step_norms.append(np.linalg.norm(A, "fro"))
        state = state @ expm(A)
        scar_norms.append(np.linalg.norm(state - I, "fro"))

    scar_norms = np.array(scar_norms)
    step_norms = np.array(step_norms)

    # Final state: log-matrix direction
    log_h = safe_logm(state)
    log_coeffs = skew_to_vec(log_h)

    # Eigenvalue spectrum
    eigvals = np.sort(np.abs(np.linalg.eigvals(log_h)))[::-1]
    eig_mags = eigvals.real[:n // 2]
    if len(eig_mags) < n // 2:
        eig_mags = np.pad(eig_mags, (0, n // 2 - len(eig_mags)))

    # Commutator norms
    comm_norms = []
    for i in range(len(skews) - 1):
        C = skews[i] @ skews[i+1] - skews[i+1] @ skews[i]
        comm_norms.append(np.linalg.norm(C, "fro"))

    # Trajectory statistics
    max_scar = scar_norms.max() if len(scar_norms) > 0 else 0.0
    final_scar = scar_norms[-1] if len(scar_norms) > 0 else 0.0
    scar_mean = scar_norms.mean() if len(scar_norms) > 0 else 0.0

    if len(scar_norms) > 1:
        scar_diffs = np.diff(scar_norms)
        max_jump = scar_diffs.max()
        max_drop = scar_diffs.min()
        scar_vol = scar_diffs.std()
    else:
        max_jump = max_drop = scar_vol = 0.0

    features = np.concatenate([
        log_coeffs,  # direction of final rotation
        eig_mags,    # rotation plane magnitudes
        [final_scar, np.trace(state), max_scar, scar_mean],
        [max_jump, max_drop, scar_vol],
        [sum(comm_norms), max(comm_norms) if comm_norms else 0, np.mean(comm_norms) if comm_norms else 0],
        [sum(step_norms), max(step_norms) if len(step_norms) > 0 else 0],
    ])
    return features


def raw_embedding_features(vecs, n_target):
    """
    Non-group baseline: sequence-level statistics from raw embeddings.
    Matched to same dimensionality as holonomy features.
    """
    mean_v = vecs.mean(axis=0)
    max_v = vecs.max(axis=0)
    std_v = vecs.std(axis=0)
    min_v = vecs.min(axis=0)

    # Pairwise similarities
    cos_sims = []
    l2_dists = []
    for i in range(len(vecs) - 1):
        n1 = np.linalg.norm(vecs[i])
        n2 = np.linalg.norm(vecs[i+1])
        cos_sims.append(np.dot(vecs[i], vecs[i+1]) / (n1 * n2 + 1e-12))
        l2_dists.append(np.linalg.norm(vecs[i] - vecs[i+1]))

    if cos_sims:
        seq_stats = [
            np.mean(cos_sims), np.std(cos_sims), np.min(cos_sims), np.max(cos_sims),
            np.mean(l2_dists), np.std(l2_dists), np.min(l2_dists), np.max(l2_dists),
        ]
    else:
        seq_stats = [0] * 8

    # Norms
    norms = np.linalg.norm(vecs, axis=1)
    norm_stats = [norms.mean(), norms.std(), norms.max(), norms.min()]

    # Build to target dimensionality
    per_dim = max(1, (n_target - 12) // 4)
    raw = np.concatenate([
        mean_v[:per_dim],
        max_v[:per_dim],
        std_v[:per_dim],
        min_v[:per_dim],
        seq_stats,
        norm_stats,
    ])
    if len(raw) < n_target:
        raw = np.pad(raw, (0, n_target - len(raw)))
    return raw[:n_target]


# ---------------------------------------------------------------------------
# Scenario generator (from pre-split vectors only)
# ---------------------------------------------------------------------------

def make_scenarios(b_vecs, h_vecs, n_scenarios=200, seed=42):
    """Build scenarios using ONLY the provided vectors (no leakage)."""
    rng = np.random.RandomState(seed)
    scenarios = []
    n_b, n_h = len(b_vecs), len(h_vecs)

    for i in range(n_scenarios):
        t = i % 4
        if t == 0:  # all benign
            idxs = rng.choice(n_b, 8, replace=True)
            vecs = b_vecs[idxs]
            label = 0
            desc = "benign"
        elif t == 1:  # single attack in benign context
            b_idx = rng.choice(n_b, 6, replace=True)
            h_idx = rng.choice(n_h)
            pos = rng.randint(2, 5)
            vecs = np.vstack([b_vecs[b_idx[:pos]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[pos:]]])
            label = 1
            desc = "single"
        elif t == 2:  # attack + long recovery (7 benign after)
            b_idx = rng.choice(n_b, 9, replace=True)
            h_idx = rng.choice(n_h)
            vecs = np.vstack([b_vecs[b_idx[:2]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[2:]]])
            label = 1
            desc = "recover"
        else:  # escalation (2 attacks)
            b_idx = rng.choice(n_b, 5, replace=True)
            h_idx = rng.choice(n_h, min(2, n_h), replace=False)
            if len(h_idx) < 2:
                h_idx = np.array([h_idx[0], h_idx[0]])
            vecs = np.vstack([
                b_vecs[b_idx[:2]], h_vecs[h_idx[0]:h_idx[0]+1],
                b_vecs[b_idx[2:4]], h_vecs[h_idx[1]:h_idx[1]+1],
                b_vecs[b_idx[4:5]]
            ])
            label = 1
            desc = "escalate"
        scenarios.append({"vecs": vecs, "label": label, "type": desc})
    return scenarios


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_results(labels, preds):
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
    }


def eval_by_type(scenarios, preds):
    results = {}
    for t in ["benign", "single", "recover", "escalate"]:
        sub_idx = [i for i, s in enumerate(scenarios) if s["type"] == t]
        if not sub_idx:
            continue
        correct = sum(1 for i in sub_idx if preds[i] == (scenarios[i]["label"] == 1))
        results[t] = f"{correct}/{len(sub_idx)}"
    return results


# ---------------------------------------------------------------------------
# Main: sample-level split, honest evaluation
# ---------------------------------------------------------------------------

def run_experiment(name, b_vecs, h_vecs, n_b_texts, n_h_texts):
    """
    HONEST evaluation with sample-level train/test split.
    No embedding vector appears in both train and test.
    """
    print(f"\n{'='*70}")
    print(f"  {name}: {len(b_vecs)} benign, {len(h_vecs)} harmful")
    print(f"{'='*70}")

    # SAMPLE-LEVEL SPLIT (the key fix)
    rng = np.random.RandomState(42)
    n_b, n_h = len(b_vecs), len(h_vecs)

    b_perm = rng.permutation(n_b)
    h_perm = rng.permutation(n_h)

    b_split = int(0.6 * n_b)
    h_split = int(0.6 * n_h)

    b_train, b_test = b_vecs[b_perm[:b_split]], b_vecs[b_perm[b_split:]]
    h_train, h_test = h_vecs[h_perm[:h_split]], h_vecs[h_perm[h_split:]]

    print(f"  Split: train={len(b_train)}b/{len(h_train)}h, test={len(b_test)}b/{len(h_test)}h")
    print(f"  NO embedding vector shared between train and test")

    # Build scenarios from each split
    train_scenarios = make_scenarios(b_train, h_train, 400, seed=42)
    test_scenarios = make_scenarios(b_test, h_test, 400, seed=99)

    test_labels = np.array([s["label"] for s in test_scenarios])
    train_labels = np.array([s["label"] for s in train_scenarios])

    # ===================================================================
    # Baseline: Probe-only (trained on train split)
    # ===================================================================
    diff = h_train.mean(0) - b_train.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)
    probe_thresh = (b_train @ probe_dir).mean() + (h_train @ probe_dir).mean()
    probe_thresh /= 2.0

    def probe_detect(vecs):
        scores = vecs @ probe_dir
        return bool(scores.max() > probe_thresh)

    probe_preds = np.array([int(probe_detect(s["vecs"])) for s in test_scenarios])
    probe_r = eval_results(test_labels, probe_preds)
    probe_bt = eval_by_type(test_scenarios, [bool(p) for p in probe_preds])
    print(f"\n  [Baseline] Probe-only:  F1={probe_r['f1']:.3f}  P={probe_r['prec']:.3f}  R={probe_r['rec']:.3f}")
    print(f"    By type: {probe_bt}")

    # ===================================================================
    # Test multiple group dimensions
    # ===================================================================
    results_table = []

    for n_group in [6, 10, 16]:
        n_coeffs = n_group * (n_group - 1) // 2
        if n_coeffs > b_vecs.shape[1]:
            continue

        for scale in [0.1]:
            # Holonomy features
            h_train_f = np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scenarios])
            h_test_f = np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios])
            h_train_f = np.nan_to_num(h_train_f, nan=0.0, posinf=1e6, neginf=-1e6)
            h_test_f = np.nan_to_num(h_test_f, nan=0.0, posinf=1e6, neginf=-1e6)

            n_feats = h_train_f.shape[1]

            # Raw embedding features (same dimensionality)
            r_train_f = np.array([raw_embedding_features(s["vecs"], n_feats) for s in train_scenarios])
            r_test_f = np.array([raw_embedding_features(s["vecs"], n_feats) for s in test_scenarios])
            r_train_f = np.nan_to_num(r_train_f, nan=0.0, posinf=1e6, neginf=-1e6)
            r_test_f = np.nan_to_num(r_test_f, nan=0.0, posinf=1e6, neginf=-1e6)

            for feat_name, f_train, f_test in [
                (f"Holonomy SO({n_group})", h_train_f, h_test_f),
                (f"Raw emb ({n_feats}d)", r_train_f, r_test_f),
            ]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(f_train)
                X_test = scaler.transform(f_test)

                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X_train, train_labels)
                preds = clf.predict(X_test)

                r = eval_results(test_labels, preds)
                bt = eval_by_type(test_scenarios, [bool(p) for p in preds])
                delta = r["f1"] - probe_r["f1"]
                marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")

                print(f"\n  [{feat_name}] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
                print(f"    By type: {bt}")
                results_table.append((feat_name, r, bt))

            # Group structure lift for this n_group
            holo_r = results_table[-2][1]
            raw_r = results_table[-1][1]
            lift = holo_r["f1"] - raw_r["f1"]
            print(f"\n  >> SO({n_group}) group lift over raw: {lift:+.3f} F1")

    # ===================================================================
    # Paired bootstrap confidence interval for the best holonomy
    # ===================================================================
    print(f"\n  --- Statistical significance (bootstrap) ---")
    n_group = 10
    scale = 0.1
    n_coeffs = n_group * (n_group - 1) // 2

    if n_coeffs <= b_vecs.shape[1]:
        h_test_f = np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios])
        h_test_f = np.nan_to_num(h_test_f, nan=0.0, posinf=1e6, neginf=-1e6)
        h_train_f = np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scenarios])
        h_train_f = np.nan_to_num(h_train_f, nan=0.0, posinf=1e6, neginf=-1e6)

        n_feats = h_test_f.shape[1]
        r_test_f = np.array([raw_embedding_features(s["vecs"], n_feats) for s in test_scenarios])
        r_test_f = np.nan_to_num(r_test_f, nan=0.0, posinf=1e6, neginf=-1e6)
        r_train_f = np.array([raw_embedding_features(s["vecs"], n_feats) for s in train_scenarios])
        r_train_f = np.nan_to_num(r_train_f, nan=0.0, posinf=1e6, neginf=-1e6)

        # Train both classifiers
        sc_h = StandardScaler()
        clf_h = LogisticRegression(max_iter=1000, C=1.0)
        clf_h.fit(sc_h.fit_transform(h_train_f), train_labels)
        holo_preds = clf_h.predict(sc_h.transform(h_test_f))

        sc_r = StandardScaler()
        clf_r = LogisticRegression(max_iter=1000, C=1.0)
        clf_r.fit(sc_r.fit_transform(r_train_f), train_labels)
        raw_preds = clf_r.predict(sc_r.transform(r_test_f))

        # Paired bootstrap
        n_boot = 1000
        rng = np.random.RandomState(42)
        deltas = []
        for _ in range(n_boot):
            idx = rng.choice(len(test_labels), len(test_labels), replace=True)
            h_f1 = f1_score(test_labels[idx], holo_preds[idx], zero_division=0)
            r_f1 = f1_score(test_labels[idx], raw_preds[idx], zero_division=0)
            deltas.append(h_f1 - r_f1)

        deltas = np.array(deltas)
        ci_low = np.percentile(deltas, 2.5)
        ci_high = np.percentile(deltas, 97.5)
        mean_delta = deltas.mean()
        p_positive = (deltas > 0).mean()

        print(f"  Holonomy vs Raw embeddings (SO(10), {n_feats}d each):")
        print(f"    Mean Δ F1: {mean_delta:+.3f}")
        print(f"    95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
        print(f"    P(holonomy > raw): {p_positive:.3f}")

        if ci_low > 0:
            print(f"    >> STATISTICALLY SIGNIFICANT: holonomy beats raw embeddings")
        elif ci_high < 0:
            print(f"    >> STATISTICALLY SIGNIFICANT: raw embeddings beat holonomy")
        else:
            print(f"    >> NOT SIGNIFICANT: confidence interval crosses zero")

        # Also: holonomy vs probe
        deltas_probe = []
        for _ in range(n_boot):
            idx = rng.choice(len(test_labels), len(test_labels), replace=True)
            h_f1 = f1_score(test_labels[idx], holo_preds[idx], zero_division=0)
            p_f1 = f1_score(test_labels[idx], probe_preds[idx], zero_division=0)
            deltas_probe.append(h_f1 - p_f1)

        deltas_probe = np.array(deltas_probe)
        ci_low_p = np.percentile(deltas_probe, 2.5)
        ci_high_p = np.percentile(deltas_probe, 97.5)

        print(f"\n  Holonomy vs Probe-only:")
        print(f"    Mean Δ F1: {deltas_probe.mean():+.3f}")
        print(f"    95% CI: [{ci_low_p:+.3f}, {ci_high_p:+.3f}]")
        if ci_low_p > 0:
            print(f"    >> STATISTICALLY SIGNIFICANT: holonomy beats probe")
        else:
            print(f"    >> NOT SIGNIFICANT: confidence interval crosses zero")

    # ===================================================================
    # Per-type analysis: where specifically does holonomy help?
    # ===================================================================
    print(f"\n  --- Per-type: Probe vs Holonomy SO(10) ---")
    if n_coeffs <= b_vecs.shape[1]:
        for stype in ["benign", "single", "recover", "escalate"]:
            t_idx = [i for i, s in enumerate(test_scenarios) if s["type"] == stype]
            if not t_idx:
                continue
            p_correct = sum(1 for i in t_idx if probe_preds[i] == test_scenarios[i]["label"])
            h_correct = sum(1 for i in t_idx if holo_preds[i] == test_scenarios[i]["label"])
            total = len(t_idx)
            diff_n = h_correct - p_correct
            marker = f"+{diff_n}" if diff_n > 0 else str(diff_n)
            print(f"    {stype:<12} probe={p_correct}/{total}  holonomy={h_correct}/{total}  ({marker})")

    # ===================================================================
    # Cross-validation (additional honest check)
    # ===================================================================
    print(f"\n  --- 5-fold CV on test scenarios only ---")
    if n_coeffs <= b_vecs.shape[1]:
        # Recompute features for ALL test scenarios
        all_feats_h = np.nan_to_num(
            np.array([holonomy_features(s["vecs"], 10, 0.1) for s in test_scenarios]),
            nan=0.0, posinf=1e6, neginf=-1e6
        )
        n_f = all_feats_h.shape[1]
        all_feats_r = np.nan_to_num(
            np.array([raw_embedding_features(s["vecs"], n_f) for s in test_scenarios]),
            nan=0.0, posinf=1e6, neginf=-1e6
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for feat_name, feats in [("Holonomy SO(10)", all_feats_h), ("Raw embeddings", all_feats_r)]:
            fold_f1s = []
            for tr_idx, te_idx in kf.split(feats, test_labels):
                sc = StandardScaler()
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(sc.fit_transform(feats[tr_idx]), test_labels[tr_idx])
                preds = clf.predict(sc.transform(feats[te_idx]))
                fold_f1s.append(f1_score(test_labels[te_idx], preds, zero_division=0))
            mean_f = np.mean(fold_f1s)
            std_f = np.std(fold_f1s)
            print(f"    [{feat_name}] F1={mean_f:.3f}±{std_f:.3f}")

        # Note about data leakage in CV
        print(f"    NOTE: CV on test scenarios still shares embedding vectors across folds")
        print(f"    The sample-level split above is the primary honest evaluation")


def main():
    print("=" * 70)
    print("HOLONOMY HONEST: Sample-level split, no fake erasure, stat tests")
    print("=" * 70)

    t0 = time.time()

    # Load InjecAgent
    benign_inj, harmful_inj = [], []
    with open("/tmp/InjecAgent/data/user_cases.jsonl") as f:
        for line in f:
            benign_inj.append(json.loads(line)["User Instruction"])
    for fn in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
        with open(f"/tmp/InjecAgent/data/{fn}") as f:
            for line in f:
                harmful_inj.append(json.loads(line)["Attacker Instruction"])
    for fn in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(f"/tmp/InjecAgent/data/{fn}") as f:
            for c in json.load(f):
                harmful_inj.append(c["Attacker Instruction"])

    # Load TensorTrust
    tt_b, tt_h = [], []
    with open("/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tt_b.append(d["pre_prompt"])
            tt_h.append(d["attack"])

    for dataset_name, b_texts, h_texts in [
        ("InjecAgent", benign_inj, harmful_inj),
        ("TensorTrust", tt_b, tt_h),
    ]:
        print(f"\n  Encoding {dataset_name}...")
        all_vecs = encode(b_texts + h_texts)
        b_vecs = all_vecs[:len(b_texts)]
        h_vecs = all_vecs[len(b_texts):]

        run_experiment(dataset_name, b_vecs, h_vecs, len(b_texts), len(h_texts))

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("WHAT WE CAN HONESTLY CLAIM (if results support it):")
    print("  - Holonomy features in SO(10)+ beat raw embedding features")
    print("    with statistical significance (bootstrap CI > 0)")
    print("  - The group structure provides non-trivial sequence composition")
    print("  - This holds under sample-level train/test split (no leakage)")
    print("")
    print("WHAT WE CANNOT CLAIM:")
    print("  - Non-erasability (monotonic invariants don't discriminate)")
    print("  - Multi-step persistence (marginal improvement, not significant)")
    print("  - Perfect detection (only achievable with data leakage)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
    assert_logm_fallbacks_acceptable()
