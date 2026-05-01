"""
Holonomy FINAL: Definitive honest evaluation
=============================================

Larger datasets, sample-level splits, statistical tests, and an honest
test of "signal persistence" (does holonomy retain attack signal through
recovery better than raw embeddings?).

Point 3 reframe: not "non-erasable" (mathematically false), but
"persistent" — the non-linear group product retains attack information
through benign recovery messages better than linear composition.

Testable prediction: as recovery length increases, holonomy detection
degrades SLOWER than raw embedding detection.
"""
import json
import hashlib
import time
import os

import numpy as np
from scipy.linalg import expm, logm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
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
# Lie algebra
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

def safe_logm(R):
    try:
        L = logm(R)
        return np.real((L - L.T) / 2.0)
    except Exception:
        return np.zeros_like(R)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def holonomy_features(vecs, n=10, scale=0.1):
    """Group-based trajectory features."""
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

    log_h = safe_logm(state)
    log_coeffs = skew_to_vec(log_h)

    eigvals = np.sort(np.abs(np.linalg.eigvals(log_h)))[::-1]
    eig_mags = eigvals.real[:n // 2]
    if len(eig_mags) < n // 2:
        eig_mags = np.pad(eig_mags, (0, n // 2 - len(eig_mags)))

    comm_norms = []
    for i in range(len(skews) - 1):
        C = skews[i] @ skews[i+1] - skews[i+1] @ skews[i]
        comm_norms.append(np.linalg.norm(C, "fro"))

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

    return np.concatenate([
        log_coeffs, eig_mags,
        [final_scar, np.trace(state), max_scar, scar_mean],
        [max_jump, max_drop, scar_vol],
        [sum(comm_norms), max(comm_norms) if comm_norms else 0,
         np.mean(comm_norms) if comm_norms else 0],
        [sum(step_norms), max(step_norms) if len(step_norms) > 0 else 0],
    ])


def raw_embedding_features(vecs, n_target):
    """Non-group baseline: sequence statistics from raw embeddings."""
    mean_v = vecs.mean(axis=0)
    max_v = vecs.max(axis=0)
    std_v = vecs.std(axis=0)
    min_v = vecs.min(axis=0)

    cos_sims, l2_dists = [], []
    for i in range(len(vecs) - 1):
        n1, n2 = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[i+1])
        cos_sims.append(np.dot(vecs[i], vecs[i+1]) / (n1 * n2 + 1e-12))
        l2_dists.append(np.linalg.norm(vecs[i] - vecs[i+1]))

    if cos_sims:
        seq_stats = [np.mean(cos_sims), np.std(cos_sims), np.min(cos_sims), np.max(cos_sims),
                     np.mean(l2_dists), np.std(l2_dists), np.min(l2_dists), np.max(l2_dists)]
    else:
        seq_stats = [0] * 8

    norms = np.linalg.norm(vecs, axis=1)
    norm_stats = [norms.mean(), norms.std(), norms.max(), norms.min()]

    per_dim = max(1, (n_target - 12) // 4)
    raw = np.concatenate([mean_v[:per_dim], max_v[:per_dim], std_v[:per_dim],
                          min_v[:per_dim], seq_stats, norm_stats])
    if len(raw) < n_target:
        raw = np.pad(raw, (0, n_target - len(raw)))
    return raw[:n_target]


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def make_scenarios(b_vecs, h_vecs, n_per_type=100, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    n_b, n_h = len(b_vecs), len(h_vecs)

    for i in range(n_per_type):
        # Benign
        idxs = rng.choice(n_b, 8, replace=True)
        scenarios.append({"vecs": b_vecs[idxs], "label": 0, "type": "benign"})

        # Single attack
        b_idx = rng.choice(n_b, 6, replace=True)
        h_idx = rng.choice(n_h)
        pos = rng.randint(2, 5)
        vecs = np.vstack([b_vecs[b_idx[:pos]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[pos:]]])
        scenarios.append({"vecs": vecs, "label": 1, "type": "single"})

        # Attack + recovery
        b_idx = rng.choice(n_b, 9, replace=True)
        h_idx = rng.choice(n_h)
        vecs = np.vstack([b_vecs[b_idx[:2]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[2:]]])
        scenarios.append({"vecs": vecs, "label": 1, "type": "recover"})

        # Escalation
        b_idx = rng.choice(n_b, 5, replace=True)
        h_idx = rng.choice(n_h, min(2, n_h), replace=n_h < 2)
        if len(h_idx) < 2:
            h_idx = np.array([h_idx[0], h_idx[0]])
        vecs = np.vstack([b_vecs[b_idx[:2]], h_vecs[h_idx[0]:h_idx[0]+1],
                          b_vecs[b_idx[2:4]], h_vecs[h_idx[1]:h_idx[1]+1],
                          b_vecs[b_idx[4:5]]])
        scenarios.append({"vecs": vecs, "label": 1, "type": "escalate"})

    return scenarios


# ---------------------------------------------------------------------------
# Signal persistence test (the honest point 3)
# ---------------------------------------------------------------------------

def signal_persistence_test(b_vecs, h_vecs, n_group=10, scale=0.1):
    """
    THE HONEST NON-ERASABILITY TEST:

    As we add more benign recovery messages after an attack,
    does holonomy retain the attack signal better than raw embeddings?

    Protocol:
    - Fix an attack at position 2 in the conversation
    - Vary recovery length: 0, 2, 4, 6, 8, 10 benign messages after
    - Train classifier for each recovery length
    - Measure: detection rate as function of recovery length
    - Compare: holonomy vs raw embeddings
    - If holonomy degrades SLOWER, the group structure provides persistence
    """
    print(f"\n  --- SIGNAL PERSISTENCE TEST (honest point 3) ---")
    print(f"  Protocol: attack at position 2, vary benign recovery length")
    print(f"  If holonomy degrades slower → group structure persists signal\n")

    rng = np.random.RandomState(42)
    n_b, n_h = len(b_vecs), len(h_vecs)

    # Split within this test
    b_split = int(0.6 * n_b)
    h_split = int(0.6 * n_h)
    b_perm = rng.permutation(n_b)
    h_perm = rng.permutation(n_h)
    b_tr, b_te = b_vecs[b_perm[:b_split]], b_vecs[b_perm[b_split:]]
    h_tr, h_te = h_vecs[h_perm[:h_split]], h_vecs[h_perm[h_split:]]

    recovery_lengths = [0, 2, 4, 6, 8, 10]

    print(f"  {'Recovery':<10} {'Holonomy F1':>12} {'Raw Emb F1':>12} {'Probe F1':>10} {'Holo-Raw':>10} {'Holo-Probe':>10}")
    print(f"  {'-'*60}")

    holo_f1s = []
    raw_f1s = []
    probe_f1s = []

    # Train probe on train split
    diff = h_tr.mean(0) - b_tr.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)
    probe_thresh = (b_tr @ probe_dir).mean() + (h_tr @ probe_dir).mean()
    probe_thresh /= 2.0

    for rec_len in recovery_lengths:
        # Generate scenarios with this recovery length
        def gen_scenarios(b_v, h_v, n_per_type=100, seed=42):
            rng2 = np.random.RandomState(seed)
            scenarios = []
            nb, nh = len(b_v), len(h_v)
            prefix_len = 2
            total_len = prefix_len + 1 + rec_len  # prefix + attack + recovery

            for _ in range(n_per_type):
                # Benign: same total length
                b_idx = rng2.choice(nb, total_len, replace=True)
                scenarios.append({"vecs": b_v[b_idx], "label": 0, "type": "benign"})

                # Attack at position 2 + recovery
                b_idx = rng2.choice(nb, prefix_len + rec_len, replace=True)
                h_idx = rng2.choice(nh)
                vecs = np.vstack([
                    b_v[b_idx[:prefix_len]],
                    h_v[h_idx:h_idx+1],
                    b_v[b_idx[prefix_len:prefix_len + rec_len]] if rec_len > 0 else np.empty((0, b_v.shape[1]))
                ])
                scenarios.append({"vecs": vecs, "label": 1, "type": "attack"})

            return scenarios

        train_scen = gen_scenarios(b_tr, h_tr, 150, seed=42)
        test_scen = gen_scenarios(b_te, h_te, 150, seed=99)

        train_labels = np.array([s["label"] for s in train_scen])
        test_labels = np.array([s["label"] for s in test_scen])

        # Compute holonomy features
        n_coeffs = n_group * (n_group - 1) // 2
        h_tr_f = np.nan_to_num(np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scen]),
                               nan=0.0, posinf=1e6, neginf=-1e6)
        h_te_f = np.nan_to_num(np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scen]),
                               nan=0.0, posinf=1e6, neginf=-1e6)

        n_f = h_tr_f.shape[1]

        # Raw embedding features
        r_tr_f = np.nan_to_num(np.array([raw_embedding_features(s["vecs"], n_f) for s in train_scen]),
                               nan=0.0, posinf=1e6, neginf=-1e6)
        r_te_f = np.nan_to_num(np.array([raw_embedding_features(s["vecs"], n_f) for s in test_scen]),
                               nan=0.0, posinf=1e6, neginf=-1e6)

        # Train and eval holonomy
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.fit_transform(h_tr_f), train_labels)
        h_preds = clf.predict(sc.transform(h_te_f))
        h_f1 = f1_score(test_labels, h_preds, zero_division=0)

        # Train and eval raw
        sc2 = StandardScaler()
        clf2 = LogisticRegression(max_iter=1000, C=1.0)
        clf2.fit(sc2.fit_transform(r_tr_f), train_labels)
        r_preds = clf2.predict(sc2.transform(r_te_f))
        r_f1 = f1_score(test_labels, r_preds, zero_division=0)

        # Probe baseline
        def probe_detect(vecs):
            scores = vecs @ probe_dir
            return bool(scores.max() > probe_thresh)
        p_preds = np.array([int(probe_detect(s["vecs"])) for s in test_scen])
        p_f1 = f1_score(test_labels, p_preds, zero_division=0)

        holo_f1s.append(h_f1)
        raw_f1s.append(r_f1)
        probe_f1s.append(p_f1)

        print(f"  {rec_len:<10} {h_f1:>12.3f} {r_f1:>12.3f} {p_f1:>10.3f} {h_f1-r_f1:>+10.3f} {h_f1-p_f1:>+10.3f}")

    # Compute degradation rate
    holo_f1s = np.array(holo_f1s)
    raw_f1s = np.array(raw_f1s)
    probe_f1s = np.array(probe_f1s)

    # Linear regression of F1 vs recovery length
    from numpy.polynomial.polynomial import polyfit
    rec = np.array(recovery_lengths, dtype=float)

    holo_slope = polyfit(rec, holo_f1s, 1)[1]
    raw_slope = polyfit(rec, raw_f1s, 1)[1]
    probe_slope = polyfit(rec, probe_f1s, 1)[1]

    print(f"\n  Degradation rate (F1 per recovery message):")
    print(f"    Holonomy:  {holo_slope:+.4f}/msg")
    print(f"    Raw emb:   {raw_slope:+.4f}/msg")
    print(f"    Probe:     {probe_slope:+.4f}/msg")

    if holo_slope > raw_slope:  # less negative = slower degradation
        print(f"\n  >> HOLONOMY DEGRADES SLOWER ({holo_slope:+.4f} vs {raw_slope:+.4f})")
        print(f"  >> Group structure provides signal persistence!")
        return True
    else:
        print(f"\n  >> Raw embeddings degrade slower or equal")
        print(f"  >> No evidence of group-based signal persistence")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(name, b_vecs, h_vecs):
    print(f"\n{'='*70}")
    print(f"  {name}: {len(b_vecs)} benign, {len(h_vecs)} harmful")
    print(f"{'='*70}")

    # Sample-level split
    rng = np.random.RandomState(42)
    n_b, n_h = len(b_vecs), len(h_vecs)
    b_perm = rng.permutation(n_b)
    h_perm = rng.permutation(n_h)
    b_split = int(0.6 * n_b)
    h_split = int(0.6 * n_h)
    b_train, b_test = b_vecs[b_perm[:b_split]], b_vecs[b_perm[b_split:]]
    h_train, h_test = h_vecs[h_perm[:h_split]], h_vecs[h_perm[h_split:]]

    print(f"  Split: train={len(b_train)}b/{len(h_train)}h, test={len(b_test)}b/{len(h_test)}h")

    # Build scenarios
    train_scenarios = make_scenarios(b_train, h_train, 150, seed=42)
    test_scenarios = make_scenarios(b_test, h_test, 150, seed=99)
    train_labels = np.array([s["label"] for s in train_scenarios])
    test_labels = np.array([s["label"] for s in test_scenarios])

    # Probe baseline
    diff = h_train.mean(0) - b_train.mean(0)
    probe_dir = diff / (np.linalg.norm(diff) + 1e-12)
    probe_thresh = (b_train @ probe_dir).mean() + (h_train @ probe_dir).mean()
    probe_thresh /= 2.0

    def probe_detect(vecs):
        return bool((vecs @ probe_dir).max() > probe_thresh)

    probe_preds = np.array([int(probe_detect(s["vecs"])) for s in test_scenarios])
    probe_r = {
        "f1": f1_score(test_labels, probe_preds, zero_division=0),
        "prec": precision_score(test_labels, probe_preds, zero_division=0),
        "rec": recall_score(test_labels, probe_preds, zero_division=0),
    }
    print(f"\n  [Probe-only] F1={probe_r['f1']:.3f}  P={probe_r['prec']:.3f}  R={probe_r['rec']:.3f}")
    bt = {}
    for t in ["benign", "single", "recover", "escalate"]:
        idx = [i for i, s in enumerate(test_scenarios) if s["type"] == t]
        if idx:
            correct = sum(1 for i in idx if probe_preds[i] == test_scenarios[i]["label"])
            bt[t] = f"{correct}/{len(idx)}"
    print(f"    By type: {bt}")

    # Test holonomy and raw for SO(10)
    n_group = 10
    scale = 0.1

    h_tr_f = np.nan_to_num(np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scenarios]),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    h_te_f = np.nan_to_num(np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios]),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    n_f = h_tr_f.shape[1]

    r_tr_f = np.nan_to_num(np.array([raw_embedding_features(s["vecs"], n_f) for s in train_scenarios]),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    r_te_f = np.nan_to_num(np.array([raw_embedding_features(s["vecs"], n_f) for s in test_scenarios]),
                           nan=0.0, posinf=1e6, neginf=-1e6)

    results = {}
    for feat_name, f_tr, f_te in [
        (f"Holonomy SO({n_group})", h_tr_f, h_te_f),
        (f"Raw emb ({n_f}d)", r_tr_f, r_te_f),
    ]:
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.fit_transform(f_tr), train_labels)
        preds = clf.predict(sc.transform(f_te))

        r = {
            "f1": f1_score(test_labels, preds, zero_division=0),
            "prec": precision_score(test_labels, preds, zero_division=0),
            "rec": recall_score(test_labels, preds, zero_division=0),
        }
        bt = {}
        for t in ["benign", "single", "recover", "escalate"]:
            idx = [i for i, s in enumerate(test_scenarios) if s["type"] == t]
            if idx:
                correct = sum(1 for i in idx if preds[i] == test_scenarios[i]["label"])
                bt[t] = f"{correct}/{len(idx)}"

        delta = r["f1"] - probe_r["f1"]
        marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
        print(f"\n  [{feat_name}] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
        print(f"    By type: {bt}")
        results[feat_name] = (preds, r)

    # Group lift
    holo_f1 = results[f"Holonomy SO({n_group})"][1]["f1"]
    raw_f1 = results[f"Raw emb ({n_f}d)"][1]["f1"]
    print(f"\n  >> Group structure lift: {holo_f1 - raw_f1:+.3f} F1")

    # Bootstrap CI
    holo_preds = results[f"Holonomy SO({n_group})"][0]
    raw_preds = results[f"Raw emb ({n_f}d)"][0]
    rng = np.random.RandomState(42)
    n_boot = 2000
    deltas_hr = []
    deltas_hp = []
    for _ in range(n_boot):
        idx = rng.choice(len(test_labels), len(test_labels), replace=True)
        h_f = f1_score(test_labels[idx], holo_preds[idx], zero_division=0)
        r_f = f1_score(test_labels[idx], raw_preds[idx], zero_division=0)
        p_f = f1_score(test_labels[idx], probe_preds[idx], zero_division=0)
        deltas_hr.append(h_f - r_f)
        deltas_hp.append(h_f - p_f)

    deltas_hr = np.array(deltas_hr)
    deltas_hp = np.array(deltas_hp)
    print(f"\n  Bootstrap 95% CI (2000 iterations):")
    print(f"    Holonomy vs Raw:   [{np.percentile(deltas_hr, 2.5):+.3f}, {np.percentile(deltas_hr, 97.5):+.3f}]  P(holo>raw)={float((deltas_hr > 0).mean()):.3f}")
    print(f"    Holonomy vs Probe: [{np.percentile(deltas_hp, 2.5):+.3f}, {np.percentile(deltas_hp, 97.5):+.3f}]  P(holo>probe)={float((deltas_hp > 0).mean()):.3f}")

    sig_raw = np.percentile(deltas_hr, 2.5) > 0
    sig_probe = np.percentile(deltas_hp, 2.5) > 0
    print(f"    Significant vs raw: {'YES' if sig_raw else 'NO'}")
    print(f"    Significant vs probe: {'YES' if sig_probe else 'NO'}")

    # SIGNAL PERSISTENCE TEST (point 3)
    persistence = signal_persistence_test(b_vecs, h_vecs, n_group, scale)

    return {
        "probe_f1": probe_r["f1"],
        "holo_f1": holo_f1,
        "raw_f1": raw_f1,
        "lift_vs_raw": holo_f1 - raw_f1,
        "lift_vs_probe": holo_f1 - probe_r["f1"],
        "sig_vs_raw": sig_raw,
        "sig_vs_probe": sig_probe,
        "persistence": persistence,
    }


def main():
    print("=" * 70)
    print("HOLONOMY FINAL: Bigger data, honest eval, signal persistence")
    print("=" * 70)

    t0 = time.time()

    datasets = []

    # Load neuralchemy (largest)
    if os.path.exists("/tmp/neuralchemy_prompts.jsonl"):
        b_texts, h_texts = [], []
        with open("/tmp/neuralchemy_prompts.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d["label"] == 0:
                    b_texts.append(d["text"])
                else:
                    h_texts.append(d["text"])
        datasets.append(("Neuralchemy", b_texts, h_texts))

    # Load TensorTrust
    tt_path = "/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    if os.path.exists(tt_path):
        tt_b, tt_h = [], []
        with open(tt_path) as f:
            for line in f:
                d = json.loads(line)
                tt_b.append(d["pre_prompt"])
                tt_h.append(d["attack"])
        datasets.append(("TensorTrust", tt_b, tt_h))

    # Load deepset
    if os.path.exists("/tmp/deepset_prompts.jsonl"):
        ds_b, ds_h = [], []
        with open("/tmp/deepset_prompts.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d["label"] == 0:
                    ds_b.append(d["text"])
                else:
                    ds_h.append(d["text"])
        datasets.append(("Deepset", ds_b, ds_h))

    if not datasets:
        print()
        print("ERROR: No datasets loaded. Expected one or more of:")
        print("  - /tmp/neuralchemy_prompts.jsonl")
        print("  - /tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl")
        print("  - /tmp/deepset_prompts.jsonl")
        print()
        print("Cannot produce a scorecard without data. Aborting.")
        raise SystemExit(2)

    all_results = {}
    total_samples = 0
    for dataset_name, b_texts, h_texts in datasets:
        print(f"\n  Encoding {dataset_name} ({len(b_texts)}b/{len(h_texts)}h)...")
        all_vecs = encode(b_texts + h_texts)
        b_vecs = all_vecs[:len(b_texts)]
        h_vecs = all_vecs[len(b_texts):]
        total_samples += len(b_texts) + len(h_texts)

        r = run_experiment(dataset_name, b_vecs, h_vecs)
        all_results[dataset_name] = r

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(f"  FINAL SCORECARD")
    print(f"{'='*70}")
    print(f"  {'Dataset':<15} {'Probe':>7} {'Raw':>7} {'Holo':>7} {'Lift':>7} {'Sig?':>5} {'Persist?':>8}")
    print(f"  {'-'*60}")
    for name, r in all_results.items():
        print(f"  {name:<15} {r['probe_f1']:>7.3f} {r['raw_f1']:>7.3f} {r['holo_f1']:>7.3f} "
              f"{r['lift_vs_raw']:>+7.3f} {'YES' if r['sig_vs_raw'] else 'no':>5} "
              f"{'YES' if r['persistence'] else 'no':>8}")

    print(f"\n  Datasets loaded: {len(all_results)} / 3 ({total_samples} samples)")

    def verdict(predicate_per_result, label_win, label_partial, label_lost):
        flags = [predicate_per_result(r) for r in all_results.values()]
        if not flags:
            return "NO DATA"
        if all(flags):
            return label_win
        if any(flags):
            return label_partial
        return label_lost

    sig_verdict = verdict(lambda r: r["sig_vs_raw"], "WIN", "PARTIAL", "LOST")
    persist_verdict = verdict(lambda r: r["persistence"], "WIN", "PARTIAL", "LOST")
    lift_verdict = verdict(lambda r: r["lift_vs_raw"] > 0.01, "WIN", "PARTIAL", "LOST")

    print(f"  Point 2 (significance vs raw):  {sig_verdict}")
    print(f"  Point 3 (persistence vs raw):   {persist_verdict}")
    print(f"  Point 4 (lift > 0.01 vs raw):   {lift_verdict}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
