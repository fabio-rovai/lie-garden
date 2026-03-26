"""
Holonomy v2: Higher-dimensional groups + matrix-level features
==============================================================

Three fundamental fixes over v1:
1. SO(10) instead of SO(4) — uses 45 algebra coefficients instead of 6
2. Full log-matrix features instead of scalar norm — preserves direction
3. Confinement region (Mahalanobis) instead of threshold — geometric test

The key mathematical insight: holonomy measures the SHAPE of a path through
the group, not just its endpoint distance from identity. A scalar norm
collapses this to a single number, losing all the geometric information
that makes holonomy interesting.

By using the full log-matrix as features and training a classifier on
conversation-level holonomy matrices, we let the classifier find the
discriminative DIRECTION in the rotation — not just its magnitude.
"""
import json
import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, logm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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
# Core: High-dimensional holonomy
# ---------------------------------------------------------------------------

def vec_to_skew(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Map coefficient vector to n×n skew-symmetric matrix."""
    A = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(coeffs):
                A[i, j] = coeffs[idx]
                A[j, i] = -coeffs[idx]
            idx += 1
    return A


def skew_to_vec(A: np.ndarray) -> np.ndarray:
    """Extract upper-triangle coefficients from skew-symmetric matrix."""
    n = A.shape[0]
    coeffs = []
    for i in range(n):
        for j in range(i + 1, n):
            coeffs.append(A[i, j])
    return np.array(coeffs)


def compute_holonomy(vecs: np.ndarray, n: int, scale: float = 0.1) -> np.ndarray:
    """Compute cumulative holonomy matrix for a sequence of vectors."""
    n_coeffs = n * (n - 1) // 2
    state = np.eye(n)
    for v in vecs:
        coeffs = v[:n_coeffs] * scale
        A = vec_to_skew(coeffs, n)
        state = state @ expm(A)
    return state


def safe_logm(R: np.ndarray) -> np.ndarray:
    """Compute log of rotation matrix, return real skew-symmetric part."""
    try:
        L = logm(R)
        # Extract skew-symmetric part (should already be skew for SO(n))
        return (L - L.T) / 2.0
    except Exception:
        return np.zeros_like(R)


# ---------------------------------------------------------------------------
# Feature extraction from holonomy
# ---------------------------------------------------------------------------

def holonomy_features(vecs: np.ndarray, n: int, scale: float = 0.1) -> np.ndarray:
    """
    Extract rich features from the holonomy matrix:
    1. Upper triangle of log(holonomy) — the algebra element (direction + magnitude)
    2. Frobenius norm of log — overall rotation angle
    3. Eigenvalue spectrum of the algebra element — rotation in each plane
    4. Trace of holonomy — conjugation-invariant
    5. Max commutator norm in the sequence — curvature measure
    """
    n_coeffs = n * (n - 1) // 2

    # Compute holonomy
    state = np.eye(n)
    skews = []
    for v in vecs:
        coeffs = v[:n_coeffs] * scale
        A = vec_to_skew(coeffs, n)
        skews.append(A)
        state = state @ expm(A)

    # Feature 1: Log-holonomy coefficients (the key one — preserves direction)
    log_h = safe_logm(state)
    log_coeffs = skew_to_vec(log_h)  # n_coeffs values

    # Feature 2: Frobenius norm
    fro_norm = np.linalg.norm(log_h, "fro")

    # Feature 3: Eigenvalue spectrum of log_h
    # For a skew-symmetric matrix, eigenvalues are purely imaginary ±iλ_k
    eigvals = np.sort(np.abs(np.linalg.eigvals(log_h)))[::-1]
    eig_real = eigvals.real[:n//2]  # Take n//2 largest magnitudes
    if len(eig_real) < n // 2:
        eig_real = np.pad(eig_real, (0, n // 2 - len(eig_real)))

    # Feature 4: Trace of holonomy (conjugation-invariant, related to angle)
    trace_h = np.trace(state)

    # Feature 5: Max and mean commutator norms (curvature along path)
    comm_norms = []
    for i in range(len(skews) - 1):
        C = skews[i] @ skews[i+1] - skews[i+1] @ skews[i]
        comm_norms.append(np.linalg.norm(C, "fro"))
    max_comm = max(comm_norms) if comm_norms else 0.0
    mean_comm = np.mean(comm_norms) if comm_norms else 0.0

    # Feature 6: Holonomy at intermediate points (captures trajectory shape)
    mid = len(vecs) // 2
    if mid > 0:
        mid_state = np.eye(n)
        for v in vecs[:mid]:
            coeffs = v[:n_coeffs] * scale
            A = vec_to_skew(coeffs, n)
            mid_state = mid_state @ expm(A)
        mid_norm = np.linalg.norm(mid_state - np.eye(n), "fro")
    else:
        mid_norm = 0.0

    # Combine all features
    features = np.concatenate([
        log_coeffs,           # n_coeffs values (direction of rotation)
        [fro_norm],           # 1 value
        eig_real,             # n//2 values
        [trace_h],            # 1 value
        [max_comm, mean_comm],  # 2 values
        [mid_norm],           # 1 value (trajectory shape)
    ])

    return features


# ---------------------------------------------------------------------------
# Confinement region (Mahalanobis distance from benign distribution)
# ---------------------------------------------------------------------------

class ConfinementRegion:
    """
    Define a "safe region" in the Lie algebra from benign-only conversations.
    Test new conversations by Mahalanobis distance from this region.

    This is the mathematically principled use of holonomy:
    holonomy stays within a region iff the curvature is bounded.
    """
    def __init__(self):
        self.mean = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, benign_features: np.ndarray, contamination: float = 0.05):
        """Fit the benign region from benign conversation features."""
        self.mean = benign_features.mean(axis=0)
        cov = np.cov(benign_features, rowvar=False)
        # Regularize
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

        # Set threshold at the (1-contamination) percentile of benign distances
        distances = self._distances(benign_features)
        self.threshold = np.percentile(distances, (1 - contamination) * 100)

    def _distances(self, features: np.ndarray) -> np.ndarray:
        diff = features - self.mean
        # Mahalanobis: sqrt(diff @ cov_inv @ diff.T) for each row
        return np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """1 = outside confinement (attack), 0 = inside (benign)."""
        distances = self._distances(features)
        return (distances > self.threshold).astype(int)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distances (higher = more anomalous)."""
        return self._distances(features)


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def make_scenarios(b_vecs, h_vecs, n_scenarios=200, seed=42):
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
        elif t == 1:  # single attack
            b_idx = rng.choice(n_b, 6, replace=True)
            h_idx = rng.choice(n_h)
            pos = rng.randint(2, 5)
            vecs = np.vstack([b_vecs[b_idx[:pos]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[pos:]]])
            label = 1
            desc = "single"
        elif t == 2:  # attack + recovery (THE hard case)
            b_idx = rng.choice(n_b, 7, replace=True)
            h_idx = rng.choice(n_h)
            vecs = np.vstack([b_vecs[b_idx[:2]], h_vecs[h_idx:h_idx+1], b_vecs[b_idx[2:]]])
            label = 1
            desc = "recover"
        else:  # escalation
            b_idx = rng.choice(n_b, 5, replace=True)
            h_idx = rng.choice(n_h, 2, replace=False)
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
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(name, b_vecs, h_vecs, probe_dir, probe_thresh):
    print(f"\n{'='*70}")
    print(f"  {name}: {len(b_vecs)} benign, {len(h_vecs)} harmful")
    print(f"{'='*70}")

    # Generate scenarios with two seeds: one for train, one for test
    train_scenarios = make_scenarios(b_vecs, h_vecs, 400, seed=42)
    test_scenarios = make_scenarios(b_vecs, h_vecs, 400, seed=99)

    # ===================================================================
    # Baseline: Probe-only (per-step max score)
    # ===================================================================
    def probe_detect(vecs):
        scores = vecs @ probe_dir
        return bool(scores.max() > probe_thresh)

    test_labels = [s["label"] for s in test_scenarios]
    probe_preds = [int(probe_detect(s["vecs"])) for s in test_scenarios]
    probe_r = eval_results(test_labels, probe_preds)
    probe_bt = eval_by_type(test_scenarios, [bool(p) for p in probe_preds])
    print(f"\n  [Baseline] Probe-only:  F1={probe_r['f1']:.3f}  P={probe_r['prec']:.3f}  R={probe_r['rec']:.3f}")
    print(f"    By type: {probe_bt}")

    # ===================================================================
    # Method 1: High-dim holonomy + LogReg on matrix features
    # ===================================================================
    for n_group in [6, 10, 16]:
        n_coeffs = n_group * (n_group - 1) // 2
        if n_coeffs > b_vecs.shape[1]:
            continue

        # Try multiple scales
        for scale in [0.05, 0.1, 0.2]:
            # Extract features for train and test
            train_feats = np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scenarios])
            test_feats = np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios])
            train_labels = np.array([s["label"] for s in train_scenarios])

            # Handle NaN/Inf
            train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=1e6, neginf=-1e6)
            test_feats = np.nan_to_num(test_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            # Standardize
            scaler = StandardScaler()
            train_feats_s = scaler.fit_transform(train_feats)
            test_feats_s = scaler.transform(test_feats)

            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(train_feats_s, train_labels)

            preds = clf.predict(test_feats_s)
            r = eval_results(test_labels, preds)
            bt = eval_by_type(test_scenarios, [bool(p) for p in preds])

            delta = r["f1"] - probe_r["f1"]
            marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
            print(f"\n  [SO({n_group}) s={scale}] LogReg:  F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
            print(f"    By type: {bt}")
            n_feats = train_feats.shape[1]
            print(f"    Features: {n_feats} ({n_coeffs} log-coeffs + {n_feats - n_coeffs} invariants)")

            # What does holonomy catch that probe doesn't?
            recover_idx = [i for i, s in enumerate(test_scenarios) if s["type"] == "recover"]
            if recover_idx:
                holo_only = sum(1 for i in recover_idx
                               if preds[i] == 1 and probe_preds[i] == 0)
                probe_only = sum(1 for i in recover_idx
                                if preds[i] == 0 and probe_preds[i] == 1)
                print(f"    Recovery: holonomy-only={holo_only}, probe-only={probe_only}")

    # ===================================================================
    # Method 2: Confinement region (Mahalanobis from benign distribution)
    # ===================================================================
    print(f"\n  --- Confinement Region (Mahalanobis) ---")
    for n_group in [6, 10]:
        n_coeffs = n_group * (n_group - 1) // 2
        if n_coeffs > b_vecs.shape[1]:
            continue

        for scale in [0.05, 0.1]:
            # Generate many benign-only scenarios for fitting
            benign_only = [s for s in train_scenarios if s["label"] == 0]
            benign_feats = np.array([holonomy_features(s["vecs"], n_group, scale) for s in benign_only])
            benign_feats = np.nan_to_num(benign_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            region = ConfinementRegion()
            region.fit(benign_feats, contamination=0.05)

            test_feats = np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios])
            test_feats = np.nan_to_num(test_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            preds = region.predict(test_feats)
            r = eval_results(test_labels, preds)
            bt = eval_by_type(test_scenarios, [bool(p) for p in preds])

            delta = r["f1"] - probe_r["f1"]
            marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
            print(f"\n  [Confine SO({n_group}) s={scale}] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
            print(f"    By type: {bt}")

    # ===================================================================
    # Method 3: Holonomy + Probe combined features
    # ===================================================================
    print(f"\n  --- Combined: Holonomy + Probe features ---")
    n_group = 10
    scale = 0.1
    n_coeffs = n_group * (n_group - 1) // 2
    if n_coeffs <= b_vecs.shape[1]:
        # Add probe-derived features to holonomy features
        def combined_features(vecs, n=n_group, s=scale):
            holo_f = holonomy_features(vecs, n, s)
            # Probe features: max score, mean score, score variance, #above-thresh
            scores = vecs @ probe_dir
            probe_f = np.array([
                scores.max(),
                scores.mean(),
                scores.std(),
                (scores > probe_thresh).sum() / len(scores),
                # Score trajectory features
                np.diff(scores).max() if len(scores) > 1 else 0.0,  # max jump
                np.diff(scores).min() if len(scores) > 1 else 0.0,  # max drop
            ])
            return np.concatenate([holo_f, probe_f])

        train_feats = np.array([combined_features(s["vecs"]) for s in train_scenarios])
        test_feats = np.array([combined_features(s["vecs"]) for s in test_scenarios])
        train_labels = np.array([s["label"] for s in train_scenarios])

        train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        test_feats = np.nan_to_num(test_feats, nan=0.0, posinf=1e6, neginf=-1e6)

        scaler = StandardScaler()
        train_feats_s = scaler.fit_transform(train_feats)
        test_feats_s = scaler.transform(test_feats)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(train_feats_s, train_labels)

        preds = clf.predict(test_feats_s)
        r = eval_results(test_labels, preds)
        bt = eval_by_type(test_scenarios, [bool(p) for p in preds])

        delta = r["f1"] - probe_r["f1"]
        marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
        print(f"\n  [Combined SO(10)+Probe] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
        print(f"    By type: {bt}")

        # Feature importance — which features matter?
        coef = np.abs(clf.coef_[0])
        n_holo = len(holonomy_features(train_scenarios[0]["vecs"], n_group, scale))
        holo_importance = coef[:n_holo].sum()
        probe_importance = coef[n_holo:].sum()
        print(f"    Feature importance: holonomy={holo_importance:.3f}, probe={probe_importance:.3f}")
        print(f"    Holonomy fraction: {holo_importance / (holo_importance + probe_importance):.1%}")

        # Ablation: probe features only (same classifier)
        probe_only_train = train_feats[:, n_holo:]
        probe_only_test = test_feats[:, n_holo:]
        scaler2 = StandardScaler()
        probe_only_train_s = scaler2.fit_transform(probe_only_train)
        probe_only_test_s = scaler2.transform(probe_only_test)
        clf2 = LogisticRegression(max_iter=1000, C=1.0)
        clf2.fit(probe_only_train_s, train_labels)
        preds2 = clf2.predict(probe_only_test_s)
        r2 = eval_results(test_labels, preds2)
        bt2 = eval_by_type(test_scenarios, [bool(p) for p in preds2])
        delta2 = r2["f1"] - probe_r["f1"]
        print(f"\n  [Ablation: Probe-feats only] F1={r2['f1']:.3f}  P={r2['prec']:.3f}  R={r2['rec']:.3f}  Δ={delta2:+.3f}")
        print(f"    By type: {bt2}")

        # The key question: does combined beat probe-feats-only?
        holo_lift = r["f1"] - r2["f1"]
        print(f"\n  >> Holonomy marginal lift over probe features: {holo_lift:+.3f} F1")
        if holo_lift > 0.01:
            print(f"  >> HOLONOMY ADDS GENUINE VALUE!")

            # Detailed: what specifically does holonomy catch?
            combined_only = sum(1 for i in range(len(test_scenarios))
                              if preds[i] == 1 and preds2[i] == 0 and test_labels[i] == 1)
            probe_feat_only = sum(1 for i in range(len(test_scenarios))
                                if preds[i] == 0 and preds2[i] == 1 and test_labels[i] == 1)
            print(f"  >> Combined catches {combined_only} attacks that probe-feats miss")
            print(f"  >> Probe-feats catches {probe_feat_only} attacks that combined misses")

            # Which types benefit?
            for t in ["benign", "single", "recover", "escalate"]:
                t_idx = [i for i, s in enumerate(test_scenarios) if s["type"] == t]
                if not t_idx:
                    continue
                c_correct = sum(1 for i in t_idx if preds[i] == (test_scenarios[i]["label"] == 1))
                p_correct = sum(1 for i in t_idx if preds2[i] == (test_scenarios[i]["label"] == 1))
                if c_correct != p_correct:
                    print(f"     {t}: combined={c_correct}/{len(t_idx)}, probe-only={p_correct}/{len(t_idx)}")
        else:
            print(f"  >> Holonomy adds no marginal value over probe features alone")

    # ===================================================================
    # Method 4: Cross-validated holonomy-only (honest test)
    # ===================================================================
    print(f"\n  --- Cross-validated holonomy-only (no probe info) ---")
    n_group = 10
    scale = 0.1
    n_coeffs = n_group * (n_group - 1) // 2
    if n_coeffs <= b_vecs.shape[1]:
        all_scenarios = train_scenarios + test_scenarios
        all_feats = np.array([holonomy_features(s["vecs"], n_group, scale) for s in all_scenarios])
        all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        all_labels = np.array([s["label"] for s in all_scenarios])

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_f1s = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_feats, all_labels)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(all_feats[train_idx])
            X_test = scaler.transform(all_feats[test_idx])
            y_train = all_labels[train_idx]
            y_test = all_labels[test_idx]

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1 = f1_score(y_test, preds, zero_division=0)
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        delta = mean_f1 - probe_r["f1"]
        marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
        print(f"  [CV SO(10) holonomy-only] F1={mean_f1:.3f}±{std_f1:.3f}  Δ={delta:+.3f} {marker}")
        print(f"    Folds: {[f'{f:.3f}' for f in fold_f1s]}")

    # ===================================================================
    # CRITICAL ABLATION: Raw embedding features vs holonomy features
    # Same classifier, same feature count — is the GROUP doing anything?
    # ===================================================================
    print(f"\n  --- CRITICAL ABLATION: Is the group structure doing anything? ---")
    print(f"  (Same LogReg classifier, comparable feature count)")
    n_group = 10
    scale = 0.1
    n_coeffs = n_group * (n_group - 1) // 2

    if n_coeffs <= b_vecs.shape[1]:
        # Holonomy features (sequence → group → log-matrix → features)
        holo_train = np.array([holonomy_features(s["vecs"], n_group, scale) for s in train_scenarios])
        holo_test = np.array([holonomy_features(s["vecs"], n_group, scale) for s in test_scenarios])
        holo_train = np.nan_to_num(holo_train, nan=0.0, posinf=1e6, neginf=-1e6)
        holo_test = np.nan_to_num(holo_test, nan=0.0, posinf=1e6, neginf=-1e6)
        n_holo_feats = holo_train.shape[1]

        # Raw embedding features: same dimensionality, but skip the group
        # Use: mean embedding, max embedding, std embedding, pairwise diffs
        def raw_embedding_features(vecs, n_feats_target):
            """Extract sequence-level features from raw embeddings (no group)."""
            mean_v = vecs.mean(axis=0)
            max_v = vecs.max(axis=0)
            std_v = vecs.std(axis=0)
            # Pairwise cosine similarities between consecutive messages
            cos_sims = []
            for i in range(len(vecs) - 1):
                d = np.dot(vecs[i], vecs[i+1]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i+1]) + 1e-12)
                cos_sims.append(d)
            cos_stats = [np.mean(cos_sims), np.std(cos_sims), np.min(cos_sims), np.max(cos_sims)] if cos_sims else [0,0,0,0]

            # Take first n_feats_target dimensions from concatenation
            raw = np.concatenate([
                mean_v[:n_feats_target//3],
                max_v[:n_feats_target//3],
                std_v[:n_feats_target//3],
                cos_stats,
            ])
            # Pad or truncate to match n_feats_target
            if len(raw) < n_feats_target:
                raw = np.pad(raw, (0, n_feats_target - len(raw)))
            return raw[:n_feats_target]

        raw_train = np.array([raw_embedding_features(s["vecs"], n_holo_feats) for s in train_scenarios])
        raw_test = np.array([raw_embedding_features(s["vecs"], n_holo_feats) for s in test_scenarios])
        raw_train = np.nan_to_num(raw_train, nan=0.0, posinf=1e6, neginf=-1e6)
        raw_test = np.nan_to_num(raw_test, nan=0.0, posinf=1e6, neginf=-1e6)

        train_labels_arr = np.array([s["label"] for s in train_scenarios])

        # Train and evaluate both
        for feat_name, f_train, f_test in [
            ("Holonomy SO(10)", holo_train, holo_test),
            ("Raw embeddings", raw_train, raw_test),
        ]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(f_train)
            X_test = scaler.transform(f_test)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train, train_labels_arr)
            preds = clf.predict(X_test)
            r = eval_results(test_labels, preds)
            bt = eval_by_type(test_scenarios, [bool(p) for p in preds])
            delta = r["f1"] - probe_r["f1"]
            marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
            print(f"  [{feat_name} ({f_train.shape[1]}d)] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
            print(f"    By type: {bt}")

        # Also: just first-N embedding coefficients through LogReg (no sequence processing)
        # This tests whether the 45 coefficients used by SO(10) are just "the best features"
        def first_n_features(vecs, n):
            """Just mean of first N embedding dims — no group structure."""
            return vecs[:, :n].mean(axis=0)

        fn_train = np.array([first_n_features(s["vecs"], n_coeffs) for s in train_scenarios])
        fn_test = np.array([first_n_features(s["vecs"], n_coeffs) for s in test_scenarios])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(fn_train)
        X_test = scaler.transform(fn_test)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, train_labels_arr)
        preds = clf.predict(X_test)
        r = eval_results(test_labels, preds)
        bt = eval_by_type(test_scenarios, [bool(p) for p in preds])
        delta = r["f1"] - probe_r["f1"]
        marker = "**WIN**" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "lose")
        print(f"  [Mean first-{n_coeffs} emb dims ({n_coeffs}d)] F1={r['f1']:.3f}  P={r['prec']:.3f}  R={r['rec']:.3f}  Δ={delta:+.3f} {marker}")
        print(f"    By type: {bt}")

        holo_f1 = eval_results(test_labels, LogisticRegression(max_iter=1000).fit(
            StandardScaler().fit_transform(holo_train),
            train_labels_arr
        ).predict(StandardScaler().fit(holo_train).transform(holo_test)))["f1"]
        raw_f1 = eval_results(test_labels, LogisticRegression(max_iter=1000).fit(
            StandardScaler().fit_transform(raw_train),
            train_labels_arr
        ).predict(StandardScaler().fit(raw_train).transform(raw_test)))["f1"]

        print(f"\n  >> GROUP STRUCTURE LIFT: {holo_f1 - raw_f1:+.3f} F1 (holonomy vs raw embeddings)")
        if holo_f1 > raw_f1 + 0.01:
            print(f"  >> THE GROUP OPERATION ADDS GENUINE VALUE!")
        else:
            print(f"  >> The group operation may not add value — raw embeddings perform similarly")


def main():
    print("=" * 70)
    print("HOLONOMY v2: Higher-dimensional groups + matrix features")
    print("Can geometric features beat a simple probe?")
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

    for name, b_texts, h_texts in [
        ("InjecAgent", benign_inj, harmful_inj),
        ("TensorTrust", tt_b, tt_h),
    ]:
        print(f"\n  Encoding {name}...")
        all_vecs = encode(b_texts + h_texts)
        b_vecs = all_vecs[:len(b_texts)]
        h_vecs = all_vecs[len(b_texts):]

        # Train probe baseline
        diff = h_vecs.mean(0) - b_vecs.mean(0)
        probe_dir = diff / (np.linalg.norm(diff) + 1e-12)
        probe_thresh = (b_vecs @ probe_dir).mean() + (h_vecs @ probe_dir).mean()
        probe_thresh /= 2.0

        run_experiment(name, b_vecs, h_vecs, probe_dir, probe_thresh)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print("VERDICT: If any method shows positive Δ consistently,")
    print("holonomy captures geometric structure the probe misses.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
