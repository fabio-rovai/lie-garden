"""
Can holonomy ACTUALLY detect something the probe can't?
========================================================
Tests three formulations:
1. Current: cumulative scar (||state - ref||) — already shown to be noisy
2. Windowed holonomy: closed-loop holonomy over sliding windows
3. Pairwise curvature: commutator [A_i, A_{i+1}] between consecutive messages

The hypothesis: benign messages live in a near-commutative subspace of the
Lie algebra, but attacks break commutativity. If true, this is a GENUINE
geometric signal that a per-step probe cannot capture.
"""
import json
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, logm
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Embedding (cached model)
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
        # Hash fallback
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
# Directional Probe (baseline)
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0

    def train(self, benign_vecs: np.ndarray, harmful_vecs: np.ndarray):
        diff = harmful_vecs.mean(0) - benign_vecs.mean(0)
        norm = np.linalg.norm(diff)
        self.direction = diff / norm if norm > 1e-12 else diff
        b_scores = benign_vecs @ self.direction
        h_scores = harmful_vecs @ self.direction
        self.threshold = (b_scores.mean() + h_scores.mean()) / 2

    def predict(self, vecs: np.ndarray) -> np.ndarray:
        scores = vecs @ self.direction
        return (scores > self.threshold).astype(int)


# ---------------------------------------------------------------------------
# Text → Lie algebra mapping
# ---------------------------------------------------------------------------

def vec_to_skew(vec: np.ndarray, n: int, scale: float = 0.1) -> np.ndarray:
    """Map embedding vector to so(n) skew-symmetric matrix."""
    n_coeffs = n * (n - 1) // 2
    coeffs = vec[:n_coeffs] if len(vec) >= n_coeffs else np.resize(vec, n_coeffs)
    A = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = coeffs[idx] * scale
            A[j, i] = -coeffs[idx] * scale
            idx += 1
    return A


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """||[A, B]||_F = ||AB - BA||_F — measures non-commutativity."""
    C = A @ B - B @ A
    return float(np.linalg.norm(C, "fro"))


# ---------------------------------------------------------------------------
# Formulation 1: Cumulative scar (current — baseline)
# ---------------------------------------------------------------------------

def cumulative_scar_test(vecs: np.ndarray, labels: np.ndarray,
                         benign_vecs: np.ndarray, n: int = 4):
    """Current approach: accumulate state, measure ||state - ref||."""
    # Calibrate reference from benign
    ref = np.eye(n)
    for v in benign_vecs[:10]:
        A = vec_to_skew(v, n)
        ref = ref @ expm(A)

    state = np.eye(n)
    scars = []
    for v in vecs:
        A = vec_to_skew(v, n)
        state = state @ expm(A)
        scars.append(np.linalg.norm(state - ref, "fro"))

    return np.array(scars)


# ---------------------------------------------------------------------------
# Formulation 2: Windowed closed-loop holonomy
# ---------------------------------------------------------------------------

def windowed_holonomy(vecs: np.ndarray, n: int = 4, window: int = 5):
    """
    For each position i, compute the holonomy of the loop
    [v_{i-w}, ..., v_i, v_i^{-1}, ..., v_{i-w}^{-1}].
    A closed loop through identity on a flat connection returns I.
    Curvature makes it deviate.
    """
    scores = []
    for i in range(len(vecs)):
        start = max(0, i - window + 1)
        window_vecs = vecs[start:i+1]

        # Forward path
        forward = np.eye(n)
        for v in window_vecs:
            A = vec_to_skew(v, n)
            forward = forward @ expm(A)

        # The holonomy defect: how far is forward @ forward^{-T} from I?
        # Actually: holonomy = forward product. Deviation from "expected" (identity
        # or reference) is the signal.
        # Better: compare to a reference loop of same length from benign data
        scores.append(np.linalg.norm(forward - np.eye(n), "fro"))

    return np.array(scores)


# ---------------------------------------------------------------------------
# Formulation 3: Pairwise curvature (commutator-based)
# ---------------------------------------------------------------------------

def pairwise_curvature(vecs: np.ndarray, n: int = 8, window: int = 3):
    """
    For each position i, compute the average commutator norm over a window:
    score_i = mean(||[A_j, A_{j+1}]|| for j in [i-w..i])

    Hypothesis: benign text lives in a near-commutative subspace.
    Attacks break commutativity.
    """
    # Pre-compute all skew matrices
    skews = [vec_to_skew(v, n) for v in vecs]

    scores = []
    for i in range(len(vecs)):
        if i == 0:
            scores.append(0.0)
            continue
        start = max(0, i - window + 1)
        norms = []
        for j in range(start, i):
            norms.append(commutator_norm(skews[j], skews[j+1]))
        scores.append(np.mean(norms) if norms else 0.0)

    return np.array(scores)


# ---------------------------------------------------------------------------
# Formulation 4: Differential curvature (benign-relative)
# ---------------------------------------------------------------------------

def differential_curvature(vecs: np.ndarray, benign_vecs: np.ndarray,
                           n: int = 8, window: int = 3):
    """
    Compute pairwise curvature, but SUBTRACT the expected benign curvature.
    This way, normal conversation curvature is ~0, attacks spike.
    """
    # Compute baseline curvature from benign text
    benign_skews = [vec_to_skew(v, n) for v in benign_vecs[:50]]
    benign_curvs = []
    for j in range(len(benign_skews) - 1):
        benign_curvs.append(commutator_norm(benign_skews[j], benign_skews[j+1]))
    baseline = np.mean(benign_curvs) if benign_curvs else 0.0
    baseline_std = np.std(benign_curvs) if benign_curvs else 1.0

    # Compute test curvature relative to baseline
    skews = [vec_to_skew(v, n) for v in vecs]
    scores = []
    for i in range(len(vecs)):
        if i == 0:
            scores.append(0.0)
            continue
        start = max(0, i - window + 1)
        norms = []
        for j in range(start, i):
            norms.append(commutator_norm(skews[j], skews[j+1]))
        raw = np.mean(norms) if norms else 0.0
        # Z-score relative to benign baseline
        scores.append((raw - baseline) / baseline_std if baseline_std > 0 else 0.0)

    return np.array(scores)


# ---------------------------------------------------------------------------
# Multi-step scenario generator
# ---------------------------------------------------------------------------

def make_multi_step_scenarios(benign_vecs, harmful_vecs, n_scenarios=100):
    """
    Create realistic multi-step scenarios:
    - Type A: all benign (should be negative)
    - Type B: single attack in benign context (should be positive)
    - Type C: attack then recovery (should be positive — this is the hard one)
    - Type D: gradual escalation (multiple attacks)
    """
    scenarios = []
    rng = np.random.RandomState(42)

    n_b = len(benign_vecs)
    n_h = len(harmful_vecs)

    for i in range(n_scenarios):
        scenario_type = i % 4

        if scenario_type == 0:
            # Type A: all benign
            idxs = rng.choice(n_b, size=8, replace=True)
            vecs = benign_vecs[idxs]
            label = 0
            desc = "all_benign"

        elif scenario_type == 1:
            # Type B: benign-attack-benign
            b_idxs = rng.choice(n_b, size=6, replace=True)
            h_idx = rng.choice(n_h)
            pos = rng.randint(2, 5)
            vecs = np.concatenate([
                benign_vecs[b_idxs[:pos]],
                harmful_vecs[h_idx:h_idx+1],
                benign_vecs[b_idxs[pos:]]
            ])
            label = 1
            desc = "single_attack"

        elif scenario_type == 2:
            # Type C: attack then long recovery
            b_idxs = rng.choice(n_b, size=7, replace=True)
            h_idx = rng.choice(n_h)
            vecs = np.concatenate([
                benign_vecs[b_idxs[:2]],
                harmful_vecs[h_idx:h_idx+1],
                benign_vecs[b_idxs[2:]]  # 5 recovery messages
            ])
            label = 1
            desc = "attack_recover"

        else:
            # Type D: gradual escalation (2 attacks)
            b_idxs = rng.choice(n_b, size=5, replace=True)
            h_idxs = rng.choice(n_h, size=2, replace=False)
            vecs = np.concatenate([
                benign_vecs[b_idxs[:2]],
                harmful_vecs[h_idxs[0]:h_idxs[0]+1],
                benign_vecs[b_idxs[2:4]],
                harmful_vecs[h_idxs[1]:h_idxs[1]+1],
                benign_vecs[b_idxs[4:5]]
            ])
            label = 1
            desc = "escalation"

        scenarios.append({"vecs": vecs, "label": label, "type": desc})

    return scenarios


# ---------------------------------------------------------------------------
# Evaluate a scoring method on multi-step scenarios
# ---------------------------------------------------------------------------

def eval_method(scenarios, score_fn, threshold):
    """Score each scenario with max score, predict positive if > threshold."""
    preds = []
    labels = []
    for s in scenarios:
        scores = score_fn(s["vecs"])
        preds.append(1 if scores.max() > threshold else 0)
        labels.append(s["label"])

    labels = np.array(labels)
    preds = np.array(preds)

    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "tp": int(((preds == 1) & (labels == 1)).sum()),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
        "tn": int(((preds == 0) & (labels == 0)).sum()),
    }


def find_best_threshold(scenarios, score_fn, n_thresholds=50):
    """Find threshold that maximizes F1."""
    all_scores = []
    for s in scenarios:
        scores = score_fn(s["vecs"])
        all_scores.append(scores.max())
    all_scores = np.array(all_scores)

    best_f1 = 0
    best_t = 0
    for t in np.linspace(all_scores.min(), all_scores.max(), n_thresholds):
        preds = (all_scores > t).astype(int)
        labels = np.array([s["label"] for s in scenarios])
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CAN HOLONOMY DETECT WHAT A PROBE CAN'T?")
    print("Testing on REAL InjecAgent + TensorTrust data")
    print("=" * 70)

    # Load data
    benign_inj, harmful_inj = [], []
    with open("/tmp/InjecAgent/data/user_cases.jsonl") as f:
        for line in f:
            benign_inj.append(json.loads(line)["User Instruction"])
    for fname in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
        with open(f"/tmp/InjecAgent/data/{fname}") as f:
            for line in f:
                harmful_inj.append(json.loads(line)["Attacker Instruction"])
    for fname in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(f"/tmp/InjecAgent/data/{fname}") as f:
            for c in json.load(f):
                harmful_inj.append(c["Attacker Instruction"])

    benign_tt, harmful_tt = [], []
    with open("/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            benign_tt.append(d["pre_prompt"])
            harmful_tt.append(d["attack"])

    for name, benign_texts, harmful_texts in [
        ("InjecAgent", benign_inj, harmful_inj),
        ("TensorTrust", benign_tt, harmful_tt),
    ]:
        print(f"\n{'='*70}")
        print(f"  {name}: {len(benign_texts)} benign, {len(harmful_texts)} harmful")
        print(f"{'='*70}")

        # Encode all texts at once (fast batch)
        print("  Encoding...")
        all_texts = benign_texts + harmful_texts
        all_vecs = encode(all_texts)
        b_vecs = all_vecs[:len(benign_texts)]
        h_vecs = all_vecs[len(benign_texts):]

        # Train probe
        probe = Probe()
        probe.train(b_vecs, h_vecs)

        # Generate multi-step scenarios
        print("  Generating scenarios...")
        scenarios = make_multi_step_scenarios(b_vecs, h_vecs, n_scenarios=200)

        # Method 0: Probe-only (per-step max)
        def probe_score(vecs):
            scores = vecs @ probe.direction
            return scores - probe.threshold  # positive = harmful

        # Method 1: Cumulative scar
        def scar_score(vecs):
            return cumulative_scar_test(vecs, None, b_vecs, n=4)

        # Method 2: Windowed holonomy
        def wind_score(vecs):
            return windowed_holonomy(vecs, n=4, window=4)

        # Method 3: Pairwise curvature (so(8))
        def curv_score(vecs):
            return pairwise_curvature(vecs, n=8, window=3)

        # Method 4: Differential curvature
        def diff_curv_score(vecs):
            return differential_curvature(vecs, b_vecs, n=8, window=3)

        # Method 5: Probe + curvature combined
        def combined_score(vecs):
            p = probe_score(vecs)
            c = pairwise_curvature(vecs, n=8, window=3)
            # Normalize both to [0,1] range
            p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)
            c_norm = (c - c.min()) / (c.max() - c.min() + 1e-12)
            return p_norm + c_norm

        methods = [
            ("Probe-only", probe_score),
            ("Cumulative scar", scar_score),
            ("Windowed holonomy", wind_score),
            ("Pairwise curvature", curv_score),
            ("Differential curvature", diff_curv_score),
            ("Probe + curvature", combined_score),
        ]

        print(f"\n  {'Method':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  Thresh")
        print(f"  {'-'*85}")

        for method_name, score_fn in methods:
            # Find best threshold on the scenarios themselves (optimistic — just to see ceiling)
            best_t, best_f1 = find_best_threshold(scenarios, score_fn)
            result = eval_method(scenarios, score_fn, best_t)
            print(f"  {method_name:<25} {result['f1']:>6.3f} {result['prec']:>6.3f} "
                  f"{result['rec']:>6.3f} {result['tp']:>4} {result['fp']:>4} "
                  f"{result['fn']:>4} {result['tn']:>4}  {best_t:.4f}")

        # Breakdown by scenario type
        print(f"\n  Detection by scenario type (using best method's threshold):")
        # Find best geometric method
        best_geo_name = None
        best_geo_f1 = 0
        best_geo_fn = None
        best_geo_t = 0
        for method_name, score_fn in methods[1:]:  # skip probe
            t, f1 = find_best_threshold(scenarios, score_fn)
            if f1 > best_geo_f1:
                best_geo_f1 = f1
                best_geo_name = method_name
                best_geo_fn = score_fn
                best_geo_t = t

        if best_geo_fn:
            probe_t, _ = find_best_threshold(scenarios, probe_score)
            for stype in ["all_benign", "single_attack", "attack_recover", "escalation"]:
                type_scenarios = [s for s in scenarios if s["type"] == stype]
                if not type_scenarios:
                    continue

                probe_correct = sum(
                    1 for s in type_scenarios
                    if (probe_score(s["vecs"]).max() > probe_t) == (s["label"] == 1)
                )
                geo_correct = sum(
                    1 for s in type_scenarios
                    if (best_geo_fn(s["vecs"]).max() > best_geo_t) == (s["label"] == 1)
                )
                total = len(type_scenarios)
                print(f"    {stype:<20} Probe: {probe_correct}/{total}  "
                      f"{best_geo_name}: {geo_correct}/{total}")

    print(f"\n{'='*70}")
    print("If any geometric method beats probe-only, holonomy has value.")
    print("If none do, the concept needs fundamental rethinking.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
