"""
Honest Ablation: Probe-Only vs Probe+Holonomy on Real Datasets
================================================================
Tests whether holonomy adds any detection power over the directional probe alone.
Uses REAL data from InjecAgent and TensorTrust (not fabricated samples).

This directly addresses the reviewer's criticism: "The holonomy adds nothing."
"""
import json
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Components (same as arena_agent.py / gravrail.py)
# ---------------------------------------------------------------------------

@dataclass
class DirectionalProbe:
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    separation: float = 0.0

    @staticmethod
    def _hash_encode(texts: list[str], dim: int = 256) -> np.ndarray:
        vecs = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.array(
                [int.from_bytes(h[i:i + 2], "little") / 65535.0
                 for i in range(0, min(len(h), dim * 2), 2)]
            )
            if len(vec) < dim:
                vec = np.resize(vec, dim)
            vecs.append(vec)
        return np.array(vecs)

    _model = None  # class-level cache

    @staticmethod
    def _encode(texts: list[str]) -> np.ndarray:
        try:
            from model2vec import StaticModel
            if DirectionalProbe._model is None:
                DirectionalProbe._model = StaticModel.from_pretrained(
                    "minishlab/potion-base-32M"
                )
            return np.array(DirectionalProbe._model.encode(texts))
        except ImportError:
            return DirectionalProbe._hash_encode(texts)

    def train(self, benign: list[str], harmful: list[str]) -> None:
        all_vecs = self._encode(benign + harmful)
        b = all_vecs[:len(benign)]
        h = all_vecs[len(benign):]
        diff = h.mean(axis=0) - b.mean(axis=0)
        norm = np.linalg.norm(diff)
        self.direction = diff / norm if norm > 1e-12 else diff
        self.threshold = (b @ self.direction).mean() + (h @ self.direction).mean()
        self.threshold /= 2.0
        self.separation = (h @ self.direction).mean() - (b @ self.direction).mean()

    def score_text(self, text: str) -> float:
        vec = self._encode([text])[0]
        return float(np.dot(vec, self.direction))

    def is_harmful(self, text: str) -> bool:
        return self.score_text(text) > self.threshold


@dataclass
class HolonomyTracker:
    n: int = 4
    state: np.ndarray = field(default_factory=lambda: np.eye(4))
    ref_state: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.state = np.eye(self.n)
        self.ref_state = np.eye(self.n)

    def _text_to_skew(self, text: str, probe: DirectionalProbe) -> np.ndarray:
        vec = probe._encode([text])[0]
        n = self.n
        n_coeffs = n * (n - 1) // 2
        coeffs = vec[:n_coeffs] if len(vec) >= n_coeffs else np.resize(vec, n_coeffs)
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = coeffs[idx] * 0.1
                A[j, i] = -coeffs[idx] * 0.1
                idx += 1
        return A

    def step(self, text: str, probe: DirectionalProbe) -> float:
        A = self._text_to_skew(text, probe)
        self.state = self.state @ expm(A)
        return self.scar

    def calibrate(self, benign_texts: list[str], probe: DirectionalProbe) -> None:
        self.ref_state = np.eye(self.n)
        for text in benign_texts:
            A = self._text_to_skew(text, probe)
            self.ref_state = self.ref_state @ expm(A)

    @property
    def scar(self) -> float:
        return float(np.linalg.norm(self.state - self.ref_state, "fro"))


# ---------------------------------------------------------------------------
# Load REAL datasets
# ---------------------------------------------------------------------------

def load_injecagent():
    """Load real InjecAgent data."""
    benign = []
    harmful = []

    # Benign: user instructions
    with open("/tmp/InjecAgent/data/user_cases.jsonl") as f:
        for line in f:
            d = json.loads(line)
            benign.append(d["User Instruction"])

    # Harmful: attacker instructions (both DH and DS)
    for fname in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
        with open(f"/tmp/InjecAgent/data/{fname}") as f:
            for line in f:
                d = json.loads(line)
                harmful.append(d["Attacker Instruction"])

    # Also load test cases for more samples
    for fname in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(f"/tmp/InjecAgent/data/{fname}") as f:
            cases = json.load(f)
            for c in cases:
                # The injected instruction embedded in the tool response
                harmful.append(c["Attacker Instruction"])

    print(f"InjecAgent: {len(benign)} benign, {len(harmful)} harmful")
    return benign, harmful


def load_tensortrust():
    """Load real TensorTrust data."""
    benign = []
    harmful = []

    # Benign: pre_prompts (defender system prompts) — legitimate instructions
    # Harmful: attacks (injection attempts)
    with open("/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            benign.append(d["pre_prompt"])
            harmful.append(d["attack"])

    # Also add extraction attacks
    with open("/tmp/tensor-trust-data/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["pre_prompt"] not in benign:
                benign.append(d["pre_prompt"])
            harmful.append(d["attack"])

    print(f"TensorTrust: {len(benign)} benign, {len(harmful)} harmful")
    return benign, harmful


# ---------------------------------------------------------------------------
# Test 1: Per-step detection (probe-only vs probe+holonomy)
# ---------------------------------------------------------------------------

def test_per_step(probe, tracker, test_texts, test_labels, scar_threshold=0.3):
    """Compare probe-only vs probe+holonomy on single messages."""
    probe_preds = []
    combined_preds = []

    for text in test_texts:
        p_harmful = probe.is_harmful(text)
        scar = tracker.step(text, probe)

        probe_preds.append(1 if p_harmful else 0)
        combined_preds.append(1 if (p_harmful or scar > scar_threshold) else 0)

    probe_acc = accuracy_score(test_labels, probe_preds)
    probe_prec = precision_score(test_labels, probe_preds, zero_division=0)
    probe_rec = recall_score(test_labels, probe_preds, zero_division=0)
    probe_f1 = f1_score(test_labels, probe_preds, zero_division=0)

    comb_acc = accuracy_score(test_labels, combined_preds)
    comb_prec = precision_score(test_labels, combined_preds, zero_division=0)
    comb_rec = recall_score(test_labels, combined_preds, zero_division=0)
    comb_f1 = f1_score(test_labels, combined_preds, zero_division=0)

    # Check: how many did holonomy catch that probe missed?
    holonomy_only = sum(
        1 for p, c, l in zip(probe_preds, combined_preds, test_labels)
        if c == 1 and p == 0 and l == 1
    )
    holonomy_fp = sum(
        1 for p, c, l in zip(probe_preds, combined_preds, test_labels)
        if c == 1 and p == 0 and l == 0
    )

    return {
        "probe": {"acc": probe_acc, "prec": probe_prec, "rec": probe_rec, "f1": probe_f1},
        "combined": {"acc": comb_acc, "prec": comb_prec, "rec": comb_rec, "f1": comb_f1},
        "holonomy_only_tp": holonomy_only,
        "holonomy_only_fp": holonomy_fp,
    }


# ---------------------------------------------------------------------------
# Test 2: Multi-step attack-and-recover
# ---------------------------------------------------------------------------

def test_multi_step(probe, benign_texts, harmful_texts, scar_threshold=0.3):
    """
    Simulate attack-and-recover: benign → attack → benign recovery.
    Test whether the holonomy scar persists after recovery.
    """
    results = []

    for i in range(min(50, len(harmful_texts))):
        tracker = HolonomyTracker(n=4)
        tracker.calibrate(benign_texts[:5], probe)

        # Phase 1: benign messages
        for b in benign_texts[:3]:
            tracker.step(b, probe)
        scar_before = tracker.scar

        # Phase 2: attack
        attack_text = harmful_texts[i]
        tracker.step(attack_text, probe)
        scar_during = tracker.scar
        probe_caught = probe.is_harmful(attack_text)

        # Phase 3: recovery with benign messages
        for b in benign_texts[:5]:
            tracker.step(b, probe)
        scar_after = tracker.scar

        # Did the scar persist?
        scar_persisted = scar_after > scar_threshold
        probe_still_flags = False  # probe is per-step, can't flag after recovery

        results.append({
            "scar_before": scar_before,
            "scar_during": scar_during,
            "scar_after": scar_after,
            "scar_persisted": scar_persisted,
            "probe_caught_attack": probe_caught,
        })

    persist_count = sum(r["scar_persisted"] for r in results)
    probe_caught = sum(r["probe_caught_attack"] for r in results)
    total = len(results)

    return {
        "total_scenarios": total,
        "probe_caught_during_attack": probe_caught,
        "scar_persisted_after_recovery": persist_count,
        "scar_persistence_rate": persist_count / total if total > 0 else 0,
        "avg_scar_before": np.mean([r["scar_before"] for r in results]),
        "avg_scar_during": np.mean([r["scar_during"] for r in results]),
        "avg_scar_after": np.mean([r["scar_after"] for r in results]),
    }


# ---------------------------------------------------------------------------
# Test 3: Can an adversary erase the scar?
# ---------------------------------------------------------------------------

def test_scar_erasability(probe, benign_texts, harmful_texts):
    """
    Test reviewer's claim: attacker can compute state^(-1) * ref_state
    to erase the scar.
    """
    tracker = HolonomyTracker(n=4)
    tracker.calibrate(benign_texts[:5], probe)

    # Process some benign then an attack
    for b in benign_texts[:3]:
        tracker.step(b, probe)
    attack = harmful_texts[0]
    tracker.step(attack, probe)
    scar_after_attack = tracker.scar

    # The adversary's theoretical move: compute inverse
    # state^(-1) * ref_state gives the "correction" needed
    correction = np.linalg.inv(tracker.state) @ tracker.ref_state

    # But can the adversary actually execute this?
    # They'd need to craft text that maps to exactly the right skew matrix
    # such that expm(A_crafted) = correction
    # Let's check: is correction orthogonal?
    orth_error = np.linalg.norm(correction @ correction.T - np.eye(4))

    # Try: apply the exact correction
    tracker.state = tracker.state @ correction
    scar_after_correction = tracker.scar

    return {
        "scar_after_attack": scar_after_attack,
        "scar_after_perfect_correction": scar_after_correction,
        "correction_is_orthogonal": orth_error < 1e-10,
        "conclusion": (
            "YES, the scar IS erasable with perfect state knowledge. "
            "The reviewer is correct: non-commutativity does NOT prevent "
            "erasure when the adversary knows (or can infer) the state."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("HONEST ABLATION: Does holonomy add value over probe-only?")
    print("Using REAL InjecAgent and TensorTrust data")
    print("=" * 70)

    # Load real datasets
    inj_benign, inj_harmful = load_injecagent()
    tt_benign, tt_harmful = load_tensortrust()

    for dataset_name, benign, harmful in [
        ("InjecAgent", inj_benign, inj_harmful),
        ("TensorTrust", tt_benign, tt_harmful),
    ]:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        # Split: train on first half, test on second half
        n_b_train = max(5, len(benign) // 2)
        n_h_train = max(5, len(harmful) // 2)
        train_benign = benign[:n_b_train]
        train_harmful = harmful[:n_h_train]
        test_benign = benign[n_b_train:]
        test_harmful = harmful[n_h_train:]

        # Ensure we have test data
        if not test_benign:
            test_benign = benign[:n_b_train]  # reuse if too small
        if not test_harmful:
            test_harmful = harmful[:n_h_train]

        # Train probe
        probe = DirectionalProbe()
        probe.train(train_benign, train_harmful)
        print(f"\nProbe separation: {probe.separation:.4f}")

        # Create tracker
        tracker = HolonomyTracker(n=4)
        tracker.calibrate(train_benign[:5], probe)

        # Test 1: Per-step detection
        print(f"\n--- Test 1: Per-Step Detection ---")
        test_texts = test_benign + test_harmful
        test_labels = [0] * len(test_benign) + [1] * len(test_harmful)
        result = test_per_step(probe, tracker, test_texts, test_labels)

        print(f"  Probe-only:      Acc={result['probe']['acc']:.3f}  "
              f"Prec={result['probe']['prec']:.3f}  "
              f"Rec={result['probe']['rec']:.3f}  "
              f"F1={result['probe']['f1']:.3f}")
        print(f"  Probe+Holonomy:  Acc={result['combined']['acc']:.3f}  "
              f"Prec={result['combined']['prec']:.3f}  "
              f"Rec={result['combined']['rec']:.3f}  "
              f"F1={result['combined']['f1']:.3f}")
        print(f"  Holonomy-only TPs: {result['holonomy_only_tp']}  "
              f"Holonomy-only FPs: {result['holonomy_only_fp']}")

        delta_f1 = result['combined']['f1'] - result['probe']['f1']
        if abs(delta_f1) < 0.01:
            print(f"  >> VERDICT: Holonomy adds NOTHING (delta F1 = {delta_f1:+.4f})")
        elif delta_f1 > 0.01:
            print(f"  >> VERDICT: Holonomy HELPS (delta F1 = {delta_f1:+.4f})")
        else:
            print(f"  >> VERDICT: Holonomy HURTS (delta F1 = {delta_f1:+.4f})")

        # Test 2: Multi-step attack-and-recover
        print(f"\n--- Test 2: Multi-Step Attack-and-Recover ---")
        ms_result = test_multi_step(probe, test_benign, test_harmful)
        print(f"  Scenarios: {ms_result['total_scenarios']}")
        print(f"  Probe caught during attack: {ms_result['probe_caught_during_attack']}")
        print(f"  Scar persisted after recovery: {ms_result['scar_persisted_after_recovery']}")
        print(f"  Persistence rate: {ms_result['scar_persistence_rate']:.1%}")
        print(f"  Avg scar: before={ms_result['avg_scar_before']:.4f}  "
              f"during={ms_result['avg_scar_during']:.4f}  "
              f"after={ms_result['avg_scar_after']:.4f}")

        if ms_result['scar_persistence_rate'] < 0.5:
            print(f"  >> VERDICT: Scar does NOT reliably persist (rate={ms_result['scar_persistence_rate']:.1%})")
        else:
            print(f"  >> VERDICT: Scar persists in {ms_result['scar_persistence_rate']:.1%} of scenarios")

    # Test 3: Scar erasability (dataset-independent)
    print(f"\n{'=' * 70}")
    print(f"  Test 3: Scar Erasability (Reviewer's Claim)")
    print(f"{'=' * 70}")
    probe_er = DirectionalProbe()
    probe_er.train(inj_benign, inj_harmful[:len(inj_benign)])
    er_result = test_scar_erasability(probe_er, inj_benign, inj_harmful)
    print(f"  Scar after attack: {er_result['scar_after_attack']:.6f}")
    print(f"  Scar after perfect correction: {er_result['scar_after_perfect_correction']:.2e}")
    print(f"  Correction is orthogonal: {er_result['correction_is_orthogonal']}")
    print(f"  >> {er_result['conclusion']}")

    print(f"\n{'=' * 70}")
    print(f"  OVERALL CONCLUSION")
    print(f"{'=' * 70}")
    print(f"  This test uses REAL data from InjecAgent and TensorTrust.")
    print(f"  If holonomy adds nothing, the eval should not claim it does.")
    print(f"  If the scar is erasable, the 'non-erasability' claim is false.")


if __name__ == "__main__":
    main()
