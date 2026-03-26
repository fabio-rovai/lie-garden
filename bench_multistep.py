from __future__ import annotations
"""
GravRail Multi-Step Combined Detector Benchmark
=================================================
Tests the FULL combined detector (directional + holonomy) on TensorTrust,
not just the single-step directional probe.

Insight being tested:
- Per-step directional probe plateaus at F1≈0.72 on TensorTrust
- The combined detector (directional + holonomy scar) should perform better
- Training on 3-5 step windows instead of individual messages may improve detection
- The algebra mapping conditioned on group state adds contextual signal

This benchmark simulates realistic multi-step conversation sequences:
  benign context → attack injection → potential recovery
and measures whether the combined geometric signals detect what
the per-step probe misses.
"""
import numpy as np
import json
import time
from pathlib import Path
from scipy.linalg import expm
from dataclasses import dataclass, field


def vec_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tensor_trust(data_dir="/tmp/tensor-trust-data"):
    p = Path(data_dir) / "benchmarks" / "hijacking-robustness" / "v1" / "hijacking_robustness_dataset.jsonl"
    attacks, defenses = [], []
    with open(p) as f:
        for line in f:
            case = json.loads(line)
            attack_text = case.get("attack", "")
            if attack_text and len(attack_text.strip()) > 10:
                attacks.append(attack_text)
            pre = case.get("pre_prompt", "")
            if pre and len(pre.strip()) > 10:
                defenses.append(pre)
    return defenses, attacks


def load_injecagent(data_dir="/tmp/InjecAgent/data"):
    p = Path(data_dir)
    attacks, benign = [], []
    for fname in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(p / fname) as f:
            cases = json.load(f)
        for case in cases:
            attacks.append(case["Attacker Instruction"])
    with open(p / "user_cases.jsonl") as f:
        for line in f:
            case = json.loads(line)
            benign.append(case["User Instruction"])
    return benign, attacks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model2vec():
    from model2vec import StaticModel
    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    def embed(texts):
        return np.array(model.encode(texts))
    return embed


# ---------------------------------------------------------------------------
# Core detectors
# ---------------------------------------------------------------------------

@dataclass
class DirectionalProbe:
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    separation: float = 0.0
    proj: np.ndarray | None = None
    dim: int = 0

    def train(self, benign_vecs, harmful_vecs, dim=None):
        full_dim = benign_vecs.shape[1]
        dim = dim or full_dim
        self.dim = dim

        if dim < full_dim:
            np.random.seed(42)
            self.proj = np.random.randn(full_dim, dim) / np.sqrt(dim)
            b = benign_vecs @ self.proj
            h = harmful_vecs @ self.proj
        else:
            self.proj = None
            b = benign_vecs
            h = harmful_vecs

        diff = h.mean(axis=0) - b.mean(axis=0)
        norm = np.linalg.norm(diff)
        self.direction = diff / norm if norm > 1e-12 else diff
        b_scores = b @ self.direction
        h_scores = h @ self.direction
        self.threshold = (b_scores.mean() + h_scores.mean()) / 2.0
        self.separation = h_scores.mean() - b_scores.mean()

    def score(self, vec):
        if self.proj is not None:
            vec = vec @ self.proj
        return float(np.dot(vec, self.direction))

    def is_harmful(self, vec):
        return self.score(vec) > self.threshold


@dataclass
class HolonomyTracker:
    n: int = 4
    state: np.ndarray = field(default_factory=lambda: np.eye(4))
    ref_state: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: float = 0.1

    def reset(self):
        self.state = np.eye(self.n)

    def _vec_to_skew(self, vec):
        n = self.n
        n_coeffs = n * (n - 1) // 2
        coeffs = vec[:n_coeffs] if len(vec) >= n_coeffs else np.resize(vec, n_coeffs)
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = coeffs[idx] * self.scale
                A[j, i] = -coeffs[idx] * self.scale
                idx += 1
        return A

    def calibrate(self, benign_vecs):
        self.ref_state = np.eye(self.n)
        for vec in benign_vecs:
            A = self._vec_to_skew(vec)
            self.ref_state = self.ref_state @ expm(A)

    def step(self, vec):
        A = self._vec_to_skew(vec)
        self.state = self.state @ expm(A)
        return self.scar

    @property
    def scar(self):
        return float(np.linalg.norm(self.state - self.ref_state, 'fro'))


@dataclass
class ContextualHolonomyTracker(HolonomyTracker):
    """Holonomy tracker where the algebra mapping is conditioned on group state.

    Instead of a static mapping text→skew, the mapping rotates the embedding
    by the current state before projecting to the Lie algebra. This means the
    same text produces different algebra elements depending on conversation
    history — making it harder for an attacker to predict and cancel.
    """

    def _vec_to_skew(self, vec):
        n = self.n
        n_coeffs = n * (n - 1) // 2
        coeffs = vec[:n_coeffs] if len(vec) >= n_coeffs else np.resize(vec, n_coeffs)

        # Contextual conditioning: rotate coefficients by current state
        # Extract upper-triangle of state as modulation weights
        state_flat = []
        for i in range(n):
            for j in range(i + 1, n):
                state_flat.append(self.state[i, j])
        state_flat = np.array(state_flat)
        if len(state_flat) == len(coeffs):
            # Modulate: coefficients are shifted by state context
            coeffs = coeffs * (1.0 + 0.5 * np.tanh(state_flat))

        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = coeffs[idx] * self.scale
                A[j, i] = -coeffs[idx] * self.scale
                idx += 1
        return A


# ---------------------------------------------------------------------------
# Sequence-level probe (trains on 3-5 step windows)
# ---------------------------------------------------------------------------

@dataclass
class SequenceProbe:
    """Directional probe trained on projected step sequences, not individual messages.

    For each window of W steps, we project each step through the group (expm),
    then flatten the resulting state trajectory into a feature vector.
    This captures multi-step patterns that per-message probes miss.
    """
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    separation: float = 0.0
    window_size: int = 3
    so_n: int = 4
    scale: float = 0.1

    def _sequence_to_features(self, vecs):
        """Convert a sequence of embeddings to a group trajectory feature vector."""
        n = self.so_n
        n_coeffs = n * (n - 1) // 2
        state = np.eye(n)
        features = []

        for vec in vecs:
            coeffs = vec[:n_coeffs] if len(vec) >= n_coeffs else np.resize(vec, n_coeffs)
            A = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    A[i, j] = coeffs[idx] * self.scale
                    A[j, i] = -coeffs[idx] * self.scale
                    idx += 1
            state = state @ expm(A)
            # Extract upper triangle as features (captures trajectory shape)
            upper = []
            for i in range(n):
                for j in range(i, n):
                    upper.append(state[i, j])
            features.extend(upper)

        return np.array(features)

    def train(self, benign_sequences, harmful_sequences):
        """Train on labeled sequences (each is a list of embedding vectors)."""
        b_features = np.array([self._sequence_to_features(seq) for seq in benign_sequences])
        h_features = np.array([self._sequence_to_features(seq) for seq in harmful_sequences])

        diff = h_features.mean(axis=0) - b_features.mean(axis=0)
        norm = np.linalg.norm(diff)
        self.direction = diff / norm if norm > 1e-12 else diff
        b_scores = b_features @ self.direction
        h_scores = h_features @ self.direction
        self.threshold = (b_scores.mean() + h_scores.mean()) / 2.0
        self.separation = h_scores.mean() - b_scores.mean()

    def score_sequence(self, vecs):
        features = self._sequence_to_features(vecs)
        return float(np.dot(features, self.direction))

    def is_harmful(self, vecs):
        return self.score_sequence(vecs) > self.threshold


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------

def evaluate_single_step(probe, test_vecs, test_labels):
    """Per-step directional probe only (baseline)."""
    tp = fp = tn = fn = 0
    for vec, label in zip(test_vecs, test_labels):
        predicted = probe.is_harmful(vec)
        actual = label == "attack"
        if actual and predicted: tp += 1
        elif actual and not predicted: fn += 1
        elif not actual and predicted: fp += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_combined(probe, tracker_class, test_sequences, scar_threshold=0.3,
                      dir_weight=0.6, hol_weight=0.4, calibration_vecs=None):
    """Combined detector: directional + holonomy on conversation sequences.

    Each test_sequence is (list_of_vecs, label) where label is for the sequence.
    A sequence is 'attack' if any step in it is an attack.
    """
    tp = fp = tn = fn = 0
    details = []

    for vecs, label in test_sequences:
        tracker = tracker_class(n=4)
        if calibration_vecs is not None:
            tracker.calibrate(calibration_vecs[:5])
        else:
            tracker.reset()

        max_dir_score = -float('inf')
        any_harmful = False

        for vec in vecs:
            dir_score = probe.score(vec)
            if dir_score > max_dir_score:
                max_dir_score = dir_score
            if probe.is_harmful(vec):
                any_harmful = True
            tracker.step(vec)

        scar = tracker.scar
        dir_norm = max(0.0, (max_dir_score - probe.threshold) / max(probe.separation, 1e-6))
        scar_norm = scar / max(scar_threshold, 1e-6)
        combined = dir_weight * dir_norm + hol_weight * scar_norm

        # Combined decision: either directional OR holonomy flags it
        detected = any_harmful or scar > scar_threshold

        actual = label == "attack"
        if actual and detected: tp += 1
        elif actual and not detected: fn += 1
        elif not actual and detected: fp += 1
        else: tn += 1

        details.append({
            "label": label, "detected": detected,
            "max_dir": max_dir_score, "scar": scar,
            "combined": combined, "any_harmful": any_harmful,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "details": details}


def evaluate_sequence_probe(seq_probe, test_sequences):
    """Evaluate the sequence-level probe on conversation windows."""
    tp = fp = tn = fn = 0
    for vecs, label in test_sequences:
        predicted = seq_probe.is_harmful(vecs)
        actual = label == "attack"
        if actual and predicted: tp += 1
        elif actual and not predicted: fn += 1
        elif not actual and predicted: fp += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_conversation_sequences(benign_vecs, attack_vecs, window=3, n_sequences=200):
    """Build realistic conversation sequences for evaluation.

    Creates three types:
    1. Pure benign: [benign, benign, benign] — should NOT trigger
    2. Attack embedded: [benign, attack, benign] — SHOULD trigger
    3. Multi-attack: [attack, attack, benign] — SHOULD trigger (easy)
    """
    np.random.seed(42)
    sequences = []
    n_b = len(benign_vecs)
    n_a = len(attack_vecs)

    per_type = n_sequences // 3

    # Type 1: Pure benign
    for _ in range(per_type):
        idxs = np.random.choice(n_b, window, replace=True)
        vecs = [benign_vecs[i] for i in idxs]
        sequences.append((vecs, "benign"))

    # Type 2: Attack embedded in benign context (hardest case)
    for _ in range(per_type):
        attack_idx = np.random.randint(n_a)
        benign_idxs = np.random.choice(n_b, window - 1, replace=True)
        # Insert attack at random position
        pos = np.random.randint(0, window)
        vecs = []
        b_iter = iter(benign_idxs)
        for i in range(window):
            if i == pos:
                vecs.append(attack_vecs[attack_idx])
            else:
                vecs.append(benign_vecs[next(b_iter)])
        sequences.append((vecs, "attack"))

    # Type 3: Multi-attack burst
    for _ in range(n_sequences - 2 * per_type):
        n_attacks = np.random.randint(2, min(window, 4) + 1)
        attack_idxs = np.random.choice(n_a, n_attacks, replace=True)
        benign_idxs = np.random.choice(n_b, window - n_attacks, replace=True)
        all_vecs = list(attack_vecs[attack_idxs]) + list(benign_vecs[benign_idxs])
        np.random.shuffle(all_vecs)
        sequences.append((list(all_vecs), "attack"))

    return sequences


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("  MULTI-STEP COMBINED DETECTOR BENCHMARK")
    print("  Testing: directional probe vs combined (dir + holonomy) vs sequence probe")
    print("=" * 90)

    # Load data
    print("\nLoading datasets...")
    tt_benign, tt_attacks = load_tensor_trust()
    ij_benign, ij_attacks = load_injecagent()
    print(f"  TensorTrust: {len(tt_benign)} benign, {len(tt_attacks)} attacks")
    print(f"  InjecAgent: {len(ij_benign)} benign, {len(ij_attacks)} attacks")

    # Load embeddings
    print("\nLoading Model2Vec...")
    embed_fn = load_model2vec()

    for dataset_name, benign_texts, attack_texts in [
        ("TensorTrust", tt_benign, tt_attacks),
        ("InjecAgent", ij_benign, ij_attacks),
    ]:
        print(f"\n{'=' * 90}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'=' * 90}")

        # Embed
        all_texts = benign_texts + attack_texts
        all_vecs = embed_fn(all_texts)
        benign_vecs = all_vecs[:len(benign_texts)]
        attack_vecs = all_vecs[len(benign_texts):]
        full_dim = all_vecs.shape[1]

        # Train/test split
        np.random.seed(42)
        n_train_b = max(20, int(len(benign_vecs) * 0.8))
        n_train_a = max(20, int(len(attack_vecs) * 0.8))
        idx_b = np.random.permutation(len(benign_vecs))
        idx_a = np.random.permutation(len(attack_vecs))

        train_b = benign_vecs[idx_b[:n_train_b]]
        test_b = benign_vecs[idx_b[n_train_b:]]
        train_a = attack_vecs[idx_a[:n_train_a]]
        test_a = attack_vecs[idx_a[n_train_a:]]

        if len(test_b) < 10:
            test_b = train_b[-20:]
        if len(test_a) < 10:
            test_a = train_a[-20:]

        print(f"  Train: {len(train_b)} benign, {len(train_a)} attack")
        print(f"  Test: {len(test_b)} benign, {len(test_a)} attack")

        # --- Signal 1: Per-step directional probe (baseline) ---
        for dim in [30, 50, full_dim]:
            if dim > full_dim:
                continue

            probe = DirectionalProbe()
            probe.train(train_b, train_a, dim=dim)

            test_vecs = np.vstack([test_b, test_a])
            test_labels = ["benign"] * len(test_b) + ["attack"] * len(test_a)
            r = evaluate_single_step(probe, test_vecs, test_labels)
            print(f"\n  [1] PER-STEP DIRECTIONAL (dim={dim}): "
                  f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                  f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

        # --- For combined tests, use full dim ---
        probe = DirectionalProbe()
        probe.train(train_b, train_a, dim=full_dim)

        # Build conversation sequences for multi-step eval
        for window in [3, 5]:
            sequences = build_conversation_sequences(
                test_b, test_a, window=window, n_sequences=300
            )
            n_benign_seq = sum(1 for _, l in sequences if l == "benign")
            n_attack_seq = sum(1 for _, l in sequences if l == "attack")
            print(f"\n  Conversation sequences (window={window}): "
                  f"{n_benign_seq} benign, {n_attack_seq} attack")

            # --- Signal 2: Combined detector (directional + holonomy) ---
            for scar_threshold in [0.1, 0.3, 0.5]:
                r = evaluate_combined(
                    probe, HolonomyTracker, sequences,
                    scar_threshold=scar_threshold,
                    calibration_vecs=train_b,
                )
                print(f"  [2] COMBINED dir+hol (w={window}, scar_t={scar_threshold}): "
                      f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                      f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

                # Show how many were caught by holonomy but missed by directional
                hol_only = sum(1 for d in r['details']
                               if d['label'] == 'attack' and d['detected']
                               and not d['any_harmful'] and d['scar'] > scar_threshold)
                dir_only = sum(1 for d in r['details']
                               if d['label'] == 'attack' and d['detected']
                               and d['any_harmful'] and d['scar'] <= scar_threshold)
                both = sum(1 for d in r['details']
                           if d['label'] == 'attack' and d['detected']
                           and d['any_harmful'] and d['scar'] > scar_threshold)
                missed = sum(1 for d in r['details']
                             if d['label'] == 'attack' and not d['detected'])
                if hol_only > 0:
                    print(f"       ↳ Caught by holonomy ONLY (dir missed): {hol_only}")
                if both > 0:
                    print(f"       ↳ Caught by BOTH signals: {both}")
                if missed > 0:
                    print(f"       ↳ Missed by combined: {missed}")

            # --- Signal 2b: Contextual holonomy ---
            r = evaluate_combined(
                probe, ContextualHolonomyTracker, sequences,
                scar_threshold=0.3,
                calibration_vecs=train_b,
            )
            print(f"  [2b] CONTEXTUAL holonomy (w={window}, scar_t=0.3): "
                  f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                  f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
            hol_only = sum(1 for d in r['details']
                           if d['label'] == 'attack' and d['detected']
                           and not d['any_harmful'] and d['scar'] > 0.3)
            if hol_only > 0:
                print(f"       ↳ Caught by contextual holonomy ONLY: {hol_only}")

            # --- Signal 3: Sequence-level probe ---
            # Build training sequences
            train_seqs_benign = build_conversation_sequences(
                train_b, train_b[:20],  # benign-only for "benign" sequences
                window=window, n_sequences=100
            )
            # Override: make all benign
            train_benign_seqs = [(vecs, "benign") for vecs, _ in train_seqs_benign[:50]]
            train_harmful_seqs = build_conversation_sequences(
                train_b, train_a, window=window, n_sequences=50
            )
            train_harmful_seqs = [(vecs, label) for vecs, label in train_harmful_seqs if label == "attack"]

            if len(train_harmful_seqs) >= 10 and len(train_benign_seqs) >= 10:
                seq_probe = SequenceProbe(window_size=window)
                seq_probe.train(
                    [vecs for vecs, _ in train_benign_seqs],
                    [vecs for vecs, _ in train_harmful_seqs],
                )
                r = evaluate_sequence_probe(seq_probe, sequences)
                print(f"  [3] SEQUENCE PROBE (w={window}): "
                      f"F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                      f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

    # --- Detailed TensorTrust analysis: which attacks does holonomy catch? ---
    print(f"\n\n{'=' * 90}")
    print("  DETAILED ANALYSIS: TensorTrust attacks missed by directional, caught by holonomy")
    print(f"{'=' * 90}")

    all_tt = embed_fn(tt_benign + tt_attacks)
    tt_b_vecs = all_tt[:len(tt_benign)]
    tt_a_vecs = all_tt[len(tt_benign):]

    probe = DirectionalProbe()
    probe.train(tt_b_vecs[:int(len(tt_b_vecs)*0.8)], tt_a_vecs[:int(len(tt_a_vecs)*0.8)], dim=full_dim)

    # Find attacks missed by directional probe
    missed_by_dir = []
    for i, (vec, text) in enumerate(zip(tt_a_vecs, tt_attacks)):
        if not probe.is_harmful(vec):
            missed_by_dir.append((i, text, probe.score(vec)))

    print(f"\n  Total TensorTrust attacks: {len(tt_attacks)}")
    print(f"  Missed by directional probe: {len(missed_by_dir)} ({100*len(missed_by_dir)/len(tt_attacks):.1f}%)")

    if missed_by_dir:
        print(f"\n  Sample attacks that EVADE per-step detection:")
        for idx, text, score in missed_by_dir[:10]:
            short = text[:100] + "..." if len(text) > 100 else text
            print(f"    [{idx}] score={score:.4f} (thresh={probe.threshold:.4f}): {short}")

    # Now test: do these missed attacks get caught in multi-step context?
    caught_by_holonomy = 0
    for idx, text, dir_score in missed_by_dir:
        # Simulate: 2 benign → 1 attack → 1 benign
        b_idxs = np.random.choice(len(tt_b_vecs), 3, replace=True)
        seq_vecs = [tt_b_vecs[b_idxs[0]], tt_b_vecs[b_idxs[1]],
                     tt_a_vecs[idx], tt_b_vecs[b_idxs[2]]]

        tracker = HolonomyTracker(n=4)
        tracker.calibrate(tt_b_vecs[:5])
        for v in seq_vecs:
            tracker.step(v)

        if tracker.scar > 0.3:
            caught_by_holonomy += 1

    if missed_by_dir:
        print(f"\n  Of {len(missed_by_dir)} dir-missed attacks:")
        print(f"    Caught by holonomy in multi-step context: {caught_by_holonomy} "
              f"({100*caught_by_holonomy/len(missed_by_dir):.1f}%)")
        print(f"    Still missed by combined: {len(missed_by_dir) - caught_by_holonomy}")
        total_caught = (len(tt_attacks) - len(missed_by_dir)) + caught_by_holonomy
        combined_recall = total_caught / len(tt_attacks)
        print(f"\n  COMBINED recall (dir + holonomy): {combined_recall:.3f} "
              f"(vs directional-only: {(len(tt_attacks) - len(missed_by_dir)) / len(tt_attacks):.3f})")


if __name__ == "__main__":
    main()
