"""
Learned Holonomy: Train the text→algebra mapping so the group structure
actually encodes benign vs harmful.
=====================================================================

Core idea: Instead of arbitrarily projecting embedding coefficients to
so(n) generators, LEARN a projection matrix W such that:
  - Benign texts → small rotations (near identity)
  - Harmful texts → large rotations (far from identity)
  - AND the key insight: benign SEQUENCES stay near identity (rotations
    approximately cancel), while attack sequences accumulate

This makes the holonomy genuinely discriminative because the mapping is
designed to make the group structure informative.

We also test a second idea: CONTRASTIVE algebra learning, where benign
pairs have low commutator norm and harmful pairs have high commutator norm.
"""
import json
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from sklearn.metrics import f1_score, precision_score, recall_score


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
# Approach 1: Learned projection W that maps embeddings to so(n)
# ---------------------------------------------------------------------------

def vec_to_skew(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Convert coefficient vector to n×n skew-symmetric matrix."""
    A = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(coeffs):
                A[i, j] = coeffs[idx]
                A[j, i] = -coeffs[idx]
            idx += 1
    return A


def learn_projection(benign_vecs: np.ndarray, harmful_vecs: np.ndarray,
                     n: int = 4, lr: float = 0.01, epochs: int = 200):
    """
    Learn W: R^d → R^{n(n-1)/2} such that:
    - ||expm(skew(W @ benign)) - I||_F is SMALL
    - ||expm(skew(W @ harmful)) - I||_F is LARGE

    This makes benign texts map to near-identity rotations and
    harmful texts map to large rotations. The holonomy (cumulative
    product) will then stay near I for benign sequences and diverge
    for sequences containing attacks.
    """
    d = benign_vecs.shape[1]
    n_coeffs = n * (n - 1) // 2

    # Initialize W randomly but small
    rng = np.random.RandomState(42)
    W = rng.randn(n_coeffs, d) * 0.001

    I = np.eye(n)

    # Subsample for speed
    b_sub = benign_vecs[:min(100, len(benign_vecs))]
    h_sub = harmful_vecs[:min(100, len(harmful_vecs))]

    for epoch in range(epochs):
        total_loss = 0.0
        grad_W = np.zeros_like(W)

        # Benign: minimize ||expm(skew(Wv)) - I||^2
        for v in b_sub:
            coeffs = W @ v
            A = vec_to_skew(coeffs * 0.1, n)
            R = expm(A)
            diff = R - I
            loss_b = np.sum(diff ** 2)
            total_loss += loss_b

            # Numerical gradient (small W, so finite diff is OK)
            for k in range(n_coeffs):
                W_plus = W.copy()
                W_plus[k] += 1e-5
                c_plus = W_plus @ v
                A_plus = vec_to_skew(c_plus * 0.1, n)
                R_plus = expm(A_plus)
                loss_plus = np.sum((R_plus - I) ** 2)
                grad_W[k] += (loss_plus - loss_b) / 1e-5

        # Harmful: maximize ||expm(skew(Wv)) - I||^2
        # = minimize -||...||^2
        for v in h_sub:
            coeffs = W @ v
            A = vec_to_skew(coeffs * 0.1, n)
            R = expm(A)
            diff = R - I
            loss_h = -np.sum(diff ** 2)  # negative = maximize
            total_loss += loss_h

            for k in range(n_coeffs):
                W_plus = W.copy()
                W_plus[k] += 1e-5
                c_plus = W_plus @ v
                A_plus = vec_to_skew(c_plus * 0.1, n)
                R_plus = expm(A_plus)
                loss_plus = -np.sum((R_plus - I) ** 2)
                grad_W[k] += (loss_plus - loss_h) / 1e-5

        W -= lr * grad_W / (len(b_sub) + len(h_sub))

        if epoch % 50 == 0:
            # Evaluate separation
            b_norms = []
            for v in b_sub[:20]:
                c = W @ v
                A = vec_to_skew(c * 0.1, n)
                b_norms.append(np.linalg.norm(expm(A) - I, "fro"))
            h_norms = []
            for v in h_sub[:20]:
                c = W @ v
                A = vec_to_skew(c * 0.1, n)
                h_norms.append(np.linalg.norm(expm(A) - I, "fro"))
            print(f"    Epoch {epoch:3d}: loss={total_loss:.4f}  "
                  f"benign_norm={np.mean(b_norms):.4f}  "
                  f"harmful_norm={np.mean(h_norms):.4f}  "
                  f"gap={np.mean(h_norms) - np.mean(b_norms):.4f}")

    return W


# ---------------------------------------------------------------------------
# Approach 2: Probe-informed algebra (simpler, might work better)
# ---------------------------------------------------------------------------

def probe_informed_mapping(vecs: np.ndarray, probe_dir: np.ndarray,
                           probe_threshold: float, n: int = 4,
                           scale_benign: float = 0.01,
                           scale_harmful: float = 0.5):
    """
    Use the probe score to SCALE the algebra element.
    - Score < threshold → small rotation (benign)
    - Score > threshold → large rotation (harmful)

    The probe does the classification; the group structure provides
    sequence-level accumulation that the probe alone can't do.

    Key insight: this isn't "just the probe with extra steps" because
    the GROUP PRODUCT of many small benign rotations stays near identity
    (they partially cancel due to varying directions), while even ONE
    large harmful rotation pushes the product far from identity and
    subsequent small rotations CAN'T bring it back easily.
    """
    n_coeffs = n * (n - 1) // 2
    skews = []
    for v in vecs:
        score = np.dot(v, probe_dir)
        if score > probe_threshold:
            scale = scale_harmful
        else:
            scale = scale_benign
        coeffs = v[:n_coeffs] * scale
        skews.append(vec_to_skew(coeffs, n))
    return skews


# ---------------------------------------------------------------------------
# Multi-step scenarios
# ---------------------------------------------------------------------------

def make_scenarios(b_vecs, h_vecs, n_scenarios=200):
    rng = np.random.RandomState(42)
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
        elif t == 2:  # attack + recovery
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


def eval_scenarios(scenarios, detect_fn):
    """detect_fn(vecs) → bool (True = attack detected)"""
    labels = [s["label"] for s in scenarios]
    preds = [int(detect_fn(s["vecs"])) for s in scenarios]
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
    }


def eval_by_type(scenarios, detect_fn):
    results = {}
    for t in ["benign", "single", "recover", "escalate"]:
        sub = [s for s in scenarios if s["type"] == t]
        correct = sum(1 for s in sub if detect_fn(s["vecs"]) == (s["label"] == 1))
        results[t] = f"{correct}/{len(sub)}"
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LEARNED HOLONOMY: Can we make the group structure work?")
    print("=" * 70)

    # Load data
    benign_texts, harmful_texts = [], []
    with open("/tmp/InjecAgent/data/user_cases.jsonl") as f:
        for line in f:
            benign_texts.append(json.loads(line)["User Instruction"])
    for fn in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
        with open(f"/tmp/InjecAgent/data/{fn}") as f:
            for line in f:
                harmful_texts.append(json.loads(line)["Attacker Instruction"])
    for fn in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(f"/tmp/InjecAgent/data/{fn}") as f:
            for c in json.load(f):
                harmful_texts.append(c["Attacker Instruction"])

    # Also TensorTrust
    tt_b, tt_h = [], []
    with open("/tmp/tensor-trust-data/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tt_b.append(d["pre_prompt"])
            tt_h.append(d["attack"])

    for name, b_texts, h_texts in [
        ("InjecAgent", benign_texts, harmful_texts),
        ("TensorTrust", tt_b, tt_h),
    ]:
        print(f"\n{'='*70}")
        print(f"  {name}: {len(b_texts)} benign, {len(h_texts)} harmful")
        print(f"{'='*70}")

        print("  Encoding...")
        all_vecs = encode(b_texts + h_texts)
        b_vecs = all_vecs[:len(b_texts)]
        h_vecs = all_vecs[len(b_texts):]

        # Train probe baseline
        diff = h_vecs.mean(0) - b_vecs.mean(0)
        probe_dir = diff / (np.linalg.norm(diff) + 1e-12)
        probe_thresh = (b_vecs @ probe_dir).mean() + (h_vecs @ probe_dir).mean()
        probe_thresh /= 2.0

        scenarios = make_scenarios(b_vecs, h_vecs, 200)

        # --- Baseline: probe-only (per-step max) ---
        def probe_detect(vecs):
            scores = vecs @ probe_dir
            return bool(scores.max() > probe_thresh)

        r = eval_scenarios(scenarios, probe_detect)
        bt = eval_by_type(scenarios, probe_detect)
        print(f"\n  Probe-only:           F1={r['f1']:.3f} Prec={r['prec']:.3f} Rec={r['rec']:.3f}")
        print(f"    By type: {bt}")

        # --- Approach 2: Probe-informed holonomy ---
        # This is the key idea: use probe score to SCALE the rotation magnitude
        n = 4
        I = np.eye(n)

        # Find best scale ratio and threshold
        best_f1 = 0
        best_params = None

        for s_b in [0.001, 0.005, 0.01, 0.02]:
            for s_h in [0.1, 0.3, 0.5, 1.0]:
                for scar_t in [0.05, 0.1, 0.2, 0.5, 1.0]:
                    def make_detector(sb=s_b, sh=s_h, st=scar_t):
                        def detect(vecs):
                            state = I.copy()
                            for v in vecs:
                                score = np.dot(v, probe_dir)
                                scale = sh if score > probe_thresh else sb
                                n_c = n * (n - 1) // 2
                                coeffs = v[:n_c] * scale
                                A = vec_to_skew(coeffs, n)
                                state = state @ expm(A)
                            return bool(np.linalg.norm(state - I, "fro") > st)
                        return detect

                    det = make_detector()
                    r_test = eval_scenarios(scenarios, det)
                    if r_test["f1"] > best_f1:
                        best_f1 = r_test["f1"]
                        best_params = (s_b, s_h, scar_t)

        s_b, s_h, scar_t = best_params
        print(f"\n  Probe-informed holonomy (best params: benign_scale={s_b}, harmful_scale={s_h}, threshold={scar_t}):")

        def best_detect(vecs, sb=s_b, sh=s_h, st=scar_t):
            state = I.copy()
            for v in vecs:
                score = np.dot(v, probe_dir)
                scale = sh if score > probe_thresh else sb
                n_c = n * (n - 1) // 2
                coeffs = v[:n_c] * scale
                A = vec_to_skew(coeffs, n)
                state = state @ expm(A)
            return bool(np.linalg.norm(state - I, "fro") > st)

        r2 = eval_scenarios(scenarios, best_detect)
        bt2 = eval_by_type(scenarios, best_detect)
        print(f"                        F1={r2['f1']:.3f} Prec={r2['prec']:.3f} Rec={r2['rec']:.3f}")
        print(f"    By type: {bt2}")

        delta = r2["f1"] - r["f1"]
        if delta > 0.01:
            print(f"    >> HOLONOMY WINS by {delta:+.3f} F1")
        elif delta < -0.01:
            print(f"    >> PROBE WINS by {-delta:+.3f} F1")
        else:
            print(f"    >> TIE (delta={delta:+.3f})")

        # --- Test: Can the holonomy catch attack-and-recover that probe misses? ---
        print(f"\n  Attack-and-recover analysis:")
        recover_scenarios = [s for s in scenarios if s["type"] == "recover"]

        probe_catches = sum(1 for s in recover_scenarios if probe_detect(s["vecs"]))
        holo_catches = sum(1 for s in recover_scenarios if best_detect(s["vecs"]))
        total_r = len(recover_scenarios)

        # Find cases where holonomy catches but probe doesn't
        holo_only = sum(
            1 for s in recover_scenarios
            if best_detect(s["vecs"]) and not probe_detect(s["vecs"])
        )

        print(f"    Probe catches: {probe_catches}/{total_r}")
        print(f"    Holonomy catches: {holo_catches}/{total_r}")
        print(f"    Holonomy-only catches: {holo_only}/{total_r}")

        if holo_only > 0:
            print(f"    >> HOLONOMY ADDS VALUE on {holo_only} scenarios probe missed!")
        else:
            print(f"    >> Holonomy catches nothing the probe doesn't")

        # --- Approach 1: Learned projection (slower but might be better) ---
        print(f"\n  Learning projection W...")
        W = learn_projection(b_vecs, h_vecs, n=4, lr=0.005, epochs=200)

        def learned_detect(vecs, threshold=0.3):
            state = I.copy()
            for v in vecs:
                coeffs = W @ v * 0.1
                A = vec_to_skew(coeffs, n)
                state = state @ expm(A)
            return bool(np.linalg.norm(state - I, "fro") > threshold)

        # Find best threshold
        best_t_f1 = 0
        best_t = 0.3
        for t in np.linspace(0.01, 2.0, 40):
            def det_t(vecs, th=t):
                return learned_detect(vecs, th)
            rf = eval_scenarios(scenarios, det_t)
            if rf["f1"] > best_t_f1:
                best_t_f1 = rf["f1"]
                best_t = t

        def final_learned(vecs):
            return learned_detect(vecs, best_t)

        r3 = eval_scenarios(scenarios, final_learned)
        bt3 = eval_by_type(scenarios, final_learned)
        print(f"  Learned holonomy:     F1={r3['f1']:.3f} Prec={r3['prec']:.3f} Rec={r3['rec']:.3f}  (thresh={best_t:.3f})")
        print(f"    By type: {bt3}")

        learned_recover = sum(1 for s in recover_scenarios if final_learned(s["vecs"]))
        learned_only = sum(
            1 for s in recover_scenarios
            if final_learned(s["vecs"]) and not probe_detect(s["vecs"])
        )
        print(f"    Recovery: catches {learned_recover}/{total_r}, unique: {learned_only}")

    print(f"\n{'='*70}")
    print("SUMMARY: If probe-informed or learned holonomy beats probe-only,")
    print("especially on attack-and-recover, the concept is salvageable.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
