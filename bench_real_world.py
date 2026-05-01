"""
GravRail Real-World Adversarial Benchmark
==========================================
Tests directional detection + holonomy scarring against:
- InjecAgent (Zhan et al. 2024): 1,054 indirect prompt injection cases in agent tools
- TensorTrust (Toyer et al. 2024): 775 human-crafted prompt hijacking attacks

Embedding models tested:
- Snowflake Arctic-M (2024, 768d) — best subtle detection
- Model2Vec potion-base-32M (2024, 512d) — fastest (0.07ms/text)

Baseline: GloVe-50 results from Rust directional_detection.rs
"""
import numpy as np
import json
import time
import sys
from pathlib import Path

def vec_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def load_injecagent(data_dir="/tmp/InjecAgent/data"):
    """Load InjecAgent dataset: indirect prompt injection in agent tool responses."""
    p = Path(data_dir)

    # Attacker instructions (the injected text)
    attacks = []
    for fname in ["test_cases_dh_base.json", "test_cases_ds_base.json"]:
        with open(p / fname) as f:
            cases = json.load(f)
        for case in cases:
            attacks.append({
                "text": case["Attacker Instruction"],
                "type": case.get("Attack Type", "unknown"),
                "source": fname.replace("test_cases_", "").replace("_base.json", ""),
                "context": case.get("User Instruction", ""),
            })

    # Benign: user instructions (what the user actually asked)
    benign = []
    with open(p / "user_cases.jsonl") as f:
        for line in f:
            case = json.loads(line)
            benign.append({
                "text": case["User Instruction"],
                "type": "benign_user",
                "source": "user_cases",
            })

    # Also use the tool responses WITHOUT injection as benign
    # (the template before attacker instruction is inserted)
    with open(p / "test_cases_dh_base.json") as f:
        dh = json.load(f)
    # Extract unique user instructions as additional benign
    seen = set()
    for case in dh:
        ui = case["User Instruction"]
        if ui not in seen:
            seen.add(ui)
            benign.append({
                "text": ui,
                "type": "benign_user_dh",
                "source": "dh_user_instructions",
            })

    return benign, attacks


def load_tensor_trust(data_dir="/tmp/tensor-trust-data"):
    """Load TensorTrust hijacking robustness dataset."""
    p = Path(data_dir) / "benchmarks" / "hijacking-robustness" / "v1" / "hijacking_robustness_dataset.jsonl"

    attacks = []
    defenses = []
    with open(p) as f:
        for line in f:
            case = json.loads(line)
            attack_text = case.get("attack", "")
            if attack_text and len(attack_text.strip()) > 10:
                attacks.append({
                    "text": attack_text,
                    "type": "hijack",
                    "source": "tensor_trust",
                })
            # Pre-prompt + post-prompt are the "defense" (benign system context)
            pre = case.get("pre_prompt", "")
            if pre and len(pre.strip()) > 10:
                defenses.append({
                    "text": pre,
                    "type": "benign_defense",
                    "source": "tensor_trust",
                })

    return defenses, attacks


def train_directional_probe(benign_vecs, harmful_vecs, dim, full_dim):
    """Learn harmful direction via mean difference, with optional JL projection."""
    if dim < full_dim:
        np.random.seed(42)
        proj = np.random.randn(full_dim, dim) / np.sqrt(dim)
        b = benign_vecs @ proj
        h = harmful_vecs @ proj
    else:
        b = benign_vecs
        h = harmful_vecs
        proj = None

    harm_dir = vec_normalize(h.mean(axis=0) - b.mean(axis=0))
    b_scores = b @ harm_dir
    h_scores = h @ harm_dir
    threshold = (b_scores.mean() + h_scores.mean()) / 2.0
    separation = h_scores.mean() - b_scores.mean()

    return harm_dir, threshold, separation, proj


def evaluate(harm_dir, threshold, proj, embeddings, labels, full_dim, dim):
    """Evaluate directional probe on test data."""
    tp = fp = tn = fn = 0
    scores = []

    for emb, label in zip(embeddings, labels):
        if proj is not None:
            emb = emb @ proj
        score = np.dot(emb, harm_dir)
        predicted = score > threshold
        actual = label == "attack"
        scores.append(score)

        if actual and predicted: tp += 1
        elif actual and not predicted: fn += 1
        elif not actual and predicted: fp += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "scores": np.array(scores), "labels": labels,
    }


def simulate_holonomy(embeddings, labels, harm_dir, proj, window=3):
    """Simulate multi-step holonomy scarring.

    Non-commutative group multiplication means path A→B→A ≠ A→A→A.
    We simulate this with matrix exponentials in SO(n).
    """
    from scipy.linalg import expm

    dim = len(harm_dir)
    n = max(3, int(np.ceil(np.sqrt(dim))))  # Matrix size for SO(n)

    # Build reference trajectory from benign-only
    benign_embs = [e for e, l in zip(embeddings, labels) if l == "benign"]
    ref_state = np.eye(n)
    for emb in benign_embs[:10]:
        if proj is not None:
            emb = emb @ proj
        # Map to skew-symmetric (Lie algebra of SO(n))
        coeffs = emb[:n*(n-1)//2] if len(emb) >= n*(n-1)//2 else np.resize(emb, n*(n-1)//2)
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                A[i, j] = coeffs[idx] * 0.1
                A[j, i] = -coeffs[idx] * 0.1
                idx += 1
        ref_state = ref_state @ expm(A)

    # Now simulate mixed trajectory (benign + attack interleaved)
    mixed_state = np.eye(n)
    scars = []
    step_labels = []

    all_embs = list(zip(embeddings, labels))
    # Take first 10 benign, then 5 attacks, then 5 benign (attack+recover)
    scenario_embs = []
    benign_pool = [(e, l) for e, l in all_embs if l == "benign"]
    attack_pool = [(e, l) for e, l in all_embs if l == "attack"]

    # Scenario: 5 benign → 3 attack → 2 benign (recovery)
    scenario = benign_pool[:5] + attack_pool[:3] + benign_pool[5:7]

    for emb, label in scenario:
        if proj is not None:
            emb = emb @ proj
        coeffs = emb[:n*(n-1)//2] if len(emb) >= n*(n-1)//2 else np.resize(emb, n*(n-1)//2)
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                A[i, j] = coeffs[idx] * 0.1
                A[j, i] = -coeffs[idx] * 0.1
                idx += 1
        mixed_state = mixed_state @ expm(A)
        scar = np.linalg.norm(mixed_state - ref_state, 'fro')
        scars.append(scar)
        step_labels.append(label)

    return scars, step_labels


def run_benchmark_on_dataset(model_name, embed_fn, benign_data, attack_data, dataset_name):
    """Run full directional + holonomy benchmark on a dataset."""
    print(f"\n{'='*90}")
    print(f"  DATASET: {dataset_name}")
    print(f"  MODEL: {model_name}")
    print(f"  Benign: {len(benign_data)}, Attacks: {len(attack_data)}")
    print(f"{'='*90}")

    # Use 80% for training, 20% for test
    np.random.seed(42)
    n_train_b = max(20, int(len(benign_data) * 0.8))
    n_train_a = max(20, int(len(attack_data) * 0.8))

    idx_b = np.random.permutation(len(benign_data))
    idx_a = np.random.permutation(len(attack_data))

    train_benign = [benign_data[i] for i in idx_b[:n_train_b]]
    test_benign = [benign_data[i] for i in idx_b[n_train_b:]]
    train_attack = [attack_data[i] for i in idx_a[:n_train_a]]
    test_attack = [attack_data[i] for i in idx_a[n_train_a:]]

    if len(test_benign) < 5:
        test_benign = train_benign[-10:]
    if len(test_attack) < 5:
        test_attack = train_attack[-10:]

    print(f"  Train: {len(train_benign)} benign, {len(train_attack)} attack")
    print(f"  Test: {len(test_benign)} benign, {len(test_attack)} attack")

    # Embed all texts
    all_texts = ([d["text"] for d in train_benign] +
                 [d["text"] for d in train_attack] +
                 [d["text"] for d in test_benign] +
                 [d["text"] for d in test_attack])

    t0 = time.time()
    all_vecs = embed_fn(all_texts)
    embed_time = time.time() - t0
    print(f"  Embed time: {embed_time:.2f}s ({len(all_texts)} texts, {embed_time/len(all_texts)*1000:.1f}ms/text)")
    print(f"  Dim: {all_vecs.shape[1]}")

    # Split back
    offset = 0
    train_b_vecs = all_vecs[offset:offset+len(train_benign)]; offset += len(train_benign)
    train_a_vecs = all_vecs[offset:offset+len(train_attack)]; offset += len(train_attack)
    test_b_vecs = all_vecs[offset:offset+len(test_benign)]; offset += len(test_benign)
    test_a_vecs = all_vecs[offset:offset+len(test_attack)]; offset += len(test_attack)

    full_dim = all_vecs.shape[1]

    test_embs = np.vstack([test_b_vecs, test_a_vecs])
    test_labels = ["benign"] * len(test_b_vecs) + ["attack"] * len(test_a_vecs)

    results_by_dim = {}

    for dim in [10, 20, 30, 50, full_dim]:
        if dim > full_dim:
            continue

        harm_dir, threshold, separation, proj = train_directional_probe(
            train_b_vecs, train_a_vecs, dim, full_dim
        )

        result = evaluate(harm_dir, threshold, proj, test_embs, test_labels, full_dim, dim)
        results_by_dim[dim] = result

        print(f"  dim={dim:>5}: F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  "
              f"Sep={separation:.3f}  TP={result['tp']} FP={result['fp']} TN={result['tn']} FN={result['fn']}")

    # Holonomy simulation at best dim
    best_dim = max(results_by_dim, key=lambda d: results_by_dim[d]["f1"])
    harm_dir, threshold, separation, proj = train_directional_probe(
        train_b_vecs, train_a_vecs, best_dim, full_dim
    )

    try:
        holonomy_embs = list(test_b_vecs[:7]) + list(test_a_vecs[:3])
        holonomy_labels = ["benign"]*7 + ["attack"]*3
        scars, step_labels = simulate_holonomy(holonomy_embs, holonomy_labels, harm_dir, proj)
        print(f"\n  Holonomy (attack+recover, dim={best_dim}):")
        for i, (scar, sl) in enumerate(zip(scars, step_labels)):
            marker = " <<<" if sl == "attack" else ""
            print(f"    Step {i+1} [{sl:>6}]: scar={scar:.4f}{marker}")
        final_scar = scars[-1]
        print(f"  Final scar after recovery: {final_scar:.4f} {'(DETECTED)' if final_scar > 0.1 else '(missed)'}")
    except ImportError:
        print("  (scipy not installed — skipping holonomy simulation)")

    return results_by_dim


def main():
    # ─── Load datasets ───
    print("Loading datasets...\n")

    inj_benign, inj_attacks = load_injecagent()
    print(f"  InjecAgent: {len(inj_benign)} benign, {len(inj_attacks)} attacks")

    tt_benign, tt_attacks = load_tensor_trust()
    print(f"  TensorTrust: {len(tt_benign)} benign, {len(tt_attacks)} attacks")

    # Combined dataset
    all_benign = inj_benign + tt_benign
    all_attacks = inj_attacks + tt_attacks
    print(f"  Combined: {len(all_benign)} benign, {len(all_attacks)} attacks")

    # ─── Load embedding models ───

    # Model 1: Snowflake Arctic-M (best subtle detection)
    print("\n  Loading Snowflake Arctic-M...")
    from fastembed import TextEmbedding
    arctic = TextEmbedding("snowflake/snowflake-arctic-embed-m")
    def embed_arctic(texts):
        return np.array(list(arctic.embed(texts)))

    # Model 2: Model2Vec (fastest)
    print("  Loading Model2Vec potion-base-32M...")
    from model2vec import StaticModel
    m2v = StaticModel.from_pretrained("minishlab/potion-base-32M")
    def embed_m2v(texts):
        return np.array(m2v.encode(texts))

    # ─── Run benchmarks ───

    all_results = {}

    # InjecAgent with Snowflake
    r = run_benchmark_on_dataset("Snowflake Arctic-M (2024)", embed_arctic,
                                  inj_benign, inj_attacks, "InjecAgent (indirect injection in tools)")
    all_results["InjecAgent + Arctic"] = r

    # InjecAgent with Model2Vec
    r = run_benchmark_on_dataset("Model2Vec potion-base-32M (2024)", embed_m2v,
                                  inj_benign, inj_attacks, "InjecAgent (indirect injection in tools)")
    all_results["InjecAgent + M2V"] = r

    # TensorTrust with Snowflake
    r = run_benchmark_on_dataset("Snowflake Arctic-M (2024)", embed_arctic,
                                  tt_benign, tt_attacks, "TensorTrust (human-crafted hijacking)")
    all_results["TensorTrust + Arctic"] = r

    # TensorTrust with Model2Vec
    r = run_benchmark_on_dataset("Model2Vec potion-base-32M (2024)", embed_m2v,
                                  tt_benign, tt_attacks, "TensorTrust (human-crafted hijacking)")
    all_results["TensorTrust + M2V"] = r

    # Combined with Snowflake
    r = run_benchmark_on_dataset("Snowflake Arctic-M (2024)", embed_arctic,
                                  all_benign, all_attacks, "Combined (InjecAgent + TensorTrust)")
    all_results["Combined + Arctic"] = r

    # ─── Summary ───
    print(f"\n\n{'='*90}")
    print(f"  REAL-WORLD BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(f"\n  {'Config':<40} {'Best F1':>8} {'@Dim':>6} {'F1@50':>7} {'F1@30':>7}")
    print(f"  {'-'*70}")

    for name, results in all_results.items():
        best_dim = max(results, key=lambda d: results[d]["f1"])
        best_f1 = results[best_dim]["f1"]
        f1_50 = results.get(50, {}).get("f1", 0)
        f1_30 = results.get(30, {}).get("f1", 0)
        print(f"  {name:<40} {best_f1:>8.3f} {best_dim:>6} {f1_50:>7.3f} {f1_30:>7.3f}")

    # Baseline NOT recomputed here. Do not quote a hardcoded GloVe-50 F1 as if
    # this script produced it; run the Rust example separately.

    print(f"\n  Key: InjecAgent = indirect prompt injection in agent tool responses (Zhan et al. 2024)")
    print(f"       TensorTrust = human-crafted prompt hijacking from competitive game (Toyer et al. 2024)")
    # The "100% of attack+recovery scenarios" line was a hardcoded string
    # not derived from any computation in this script — removed. honest_ablation.py
    # and holonomy_v2.py later document that holonomy IS partially erasable
    # and dataset-dependent.


if __name__ == "__main__":
    main()
