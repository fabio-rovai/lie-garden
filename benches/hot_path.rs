use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::DMatrix;
use gravrail::lie::group::{LieGroup, GroupType};
use gravrail::lie::algebra::LieAlgebra;
use gravrail::circuit::define::Circuit;
use gravrail::circuit::map::map_to_algebra;
use gravrail::crypto::sign::{sign, verify};
use gravrail::crypto::commit::{commit, verify_commit};
use gravrail::crypto::zkproof::{prove_knowledge, verify_knowledge};
use gravrail::gauge::bundle::FiberBundle;
use gravrail::gauge::connection::Connection;

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    let so3 = LieGroup::new(GroupType::SO, 3);
    let coeffs_3 = vec![0.3, -0.2, 0.5];
    group.bench_function("SO3_rodrigues", |b| {
        b.iter(|| so3.exp(black_box(&coeffs_3)))
    });

    let so4 = LieGroup::new(GroupType::SO, 4);
    let coeffs_6 = vec![0.1, -0.2, 0.3, 0.15, -0.1, 0.25];
    group.bench_function("SO4_pade", |b| {
        b.iter(|| so4.exp(black_box(&coeffs_6)))
    });

    let se3 = LieGroup::new(GroupType::SE, 3);
    let coeffs_se3 = vec![0.3, 0.5, -0.2];
    group.bench_function("SE3", |b| {
        b.iter(|| se3.exp(black_box(&coeffs_se3)))
    });

    let gl2 = LieGroup::new(GroupType::GL, 2);
    let coeffs_gl2 = vec![0.1, 0.2, -0.1, 0.3];
    group.bench_function("GL2", |b| {
        b.iter(|| gl2.exp(black_box(&coeffs_gl2)))
    });

    group.finish();
}

fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("log");

    let so3 = LieGroup::new(GroupType::SO, 3);
    let elem = so3.exp(&vec![0.3, -0.2, 0.5]);
    group.bench_function("SO3_rodrigues", |b| {
        b.iter(|| so3.log(black_box(&elem)))
    });

    let so4 = LieGroup::new(GroupType::SO, 4);
    let elem4 = so4.exp(&vec![0.1, -0.2, 0.3, 0.15, -0.1, 0.25]);
    group.bench_function("SO4_pade", |b| {
        b.iter(|| so4.log(black_box(&elem4)))
    });

    group.finish();
}

fn bench_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiply");

    let so3 = LieGroup::new(GroupType::SO, 3);
    let a = so3.exp(&vec![0.3, -0.2, 0.5]);
    let b = so3.exp(&vec![-0.1, 0.4, 0.2]);
    group.bench_function("SO3", |b_iter| {
        b_iter.iter(|| so3.multiply(black_box(&a), black_box(&b)))
    });

    let se3 = LieGroup::new(GroupType::SE, 3);
    let a_se = se3.exp(&vec![0.3, 0.5, -0.2]);
    let b_se = se3.exp(&vec![-0.1, 0.2, 0.4]);
    group.bench_function("SE3", |b_iter| {
        b_iter.iter(|| se3.multiply(black_box(&a_se), black_box(&b_se)))
    });

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse");

    let so3 = LieGroup::new(GroupType::SO, 3);
    let a = so3.exp(&vec![0.3, -0.2, 0.5]);
    group.bench_function("SO3_transpose", |b| {
        b.iter(|| so3.inverse(black_box(&a)))
    });

    let gl2 = LieGroup::new(GroupType::GL, 2);
    let a_gl = gl2.exp(&vec![0.1, 0.2, -0.1, 0.3]);
    group.bench_function("GL2_lu", |b| {
        b.iter(|| gl2.inverse(black_box(&a_gl)))
    });

    let se3 = LieGroup::new(GroupType::SE, 3);
    let a_se = se3.exp(&vec![0.3, 0.5, -0.2]);
    group.bench_function("SE3", |b| {
        b.iter(|| se3.inverse(black_box(&a_se)))
    });

    group.finish();
}

fn bench_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("project_to_group");

    let so3 = LieGroup::new(GroupType::SO, 3);
    let noisy = {
        let clean = so3.exp(&vec![0.3, -0.2, 0.5]);
        let data: Vec<f64> = clean.matrix().iter().enumerate()
            .map(|(i, &v)| v + (i as f64) * 0.001)
            .collect();
        let m = DMatrix::from_row_slice(3, 3, &data);
        gravrail::lie::group::GroupElement::from_matrix(&m)
    };
    group.bench_function("SO3_svd", |b| {
        b.iter(|| so3.project_to_group(black_box(&noisy)))
    });

    group.finish();
}

fn bench_map_to_algebra(c: &mut Criterion) {
    c.bench_function("map_to_algebra_dim3", |b| {
        b.iter(|| map_to_algebra(black_box("The agent should move left"), black_box(3), 1.0))
    });

    c.bench_function("map_to_algebra_dim6", |b| {
        b.iter(|| map_to_algebra(black_box("The agent should move left"), black_box(6), 1.0))
    });
}

fn bench_full_step_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_step");

    // SO(3) pipeline: map → constrain → exp → multiply
    let so3 = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(so3.clone(), None);
    let state = so3.identity();
    group.bench_function("SO3_unconstrained", |b| {
        b.iter(|| {
            let coeffs = map_to_algebra(black_box("agent output text"), circuit.group.algebra_dim(), 1.0);
            let constrained = circuit.constrain_algebra(&coeffs);
            let step = circuit.group.exp(&constrained);
            circuit.group.multiply(black_box(&state), &step)
        })
    });

    // SO(3) with generator mask
    let mask = vec![true, false, true]; // only 2 of 3 generators active
    let circuit_masked = Circuit::new(so3.clone(), Some(mask));
    group.bench_function("SO3_masked", |b| {
        b.iter(|| {
            let coeffs = map_to_algebra(black_box("agent output text"), circuit_masked.group.algebra_dim(), 1.0);
            let constrained = circuit_masked.constrain_algebra(&coeffs);
            let step = circuit_masked.group.exp(&constrained);
            circuit_masked.group.multiply(black_box(&state), &step)
        })
    });

    // SE(3) pipeline
    let se3 = LieGroup::new(GroupType::SE, 3);
    let circuit_se = Circuit::new(se3.clone(), None);
    let state_se = se3.identity();
    group.bench_function("SE3", |b| {
        b.iter(|| {
            let coeffs = map_to_algebra(black_box("agent output text"), circuit_se.group.algebra_dim(), 1.0);
            let constrained = circuit_se.constrain_algebra(&coeffs);
            let step = circuit_se.group.exp(&constrained);
            circuit_se.group.multiply(black_box(&state_se), &step)
        })
    });

    group.finish();
}

fn bench_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("crypto");

    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let public = g.exp(&secret);

    group.bench_function("schnorr_sign", |b| {
        b.iter(|| sign(black_box(&g), black_box(b"benchmark message"), black_box(&secret)))
    });

    let sig = sign(&g, b"benchmark message", &secret);
    group.bench_function("schnorr_verify", |b| {
        b.iter(|| verify(black_box(&g), black_box(b"benchmark message"), black_box(&public), black_box(&sig)))
    });

    let value = vec![0.3, -0.2, 0.5];
    let randomness = vec![0.1, 0.4, -0.1];
    group.bench_function("pedersen_commit", |b| {
        b.iter(|| commit(black_box(&g), black_box(&value), black_box(&randomness)))
    });

    let c_val = commit(&g, &value, &randomness);
    group.bench_function("pedersen_verify", |b| {
        b.iter(|| verify_commit(black_box(&g), black_box(&c_val), black_box(&value), black_box(&randomness)))
    });

    group.bench_function("zk_prove", |b| {
        b.iter(|| prove_knowledge(black_box(&g), black_box(&secret), black_box(&public)))
    });

    let proof = prove_knowledge(&g, &secret, &public);
    group.bench_function("zk_verify", |b| {
        b.iter(|| verify_knowledge(black_box(&g), black_box(&public), black_box(&proof)))
    });

    group.finish();
}

fn bench_holonomy(c: &mut Criterion) {
    let g = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(g.clone(), 2);
    let conn = Connection::with_curvature(&bundle, 0.5);
    let path = vec![
        vec![1.0, 0.0], vec![0.0, 1.0],
        vec![-1.0, 0.0], vec![0.0, -1.0],
    ];

    c.bench_function("holonomy_SO3_square_path", |b| {
        b.iter(|| conn.holonomy(black_box(&path)))
    });
}

fn bench_bracket(c: &mut Criterion) {
    let alg = LieAlgebra::new(GroupType::SO, 3);
    let x = vec![0.3, -0.2, 0.5];
    let y = vec![-0.1, 0.4, 0.2];

    c.bench_function("lie_bracket_SO3", |b| {
        b.iter(|| alg.bracket(black_box(&x), black_box(&y)))
    });
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for n in [2, 3, 4, 5] {
        let g = LieGroup::new(GroupType::SO, n);
        let dim = g.algebra_dim();
        let coeffs: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1 - 0.2).collect();
        group.bench_with_input(BenchmarkId::new("SO_exp", n), &n, |b, _| {
            b.iter(|| g.exp(black_box(&coeffs)))
        });
    }

    for n in [2, 3, 4, 5] {
        let g = LieGroup::new(GroupType::SO, n);
        let dim = g.algebra_dim();
        let coeffs: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1 - 0.2).collect();
        let a = g.exp(&coeffs);
        let b_elem = g.exp(&coeffs.iter().map(|x| -x).collect::<Vec<_>>());
        group.bench_with_input(BenchmarkId::new("SO_multiply", n), &n, |b, _| {
            b.iter(|| g.multiply(black_box(&a), black_box(&b_elem)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_exp,
    bench_log,
    bench_multiply,
    bench_inverse,
    bench_project,
    bench_map_to_algebra,
    bench_full_step_pipeline,
    bench_crypto,
    bench_holonomy,
    bench_bracket,
    bench_scaling,
);
criterion_main!(benches);
