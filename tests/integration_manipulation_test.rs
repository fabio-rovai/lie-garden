use gravrail::gauge::bundle::FiberBundle;
use gravrail::gauge::connection::Connection;
use gravrail::lie::group::{LieGroup, GroupType};

#[test]
fn holonomy_detects_manipulation() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group.clone(), 2);
    // Non-flat connection = there's curvature/manipulation in the system
    let conn = Connection::with_curvature(&bundle, 0.5);
    let path = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
    let h = conn.holonomy(&path);
    let id = group.identity();
    // Non-trivial holonomy detected
    let deviation: f64 = h.matrix().iter().zip(id.matrix().iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(deviation > 1e-6, "Non-flat connection must produce non-trivial holonomy, got deviation={}", deviation);
}

#[test]
fn flat_connection_no_manipulation() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group.clone(), 2);
    let conn = Connection::flat(&bundle);
    let path = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
    let h = conn.holonomy(&path);
    let id = group.identity();
    let deviation: f64 = h.matrix().iter().zip(id.matrix().iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(deviation < 1e-6, "Flat connection must have trivial holonomy, got deviation={}", deviation);
}

#[test]
fn lineage_tamper_detection() {
    let dir = tempfile::tempdir().unwrap();
    let db = gravrail::state::StateDb::open(&dir.path().join("test.db")).unwrap();
    let log = gravrail::lineage::LineageLog::new(db.clone());
    let session = log.new_session();

    log.record(&session, "step", "circuit_step", "agent_1 moved");
    log.record(&session, "step", "circuit_step", "agent_2 moved");
    assert!(log.verify_chain(&session), "Clean chain must verify");

    // Tamper with a record
    db.conn().unwrap().execute(
        "UPDATE lineage_events SET details = 'TAMPERED' WHERE seq = 1 AND session_id = ?1",
        rusqlite::params![session],
    ).unwrap();
    assert!(!log.verify_chain(&session), "Tampered chain must fail verification");
}
