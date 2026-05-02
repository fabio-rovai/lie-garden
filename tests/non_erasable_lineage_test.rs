//! Cryptographic non-erasability of the lineage chain.
//!
//! The "non-erasable geometric scar" claim was previously stated as a
//! property of the holonomy state itself. That claim is not defensible:
//! group multiplication is reversible, so an attacker who knows the
//! reference state can compute `state^{-1} * ref_state` and return the
//! agent state to identity, geometrically erasing the scar.
//!
//! What IS non-erasable is the **append-only cryptographic record** of
//! the trajectory. Once a step is recorded in the Merkle hash chain, it
//! cannot be removed without breaking SHA-256 — and any later snapshot
//! of the chain by an auditor preserves the deviation forever.
//!
//! This test demonstrates that property concretely:
//!
//!   1. The agent records a benign step, then a deviation, then a
//!      "recovery" that geometrically reverses the deviation.
//!   2. The lineage chain still contains the deviation event.
//!   3. `verify_chain` returns true because the hash chain is consistent.
//!   4. Attempting to remove the deviation from the SQLite store leaves
//!      the chain inconsistent and `verify_chain` returns false.
//!
//! Translation: an attacker can erase the geometric scar but cannot
//! erase the record without leaving falsifiable evidence.

use gravrail::lineage::LineageLog;
use gravrail::state::StateDb;
use std::sync::Arc;
use tempfile::tempdir;

fn open_lineage() -> (LineageLog, tempfile::TempDir, Arc<StateDb>) {
    let dir = tempdir().unwrap();
    let db = Arc::new(StateDb::open(&dir.path().join("test.db")).unwrap());
    let lineage = LineageLog::new((*db).clone());
    (lineage, dir, db)
}

#[test]
fn lineage_records_geometric_deviation_and_recovery() {
    let (lineage, _dir, db) = open_lineage();
    let session = lineage.new_session();

    // 1. Benign step: agent state is on the manifold, near identity.
    lineage.record(&session, "step", "grav_step", "coeffs=[0.01,0.02,0.03] benign");

    // 2. Deviation: a step that significantly perturbs the state away
    //    from the calibration baseline. This is the "attack" event.
    lineage.record(&session, "step", "grav_step", "coeffs=[0.7,-0.5,0.9] DEVIATION");

    // 3. Recovery: an attacker computes state^-1 * ref_state and applies
    //    it. Geometrically the state is back near identity. But the
    //    attacker still has to record the recovery step OR omit it.
    //    Either way, the deviation event in step (2) is now committed
    //    to the chain.
    lineage.record(&session, "step", "grav_step", "coeffs=[-0.7,0.5,-0.9] recovery=erase_attempt");

    // 4. The chain contains all three events and verifies cleanly.
    let conn = db.conn().unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM lineage_events WHERE session_id = ?1",
            rusqlite::params![&session],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 3, "lineage should record all three events");
    drop(conn);

    assert!(
        lineage.verify_chain(&session),
        "chain must verify cleanly while all three events are present"
    );

    // 5. The deviation event is still there for an auditor to find,
    //    even though the agent's geometric state is now ~identity.
    let conn = db.conn().unwrap();
    let mut stmt = conn
        .prepare("SELECT details FROM lineage_events WHERE session_id = ?1 ORDER BY seq")
        .unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params![&session], |r| r.get::<_, String>(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);
    drop(conn);
    assert!(
        rows.iter().any(|d| d.contains("DEVIATION")),
        "auditor must still see the deviation event after geometric recovery"
    );
}

#[test]
fn attacker_cannot_silently_remove_deviation_event() {
    let (lineage, _dir, db) = open_lineage();
    let session = lineage.new_session();

    lineage.record(&session, "step", "grav_step", "benign 1");
    lineage.record(&session, "step", "grav_step", "DEVIATION attack");
    lineage.record(&session, "step", "grav_step", "benign 3");

    assert!(
        lineage.verify_chain(&session),
        "chain verifies before tampering"
    );

    // Attacker deletes the deviation event from the SQLite store. This
    // is the strongest erasure capability — direct DB access, no key
    // material required.
    let conn = db.conn().unwrap();
    conn.execute(
        "DELETE FROM lineage_events WHERE session_id = ?1 AND seq = 2",
        rusqlite::params![&session],
    )
    .unwrap();
    drop(conn);

    // The chain no longer verifies: the JOIN in verify_chain drops the
    // hash row whose event was deleted, leaving stored_prev for seq=3
    // pointing at a hash whose event no longer exists in the join. The
    // verifier returns false.
    assert!(
        !lineage.verify_chain(&session),
        "deleting the deviation event must invalidate the chain"
    );
}

#[test]
fn attacker_cannot_silently_replace_deviation_event() {
    let (lineage, _dir, db) = open_lineage();
    let session = lineage.new_session();

    lineage.record(&session, "step", "grav_step", "benign 1");
    lineage.record(&session, "step", "grav_step", "DEVIATION attack");
    lineage.record(&session, "step", "grav_step", "benign 3");

    assert!(lineage.verify_chain(&session));

    // Attacker rewrites the deviation event to look benign while leaving
    // the hash row untouched. The recomputed SHA-256 over the new
    // payload no longer matches the stored hash.
    let conn = db.conn().unwrap();
    conn.execute(
        "UPDATE lineage_events SET details = 'benign-now' WHERE session_id = ?1 AND seq = 2",
        rusqlite::params![&session],
    )
    .unwrap();
    drop(conn);

    assert!(
        !lineage.verify_chain(&session),
        "rewriting the deviation event must invalidate the hash chain"
    );
}

#[test]
fn empty_session_does_not_verify() {
    // Cross-check the bf2dfc7 fix: an empty session is not "verified
    // clean", it's "no evidence" — verify_chain returns false.
    let (lineage, _dir, _db) = open_lineage();
    let session = lineage.new_session();
    assert!(
        !lineage.verify_chain(&session),
        "empty session must not pass verification"
    );
}
