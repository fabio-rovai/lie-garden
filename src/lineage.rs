use crate::state::StateDb;
use crate::util;
use chrono::Utc;
use sha2::{Sha256, Digest};

pub struct LineageLog {
    db: StateDb,
}

impl LineageLog {
    pub fn new(db: StateDb) -> Self {
        Self { db }
    }

    pub fn new_session(&self) -> String {
        util::rand_id()
    }

    pub fn record(&self, session_id: &str, event_type: &str, operation: &str, details: &str) {
        let conn = match self.db.conn() {
            Ok(c) => c,
            Err(_) => return,
        };
        let seq: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM lineage_events WHERE session_id = ?1",
                rusqlite::params![session_id],
                |r| r.get(0),
            )
            .unwrap_or(1);
        let ts = Utc::now().timestamp();

        let prev_hash: String = conn
            .query_row(
                "SELECT hash FROM lineage_hashes WHERE session_id = ?1 ORDER BY seq DESC LIMIT 1",
                rusqlite::params![session_id],
                |r| r.get(0),
            )
            .unwrap_or_else(|_| "genesis".to_string());

        let _ = conn.execute(
            "INSERT INTO lineage_events (session_id, seq, timestamp, event_type, operation, details)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![session_id, seq, ts.to_string(), event_type, operation, details],
        );

        let payload = format!("{}:{}:{}:{}:{}:{}:{}", prev_hash, session_id, seq, ts, event_type, operation, details);
        let hash = format!("{:x}", Sha256::digest(payload.as_bytes()));
        let _ = conn.execute(
            "INSERT INTO lineage_hashes (session_id, seq, hash, prev_hash) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![session_id, seq, hash, prev_hash],
        );
    }

    pub fn get_compact(&self, session_id: &str) -> String {
        let conn = match self.db.conn() {
            Ok(c) => c,
            Err(_) => return String::new(),
        };
        let mut stmt = match conn.prepare(
            "SELECT seq, timestamp, event_type, operation, details
             FROM lineage_events WHERE session_id = ?1 ORDER BY seq ASC",
        ) {
            Ok(s) => s,
            Err(_) => return String::new(),
        };
        let rows: Vec<String> = match stmt.query_map(rusqlite::params![session_id], |row| {
            let seq: i64 = row.get(0)?;
            let ts: String = row.get(1)?;
            let etype: String = row.get(2)?;
            let op: String = row.get(3)?;
            let details: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
            Ok(format!("{}:{}:{}:{}:{}:{}", session_id, seq, ts, etype, op, details))
        }) {
            Ok(mapped) => mapped.filter_map(|r| r.ok()).collect(),
            Err(_) => return String::new(),
        };
        rows.join("\n") + "\n"
    }

    /// Verify the hash chain is intact — returns true if no tampering detected.
    pub fn verify_chain(&self, session_id: &str) -> bool {
        let conn = match self.db.conn() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let mut stmt = match conn.prepare(
            "SELECT h.seq, h.hash, h.prev_hash, e.timestamp, e.event_type, e.operation, e.details
             FROM lineage_hashes h
             JOIN lineage_events e ON e.session_id = h.session_id AND e.seq = h.seq
             WHERE h.session_id = ?1 ORDER BY h.seq ASC",
        ) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let mut prev = "genesis".to_string();
        match stmt.query_map(rusqlite::params![session_id], |row| {
            let seq: i64 = row.get(0)?;
            let hash: String = row.get(1)?;
            let stored_prev: String = row.get(2)?;
            let ts: String = row.get(3)?;
            let etype: String = row.get(4)?;
            let op: String = row.get(5)?;
            let details: String = row.get::<_, Option<String>>(6)?.unwrap_or_default();
            Ok((seq, hash, stored_prev, ts, etype, op, details))
        }) {
            Ok(mapped) => mapped
                .filter_map(|r| r.ok())
                .all(|(seq, hash, stored_prev, ts, etype, op, details)| {
                    if stored_prev != prev {
                        return false;
                    }
                    let payload = format!("{}:{}:{}:{}:{}:{}:{}", prev, session_id, seq, ts, etype, op, details);
                    let expected = format!("{:x}", Sha256::digest(payload.as_bytes()));
                    let ok = hash == expected;
                    prev = hash;
                    ok
                }),
            Err(_) => false,
        }
    }
}
