use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS circuits (
    id TEXT PRIMARY KEY,
    group_type TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    config TEXT NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    circuit_id TEXT NOT NULL,
    state TEXT NOT NULL,
    step_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (circuit_id) REFERENCES circuits(id)
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    algebra_element TEXT NOT NULL,
    group_element TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS lineage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS tool_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS gauge_bundles (
    id TEXT PRIMARY KEY,
    circuit_id TEXT NOT NULL,
    connection_config TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (circuit_id) REFERENCES circuits(id)
);

CREATE TABLE IF NOT EXISTS crypto_keys (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    public_key TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS lineage_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    circuit_id   TEXT    NOT NULL,
    step         INTEGER NOT NULL,
    input_norm   REAL,
    input_hol    REAL,
    output_norm  REAL    NOT NULL,
    output_hol   REAL,
    drift        REAL,
    algebra      BLOB    NOT NULL,
    ts           INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lineage_session ON lineage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_lineage_seq ON lineage_events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_trajectories_agent ON trajectories(agent_id, step);
CREATE INDEX IF NOT EXISTS idx_tool_feedback ON tool_feedback(tool, rule_id, entity);
CREATE INDEX IF NOT EXISTS idx_lineage_hashes ON lineage_hashes(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_traj_steps_session ON trajectory_steps(session_id, step);
";

#[derive(Clone)]
pub struct StateDb {
    conn: Arc<Mutex<Connection>>,
}

impl StateDb {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|e| anyhow::anyhow!("mutex poisoned: {e}"))
    }

    /// Insert one trajectory step row.
    pub fn insert_trajectory_step(
        &self,
        session_id: &str,
        circuit_id: &str,
        step: u64,
        input_norm: Option<f64>,
        input_hol: Option<f64>,
        output_norm: f64,
        output_hol: Option<f64>,
        drift: Option<f64>,
        algebra: &[f64],
    ) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        let algebra_bytes: Vec<u8> = algebra.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        conn.execute(
            "INSERT INTO trajectory_steps
             (session_id, circuit_id, step, input_norm, input_hol, output_norm, output_hol, drift, algebra, ts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                session_id, circuit_id, step as i64,
                input_norm, input_hol,
                output_norm, output_hol,
                drift, algebra_bytes, ts
            ],
        )?;
        Ok(())
    }

    /// Query all trajectory steps for a session, ordered by step.
    pub fn query_trajectory_steps(&self, session_id: &str) -> anyhow::Result<Vec<TrajectoryStep>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT session_id, circuit_id, step, input_norm, input_hol,
                    output_norm, output_hol, drift, ts
             FROM trajectory_steps WHERE session_id = ?1 ORDER BY step ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok(TrajectoryStep {
                session_id: row.get(0)?,
                circuit_id: row.get(1)?,
                step: row.get(2)?,
                input_norm: row.get(3)?,
                input_hol: row.get(4)?,
                output_norm: row.get(5)?,
                output_hol: row.get(6)?,
                drift: row.get(7)?,
                ts: row.get(8)?,
            })
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// List all distinct session IDs in trajectory_steps, most recent first.
    pub fn list_trajectory_sessions(&self) -> anyhow::Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT session_id FROM trajectory_steps GROUP BY session_id ORDER BY MAX(ts) DESC",
        )?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut result = Vec::new();
        for row in rows { result.push(row?); }
        Ok(result)
    }
}

/// A single step recorded in the trajectory_steps table.
#[derive(Debug)]
pub struct TrajectoryStep {
    pub session_id: String,
    pub circuit_id: String,
    pub step: i64,
    pub input_norm: Option<f64>,
    pub input_hol: Option<f64>,
    pub output_norm: f64,
    pub output_hol: Option<f64>,
    pub drift: Option<f64>,
    pub ts: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_trajectory_step() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(&dir.path().join("test.db")).unwrap();
        db.insert_trajectory_step(
            "session-abc",
            "circuit-xyz",
            1,
            Some(3.2),
            Some(0.05),
            4.1,
            Some(0.08),
            Some(0.11),
            &[0.1f64, 0.2, 0.3],
        ).unwrap();
        let steps = db.query_trajectory_steps("session-abc").unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].step, 1);
        assert!((steps[0].output_norm - 4.1).abs() < 1e-6);
    }
}
