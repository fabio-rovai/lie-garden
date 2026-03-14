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

CREATE INDEX IF NOT EXISTS idx_lineage_session ON lineage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_lineage_seq ON lineage_events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_trajectories_agent ON trajectories(agent_id, step);
CREATE INDEX IF NOT EXISTS idx_tool_feedback ON tool_feedback(tool, rule_id, entity);
CREATE INDEX IF NOT EXISTS idx_lineage_hashes ON lineage_hashes(session_id, seq);
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
}
