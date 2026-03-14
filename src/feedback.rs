use crate::state::StateDb;

/// What to do with an issue based on feedback history.
#[derive(Debug, PartialEq)]
pub enum FeedbackAction {
    /// Report at original severity
    Keep,
    /// Downgrade severity one level (warning -> info)
    Downgrade,
    /// Suppress entirely (omit from output)
    Suppress,
}

const SUPPRESS_THRESHOLD: i64 = 3;
const DOWNGRADE_THRESHOLD: i64 = 2;

/// Check feedback history for a (tool, rule_id, entity) tuple.
pub fn get_feedback_adjustment(db: &StateDb, tool: &str, rule_id: &str, entity: &str) -> FeedbackAction {
    let conn = match db.conn() {
        Ok(c) => c,
        Err(_) => return FeedbackAction::Keep,
    };
    let dismiss_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM tool_feedback WHERE tool = ?1 AND rule_id = ?2 AND entity = ?3 AND accepted = 0",
            rusqlite::params![tool, rule_id, entity],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let accept_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM tool_feedback WHERE tool = ?1 AND rule_id = ?2 AND entity = ?3 AND accepted = 1",
            rusqlite::params![tool, rule_id, entity],
            |r| r.get(0),
        )
        .unwrap_or(0);

    if accept_count > 0 {
        return FeedbackAction::Keep;
    }
    if dismiss_count >= SUPPRESS_THRESHOLD {
        return FeedbackAction::Suppress;
    }
    if dismiss_count >= DOWNGRADE_THRESHOLD {
        return FeedbackAction::Downgrade;
    }
    FeedbackAction::Keep
}

/// Record feedback for a tool issue.
pub fn record_tool_feedback(db: &StateDb, tool: &str, rule_id: &str, entity: &str, accepted: bool) -> anyhow::Result<String> {
    let conn = db.conn()?;
    conn.execute(
        "INSERT INTO tool_feedback (tool, rule_id, entity, accepted) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![tool, rule_id, entity, accepted as i32],
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "tool": tool,
        "rule_id": rule_id,
        "entity": entity,
        "accepted": accepted,
    })
    .to_string())
}

