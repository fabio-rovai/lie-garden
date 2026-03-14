// CLI subcommand: gravrail proxy

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::circuit::define::Circuit;
use crate::lie::group::{GroupType, LieGroup};
use super::auth::SessionAuth;
use super::confine::ConfinementPipeline;
use super::server::{ProxyState, build_router};
use super::watchdog::{WatchdogConfig, create_watchdog};

/// Configuration for the proxy subcommand.
pub struct ProxyConfig {
    pub circuit_id: Option<String>,
    pub group_type: String,
    pub group_dim: usize,
    pub upstream_url: String,
    pub port: u16,
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub max_state_norm: Option<f64>,
    pub prove_every_step: bool,
}

/// Run the proxy server with the given configuration.
pub async fn run_proxy(config: ProxyConfig) -> anyhow::Result<()> {
    // 1. Parse group_type string to GroupType enum
    let group_type = match config.group_type.to_uppercase().as_str() {
        "SO" => GroupType::SO,
        "SE" => GroupType::SE,
        "GL" => GroupType::GL,
        other => anyhow::bail!("Unknown group type '{}'. Expected SO, SE, or GL.", other),
    };

    // 2. Create Circuit with LieGroup
    let group = LieGroup::new(group_type, config.group_dim);
    let circuit = Circuit::new(group, None);
    let circuit_id = config.circuit_id.unwrap_or_else(|| circuit.id.clone());

    // 3. Create SessionAuth, print token to stderr
    let auth = SessionAuth::new();
    eprintln!("[gravrail-proxy] session token: {}", auth.token());

    // 4. Create watchdog with config intervals
    let watchdog_config = WatchdogConfig {
        interval: Duration::from_millis(config.heartbeat_interval_ms),
        timeout: Duration::from_millis(config.heartbeat_timeout_ms),
    };
    let (watchdog_handle, watchdog_loop) = create_watchdog(watchdog_config);

    // 5. Create ConfinementPipeline
    let pipeline = ConfinementPipeline::new(circuit, config.max_state_norm, config.prove_every_step);

    // 6. Build Arc<ProxyState>
    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: config.upstream_url.clone(),
        client: reqwest::Client::new(),
    });

    // 7. Spawn watchdog future on tokio
    tokio::spawn(watchdog_loop);

    // 8. Spawn proxy heartbeat sender loop
    let heartbeat_state = Arc::clone(&state);
    let heartbeat_interval = Duration::from_millis(config.heartbeat_interval_ms);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(heartbeat_interval);
        loop {
            interval.tick().await;
            if !heartbeat_state.watchdog.is_alive() {
                break;
            }
            if heartbeat_state.watchdog.send_ping().await.is_err() {
                break;
            }
        }
    });

    // 9. Build Axum router
    let router = build_router(Arc::clone(&state));

    // 10. Bind TcpListener and serve
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // 11. Print startup info to stderr
    eprintln!("[gravrail-proxy] circuit: {} ({:?}, dim={})", circuit_id, group_type, config.group_dim);
    eprintln!("[gravrail-proxy] upstream: {}", config.upstream_url);
    eprintln!("[gravrail-proxy] listening on port {}", config.port);
    if config.prove_every_step {
        eprintln!("[gravrail-proxy] STARK proof enabled for every response");
    }
    if let Some(norm) = config.max_state_norm {
        eprintln!("[gravrail-proxy] max state norm: {}", norm);
    }

    // 12. Write PID lock file with exclusive flock
    let lock_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".gravrail");
    std::fs::create_dir_all(&lock_dir)?;
    let lock_path = lock_dir.join("proxy.lock");

    use std::io::Write;
    let lock_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)?;

    // Try exclusive lock (non-blocking)
    use std::os::unix::io::AsRawFd;
    let fd = lock_file.as_raw_fd();
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        anyhow::bail!("Another GravProxy instance is already running (lock file: {})", lock_path.display());
    }
    write!(&lock_file, "{}", std::process::id())?;
    eprintln!("[gravrail-proxy] PID lock: {}", lock_path.display());

    // Keep _lock_file alive so flock persists for the process lifetime
    let _lock_file = lock_file;

    axum::serve(listener, router).await?;

    Ok(())
}
