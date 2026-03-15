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
    /// Port to bind. Use 0 to let the OS assign a free port.
    pub port: u16,
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub max_state_norm: Option<f64>,
    pub prove_every_step: bool,
    /// Holonomy sliding window size.
    pub holonomy_window_size: usize,
    /// If true, write {"port":N,"token":"..."} to stdout once the server is ready.
    pub json_startup: bool,
    /// Output-side holonomy threshold. 0.0 = disabled.
    pub holonomy_threshold: f64,
    /// Input-side holonomy threshold. 0.0 = disabled.
    pub input_holonomy_threshold: f64,
    /// Input-side norm threshold. 0.0 = disabled.
    pub input_norm_threshold: f64,
}

/// Run the proxy server with the given configuration, calling `token_cb` with
/// the session token before the server starts accepting connections.
///
/// This variant is primarily intended for integration tests that need to
/// capture the session token programmatically rather than reading it from
/// stderr.
pub async fn run_proxy_returning_token(
    config: ProxyConfig,
    token_cb: impl FnOnce(String) + Send + 'static,
) -> anyhow::Result<()> {
    // 1. Parse group_type string to GroupType enum
    let group_type = match config.group_type.to_uppercase().as_str() {
        "SO" => GroupType::SO,
        "SE" => GroupType::SE,
        "GL" => GroupType::GL,
        other => anyhow::bail!("Unknown group type '{}'. Expected SO, SE, or GL.", other),
    };

    // Load user config (CLI flags take precedence over config file)
    let file_cfg = crate::config::Config::load_user_default();
    let pfc = file_cfg.proxy;

    let effective_max_state_norm = config.max_state_norm.or(pfc.max_state_norm);
    let effective_holonomy_threshold = if config.holonomy_threshold > 0.0 { config.holonomy_threshold } else { pfc.holonomy_threshold };
    let effective_input_holonomy_threshold = if config.input_holonomy_threshold > 0.0 { config.input_holonomy_threshold } else { pfc.input_holonomy_threshold };
    let effective_input_norm_threshold = if config.input_norm_threshold > 0.0 { config.input_norm_threshold } else { pfc.input_norm_threshold };
    let effective_window_size = if config.holonomy_window_size != 8 { config.holonomy_window_size } else { if pfc.holonomy_window != 0 { pfc.holonomy_window } else { 8 } };

    // 2. Create Circuit with LieGroup
    let group = LieGroup::new(group_type, config.group_dim);
    let circuit = Circuit::new(group, None);
    let circuit_id = config.circuit_id.unwrap_or_else(|| circuit.id.clone());

    // 3. Create SessionAuth, fire callback with token, then print to stderr
    let auth = SessionAuth::new();
    let token = auth.token().to_owned();
    token_cb(token.clone());
    eprintln!("[gravrail-proxy] session token: {}", token);

    // 4. Create watchdog with config intervals
    let watchdog_config = WatchdogConfig {
        interval: Duration::from_millis(config.heartbeat_interval_ms),
        timeout: Duration::from_millis(config.heartbeat_timeout_ms),
    };
    let (watchdog_handle, watchdog_loop) = create_watchdog(watchdog_config);

    // 5. Create ConfinementPipeline
    let pipeline = ConfinementPipeline::new(circuit, effective_max_state_norm, config.prove_every_step, effective_window_size);

    // 6. Build Arc<ProxyState>
    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: config.upstream_url.clone(),
        client: reqwest::Client::new(),
        holonomy_threshold: effective_holonomy_threshold,
        input_holonomy_threshold: effective_input_holonomy_threshold,
        input_norm_threshold: effective_input_norm_threshold,
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

    axum::serve(listener, router).await?;

    Ok(())
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

    // Load user config (CLI flags take precedence over config file)
    let file_cfg = crate::config::Config::load_user_default();
    let pfc = file_cfg.proxy;

    let effective_max_state_norm = config.max_state_norm.or(pfc.max_state_norm);
    let effective_holonomy_threshold = if config.holonomy_threshold > 0.0 { config.holonomy_threshold } else { pfc.holonomy_threshold };
    let effective_input_holonomy_threshold = if config.input_holonomy_threshold > 0.0 { config.input_holonomy_threshold } else { pfc.input_holonomy_threshold };
    let effective_input_norm_threshold = if config.input_norm_threshold > 0.0 { config.input_norm_threshold } else { pfc.input_norm_threshold };
    let effective_window_size = if config.holonomy_window_size != 8 { config.holonomy_window_size } else { if pfc.holonomy_window != 0 { pfc.holonomy_window } else { 8 } };

    // 2. Create Circuit with LieGroup
    let group = LieGroup::new(group_type, config.group_dim);
    let circuit = Circuit::new(group, None);
    let circuit_id = config.circuit_id.unwrap_or_else(|| circuit.id.clone());

    // 3. Create SessionAuth, save token string before auth is moved into state
    let auth = SessionAuth::new();
    let token_str = auth.token().to_owned();
    eprintln!("[gravrail-proxy] session token: {}", token_str);

    // 4. Create watchdog with config intervals
    let watchdog_config = WatchdogConfig {
        interval: Duration::from_millis(config.heartbeat_interval_ms),
        timeout: Duration::from_millis(config.heartbeat_timeout_ms),
    };
    let (watchdog_handle, watchdog_loop) = create_watchdog(watchdog_config);

    // 5. Create ConfinementPipeline
    let pipeline = ConfinementPipeline::new(circuit, effective_max_state_norm, config.prove_every_step, effective_window_size);

    // 6. Build Arc<ProxyState>
    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: config.upstream_url.clone(),
        client: reqwest::Client::new(),
        holonomy_threshold: effective_holonomy_threshold,
        input_holonomy_threshold: effective_input_holonomy_threshold,
        input_norm_threshold: effective_input_norm_threshold,
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

    // 10. Bind TcpListener and serve (port 0 → OS assigns free port)
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let actual_port = listener.local_addr()?.port();

    // 11. Print startup info to stderr
    eprintln!("[gravrail-proxy] circuit: {} ({:?}, dim={})", circuit_id, group_type, config.group_dim);
    eprintln!("[gravrail-proxy] upstream: {}", config.upstream_url);
    eprintln!("[gravrail-proxy] listening on port {}", actual_port);
    if config.prove_every_step {
        eprintln!("[gravrail-proxy] STARK proof enabled for every response");
    }
    if let Some(norm) = config.max_state_norm {
        eprintln!("[gravrail-proxy] max state norm: {}", norm);
    }

    // If json_startup: write machine-readable startup info to stdout so callers
    // (e.g. VS Code plugins spawning the proxy as a subprocess) can parse port + token.
    if config.json_startup {
        println!(
            "{{\"port\":{},\"token\":\"{}\"}}",
            actual_port, token_str
        );
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
