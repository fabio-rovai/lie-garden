// Heartbeat protocol and mutual liveness detection
//
// The watchdog provides HMAC-signed heartbeats between proxy and watchdog task.
// If either side stops responding within the timeout, `killed` is set and the
// system shuts down. This prevents silent proxy termination to bypass confinement.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

type HmacSha256 = Hmac<Sha256>;

/// An HMAC-signed heartbeat message.
#[derive(Debug, Clone)]
pub struct Heartbeat {
    pub seq: u64,
    pub timestamp_ms: u64,
    pub signature: Vec<u8>,
}

/// Configuration for the watchdog timing.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// How often to send heartbeats.
    pub interval: Duration,
    /// How long to wait without a ping before declaring death.
    pub timeout: Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            timeout: Duration::from_secs(3),
        }
    }
}

/// Proxy-side handle to the watchdog.
///
/// The proxy holds this to check liveness, send pings, and kill the watchdog.
/// Dropping all senders (or calling `kill()`) causes the watchdog loop to exit.
pub struct WatchdogHandle {
    pub killed: Arc<AtomicBool>,
    pub last_watchdog_seq: Arc<AtomicU64>,
    /// Sender for proxy -> watchdog pings.
    pub ping_tx: mpsc::Sender<Heartbeat>,
    /// Receiver for watchdog -> proxy heartbeats.
    pub heartbeat_rx: mpsc::Receiver<Heartbeat>,
    /// HMAC key shared with the watchdog.
    pub hmac_key: Vec<u8>,
}

impl WatchdogHandle {
    /// Returns `true` if the watchdog is still alive.
    pub fn is_alive(&self) -> bool {
        !self.killed.load(Ordering::SeqCst)
    }

    /// Kill the watchdog immediately.
    pub fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
    }

    /// Send a signed ping to the watchdog to prove the proxy is alive.
    pub async fn send_ping(&self) -> Result<(), mpsc::error::SendError<Heartbeat>> {
        let seq = self.last_watchdog_seq.load(Ordering::SeqCst);
        let hb = sign_heartbeat(seq, &self.hmac_key);
        self.ping_tx.send(hb).await
    }

    /// Try to receive the next heartbeat (non-blocking).
    pub async fn recv_heartbeat(&mut self) -> Option<Heartbeat> {
        self.heartbeat_rx.recv().await
    }

    /// Get the shared HMAC key for verification.
    pub fn hmac_key(&self) -> &[u8] {
        &self.hmac_key
    }

    /// Current watchdog sequence number.
    pub fn last_watchdog_seq(&self) -> u64 {
        self.last_watchdog_seq.load(Ordering::SeqCst)
    }
}

/// Sign a heartbeat with HMAC-SHA256.
///
/// The signed message is `seq || timestamp_ms` as little-endian bytes.
pub fn sign_heartbeat(seq: u64, key: &[u8]) -> Heartbeat {
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut mac =
        HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(&seq.to_le_bytes());
    mac.update(&timestamp_ms.to_le_bytes());
    let signature = mac.finalize().into_bytes().to_vec();

    Heartbeat {
        seq,
        timestamp_ms,
        signature,
    }
}

/// Verify a heartbeat's HMAC-SHA256 signature.
pub fn verify_heartbeat(hb: &Heartbeat, key: &[u8]) -> bool {
    let mut mac =
        HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(&hb.seq.to_le_bytes());
    mac.update(&hb.timestamp_ms.to_le_bytes());
    mac.verify_slice(&hb.signature).is_ok()
}

/// Create a watchdog pair: a handle for the proxy and a future for the watchdog loop.
///
/// The returned future must be spawned on a tokio runtime. It will run until:
/// - The proxy stops pinging (timeout exceeded)
/// - The proxy drops its handle (channel closes)
/// - `kill()` is called
pub fn create_watchdog(
    config: WatchdogConfig,
) -> (WatchdogHandle, impl std::future::Future<Output = ()>) {
    let killed = Arc::new(AtomicBool::new(false));
    let sequence = Arc::new(AtomicU64::new(0));

    // Generate a random HMAC key.
    let hmac_key: Vec<u8> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..32).map(|_| rng.r#gen()).collect()
    };

    // proxy -> watchdog pings (now HMAC-signed Heartbeats)
    let (ping_tx, mut ping_rx) = mpsc::channel::<Heartbeat>(16);
    // watchdog -> proxy heartbeats
    let (heartbeat_tx, heartbeat_rx) = mpsc::channel::<Heartbeat>(16);

    let handle = WatchdogHandle {
        killed: Arc::clone(&killed),
        last_watchdog_seq: Arc::clone(&sequence),
        ping_tx,
        heartbeat_rx,
        hmac_key: hmac_key.clone(),
    };

    let loop_killed = Arc::clone(&killed);
    let loop_sequence = Arc::clone(&sequence);

    let watchdog_loop = async move {
        let mut last_ping = Instant::now();
        let mut interval = tokio::time::interval(config.interval);
        // Don't pile up ticks if we're slow.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            // Check kill flag.
            if loop_killed.load(Ordering::SeqCst) {
                break;
            }

            tokio::select! {
                _ = interval.tick() => {
                    // Check timeout.
                    if last_ping.elapsed() > config.timeout {
                        loop_killed.store(true, Ordering::SeqCst);
                        break;
                    }

                    // Send heartbeat.
                    let seq = loop_sequence.fetch_add(1, Ordering::SeqCst);
                    let hb = sign_heartbeat(seq, &hmac_key);
                    if heartbeat_tx.send(hb).await.is_err() {
                        // Proxy dropped its receiver — proxy is dead.
                        loop_killed.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                result = ping_rx.recv() => {
                    match result {
                        Some(hb) => {
                            // Only reset timer if HMAC signature is valid.
                            if verify_heartbeat(&hb, &hmac_key) {
                                last_ping = Instant::now();
                            }
                        }
                        None => {
                            // Channel closed — proxy dropped its sender.
                            loop_killed.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            }
        }
    };

    (handle, watchdog_loop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_heartbeat() {
        let key = b"test-secret-key-for-hmac";
        let hb = sign_heartbeat(42, key);
        assert_eq!(hb.seq, 42);
        assert!(verify_heartbeat(&hb, key));
    }

    #[test]
    fn test_tampered_heartbeat_rejected() {
        let key = b"test-secret-key-for-hmac";
        let mut hb = sign_heartbeat(1, key);
        // Tamper with the sequence number.
        hb.seq = 999;
        assert!(!verify_heartbeat(&hb, key));
    }

    #[test]
    fn test_wrong_key_rejected() {
        let key_a = b"key-alpha-aaaaaaaaaaaaaa";
        let key_b = b"key-bravo-bbbbbbbbbbbbbb";
        let hb = sign_heartbeat(7, key_a);
        assert!(!verify_heartbeat(&hb, key_b));
    }

    #[tokio::test]
    async fn test_watchdog_starts_alive() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(300),
        };
        let (handle, _loop_fut) = create_watchdog(config);
        assert!(handle.is_alive());
    }

    #[tokio::test]
    async fn test_watchdog_kill() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(300),
        };
        let (handle, _loop_fut) = create_watchdog(config);
        assert!(handle.is_alive());
        handle.kill();
        assert!(!handle.is_alive());
    }

    #[tokio::test]
    async fn test_watchdog_detects_proxy_death() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(150),
        };
        let (handle, loop_fut) = create_watchdog(config);
        let killed = Arc::clone(&handle.killed);

        // Spawn the watchdog loop.
        let join = tokio::spawn(loop_fut);

        // Drop the handle — this closes the ping sender and heartbeat receiver.
        drop(handle);

        // The watchdog loop should detect the closed channel and exit.
        tokio::time::timeout(Duration::from_millis(500), join)
            .await
            .expect("watchdog should exit promptly")
            .expect("watchdog task should not panic");

        assert!(killed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_mutual_heartbeat_keeps_alive() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(300),
        };
        let (mut handle, loop_fut) = create_watchdog(config);

        // Spawn the watchdog loop.
        let join = tokio::spawn(loop_fut);

        // Exchange heartbeats for a while.
        for _ in 0..6 {
            tokio::time::sleep(Duration::from_millis(40)).await;
            // Proxy sends ping.
            handle.send_ping().await.expect("ping should succeed");
            // Proxy receives heartbeat (drain available).
            while let Ok(hb) = handle.heartbeat_rx.try_recv() {
                assert!(verify_heartbeat(&hb, handle.hmac_key()));
            }
        }

        // Should still be alive after active exchange.
        assert!(handle.is_alive());

        // Clean shutdown.
        handle.kill();
        tokio::time::timeout(Duration::from_millis(500), join)
            .await
            .expect("watchdog should exit")
            .expect("watchdog task should not panic");
    }
}
