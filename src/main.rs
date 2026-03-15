use clap::{Parser, Subcommand};
use gravrail::config::expand_tilde;
use gravrail::server::GravRailServer;
use gravrail::state::StateDb;
use rmcp::ServiceExt;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gravrail", about = "Deterministic geometric confinement for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize data directory and database
    Init {
        #[arg(long, default_value = "~/.gravrail")]
        data_dir: String,
    },
    /// Start MCP server (stdio)
    Serve {
        #[arg(long, default_value = "~/.gravrail")]
        data_dir: String,
    },
    /// Interactive chat session with a confined LLM.
    Chat {
        /// Upstream LLM URL (OpenAI-compatible)
        #[arg(long, default_value = "http://localhost:11434/v1")]
        upstream: String,

        /// Model name
        #[arg(long, default_value = "llama3")]
        model: String,

        /// Lie group type: SO, SE, or GL
        #[arg(long, default_value = "SO")]
        group_type: String,

        /// Group dimension
        #[arg(long, default_value_t = 3usize)]
        group_dim: usize,

        /// Maximum state norm for reachability (None = unlimited)
        #[arg(long)]
        max_state_norm: Option<f64>,

        /// Generate STARK proof for every response
        #[arg(long, default_value_t = false)]
        prove: bool,
    },
    /// Start confinement proxy in front of an LLM API
    Proxy {
        /// Upstream LLM API URL (e.g. http://localhost:11434)
        #[arg(long)]
        upstream: String,

        /// Port for the proxy to listen on
        #[arg(long, default_value_t = 8340)]
        port: u16,

        /// Lie group type: SO, SE, or GL
        #[arg(long, default_value = "SO")]
        group_type: String,

        /// Group matrix dimension
        #[arg(long, default_value_t = 3)]
        group_dim: usize,

        /// Heartbeat interval in milliseconds
        #[arg(long, default_value_t = 1000)]
        heartbeat_interval: u64,

        /// Heartbeat timeout in milliseconds
        #[arg(long, default_value_t = 3000)]
        heartbeat_timeout: u64,

        /// Maximum state norm (reachability bound)
        #[arg(long)]
        max_state_norm: Option<f64>,

        /// Generate STARK proof for every response
        #[arg(long, default_value_t = false)]
        prove: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat { upstream, model, group_type, group_dim, max_state_norm, prove } => {
            gravrail::chat::run_chat(gravrail::chat::ChatConfig {
                upstream_url: upstream,
                model,
                group_type,
                group_dim,
                max_state_norm,
                prove_every_step: prove,
            })
            .await?;
        }
        Commands::Init { data_dir } => {
            let path = PathBuf::from(expand_tilde(&data_dir));
            std::fs::create_dir_all(&path)?;
            let _db = StateDb::open(&path.join("state.db"))?;
            println!("GravRail initialized at {}", path.display());
        }
        Commands::Serve { data_dir } => {
            let path = PathBuf::from(expand_tilde(&data_dir));
            std::fs::create_dir_all(&path)?;
            let db = StateDb::open(&path.join("state.db"))?;
            let server = GravRailServer::new(db);
            eprintln!("GravRail MCP server starting ({} tools registered)", server.list_tool_definitions().len());
            let service = server.serve(rmcp::transport::stdio()).await?;
            service.waiting().await?;
        }
        Commands::Proxy {
            upstream,
            port,
            group_type,
            group_dim,
            heartbeat_interval,
            heartbeat_timeout,
            max_state_norm,
            prove,
        } => {
            let config = gravrail::proxy::cli::ProxyConfig {
                circuit_id: None,
                group_type,
                group_dim,
                upstream_url: upstream,
                port,
                heartbeat_interval_ms: heartbeat_interval,
                heartbeat_timeout_ms: heartbeat_timeout,
                max_state_norm,
                prove_every_step: prove,
            };
            gravrail::proxy::cli::run_proxy(config).await?;
        }
    }
    Ok(())
}
