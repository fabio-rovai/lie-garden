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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
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
    }
    Ok(())
}
