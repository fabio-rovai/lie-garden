use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("failed to parse config: {}", path.display()))?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub data_dir: String,
    pub max_reachability_depth: usize,
    pub max_reachability_breadth: usize,
    pub step_timeout_ms: u64,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            data_dir: "~/.gravrail".into(),
            max_reachability_depth: 1000,
            max_reachability_breadth: 10000,
            step_timeout_ms: 5000,
        }
    }
}

pub fn expand_tilde(path: &str) -> String {
    if (path.starts_with("~/") || path == "~")
        && let Some(home) = std::env::var_os("HOME")
    {
        return path.replacen("~", &home.to_string_lossy(), 1);
    }
    path.to_string()
}
