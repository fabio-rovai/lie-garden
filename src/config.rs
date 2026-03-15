use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
    pub proxy: ProxyFileConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            proxy: ProxyFileConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("failed to parse config: {}", path.display()))?;
        Ok(config)
    }

    /// Load from `~/.gravrail/config.toml`. Returns `Config::default()` if the
    /// file does not exist or cannot be parsed.
    pub fn load_user_default() -> Self {
        let path = match dirs::home_dir() {
            Some(h) => h.join(".gravrail").join("config.toml"),
            None => return Config::default(),
        };
        if !path.exists() {
            return Config::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
            Err(_) => Config::default(),
        }
    }

    /// Write (or replace) the `[proxy]` section in `~/.gravrail/config.toml`
    /// without destroying other sections.
    pub fn write_proxy_section(proxy: &ProxyFileConfig) -> Result<()> {
        let dir = dirs::home_dir()
            .context("cannot determine home directory")?
            .join(".gravrail");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create directory: {}", dir.display()))?;
        let path = dir.join("config.toml");

        let existing = if path.exists() {
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?
        } else {
            String::new()
        };

        let mut stripped = remove_toml_section(&existing, "proxy");
        // Ensure a trailing newline before appending
        if !stripped.is_empty() && !stripped.ends_with('\n') {
            stripped.push('\n');
        }

        // Serialise proxy config to TOML key=value pairs
        let proxy_toml = build_proxy_toml(proxy);
        stripped.push_str("[proxy]\n");
        stripped.push_str(&proxy_toml);

        std::fs::write(&path, &stripped)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }
}

/// Remove a `[section]` block (and all its key/value lines) from TOML content.
fn remove_toml_section(content: &str, section: &str) -> String {
    let header = format!("[{}]", section);
    let mut result = Vec::new();
    let mut in_section = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == header {
            in_section = true;
            continue;
        }
        if in_section && trimmed.starts_with('[') {
            in_section = false;
        }
        if !in_section {
            result.push(line);
        }
    }

    // Re-join preserving original line endings
    let mut out = result.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Build TOML key=value lines for `ProxyFileConfig` without the section header.
fn build_proxy_toml(p: &ProxyFileConfig) -> String {
    let mut lines = Vec::new();
    if let Some(v) = p.max_state_norm {
        lines.push(format!("max_state_norm = {}", v));
    }
    lines.push(format!("holonomy_threshold = {}", p.holonomy_threshold));
    lines.push(format!("input_holonomy_threshold = {}", p.input_holonomy_threshold));
    lines.push(format!("input_norm_threshold = {}", p.input_norm_threshold));
    lines.push(format!("holonomy_window = {}", p.holonomy_window));
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(default)]
pub struct ProxyFileConfig {
    pub max_state_norm: Option<f64>,
    pub holonomy_threshold: f64,
    pub input_holonomy_threshold: f64,
    pub input_norm_threshold: f64,
    pub holonomy_window: usize,
}

impl Default for ProxyFileConfig {
    fn default() -> Self {
        Self {
            max_state_norm: None,
            holonomy_threshold: 0.0,
            input_holonomy_threshold: 0.0,
            input_norm_threshold: 0.0,
            holonomy_window: 8,
        }
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_proxy_config_load() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"
[proxy]
max_state_norm = 3.14
holonomy_threshold = 0.5
input_holonomy_threshold = 0.25
input_norm_threshold = 1.5
holonomy_window = 12
"#
        )
        .unwrap();

        let cfg = Config::load(f.path()).unwrap();
        let p = &cfg.proxy;
        assert!((p.max_state_norm.unwrap() - 3.14).abs() < 1e-9);
        assert!((p.holonomy_threshold - 0.5).abs() < 1e-9);
        assert!((p.input_holonomy_threshold - 0.25).abs() < 1e-9);
        assert!((p.input_norm_threshold - 1.5).abs() < 1e-9);
        assert_eq!(p.holonomy_window, 12);
    }

    #[test]
    fn test_proxy_config_defaults() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "").unwrap();

        let cfg = Config::load(f.path()).unwrap();
        let p = &cfg.proxy;
        assert!(p.max_state_norm.is_none());
        assert_eq!(p.holonomy_threshold, 0.0);
        assert_eq!(p.input_holonomy_threshold, 0.0);
        assert_eq!(p.input_norm_threshold, 0.0);
        assert_eq!(p.holonomy_window, 8);
    }

    #[test]
    fn test_remove_toml_section_removes_proxy() {
        let content = "[general]\ndata_dir = \"~/.gravrail\"\n\n[proxy]\nholonomy_threshold = 0.5\n\n[other]\nfoo = 1\n";
        let result = remove_toml_section(content, "proxy");
        assert!(!result.contains("[proxy]"));
        assert!(!result.contains("holonomy_threshold"));
        assert!(result.contains("[general]"));
        assert!(result.contains("[other]"));
    }

    #[test]
    fn test_remove_toml_section_no_section() {
        let content = "[general]\ndata_dir = \"~/.gravrail\"\n";
        let result = remove_toml_section(content, "proxy");
        assert!(result.contains("[general]"));
    }
}
