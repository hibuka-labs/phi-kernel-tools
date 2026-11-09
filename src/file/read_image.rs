//! `read_image` — one entry point for a picture the model should look at,
//! whether it lives on disk or behind an http(s) URL.
//!
//! The result is the vision channel's native shape: a metadata text line
//! (dimensions, format, size) plus a `Content::Image` carrying the bytes as
//! base64 — the engine's image budget shrinks and judges it on the way to
//! the wire. Three guards sit here:
//!
//! - **Type is sniffed, not named**: magic bytes decide PNG/JPEG; an HTML
//!   page wearing `image/png` is refused, a `.jpg` holding a PNG is served
//!   as the PNG it is.
//! - **URLs are fetched SSRF-safe** ([`url_fetch`]): internal addresses and
//!   redirects into them are refused, 30 s and 20 MB ceilings enforced.
//! - **The capability gate is honest**: a model without vision gets the
//!   metadata and an explicit "switch with /model" — the picture is never
//!   attached and silently unreadable (and never silently dropped: the
//!   degradation SAYS there was a picture).

use std::path::PathBuf;

use agent_base::image_ops::{header_dimensions, sniff_mime, to_base64};
use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::resolve_path;
use super::url_fetch::{self, FetchPolicy, MAX_DOWNLOAD_BYTES};

/// Reads one image (PNG/JPEG) from a local path or an http(s) URL and hands
/// it to the model through the vision channel.
pub struct ReadImageTool {
    workspace_root: PathBuf,
    /// Opt-in path confinement (same contract as the other file tools).
    confined: bool,
    /// SSRF fetch policy — default-strict; the loopback knob is a test seam.
    fetch_policy: FetchPolicy,
}

impl ReadImageTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            confined: false,
            fetch_policy: FetchPolicy::default(),
        }
    }

    /// Enable path confinement: local paths must stay inside the workspace.
    pub fn with_confinement(mut self) -> Self {
        self.confined = true;
        self
    }

    /// TEST SEAM: relax the SSRF policy for fixture servers. Never called
    /// in production wiring.
    pub fn with_fetch_policy(mut self, policy: FetchPolicy) -> Self {
        self.fetch_policy = policy;
        self
    }
}

/// Human byte count for the metadata line ("2.1 MB", "350 KB").
fn pretty_bytes(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0} KB", n as f64 / 1_000.0)
    } else {
        format!("{n} B")
    }
}

/// Load the raw bytes of one picture: local file (respects confinement) or
/// http(s) URL (SSRF-safe). `source` is what the metadata line shows.
async fn load_bytes(tool: &ReadImageTool, path: &str) -> Result<(Vec<u8>, String), String> {
    if path.contains("://") {
        // Anything that looks like a URL goes to the URL face — which
        // refuses non-http(s) schemes. A `file://` or `ftp://` target is
        // never read as a local path by accident.
        let fetched = url_fetch::fetch_image(path, tool.fetch_policy).await?;
        return Ok((fetched.bytes, fetched.final_url));
    }
    let file_path = resolve_path(&tool.workspace_root, path, tool.confined)?;
    if !file_path.exists() {
        return Err(format!("File not found: {path}"));
    }
    if !file_path.is_file() {
        return Err(format!("Path is not a file: {path}"));
    }
    let meta = std::fs::metadata(&file_path).map_err(|e| format!("Cannot stat '{path}': {e}"))?;
    if meta.len() as usize > MAX_DOWNLOAD_BYTES {
        return Err(format!(
            "Image refused: '{}' is {} which exceeds the {} byte cap",
            path,
            pretty_bytes(meta.len() as usize),
            MAX_DOWNLOAD_BYTES
        ));
    }
    let bytes = std::fs::read(&file_path).map_err(|e| format!("Failed to read '{path}': {e}"))?;
    Ok((bytes, path.to_string()))
}

/// The one text line every outcome carries: what the picture is.
fn metadata_line(source: &str, dims: (u32, u32), mime: &str, size: usize) -> String {
    let format = if mime == "image/png" { "PNG" } else { "JPEG" };
    format!(
        "Image: {}x{} {}, {} ({})",
        dims.0,
        dims.1,
        format,
        pretty_bytes(size),
        source
    )
}

#[async_trait]
impl Tool for ReadImageTool {
    fn name(&self) -> &'static str {
        "read_image"
    }

    fn description(&self) -> &'static str {
        "Look at one image: a local path (workspace-relative or absolute) or an http(s) URL. Returns the picture for you to see plus a metadata line (WxH, format, size). PNG and JPEG only — the type is decided by the file's magic bytes, not its name. Use this to inspect screenshots, diagrams, photos or any visual material."
    }

    fn schema(&self) -> Value {
        let mut props = serde_json::Map::new();
        props.insert(
            "path".to_string(),
            json!({
                "type": "string",
                "description": "The image to look at: a local path (e.g. 'screenshots/error.png' or '/abs/path/photo.jpg') or an http(s) URL (e.g. 'https://example.com/diagram.png')."
            }),
        );
        json!({
            "type": "object",
            "properties": props,
            "required": ["path"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description:
                "Look at one image (local path or http(s) URL) through the vision channel."
                    .to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if path.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No image path provided.".to_string(),
            )]);
        }

        let (bytes, source) = match load_bytes(self, &path).await {
            Ok(loaded) => loaded,
            Err(e) => return Ok(vec![Content::text(format!("[Error]: {e}"))]),
        };

        // Type is decided by magic bytes: an HTML page wearing image/png is
        // refused; a .jpg holding a PNG is served as the PNG it is.
        let Some(mime) = sniff_mime(&bytes) else {
            return Ok(vec![Content::text(format!(
                "[Error]: '{source}' is not a PNG/JPEG image (magic bytes do not match) — refusing to attach it."
            ))]);
        };
        let Some(dims) = header_dimensions(&bytes) else {
            return Ok(vec![Content::text(format!(
                "[Error]: '{source}' has an unreadable image header — refusing to attach it."
            ))]);
        };

        let meta = metadata_line(&source, dims, mime, bytes.len());
        tracing::info!(source = %source, width = dims.0, height = dims.1, mime, size = bytes.len(), "read_image");

        // Capability gate: a model without vision can only read words. The
        // degradation says the picture exists and how to get it seen — the
        // attachment is withheld, never silently dropped.
        if let Some(provider) = ctx.llm_client.as_ref()
            && !provider.capabilities().supports_vision
        {
            return Ok(vec![Content::text(format!(
                "{meta}\n[this model cannot see images, so the picture is NOT attached — \
                 switch to a vision model with /model and call read_image again]"
            ))]);
        }

        Ok(vec![
            Content::text(meta),
            Content::image(to_base64(&bytes), mime),
        ])
    }
}

#[cfg(test)]
#[path = "read_image_tests.rs"]
mod tests;
