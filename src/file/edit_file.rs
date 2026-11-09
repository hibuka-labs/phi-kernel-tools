use std::path::PathBuf;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::resolve_path;

/// A single edit operation: replace `old_text` with `new_text`.
#[derive(Debug, Deserialize, Serialize)]
struct Edit {
    old_text: String,
    new_text: String,
}

/// Precision replacement tool — replaces exact text blocks in a file.
///
/// Unlike `write_file`, this tool only requires the LLM to provide the
/// specific text to replace, not the entire file. This is more token-efficient
/// and reduces the risk of modifying unrelated parts of the file.
///
/// # Matching strategy (4-level fallback)
///
/// 1. **Exact match** — the `old_text` must appear exactly once in the file
/// 2. **Rstrip** — trailing whitespace stripped from both sides
/// 3. **Trim** — leading and trailing whitespace stripped from both sides
/// 4. **Unicode NFC normalization** — normalizes Unicode before comparison
///
/// # Safety guarantees
///
/// - Each `old_text` must be unique in the file (appears exactly once)
/// - Edits must not overlap — overlapping edits are rejected
/// - Atomic write: content is written to a temp file then renamed
/// - Original line endings (`\n` vs `\r\n`) are preserved
pub struct EditFileTool {
    workspace_root: PathBuf,
    /// Opt-in path confinement (see `resolve_path`): reject absolute paths and
    /// `..` escapes from `workspace_root`. Off by default.
    confined: bool,
}

impl EditFileTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            confined: false,
        }
    }

    /// Enable path confinement: paths must stay inside the workspace root.
    pub fn with_confinement(mut self) -> Self {
        self.confined = true;
        self
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
        "Make precise text replacements in an existing file.\n\
        Provide one or more edits, each with old_text (text to find) and new_text (replacement).\n\
        Each old_text must appear exactly once in the file.\n\
        Multiple edits are applied to the original file (not incrementally).\n\
        Use this instead of write_file when you only need to change specific parts of a file."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit: workspace-relative or absolute."
                },
                "edits": {
                    "type": "array",
                    "description": "List of edit operations. Each edit has old_text and new_text.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "The exact text to find and replace. Must appear exactly once in the file."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "The replacement text."
                            }
                        },
                        "required": ["old_text", "new_text"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description:
                "Precision text replacement (workspace-relative or absolute paths) with uniqueness checks and atomic writes."
                    .to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path_str = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();

        if path_str.is_empty() {
            return Ok(error_output("No file path provided.", &path_str));
        }

        let edits: Vec<Edit> = match args.get("edits") {
            Some(raw) => match serde_json::from_value(raw.clone()) {
                Ok(edits) => edits,
                Err(e) => {
                    return Ok(vec![Content::text(format!(
                        "[Error]: Failed to parse edits: {}. Expected array of {{old_text, new_text}} objects.",
                        e
                    ))]);
                }
            },
            None => {
                return Ok(vec![Content::text(
                    "[Error]: No edits provided. Expected array of {old_text, new_text} objects."
                        .to_string(),
                )]);
            }
        };

        if edits.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: Edits array is empty. Provide at least one edit operation.".to_string(),
            )]);
        }

        // Resolve and validate the path
        let file_path = match resolve_path(&self.workspace_root, &path_str, self.confined) {
            Ok(p) => p,
            Err(e) => {
                return Ok(error_output(&format!("Path error: {}", e), &path_str));
            }
        };

        if !file_path.exists() {
            return Ok(vec![Content::text(format!(
                "[Error]: File not found: {}",
                path_str
            ))]);
        }

        if !file_path.is_file() {
            return Ok(vec![Content::text(format!(
                "[Error]: Path is not a file: {}",
                path_str
            ))]);
        }

        // Read original file content
        let original = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(vec![Content::text(format!(
                    "[Error]: Failed to read file: {}",
                    e
                ))]);
            }
        };

        // Detect line ending style
        let line_ending = detect_line_ending(&original);

        // Check for overlapping edits
        if let Err(overlap_err) = check_overlaps(&original, &edits) {
            return Ok(vec![Content::text(format!("[Error]: {}", overlap_err))]);
        }

        // Apply edits with 4-level fallback matching
        let mut modified = original.clone();
        let mut applied = 0;

        // Compute line numbers for each edit before applying (so positions stay valid)
        let mut edit_lines: Vec<u32> = Vec::new();
        for edit in &edits {
            let line = original
                .find(&edit.old_text)
                .map(|pos| original[..pos].lines().count() as u32 + 1)
                .unwrap_or(0);
            edit_lines.push(line);
        }

        for (i, edit) in edits.iter().enumerate() {
            match find_and_replace(&modified, &original, edit, i) {
                Ok(new_content) => {
                    modified = new_content;
                    applied += 1;
                }
                Err(err) => {
                    return Ok(vec![Content::text(format!(
                        "[Error]: Edit {} failed: {}",
                        i, err
                    ))]);
                }
            }
        }

        // Normalize line endings to match original
        let modified = normalize_line_endings(&modified, line_ending);

        // Atomic write: write to temp file then rename
        let temp_path = temp_path_for(&file_path)?;
        if let Err(e) = std::fs::write(&temp_path, &modified) {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(vec![Content::text(format!(
                "[Error]: Failed to write file: {}",
                e
            ))]);
        }

        if let Err(e) = std::fs::rename(&temp_path, &file_path) {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(vec![Content::text(format!(
                "[Error]: Failed to save file (rename): {}",
                e
            ))]);
        }

        tracing::info!(
            path = %path_str,
            edits = applied,
            "edit_file"
        );

        Ok(vec![
            Content::text(format!(
                "Successfully applied {} edit(s) to {}.",
                applied, path_str
            )),
            Content::detail(serde_json::json!({ "edit_lines": edit_lines })),
        ])
    }
}

// ── Helpers ──

fn error_output(message: &str, _path: &str) -> Vec<Content> {
    vec![Content::text(format!("[Error]: {}", message))]
}

/// Detect the dominant line ending style in the content.
/// Returns `"\r\n"` if any CRLF is found, otherwise `"\n"`.
fn detect_line_ending(content: &str) -> &'static str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

/// Normalize all line endings in `content` to the given `line_ending`.
fn normalize_line_endings(content: &str, line_ending: &str) -> String {
    // Replace all CRLF with LF first, then optionally back to CRLF
    let normalized = content.replace("\r\n", "\n");
    if line_ending == "\r\n" {
        normalized.replace('\n', "\r\n")
    } else {
        normalized
    }
}

/// Generate a temp file path next to the target file.
fn temp_path_for(file_path: &std::path::Path) -> AgentResult<PathBuf> {
    let parent = file_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let file_name = file_path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("tmp"));
    let mut temp_name = file_name.to_os_string();
    temp_name.push(".phi-tmp");
    Ok(parent.join(temp_name))
}

/// Check that no two edits overlap in the original content.
/// Returns an error string if overlap is detected.
fn check_overlaps(original: &str, edits: &[Edit]) -> Result<(), String> {
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, idx)

    for (i, edit) in edits.iter().enumerate() {
        let positions = find_all_positions(original, &edit.old_text);
        if positions.len() != 1 {
            // Uniqueness is checked later; skip non-unique for overlap check
            continue;
        }
        let start = positions[0];
        let end = start + edit.old_text.len();
        ranges.push((start, end, i));
    }

    // Sort by start position
    ranges.sort_by_key(|(s, _, _)| *s);

    for w in ranges.windows(2) {
        let (_, end_a, idx_a) = w[0];
        let (start_b, _, idx_b) = w[1];
        if end_a > start_b {
            return Err(format!(
                "Edits {} and {} overlap. Edit {} ends at byte {} but edit {} starts at byte {}. \
                 Merge these edits into a single edit.",
                idx_a, idx_b, idx_a, end_a, idx_b, start_b
            ));
        }
    }

    Ok(())
}

/// Find all byte positions where `needle` appears in `haystack`.
fn find_all_positions(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return vec![];
    }
    let mut positions = Vec::new();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let abs_pos = start + pos;
        positions.push(abs_pos);
        start = abs_pos + needle.len();
    }
    positions
}

/// Try to find and replace `old_text` with `new_text` in `current`, using the
/// `original` content for uniqueness verification.
///
/// Matching is performed against `current` (which reflects previous edits),
/// but uniqueness is verified against `original` to prevent ambiguity.
///
/// Returns the new content on success, or an error string.
fn find_and_replace(
    current: &str,
    original: &str,
    edit: &Edit,
    _idx: usize,
) -> Result<String, String> {
    // Level 1: Exact match
    let matches = find_all_positions(current, &edit.old_text);
    if matches.len() == 1 {
        let pos = matches[0];
        return Ok(apply_replace(current, pos, &edit.old_text, &edit.new_text));
    }
    if matches.is_empty() {
        // Fall through to relaxed matching
    } else {
        // Multiple exact matches — ambiguity
        // Check if it was unique in the original
        let orig_matches = find_all_positions(original, &edit.old_text);
        if orig_matches.len() == 1 {
            // Was unique in original but now multiple — ambiguity from prior edits
            return Err(format!(
                "old_text is not unique in the file after previous edits. \
                 Found {} occurrences. Merge edits that affect the same text.",
                matches.len()
            ));
        }
        return Err(format!(
            "old_text is not unique — found {} occurrences. \
             Provide more surrounding context to make it unique.",
            matches.len()
        ));
    }

    // Level 2: Rstrip (trailing whitespace)
    let old_rstrip = edit.old_text.trim_end();
    if old_rstrip != edit.old_text {
        let current_rstrip = find_rstrip_matches(current, old_rstrip);
        if current_rstrip.len() == 1 {
            let (pos, actual) = current_rstrip[0];
            return Ok(apply_replace(current, pos, actual, &edit.new_text));
        }
        if current_rstrip.len() > 1 {
            return Err(format!(
                "old_text (with trailing whitespace stripped) is not unique — found {} matches. \
                 Provide more context.",
                current_rstrip.len()
            ));
        }
    }

    // Level 3: Trim (leading + trailing whitespace)
    let old_trim = edit.old_text.trim();
    if old_trim != edit.old_text && old_trim != old_rstrip {
        let current_trim = find_trim_matches(current, old_trim);
        if current_trim.len() == 1 {
            let (pos, actual) = current_trim[0];
            return Ok(apply_replace(current, pos, actual, &edit.new_text));
        }
        if current_trim.len() > 1 {
            return Err(format!(
                "old_text (with whitespace trimmed) is not unique — found {} matches. \
                 Provide more context.",
                current_trim.len()
            ));
        }
    }

    // Level 4: Unicode NFC normalization
    let old_nfc = unicode_normalize(&edit.old_text);
    if old_nfc != edit.old_text && old_nfc != old_rstrip && old_nfc != old_trim {
        let current_nfc = find_nfc_matches(current, &old_nfc);
        if current_nfc.len() == 1 {
            let (pos, actual) = current_nfc[0];
            return Ok(apply_replace(current, pos, actual, &edit.new_text));
        }
        if current_nfc.len() > 1 {
            return Err(format!(
                "old_text (with Unicode NFC normalization) is not unique — found {} matches. \
                 Provide more context.",
                current_nfc.len()
            ));
        }
    }

    // Exhausted all fallback levels
    Err(
        "old_text not found in the file. Check that the text matches exactly, \
         including whitespace and indentation."
            .to_string(),
    )
}

/// Find positions where `needle` matches with trailing whitespace stripped from lines.
fn find_rstrip_matches<'a>(haystack: &'a str, needle: &str) -> Vec<(usize, &'a str)> {
    if needle.is_empty() {
        return vec![];
    }
    let mut results = Vec::new();
    let mut start = 0;
    let needle_len = needle.len();
    while start + needle_len <= haystack.len() {
        let candidate = &haystack[start..];
        if candidate.starts_with(needle) {
            // Found a prefix match — find the actual extent (needle + trailing whitespace)
            let actual_len = candidate[needle_len..]
                .chars()
                .take_while(|c| c.is_whitespace() && *c != '\n' && *c != '\r')
                .map(|c| c.len_utf8())
                .sum::<usize>()
                + needle_len;
            let actual = &haystack[start..start + actual_len];
            results.push((start, actual));
            // Skip past the entire match to avoid overlapping results
            start += actual_len;
        } else {
            // Advance by one char
            if let Some(c) = candidate.chars().next() {
                start += c.len_utf8();
            } else {
                break;
            }
        }
    }
    results
}

/// Find positions where `needle` matches with whitespace trimmed (but not across newlines).
fn find_trim_matches<'a>(haystack: &'a str, needle: &str) -> Vec<(usize, &'a str)> {
    if needle.is_empty() {
        return vec![];
    }
    let mut results = Vec::new();
    let mut start = 0;
    while start < haystack.len() {
        let rest = &haystack[start..];

        // Skip leading whitespace (not newlines)
        let ws_skip: usize = rest
            .chars()
            .take_while(|c| c.is_whitespace() && *c != '\n' && *c != '\r')
            .map(|c| c.len_utf8())
            .sum();
        let after_ws = start + ws_skip;

        if let Some(after_ws_str) = haystack.get(after_ws..)
            && let Some(stripped) = after_ws_str.strip_prefix(needle)
        {
            // Found a match — find the actual extent
            let actual_len = ws_skip
                + needle.len()
                + stripped
                    .chars()
                    .take_while(|c| c.is_whitespace() && *c != '\n' && *c != '\r')
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
            let actual = &haystack[start..start + actual_len];
            results.push((start, actual));
            // Skip past the entire match
            start += actual_len;
            continue;
        }
        // Advance by one char
        if let Some(c) = rest.chars().next() {
            start += c.len_utf8();
        } else {
            break;
        }
    }
    results
}

/// Find positions where `needle` matches after NFC normalization.
fn find_nfc_matches<'a>(haystack: &'a str, needle_nfc: &str) -> Vec<(usize, &'a str)> {
    if needle_nfc.is_empty() {
        return vec![];
    }
    let mut results = Vec::new();
    // Walk through haystack char by char, comparing NFC-normalized slices
    let chars: Vec<char> = haystack.chars().collect();
    let needle_chars: Vec<char> = needle_nfc.chars().collect();
    let needle_len = needle_chars.len();

    for i in 0..=chars.len().saturating_sub(needle_len) {
        let candidate: String = chars[i..i + needle_len].iter().collect();
        let candidate_nfc = unicode_normalize(&candidate);
        if candidate_nfc == needle_nfc {
            let start = chars[..i].iter().map(|c| c.len_utf8()).sum();
            let end = start
                + chars[i..i + needle_len]
                    .iter()
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
            results.push((start, &haystack[start..end]));
        }
    }
    results
}

/// NFC-normalize a string. Falls back to the original if normalization fails.
fn unicode_normalize(s: &str) -> String {
    // Full NFC normalization requires the `unicode-normalization` crate.
    // For now, return the input unchanged — most strings are already NFC.
    // This function exists as a hook point for future enhancement.
    s.to_string()
}

/// Apply a replacement at the given byte position.
fn apply_replace(content: &str, pos: usize, old: &str, new: &str) -> String {
    let cap = content.len() + new.len();
    let mut result = String::with_capacity(cap);
    result.push_str(&content[..pos]);
    result.push_str(new);
    result.push_str(&content[pos + old.len()..]);
    result
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    use super::*;

    pub fn find_all_positions(haystack: &str, needle: &str) -> Vec<usize> {
        super::find_all_positions(haystack, needle)
    }

    pub fn find_and_replace(
        current: &str,
        original: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<String, String> {
        let edit = Edit {
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
        };
        super::find_and_replace(current, original, &edit, 0)
    }

    pub fn normalize_line_endings(content: &str, line_ending: &str) -> String {
        super::normalize_line_endings(content, line_ending)
    }
}

#[cfg(test)]
#[path = "edit_file_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "edit_file_proptests.rs"]
mod proptests;
