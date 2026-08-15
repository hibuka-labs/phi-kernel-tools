use std::process::Stdio;
use std::time::Duration;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Local shell command execution tool.
///
/// Executes arbitrary commands via `sh -c`, with support for timeout,
/// cancellation, and working directory.
pub struct LocalShellTool {
    timeout_ms: u64,
}

impl LocalShellTool {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }
}

fn format_result(
    command: &str,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    timed_out: bool,
) -> String {
    let stdout = stdout.trim();
    let stderr = stderr.trim();

    if timed_out {
        return format!(
            "[Command Timed Out]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        );
    }

    match exit_code {
        Some(0) => match (stdout.is_empty(), stderr.is_empty()) {
            (true, true) => "Command executed successfully with no output.".to_string(),
            (false, true) => stdout.to_string(),
            (true, false) => format!("stderr:\n{}", stderr),
            (false, false) => format!("stdout:\n{}\n\nstderr:\n{}", stdout, stderr),
        },
        Some(code) => format!(
            "[Command Failed (exit code: {})]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            code,
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        ),
        None => format!(
            "[Command Terminated]\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
            command,
            if stdout.is_empty() { "(empty)" } else { stdout },
            if stderr.is_empty() { "(empty)" } else { stderr },
        ),
    }
}

/// Truncate a shell output string to the per-call output budget
/// (`ToolContext::max_output_chars`), keeping both the head and the tail.
///
/// Compile/test errors cluster at the *end* of a command's output, while listing
/// output is meaningful at the *start*, so both ends survive and the elided
/// middle is marked. This mirrors `read_file`/`list_files`: the tool
/// self-truncates so the engine's hard reject never fires for a noisy command.
fn truncate_output(s: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }

    const MARKER: &str = "...[output truncated]";
    // Reserve the marker + two joining newlines; head gets 1/3, tail 2/3.
    let available = max_chars.saturating_sub(MARKER.chars().count() + 2);
    let head_chars = available / 3;
    let tail_chars = available - head_chars;

    let head: String = s.chars().take(head_chars).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    format!("{head}\n{MARKER}\n{tail}")
}

#[async_trait]
impl Tool for LocalShellTool {
    fn name(&self) -> &'static str {
        "execute_command"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command locally. Use for file operations, code compilation, Git operations, system info queries, etc. For commands that may produce large output, consider limiting lines (e.g. journalctl -n 50, grep ... | head -n 30)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. For commands that may produce large output, consider limiting lines: cat large files with | tail -n 30, find / ls -R with | head -n 50, grep over large scope with | head -n 30."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory. Uses the current directory if not specified."
                }
            },
            "required": ["command"]
        })
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name().to_string(),
            description: "Execute a shell command locally. Use for file operations, code compilation, Git operations, system info queries, etc.".to_string(),
            origin: "phi-kernel-tools".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();

        if command.is_empty() {
            return Ok(vec![Content::text(
                "[Error]: No command provided.".to_string(),
            )]);
        }

        tracing::info!(command = %command, timeout_ms = self.timeout_ms, "execute_command start");

        let working_dir = args.get("working_dir").and_then(Value::as_str);

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(&command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .kill_on_drop(true);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        // spawn + timeout + kill pattern: explicitly kill child process on timeout
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, command = %command, "execute_command: spawn failed");
                return Ok(vec![Content::text(format!(
                    "[Error]: Command execution failed: {}",
                    e
                ))]);
            }
        };

        let pid = child.id();
        let sleep = tokio::time::sleep(Duration::from_millis(self.timeout_ms));
        tokio::pin!(sleep);

        let output = tokio::select! {
            result = child.wait_with_output() => {
                match result {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                        let exit_code = output.status.code();

                        tracing::info!(
                            command = %command,
                            exit_code = exit_code,
                            stdout_len = stdout.len(),
                            stderr_len = stderr.len(),
                            "execute_command: done"
                        );

                        let summary = format_result(&command, &stdout, &stderr, exit_code, false);
                        let summary = truncate_output(&summary, ctx.max_output_chars);
                        Ok(vec![Content::text(summary)])
                    }
                    Err(e) => {
                        tracing::error!(error = %e, command = %command, "execute_command: wait failed");
                        Ok(vec![Content::text(format!(
                            "[Error]: Command execution failed: {}",
                            e
                        ))])
                    }
                }
            }
            _ = &mut sleep => {
                // Timeout — kill the child process by pid (child has been moved by wait_with_output)
                if let Some(pid) = pid {
                    let _ = tokio::process::Command::new("kill")
                        .arg("-9")
                        .arg(pid.to_string())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .await;
                }
                tracing::warn!(command = %command, timeout_ms = self.timeout_ms, "execute_command: timed out and killed");
                Ok(vec![Content::text(format!(
                    "[Command Timed Out after {}ms]\ncommand: {}",
                    self.timeout_ms, command
                ))])
            }
        };

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    #[test]
    fn test_format_result_success() {
        let result = format_result("echo hello", "hello", "", Some(0), false);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_format_result_stderr_only() {
        let result = format_result("cmd", "", "error output", Some(0), false);
        assert_eq!(result, "stderr:\nerror output");
    }

    #[test]
    fn test_format_result_stdout_and_stderr() {
        let result = format_result("cmd", "out", "err", Some(0), false);
        assert_eq!(result, "stdout:\nout\n\nstderr:\nerr");
    }

    #[test]
    fn test_format_result_terminated() {
        let result = format_result("cmd", "", "sigterm", None, false);
        assert!(result.contains("Command Terminated"));
    }

    #[test]
    fn test_format_result_no_output() {
        let result = format_result("true", "", "", Some(0), false);
        assert!(result.contains("no output"));
    }

    #[test]
    fn test_format_result_failure() {
        let result = format_result("false", "", "error", Some(1), false);
        assert!(result.contains("Command Failed"));
        assert!(result.contains("exit code: 1"));
    }

    #[test]
    fn test_format_result_timeout() {
        let result = format_result("sleep 100", "", "", None, true);
        assert!(result.contains("Command Timed Out"));
    }

    #[test]
    fn test_name() {
        let tool = LocalShellTool::new(30000);
        assert_eq!(tool.name(), "execute_command");
    }

    #[test]
    fn test_definition() {
        let tool = LocalShellTool::new(30000);
        assert_eq!(tool.name(), "execute_command");
        assert!(tool.description().contains("shell"));
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("command")));
    }

    #[test]
    fn test_metadata() {
        let tool = LocalShellTool::new(30000);
        let meta = tool.metadata();
        assert_eq!(meta.name, "execute_command");
        assert_eq!(meta.origin, "phi-kernel-tools");
        assert_eq!(meta.version, env!("CARGO_PKG_VERSION"));
        assert!(meta.description.contains("shell"));
        assert!(meta.requirements.is_empty());
    }

    #[tokio::test]
    async fn test_call_echo() {
        let tool = LocalShellTool::new(30000);
        let result = tool
            .call(&json!({"command": "echo hello"}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&result).contains("hello"));
    }

    #[tokio::test]
    async fn test_call_empty_command() {
        let tool = LocalShellTool::new(30000);
        let result = tool
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&result).contains("No command provided"));
    }

    #[tokio::test]
    async fn test_call_failing_command() {
        let tool = LocalShellTool::new(30000);
        let result = tool
            .call(&json!({"command": "exit 3"}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&result).contains("exit code: 3"));
    }

    #[tokio::test]
    async fn test_call_working_dir() {
        let tool = LocalShellTool::new(30000);
        let result = tool
            .call(
                &json!({"command": "pwd", "working_dir": "/"}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap();
        let text = content_text(&result);
        assert!(!text.contains("[Error]"));
        assert!(text.contains('/'));
    }

    #[tokio::test]
    async fn test_call_timeout() {
        let tool = LocalShellTool::new(50);
        let result = tool
            .call(&json!({"command": "sleep 30"}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&result).contains("Timed Out"));
    }

    #[test]
    fn test_truncate_output_unchanged_when_under_budget() {
        assert_eq!(truncate_output("short", Some(100)), "short");
        assert_eq!(truncate_output("short", None), "short");
    }

    #[test]
    fn test_truncate_output_keeps_head_and_tail() {
        let s: String = (0..1000).map(|i| format!("L{i:03}\n")).collect();
        let out = truncate_output(&s, Some(200));
        let n = out.chars().count();
        assert!(n <= 200, "output exceeds budget: {n}");
        assert!(out.starts_with("L000"), "head missing:\n{out}");
        assert!(out.trim_end().ends_with("L999"), "tail missing:\n{out}");
        assert!(out.contains("output truncated"), "marker missing:\n{out}");
    }

    #[tokio::test]
    async fn test_call_self_truncates_large_output() {
        let tool = LocalShellTool::new(30000);
        let mut ctx = ToolContext::for_test();
        ctx.max_output_chars = Some(200);
        let result = tool
            .call(&json!({"command": "seq 1 1000"}), &ctx)
            .await
            .unwrap();
        let text = content_text(&result);
        let n = text.chars().count();
        assert!(n <= 200, "output exceeds budget: {n}");
        assert!(text.contains("output truncated"), "marker missing:\n{text}");
        assert!(text.starts_with('1'), "head missing:\n{text}");
        assert!(text.trim_end().ends_with("1000"), "tail missing:\n{text}");
    }
}
