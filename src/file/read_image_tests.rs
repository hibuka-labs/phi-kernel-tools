//! `read_image` unit face: type-by-magic, the metadata line, the base64
//! handoff into the vision channel, and the capability degradation.

use std::path::PathBuf;
use std::sync::Arc;

use agent_base::image_ops::from_base64;
use agent_base::llm_trait::LlmProvider;
use agent_base::llm_trait::capabilities::{Capabilities, ProviderInfo};
use agent_base::llm_trait::error::LlmError;
use agent_base::llm_trait::request::ChatRequest;
use agent_base::llm_trait::response::{ChatResponse, ChatStream};
use agent_base::{Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::json;

use super::ReadImageTool;

/// A provider whose only job is to declare whether it can see pictures.
struct StubProvider {
    vision: bool,
}

#[async_trait]
impl LlmProvider for StubProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            Vec::<Result<_, LlmError>>::new(),
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: "stub".to_string(),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: self.vision,
            ..Default::default()
        }
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

fn png_bytes(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba([20, 40, 60, 255]));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode test png");
    out
}

fn ctx_with(vision: Option<bool>) -> ToolContext {
    let mut ctx = ToolContext::for_test();
    if let Some(vision) = vision {
        ctx.llm_client = Some(Arc::new(StubProvider { vision }));
    }
    ctx
}

async fn run(path_arg: &str, ctx: &ToolContext) -> Vec<Content> {
    let tool = ReadImageTool::new(PathBuf::from("/tmp"));
    tool.call(&json!({ "path": path_arg }), ctx)
        .await
        .expect("read_image must answer, never error out")
}

fn text_of(contents: &[Content]) -> String {
    contents
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn image_of(contents: &[Content]) -> Option<(String, String)> {
    contents.iter().find_map(|c| match c {
        Content::Image { data, mime_type } => Some((data.clone(), mime_type.clone())),
        _ => None,
    })
}

/// protects: a local PNG comes back as the vision channel's native shape —
/// metadata line (WxH, format, size) + `Content::Image` whose base64 decodes
/// to the file's bytes (AC1). fails_when: the picture is renamed, re-encoded
/// or summarized instead of attached. no_claim: what the model does with it.
#[tokio::test]
async fn a_local_png_returns_metadata_and_the_image() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bytes = png_bytes(37, 11);
    let file = dir.path().join("shot.png");
    std::fs::write(&file, &bytes).expect("write fixture");

    let contents = run(file.to_str().unwrap(), &ctx_with(None)).await;
    let text = text_of(&contents);
    assert!(
        text.contains("Image: 37x11 PNG"),
        "metadata names the dimensions: {text}"
    );
    let (data, mime) = image_of(&contents).expect("the picture is attached");
    assert_eq!(mime, "image/png");
    assert_eq!(
        from_base64(&data).expect("valid base64"),
        bytes,
        "bytes intact"
    );
}

/// protects: magic bytes decide the type, not the file name — a PNG served
/// as `photo.jpg` is taken (and labeled PNG), and text/html wearing
/// `.png` is refused (AC2's two impostor cases). fails_when: an extension
/// or a claimed MIME can smuggle a wrong-typed payload. no_claim: what
/// happens to bytes of a correctly typed file.
#[tokio::test]
async fn magic_bytes_decide_the_type_not_the_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let png = png_bytes(4, 4);

    let misnamed = dir.path().join("photo.jpg");
    std::fs::write(&misnamed, &png).expect("write fixture");
    let contents = run(misnamed.to_str().unwrap(), &ctx_with(None)).await;
    let text = text_of(&contents);
    assert!(
        text.contains("Image: 4x4 PNG"),
        "a .jpg holding a PNG is served as the PNG it is: {text}"
    );
    assert!(image_of(&contents).is_some(), "the picture travels");

    let impostor = dir.path().join("shot.png");
    std::fs::write(&impostor, b"<html><body>not a picture</body></html>").expect("write fixture");
    let contents = run(impostor.to_str().unwrap(), &ctx_with(None)).await;
    let text = text_of(&contents);
    assert!(
        text.contains("magic bytes do not match"),
        "an HTML impostor is refused: {text}"
    );
    assert!(image_of(&contents).is_none(), "nothing is attached");
}

/// protects: the capability gate degrades to words — a supports_vision=false
/// model gets the metadata line and an explicit /model pointer, and the
/// result carries ZERO images so the request body cannot grow one (AC3).
/// fails_when: the picture is attached anyway (silently unreadable) or the
/// picture vanishes without a word (silent drop). no_claim: which models
/// lack vision — that lives in the registry.
#[tokio::test]
async fn a_visionless_model_gets_words_not_the_picture() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("shot.png");
    std::fs::write(&file, png_bytes(2, 2)).expect("write fixture");

    let contents = run(file.to_str().unwrap(), &ctx_with(Some(false))).await;
    assert!(image_of(&contents).is_none(), "no picture is attached");
    let text = text_of(&contents);
    assert!(
        text.contains("Image: 2x2 PNG"),
        "the metadata still lands: {text}"
    );
    assert!(
        text.contains("cannot see images"),
        "the degradation is explicit: {text}"
    );
    assert!(text.contains("/model"), "it points at the fix: {text}");
}

/// protects: the gate does not over-fire — a vision-capable model (and a
/// caller with no provider wired at all) gets the picture. fails_when: the
/// gate blocks a model that CAN see. no_claim: registry accuracy.
#[tokio::test]
async fn a_vision_model_gets_the_picture() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("shot.png");
    std::fs::write(&file, png_bytes(2, 2)).expect("write fixture");

    for ctx in [ctx_with(Some(true)), ctx_with(None)] {
        let contents = run(file.to_str().unwrap(), &ctx).await;
        assert!(
            image_of(&contents).is_some(),
            "a seeing caller gets the picture"
        );
    }
}

/// protects: a missing file answers as a readable error line — the tool
/// returns text, the run continues. fails_when: the call hard-errors the
/// turn or reports success on nothing. no_claim: the exact wording beyond
/// the [Error] marker.
#[tokio::test]
async fn missing_and_non_file_targets_say_so() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("nope.png").to_str().unwrap().to_string();
    let text = text_of(&run(&missing, &ctx_with(None)).await);
    assert!(
        text.contains("[Error]"),
        "missing file is an error line: {text}"
    );

    let not_a_file = dir.path().to_str().unwrap().to_string();
    let text = text_of(&run(&not_a_file, &ctx_with(None)).await);
    assert!(
        text.contains("[Error]"),
        "a directory is an error line: {text}"
    );
}

/// protects: the empty-argument face. fails_when: an empty `path` reaches
/// the loader. no_claim: message wording.
#[tokio::test]
async fn an_empty_path_is_refused_before_any_io() {
    let text = text_of(&run("", &ctx_with(None)).await);
    assert!(text.contains("[Error]"), "got: {text}");
}
