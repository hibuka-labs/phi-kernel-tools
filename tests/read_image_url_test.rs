//! `read_image` URL face (批5 #14 AC2): the SSRF pipeline end-to-end against
//! local fixture servers. The fixture server is loopback, reached through the
//! `allow_loopback` TEST SEAM — every OTHER internal class (10.*, metadata
//! endpoints, …) stays refused mid-pipeline, which is exactly what the
//! redirect tests pin.

use std::path::PathBuf;

use agent_base::{Content, Tool, ToolContext};
use phi_kernel_tools::file::{FetchPolicy, ReadImageTool};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn png_bytes(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba([5, 10, 15, 255]));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode test png");
    out
}

fn seam_tool() -> ReadImageTool {
    ReadImageTool::new(PathBuf::from("/tmp")).with_fetch_policy(FetchPolicy {
        allow_loopback: true,
    })
}

async fn run(url: &str) -> Vec<Content> {
    seam_tool()
        .call(&json!({ "path": url }), &ToolContext::for_test())
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

/// protects: the happy URL leg — a served PNG comes back attached with its
/// metadata (AC2's "公网图成功" shape; the fixture server stands in at
/// loopback via the seam). fails_when: the fetch pipeline corrupts or drops
/// a good picture. no_claim: that any specific public host is reachable.
#[tokio::test]
async fn a_served_image_is_fetched_and_attached() {
    let bytes = png_bytes(12, 7);
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/img.png"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
        .mount(&server)
        .await;

    let contents = run(&format!("{}/img.png", server.uri())).await;
    let text = text_of(&contents);
    assert!(
        text.contains("Image: 12x7 PNG"),
        "metadata names the dimensions: {text}"
    );
    let (data, mime) = image_of(&contents).expect("the picture is attached");
    assert_eq!(mime, "image/png");
    assert_eq!(
        agent_base::image_ops::from_base64(&data).expect("valid base64"),
        bytes
    );
}

/// protects: a redirect into an internal network is refused exactly like a
/// direct internal request (AC2 "重定向跳内网拒") — the second hop is
/// re-resolved and re-checked, and the refusal is loud with nothing
/// attached. fails_when: only the first hop is checked (the classic
/// open-redirect SSRF). no_claim: that legitimate cross-host redirects of
/// public CDNs work — pinned by the happy leg only.
#[tokio::test]
async fn a_redirect_into_an_internal_address_is_refused() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/jump"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("Location", "http://10.0.0.1/metadata"),
        )
        .mount(&server)
        .await;

    let contents = run(&format!("{}/jump", server.uri())).await;
    let text = text_of(&contents);
    assert!(text.contains("[Error]"), "the refusal is loud: {text}");
    assert!(
        text.contains("non-public address 10.0.0.1"),
        "the internal hop is named: {text}"
    );
    assert!(image_of(&contents).is_none(), "nothing is attached");
}

/// protects: the redirect ceiling — more than 3 hops refuses the fetch
/// (the spec's 3-hop cap), even though every hop is individually allowed.
/// fails_when: a redirect loop chases forever or the fetch hangs.
/// no_claim: that 3 is the right number (named constant, adjustable).
#[tokio::test]
async fn more_than_three_redirects_are_refused() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/loop"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("Location", format!("{}/loop", server.uri())),
        )
        .mount(&server)
        .await;

    let contents = run(&format!("{}/loop", server.uri())).await;
    let text = text_of(&contents);
    assert!(
        text.contains("more than 3 redirects"),
        "the ceiling refuses a loop: {text}"
    );
    assert!(image_of(&contents).is_none());
}

/// protects: the 20 MB download cap — a body over the line is refused whole
/// (AC2 "超 20MB 拒"), loudly, with nothing attached. fails_when: a huge
/// body is buffered into memory anyway or partially attached. no_claim: that
/// 20 MB is the right line (named constant, adjustable).
#[tokio::test]
async fn a_body_over_the_cap_is_refused() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/big.png"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0x89u8; 20 * 1024 * 1024 + 1]))
        .mount(&server)
        .await;

    let contents = run(&format!("{}/big.png", server.uri())).await;
    let text = text_of(&contents);
    assert!(text.contains("exceeds"), "the cap refuses the body: {text}");
    assert!(image_of(&contents).is_none());
}

/// protects: non-http(s) schemes never reach the network — a `file://` or
/// other scheme is refused up front. fails_when: a scheme handler smuggles
/// local reads through the URL face (local paths have their own, confined
/// face). no_claim: the exact error wording.
#[tokio::test]
async fn non_http_schemes_are_refused() {
    for url in ["file:///etc/passwd", "ftp://example.com/a.png"] {
        let contents = run(url).await;
        let text = text_of(&contents);
        assert!(
            text.contains("refused"),
            "scheme must be refused: {url} -> {text}"
        );
        assert!(image_of(&contents).is_none());
    }
}
