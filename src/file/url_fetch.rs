//! SSRF-safe HTTP fetching for tool-driven image downloads (`read_image`'s
//! URL face).
//!
//! A URL argument is attacker-influenced text: `http://169.254.169.254/…`
//! is the cloud metadata endpoint and `http://127.0.0.1:6379/` is every
//! local service. The fetch therefore (1) resolves the host and rejects it
//! if ANY resolved address is not publicly routable (loopback, RFC1918,
//! link-local, ULA, multicast, CGNAT, …), (2) re-checks and re-pins on EVERY
//! redirect hop (≤ 3), pinning the checked address into the connection so a
//! second DNS answer cannot rebind between check and connect, (3) enforces a
//! 30 s total timeout and a 20 MB streamed download cap.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use reqwest::redirect::Policy;

/// Largest body a URL fetch will buffer: a screenshot, not a video.
pub const MAX_DOWNLOAD_BYTES: usize = 20 * 1024 * 1024;

/// Total wall-clock budget for one fetch, redirects included.
pub const TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

/// How many redirects a fetch may follow before refusing (the spec's 3-hop
/// ceiling).
pub const MAX_REDIRECTS: usize = 3;

/// What a fetch is allowed to reach. Production runs the default: only
/// publicly routable addresses.
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchPolicy {
    /// Admit loopback targets — a TEST SEAM for local fixture servers, never
    /// set in production. Every other internal class stays rejected.
    pub allow_loopback: bool,
}

/// The outcome of a successful fetch: raw bytes plus the final URL after
/// redirects (for the metadata line the model reads).
pub struct FetchedImage {
    pub bytes: Vec<u8>,
    pub final_url: String,
}

/// Is this address safe for a tool-driven fetch to connect to? Everything
/// that is not publicly routable is refused: loopback, RFC1918 private,
/// link-local, CGNAT, ULA, multicast, broadcast, unspecified, reserved, and
/// IPv4-mapped IPv6 (judged by the embedded v4).
pub fn is_publicly_routable(ip: IpAddr, policy: FetchPolicy) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4, policy),
        IpAddr::V6(v6) => is_public_v6(v6, policy),
    }
}

fn is_public_v4(ip: Ipv4Addr, policy: FetchPolicy) -> bool {
    if ip.is_loopback() {
        return policy.allow_loopback;
    }
    let [a, b, c, _] = ip.octets();
    let blocked = a == 0                                    // 0.0.0.0/8 "this network"
        || a == 10                                          // 10.0.0.0/8 RFC1918
        || (a == 100 && (b & 0xC0) == 64)                   // 100.64.0.0/10 CGNAT
        || (a == 169 && b == 254)                           // 169.254.0.0/16 link-local
        || (a == 172 && (16..32).contains(&b))              // 172.16.0.0/12 RFC1918
        || (a == 192 && b == 0 && c == 0)                   // 192.0.0.0/24 IETF
        || (a == 192 && b == 0 && c == 2)                   // 192.0.2.0/24 TEST-NET-1
        || (a == 192 && b == 168)                           // 192.168.0.0/16 RFC1918
        || (a == 198 && (b == 18 || b == 19))               // 198.18.0.0/15 benchmarking
        || (a == 198 && b == 51 && c == 100)                // 198.51.100.0/24 TEST-NET-2
        || (a == 203 && b == 0 && c == 113)                 // 203.0.113.0/24 TEST-NET-3
        || ip.is_multicast()                                // 224.0.0.0/4
        || a >= 240; // 240.0.0.0/4 reserved + broadcast
    !blocked
}

fn is_public_v6(ip: Ipv6Addr, policy: FetchPolicy) -> bool {
    // ::1 first: the v4 translation below reads it as 0.0.0.1 (an
    // IPv4-compatible form) and would deny the loopback seam its say.
    if ip.is_loopback() {
        return policy.allow_loopback;
    }
    // IPv4-mapped (`::ffff:a.b.c.d`) and NAT64 (`64:ff9b::a.b.c.d`) carry a
    // v4 target — judge that one, or `::ffff:127.0.0.1` sneaks through.
    if let Some(v4) = ip.to_ipv4_mapped().or_else(|| ip.to_ipv4()) {
        return is_public_v4(v4, policy);
    }
    let seg = ip.segments();
    if seg[0] == 0x64 && seg[1] == 0xff9b {
        let v4 = Ipv4Addr::new(
            (seg[6] >> 8) as u8,
            (seg[6] & 0xff) as u8,
            (seg[7] >> 8) as u8,
            (seg[7] & 0xff) as u8,
        );
        return is_public_v4(v4, policy);
    }
    let blocked = ip.is_unspecified()
        || ip.is_multicast()                // ff00::/8
        || (seg[0] & 0xfe00) == 0xfc00      // fc00::/7 ULA
        || (seg[0] & 0xffc0) == 0xfe80; // fe80::/10 link-local
    !blocked
}

/// Resolve `host:port` and pick the one address the fetch may connect to:
/// every resolved address must pass [`is_publicly_routable`] (one internal
/// answer poisons the host — the strict reading), and the chosen address is
/// returned for pinning.
async fn resolve_checked(host: &str, port: u16, policy: FetchPolicy) -> Result<SocketAddr, String> {
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("DNS lookup failed for '{host}': {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!("DNS lookup returned no addresses for '{host}'"));
    }
    for addr in &addrs {
        if !is_publicly_routable(addr.ip(), policy) {
            return Err(format!(
                "refused: '{host}' resolves to non-public address {} (SSRF guard)",
                addr.ip()
            ));
        }
    }
    Ok(addrs[0])
}

/// Stream one GET with no automatic redirects and a hard byte cap. Returns
/// the body bytes and the `Location` header when the status is a redirect.
async fn fetch_once(
    url: &reqwest::Url,
    policy: FetchPolicy,
) -> Result<(Vec<u8>, Option<String>), String> {
    let host = url
        .host_str()
        .ok_or_else(|| format!("URL '{url}' has no host"))?
        .to_string();
    let port = url
        .port_or_known_default()
        .ok_or_else(|| format!("URL '{url}' has no port"))?;

    // Check-then-pin: `resolve` makes the connection go to the address that
    // was just checked, so a second DNS answer cannot rebind the fetch.
    let pinned = resolve_checked(&host, port, policy).await?;
    let client = reqwest::Client::builder()
        .redirect(Policy::none())
        .resolve(&host, pinned)
        .timeout(TOTAL_TIMEOUT)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    let resp = client
        .get(url.clone())
        .send()
        .await
        .map_err(|e| format!("request to '{url}' failed: {e}"))?;

    let status = resp.status();
    if status.is_redirection() {
        let location = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        return Ok((Vec::new(), location));
    }
    if !status.is_success() {
        return Err(format!("'{url}' answered HTTP {status}"));
    }
    if resp
        .content_length()
        .is_some_and(|len| len as usize > MAX_DOWNLOAD_BYTES)
    {
        return Err(format!(
            "download refused: body exceeds the {MAX_DOWNLOAD_BYTES} byte cap"
        ));
    }

    let mut resp = resp;
    let mut bytes = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("download from '{url}' failed: {e}"))?
    {
        if bytes.len() + chunk.len() > MAX_DOWNLOAD_BYTES {
            return Err(format!(
                "download refused: body exceeds the {MAX_DOWNLOAD_BYTES} byte cap"
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((bytes, None))
}

/// Fetch a URL's bytes under the SSRF policy: redirects are followed by hand
/// (≤ [`MAX_REDIRECTS`]) so every hop is resolved, checked and pinned anew —
/// a redirect into an internal network is refused exactly like a direct
/// request to one.
pub async fn fetch_image(url: &str, policy: FetchPolicy) -> Result<FetchedImage, String> {
    let run = fetch_image_inner(url, policy);
    tokio::time::timeout(TOTAL_TIMEOUT, run)
        .await
        .map_err(|_| format!("fetch timed out after {}s", TOTAL_TIMEOUT.as_secs()))?
}

async fn fetch_image_inner(url: &str, policy: FetchPolicy) -> Result<FetchedImage, String> {
    let mut current: reqwest::Url = url
        .parse()
        .map_err(|e| format!("invalid URL '{url}': {e}"))?;
    if current.scheme() != "http" && current.scheme() != "https" {
        return Err(format!(
            "refused: scheme '{}' is not http(s)",
            current.scheme()
        ));
    }

    for hop in 0..=MAX_REDIRECTS {
        let (bytes, location) = fetch_once(&current, policy).await?;
        let Some(location) = location else {
            return Ok(FetchedImage {
                bytes,
                final_url: current.to_string(),
            });
        };
        if hop == MAX_REDIRECTS {
            return Err(format!(
                "refused: more than {MAX_REDIRECTS} redirects (last: {location})"
            ));
        }
        current = current
            .join(&location)
            .map_err(|e| format!("bad redirect target '{location}': {e}"))?;
        if current.scheme() != "http" && current.scheme() != "https" {
            return Err(format!(
                "refused: redirect to non-http(s) scheme '{}'",
                current.scheme()
            ));
        }
    }
    Err(format!("refused: more than {MAX_REDIRECTS} redirects"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(s: &str) -> IpAddr {
        s.parse().expect("valid v4 fixture")
    }

    fn v6(s: &str) -> IpAddr {
        s.parse().expect("valid v6 fixture")
    }

    /// protects: the SSRF address table — every AC2 class is refused
    /// (127.0.0.1 / 10.* / 169.254.* / ::1, plus RFC1918, ULA, CGNAT,
    /// multicast, mapped-v4 loopback). fails_when: one internal class slips
    /// through as "public". no_claim: which public hosts are worth fetching.
    #[test]
    fn internal_addresses_are_refused() {
        let strict = FetchPolicy::default();
        for bad in [
            "127.0.0.1",
            "127.1.2.3",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254", // cloud metadata endpoint
            "0.0.0.0",
            "100.64.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "255.255.255.255",
        ] {
            assert!(
                !is_publicly_routable(v4(bad), strict),
                "{bad} must be refused"
            );
        }
        for bad in ["::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "ff02::1"] {
            assert!(
                !is_publicly_routable(v6(bad), strict),
                "{bad} must be refused"
            );
        }
        // IPv4-mapped loopback smuggled in v6 form is judged by its v4 part.
        assert!(!is_publicly_routable(v6("::ffff:127.0.0.1"), strict));
        assert!(!is_publicly_routable(v6("::ffff:10.0.0.1"), strict));
        // Same for the NAT64 embedding (64:ff9b::127.0.0.1).
        assert!(!is_publicly_routable(v6("64:ff9b::7f00:1"), strict));
    }

    /// protects: ordinary public addresses pass — the guard is not a blunt
    /// "no network" denial. fails_when: a real public image URL can never be
    /// fetched. no_claim: that any particular public host is up.
    #[test]
    fn public_addresses_pass() {
        let strict = FetchPolicy::default();
        for good in [
            "1.1.1.1",
            "8.8.8.8",
            "93.184.216.34",
            "172.32.0.1",
            "172.15.0.1",
        ] {
            assert!(is_publicly_routable(v4(good), strict), "{good} must pass");
        }
        for good in ["2606:4700::1111", "2001:4860:4860::8888"] {
            assert!(is_publicly_routable(v6(good), strict), "{good} must pass");
        }
    }

    /// protects: the loopback test seam — `allow_loopback` admits 127.0.0.1
    /// and ::1 ONLY for fixture servers, and only those. fails_when: the
    /// seam also admits RFC1918 or link-local (a test flag must not become a
    /// production hole). no_claim: anything about production policy.
    #[test]
    fn the_loopback_seam_opens_only_loopback() {
        let seam = FetchPolicy {
            allow_loopback: true,
        };
        assert!(is_publicly_routable(v4("127.0.0.1"), seam));
        assert!(is_publicly_routable(v6("::1"), seam));
        assert!(!is_publicly_routable(v4("10.0.0.1"), seam));
        assert!(!is_publicly_routable(v4("169.254.169.254"), seam));
    }
}
