//! Tests for [`super`] — the answer-policy table: tier mapping, the hard
//! boundary, action-key granularity, and key wire bytes.

use super::*;

fn confirm() -> PromptClass {
    PromptClass {
        kind: PromptKind::Confirm,
        default_side: Some('Y'),
        suggested: SuggestedInput::Text("y\n"),
    }
}

fn pager() -> PromptClass {
    PromptClass {
        kind: PromptKind::Pager,
        default_side: None,
        suggested: SuggestedInput::Text("q\n"),
    }
}

/// protects: the spec's approved table — pager auto, confirm/menu/press_enter
///   and free input cache.
/// fails_when: any tier deviates from the approved mapping.
/// no_claim: what a particular host sets (products may override fields).
#[test]
fn spec_default_table_tiers() {
    let p = PromptPolicy::spec_default();
    assert_eq!(p.tier_for(Some(pager())), PolicyTier::Auto);
    assert_eq!(p.tier_for(Some(confirm())), PolicyTier::Cache);
    assert_eq!(
        p.tier_for(Some(PromptClass {
            kind: PromptKind::Menu,
            default_side: None,
            suggested: SuggestedInput::None
        })),
        PolicyTier::Cache
    );
    assert_eq!(
        p.tier_for(Some(PromptClass {
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter")
        })),
        PolicyTier::Cache
    );
    assert_eq!(p.tier_for(None), PolicyTier::Cache);
}

/// protects: the hard boundary — secrets and host-key confirmations stop at
///   every tier, even when the product sets every table field to Auto.
/// fails_when: any of Password / SshHostKey / Unknown maps to Auto or Cache.
/// no_claim: the human handoff UX beyond the `needs user` contract.
#[test]
fn hard_boundary_never_configurable() {
    let wide_open = PromptPolicy {
        pager: PolicyTier::Auto,
        confirm: PolicyTier::Auto,
        menu: PolicyTier::Auto,
        press_enter: PolicyTier::Auto,
        free_input: PolicyTier::Auto,
    };
    for kind in [
        PromptKind::Password,
        PromptKind::SshHostKey,
        PromptKind::Unknown,
    ] {
        let class = PromptClass {
            kind,
            default_side: None,
            suggested: SuggestedInput::None,
        };
        assert_eq!(
            wide_open.tier_for(Some(class)),
            PolicyTier::Stop,
            "{kind:?} must never leave the stop tier"
        );
    }
}

/// protects: action-key granularity — `AllowAlways` on one answer never
///   pre-approves a different answer, task, or class.
/// fails_when: two sends with a differing (task, class, input) share a key,
///   or two identical sends produce different keys.
/// no_claim: where the cache lives (host-side).
#[test]
fn action_key_granularity() {
    let p = PromptPolicy::default();
    let y = SendInput::Text("y\n".to_string());
    let n = SendInput::Text("n\n".to_string());
    let base = p.send_action_key("bg_ab12", Some(confirm()), &y);
    assert_eq!(base, p.send_action_key("bg_ab12", Some(confirm()), &y));
    assert_ne!(base, p.send_action_key("bg_ab12", Some(confirm()), &n));
    assert_ne!(base, p.send_action_key("bg_cd34", Some(confirm()), &y));
    let free = SendInput::Key("enter".to_string());
    assert_ne!(base, p.send_action_key("bg_ab12", None, &free));
}

/// protects: the consent request shape a cache-tier send produces — carries
///   the action key so `AllowAlways` can cache it.
/// fails_when: the request loses the action key or gains a wrong key.
/// no_claim: how a host renders the dialog.
#[test]
fn approval_request_carries_action_key() {
    let p = PromptPolicy::default();
    let input = SendInput::Text("y\n".to_string());
    let req = p.approval_request("bg_ab12", Some(confirm()), &input);
    assert_eq!(
        req.action_key.as_deref(),
        Some(
            p.send_action_key("bg_ab12", Some(confirm()), &input)
                .as_str()
        )
    );
    assert_eq!(req.title, "Task bg_ab12 is waiting for input");
    // The picture a human reads: the class and the answer, not the cache key.
    assert!(
        req.message.contains("confirm") && req.message.contains("text \"y\\n\""),
        "message names the class and the answer: {}",
        req.message
    );
}

/// protects: key wire bytes — Enter is CR, arrows are CSI, control keys are
///   raw control bytes.
/// fails_when: any key name maps to wrong bytes or an unknown name maps to a
///   panic instead of a fallback.
/// no_claim: terminal line-discipline behavior beyond the byte contract.
#[test]
fn key_wire_bytes() {
    assert_eq!(key_bytes("enter"), b"\r".to_vec());
    assert_eq!(key_bytes("tab"), b"\t".to_vec());
    assert_eq!(key_bytes("up"), b"\x1b[A".to_vec());
    assert_eq!(key_bytes("down"), b"\x1b[B".to_vec());
    assert_eq!(key_bytes("ctrl_c"), b"\x03".to_vec());
    assert_eq!(key_bytes("ctrl_d"), b"\x04".to_vec());
    assert_eq!(key_bytes("escape"), b"\x1b".to_vec());
    for name in KEY_NAMES {
        let _ = key_bytes(name);
    }
}

/// protects: `SendInput::wire_bytes` parity — text passes through verbatim
///   (a missing newline stays missing; the wire says so), keys use the map.
/// fails_when: text is silently newline-terminated or key bytes drift.
/// no_claim: what the receiving program does with the bytes.
#[test]
fn send_input_wire_bytes() {
    assert_eq!(SendInput::Text("y".to_string()).wire_bytes(), b"y".to_vec());
    assert_eq!(
        SendInput::Text("y\n".to_string()).wire_bytes(),
        b"y\n".to_vec()
    );
    assert_eq!(
        SendInput::Key("enter".to_string()).wire_bytes(),
        b"\r".to_vec()
    );
}
