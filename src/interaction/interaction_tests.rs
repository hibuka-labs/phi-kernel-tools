//! Tests for [`super`] — prompt matcher + vocabulary (split out of
//! `mod.rs` to keep every source file under the 1000-line rule; `#[path]`
//! keeps private-item access).

use super::*;

/// protects: the vocabulary table reports kind + default_side + suggested
///   exactly as pinned — the 11 ops-tool rows kept their detection coverage
///   (with the `[y/N]` auto-y bug fixed), and the new families land.
/// fails_when: any row's classification, default side, or suggested input
///   drifts; a `[y/N]` suggestion ever becomes an auto-send.
/// no_claim: what an auto sender does with the suggestion (policy lives in
///   the product layer).
#[test]
fn vocabulary_table() {
    struct Case {
        input: &'static str,
        kind: PromptKind,
        default_side: Option<char>,
        suggested: SuggestedInput,
    }
    let table = [
        // ── 11 original rows (semantic coverage) ──
        Case {
            input: "The authenticity of host 'x' can't be established.\nAre you sure you want to continue connecting (yes/no/[fingerprint])?",
            kind: PromptKind::SshHostKey,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Enter password: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Password: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Passphrase: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "[sudo] password for root: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Do you want to continue? [Y/n]",
            kind: PromptKind::Confirm,
            default_side: Some('Y'),
            suggested: SuggestedInput::Text("y\n"),
        },
        Case {
            input: "Remove 123 packages? [y/N]",
            kind: PromptKind::Confirm,
            default_side: Some('N'),
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Proceed with installation? (yes/no)",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "some output\n(END)",
            kind: PromptKind::Pager,
            default_side: None,
            suggested: SuggestedInput::Text("q\n"),
        },
        Case {
            input: "long output\n:\n",
            kind: PromptKind::Pager,
            default_side: None,
            suggested: SuggestedInput::Text("q\n"),
        },
        Case {
            input: "│       <Ok>          <Cancel>          │",
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter"),
        },
        Case {
            input: "Do you want to proceed?\n<Yes>  <No>",
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter"),
        },
        Case {
            input: "Installation complete! [ OK ]",
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter"),
        },
        Case {
            input: "Press Enter to continue...",
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter"),
        },
        // ── new families: bracket/paren case normalization ──
        Case {
            input: "Continue? (y/n)",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Continue? (Y/n)",
            kind: PromptKind::Confirm,
            default_side: Some('Y'),
            suggested: SuggestedInput::Text("y\n"),
        },
        Case {
            input: "Continue? (y/N)",
            kind: PromptKind::Confirm,
            default_side: Some('N'),
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Continue? [y/n]",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Continue? [Y/N]",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Delete everything? (Yes/No)",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        // ── new families: CJK confirm ──
        Case {
            input: "是否继续安装？(是/否)",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "覆盖已存在的文件（是/否）",
            kind: PromptKind::Confirm,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        // ── new families: numeric menu ──
        Case {
            input: "Enter choice [1-3]: ",
            kind: PromptKind::Menu,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Please press 1/2/3 to continue",
            kind: PromptKind::Menu,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "select an option\n[1/2/3]\n",
            kind: PromptKind::Menu,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "请输入编号后回车",
            kind: PromptKind::Menu,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        // ── new families: CJK press-enter + pager extra ──
        Case {
            input: "安装完成，按回车继续",
            kind: PromptKind::PressEnter,
            default_side: None,
            suggested: SuggestedInput::Key("enter"),
        },
        Case {
            input: "log line\n--More--(42%)",
            kind: PromptKind::Pager,
            default_side: None,
            suggested: SuggestedInput::Text("q\n"),
        },
        // ── new families: 2FA/OTP are secrets too ──
        Case {
            input: "Verification code: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
        Case {
            input: "Two-factor code: ",
            kind: PromptKind::Password,
            default_side: None,
            suggested: SuggestedInput::None,
        },
    ];
    for c in &table {
        assert_eq!(
            match_prompt(c.input),
            Some(PromptClass {
                kind: c.kind,
                default_side: c.default_side,
                suggested: c.suggested,
            }),
            "input: {:?}",
            c.input
        );
    }
}

/// protects: no-match tails return None (detection precision half of the
///   vocabulary contract).
/// fails_when: any of these tails starts matching a prompt class.
/// no_claim: the foreground early-kill path (covered in local_shell tests).
#[test]
fn no_match_fixtures() {
    let misses = [
        "Pulling from library/nginx\nDigest: sha256:abc123\nStatus: Downloaded",
        "npm warn deprecated foo@1.0.0\nadded 269 packages in 9s",
        "Compiling phi-kernel-tools v0.13.0\nFinished dev target in 12.4s",
        "error[E0308]: mismatched types\n  --> src/lib.rs:10:5",
        "warning: unused variable: `x`",
        "total 24\ndrwxr-xr-x  5 user  staff   160 Nov 11 10:00 src",
        "Are you sure? The fingerprint prompt appears once per host",
        "The password: example123 is stored in config", // mid-line, not a prompt line
        "1/2/3 of the tests passed",
        "END of file reached",
        "the continue button is disabled until input arrives",
        "type: module\nmain: index.js",
        "option: verbose\nselection: all",
        "choose 1, 2 or 3 from the menu below and run again",
        "yes/no answers are required by the installer",
    ];
    for input in &misses {
        assert_eq!(match_prompt(input), None, "input: {:?}", input);
    }
}

/// protects: the priority ladder — Password beats everything, SshHostKey
///   beats Confirm (its text carries a yes/no question).
/// fails_when: a combined tail reports the lower-priority class.
/// no_claim: what happens when two prompts are on screen in different regions.
#[test]
fn priority_ladder() {
    let both = "Continue? [Y/n]\nPassword: ";
    let hit = match_prompt(both).unwrap();
    assert_eq!(hit.kind, PromptKind::Password);
    assert_eq!(hit.suggested, SuggestedInput::None);

    let ssh = "Are you sure you want to continue connecting (yes/no)?";
    let hit = match_prompt(ssh).unwrap();
    assert_eq!(hit.kind, PromptKind::SshHostKey);

    let pager_over_menu = "Enter choice [1-3]\n(END)";
    let hit = match_prompt(pager_over_menu).unwrap();
    assert_eq!(hit.kind, PromptKind::Pager);
}

/// protects: the `[y/N]` fix — a decline-default confirm never carries an
///   auto-send suggestion (the old ops matcher replied `y` here).
/// fails_when: `suggested` for `[y/N]` is anything but `None`.
/// no_claim: the policy tier this class lands in.
#[test]
fn decline_default_never_auto_sends() {
    let hit = match_prompt("Remove 123 packages? [y/N]").unwrap();
    assert_eq!(hit.default_side, Some('N'));
    assert_eq!(hit.suggested, SuggestedInput::None);
}

/// protects: ANSI stripping for CSI, cursor and OSC sequences before match.
/// fails_when: escapes survive into the matched text or swallow prompt text.
/// no_claim: exotic escape forms beyond the three families.
#[test]
fn strip_ansi_families() {
    let input = "Do you want to continue? \x1b[1m[Y/n]\x1b[0m";
    let stripped = strip_ansi(input);
    assert!(stripped.contains("[Y/n]"));
    assert!(!stripped.contains('\u{1b}'));
    assert!(match_prompt(&stripped).is_some());

    let stripped = strip_ansi("\x1b[2J\x1b[HHello World");
    assert_eq!(stripped, "Hello World");

    let stripped = strip_ansi("\x1b]0;Terminal Title\x07Press Enter");
    assert_eq!(stripped, "Press Enter");
}

/// protects: the `[waiting for input]` block wire shape — class, default,
///   suggested and tail all present, English vocabulary.
/// fails_when: a field is renamed, dropped, or the note wording changes.
/// no_claim: who consumes the suggestion.
#[test]
fn waiting_block_shape() {
    let class = PromptClass {
        kind: PromptKind::Confirm,
        default_side: Some('Y'),
        suggested: SuggestedInput::Text("y\n"),
    };
    let block = format_waiting_block(&class, "Continue? [Y/n]");
    assert!(block.starts_with("[waiting for input]\n"));
    assert!(block.contains("class: confirm"));
    assert!(block.contains("default: y"));
    assert!(block.contains("suggested: y\\n"));
    assert!(block.contains("tail:\nContinue? [Y/n]"));

    let class = PromptClass {
        kind: PromptKind::Password,
        default_side: None,
        suggested: SuggestedInput::None,
    };
    let block = format_waiting_block(&class, "Password: ");
    assert!(block.contains("class: password"));
    assert!(block.contains("default: none"));
    assert!(block.contains("suggested: none"));

    let class = PromptClass {
        kind: PromptKind::PressEnter,
        default_side: None,
        suggested: SuggestedInput::Key("enter"),
    };
    let block = format_waiting_block(&class, "<Ok>");
    assert!(block.contains("suggested: <enter>"));
}

/// protects: tail extraction stays on char boundaries (CJK) and keeps the
///   end of the stream.
/// fails_when: slicing panics on multibyte chars or drops the tail.
/// no_claim: stdout/stderr interleaving order.
#[test]
fn output_tail_char_boundary() {
    let tail = output_tail("输出一二三四五", "", 3);
    assert_eq!(tail, "三四五");
    let tail = output_tail("hello world", "err text", 200);
    assert!(tail.starts_with("hello world"));
    assert!(tail.ends_with("err text"));
    let tail = output_tail("abcdef", "", 3);
    assert_eq!(tail, "def");
}
