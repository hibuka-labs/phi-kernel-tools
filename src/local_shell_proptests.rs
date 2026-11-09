//! Property tests for [`super`] — local shell helpers (split out of
//! `local_shell.rs` to keep every source file under the 1000-line rule;
//! `#[path]` keeps private-item access).

use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn truncate_output_respects_budget(s in ".*", max in 30..200usize) {
        // max must be large enough to fit the marker + head/tail
        let result = truncate_output(&s, Some(max));
        prop_assert!(result.chars().count() <= max,
            "output {} chars exceeds budget {}", result.chars().count(), max);
    }

    #[test]
    fn truncate_output_no_panic(s in ".*", max in 0..500usize) {
        let _ = truncate_output(&s, Some(max));
    }

    #[test]
    fn format_result_does_not_panic(
        command in ".*",
        stdout in ".*",
        stderr in ".*",
        exit_code in proptest::option::of(0i32..128),
        timed_out in proptest::bool::ANY,
    ) {
        let _ = format_result(&command, &stdout, &stderr, exit_code, timed_out);
    }

    #[test]
    fn truncate_output_short_string_unchanged(s in "[a-z]{0,50}") {
        let result = truncate_output(&s, Some(100));
        prop_assert_eq!(result, s);
    }
}
