//! Property tests for [`super`] — edit_file helpers (split out of `edit_file.rs`
//! to keep every source file under the 1000-line rule; `#[path]` keeps
//! private-item access).

use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn find_all_positions_does_not_panic(haystack in ".*", needle in ".*") {
        let _ = find_all_positions(&haystack, &needle);
    }

    #[test]
    fn find_all_positions_results_non_overlapping(haystack in "[a-z]{0,100}", needle in "[a-z]{1,10}") {
        let positions = find_all_positions(&haystack, &needle);
        // Results should be non-overlapping
        for w in positions.windows(2) {
            prop_assert!(w[0] + needle.len() <= w[1],
                "overlapping results: {} + {} > {}", w[0], needle.len(), w[1]);
        }
    }

    #[test]
    fn normalize_line_endings_lf_idempotent(s in "[^\r]{0,100}") {
        let once = normalize_line_endings(&s, "\n");
        let twice = normalize_line_endings(&once, "\n");
        prop_assert_eq!(once, twice);
    }

    #[test]
    fn normalize_line_endings_crlf_produces_only_crlf(s in ".*") {
        let result = normalize_line_endings(&s, "\r\n");
        // Should not contain bare \n (only \r\n)
        prop_assert!(!result.contains("\r\n\r"), "double CR found");
        // Every \n should be preceded by \r
        for (i, byte) in result.bytes().enumerate() {
            if byte == b'\n' && i > 0 {
                prop_assert_eq!(result.as_bytes()[i - 1], b'\r',
                    "bare \\n at position {}", i);
            }
        }
    }
}
