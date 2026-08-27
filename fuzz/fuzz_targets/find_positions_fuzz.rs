#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    // Split on a delimiter to get two strings for haystack and needle
    if let Some(idx) = data.find('\0') {
        let haystack = &data[..idx];
        let needle = &data[idx + 1..];
        let positions = phi_kernel_tools::file::fuzz::edit_file::find_all_positions(haystack, needle);

        // Verify invariants
        for (i, &pos) in positions.iter().enumerate() {
            // Position must be within bounds
            assert!(pos <= haystack.len(), "position {} out of bounds", pos);
            // Positions must be in ascending order
            if i > 0 {
                assert!(positions[i - 1] < pos, "positions not ascending");
            }
            // Positions must not overlap
            if i > 0 {
                assert!(
                    positions[i - 1] + needle.len() <= pos,
                    "positions overlap"
                );
            }
        }
    }
});
