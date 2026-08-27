#![no_main]

use libfuzzer_sys::fuzz_target;
use phi_kernel_tools::file::fuzz::edit_file::{find_all_positions, find_and_replace, normalize_line_endings};

fuzz_target!(|data: &[u8]| {
    if data.len() > 2048 {
        return;
    }
    if let Ok(s) = std::str::from_utf8(data) {
        let parts: Vec<&str> = s.split('\0').collect();
        if parts.len() >= 2 {
            // Fuzz find_all_positions (highest priority — validates UTF-8 bug fix)
            let haystack = parts[0];
            let needle = parts[1];
            let positions = find_all_positions(haystack, needle);
            for &pos in &positions {
                assert!(pos <= haystack.len());
            }

            // Fuzz normalize_line_endings
            if parts.len() >= 3 {
                let _ = normalize_line_endings(parts[0], parts[2]);
            }

            // Fuzz find_and_replace with size guard
            if parts.len() >= 4 && parts[0].len() < 512 && parts[1].len() < 512 {
                let _ = find_and_replace(
                    parts[0],  // current
                    parts[1],  // original
                    parts[2],  // old_text
                    &parts[3..].join("\0"),  // new_text
                );
            }
        }
    }
});
