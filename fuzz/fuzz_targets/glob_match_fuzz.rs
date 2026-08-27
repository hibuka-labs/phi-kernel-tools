#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    if let Some(idx) = data.find('\0') {
        let pattern = &data[..idx];
        let name = &data[idx + 1..];
        let _result = phi_kernel_tools::file::fuzz::list_files::glob_match(pattern, name);
    }
});
