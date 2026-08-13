//! Benchmarks: kernel file tools (read_file, write_file, list_files).

use agent_base::{Tool, ToolContext};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use phi_kernel_tools::file::{ListFilesTool, ReadFileTool, WriteFileTool};
use tempfile::TempDir;

fn make_ctx() -> ToolContext {
    ToolContext::for_test()
}

fn bench_read_file(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp = TempDir::new().unwrap();
    let file_path = temp.path().join("test.txt");
    let content = "This is a test file.\n".repeat(1000);
    std::fs::write(&file_path, &content).unwrap();

    let tool = ReadFileTool::new(temp.path().to_path_buf());
    let ctx = make_ctx();

    c.bench_function("file/read_1000_lines", |b| {
        b.iter(|| {
            let args = serde_json::json!({"path": "test.txt"});
            let _ = black_box(rt.block_on(tool.call(&args, &ctx)));
        });
    });
}

fn bench_write_file(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp = TempDir::new().unwrap();
    let tool = WriteFileTool::new(temp.path().to_path_buf());
    let ctx = make_ctx();
    let content = "Hello, world!\n".repeat(100);

    c.bench_function("file/write_100_lines", |b| {
        b.iter(|| {
            let args = serde_json::json!({"path": "bench_output.txt", "content": content});
            let _ = black_box(rt.block_on(tool.call(&args, &ctx)));
        });
    });
}

fn bench_list_files(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp = TempDir::new().unwrap();
    // Create 100 files
    for i in 0..100 {
        std::fs::write(temp.path().join(format!("file_{:03}.txt", i)), "data").unwrap();
    }
    let sub = temp.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    for i in 0..20 {
        std::fs::write(sub.join(format!("sub_{:03}.txt", i)), "data").unwrap();
    }

    let tool = ListFilesTool::new(temp.path().to_path_buf());
    let ctx = make_ctx();

    c.bench_function("file/list_100_flat", |b| {
        b.iter(|| {
            let args = serde_json::json!({"path": "."});
            let _ = black_box(rt.block_on(tool.call(&args, &ctx)));
        });
    });
}

criterion_group! {
    name = file_tools_benches;
    config = Criterion::default().sample_size(200);
    targets = bench_read_file, bench_write_file, bench_list_files
}
criterion_main!(file_tools_benches);
