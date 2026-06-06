use amd_bf16_tools::{BenchSuite, AccuracyReport};

fn main() {
    println!("=== AMD BF16 Tools — Benchmark Suite ===\n");

    println!("Running performance benchmarks...\n");
    let results = BenchSuite::run_all();

    println!("{:<25} | {:>10} | {:>10} | {}", "Operation", "FP32 (µs)", "BF16 (µs)", "Speedup");
    println!("{}", "-".repeat(70));
    for r in &results {
        println!("{:<25} | {:>8}µs | {:>8}µs | {:.2}x", r.name, r.fp32_us, r.bf16_us, r.speedup);
    }

    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    println!("{}", "-".repeat(70));
    println!("Average speedup: {:.2}x\n", avg_speedup);

    println!("=== Accuracy Report (BF16 vs FP32) ===\n");
    println!("{:<25} | {:>12} | {:>12}", "Operation", "Mean Abs Err", "Max Abs Err");
    println!("{}", "-".repeat(55));
    let accuracy = AccuracyReport::generate(100_000);
    for (name, mae, maxe) in &accuracy {
        println!("{:<25} | {:>12.6} | {:>12.6}", name, mae, maxe);
    }
}
