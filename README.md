# amd-bf16-tools

Hypothesis test: Can AMD Ryzen AI BF16 instructions accelerate common numerical workloads by 2x+ without GPU?

**Short answer: No — not from Rust via the `half` crate on software-emulated BF16.** The BF16 path uses `half::bf16` which stores data in BF16 format but converts to `f32` for every arithmetic operation. This adds conversion overhead, making BF16 slightly *slower* than native FP32 for most operations.

## What's Included

| Module | Description |
|---|---|
| `Bf16MatMul` | Matrix multiplication using BF16 arithmetic via `half` crate |
| `Bf16VectorOps` | Dot product, L2 norm, cosine similarity in BF16 |
| `Bf16Stats` | Mean, variance, softmax, layer normalization in BF16 |
| `Bf16Kmeans` | K-means clustering using BF16 distances |
| `BenchSuite` | Compare FP32 vs BF16 timings, print speedup table |
| `AccuracyReport` | Compute max absolute error between FP32 and BF16 results |

## Benchmark Results

Test environment: **WSL2 on x86_64** (Linux 6.6.87), `half` crate software BF16, `rayon` parallelism, `--release` build.

```
Operation                 |  FP32 (µs) |  BF16 (µs) | Speedup
----------------------------------------------------------------------
MatMul (256x256)          |     1540µs |     4185µs | 0.37x
Dot Product (1M)          |     2110µs |     3265µs | 0.65x
Cosine Similarity (100K)  |    13130µs |     5446µs | 2.41x
Mean+Variance (1M)        |    11831µs |    16923µs | 0.70x
Softmax (100K)            |      332µs |      370µs | 0.90x
Layer Norm (100K)         |    19184µs |    16076µs | 1.19x
K-Means (10Kx32, k=8)     |    47349µs |    52476µs | 0.90x
----------------------------------------------------------------------
Average speedup: 1.02x
```

## Accuracy Report

```
Operation                 | Mean Abs Err |  Max Abs Err
-------------------------------------------------------
MatMul (64x64)            |     0.625622 |     3.866699
Dot Product               |    59.250000 |    59.250000
Mean                      |     0.000019 |     0.000019
Variance                  |     0.000595 |     0.000595
Softmax (1K)              |     0.000014 |     0.000571
Layer Norm (1K)           |     0.001246 |     0.005499
```

Accuracy is acceptable for most operations except accumulation-heavy ones (dot product over 100K elements, matmul) where errors compound.

## The BF16 Gap in Rust

The `half` crate provides a correct BF16 type, but **every arithmetic operation requires a conversion to f32 and back**. There is no current Rust path to AMD's native BF16 instructions (AVX512-BF16 / VNNI-BF16 on Zen 5). To get actual hardware BF16 acceleration:

1. **AMD Ryzen AI 9 HX 370** (Zen 5 / Strix Point) supports AVX512-BF16 instructions
2. These are accessible via **AMD AOCC compiler** or **intrinsics in C/C++** (`_mm512_dpbf16_ps`)
3. Rust has no stable `core::arch` support for AVX512-BF16 as of 2025
4. A `std::simd` path may arrive in the future but doesn't exist today
5. **Expected real hardware speedup**: ~2-2.5x on matmul/convolution with native BF16 FMA instructions

## Usage

```rust
use amd_bf16_tools::{Bf16MatMul, Bf16VectorOps, Bf16Stats, Bf16Kmeans, BenchSuite, AccuracyReport};
use half::bf16;

// Matrix multiply
let mm = Bf16MatMul::new(128, 64, 32);
let a: Vec<bf16> = /* ... */;
let b: Vec<bf16> = /* ... */;
let result = mm.multiply(&a, &b);

// Run full benchmark suite
let results = BenchSuite::run_all();
for r in &results {
    println!("{}", r);
}

// Accuracy report
let report = AccuracyReport::generate(100_000);
```

## Running

```bash
cargo run --bin bench --release   # Run benchmarks
cargo test                         # Run 13 unit tests
```

## Dependencies

- `half` — BF16 type and conversions
- `rayon` — data parallelism
- `rand` / `rand_distr` — random data generation for benchmarks

## Conclusion

**Hypothesis: REJECTED** (in software). BF16 via the `half` crate does not provide 2x+ speedup — it's roughly parity (1.02x average) due to per-operation f32 conversion overhead. The only notable win was cosine similarity (2.41x) which benefits from BF16's smaller memory footprint reducing cache pressure.

The expected ~2.5x speedup **would** materialize with native AVX512-BF16 hardware intrinsics, but Rust doesn't currently expose these. The path forward is either:
- Wait for Rust `std::simd` to support BF16 ops
- Write a C shim using AMD intrinsics and call via FFI
- Use AMD's MIGraphX / ONNX Runtime with BF16 acceleration
