use half::bf16;
use std::time::Instant;

use crate::matmul::Bf16MatMul;
use crate::vector::Bf16VectorOps;
use crate::stats::Bf16Stats;
use crate::kmeans::Bf16Kmeans;

/// Benchmark result for one operation.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub fp32_us: u128,
    pub bf16_us: u128,
    pub speedup: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<25} | FP32: {:>8}µs | BF16: {:>8}µs | speedup: {:.2}x",
            self.name, self.fp32_us, self.bf16_us, self.speedup
        )
    }
}

/// Benchmark suite comparing FP32 vs BF16.
pub struct BenchSuite;

impl BenchSuite {
    /// Run all benchmarks and return results.
    pub fn run_all() -> Vec<BenchResult> {
        let mut results = Vec::new();
        results.push(Self::bench_matmul());
        results.push(Self::bench_dot());
        results.push(Self::bench_cosine());
        results.push(Self::bench_mean_var());
        results.push(Self::bench_softmax());
        results.push(Self::bench_layer_norm());
        results.push(Self::bench_kmeans());
        results
    }

    fn random_data(n: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect()
    }

    fn bench_matmul() -> BenchResult {
        let m = 256;
        let a_f = Self::random_data(m * m);
        let b_f = Self::random_data(m * m);
        let a_b: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_b: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();

        // Warmup
        let _ = Bf16MatMul::multiply_f32(&a_f, m, m, &b_f, m);
        let _ = Bf16MatMul::new(m, m, m).multiply(&a_b, &b_b);

        let iterations = 5;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16MatMul::multiply_f32(&a_f, m, m, &b_f, m);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        let mm = Bf16MatMul::new(m, m, m);
        for _ in 0..iterations {
            let _ = mm.multiply(&a_b, &b_b);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "MatMul (256x256)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_dot() -> BenchResult {
        let n = 1_000_000;
        let a_f = Self::random_data(n);
        let b_f = Self::random_data(n);
        let a_b: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_b: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16VectorOps::dot_f32(&a_f, &b_f);
        let _ = Bf16VectorOps::dot(&a_b, &b_b);

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16VectorOps::dot_f32(&a_f, &b_f);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16VectorOps::dot(&a_b, &b_b);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "Dot Product (1M)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_cosine() -> BenchResult {
        let n = 100_000;
        let a_f = Self::random_data(n);
        let b_f = Self::random_data(n);
        let a_b: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_b: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16VectorOps::cosine_similarity_f32(&a_f, &b_f);
        let _ = Bf16VectorOps::cosine_similarity(&a_b, &b_b);

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16VectorOps::cosine_similarity_f32(&a_f, &b_f);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16VectorOps::cosine_similarity(&a_b, &b_b);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "Cosine Similarity (100K)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_mean_var() -> BenchResult {
        let n = 1_000_000;
        let data_f = Self::random_data(n);
        let data_b: Vec<bf16> = data_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16Stats::mean_f32(&data_f);
        let _ = Bf16Stats::mean(&data_b);

        let iterations = 20;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::mean_f32(&data_f);
            let _ = Bf16Stats::variance_f32(&data_f);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::mean(&data_b);
            let _ = Bf16Stats::variance(&data_b);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "Mean+Variance (1M)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_softmax() -> BenchResult {
        let n = 100_000;
        let data_f = Self::random_data(n);
        let data_b: Vec<bf16> = data_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16Stats::softmax_f32(&data_f);
        let _ = Bf16Stats::softmax(&data_b);

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::softmax_f32(&data_f);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::softmax(&data_b);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "Softmax (100K)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_layer_norm() -> BenchResult {
        let n = 100_000;
        let data_f = Self::random_data(n);
        let gamma_f: Vec<f32> = (0..n).map(|_| 1.0).collect();
        let beta_f: Vec<f32> = vec![0.0; n];
        let data_b: Vec<bf16> = data_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let gamma_b: Vec<bf16> = gamma_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let beta_b: Vec<bf16> = beta_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16Stats::layer_norm_f32(&data_f, &gamma_f, &beta_f, 1e-5);
        let _ = Bf16Stats::layer_norm(&data_b, &gamma_b, &beta_b, 1e-5);

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::layer_norm_f32(&data_f, &gamma_f, &beta_f, 1e-5);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Stats::layer_norm(&data_b, &gamma_b, &beta_b, 1e-5);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "Layer Norm (100K)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }

    fn bench_kmeans() -> BenchResult {
        let n = 10_000;
        let dim = 32;
        let k = 8;
        let data_f = Self::random_data(n * dim);
        let data_b: Vec<bf16> = data_f.iter().map(|&x| bf16::from_f32(x)).collect();

        let _ = Bf16Kmeans::fit_f32(k, 10, &data_f, n, dim);
        let km = Bf16Kmeans::new(k, 10);
        let _ = km.fit(&data_b, n, dim);

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Bf16Kmeans::fit_f32(k, 10, &data_f, n, dim);
        }
        let fp32_us = start.elapsed().as_micros() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = km.fit(&data_b, n, dim);
        }
        let bf16_us = start.elapsed().as_micros() / iterations;

        BenchResult {
            name: "K-Means (10Kx32, k=8)".into(),
            fp32_us,
            bf16_us,
            speedup: fp32_us as f64 / bf16_us as f64,
        }
    }
}
