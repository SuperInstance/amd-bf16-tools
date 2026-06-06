use half::bf16;

use crate::matmul::Bf16MatMul;
use crate::vector::Bf16VectorOps;
use crate::stats::Bf16Stats;

/// Accuracy report comparing BF16 vs FP32 results.
pub struct AccuracyReport;

impl AccuracyReport {
    /// Compute max absolute error between two f32 vectors.
    pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Compute mean absolute error.
    pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sum: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum();
        sum / a.len() as f32
    }

    /// Generate a full accuracy report across all operations.
    pub fn generate(data_size: usize) -> Vec<(String, f32, f32)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_data: Vec<f32> = (0..data_size).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let mut report = Vec::new();

        // MatMul accuracy
        {
            let m = 64;
            let a_f = &random_data[..m * m];
            let b_f = &random_data[m * m..2 * m * m];
            let a_b: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
            let b_b: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16MatMul::multiply_f32(a_f, m, m, b_f, m);
            let bf16 = Bf16MatMul::new(m, m, m).multiply(&a_b, &b_b);
            let bf16_f: Vec<f32> = bf16.iter().map(|&x| f32::from(x)).collect();
            let mae = Self::mean_abs_error(&fp32, &bf16_f);
            let maxe = Self::max_abs_error(&fp32, &bf16_f);
            report.push(("MatMul (64x64)".into(), mae, maxe));
        }

        // Dot product accuracy
        {
            let a_f = &random_data[..data_size];
            let b_f = &random_data[..data_size]; // reuse for simplicity
            let a_b: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
            let b_b: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16VectorOps::dot_f32(a_f, b_f);
            let bf16 = Bf16VectorOps::dot(&a_b, &b_b);
            let err = (fp32 - bf16).abs();
            report.push(("Dot Product".into(), err, err));
        }

        // Mean accuracy
        {
            let data_b: Vec<bf16> = random_data.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16Stats::mean_f32(&random_data);
            let bf16 = Bf16Stats::mean(&data_b);
            let err = (fp32 - bf16).abs();
            report.push(("Mean".into(), err, err));
        }

        // Variance accuracy
        {
            let data_b: Vec<bf16> = random_data.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16Stats::variance_f32(&random_data);
            let bf16 = Bf16Stats::variance(&data_b);
            let err = (fp32 - bf16).abs();
            report.push(("Variance".into(), err, err));
        }

        // Softmax accuracy
        {
            let subset = &random_data[..1000];
            let data_b: Vec<bf16> = subset.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16Stats::softmax_f32(subset);
            let bf16 = Bf16Stats::softmax(&data_b);
            let mae = Self::mean_abs_error(&fp32, &bf16);
            let maxe = Self::max_abs_error(&fp32, &bf16);
            report.push(("Softmax (1K)".into(), mae, maxe));
        }

        // Layer norm accuracy
        {
            let subset = &random_data[..1000];
            let gamma: Vec<f32> = vec![1.0; 1000];
            let beta: Vec<f32> = vec![0.0; 1000];
            let data_b: Vec<bf16> = subset.iter().map(|&x| bf16::from_f32(x)).collect();
            let gamma_b: Vec<bf16> = gamma.iter().map(|&x| bf16::from_f32(x)).collect();
            let beta_b: Vec<bf16> = beta.iter().map(|&x| bf16::from_f32(x)).collect();
            let fp32 = Bf16Stats::layer_norm_f32(subset, &gamma, &beta, 1e-5);
            let bf16 = Bf16Stats::layer_norm(&data_b, &gamma_b, &beta_b, 1e-5);
            let mae = Self::mean_abs_error(&fp32, &bf16);
            let maxe = Self::max_abs_error(&fp32, &bf16);
            report.push(("Layer Norm (1K)".into(), mae, maxe));
        }

        report
    }
}
