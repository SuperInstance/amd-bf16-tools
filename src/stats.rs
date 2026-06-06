use half::bf16;
use rayon::prelude::*;

/// BF16 statistical operations: mean, variance, softmax, layer norm.
pub struct Bf16Stats;

impl Bf16Stats {
    /// Mean of BF16 values (accumulated in f32).
    pub fn mean(data: &[bf16]) -> f32 {
        let sum: f32 = data.par_iter().map(|&x| f32::from(x)).sum();
        sum / data.len() as f32
    }

    /// Variance of BF16 values.
    pub fn variance(data: &[bf16]) -> f32 {
        let m = Self::mean(data);
        let sum_sq: f32 = data.par_iter().map(|&x| (f32::from(x) - m).powi(2)).sum();
        sum_sq / data.len() as f32
    }

    /// Softmax: returns f32 vector (exponential needs full precision).
    pub fn softmax(data: &[bf16]) -> Vec<f32> {
        let vals: Vec<f32> = data.iter().map(|&x| f32::from(x)).collect();
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = vals.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Layer normalization with learnable scale (gamma) and bias (beta).
    /// Returns normalized values in f32.
    pub fn layer_norm(data: &[bf16], gamma: &[bf16], beta: &[bf16], eps: f32) -> Vec<f32> {
        assert_eq!(data.len(), gamma.len());
        assert_eq!(data.len(), beta.len());
        let m = Self::mean(data);
        let v = Self::variance(data);
        let std = (v + eps).sqrt();
        data.iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&x, &g), &b)| {
                let xn = (f32::from(x) - m) / std;
                xn * f32::from(g) + f32::from(b)
            })
            .collect()
    }

    /// FP32 reference mean.
    pub fn mean_f32(data: &[f32]) -> f32 {
        let sum: f32 = data.par_iter().sum();
        sum / data.len() as f32
    }

    /// FP32 reference variance.
    pub fn variance_f32(data: &[f32]) -> f32 {
        let m = Self::mean_f32(data);
        let sum_sq: f32 = data.par_iter().map(|&x| (x - m).powi(2)).sum();
        sum_sq / data.len() as f32
    }

    /// FP32 reference softmax.
    pub fn softmax_f32(data: &[f32]) -> Vec<f32> {
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// FP32 reference layer norm.
    pub fn layer_norm_f32(data: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let m = Self::mean_f32(data);
        let v = Self::variance_f32(data);
        let std = (v + eps).sqrt();
        data.iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&x, &g), &b)| (x - m) / std * g + b)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let v: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let m = Bf16Stats::mean(&v);
        assert!((m - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_variance() {
        let v: Vec<bf16> = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            .iter().map(|&x| bf16::from_f32(x)).collect();
        let v = Bf16Stats::variance(&v);
        assert!((v - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let v: Vec<bf16> = [1.0f32, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let s = Bf16Stats::softmax(&v);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_zero_gamma() {
        let data: Vec<bf16> = [1.0f32, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let gamma: Vec<bf16> = [0.0f32, 0.0, 0.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let beta: Vec<bf16> = [0.0f32, 0.0, 0.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let result = Bf16Stats::layer_norm(&data, &gamma, &beta, 1e-5);
        assert!(result.iter().all(|&x| x.abs() < 1e-4));
    }
}
