use half::bf16;
use rayon::prelude::*;

/// BF16 vector operations: dot product, norms, cosine similarity.
pub struct Bf16VectorOps;

impl Bf16VectorOps {
    /// Dot product of two BF16 vectors, accumulated in f32.
    pub fn dot(a: &[bf16], b: &[bf16]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&av, &bv)| f32::from(av) * f32::from(bv))
            .sum()
    }

    /// L2 norm (Euclidean) of a BF16 vector.
    pub fn l2_norm(v: &[bf16]) -> f32 {
        let sum_sq: f32 = v.par_iter().map(|&x| f32::from(x).powi(2)).sum();
        sum_sq.sqrt()
    }

    /// Cosine similarity between two BF16 vectors.
    pub fn cosine_similarity(a: &[bf16], b: &[bf16]) -> f32 {
        let d = Self::dot(a, b);
        let na = Self::l2_norm(a);
        let nb = Self::l2_norm(b);
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        d / (na * nb)
    }

    /// FP32 reference dot product.
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.par_iter().zip(b.par_iter()).map(|(&av, &bv)| av * bv).sum()
    }

    /// FP32 reference L2 norm.
    pub fn l2_norm_f32(v: &[f32]) -> f32 {
        let sum_sq: f32 = v.par_iter().map(|&x| x.powi(2)).sum();
        sum_sq.sqrt()
    }

    /// FP32 reference cosine similarity.
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        let d = Self::dot_f32(a, b);
        let na = Self::l2_norm_f32(a);
        let nb = Self::l2_norm_f32(b);
        if na == 0.0 || nb == 0.0 { return 0.0; }
        d / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a: Vec<bf16> = [1.0f32, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let b: Vec<bf16> = [4.0f32, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let d = Bf16VectorOps::dot(&a, &b);
        assert!((d - 32.0).abs() < 0.5);
    }

    #[test]
    fn test_l2_norm() {
        let v: Vec<bf16> = [3.0f32, 4.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let n = Bf16VectorOps::l2_norm(&v);
        assert!((n - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v: Vec<bf16> = [1.0f32, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let sim = Bf16VectorOps::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a: Vec<bf16> = [1.0f32, 0.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let b: Vec<bf16> = [0.0f32, 1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let sim = Bf16VectorOps::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.01);
    }
}
