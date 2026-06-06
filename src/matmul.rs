use half::bf16;
use rayon::prelude::*;

/// BF16 matrix multiplication engine.
pub struct Bf16MatMul {
    pub rows_a: usize,
    pub cols_a: usize,
    pub cols_b: usize,
}

impl Bf16MatMul {
    /// Create a new matmul context for A(rows_a x cols_a) @ B(cols_a x cols_b).
    pub fn new(rows_a: usize, cols_a: usize, cols_b: usize) -> Self {
        Self { rows_a, cols_a, cols_b }
    }

    /// Multiply two BF16 matrices. Returns result in BF16.
    pub fn multiply(&self, a: &[bf16], b: &[bf16]) -> Vec<bf16> {
        assert_eq!(a.len(), self.rows_a * self.cols_a);
        assert_eq!(b.len(), self.cols_a * self.cols_b);

        let cols_b = self.cols_b;
        let cols_a = self.cols_a;
        (0..self.rows_a * self.cols_b)
            .into_par_iter()
            .map(|idx| {
                let i = idx / cols_b;
                let j = idx % cols_b;
                let mut sum: f32 = 0.0;
                for k in 0..cols_a {
                    let av = f32::from(a[i * cols_a + k]);
                    let bv = f32::from(b[k * cols_b + j]);
                    sum += av * bv;
                }
                bf16::from_f32(sum)
            })
            .collect()
    }

    /// Reference FP32 matrix multiply for comparison.
    pub fn multiply_f32(a: &[f32], rows_a: usize, cols_a: usize, b: &[f32], cols_b: usize) -> Vec<f32> {
        assert_eq!(a.len(), rows_a * cols_a);
        assert_eq!(b.len(), cols_a * cols_b);

        (0..rows_a * cols_b)
            .into_par_iter()
            .map(|idx| {
                let i = idx / cols_b;
                let j = idx % cols_b;
                let mut sum: f32 = 0.0;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                sum
            })
            .collect()
    }

    /// Convert a slice of f32 to bf16.
    pub fn f32_to_bf16(data: &[f32]) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// Convert a slice of bf16 to f32.
    pub fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
        data.iter().map(|&x| f32::from(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_multiply() {
        // A = [[1,0],[0,1]], B = [[2,3],[4,5]] => [[2,3],[4,5]]
        let a: Vec<bf16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let b: Vec<bf16> = [2.0f32, 3.0, 4.0, 5.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mm = Bf16MatMul::new(2, 2, 2);
        let c = mm.multiply(&a, &b);
        let c32: Vec<f32> = c.iter().map(|&x| f32::from(x)).collect();
        assert!((c32[0] - 2.0).abs() < 0.01);
        assert!((c32[3] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_known_multiply() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let b: Vec<bf16> = [5.0f32, 6.0, 7.0, 8.0].iter().map(|&x| bf16::from_f32(x)).collect();
        let mm = Bf16MatMul::new(2, 2, 2);
        let c = mm.multiply(&a, &b);
        let c32: Vec<f32> = c.iter().map(|&x| f32::from(x)).collect();
        assert!((c32[0] - 19.0).abs() < 0.5);
        assert!((c32[1] - 22.0).abs() < 0.5);
        assert!((c32[2] - 43.0).abs() < 0.5);
        assert!((c32[3] - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_conversion_roundtrip() {
        let vals = vec![1.0f32, 2.5, -3.14, 100.0];
        let bf = Bf16MatMul::f32_to_bf16(&vals);
        let back = Bf16MatMul::bf16_to_f32(&bf);
        for (orig, round) in vals.iter().zip(back.iter()) {
            assert!((orig - round).abs() < 0.1 * orig.abs().max(1.0));
        }
    }
}
