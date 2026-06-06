use half::bf16;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

/// K-means clustering using BF16 distances.
pub struct Bf16Kmeans {
    pub k: usize,
    pub max_iters: usize,
}

impl Bf16Kmeans {
    pub fn new(k: usize, max_iters: usize) -> Self {
        Self { k, max_iters }
    }

    /// Run k-means on row-major data with `n` points of `dim` dimensions.
    /// Returns (centroids, assignments).
    pub fn fit(&self, data: &[bf16], n: usize, dim: usize) -> (Vec<bf16>, Vec<usize>) {
        assert_eq!(data.len(), n * dim);

        // Initialize centroids using k-means++ for deterministic-quality seeding
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<bf16> = Vec::with_capacity(self.k * dim);

        // Pick first centroid randomly
        let first_idx = rng.gen_range(0..n);
        for d in 0..dim {
            centroids.push(data[first_idx * dim + d]);
        }

        // Pick remaining centroids proportional to squared distance
        for _ in 1..self.k {
            let distances: Vec<f32> = (0..n)
                .map(|i| {
                    let mut min_dist = f32::MAX;
                    for c in 0..centroids.len() / dim {
                        let cstart = c * dim;
                        let mut dist = 0.0f32;
                        for d in 0..dim {
                            let diff = f32::from(data[i * dim + d]) - f32::from(centroids[cstart + d]);
                            dist += diff * diff;
                        }
                        min_dist = min_dist.min(dist);
                    }
                    min_dist
                })
                .collect();

            let total: f32 = distances.iter().sum();
            let mut target = rng.gen_range(0.0..total);
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                target -= d;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            for d in 0..dim {
                centroids.push(data[chosen * dim + d]);
            }
        }

        let mut assignments = vec![0usize; n];

        for _ in 0..self.max_iters {
            // Assign each point to nearest centroid (parallel)
            let new_assignments: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &data[i * dim..(i + 1) * dim];
                    let mut best = 0;
                    let mut best_dist = f32::MAX;
                    for c in 0..self.k {
                        let cstart = c * dim;
                        let mut dist = 0.0f32;
                        for d in 0..dim {
                            let diff = f32::from(point[d]) - f32::from(centroids[cstart + d]);
                            dist += diff * diff;
                        }
                        if dist < best_dist {
                            best_dist = dist;
                            best = c;
                        }
                    }
                    best
                })
                .collect();

            // Check convergence
            if new_assignments == assignments {
                break;
            }
            assignments = new_assignments;

            // Recompute centroids
            let mut sums = vec![0.0f32; self.k * dim];
            let mut counts = vec![0usize; self.k];
            for (i, &c) in assignments.iter().enumerate() {
                counts[c] += 1;
                for d in 0..dim {
                    sums[c * dim + d] += f32::from(data[i * dim + d]);
                }
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centroids[c * dim + d] = bf16::from_f32(sums[c * dim + d] / counts[c] as f32);
                    }
                }
            }
        }

        (centroids, assignments)
    }

    /// FP32 reference k-means.
    pub fn fit_f32(k: usize, max_iters: usize, data: &[f32], n: usize, dim: usize) -> (Vec<f32>, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
        let mut used = std::collections::HashSet::new();
        for _ in 0..k {
            let mut idx;
            loop {
                idx = rng.gen_range(0..n);
                if used.insert(idx) { break; }
            }
            for d in 0..dim {
                centroids.push(data[idx * dim + d]);
            }
        }

        let mut assignments = vec![0usize; n];
        for _ in 0..max_iters {
            let new_assignments: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &data[i * dim..(i + 1) * dim];
                    let mut best = 0;
                    let mut best_dist = f32::MAX;
                    for c in 0..k {
                        let cstart = c * dim;
                        let mut dist = 0.0f32;
                        for d in 0..dim {
                            let diff = point[d] - centroids[cstart + d];
                            dist += diff * diff;
                        }
                        if dist < best_dist {
                            best_dist = dist;
                            best = c;
                        }
                    }
                    best
                })
                .collect();

            if new_assignments == assignments { break; }
            assignments = new_assignments;

            let mut sums = vec![0.0f32; k * dim];
            let mut counts = vec![0usize; k];
            for (i, &c) in assignments.iter().enumerate() {
                counts[c] += 1;
                for d in 0..dim {
                    sums[c * dim + d] += data[i * dim + d];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centroids[c * dim + d] = sums[c * dim + d] / counts[c] as f32;
                    }
                }
            }
        }
        (centroids, assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_two_clusters() {
        // Two clusters far apart — gap must survive BF16 quantization + random init
        let mut data_f32 = Vec::new();
        for _ in 0..50 {
            data_f32.extend_from_slice(&[0.0f32, 0.0]);
        }
        for _ in 0..50 {
            data_f32.extend_from_slice(&[10000.0f32, 10000.0]);
        }
        let data: Vec<bf16> = data_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let km = Bf16Kmeans::new(2, 100);
        let (centroids, assignments) = km.fit(&data, 100, 2);

        assert_eq!(assignments.len(), 100);
        let g0 = assignments[0];
        let g1 = assignments[50];
        assert_ne!(g0, g1, "two groups must have different assignments");
        assert!(assignments[0..50].iter().all(|&a| a == g0), "first group inconsistent");
        assert!(assignments[50..100].iter().all(|&a| a == g1), "second group inconsistent");
    }

    #[test]
    fn test_kmeans_single_cluster() {
        // With k=1, after one recompute the centroid = mean of all points.
        // Use enough points that the initial pick doesn't matter much.
        let data_f32: Vec<f32> = vec![10.0; 100];
        let data: Vec<bf16> = data_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let km = Bf16Kmeans::new(1, 10);
        let (centroids, assignments) = km.fit(&data, 100, 1);
        assert!(assignments.iter().all(|&a| a == 0));
        let c = f32::from(centroids[0]);
        assert!((c - 10.0).abs() < 0.1, "centroid={}", c);
    }
}
