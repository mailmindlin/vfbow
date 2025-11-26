//! L2 (Euclidean) distance computation

use super::util::specialize_array;

/// Helper trait to write generic loops for L2 distance computation
pub(super) trait AccumulateL2 {
	/// Accumulator type
	type Accumulator: Sized;

	/// Create initial accumulator value (zero)
	fn init() -> Self::Accumulator;
	/// Update accumulator with distance between two elements
	fn update(self, other: Self, acc: Self::Accumulator) -> Self::Accumulator;
	/// Finalize and return distance from accumulator
	fn finish(acc: Self::Accumulator) -> f32;
}

/// An element for which L2 distance can be computed
pub(super) trait ElementL2 {
	/// Compute the squared L2 distance between two elements
	fn distance_l2(self, other: Self) -> f32;
}

#[cfg(feature="f16")]
impl ElementL2 for f16 {
	#[inline]
	fn distance_l2(self, other: Self) -> f32 {
		//TODO: when should this conversion happen?
		let delta = (self - other).to_f32();
		delta * delta
	}
}

impl ElementL2 for f32 {
	#[inline]
	fn distance_l2(self, other: Self) -> f32 {
		let delta = self - other;
		delta * delta
	}
}

impl<T: ElementL2> AccumulateL2 for T {
	type Accumulator = f32;

	#[inline(always)]
	fn init() -> Self::Accumulator { 0. }
	#[inline(always)]
	fn update(self, other: Self, acc: Self::Accumulator) -> Self::Accumulator {
		acc + self.distance_l2(other)
	}
	#[inline(always)]
	fn finish(acc: Self::Accumulator) -> f32 { acc }
}

specialize_array! {
	/// Compute the L2 distance
	pub(super) fn generic<N>(reference: &[f32], feature: &[f32]) -> f32 {
		// Substract, multiply and accumulate
		let mut sum: f32 = 0.;
		for i in 0..N {
			let diff = feature[i] - reference[i];
			sum += diff * diff
		}
		sum
	}
}


#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use super::arch::aarch_intrinsics::{float32x4_t};

specialize_array! {
	/// Compute the L2 distance between two slices using NEON
	#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
	#[target_feature(enable = "neon")]
	pub(super) fn neon<N>(reference: &[float32x4_t], feature: &[float32x4_t]) -> f32 {
		use std::arch::aarch64::{vadd_f32, vdupq_n_f32, vget_high_f32, vget_low_f32, vmlaq_f32, vpadd_f32, vsubq_f32, vget_lane_f32};
		
		//substract, multiply and accumulate
		let mut sum = vdupq_n_f32(0.);
		for i in 0..N {
			let diff = vsubq_f32(feature[i], reference[i]);
			sum = vmlaq_f32(sum, diff, diff);
		}
		// Reduce pairwise, twice
		let sum = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
		vget_lane_f32::<0>(vpadd_f32(sum, sum))
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn l2_sse_slice(reference: &[std::arch::x86_64::__m128], feature: &[std::arch::x86_64::__m128]) -> f32 {
	assert_eq!(reference.len(), feature.len());
	use std::arch::x86_64::{_mm_cvtss_f32, _mm_hadd_ps, _mm_setzero_ps, _mm_mul_ps, _mm_add_ps, _mm_sub_ps};
	//substract, multiply and accumulate
	let mut sum = _mm_setzero_ps();
	for i in 0..reference.len() {
		let diff = _mm_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm_mul_ps(diff, diff);
		sum = _mm_add_ps(sum, diff_sq);
	}

	// Reduce pairwise, twice
	let sum = _mm_hadd_ps(sum,sum);
	let sum = _mm_hadd_ps(sum,sum);
	_mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
pub(super) unsafe fn l2_sse_array<const N: usize>(reference: &[std::arch::x86_64::__m128; N], feature: &[std::arch::x86_64::__m128; N]) -> f32 {
	use std::arch::x86_64::{_mm_cvtss_f32, _mm_hadd_ps, _mm_setzero_ps, _mm_mul_ps, _mm_add_ps, _mm_sub_ps};
	//substract, multiply and accumulate
	let mut sum = _mm_setzero_ps();
	for i in 0..N {
		let diff = _mm_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm_mul_ps(diff, diff);
		sum = _mm_add_ps(sum, diff_sq);
	}

	// Reduce pairwise, twice
	let sum = _mm_hadd_ps(sum,sum);
	let sum = _mm_hadd_ps(sum,sum);
	_mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn l2_avx_slice(reference: &[std::arch::x86_64::__m256], feature: &[std::arch::x86_64::__m256]) -> f32 {
	use std::{arch::x86_64::{_mm256_add_ps, _mm256_hadd_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_store_ps, _mm256_sub_ps}, ptr};
	assert_eq!(reference.len(), feature.len());
	
	//substract, multiply and accumulate
	let mut sum = _mm256_setzero_ps();
	for i in 0..reference.len() {
		let diff = _mm256_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm256_mul_ps(diff, diff);
		sum = _mm256_add_ps(sum, diff_sq);
	}
	// Reduce pairwise, twice
	let sum = _mm256_hadd_ps(sum,sum);
	let sum = _mm256_hadd_ps(sum,sum);
	//TODO: it might be worth doing another AVX reduce + element load instead of this
	#[repr(align(32))]
	struct Memory([f32; 8]);
	let mut memory = Memory([0.; 8]);
	_mm256_store_ps(ptr::addr_of_mut!(memory.0[0]), sum);
	memory.0[0] + memory.0[4]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn l2_avx_array<const N: usize>(reference: &[std::arch::x86_64::__m256; N], feature: &[std::arch::x86_64::__m256; N]) -> f32 {
	use std::{arch::x86_64::{_mm256_add_ps, _mm256_hadd_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_store_ps, _mm256_sub_ps}, ptr};
	
	//substract, multiply and accumulate
	let mut sum = _mm256_setzero_ps();
	for i in 0..N {
		let diff = _mm256_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm256_mul_ps(diff, diff);
		sum = _mm256_add_ps(sum, diff_sq);
	}
	// Reduce pairwise, twice
	let sum = _mm256_hadd_ps(sum,sum);
	let sum = _mm256_hadd_ps(sum,sum);
	//TODO: it might be worth doing another AVX reduce + element load instead of this
	#[repr(align(32))]
	struct Memory([f32; 8]);
	let mut memory = Memory([0.; 8]);
	_mm256_store_ps(ptr::addr_of_mut!(memory.0[0]), sum);
	memory.0[0] + memory.0[4]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn l2_avx512_slice(reference: &[std::arch::x86_64::__m512], feature: &[std::arch::x86_64::__m512]) -> f32 {
	use std::arch::x86_64::{_mm512_add_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_setzero_ps, _mm512_sub_ps};
	assert_eq!(reference.len(), feature.len());
	
	//substract, multiply and accumulate
	let mut sum = _mm512_setzero_ps();
	for i in 0..reference.len() {
		let diff = _mm512_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm512_mul_ps(diff, diff);
		sum = _mm512_add_ps(sum, diff_sq);
	}
	// Reduce pairwise, twice
	_mm512_reduce_add_ps(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn l2_avx512_array<const N: usize>(reference: &[std::arch::x86_64::__m512; N], feature: &[std::arch::x86_64::__m512; N]) -> f32 {
	use std::arch::x86_64::{_mm512_add_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_setzero_ps, _mm512_sub_ps};
	
	//substract, multiply and accumulate
	let mut sum = _mm512_setzero_ps();
	for i in 0..N {
		let diff = _mm512_sub_ps(feature[i], reference[i]);
		let diff_sq = _mm512_mul_ps(diff, diff);
		sum = _mm512_add_ps(sum, diff_sq);
	}
	// Reduce pairwise, twice
	_mm512_reduce_add_ps(sum)
}