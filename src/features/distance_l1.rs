//! L1 (Hamming) distance computations

/// Trait to help write generic code to compute L1 (Hamming) distances
/// 
/// The idea here is that we can use this trait to write generic loops like:
/// ```ignore
/// let mut acc = T::init();
/// for (a, b) in reference.iter().zip(feature.iter()) {
///    acc = a.update(*b, acc);
/// }
/// let distance = T::finish(acc);
/// ```
pub(super) trait AccumulateL1 {
	/// Accumulator type
	type Accumulator: Sized;

	/// Create initial accumulator value (zero)
	fn init() -> Self::Accumulator;
	/// Update accumulator with distance between two elements
	fn update(self, other: Self, acc: Self::Accumulator) -> Self::Accumulator;
	/// Finalize and return distance from accumulator
	fn finish(acc: Self::Accumulator) -> u32;
}

/// An element that can compute L1 distance
pub(super) trait ElementL1 {
	/// Compute L1 distance between two elements
	fn distance_l1(self, other: Self) -> u32;
}

impl ElementL1 for u8 {
	#[inline]
	fn distance_l1(self, other: Self) -> u32 {
		(self ^ other).count_ones()
	}
}

impl ElementL1 for u32 {
	#[inline]
	fn distance_l1(self, other: Self) -> u32 {
		(self ^ other).count_ones()
	}
}

impl ElementL1 for u64 {
	#[inline]
	fn distance_l1(self, other: Self) -> u32 {
		(self ^ other).count_ones()
	}
}


impl<T: ElementL1> AccumulateL1 for T {
	type Accumulator = u32;

	#[inline(always)]
	fn init() -> Self::Accumulator { 0 }
	#[inline(always)]
	fn update(self, other: Self, acc: Self::Accumulator) -> Self::Accumulator {
		acc + self.distance_l1(other)
	}
	#[inline(always)]
	fn finish(acc: Self::Accumulator) -> u32 { acc }
}

/*//generic hamming distance calculator
pub(super) fn l1_x8(reference: &[u64], feature: &[u64]) -> u64 {
	assert_eq!(reference.len(), feature.len());
	reference.iter().zip(feature.iter())
		.map(|(x, y)| (x ^ y).count_ones() as u64)
		.sum()
}

fn l1_x4(reference: &[u32], feature: &[u32]) -> u32 {
	assert_eq!(reference.len(), feature.len());
	reference.iter().zip(feature.iter())
		.map(|(x, y)| (x ^ y).count_ones())
		.sum()
}

fn l1_array<const N: usize>(reference: &[u64; N], feature: &[u64; N]) -> u32 {
	reference.iter().zip(feature.iter())
		.map(|(x, y)| (x ^ y).count_ones())
		.sum()
}

//TODO: should we just use l1_array specializations instead?
//for orb
fn l1_x32(reference: &[u64; 4], feature: &[u64; 4]) -> u32 {
	(reference[0] ^ feature[0]).count_ones()
	+ (reference[1] ^ feature[1]).count_ones()
	+ (reference[2] ^ feature[2]).count_ones()
	+ (reference[3] ^ feature[3]).count_ones()
}

 //for akaze
fn l1_x64(reference: &[u64; 8], feature: &[u64; 8]) -> u32 {
	(reference[0] ^ feature[0]).count_ones()
	+ (reference[1] ^ feature[1]).count_ones()
	+ (reference[2] ^ feature[2]).count_ones()
	+ (reference[3] ^ feature[3]).count_ones()
	+ (reference[4] ^ feature[4]).count_ones()
	+ (reference[5] ^ feature[5]).count_ones()
	+ (reference[6] ^ feature[6]).count_ones()
	+ (reference[7] ^ feature[7]).count_ones()
}*/

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn l1_neon_array<const N: usize>(reference: &[std::arch::aarch64::uint8x16_t; N], feature: &[std::arch::aarch64::uint8x16_t; N]) -> u32 {
	use std::arch::aarch64::*;
	let mut acc = vdupq_n_u16(0);
	//TODO: fix for overflows when N>2**16
	for i in 0..N {
		let delta = veorq_u8(reference[i], feature[i]);
		let counts = vcntq_u8(delta);
		acc = vpadalq_u8(acc, counts);
	}
	
	// Pairwise reduce
	let acc = vpaddlq_u16(acc);
	let acc  = vadd_u32(vget_high_u32(acc), vget_low_u32(acc));
	vget_lane_u32::<0>(vpadd_u32(acc, acc))
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn l1_neon_slice(reference: &[std::arch::aarch64::uint8x16_t], feature: &[std::arch::aarch64::uint8x16_t]) -> u32 {
	use std::arch::aarch64::*;
	assert_eq!(reference.len(), feature.len());
	let mut acc = vdupq_n_u16(0);
	//TODO: fix for overflows when N>2**16
	for i in 0..reference.len() {
		let delta = veorq_u8(reference[i], feature[i]);
		let counts = vcntq_u8(delta);
		acc = vpadalq_u8(acc, counts);
	}
	
	// Pairwise reduce
	let acc = vpaddlq_u16(acc);
	let acc  = vadd_u32(vget_high_u32(acc), vget_low_u32(acc));
	vget_lane_u32::<0>(vpadd_u32(acc, acc))
}