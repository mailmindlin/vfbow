use std::{fmt::Debug, num::NonZeroUsize};

use ndarray::{linalg::Dot, Array1, ArrayView1};
use num_traits::Zero;

use crate::{features::FeatureType, util::DescriptorType};

use super::feature::FeatureInfo;

#[allow(private_bounds)]
pub trait VocabElement: DistFunc + FeatureType + Clone + Zero + Debug {
	const MIN_ALIGNMENT: usize;
	const TYPE: DescriptorType;
	fn prefer_alignment(_ncols: NonZeroUsize) -> usize {
		Self::MIN_ALIGNMENT
	}
}
impl VocabElement for u8 {
	const MIN_ALIGNMENT: usize = 8;
	const TYPE: DescriptorType = DescriptorType::Uint8;
	fn prefer_alignment(ncols: NonZeroUsize) -> usize {
		#[cfg(any(target_arch="x86_64", target_arch="x86"))]
		use std::arch::{x86_64::{__m128i, __m256i, __m512i}, is_x86_feature_detected};
		#[cfg(target_arch="aarch64")]
		use std::arch::{aarch64::uint8x16_t, is_aarch64_feature_detected};
		//TODO: check that these work
		#[cfg(target_arch="x86")]
		use std::arch::{x86::{__m128i, __m256i, __m512i}, is_x86_feature_detected};
		#[cfg(target_arch="arm")]
		use std::arch::{arm::uint8x16_t, is_arm_feature_detected as is_aarch64_feature_detected};

		let ncols = ncols.get();
		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(64) && is_x86_feature_detected!("avx512f") {
			return align_of::<__m512i>();
		}
		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(32) && is_x86_feature_detected!("avx") {
			return align_of::<__m256i>();
		}
		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(32) && is_x86_feature_detected!("sse") {
			return align_of::<__m128i>();
		}

		#[cfg(any(target_arch="aarch64", target_arch="arm"))]
		if ncols.is_multiple_of(16) && is_aarch64_feature_detected!("neon") {
			// NEON 
			return align_of::<uint8x16_t>();
		}

		// Try using u128
		// TODO does this have any performance benefit?
		if ncols.is_multiple_of(16) {
			align_of::<u128>()
		} else if ncols.is_multiple_of(8) {
			align_of::<u64>()
		} else {
			align_of::<u8>()
		}
	}
}
impl VocabElement for f32 {
	const MIN_ALIGNMENT: usize = 32;
	const TYPE: DescriptorType = DescriptorType::Float32;

	fn prefer_alignment(ncols: NonZeroUsize) -> usize {
		#[cfg(any(target_arch="x86_64", target_arch="x86"))]
		use std::arch::{x86_64::{__m128, __m256, __m512}, is_x86_feature_detected};
		#[cfg(target_arch="aarch64")]
		use std::arch::{aarch64::{float32x4_t, float32x2_t}, is_aarch64_feature_detected};
		//TODO: check that these work
		#[cfg(target_arch="x86")]
		use std::arch::{x86::{__m128, __m256, __m512}, is_x86_feature_detected};
		#[cfg(target_arch="arm")]
		use std::arch::{arm::{float32x4_t, float32x2_t}, is_arm_feature_detected as is_aarch64_feature_detected};

		let ncols = ncols.get();

		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(16) && is_x86_feature_detected!("avx512f") {
			return align_of::<__m512>();
		}
		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(8) && is_x86_feature_detected!("avx") {
			return align_of::<__m256>();
		}
		#[cfg(any(target_arch="x86", target_arch="x86_64"))]
		if ncols.is_multiple_of(4) && is_x86_feature_detected!("sse") {
			return align_of::<__m128>();
		}

		#[cfg(any(target_arch="aarch64", target_arch="arm"))]
		if ncols.is_multiple_of(2) && is_aarch64_feature_detected!("neon") {
			// NEON
			return if ncols.is_multiple_of(4) {
				align_of::<float32x4_t>()
			} else {
				align_of::<float32x2_t>()
			}
		}

		align_of::<f32>()
	}
}


pub(super) trait DistFunc: Sized {
	fn dist_func(a: ArrayView1<Self>, b: ArrayView1<Self>) -> f32;
	fn mean_values(features: &FeatureInfo<Self>, indices: &[u32]) -> Array1<Self>;
}

impl DistFunc for f32 {
	fn dist_func(a: ArrayView1<Self>, b: ArrayView1<Self>) -> f32 {
		Dot::dot(&a, &b)
	}

	fn mean_values(features: &FeatureInfo<Self>, indices: &[u32]) -> Array1<Self> {
		features.mean_value(indices.iter().map(|idx| *idx as usize))
	}
}

impl DistFunc for u8 {
	fn dist_func(a: ArrayView1<Self>, b: ArrayView1<Self>) -> f32 {
		assert_eq!(a.len(), b.len());
		//TODO: We can do this without memory allocations
		let x = (&a ^ &b)
			.mapv_into_any(|x| x.count_ones())
			.sum();
		x as f32
		/*const uchar *pa = a.ptr<uchar>(); // a & b are actually CV_8U
		const uchar *pb = b.ptr<uchar>();
		for(int i=0;i<a.cols;i++,pa++,pb++){
			uchar v=(*pa)^(*pb);
	#ifdef __GNUG__
			ret+=__builtin_popcount(v);//only in g++
	#else
			ret+=(v& (1))!=0;
			ret+=(v& (2))!=0;
			ret+=(v& (4))!=0;
			ret+=(v& (8))!=0;
			ret+=(v& (16))!=0;
			ret+=(v& (32))!=0;
			ret+=(v& (64))!=0;
			ret+=(v& (128))!=0;
	#endif
		}
		return ret;*/
	}

	fn mean_values(features: &FeatureInfo<Self>, indices: &[u32]) -> Array1<Self> {
		features.mean_value(indices.iter().map(|idx| *idx as usize))
	}
}