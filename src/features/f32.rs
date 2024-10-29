#[cfg(target_arch="aarch64")]
use std::arch::{is_aarch64_feature_detected, aarch64::float32x4_t};
#[cfg(target_arch="arm")]
use std::arch::{is_arm_feature_detected, arm::float32x4_t};
#[cfg(any(target_arch="x86_64"))]
use std::arch::{is_x86_feature_detected, x86_64::{__m128, __m256, __m512}};
#[cfg(target_arch="x86")]
use std::arch::{is_x86_feature_detected, x86::{__m128, __m256, __m512}};
use std::{any, borrow::Cow, mem::MaybeUninit};

use ndarray::ArrayView1;

#[cfg(any(target_arch="x86_64", target_arch="x86"))]
use crate::features::distance_l2::{l2_avx512_array, l2_avx_array, l2_sse_array};
use crate::features::{shared::ToArray, Features};
use crate::util::serde::{read_u32ish, write_u32ish};
#[cfg(any(target_arch="aarch64", target_arch="arm"))]
use super::distance_l2::{l2_neon_slice, l2_neon_array};
use super::{distance_l2::{l2_array, l2_slice, AccumulateL2}, shared::is_slice_packed};
use crate::{Deserialize, Serialize};

use super::{DistanceQuery, FeaturesGeneric};
use super::shared::{AlignQuery, FeatureDistance, FromArray, L2};

/// When features_len is a nice multiple of 8, we can store them as `u64`s like pretend SIMD
type PackedArray<const N: usize> = [f32; N];

impl<const N: usize> FromArray<f32> for PackedArray<N> {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, f32>) -> &'a mut Self {
		assert_eq!(array.len(), N);
		let mut result = [0.; N];
		if let Some(slice) = array.as_slice() {
			result.copy_from_slice(slice);
		} else {
			for (src, dst) in array.iter().copied().zip(&mut result) {
				//TODO: should we preserve the host order?
				*dst = src;
			}
		}
		dst.write(result)
	}

	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, slice: &[f32]) -> &'a mut Self {
		assert_eq!(slice.len(), 8 * N);
		let result = slice.try_into()
			.unwrap();

		dst.write(result)
	}
	
	fn as_slice<'a>(&'a self) -> Cow<'a, [f32]> {
		Cow::Borrowed(self)
	}
}


impl<const N: usize> FeatureDistance for PackedArray<N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		l2_array::<N>(self, other)
	}
}

impl FeatureDistance for [f32] {
	type Metric = L2;
	type Distance = f32;

	fn distance(&self, other: &Self) -> Self::Distance {
		l2_slice(self, other)
	}
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
impl FeatureDistance for [float32x4_t] {
	type Metric = L2;
	type Distance = f32;

	fn distance(&self, other: &Self) -> Self::Distance {
		debug_assert!(is_aarch64_feature_detected!("neon"));
		unsafe {
			l2_neon_slice(self, other)
		}
	}
}


#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct TransmuteArray<E: Sized, const N: usize>([E; N]);

impl<E: Sized + Copy, const N: usize> FromArray<f32> for TransmuteArray<E, N> {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, f32>) -> &'a mut Self {
		if let Some(slice) = array.as_slice() {
			Self::from_slice(dst, slice)
		} else {
			assert!(size_of::<E>().is_multiple_of(size_of::<f32>()));
			let feature_len = N * size_of::<E>() / size_of::<f32>();
			assert_ne!(size_of::<E>(), 0, "Can't use ZSTs");
			assert_ne!(N, 0, "Empty feature");

			let dst_u8 = {
				// I can't imagine a platform where this isn't true
				assert!(align_of::<E>() >= align_of::<f32>());
				// Check that our array is packed (we could write code to deal with this, but I don't think we need to)
				assert!(is_slice_packed::<E>(), "Array not packed");
				dst.as_bytes_mut()
			};
			
			// Slow path: we have to copy from a non-contiguous view
			println!("Warn: transmute slow");
			assert_ne!(array.len(), feature_len, "Invalid feature size (actual: {}, expected: {feature_len})", array.len());
			//TODO: are there any meaningful optimizations we can do here?
			// for (src, dst) in array.iter().zip(&mut dst_u8[..F]) {
			// 	dst.write(*src);
			// }

			// unsafe { dst.assume_init_mut() }
			todo!("{} from_array", any::type_name::<Self>())
		}
	}

	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, slice: &[f32]) -> &'a mut Self {
		todo!("{} from_slice", any::type_name::<Self>())
		/*assert_ne!(size_of::<E>(), 0, "Can't use ZSTs");
		
		let dst_u8 = {
			// I can't imagine a platform where this isn't true
			assert!(align_of::<E>() >= align_of::<u8>());
			// Check that our array is packed (we could write code to deal with this, but I don't think we need to)
			assert!(align_of::<E>() <= size_of::<E>(), "Array not packed");
			dst.as_bytes_mut()
		};

		assert_eq!(slice.len(), F, "Invalid feature size (actual: {}, expected: {F}) for {}", slice.len(), std::any::type_name::<Self>());

		if N * size_of::<E>() == F {
			// Fast path: we just do a memcpy
			MaybeUninit::copy_from_slice(dst_u8, slice);
		} else {
			// Pad with zeros to ensure rust's safety guarantees
			// Fast-enough path: we do a memcpy + memset
			MaybeUninit::copy_from_slice(&mut dst_u8[..F], slice);
			MaybeUninit::fill(&mut dst_u8[F..], 0);
		}
		unsafe { dst.assume_init_mut() }*/
	}

	fn as_slice<'a>(&'a self) -> Cow<'a, [f32]> {
		if is_slice_packed::<E>() && is_slice_packed::<[E; N]>() {
			// No padding, transmute is safe
			let (pfx, bytes, sfx) = unsafe { self.0.align_to::<u8>() };
			assert!(pfx.is_empty());
			assert!(sfx.is_empty());

			// Cow::Borrowed(bytes)
		} else {
		}
		todo!("{} as_slice", any::type_name::<Self>())
	}
}

impl<E: AccumulateL2 + Sized + Copy, const N: usize> FeatureDistance for TransmuteArray<E, N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		//TODO: include detection on AccumulateL1Unsafe?
		let mut acc = E::init();
		for i in 0..N {
			acc = E::update(self.0[i], other.0[i], acc);
		}
		E::finish(acc)
	}
}


#[cfg(any(target_arch="x86_64", target_arch="x86"))]
impl<const N: usize> FeatureDistance for TransmuteArray<__m128, N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		//TODO: include detection on AccumulateL1Unsafe?
		debug_assert!(is_x86_feature_detected!("sse"));
		unsafe {
			l2_sse_array::<N>(&self.0, &other.0)
		}
	}
}

#[cfg(any(target_arch="x86_64", target_arch="x86"))]
impl<const N: usize> FeatureDistance for TransmuteArray<__m256, N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		//TODO: include detection on AccumulateL1Unsafe?
		debug_assert!(is_x86_feature_detected!("sse"));
		unsafe {
			l2_avx_array::<N>(&self.0, &other.0)
		}
	}
}

#[cfg(any(target_arch="x86_64", target_arch="x86"))]
impl<const N: usize> FeatureDistance for TransmuteArray<__m512, N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		//TODO: include detection on AccumulateL1Unsafe?
		debug_assert!(is_x86_feature_detected!("sse"));
		unsafe {
			l2_avx512_array::<N>(&self.0, &other.0)
		}
	}
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
impl<const N: usize> FeatureDistance for TransmuteArray<float32x4_t, N> {
	type Metric = L2;
	type Distance = f32;
	fn distance(&self, other: &Self) -> f32 {
		//TODO: include detection on AccumulateL1Unsafe?
		debug_assert!(is_aarch64_feature_detected!("neon"));
		unsafe {
			l2_neon_array::<N>(&self.0, &other.0)
		}
	}
}



pub(crate) enum QueryF32<'a> {
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon64(AlignQuery<'a, TransmuteArray<float32x4_t, 16>>),
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Sse64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m128, 16>>),
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m256, 8>>),
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx512_64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m512, 4>>),
	Array64(AlignQuery<'a, PackedArray<64>>),
	Generic(AlignQuery<'a, [f32], Vec<f32>>),
}

pub(crate) enum FeaturesF32 {
	// Specialize [f32; 64] because of SURF
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon64(Vec<TransmuteArray<float32x4_t, 16>>),
	/// `[f32; 64]` => `[__m128; 16]` (requires SSE2)
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Sse64(Vec<TransmuteArray<std::arch::x86_64::__m128, 16>>),
	/// `[f32; 64]` => `[__m256; 8]` (requires AVX)
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx64(Vec<TransmuteArray<std::arch::x86_64::__m256, 8>>),
	/// `[f32; 64]` => `[__m512; 4]` (requires AVX512F)
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx512_64(Vec<TransmuteArray<std::arch::x86_64::__m512, 4>>),
	/// SURF `[f32; 64]`
	Array64(Vec<PackedArray<64>>),
	//TODO: specific generics
	Generic {
		feature_len: usize,
		/// Invariant: data is always a multiple of feature_len
		data: Vec<f32>,
	},
}

impl<'a> DistanceQuery for QueryF32<'a> {
	fn min_index(&self, offset: usize, len: usize) -> usize {
		match self {
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(q) => q.min_index(offset, len),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(q) => q.min_index(offset, len),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(q) => q.min_index(offset, len),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(q) => q.min_index(offset, len),
			Self::Array64(q) => q.min_index(offset, len),
			Self::Generic(q) => q.min_index(offset, len),
		}
	}
}

impl FeaturesF32 {
	pub(super) fn len(&self) -> usize {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(vec) => vec.len(),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(vec) => vec.len(),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(vec) => vec.len(),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(vec) => vec.len(),
			Self::Array64(vec) => vec.len(),
			Self::Generic { data, .. } => data.len(),
		}
	}

	pub(super) fn storage(&self) -> &'static str {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(..) => "neon_64",
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(..) => "sse_64",
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(..) => "avx_64",
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(..) => "avx512_64",
			Self::Array64(..) => "array_64",
			Self::Generic { .. } => "generic",
		}
	}

	pub(super) fn feature_len(&self) -> usize {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(..) => 64,
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(..) => 64,
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(..) => 64,
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(..) => 64,
			Self::Array64(..) => 64,
			Self::Generic { feature_len, .. } => *feature_len,
		}
	}
}

impl Serialize for FeaturesF32 {
	fn write_to(&self, mut dst: impl std::io::Write) -> std::io::Result<()> {
		write_u32ish(self.len(), &mut dst)?;
		write_u32ish(self.feature_len(), &mut dst)?;

		fn write_features<'a, T: ToArray<f32> + ?Sized + 'a>(features: impl IntoIterator<Item = &'a T>, mut dst: impl std::io::Write) -> std::io::Result<()> {
			//TODO: write vectored?
			for feature in features {
				let slice = T::as_slice(feature);
				for item in slice.as_ref() {
					dst.write_all(&item.to_le_bytes())?;
				}
			}
			Ok(())
		}

		match self {
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(vec) => write_features(vec, dst),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(vec) => write_features(vec, dst),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(vec) => write_features(vec, dst),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(vec) => write_features(vec, dst),
			Self::Array64(vec) => write_features(vec, dst),
			Self::Generic { feature_len, data } => {
				debug_assert!(data.len().is_multiple_of(*feature_len));
				write_features(data.chunks(*feature_len), dst)
			}
		}
	}
}

impl Deserialize for FeaturesF32 {
	fn read_from(mut src: impl std::io::Read) -> std::io::Result<Self> {
		let len = read_u32ish(&mut src)?;
		let feature_len = read_u32ish(&mut src)?;

		let mut buf = vec![0u8; feature_len * size_of::<f32>()];
		let features = (0..len)
			.map::<std::io::Result<Vec<_>>, _>(|_| {
				src.read_exact(&mut buf)?;
				let value = buf.array_chunks::<{size_of::<f32>()}>()
					.map(|c| f32::from_le_bytes(*c))
					.collect::<Vec<f32>>();
				Ok(value)
			})
			.collect::<Result<Vec<_>, _>>()?;
		drop(buf);

		let mut result = FeaturesF32::new(len, feature_len);
		result.insert(
			features
				.iter()
				.map(ArrayView1::from)
		);

		Ok(result)
	}
}

impl super::Features<f32> for FeaturesF32 {
	type Query<'a> = QueryF32<'a>;
	fn new(capacity: usize, feature_len: usize) -> Self {
		assert_ne!(feature_len, 0, "Empty features");
		match feature_len {
			// SURF features are all 64-wide, so we specialize for that
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			64 if is_x86_feature_detected!("avx512f") => Self::Avx512_64(Vec::with_capacity(capacity)),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			64 if is_x86_feature_detected!("avx") => Self::Avx64(Vec::with_capacity(capacity)),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			64 if is_x86_feature_detected!("sse2") => Self::Sse64(Vec::with_capacity(capacity)),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			64 if is_aarch64_feature_detected!("neon") => Self::Neon64(Vec::with_capacity(capacity)),
			//TODO: do we prefer aligned slices to this?
			64 => Self::Array64(Vec::with_capacity(capacity)),

			// // Generic aligned
			// _ if is_aarch64_feature_detected!("neon") && feature_len.is_multiple_of(size_of::<float32x4_t>()) => Self::Neon(Vec::with_capacity(capacity * feature_len)),
			// Generic unaligned
			_ => Self::Generic { feature_len, data: Vec::with_capacity(capacity * feature_len) },
		}
	}
	
	fn insert<'a>(&mut self, features: impl ExactSizeIterator<Item = ArrayView1<'a, f32>>) {
		fn insert_array<'a, E: FromArray<f32>>(vec: &mut Vec<E>, features: impl ExactSizeIterator<Item = ArrayView1<'a, f32>>) {
			vec.reserve(features.len());
			
			//TODO: put in place
			for feature in features {
				// assert_eq!(feature.len(), F, "Invalid feature length");
				let mut value = MaybeUninit::zeroed();
				FromArray::from_array(&mut value, feature);
				vec.push(unsafe { value.assume_init() });
			}
		}

		match self {
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(vec) => insert_array(vec, features),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(vec) => insert_array(vec, features),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(vec) => insert_array(vec, features),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(vec) => insert_array(vec, features),
			Self::Array64(vec) => insert_array(vec, features),
			Self::Generic { feature_len, data } => {
				let feature_len = *feature_len;
				data.reserve(feature_len * features.len());
				for feature in features {
					match feature.as_slice() {
						Some(slice) => {
							assert_eq!(slice.len(), feature_len, "Invalid feature length");
							data.extend_from_slice(slice);
						},
						None => {
							let vec = feature.to_vec();
							assert_eq!(vec.len(), feature_len, "Invalid feature length");
							data.extend_from_slice(&vec);
						}
					}
				}
			},
		}
	}

	fn query<'a>(&'a self, value: ArrayView1<'a, f32>) -> Self::Query<'a> {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(vec) => QueryF32::Neon64(AlignQuery::new::<f32>(vec, value)),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(vec) => QueryF32::Avx512_64(AlignQuery::new(vec, value)),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(vec) => QueryF32::Sse64(AlignQuery::new(vec, value)),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(vec) => QueryF32::Avx64(AlignQuery::new(vec, value)),
			Self::Array64(vec) => QueryF32::Array64(AlignQuery::new(vec, value)),
			Self::Generic { .. } => {
				todo!("query generic f32")
				// QueryF32::Generic(AlignQuery::new_f32(vec, value)),
			}
		}
	}
}

impl From<FeaturesF32> for FeaturesGeneric {
	fn from(value: FeaturesF32) -> Self {
		Self::Float32(value)
	}
}


impl super::FeatureType for f32 {
	type FeaturesSpec = FeaturesF32;
	
	fn extract<'a>(generic: &'a FeaturesGeneric) -> Option<&'a Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Float32(r) => Some(r),
			_ => None,
		}
	}
	
	fn extract_mut<'a>(generic: &'a mut FeaturesGeneric) -> Option<&'a mut Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Float32(r) => Some(r),
			_ => None,
		}
	}
}