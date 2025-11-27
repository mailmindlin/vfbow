//! F32 feature type and distance functions
//! 
//! Supported metrics:
//! - L2 (Euclidean distance)

#[cfg(target_arch="aarch64")]
use std::arch::{is_aarch64_feature_detected, aarch64::float32x4_t};
#[cfg(target_arch="arm")]
use std::arch::{is_arm_feature_detected, arm::float32x4_t};
#[cfg(target_arch="x86_64")]
use std::arch::{is_x86_feature_detected, x86_64::{__m128, __m256, __m512}};
#[cfg(target_arch="x86")]
use std::arch::{is_x86_feature_detected, x86::{__m128, __m256, __m512}};
use std::{borrow::Cow, mem::{MaybeUninit, offset_of}};

use ndarray::{ArrayView1, Dim, Ix};

#[cfg(any(target_arch="x86_64", target_arch="x86"))]
use crate::features::distance_l2::{l2_avx512_array, l2_avx_array, l2_sse_array};
use crate::{features::{Features, shared::ToArray, util::{assert_alignment_geq, assert_array_packed, assert_not_zst, assert_size_multiple}}, util::convert::convert_le};
use crate::util::serde::{read_u32ish, write_u32ish};
#[cfg(any(target_arch="aarch64", target_arch="arm"))]
use super::distance_l2::{self, AccumulateL2};
use super::shared::SliceQuery;
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
		distance_l2::generic::array::<N>(self, other)
	}
}

impl FeatureDistance for [f32] {
	type Metric = L2;
	type Distance = f32;

	fn distance(&self, other: &Self) -> Self::Distance {
		distance_l2::generic::slice(self, other)
	}
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
impl FeatureDistance for [float32x4_t] {
	type Metric = L2;
	type Distance = f32;

	fn distance(&self, other: &Self) -> Self::Distance {
		super::arch::debug_ensure_neon();

		unsafe { distance_l2::neon::slice(self, other) }
	}
}

/// Assert that some type is actually packed [f32]s, and can be transmuted to/from `[f32; size_of<Self>() / size_of<f32>()]` correctly
/// 
/// # Safety
/// Given the number of elements `N = size_of::<Self>() / size_of::<f32>()`
/// 
/// This type MUST have the properties:
/// - All references `&Self` may transmuted to `&[f32; N]`
/// - All references `&[Self; X]` may be transmuted to `&[f32; X * N]`
/// - The above rules also apply to mutable references and [MaybeUninit] references.
/// - When transmuting `&mut MaybeUninit<Self>` => `&mut [MaybeUninit<f32>; X]` and writing to every element of the second array, the first reference may be considered initialized.
/// 
/// These can only hold for types that are equivalend in memory layout to `[f32; N]`.
unsafe trait TransmutePackedF32: Sized {}

/// Safety: float32x4 is a SIMD type like `[f32; 4]`
#[cfg(any(target_arch="aarch64", target_arch="arm"))]
unsafe impl TransmutePackedF32 for float32x4_t {}


/// Helper to store `[f32; X]` transmuted into `[E; N]`. Used for when we pack an array of [f32]s into a SIMD type
/// 
/// Invariant: `E` must be a size multiple of [f32], and have a greater or equal alignment.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct TransmuteArray<E: Sized, const N: usize>([E; N]);

impl<E: Sized + TransmutePackedF32, const N: usize> TransmuteArray<E, N> {
	/*const fn inner_ptr<'a>(this: &'a mut MaybeUninit<Self>) -> &'a mut [MaybeUninit<E>; N] {
		// Assert that the memory layout of Self is equivalent to [E; N]
		// I think this is guaranteed by `#[repr(transparent)]`
		const {
			let () = assert!(offset_of!(Self, 0) == 0);
			let () = assert!(size_of::<Self>() == size_of::<[E; N]>());
			let () = assert!(N != 0);
		}

		unsafe {
			this.as_mut_ptr()
				.cast::<[MaybeUninit<E>; N]>()
				.as_mut().unwrap()
		}
	}*/
	/// Transmute `MaybeUninit<Self>` to slice of uninitialized f32s safely
	/// 
	/// This can be done safely because of the requirements of [TransmutePackedF32]
	const fn element_ptr(this: &mut MaybeUninit<Self>) -> &mut [MaybeUninit<f32>] {
		const {
			// Assert that the memory layout of Self is equivalent to [E; N]
			// I think this is guaranteed by `#[repr(transparent)]`
			let () = assert!(offset_of!(Self, 0) == 0);
			let () = assert!(size_of::<Self>() == size_of::<[E; N]>());
			let () = assert!(N != 0);
			// Now prove that we can transmute to f32
			assert_array_packed::<E>();
			assert_alignment_geq::<E, f32>();
			assert_not_zst::<E>();
			assert_size_multiple::<E, f32>();
		}

		unsafe {
			let ptr = this.as_mut_ptr().cast();
			core::slice::from_raw_parts_mut(ptr, N * size_of::<E>() / size_of::<f32>())
		}
	}
}

impl<E: Sized + Copy + TransmutePackedF32, const N: usize> FromArray<f32> for TransmuteArray<E, N> {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, f32>) -> &'a mut Self {
		if let Some(slice) = array.as_slice() {
			Self::from_slice(dst, slice)
		} else {
			assert!(size_of::<E>().is_multiple_of(size_of::<f32>()));
			assert_ne!(size_of::<E>(), 0, "Can't use ZSTs");
			assert_ne!(N, 0, "Empty feature");

			let dst_slice = Self::element_ptr(dst);
			
			// Slow path: we have to copy from a non-contiguous view
			println!("Warn: transmute slow");
			assert_ne!(array.len(), dst_slice.len(), "Invalid feature size (actual: {}, expected: {})", array.len(), dst_slice.len());
			//TODO: are there any meaningful optimizations we can do here?
			let mut dst_iter = dst_slice.iter_mut();
			for (src, dst) in array.iter().zip(dst_iter.by_ref()) {
				dst.write(*src);
			}
			assert!(dst_iter.next().is_none(), "Not all elements were written to");

			// Safety: we wrote to every element of `dst`
			unsafe { dst.assume_init_mut() }
		}
	}

	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, slice: &[f32]) -> &'a mut Self {
		const {
			assert_not_zst::<E>();
			// I can't imagine a platform where this isn't true
			assert_alignment_geq::<E, f32>();
			// Check that our array is packed (we could write code to deal with this, but I don't think we need to)
			assert_array_packed::<E>();
		};

		let expected_f32s = N * size_of::<E>() / size_of::<f32>();
		assert_eq!(slice.len(), expected_f32s, "Invalid feature size (actual: {}, expected: {expected_f32s}) for {}", slice.len(), std::any::type_name::<Self>());

		// memcpy
		Self::element_ptr(dst).write_copy_of_slice(slice);
		unsafe { dst.assume_init_mut() }
	}

	fn as_slice<'a>(&'a self) -> Cow<'a, [f32]> {
		const {
			assert_array_packed::<E>();
			assert_array_packed::<[E; N]>();
		}
		// Safety: invariant of TransmutePackedF32
		let (pfx, result, sfx) = unsafe { self.0.align_to::<f32>() };
		assert!(pfx.is_empty());
		assert!(sfx.is_empty());

		Cow::Borrowed(result)
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
		super::arch::debug_ensure_neon();
		unsafe {
			distance_l2::neon::array::<N>(&self.0, &other.0)
		}
	}
}


/// Query for f32 features
/// 
/// Each variant corresponds to a variant in [FeaturesF32]
#[allow(private_interfaces)]
pub(crate) enum QueryF32<'a> {
	/// Corresponds to [FeaturesF32::Neon64]
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon64(AlignQuery<'a, TransmuteArray<float32x4_t, 16>>),
	/// Corresponds to [FeaturesF32::Sse64]
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Sse64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m128, 16>>),
	/// Corresponds to [FeaturesF32::Avx64]
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m256, 8>>),
	/// Corresponds to [FeaturesF32::Avx512_64]
	#[cfg(any(target_arch="x86_64", target_arch="x86"))]
	Avx512_64(AlignQuery<'a, TransmuteArray<std::arch::x86_64::__m512, 4>>),
	/// Corresponds to [FeaturesF32::Array64]
	Array64(AlignQuery<'a, PackedArray<64>>),
	/// Corresponds to [FeaturesF32::Generic]
	Generic(SliceQuery<'a, f32>),
}

/// Storage for f32 features
#[allow(private_interfaces)]
pub(crate) enum FeaturesF32 {
	// Specialize [f32; 64] because of SURF
	/// SURF `[f32; 64]` => `[float32x4_t; 16]` (requires NEON)
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
	/// Storage for feature lengths not specifically optimized
	//TODO: specific generics
	Generic {
		/// Feature length
		feature_len: usize,
		/// Actual feature data, stored as a flat array
		/// 
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
	/// Number of features stored
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

	/// Name describing storage type
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

	/// Feature length
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

	/// Convert to Python ndarray
	#[cfg(feature="python")]
	pub(super) fn to_ndarray<'a>(&self, py: pyo3::Python<'a>) -> pyo3::PyResult<pyo3::Bound<'a, numpy::PyArray2<f32>>> {
		todo!("Convert FeaturesF32 to numpy array")
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
				let value = convert_le(&buf)
					.unwrap()
					.collect::<Vec<_>>();
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
			Self::Generic { data, feature_len } => {
				assert_eq!(*feature_len, value.len(), "Feature length mismatch");
				QueryF32::Generic(SliceQuery::new(data, *feature_len, value))
			},
		}
	}

	fn get<'a>(&'a self, index: usize) -> Option<ndarray::CowArray<'a, f32, Dim<[Ix; 1]>>> {
		let slice = match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			Self::Neon64(vec) => ToArray::as_slice(vec.get(index)?),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx512_64(vec) => ToArray::as_slice(vec.get(index)?),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Sse64(vec) => ToArray::as_slice(vec.get(index)?),
			#[cfg(any(target_arch="x86_64", target_arch="x86"))]
			Self::Avx64(vec) => ToArray::as_slice(vec.get(index)?),
			Self::Array64(vec) => ToArray::as_slice(vec.get(index)?),
			Self::Generic { feature_len, data } => {
				let chunk = data.chunks_exact(*feature_len)
					.nth(index)?;
				ToArray::as_slice(chunk)
			},
		};
		Some(match slice {
			Cow::Borrowed(v) => ndarray::aview1(v).into(),
			Cow::Owned(v) => ndarray::Array1::from_vec(v).into(),
		})
	}
}

impl From<FeaturesF32> for FeaturesGeneric {
	fn from(value: FeaturesF32) -> Self {
		Self::Float32(value)
	}
}


impl super::FeatureType for f32 {
	type FeaturesSpec = FeaturesF32;
	
	fn extract(generic: &FeaturesGeneric) -> Option<&Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Float32(r) => Some(r),
			_ => None,
		}
	}
	
	fn extract_mut(generic: &mut FeaturesGeneric) -> Option<&mut Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Float32(r) => Some(r),
			_ => None,
		}
	}
}