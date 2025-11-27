//! U8 feature type and distance functions
//! 
//! Supported metrics:
//! - L1 (Hamming distance)
use std::{borrow::Cow, io::{IoSliceMut, Write}, mem::MaybeUninit};
#[cfg(target_arch="aarch64")]
use std::arch::aarch64::uint8x16_t;
#[cfg(target_arch="arm")]
use std::arch::arm::uint8x16_t;

use ndarray::{Array1, ArrayView1, CowArray, Dim, Ix, aview1};

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
use super::distance_l1;
use crate::{Deserialize, Serialize, features::shared::{SliceQuery, ToArray}, util::convert::convert_le};

use super::{distance_l1::AccumulateL1, shared::{is_slice_packed, AlignQuery, FeatureDistance, FromArray, ValidZeroBits, L1}, DistanceQuery, FeatureType, Features, FeaturesGeneric};

impl FeatureDistance for [u8] {
	type Distance = u32;
	type Metric = L1;
	fn distance(&self, other: &Self) -> Self::Distance {
		debug_assert_eq!(self.len(), other.len());

		let mut result = 0;
		for (a, b) in self.iter().copied().zip(other.iter().copied()) {
			result += (a ^ b).count_ones();
		}
		result
	}
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
impl FeatureDistance for [uint8x16_t] {
	type Metric = L1;
	type Distance = u32;

	fn distance(&self, other: &Self) -> Self::Distance {
		super::arch::debug_ensure_neon();
		unsafe {
			distance_l1::neon::slice(self, other)
		}
	}
}

/// When features_len is a nice multiple of 8, we can store them as `u64`s like pretend SIMD
type Packed8Array<const N: usize> = [u64; N];

impl<const N: usize> FromArray<u8> for Packed8Array<N> {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, u8>) -> &'a mut Self {
		assert_eq!(array.len(), 8 * N);

		let mut result = [0u64; N];
		if let Some(slice) = array.as_slice() {
			//TODO: should we preserve the host order?
			for (src, dst) in convert_le(slice).unwrap().zip(&mut result) {
				*dst = src;
			}
		} else {
			for (src, dst) in array.iter().copied().array_chunks::<{size_of::<u64>()}>().zip(&mut result) {
				//TODO: should we preserve the host order?
				*dst = u64::from_le_bytes(src);
			}
		}
		dst.write(result)
	}

	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, slice: &[u8]) -> &'a mut Self {
		assert_eq!(slice.len(), 8 * N);
		let mut result = [0u64; N];
		//TODO: should we preserve the host order?
		for (src, dst) in convert_le(slice).unwrap().zip(&mut result) {
			*dst = src;
		}
		dst.write(result)
	}

	fn as_slice<'a>(&'a self) -> Cow<'a, [u8]> {
		if cfg!(target_endian="little") {
			// Pretty sure this is safe
			assert!(align_of::<u64>() <= size_of::<u64>(), "[u64;N] not packed");
			let (pfx, v, sfx) = unsafe { self.align_to::<u8>() };
			assert!(pfx.is_empty());
			assert!(sfx.is_empty());

			Cow::Borrowed(v)
		} else {
			// Make a copy
			let mut result = Vec::with_capacity(N * 8);
			for v in self.iter() {
				result.extend_from_slice(&v.to_le_bytes());
			}
			Cow::Owned(result)
		}
	}
}

impl<const N: usize> FeatureDistance for Packed8Array<N> {
	type Metric = L1;
	type Distance = u32;
	fn distance(&self, other: &Self) -> u32 {
		let mut result = 0;
		for i in 0..N {
			result += (self[i] ^ other[i]).count_ones();
		}
		result
	}
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct TransmuteArray<E: Sized, const N: usize, const F: usize>([E; N]);

impl<E: Sized + ValidZeroBits + Copy, const N: usize, const F: usize> FromArray<u8> for TransmuteArray<E, N, F> {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, u8>) -> &'a mut Self {
		if let Some(slice) = array.as_slice() {
			Self::from_slice(dst, slice)
		} else {
			assert_ne!(size_of::<E>(), 0, "Can't use ZSTs");
			assert_ne!(F, 0, "Empty feature");
			debug_assert!(F <= N * size_of::<E>(), "Feature size too big");

			let dst_u8 = {
				// I can't imagine a platform where this isn't true
				assert!(align_of::<E>() >= align_of::<u8>());
				// Check that our array is packed (we could write code to deal with this, but I don't think we need to)
				assert!(align_of::<E>() <= size_of::<E>(), "Array not packed");
				dst.as_bytes_mut()
			};
			
			// Slow path: we have to copy from a non-contiguous view
			println!("Warn: transmute slow");
			assert_ne!(array.len(), F, "Invalid feature size (actual: {}, expected: {F})", array.len());
			//TODO: are there any meaningful optimizations we can do here?
			for (src, dst) in array.iter().zip(&mut dst_u8[..F]) {
				dst.write(*src);
			}

			if N * size_of::<E>() != F {
				// Pad with zeros to ensure rust's safety guarantees
				dst_u8[F..].write_filled(0);
			}
			unsafe { dst.assume_init_mut() }
		}
	}

	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, slice: &[u8]) -> &'a mut Self {
		assert_ne!(size_of::<E>(), 0, "Can't use ZSTs");
		assert_ne!(F, 0, "Empty feature");
		debug_assert!(F <= N * size_of::<E>(), "Feature size too big");
		
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
			dst_u8.write_copy_of_slice(slice);
		} else {
			// Pad with zeros to ensure rust's safety guarantees
			// Fast-enough path: we do a memcpy + memset
			dst_u8[..F].write_copy_of_slice(slice);
			dst_u8[F..].write_filled(0);
		}
		unsafe { dst.assume_init_mut() }
	}

	fn as_slice<'a>(&'a self) -> Cow<'a, [u8]> {
		if N * size_of::<E>() == F && is_slice_packed::<E>() && is_slice_packed::<[E; N]>() {
			// No padding, transmute is safe
			let (pfx, bytes, sfx) = unsafe { self.0.align_to::<u8>() };
			assert!(pfx.is_empty());
			assert!(sfx.is_empty());

			Cow::Borrowed(bytes)
		} else {
			todo!("{} as_slice copy", std::any::type_name::<Self>())
		}
	}
}

impl<E: AccumulateL1 + Sized + ValidZeroBits + Copy, const N: usize, const F: usize> FeatureDistance for TransmuteArray<E, N, F> {
	type Metric = L1;
	type Distance = u32;

	fn distance(&self, other: &Self) -> u32 {
		//TODO: include detection on AccumulateL1Unsafe?
		let mut acc = E::init();
		for i in 0..N {
			acc = E::update(self.0[i], other.0[i], acc);
		}
		E::finish(acc)
	}
}

#[cfg(any(target_arch="aarch64", target_arch="arm"))]
impl<const N: usize, const F: usize> FeatureDistance for TransmuteArray<std::arch::aarch64::uint8x16_t, N, F> {
	type Metric = L1;
	type Distance = u32;
	fn distance(&self, other: &Self) -> u32 {
		super::arch::debug_ensure_neon();
		unsafe {
			distance_l1::neon::array::<N>(&self.0, &other.0)
		}
	}
}

type Padded8Array<const N: usize, const L: usize> = TransmuteArray<u64, N, L>;

/// Distance query
/// 
/// Each variant corresponds to [FeaturesU8]
#[allow(private_interfaces)]
pub(crate) enum QueryU8<'a> {
	/// [FeaturesU8::Neon32]
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon32(AlignQuery<'a, TransmuteArray<std::arch::aarch64::uint8x16_t, 2, 32>>),
	/// [FeaturesU8::Neon61]
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon61(AlignQuery<'a, TransmuteArray<std::arch::aarch64::uint8x16_t, 4, 61>>),
	/// [FeaturesU8::Array32]
	Array32(AlignQuery<'a, Packed8Array<4>>),
	/// [FeaturesU8::Array64]
	Array64(AlignQuery<'a, Packed8Array<8>>),
	/// [FeaturesU8::Array61]
	Array61(AlignQuery<'a, Padded8Array<8, 61>>),
	/// [FeaturesU8::Generic]
	Generic(SliceQuery<'a, u8>),
}

impl<'a> DistanceQuery for QueryU8<'a> {
	fn min_index(&self, offset: usize, len: usize) -> usize {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			QueryU8::Neon32(q) => q.min_index(offset, len),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			QueryU8::Neon61(q) => q.min_index(offset, len),
			QueryU8::Array32(q) => q.min_index(offset, len),
			QueryU8::Array64(q) => q.min_index(offset, len),
			QueryU8::Array61(q) => q.min_index(offset, len),
			QueryU8::Generic(q) => q.min_index(offset, len),
		}
	}
}

/// Storage for [f32] features
/// 
/// Contains specializations for common feature sizes and CPU features
#[allow(private_interfaces)]
pub(crate) enum FeaturesU8 {
	/// ARM NEON `[uint8x16; 4]` for ORB
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon32(Vec<TransmuteArray<std::arch::aarch64::uint8x16_t, 2, 32>>),
	/// ARM NEON `[uint8x16; 4]` for ORB
	#[cfg(any(target_arch="aarch64", target_arch="arm"))]
	Neon61(Vec<TransmuteArray<std::arch::aarch64::uint8x16_t, 4, 61>>),
	// === Platform-agnostic ===
	/// Store as `[u64; 4]`, for ORB
	Array32(Vec<Packed8Array<4>>),
	/// Store as `[u64; 8]`
	Array64(Vec<Packed8Array<8>>),
	/// Store as `[u64; 8]` with padding, for AKAZE
	Array61(Vec<Padded8Array<8, 61>>),
	/// Generic storage (not specialized)
	//TODO: Generic64
	Generic {
		feature_len: usize,
		// Invariant: data.len() is always a multiple of feature_len
		data: Vec<u8>,
	},
}

impl FeaturesU8 {
	/// Feature size
	pub(super) fn feature_len(&self) -> usize {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(..) => 32,
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(..) => 61,
			FeaturesU8::Array32(..) => 32,
			FeaturesU8::Array61(..) => 61,
			FeaturesU8::Array64(..) => 64,
			FeaturesU8::Generic { feature_len, .. } => *feature_len,
		}
	}

	/// Get a string representing the kernel used for storage/computation (mostly to check if a kernel is being used)
	pub(super) fn storage(&self) -> &'static str {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(..) => "neon16_32",
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(..) => "neon16_61",
			FeaturesU8::Array32(..) => "array8_32",
			FeaturesU8::Array61(..) => "array8_61",
			FeaturesU8::Array64(..) => "array8_64",
			FeaturesU8::Generic { .. } => "generic",
		}
	}

	/// Number of features
	pub(super) fn len(&self) -> usize {
		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(v) => v.len(),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(v) => v.len(),
			FeaturesU8::Array32(v) => v.len(),
			FeaturesU8::Array61(v) => v.len(),
			FeaturesU8::Array64(v) => v.len(),
			FeaturesU8::Generic { data, ..} => data.len(),
		}
	}

	/// Convert to Python ndarray
	#[cfg(feature="python")]
	pub(super) fn to_ndarray<'a>(&self, py: pyo3::Python<'a>) -> pyo3::PyResult<pyo3::Bound<'a, numpy::PyArray2<u8>>> {
		todo!("Convert FeaturesU8 to numpy array")
		/*match self {
			// Self::Generic { data, .. } => {
			// 	let r = numpy::PyArray2::from_vec2_bound(py, data)?;
			// 	Ok(r)
			// },
			_ => {
				//TODO
				Ok(numpy::PyArray2::zeros(py, (0,0), false))
			}
		}*/
	}
}

impl Serialize for FeaturesU8 {
	fn write_to(&self, mut dst: impl std::io::Write) -> std::io::Result<()> {
		use crate::util::serde::*;
		write_u32ish(self.feature_len(), &mut dst)?;
		write_u32ish(self.len(), &mut dst)?;

		fn write_features<'a, T: ToArray<u8> + ?Sized + 'a>(features: impl IntoIterator<Item = &'a T>, mut dst: impl Write) -> std::io::Result<()> {
			//TODO: write vectored?
			for feature in features {
				dst.write_all(&T::as_slice(feature))?;
			}
			Ok(())
		}

		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(vec) => write_features(vec, dst),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(vec) => write_features(vec, dst),
			FeaturesU8::Array32(vec) => write_features(vec, dst),
			FeaturesU8::Array64(vec) => write_features(vec, dst),
			FeaturesU8::Array61(vec) => write_features(vec, dst),
			FeaturesU8::Generic { feature_len, data } => {
				debug_assert!(data.len().is_multiple_of(*feature_len));
				write_features(data.chunks(*feature_len), dst)
			}
		}
	}
}

impl Deserialize for FeaturesU8 {
	fn read_from(mut src: impl std::io::Read) -> std::io::Result<Self> {
		use crate::util::serde::*;
		let feature_len = read_u32(&mut src)? as usize;
		let num_features = read_u32(&mut src)? as usize;
		let features = if src.is_read_vectored() {
			let mut features = Vec::with_capacity(num_features);
			features.extend((0..num_features).map(|_| vec![0u8; feature_len]));
			let mut feature_bufs = features
				.iter_mut()
				.map(|feat| IoSliceMut::new(feat))
				.collect::<Vec<_>>();
			//TODO: do we have to call this repeatedly?
			let n = src.read_vectored(&mut feature_bufs)?;
			assert_eq!(n, features.len() * feature_len);
			features
		} else {
			let mut features = Vec::with_capacity(num_features);
			for _ in 0..num_features {
				let mut feature = vec![0u8; feature_len];
				src.read_exact(&mut feature)?;
				features.push(feature);
			}
			features
		};

		//TODO: I think we can elmininate this copy
		let mut result = Self::new(num_features, feature_len);
		result.insert(features.iter().map(ArrayView1::from));
		Ok(result)
	}
}

impl Features<u8> for FeaturesU8 {
	type Query<'a> = QueryU8<'a> where Self: 'a;
	fn new(capacity: usize, feature_len: usize) -> Self {
		// println!("Init FeaturesU8 with {capacity} and {feature_len}");
		assert_ne!(feature_len, 0, "Zero-size feature");

		// Select storage
		match feature_len {
			64 => Self::Array64(Vec::with_capacity(capacity)),
			
			// Specialize for AKAZE
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			61 if cfg!(target_feature="neon") => Self::Neon61(Vec::with_capacity(capacity)),
			61 => Self::Array61(Vec::with_capacity(capacity)),
			
			// Specialize for ORB
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			32 if cfg!(target_feature="neon") => Self::Neon32(Vec::with_capacity(capacity)),
			32 => Self::Array32(Vec::with_capacity(capacity)),
			_ => {
				// Fallback
				Self::Generic {
					feature_len,
					data: Vec::with_capacity(capacity),
				}
			}
		}
	}
	fn insert<'a>(&mut self, features: impl ExactSizeIterator<Item = ArrayView1<'a, u8>>) {
		fn insert_array<'a, E: FromArray<u8>>(vec: &mut Vec<E>, features: impl ExactSizeIterator<Item = ArrayView1<'a, u8>>) {
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
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(vec) => insert_array(vec, features),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(vec) => insert_array(vec, features),
			FeaturesU8::Array32(vec) => insert_array(vec, features),
			FeaturesU8::Array64(vec) => insert_array(vec, features),
			FeaturesU8::Array61(vec) => insert_array(vec, features),
			FeaturesU8::Generic { feature_len, data } => {
				let feature_len = *feature_len;
				data.reserve(features.len() * feature_len);
				for feature in features {
					match feature.as_slice() {
						Some(slice) => {
							assert_eq!(slice.len(), feature_len, "Invalid feature length");
							data.extend_from_slice(slice);
						},
						None => {
							// Maybe we can just copy from the iterator, but we could violate our invariant on a iterator panic
							let feature = feature.to_vec();
							assert_eq!(feature.len(), feature_len, "Invalid feature length");
							data.extend_from_slice(&feature);
						}
					}
				}
			},
		}
	}

	fn query<'a>(&'a self, value: ArrayView1<'a, u8>) -> Self::Query<'a> {
		assert_eq!(value.len(), self.feature_len(), "Invalid feature length");

		match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(features) => QueryU8::Neon32(AlignQuery::new(features, value)),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(features) => QueryU8::Neon61(AlignQuery::new(features, value)),
			FeaturesU8::Array32(features) => QueryU8::Array32(AlignQuery::new(features, value)),
			FeaturesU8::Array64(features) => QueryU8::Array64(AlignQuery::new(features, value)),
			FeaturesU8::Array61(features) => QueryU8::Array61(AlignQuery::new(features, value)),
			FeaturesU8::Generic { feature_len, data } => {
				assert_eq!(*feature_len, value.len(), "Value length mismatch");
				QueryU8::Generic(SliceQuery::new(data, *feature_len, value))
			},
		}
	}

	fn get<'a>(&'a self, index: usize) -> Option<CowArray<'a, u8, Dim<[Ix; 1]>>> {
		let slice = match self {
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon32(vec) => ToArray::as_slice(vec.get(index)?),
			#[cfg(any(target_arch="aarch64", target_arch="arm"))]
			FeaturesU8::Neon61(vec) => ToArray::as_slice(vec.get(index)?),
			FeaturesU8::Array32(vec) => ToArray::as_slice(vec.get(index)?),
			FeaturesU8::Array64(vec) => ToArray::as_slice(vec.get(index)?),
			FeaturesU8::Array61(vec) => ToArray::as_slice(vec.get(index)?),
			FeaturesU8::Generic { feature_len, data } => {
				let chunk = data.chunks_exact(*feature_len)
					.nth(index)?;
				ToArray::as_slice(chunk)
			},
		};
		Some(match slice {
			Cow::Borrowed(v) => aview1(v).into(),
			Cow::Owned(v) => Array1::from_vec(v).into(),
		})
	}
}

impl From<FeaturesU8> for FeaturesGeneric {
	fn from(value: FeaturesU8) -> Self {
		Self::Uint8(value)
	}
}

impl FeatureType for u8 {
	type FeaturesSpec = FeaturesU8;
	
	fn extract(generic: &FeaturesGeneric) -> Option<&Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Uint8(r) => Some(r),
			_ => None,
		}
	}
	
	fn extract_mut(generic: &mut FeaturesGeneric) -> Option<&mut Self::FeaturesSpec> {
		match generic {
			FeaturesGeneric::Uint8(r) => Some(r),
			_ => None,
		}
	}
}