//! Common traits shared in this module
use std::{borrow::{Borrow, Cow}, cmp::Ordering, mem::MaybeUninit};

use ndarray::ArrayView1;

use super::DistanceQuery;

/// Metric type
trait Metric {}

/// L1 metric (Hamming distance)
pub(super) struct L1;
impl Metric for L1 {}

/// L2 metric (Euclidean distance)
pub(super) struct L2;
impl Metric for L2 {}

/// DType that has a distance function
pub(super) trait FeatureDistance {
	/// Default metric (marker)
	#[allow(private_bounds)]
	type Metric: Metric;
	/// Distance type
	type Distance: DistanceOrd;
	/// We return u32 here because I assume the descriptors are ≤ 2^29 bytes long
	fn distance(&self, other: &Self) -> Self::Distance;
}


pub(super) trait ToArray<E> {
	fn as_slice<'a>(&'a self) -> Cow<'a, [E]> where [E]: ToOwned;
}

impl<E, T: FromArray<E>> ToArray<E> for T {
	fn as_slice<'a>(&'a self) -> Cow<'a, [E]> where [E]: ToOwned {
		FromArray::as_slice(self)
	}
}

impl<E> ToArray<E> for [E] {
	fn as_slice<'a>(&'a self) -> Cow<'a, [E]> where [E]: ToOwned {
		Cow::Borrowed(self)
	}
}

/// Helper construct self from arrays of elements
// Copy bound is so we know there's a trivial destructor
pub(super) trait FromArray<E>: Sized + Copy {
	/// Initialize `dst` with the values stored in `src`, returning a reference to the initialized value.
	/// 
	/// The returned reference MUST be the same as `dst`
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, src: ArrayView1<'_, E>) -> &'a mut Self;
	/// Initialize `dst` with the values stored in `src`, returning a reference to the initialized value.
	/// 
	/// The returned reference MUST be the same as `dst`
	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, src: &[E]) -> &'a mut Self;
	/// Get the stored values (try not to copy)
	fn as_slice<'a>(&'a self) -> Cow<'a, [E]> where [E]: ToOwned;
}

/// A [DistanceQuery] where 
pub(crate) struct AlignQuery<'a, E: ToOwned + ?Sized, F = E> {
	pub(super) features: &'a [F],
	pub(super) value: Cow<'a, E>,
}

/// Convert ArrayView1 to Cow, trying not to copy
fn array_to_cow<'a, E: FromArray<T>, T>(array: ArrayView1<'a, T>) -> Cow<'a, E> {
	// It would be really great if we didn't have to copy the array
	//TODO
	/*if let Some(slice) = array.to_slice() {
		// Check alignment
		if slice.as_ptr().is_aligned_to(align_of::<E>()) {
			assert_eq!(slice.len(), size_of::<E>());
			let (pfx, res, sfx) = unsafe { slice.align_to::<E>() };
			assert!(pfx.is_empty());
			assert!(sfx.is_empty());
			return Cow::Borrowed(res[0]);
		}
	}*/

	// Fallback: copy to vector
	Cow::Owned({
		let mut dst = MaybeUninit::zeroed();
		let dst_ptr = dst.as_ptr();
		let res = E::from_array(&mut dst, array);
		// Double check that we got the same pointer back
		debug_assert!(std::ptr::addr_eq(dst_ptr, res), "FromArray::from_array invariant violated");
		// Safety: we have an initialized reference to dst
		unsafe { dst.assume_init() }
	})
}
/// Convert ArrayView1 to Cow, trying not to copy
fn array_to_cow_simple<'a, E>(array: ArrayView1<'a, E>) -> Cow<'a, [E]> where [E]: ToOwned<Owned = Vec<E>>, E: Clone {
	// It would be really great if we didn't have to copy the array
	match array.to_slice() {
		Some(slice) => Cow::Borrowed(slice),
		None => Cow::Owned(array.to_vec()),
	}
}

#[allow(private_bounds)]
impl<'a, E: FeatureDistance + ToOwned> AlignQuery<'a, E, E> {
	/// Construct from value array
	pub(super) fn new<T>(features: &'a [E], value_array: ArrayView1<'a, T>) -> Self where E: FromArray<T> {
		Self {
			features,
			value: array_to_cow(value_array),
		}
	}
}

impl<'a, E: FeatureDistance + ToOwned + ?Sized, F: Borrow<E>> DistanceQuery for AlignQuery<'a, E, F> {
	fn min_index(&self, offset: usize, len: usize) -> usize {
		// println!("\tQuery {offset}..{} (+{len})", offset+len);
		let value: &E = &self.value;
		debug_assert!(offset.checked_add(len).expect("Index overflow") <= self.features.len(), "Index {offset}+{len}={} outside valid range 0..{}", offset+len, self.features.len());

		(0..len)
			.map(|idx| {
				let reference = self.features[offset + idx].borrow();
				let dist = value.distance(reference);
				(idx, dist)
			})
			.min_by(|(_, d1), (_, d2)| DistanceOrd::compare(d1, d2))
			.expect("Empty length")
			.0
	}
}


/// A [DistanceQuery] where the features are stored as a packed slice
pub(crate) struct SliceQuery<'a, E> where [E]: ToOwned {
	/// The value being queried
	value: Cow<'a, [E]>,

	/// Reference to the feature data
	features: &'a [E],
	feature_len: usize,
}

#[allow(private_bounds)]
impl<'a, E> SliceQuery<'a, E> where [E]: FeatureDistance + ToOwned<Owned = Vec<E>>, E: Clone {
	/// Constructor from value array
	pub(super) fn new(features: &'a [E], feature_len: usize, value_array: ArrayView1<'a, E>) -> Self {
		Self {
			features,
			feature_len,
			value: array_to_cow_simple(value_array),
		}
	}
}

impl<'a, E> DistanceQuery for SliceQuery<'a, E> where [E]: FeatureDistance + ToOwned {
	fn min_index(&self, offset: usize, len: usize) -> usize {
		// println!("\tQuery {offset}..{} (+{len})", offset+len);
		let value: &[E] = &self.value;
		debug_assert!(offset.checked_add(len).expect("Index overflow") < self.features.len(), "Index {offset}+{len}={} outside valid range 0..={}", offset+len, self.features.len());

		let chunks = self.features
			.chunks_exact(self.feature_len);
		debug_assert!(chunks.remainder().is_empty());
		chunks.skip(offset)
			.take(len)
			.map(|reference| value.distance(reference))
			.enumerate()
			.min_by(|(_, d1), (_, d2)| DistanceOrd::compare(d1, d2))
			.expect("Empty length")
			.0
	}
}

/// Like [Ord] but implemented for floats
pub(super) trait DistanceOrd {
	/// Compare two distances
	fn compare(&self, other: &Self) -> Ordering;
}

impl DistanceOrd for u32 {
	#[inline(always)]
	fn compare(&self, other: &Self) -> Ordering {
		Ord::cmp(self, other)
	}
}

impl DistanceOrd for f32 {
	fn compare(&self, other: &Self) -> Ordering {
		f32::total_cmp(self, other)
	}
}


/// Marker trait for types that all zeroes in memory is valid
/// 
/// # Safety
/// Implementing this trait for a type where all-zeroes is not a valid value
/// is undefined
// TODO: Maybe replace with bytemuck?
pub(super) unsafe trait ValidZeroBits {}
unsafe impl ValidZeroBits for u8 {}
unsafe impl ValidZeroBits for u32 {}
unsafe impl ValidZeroBits for u64 {}
#[cfg(any(target_arch="aarch64", target_arch="arm"))]
unsafe impl ValidZeroBits for std::arch::aarch64::uint8x16_t {}


/// Check if the slice `[E]` has any padding inserted
pub(super) const fn is_slice_packed<E>() -> bool {
	align_of::<E>() <= size_of::<E>() && size_of::<E>().is_multiple_of(align_of::<E>())
}
