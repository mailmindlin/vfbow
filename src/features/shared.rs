use std::{borrow::{Borrow, Cow}, cmp::Ordering, mem::MaybeUninit};

use ndarray::ArrayView1;

use super::DistanceQuery;

/// Metric type
trait Metric {}

pub(super) struct L1;
impl Metric for L1 {}
pub(super) struct L2;
impl Metric for L2 {}

pub(super) trait FeatureDistance {
	type Metric: Metric;
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

// Copy bound is so we know there's a trivial destructor
pub(super) trait FromArray<E>: Sized + Copy {
	fn from_array<'a>(dst: &'a mut MaybeUninit<Self>, array: ArrayView1<'_, E>) -> &'a mut Self;
	fn from_slice<'a>(dst: &'a mut MaybeUninit<Self>, array: &[E]) -> &'a mut Self;
	fn as_slice<'a>(&'a self) -> Cow<'a, [E]> where [E]: ToOwned;
}

pub(crate) struct AlignQuery<'a, E: ToOwned + ?Sized, F = E> {
	pub(super) features: &'a [F],
	pub(super) value: Cow<'a, E>,
}

/// Convert ArrayView1 to Cow
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
	let mut dst = MaybeUninit::zeroed();
	let dst_mut = &mut dst;
	{
		let dst_ptr = std::ptr::from_ref(dst_mut);
		let res = E::from_array(&mut dst, array);
		// Double check that we returned the same pointer
		debug_assert!(std::ptr::addr_eq(dst_ptr, res))
	}
	Cow::Owned(unsafe { dst.assume_init() })
}

#[allow(private_bounds)]
impl<'a, E: FeatureDistance + ToOwned> AlignQuery<'a, E, E> {
	pub(super) fn new<T>(features: &'a [E], array: ArrayView1<'a, T>) -> Self where E: FromArray<T> {
		Self {
			features,
			value: array_to_cow(array),
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



pub(crate) struct SliceQuery<'a, E: ToOwned + ?Sized, F = E> {
	pub(super) features: &'a [F],
	pub(super) value: Cow<'a, E>,
}

pub(super) trait DistanceOrd {
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
