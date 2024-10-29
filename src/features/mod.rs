use std::{any, fmt::Debug};

use ndarray::ArrayView1;

use crate::{util::DescriptorType, Deserialize, Serialize};

mod distance_l1;
mod distance_l2;
mod u8;
mod f32;
mod shared;

pub(crate) trait DistanceQuery {
	/// Compute the closest distance
	fn min_index(&self, offset: usize, len: usize) -> usize;
}

pub(crate) trait Features<E: 'static>: Into<FeaturesGeneric> + Serialize + Deserialize {
	type Query<'a>: DistanceQuery where Self: 'a;

	fn new(capacity: usize, feature_len: usize) -> Self;
	/// Insert features
	fn insert<'a>(&mut self, features: impl ExactSizeIterator<Item = ArrayView1<'a, E>>);
	fn query<'a>(&'a self, value: ArrayView1<'a, E>) -> Self::Query<'a>;
}



pub(crate) trait FeatureType: Sized + 'static {
	type FeaturesSpec: Features<Self>;
	fn extract<'a>(generic: &'a FeaturesGeneric) -> Option<&'a Self::FeaturesSpec>;
	fn extract_mut<'a>(generic: &'a mut FeaturesGeneric) -> Option<&'a mut Self::FeaturesSpec>;
}

pub(crate) type TypedFeatures<T> = <T as FeatureType>::FeaturesSpec;

pub(crate) enum FeaturesGeneric {
	Float32(f32::FeaturesF32),
	Uint8(u8::FeaturesU8),
}

impl Debug for FeaturesGeneric {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Float32(v) => f
				.debug_struct("Float32")
				.field("storage", &v.storage())
				.field("feature_len", &v.feature_len())
				.field("len", &v.len())
				.finish_non_exhaustive(),
			Self::Uint8(v) => f
				.debug_struct("Uint8")
				.field("storage", &v.storage())
				.field("feature_len", &v.feature_len())
				.field("len", &v.len())
				.finish_non_exhaustive(),
		}
	}
}

impl FeaturesGeneric {
	pub(crate) fn new(capacity: usize, dt: DescriptorType, feature_len: usize) -> Self {
		match dt {
			DescriptorType::Uint8 => Self::Uint8(u8::FeaturesU8::new(capacity, feature_len)),
			DescriptorType::Float32 => Self::Float32(f32::FeaturesF32::new(capacity, feature_len)),
		}
	}

	pub(crate) fn query<'a, T: FeatureType>(&'a self, value: ArrayView1<'a, T>) -> <T::FeaturesSpec as Features<T>>::Query<'a> {
		let inner = T::extract(self)
			.unwrap();//TODO: error type
		inner.query(value)
	}

	pub(crate) fn len(&self) -> usize {
		match self {
			FeaturesGeneric::Float32(feat) => feat.len(),
			FeaturesGeneric::Uint8(feat) => feat.len(),
		}
	}

	/// Insert features
	pub(crate) fn insert<'a, T: FeatureType>(&mut self, features: impl ExactSizeIterator<Item = ArrayView1<'a, T>>) where T: 'a {
		let Some(typed) = T::extract_mut(self) else {
			panic!("Inconsistent feature type: tried to insert {} into {self:?}", any::type_name::<T>());
		};
		typed.insert(features)
	}

	#[cfg(feature="python")]
	pub(crate) fn to_python<'a>(&self, py: pyo3::Python<'a>) -> pyo3::PyResult<pyo3::Bound<'a, pyo3::PyAny>> {
		match self {
			Self::Uint8(feat) => {
				let arr = feat.to_ndarray(py)?;
				Ok(arr.into_any())
			},
			Self::Float32(..) => {
				todo!("f32 to python")
			}
		}
	}
}

impl Serialize for FeaturesGeneric {
	fn write_to(&self, mut dst: impl std::io::Write) -> std::io::Result<()> {
		use crate::util::serde::*;
		match self {
			FeaturesGeneric::Float32(feat) => {
				write_u32(DescriptorType::Float32.into(), &mut dst)?;
				feat.write_to(dst)
			},
			FeaturesGeneric::Uint8(feat) => {
				write_u32(DescriptorType::Uint8.into(), &mut dst)?;
				feat.write_to(dst)
			},
		}
	}
}

impl Deserialize for FeaturesGeneric {
	fn read_from(mut src: impl std::io::Read) -> std::io::Result<Self> {
		use crate::util::serde::*;
		let dtype = read_u32(&mut src)?;
		let dtype = DescriptorType::try_from(dtype)
			.map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unknown dtype {dtype}")))?;
		match dtype {
			DescriptorType::Uint8 => u8::FeaturesU8::read_from(src).map(Self::Uint8),
			DescriptorType::Float32 => f32::FeaturesF32::read_from(src).map(Self::Float32),
		}
	}
}