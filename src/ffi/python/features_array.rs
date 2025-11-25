use std::borrow::Cow;

use ndarray::{Array2, Axis, CowArray};
use num_traits::Zero;
use numpy::{BorrowError, Ix2, PyArray, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{exceptions::{PyRuntimeError, PyTypeError, PyValueError}, inspect::types::{ModuleName, TypeInfo}, types::PyAnyMethods, Bound, FromPyObject, PyAny, PyErr, PyResult};

pub(super) enum CowArray2Any<'a> {
	U8(CowArray<'a, u8, Ix2>),
	F32(CowArray<'a, f32, Ix2>),
}

#[derive(Debug)]
pub(super) enum PyFeaturesLike<'py> {
	Empty,//TODO: get rid of this variant?
	U8(Vec<PyReadonlyArray2<'py, u8>>),
	F32(Vec<PyReadonlyArray2<'py, f32>>),
}

#[derive(Debug, thiserror::Error)]
enum ReadonlyArray2Error<'py> {
	#[error("Inconsistent feature dtypes")]
	InconsistentDtype,
	#[error("Invalid ndim (actual: {0}, expected: 2)")]
	InvalidNdims(usize),
	#[error("Unsupported dtype (actual: {0}, expected: uint8, float32)")]
	UnsuppportedDtype(Bound<'py, PyArrayDescr>),
	#[error("Unable to borrow array at index {index}: {error}")]
	Borrow {
		index: usize,
		error: BorrowError,
	},
	#[error(transparent)]
	Py(#[from] PyErr),
}
impl<'py> From<ReadonlyArray2Error<'py>> for PyErr {
	fn from(value: ReadonlyArray2Error<'py>) -> Self {
		if let ReadonlyArray2Error::Py(py) = value {
			return py;
		}

		let msg = format!("{value}");
		match value {
			ReadonlyArray2Error::Borrow { .. } => PyErr::new::<PyRuntimeError, _>(msg),
			ReadonlyArray2Error::UnsuppportedDtype(..) | ReadonlyArray2Error::InconsistentDtype => PyErr::new::<PyTypeError, _>(msg),
			_ => PyErr::new::<PyValueError, _>(msg),
		}
	}
}

impl<'py> PyFeaturesLike<'py> {
	pub(super) fn as_2d(&self) -> Option<CowArray2Any<'_>> {
		fn squish_seq<'a, E: numpy::Element + Clone + Zero>(items: &'a [PyReadonlyArray2<E>]) -> CowArray<'a, E, Ix2> {
			if items.len() == 1 {
				// One item => Cow::Borrowed
				items[0]
					.as_array()
					.into()
			} else {
				// Multiple items => copy
				let nrows = items.iter()
					.map(|item| item.shape()[0])
					.sum::<usize>();
				let ncols = items[0].shape()[1];
				let mut result = Array2::zeros(Ix2(nrows, ncols));
				let mut offset = 0;

				for item in items {
					let item = item.as_array();
					let mut dst = result.slice_axis_mut(Axis(0), (offset..offset+item.nrows()).into());
					dst.assign(&item);
					offset += item.nrows();
				}
				result.into()
			}
		}
		match self {
			PyFeaturesLike::Empty => None,
			PyFeaturesLike::U8(items) => Some(CowArray2Any::U8(squish_seq(items))),
			PyFeaturesLike::F32(items) => Some(CowArray2Any::F32(squish_seq(items))),
		}
	}

	fn insert_u8(&mut self, value: &Bound<'py, PyArray2<u8>>, index: usize) -> Result<(), ReadonlyArray2Error<'_>> {
		let value = py_read_array(value, index)?;
		match self {
			Self::Empty => {
				*self = Self::U8(vec![value]);
				Ok(())
			},
			Self::U8(me) => {
				me.push(value);
				Ok(())
			},
			_ => Err(ReadonlyArray2Error::InconsistentDtype),
		}
	}
	fn insert_f32(&mut self, value: &Bound<'py, PyArray2<f32>>, index: usize) -> Result<(), ReadonlyArray2Error<'_>> {
		let value = py_read_array(value, index)?;
		match self {
			Self::Empty => {
				*self = Self::F32(vec![value]);
				Ok(())
			},
			Self::F32(me) => {
				me.push(value);
				Ok(())
			},
			_ => Err(ReadonlyArray2Error::InconsistentDtype),
		}
	}

	fn insert_dyn(&mut self, feature: &Bound<'py, PyUntypedArray>, index: usize) -> Result<(), ReadonlyArray2Error<'_>> {
		let shape = feature.shape();
		// Skip empty arrays
		if shape.len() == 0 || shape.iter().any(|d| *d == 0) {
			// Skip empty arrays
			// println!("Skip empty");
			return Ok(());
		}
		if shape.len() != 2 {
			return Err(ReadonlyArray2Error::InvalidNdims(shape.len()));
		}

		// Check if f32
		let dtype = feature.dtype();
		if dtype.is_aligned_struct() || dtype.has_fields() {
			return Err(ReadonlyArray2Error::UnsuppportedDtype(dtype));
		}

		match dtype.kind() {
			b'f' => {
				if dtype.itemsize() != 4 {
					//TODO: support float64 input?
					return Err(ReadonlyArray2Error::UnsuppportedDtype(dtype));
				}
				if !dtype.is_native_byteorder().unwrap_or(false) {
					return Err(ReadonlyArray2Error::UnsuppportedDtype(dtype));
				}

				let as_f32 = feature.downcast::<PyArray2<f32>>()
					.map_err(|e| ReadonlyArray2Error::Py(e.into()))?;
				self.insert_f32(as_f32, index)
			},
			b'u' => {
				if dtype.itemsize() != 1 {
					//TODO: support unpacking?
					return Err(ReadonlyArray2Error::UnsuppportedDtype(dtype));
				}
				
				let as_u8 = feature.downcast::<PyArray2<u8>>()
					.map_err(|e| ReadonlyArray2Error::Py(e.into()))?;
				self.insert_u8(as_u8, index)
			},
			_ => (|| Err(ReadonlyArray2Error::UnsuppportedDtype(dtype)))(),
		}
	}
}

impl<'py> FromPyObject<'py> for PyFeaturesLike<'py> {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		let mut result = Self::Empty;
		if let Ok(features_numpy) = ob.downcast::<PyUntypedArray>() {
			// Single array
			result.insert_dyn(features_numpy, 0)?;
		} else {
			// Sequence of arrays
			for (i, feature) in ob.try_iter()?.enumerate() {
				let feature = feature?;
				match feature.downcast::<PyUntypedArray>() {
					Ok(array) => {
						result.insert_dyn(array, i)?;
					},
					Err(e) => {
						let err = PyErr::new::<PyTypeError, _>("Argument must be numpy.ndarray");
						let err_base = err.value(ob.py());
						if err_base.hasattr("add_note").unwrap_or(false) {
							// Ignore error, it can't help now
							let _ = err_base.call_method1("add_note", (format!("Original error: {e}",), ));
						}
						return Err(err);
					}
				}
			}
		}
		Ok(result)
	}

	fn type_input() -> pyo3::inspect::types::TypeInfo {
		// I don't see how to do a better union
		let mod_typing = ModuleName::Module(Cow::Borrowed("typing"));
		let mod_numpy = ModuleName::Module(Cow::Borrowed("numpy"));
		let ndarray = Cow::Borrowed("ndarray");
		let dtype = Cow::Borrowed("dtype");
		let int = TypeInfo::Class { module: ModuleName::Builtin, name: Cow::Borrowed("int"), type_vars: vec![] };
		let dim2 = TypeInfo::Tuple(Some(vec![
			int.clone(),
			int.clone(),
		]));
		// np.dtype[np.uint8]
		let dtype_u8 = TypeInfo::Class {
			module: mod_numpy.clone(),
			name: dtype.clone(),
			type_vars: vec![
				TypeInfo::Class {
					module: mod_numpy.clone(),
					name: Cow::Borrowed("uint8"),
					type_vars: vec![],
				}
			]
		};
		// np.dtype[np.float32]
		let dtype_f32 = TypeInfo::Class {
			module: mod_numpy.clone(),
			name: dtype,
			type_vars: vec![
				TypeInfo::Class {
					module: mod_numpy.clone(),
					name: Cow::Borrowed("float32"),
					type_vars: vec![],
				}
			]
		};

		// np.ndarray[tuple[T, T], dtype[uint8]]
		let ndarray_u8 = TypeInfo::Class {
			module: mod_numpy.clone(),
			name: ndarray.clone(),
			type_vars: vec![
				dim2.clone(),
				dtype_u8,
			]
		};
		// np.ndarray[tuple[T, T], dtype[float32]]
		let ndarray_f32 = TypeInfo::Class {
			module: mod_numpy,
			name: ndarray,
			type_vars: vec![
				dim2,
				dtype_f32,
			]
		};

		TypeInfo::Class {
			module: mod_typing,
			name: Cow::Borrowed("Union"),
			type_vars: vec![
				ndarray_u8.clone(),
				ndarray_f32.clone(),
				// Sequence[np.ndarray[tuple[T,T], np.dtype[np.uint8]]]
				TypeInfo::Class {
					module: ModuleName::Module(Cow::Borrowed("collections.abc")),
					name: Cow::Borrowed("Sequence"),
					type_vars: vec![ndarray_u8],
				},
				// Sequence[np.ndarray[tuple[T,T], np.dtype[np.float32]]]
				TypeInfo::Class {
					module: ModuleName::Module(Cow::Borrowed("collections.abc")),
					name: Cow::Borrowed("Sequence"),
					type_vars: vec![ndarray_f32],
				},
			]
		}
	}
}

fn py_read_array<'py, T: numpy::Element, D: ndarray::Dimension>(value: &Bound<'py, PyArray<T, D>>, index: usize) -> Result<PyReadonlyArray<'py, T, D>, ReadonlyArray2Error<'static>> {
	match value.try_readonly() {
		Ok(value) => Ok(value),
		Err(e) => Err(ReadonlyArray2Error::Borrow { index, error: e }),
	}
}