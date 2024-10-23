use std::borrow::Cow;

use numpy::{PyArray, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{exceptions::{PyRuntimeError, PyTypeError, PyValueError}, inspect::types::{ModuleName, TypeInfo}, types::PyAnyMethods, Bound, FromPyObject, PyAny, PyErr, PyResult};

#[derive(Debug)]
pub(super) enum PyReadonlyArray2Any<'py> {
	Empty,
	U8(Vec<PyReadonlyArray2<'py, u8>>),
	F32(Vec<PyReadonlyArray2<'py, f32>>),
}

impl<'py> PyReadonlyArray2Any<'py> {
	fn insert_u8(&mut self, value: &Bound<'py, PyArray2<u8>>, index: usize) -> PyResult<()> {
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
			_ => Err(PyErr::new::<PyTypeError, _>("Inconsistent feature dtypes")),
		}
	}
	fn insert_f32(&mut self, value: &Bound<'py, PyArray2<f32>>, index: usize) -> PyResult<()> {
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
			_ => Err(PyErr::new::<PyTypeError, _>("Inconsistent feature dtypes")),
		}
	}

	fn insert_dyn(&mut self, feature: &Bound<'py, PyUntypedArray>, index: usize) -> PyResult<()> {
		let shape = feature.shape();
		// Skip empty arrays
		if shape.len() == 0 || shape.iter().any(|d| *d == 0) {
			// Skip empty arrays
			println!("Skip empty");
			return Ok(());
		}
		if shape.len() != 2 {
			return Err(PyErr::new::<PyValueError, _>(format!("Expected 2D array (ndim={})", shape.len())));
		}

		// Check if f32
		let dtype = feature.dtype();
		let bad_dtype = || Err(PyErr::new::<PyTypeError, _>(format!("Unsupported dtype {dtype}")));
		if dtype.is_aligned_struct() || dtype.has_fields() {
			return bad_dtype();
		}

		match dtype.kind() {
			b'f' => {
				if dtype.itemsize() != 4 {
					//TODO: support float64 input?
					return bad_dtype();
				}
				if !dtype.is_native_byteorder().unwrap_or(false) {
					return bad_dtype();
				}

				let as_f32 = feature.downcast::<PyArray2<f32>>()?;
				self.insert_f32(as_f32, index)
			}
			b'u' => {
				if dtype.itemsize() != 1 {
					//TODO: support unpacking?
					return bad_dtype();
				}
				
				let as_u8 = feature.downcast::<PyArray2<u8>>()?;
				self.insert_u8(as_u8, index)
			},
			_ => bad_dtype(),
		}
	}
}

impl<'py> FromPyObject<'py> for PyReadonlyArray2Any<'py> {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		let mut result = Self::Empty;
		if let Ok(features_numpy) = ob.downcast::<PyUntypedArray>() {
			result.insert_dyn(features_numpy, 0)?;
		} else {
			for (i, feature) in ob.iter()?.enumerate() {
				let feature = feature?;
				let array = feature.downcast::<PyUntypedArray>()
					.map_err(|e| {
						let err = PyErr::new::<PyTypeError, _>("Argument must be numpy.ndarray");
						let err_base = err.value_bound(ob.py());
						if err_base.hasattr("add_note").unwrap_or(false) {
							// Ignore error, it can't help now
							let _ = err_base.call_method1("add_note", (format!("Original error: {e}",), ));
						}
						err
					})?;
				result.insert_dyn(array, i)?;
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

fn py_borrow_error(e: numpy::BorrowError, index: usize) -> PyResult<!> {
	let err = PyErr::new::<PyRuntimeError, _>(format!("Unable to borrow array at index {index}: {e}"));
	Err(err)
}

fn py_read_array<'py, T: numpy::Element, D: ndarray::Dimension>(value: &Bound<'py, PyArray<T, D>>, index: usize) -> PyResult<PyReadonlyArray<'py, T, D>> {
	match value.try_readonly() {
		Ok(value) => Ok(value),
		Err(e) => py_borrow_error(e, index)?,
	}
}