mod dispatch;
mod io;
mod features_array;
mod backtrace;

use std::{io::BufWriter, panic::RefUnwindSafe};

use backtrace::PanicBacktrace;
use features_array::PyReadonlyArray2Any;
use io::{PyRead, PyWrite};
use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::{exceptions::{PyRuntimeError, PyValueError}, prelude::*, pymethods, pymodule, types::{PyBytes, PyModule}, Bound, PyResult, Python};

use crate::{features::FeatureType, util::{Deserialize, Serialize}, vocabulary::{TransformError, Vocabulary}, vocabulary_creator::VocabElement, CreateVocabularyError, VocabularyCreator, VocabularyCreatorParams, FBOW, FBOW2};

#[pymethods]
impl VocabularyCreatorParams {
	/// Create new params object
	#[new]
	#[pyo3(signature = (k=32, L=None, nthreads=0, max_iters=11, verbose=false))]
	#[allow(non_snake_case)]
	fn new(k: u32, L: Option<u32>, nthreads: usize, max_iters: usize, verbose: bool) -> Self {
		Self { k, L, nthreads, max_iters, verbose }
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}
}

#[pymethods]
impl VocabularyCreator {
	#[new]
	fn py_new(params: VocabularyCreatorParams) -> Self {
		Self::new(params)
	}
	
	#[pyo3(name="create")]
	fn py_create<'py>(&self, py: Python<'py>, features: PyReadonlyArray2Any<'py>, desc_name: &str) -> PyResult<Vocabulary> {
		fn create_generic<'py, T: numpy::Element + VocabElement + Sync + RefUnwindSafe>(vc: &VocabularyCreator, py: Python<'py>, vec: Vec<PyReadonlyArray2<'py, T>>, desc_name: &str) -> PyResult<Result<Vocabulary, CreateVocabularyError>> {
			//TODO: prevent array copies
			let mut features = Vec::new();
			for arr in vec {
				features.push(arr.to_owned_array());
			}

			// Set panic hook
			PanicBacktrace::catch_backtrace(py, || vc.create::<T>(features, desc_name))
		}

		let result = match features {
			PyReadonlyArray2Any::Empty => Err(PyErr::new::<PyValueError, _>("No features provided")),
			PyReadonlyArray2Any::U8(vec) => create_generic(self, py, vec, desc_name),
			PyReadonlyArray2Any::F32(vec) => create_generic(self, py, vec, desc_name),
		}?;

		match result {
			Ok(v) => Ok(v),
			// Normal result
			Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!("Error creating vocabulary: {e}"))),
		}
	}
}

#[pymethods]
impl FBOW {
	fn __len__(&self) -> usize {
		self.len()
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	fn keys(&self) -> Vec<u32> {
		self.as_ref()
			.keys()
			.copied()
			.collect()
	}

	fn __getitem__(&self, key: u32) -> Option<f32> {
		self.as_ref().get(&key).cloned()
	}
}

#[pymethods]
impl FBOW2 {
	fn __len__(&self) -> usize {
		self.len()
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	fn __getitem__(&self, key: u32) -> Option<Vec<u32>> {
		self.as_ref().get(&key).cloned()
	}

	fn keys(&self) -> Vec<u32> {
		self.as_ref()
			.keys()
			.copied()
			.collect()
	}
}

#[pymethods]
impl Vocabulary {
	/// Read from file
	#[staticmethod]
	fn read_from(py: Python<'_>, src: PyRead) -> PyResult<Self> {
		let res = py.allow_threads(|| {
			<Self as Deserialize>::read_from(src)
		})?;
		Ok(res)
	}

	/// Deserialize bytes
	#[staticmethod]
	fn from_bytes(py: Python<'_>, src: Bound<'_, PyBytes>) -> PyResult<Self> {
		let mut bytes = src.as_bytes();
		let res = py.allow_threads(|| {
			<Self as Deserialize>::read_from(&mut bytes)
		})?;
		Ok(res)
	}

	/// Write to file
	fn write_to(&self, py: Python<'_>, dst: PyWrite) -> PyResult<()> {
		let res = py.allow_threads(|| {
			let buffered = BufWriter::new(dst);
			Serialize::write_to(self, buffered)
		})?;
		Ok(res)
	}

	/// Serialize to bytes
	fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
		py.allow_threads(|| {
			let mut bytes = Vec::new();
			Serialize::write_to(self, &mut bytes)?;
			Ok(bytes)
		})
	}

	/// Print tree to stdout
	#[pyo3(name="print_tree")]
	fn py_print_tree(&self) {
		self.print_tree();
	}

	/// Get branching factor 'k'
	#[getter(k)]
	fn py_k(&self) -> u32 {
		self.k()
	}

	/// Get number of features
	#[getter(num_features)]
	fn py_num_features(&self) -> usize {
		self.num_features()
	}

	#[getter(desc_name)]
	fn py_desc_name(&self) -> &str {
		self.desc_name()
	}

	fn __len__(&self) -> usize {
		self.size() as _
	}

	fn __str__(&self) -> String {
		format!("{self:?}")
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	#[pyo3(name="features")]
	fn py_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
		self.features().to_python(py)
	}

	#[pyo3(name="transform", signature = (features, level = None))]
	fn py_transform<'py>(&self, py: Python<'py>, features: PyReadonlyArray2Any<'py>, level: Option<usize>) -> PyResult<(Bound<'py, FBOW>, Bound<'py, FBOW2>)> {

		fn transform_inner<'py, T: numpy::Element + FeatureType + Send + Sync + RefUnwindSafe>(py: Python<'py>, vocab: &Vocabulary, features: Vec<PyReadonlyArray2<'py, T>>, level: Option<usize>) -> PyResult<Result<(FBOW, FBOW2), TransformError>> {
			let features = features.into_iter()
				.map(|feature| feature.to_owned_array())
				.collect::<Vec<_>>();
			assert_eq!(features.len(), 1);

			// Set panic hook
			PanicBacktrace::catch_backtrace(py, || vocab.transform::<T>(features[0].view(), level))
		}
		let result = match features {
			PyReadonlyArray2Any::Empty => return Err(PyErr::new::<PyValueError, _>("No features")),
			PyReadonlyArray2Any::U8(features) => transform_inner(py, self, features, level),
			PyReadonlyArray2Any::F32(features) => transform_inner(py, self, features, level),
		}?;

		match result {
			Ok((r1, r2)) => Ok((
				Bound::new(py, r1)?,
				Bound::new(py, r2)?,
			)),
			Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!("Error transforming features: {e}"))),
		}
	}
}

#[pymodule]
#[pyo3(name="vfbow")]
fn fbow_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<VocabularyCreatorParams>()?;
	m.add_class::<VocabularyCreator>()?;
	m.add_class::<Vocabulary>()?;
	m.add_class::<FBOW>()?;
	m.add_class::<FBOW2>()?;
	Ok(())
}