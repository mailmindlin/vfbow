mod dispatch;
mod io;
mod features_array;

use std::io::BufWriter;

use features_array::PyReadonlyArray2Any;
use io::{PyRead, PyWrite};
use numpy::{PyArray, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError}, prelude::*, pymethods, pymodule, types::PyModule, Bound, PyResult, Python};

use crate::{traits::{Deserialize, SelfHash, Serialize}, vocabulary::Vocabulary, VocabularyCreator, VocabularyCreatorParams, FBOW, FBOW2};

#[pymethods]
impl VocabularyCreatorParams {
	/// Create new params object
	#[new]
	#[pyo3(signature = (k=32, L=None, nthreads=0, max_iters=11, verbose=false))]
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
		println!("{features:?} / {desc_name}");
		match features {
			PyReadonlyArray2Any::Empty => Err(PyErr::new::<PyValueError, _>("No features provided")),
			PyReadonlyArray2Any::U8(vec) => {
				
				Err(PyErr::new::<PyNotImplementedError,_>("TODO"))
			},
			PyReadonlyArray2Any::F32(vec) => Err(PyErr::new::<PyNotImplementedError,_>("TODO")),
		}
	}
}

#[pymethods]
impl Vocabulary {
	#[staticmethod]
	fn read_from(py: Python<'_>, src: PyRead) -> PyResult<Self> {
		let res = py.allow_threads(|| {
			<Self as Deserialize>::read_from(src)
		})?;
		Ok(res)
	}

	fn write_to(&self, py: Python<'_>, dst: PyWrite) -> PyResult<()> {
		let res = py.allow_threads(|| {
			let buffered = BufWriter::new(dst);
			Serialize::write_to(self, buffered)
		})?;
		Ok(res)
	}

	#[getter(k)]
	fn py_k(&self) -> u32 {
		self.k()
	}

	#[pyo3(name="clear")]
	fn py_clear(&mut self) {
		self.clear();
	}

	#[getter(desc_name)]
	fn py_desc_name(&self) -> &str {
		self.desc_name()
	}

	fn __len__(&self) -> usize {
		self.size() as _
	}

	fn __hash__(&self) -> u64 {
		SelfHash::hash(self)
	}
}

#[pymodule]
#[pyo3(name="fbow_rs")]
fn fbow_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<VocabularyCreatorParams>()?;
	m.add_class::<VocabularyCreator>()?;
	m.add_class::<Vocabulary>()?;
	m.add_class::<FBOW>()?;
	m.add_class::<FBOW2>()?;
	Ok(())
}