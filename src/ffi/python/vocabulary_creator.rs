use std::{borrow::Cow, hash::{DefaultHasher, Hash, Hasher}};

use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::{exceptions::{PyRuntimeError, PyValueError}, pyclass, pymethods, Py, PyErr, PyResult, PyTraverseError, PyVisit, Python};

use crate::{vocabulary_creator::VocabElement, Vocabulary, VocabularyCreator, VocabularyCreatorParams};

use super::{PyFeaturesLike, PyVocabulary};


#[pymethods]
impl VocabularyCreatorParams {
	/// Constructor
	#[new]
	#[pyo3(signature = (k=32, L=None, nthreads=0, max_iters=11, verbose=false))]
	#[allow(non_snake_case)]
	fn __init__(k: u32, L: Option<u32>, nthreads: usize, max_iters: usize, verbose: bool) -> Self {
		Self { k, L, nthreads, max_iters, verbose }
	}

    fn __str__(&self) -> String {
        format!("{self:?}")
    }

	fn __repr__(&self) -> String {
        // Convert to Python reprs
        //TODO: proc macro for this?
        let l = match self.L {
            None => Cow::Borrowed("None"),
            Some(l) => format!("{l}").into(),
        };
        let verbose = if self.verbose { "True" } else { "False" };

        format!("VocabularyCreatorParams(k={}, L={}, nthreads={}, max_iters={}, verbose={})", self.k, l, self.nthreads, self.max_iters, verbose)
	}

    /// Get number of threads that will be used
    fn effective_threads(&self) -> usize {
        if self.nthreads == 0 {
            match std::thread::available_parallelism() {
                Err(_) => 1,
                Ok(nthreads) => nthreads.get(),
            }
        } else {
            self.nthreads
        }
    }

    /// Hash
	fn __hash__(&self, py: Python<'_>) -> u64 {
		py.detach(|| {
			let mut hasher = DefaultHasher::new();
			self.hash(&mut hasher);
			hasher.finish()
		})
	}
}

enum MaybeBoundParams {
    Bound(Py<VocabularyCreatorParams>),
    Value(VocabularyCreatorParams),
}

#[pyclass(module="vfbow", name="VocabularyCreator")]
pub(super) struct PyVocabularyCreator {
    params: MaybeBoundParams,
}

impl PyVocabularyCreator {
    /// Get as native [VocabularyCreator]
    fn as_native(&self, py: Python<'_>) -> PyResult<VocabularyCreator> {
        let params = match &self.params {
            MaybeBoundParams::Value(params) => params.clone(),
            MaybeBoundParams::Bound(bound) => bound.try_borrow(py)?.clone(),
        };
        Ok(VocabularyCreator::new(params))
    }

    /// Type-generic version of [::create]
    fn create_generic<'py, T: numpy::Element + VocabElement + Sync>(&self, py: Python<'py>, vc: VocabularyCreator, vec: Vec<PyReadonlyArray2<'py, T>>, desc_name: &str) -> PyResult<Vocabulary> {
        //TODO: prevent array copies
        let features = vec.into_iter()
            .map(|arr| arr.to_owned_array())
            .collect::<Vec<_>>();

        py.detach(|| {
            match vc.create::<T>(features, desc_name) {
                Ok(voc) => Ok(voc),
                Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!("Error creating vocabulary: {e}"))),
            }
        })
    }
}

#[pymethods]
impl PyVocabularyCreator {
	/// New VocabularyCreator with parameters
	#[new]
	#[pyo3(signature=(params = VocabularyCreatorParams::default()))]
	fn __init__(params: VocabularyCreatorParams) -> Self {
        // Note: we always make a copy of params here
		Self { params: MaybeBoundParams::Value(params) }
	}

	#[getter]
	fn params<'py>(&mut self, py: Python<'py>) -> PyResult<Py<VocabularyCreatorParams>> {
        match &self.params {
            MaybeBoundParams::Value(params) => {
                let res = Py::new(py, params.clone())?;
                self.params = MaybeBoundParams::Bound(res.clone_ref(py));
                Ok(res)
            },
            MaybeBoundParams::Bound(params) => Ok(params.clone_ref(py)),
        }
	}
	
	/// Create vocabulary from features
	fn create<'py>(&self, py: Python<'py>, features: PyFeaturesLike<'py>, desc_name: &str) -> PyResult<PyVocabulary> {
		let vc = self.as_native(py)?;

		let result = match features {
			PyFeaturesLike::Empty => return Err(PyErr::new::<PyValueError, _>("No features provided")),
			PyFeaturesLike::U8(vec) => self.create_generic(py, vc, vec, desc_name),
			PyFeaturesLike::F32(vec) => self.create_generic(py, vc, vec, desc_name),
		};
        result.map(|voc| voc.into())
	}

	fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
		if let MaybeBoundParams::Bound(bound) = &self.params {
            visit.call(bound)?;
        }
        Ok(())
	}

	fn __clear__(&mut self) {
		if matches!(&self.params, MaybeBoundParams::Bound(..)) {
            // We could add a variant for 'cleared' but don't need to
            self.params = MaybeBoundParams::Value(VocabularyCreatorParams::default());
        }
	}
}