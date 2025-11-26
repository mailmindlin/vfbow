//! Python FFI bindings, using `pyo3`
mod dispatch;
mod io;
mod features_array;
mod core;
mod vocabulary_creator;

use std::{borrow::Cow, io::BufWriter, num::NonZeroUsize, ops::Deref, panic::RefUnwindSafe, sync::Arc};

use features_array::{CowArray2Any, PyFeaturesLike};
use io::{PyRead, PyWrite};
use ndarray::CowArray;
use numpy::{Ix2, PyArrayDescr, PyReadonlyArray2};
use pyo3::{exceptions::{PyRuntimeError, PyValueError}, prelude::*, pymethods, pymodule, types::{PyBytes, PyModule}, Bound, PyResult, Python};
use rayon::prelude::*;
use vocabulary_creator::PyVocabularyCreator;

use crate::{db::{Database, QueryResult}, features::FeatureType, util::{scoring::LNorm, Scoring, Serialize}, vocabulary::{ParseValidationMode, TransformError, Vocabulary, VocabularyReadOptions}, vocabulary_creator::VocabElement, Bow, Features, VocabularyCreatorParams};

/// Python wrapper for [Vocabulary] using an Arc internally to reduce copies when construcitng [Database]
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", name="Vocabulary", frozen))]
#[derive(Clone)]
struct PyVocabulary(Arc<Vocabulary>);
impl From<Vocabulary> for PyVocabulary {
	fn from(value: Vocabulary) -> Self {
		Self(Arc::new(value))
	}
}
impl From<Arc<Vocabulary>> for PyVocabulary {
	fn from(value: Arc<Vocabulary>) -> Self {
		Self(value)
	}
}
impl Deref for PyVocabulary {
	type Target = Vocabulary;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

#[pymethods]
impl PyVocabulary {
	/// Read from file
	#[staticmethod]
	#[pyo3(signature=(src, level = None))]
	fn read_from(py: Python<'_>, src: PyRead, level: Option<ParseValidationMode>) -> PyResult<Self> {
		let options = match level {
			None => VocabularyReadOptions::default(),
			Some(level) => VocabularyReadOptions::all(level),
		};

		let res = py.detach(|| {
			Vocabulary::read_from(src, options)
				.map(|v| v.into())
		})?;
		Ok(res)
	}

	/// Deserialize bytes
	#[staticmethod]
	#[pyo3(signature=(src, level = None))]
	fn from_bytes(py: Python<'_>, src: Bound<'_, PyBytes>, level: Option<ParseValidationMode>) -> PyResult<Self> {
		let options = match level {
			None => VocabularyReadOptions::default(),
			Some(level) => VocabularyReadOptions::all(level),
		};

		let mut bytes = src.as_bytes();
		let res = py.detach(|| {
			Vocabulary::read_from(&mut bytes, options)
				.map(|v| v.into())
		})?;
		Ok(res)
	}

	/// Write to file
	fn write_to(&self, py: Python<'_>, dst: PyWrite) -> PyResult<()> {
		py.detach(|| {
			let buffered = BufWriter::new(dst);
			<Vocabulary as Serialize>::write_to(self, buffered)
		})?;
		Ok(())
	}

	/// Serialize to bytes
	fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
		py.detach(|| {
			let mut bytes = Vec::new();
			<Vocabulary as Serialize>::write_to(self, &mut bytes)?;
			Ok(bytes)
		})
	}

	/// Get descriptor dtype (np.uint8 or np.float32)
	#[getter]
	fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
		match self.desc_type() {
			crate::util::DescriptorType::Uint8 => PyArrayDescr::of::<u8>(py),
			crate::util::DescriptorType::Float32 => PyArrayDescr::of::<f32>(py),
		}
	}

	/// Print tree to stdout
	#[pyo3(name="print_tree")]
	fn py_print_tree(&self) {
		self.print_tree();
	}

	/// Branching factor 'k'
	#[getter(k)]
	fn py_k(&self) -> u32 {
		self.k()
	}

	/// Number of features
	#[getter(num_features)]
	fn py_num_features(&self) -> usize {
		self.num_features()
	}

	/// Descriptor name
	#[getter(desc_name)]
	fn py_desc_name(&self) -> &str {
		self.desc_name()
	}

	/// Total number of blocks
	fn __len__(&self) -> usize {
		self.size() as _
	}

	/// String description
	fn __str__(&self) -> String {
		format!("{:?}", self.0.as_ref())
	}

	/// String representation
	fn __repr__(&self) -> String {
		format!("{:?}", self.0.as_ref())
	}

	fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
		self.0.features()
			.to_python(py)
	}

	#[pyo3(name="transform", signature = (features, level = None))]
	fn py_transform<'py>(&self, py: Python<'py>, features: PyFeaturesLike<'py>, level: Option<usize>) -> PyResult<(Bound<'py, Bow>, Bound<'py, Features>)> {
		let Some(features) = features.as_2d() else {
			return Err(PyErr::new::<PyValueError, _>("No features"));
		};

		fn transform_inner<'py, T: numpy::Element + FeatureType + Send + Sync + RefUnwindSafe>(py: Python<'py>, vocab: &Vocabulary, features: CowArray<T, Ix2>, level: Option<usize>) -> Result<(Bow, Features), TransformError> {
			py.detach(|| vocab.transform::<T>(features.view(), level))
		}

		let result = match features {
			CowArray2Any::U8(features) => transform_inner(py, self, features, level),
			CowArray2Any::F32(features) => transform_inner(py, self, features, level),
		};

		match result {
			Ok((r1, r2)) => Ok((
				Bound::new(py, r1)?,
				Bound::new(py, r2)?,
			)),
			Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!("Error transforming features: {e}"))),
		}
	}
}

#[pymethods]
impl Database {
	#[new]
	#[pyo3(signature=(vocabulary, use_direct = true, di_levels = 0))]
	fn py_new(vocabulary: &PyVocabulary, use_direct: bool, di_levels: usize) -> Self {
		Self::new(vocabulary.0.clone(), use_direct, di_levels)
	}

	/// True if not empty
	fn __bool__(&self) -> bool {
		!self.is_empty()
	}
	fn __len__(&self) -> usize {
		self.len()
	}
	/// Get vocabulary
	#[getter(vocabulary)]
	fn py_vocabulary(&self) -> PyVocabulary {
		//TODO: make make this always the same object?
		self.vocabulary().clone().into()
	}

	/// Insert features and 
	#[pyo3(name="insert_transform")]
	fn py_insert_transform<'py>(&mut self, py: Python<'py>, features: PyFeaturesLike<'py>) -> PyResult<Bound<'py, PyAny>> {
		fn inner<'py, T: numpy::Element + VocabElement + Send + Sync + RefUnwindSafe>(py: Python<'py>, db: &mut Database, mut features: Vec<PyReadonlyArray2<'py, T>>) -> Result<Vec<usize>, TransformError> {
			assert_ne!(features.len(), 0);
			if features.len() == 1 {
				let id = {
					let py_feature = features
						.pop()
						.unwrap();
					let feature = py_feature.as_array();

					let (id, _, _) = py.detach(|| {
						db.insert_transform(feature)
					})?;
					id
				};
				//TODO: complex return type
				Ok(vec![id])
			} else {
				// Bulk insert
				// Map features
				let feat_refs = features
					.iter()
					.map(|feature| feature.as_array())
					.collect::<Vec<_>>();
				
				py.detach(|| {
					// Transform features in parallel
					let vocab = db.vocabulary();
					let levels = db.direct_index_levels();
					feat_refs
						.into_par_iter()
						.map(|feat| vocab.transform(feat, levels))
						.collect::<Result<Vec<_>, _>>()
						.map(|feat_tfs| {
							// I don't see how we gain any parallelism here
							feat_tfs
								.into_iter()
								.map(|(v, f)| {
									let mut f = Cow::Owned(f);
									db.insert(&v, &mut f)
								})
								.collect::<Vec<_>>()
						})
				})
			}
		}

		let result = match features {
			PyFeaturesLike::Empty => return Err(PyErr::new::<PyValueError, _>("No features")),
			PyFeaturesLike::U8(features) => inner(py, self, features),
			PyFeaturesLike::F32(features) => inner(py, self, features),
		};

		// Ideally we'd track the raw type of `features` and return an int/list according, but that's hard
		match result {
			Ok(r) => {
				Ok(if r.len() == 1 {
					r[0]
						.into_pyobject(py)
						.unwrap() // Infallable
						.into_any()
				} else {
					r.into_pyobject(py)?
						.into_any()
				})
			},
			Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!("Error transforming features: {e}"))),
		}
	}

	#[pyo3(name="insert")]
	fn py_insert<'py>(&mut self, py: Python<'py>, bow: &Bow, features: &Features) -> usize {
		py.detach(|| {
			let mut fv = Cow::Borrowed(features);
			self.insert(bow, &mut fv)
		})
	}

	#[pyo3(name="query", signature=(query, scoring, max_results=None, max_id=None))]
	fn py_query(&self, py: Python, query: &Bow, scoring: Scoring, max_results: Option<usize>, max_id: Option<usize>) -> PyDbQueryResults {
		// Normalize parameters
		let max_results = match max_results {
			Some(v) => match NonZeroUsize::new(v) {
				Some(v) => Some(v),
				None => return PyDbQueryResults::empty(),
			},
			None => None,
		};
		let max_id = match max_id {
			Some(v) => match NonZeroUsize::new(v) {
				Some(v) => Some(v),
				None => return PyDbQueryResults::empty(),
			},
			None => None,
		};

		let r = py.detach(|| self.query(query, scoring, max_results, max_id));
		PyDbQueryResults::new(r)
	}

	//TODO: support query_transform
	#[pyo3(name="clear")]
	fn py_clear(&mut self) {
		self.clear();
	}
}

/// Results from [Database::query()]
/// 
/// Results are sorted in ascending-score order
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", name="QueryResults", sequence, frozen))]
#[derive(Clone, Debug)]
struct PyDbQueryResults {
	results: Vec<QueryResult>,
}

impl PyDbQueryResults {
	fn empty() -> Self {
		//TODO: maybe cache this?
		Self::new(Vec::new())
	}
	fn new(results: Vec<QueryResult>) -> Self {
		Self { results }
	}
}

#[pymethods]
impl PyDbQueryResults {
	/// Get entry IDs (in ascending-score order)
	fn ids(&self, py: Python) -> Vec<usize> {
		py.detach(|| {
			self.results
				.iter()
				.map(|r| r.id)
				.collect()
		})
	}
	fn scores(&self, py: Python) -> Vec<f64> {
		py.detach(|| {
			self.results
				.iter()
				.map(|r| r.score)
				.collect()
		})
	}
	/// True if not empty
	fn __bool__(&self) -> bool {
		!self.results.is_empty()
	}
	/// Number of results
	fn __len__(&self) -> usize {
		self.results.len()
	}
	/// Get item by index
	fn __getitem__(&self, idx: usize) -> Option<QueryResult> {
		self.results.get(idx).copied()
	}
	/// Get Python list of results
	fn to_list(&self) -> Vec<QueryResult> {
		self.results.clone()
	}
}


#[pymodule]
#[pyo3(name="vfbow")]
fn fbow_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<ParseValidationMode>()?;
	m.add_class::<VocabularyCreatorParams>()?;
	m.add_class::<PyVocabularyCreator>()?;
	m.add_class::<PyVocabulary>()?;

	m.add_class::<Bow>()?;
	m.add_class::<Features>()?;
	m.add_class::<Scoring>()?;
	m.add_class::<LNorm>()?;

	// DB
	m.add_class::<Database>()?;
	m.add_class::<QueryResult>()?;
	m.add_class::<PyDbQueryResults>()?;
	Ok(())
}