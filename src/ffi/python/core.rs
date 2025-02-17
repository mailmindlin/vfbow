use std::{borrow::Cow, ops::Deref};

use numpy::PyArray1;
use pyo3::{exceptions::{PyKeyError, PyTypeError, PyValueError, PyZeroDivisionError}, inspect::types::{ModuleName, TypeInfo}, pyclass, pymethods, types::{IntoPyDict, PyAnyMethods, PyDict, PyString, PyStringMethods}, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyResult, PyTraverseError, PyVisit, Python};
use rayon::iter::Either;

use crate::{util::{scoring::LNorm, Scoring, SelfHash}, Bow, Features};

#[derive(Clone, Copy, Debug, PartialEq)]
enum ViewMode {
	Keys,
	Values,
	Items,
}

impl ViewMode {
	const fn default_sort(&self) -> SortMode {
		match self {
			ViewMode::Keys => SortMode::Keys,
			ViewMode::Values => SortMode::Values,
			ViewMode::Items => SortMode::Keys,
		}
	}
}

#[derive(Debug)]
enum AnyDict {
	Bow(Py<Bow>),
	Features(Py<Features>),
	None,
}

impl AnyDict {
	fn traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
		match self {
			AnyDict::Bow(b) => visit.call(b),
			AnyDict::Features(b) => visit.call(b),
			AnyDict::None => Ok(()),
		}
	}
	fn clear(&mut self) {
		*self = Self::None;
	}

	fn borrow(&self) -> Either<&Bow, &Features> {
		match self {
			AnyDict::Bow(b) => Either::Left(b.get()),
			AnyDict::Features(b) =>Either::Right(b.get()),
			AnyDict::None => panic!("Use after GC"),
		}
	}
}

/// Key/value/item view for 
#[pyclass(name="_PyDictView", sequence)]
#[derive(Debug)]
struct PyDictView {
	base: AnyDict,
	mode: ViewMode,
	reversed: bool,
}


#[derive(IntoPyObject)]
enum ListOutput {
	Keys(Vec<u32>),
	ValueB(Vec<f32>),
	ItemB(Vec<(u32, f32)>),
	ValueF(Vec<Vec<u32>>),
	ItemF(Vec<(u32, Vec<u32>)>),
}

impl ListOutput {
	fn to_numpy<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
		Ok(match self {
			ListOutput::Keys(items) => PyArray1::from_vec(py, items).into_any(),
			ListOutput::ValueB(items) => PyArray1::from_vec(py, items).into_any(),
			ListOutput::ItemB(items) => {
				let mut keys = Vec::with_capacity(items.len());
				let mut values = Vec::with_capacity(items.len());
				for (key, value) in items {
					keys.push(key);
					values.push(value);
				}
				// I think this is more reasonable than a 2d heterogenous array
				let keys = PyArray1::from_vec(py, keys);
				let values = PyArray1::from_vec(py, values);
				(keys, values).into_bound_py_any(py)?
			}
			// I don't *think* all the values are the same length
			ListOutput::ValueF(items) => {
				let items = items.into_iter()
					.map(|item| PyArray1::from_vec(py, item))
					.collect::<Vec<_>>();
				items.into_bound_py_any(py)?
			},
			ListOutput::ItemF(items) => {
				let mut keys = Vec::with_capacity(items.len());
				let mut values = Vec::with_capacity(items.len());
				for (key, value) in items {
					keys.push(key);
					values.push(PyArray1::from_vec(py, value));
				}

				let keys = PyArray1::from_vec(py, keys);
				(keys, values).into_bound_py_any(py)?
			},
		})
	}
}

impl PyDictView {
	fn to_py<'py>(&self, py: Python<'py>, sorted: SortModeRaw) -> PyResult<ListOutput> {
		let sorted = SortMode::from(sorted, self.mode.default_sort());

		// Early errors
		if sorted == SortMode::Values && self.mode == ViewMode::Keys {
			return Err(PyErr::new::<PyValueError, _>("Unable to sort keys view by value"));
		}

		// Helper functions
		fn copied_vec<'a, E: Copy + 'a>(src: impl IntoIterator<Item = &'a E>) -> Vec<E> {
			src
				.into_iter()
				.copied()
				.collect()
		}
		fn map_vec<E, R>(src: impl IntoIterator<Item = E>, f: impl Fn(E) -> R) -> Vec<R> {
			src
				.into_iter()
				.map(f)
				.collect()
		}

		let base = self.base.borrow();
		py.allow_threads(|| {
			Ok(match base {
				Either::Left(b) => {
					let b = b.as_ref();
					let effective_mode = match (self.mode, sorted) {
						(ViewMode::Values, SortMode::Keys) => ViewMode::Items,
						(ViewMode::Keys, SortMode::Values) => ViewMode::Items,
						_ => self.mode,
					};

					match effective_mode {
						ViewMode::Keys => {
							let mut keys = copied_vec(b.keys());
							match sorted {
								// Keys should be unique, so we shouldn't need to worry about stability
								SortMode::Keys => keys.sort_unstable(),
								SortMode::None => {},
								_ => unreachable!(),
							}

							if self.reversed {
								keys.reverse();
							}
							
							ListOutput::Keys(keys)
						},
						ViewMode::Values => {
							let mut values = copied_vec(b.values());
							match sorted {
								SortMode::Values => values.sort_by(f32::total_cmp),
								SortMode::None => {},
								SortMode::Keys => unreachable!(),
							}

							if self.reversed {
								values.reverse();
							}

							ListOutput::ValueB(values)
						},
						ViewMode::Items => {
							let mut items = b.iter()
								.map(|(k, v)| (*k, *v))
								.collect::<Vec<_>>();
							match sorted {
								SortMode::Keys => items.sort_by_key(|(k, _)| *k),
								SortMode::Values => items.sort_by(|(_, u), (_, v)| u.total_cmp(v)),
								SortMode::None => {},
							}

							if self.reversed {
								items.reverse();
							}

							// We might need to extract k/v
							match self.mode {
								ViewMode::Items => ListOutput::ItemB(items),
								ViewMode::Keys => ListOutput::Keys(map_vec(items, |(k, _)| k)),
								ViewMode::Values => ListOutput::ValueB(map_vec(items, |(_, v)| v)),
							}
						},
					}
				},
				Either::Right(f) => {
					let b = f.as_ref();
					let effective_mode = match (self.mode, sorted) {
						(ViewMode::Values, SortMode::Keys) => ViewMode::Items,
						(_, SortMode::Values) => return Err(PyErr::new::<PyValueError, _>("Unable to sort Features by value")),
						_ => self.mode,
					};

					match effective_mode {
						ViewMode::Keys => {
							let mut keys = copied_vec(b.keys());
							match sorted {
								// Keys should be unique, so we shouldn't need to worry about stability
								SortMode::Keys => keys.sort_unstable(),
								SortMode::None => {},
								_ => unreachable!(),
							}

							if self.reversed {
								keys.reverse();
							}

							ListOutput::Keys(keys)
						},
						ViewMode::Values => {
							let mut values = b.values()
								.cloned()
								.collect::<Vec<_>>();

							match sorted {
								SortMode::None => {},
								_ => unreachable!(),
							}
							if self.reversed {
								values.reverse();
							}
							ListOutput::ValueF(values)
						},
						ViewMode::Items => {
							let mut items = map_vec(b, |(k, v)| (*k, v.as_slice()));
							
							match sorted {
								SortMode::Keys => items.sort_by_key(|(k, _)| *k),
								SortMode::Values => unreachable!(),
								SortMode::None => {
									//TODO: eliminate a step copy here
								},
							}

							if self.reversed {
								items.reverse();
							}

							// We might need to extract k/v
							match self.mode {
								ViewMode::Items => ListOutput::ItemF(map_vec(items, |(k, v)| (k, v.to_vec()))),
								ViewMode::Keys => ListOutput::Keys(map_vec(items, |(k, _)| k)),
								ViewMode::Values => ListOutput::ValueF(map_vec(items, |(_, v)| v.to_vec())),
							}
						},
					}
				},
			})
		})
	}
}

#[pymethods]
impl PyDictView {
	/// True if not empty
	fn __bool__(&self) -> bool {
		match self.base.borrow() {
			Either::Left(b) => b.__bool__(),
			Either::Right(f) => f.__bool__(),
		}
	}

	/// Return number of elements in Bow
	fn __len__(&self) -> usize {
		match self.base.borrow() {
			Either::Left(b) => b.__len__(),
			Either::Right(f) => f.__len__(),
		}
	}

	fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
		// Delegate to iter(list(self))
		self.to_list(py, SortModeRaw::None)?
			.into_bound_py_any(py)?
			.call_method0("__iter__")
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	/// Efficient of all items conversion to numpy ndarray
	#[pyo3(signature = (sorted = SortModeRaw::None))]
	fn to_numpy<'py>(&self, py: Python<'py>, sorted: SortModeRaw) -> PyResult<Bound<'py, PyAny>> {
		self.to_py(py, sorted)
			.and_then(|r| r.to_numpy(py))
	}

	#[pyo3(signature = (sorted = SortModeRaw::None))]
	fn to_list<'py>(&self, py: Python<'py>, sorted: SortModeRaw) -> PyResult<ListOutput> {
		self.to_py(py, sorted)
	}
	
	fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
		self.base.traverse(visit)
	}

	fn __clear__(&mut self) {
		self.base.clear();
	}
}

/// Like [SortMode] but with default option
#[derive(Clone, Copy, Debug, PartialEq)]
enum SortModeRaw {
	/// Sort by keys
	Keys,
	/// Sort by values
	Values,
	/// Default sorting option (dependent on view type)
	Default,
	/// No sorting
	None,
}

#[derive(FromPyObject)]
enum StringOrBool<'py> {
	String(Bound<'py, PyString>),
	Bool(bool),
}

impl<'py> FromPyObject<'py> for SortModeRaw {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		if ob.is_none() {
			return Ok(Self::None);
		}
		Ok(match ob.extract::<StringOrBool>()? {
			StringOrBool::Bool(false) => Self::None,
			StringOrBool::Bool(true) => Self::Default,
			StringOrBool::String(str) => {
				let str = str.to_str()?;
				match str {
					"keys" => Self::Keys,
					"values" => Self::Values,
					_ => return Err(PyErr::new::<PyTypeError, _>("Expected None | bool | Literal['keys', 'values']")),
				}
			}
		})
	}
	fn type_input() -> TypeInfo {
		//TODO: fixme
		TypeInfo::Class {
			module: ModuleName::Module(Cow::from("typing")),
			name: Cow::from("Literal"),
			type_vars: vec![
				// TypeInfo::Class {
				// 	module: ModuleName::Builtin,
				// 	name: Cow::from(L::NAME),
				// 	type_vars: Vec::new(),
				// },
			]
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SortMode {
	/// Sort by keys
	Keys,
	/// Sort by values
	Values,
	/// No sorting
	None,
}
impl SortMode {
	const fn from(raw: SortModeRaw, default: SortMode) -> Self {
		match raw {
			SortModeRaw::Default => default,
			SortModeRaw::Keys => Self::Keys,
			SortModeRaw::Values => Self::Values,
			SortModeRaw::None => Self::None,
		}
	}
}

#[pymethods]
impl Bow {
	/// True if not empty
	fn __bool__(&self) -> bool {
		self.len() != 0
	}

	/// Return number of elements in Bow
	fn __len__(&self) -> usize {
		self.len()
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	/// Test if `key` is a valid key
	fn __contains__(&self, key: u32) -> bool {
		self.as_ref().contains_key(&key)
	}

	/*/// Get keys
	#[pyo3(signature = (sorted = false, numpy = false))]
	fn keys<'py>(&self, py: Python<'py>, sorted: bool, numpy: bool) -> Either<Vec<u32>, Bound<'py, PyArray1<u32>>> {
		let mut result = self.as_ref()
			.keys()
			.copied()
			.collect::<Vec<_>>();
		if sorted {
			// Keys should be uniqe, so we shouldn't need to worry about stability
			result.sort_unstable();
		}
		
		if numpy {
			Either::Right(PyArray1::from_vec(py, result))
		} else {
			Either::Left(result)
		}
	}

	/// Get values
	#[pyo3(signature = (sorted = None, numpy = false))]
	fn values<'py>(&self, py: Python<'py>, sorted: Option<SortMode>, numpy: bool) -> Either<Vec<f32>, Bound<'py, PyArray1<f32>>> {
		let hm = self.as_ref();
		let result = py.allow_threads(|| {
			if let Some(SortMode::Keys(..)) = sorted {
				let mut result = hm
					.iter()
					// Copy here because sizeof(u32) <= sizeof(usize)
					.map(|(key, value)| (*key, *value))
					.collect::<Vec<_>>();
				// Keys should be uniqe, so we shouldn't need to worry about stability
				result.sort_unstable_by_key(|(key, _)| *key);
				result.into_iter()
					.map(|(_, value)| value)
					.collect::<Vec<_>>()
			} else {
				let mut result = hm
					.values()
					.copied()
					.collect::<Vec<_>>();

				if let Some(SortMode::Bool(true)) | Some(SortMode::Values(..)) = sorted {
					result.sort_by(f32::total_cmp);
				}
				result
			}
		});

		if numpy {
			Either::Right(PyArray1::from_vec(py, result))
		} else {
			Either::Left(result)
		}
		
	}

	/// Get list of items
	#[pyo3(signature = (sorted = false))]
	fn items(&self, sorted: bool) -> Vec<(u32, f32)> {
		let mut result = self.iter().collect::<Vec<_>>();
		if sorted {
			// Keys should be uniqe, so we shouldn't need to worry about stability
			result.sort_unstable_by_key(|v| v.0);
		}
		result
	}*/

	/// Get item
	fn __getitem__(&self, key: u32) -> PyResult<f32> {
		match self.as_ref().get(&key) {
			Some(value) => Ok(*value),
			None => Err(PyErr::new::<PyKeyError, _>(key))
		}
	}

	/// Get item
	fn get<'py>(&self, key: u32, default: Bound<'py, PyAny>) -> Either<f32, Bound<'py, PyAny>> {
		match self.as_ref().get(&key) {
			Some(value) => Either::Left(*value),
			None => Either::Right(default),
		}
	}

	/// Compute the score between this and some other bag of words
	#[pyo3(name="score", signature = (other, metric = Scoring::L2))]
	fn py_score(&self, py: Python<'_>, other: &Bow, metric: Scoring) -> f64 {
		py.allow_threads(|| {
			self.score(other, metric)
		})
	}

	/// Compute the Ln norm of all the scores
	#[pyo3(name="norm", signature = (norm = LNorm::L2))]
	fn py_norm(&self, py: Python<'_>, norm: LNorm) -> f64 {
		py.allow_threads(|| self.norm(norm))
	}

	/// Normalize such that `fbow.normalize().norm() == 1.0`
	#[pyo3(name="normalize", signature = (norm = LNorm::L2))]
	fn py_normalize<'py>(me: Bound<'py, Self>, py: Python<'py>, norm: LNorm) -> PyResult<Bound<'py, Self>> {
		let this = me.get();

		let r = py.allow_threads(|| {
			let norm = this.norm(norm);
			if (norm - 1.) < 1e-8 {
				// Identity
				Ok(None)
			} else if norm <= 0. {
				// Error is filled in later
				Err(())
			} else {
				//TODO: is it faster to do HashMap insert and scale in the same loop?
				let mut result = this.clone();
				result.scale(norm.recip() as f32);
				Ok(Some(result))
			}
		});

		match r {
			Err(..) => Err(PyErr::new::<PyZeroDivisionError, _>("norm=0")),
			Ok(None) => Ok(me),
			Ok(Some(r)) => Bound::new(py, r),
		}
	}

	/// Keys view
	fn keys(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Bow(this), mode: ViewMode::Keys, reversed: false }
	}

	/// Values view
	fn values(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Bow(this), mode: ViewMode::Values, reversed: false }
	}

	/// Items view
	fn items(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Bow(this), mode: ViewMode::Items, reversed: false }
	}

	/// Convert to native Python dict
	fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
		self
			.iter()
			.into_py_dict(py)
	}

	/// Hash value
	fn __hash__(&self) -> u64 {
		self.hash()
	}
}

#[pyclass]
struct IntersectFeatures {
	node_ids: Vec<u32>,
	features: Vec<Py<Features>>,
}

impl IntersectFeatures {
	fn intersect<'py>(py: Python<'py>, a: FeaturesLike<'py>, b: FeaturesLike<'py>) -> PyResult<Self> {
		enum Raw<'a> {
			Features(&'a Features),
			Intersect(&'a IntersectFeatures),
		}
		impl<'a> Raw<'a> {
			fn num_refs(&self) -> usize {
				match self {
					Self::Features(..) => 1,
					Self::Intersect(f) => f.features.len(),
				}
			}
		}

		let pyr_a; // Keep PyRef's in scope
		let pyr_b;
		let a_ref = match &a {
			FeaturesLike::Features(f) => Raw::Features(f.get()),
			FeaturesLike::Intersection(f) => {
				pyr_a = f.try_borrow()?;
				Raw::Intersect(pyr_a.deref())
			},
		};
		let b_ref = match &b {
			FeaturesLike::Features(f) => Raw::Features(f.get()),
			FeaturesLike::Intersection(f) => {
				pyr_b = f.try_borrow()?;
				Raw::Intersect(pyr_b.deref())
			},
		};

		let (node_ids, cap_count) = py.allow_threads(|| {
			let keys = match (&a_ref, &b_ref) {
				(Raw::Features(a_ref), Raw::Features(b_ref)) => {
					// Zip keys
					let u = a_ref.as_ref();
					let v = b_ref.as_ref();

					// Iterate over smaller map
					let (smol, big) = if u.capacity() + u.len() <= v.capacity() + v.len() {
						(u, v)
					} else {
						(v, u)
					};
					let keys = smol.keys()
						.copied()
						.filter(|key| big.contains_key(key))
						.collect::<Vec<_>>();

					keys
				},
				_ => {
					todo!()
				},
			};
			let cap_count = a_ref.num_refs() + b_ref.num_refs();
			(keys, cap_count)
		});

		let mut features = Vec::with_capacity(cap_count);
		match a_ref {
			Raw::Features(..) => {
				let FeaturesLike::Features(f) = a else { unreachable!() };
				features.push(f.unbind());
			},
			Raw::Intersect(f) => {
				features.extend(f.features.iter().map(|f| f.clone_ref(py)));
			},
		}
		match b_ref {
			Raw::Features(..) => {
				let FeaturesLike::Features(f) = b else { unreachable!() };
				features.push(f.unbind());
			},
			Raw::Intersect(f) => {
				features.extend(f.features.iter().map(|f| f.clone_ref(py)));
			},
		}
		Ok(IntersectFeatures {
			node_ids,
			features,
		})
	}
}

#[pymethods]
impl IntersectFeatures {
	fn __bool__(&self) -> bool {
		!self.node_ids.is_empty()
	}
	fn __len__(&self) -> usize {
		self.node_ids.len()
	}
	fn keys(&self) -> &[u32] {
		&self.node_ids
	}
	fn values(&self, py: Python<'_>) -> Vec<Vec<Vec<u32>>> {
		if self.node_ids.is_empty() || self.features.is_empty() {
			return Vec::new();
		}

		let features = self.features.iter()
			.map(|f| f.bind(py).get().as_ref())
			.collect::<Vec<_>>();
		py.allow_threads(|| {
			self.node_ids
				.iter()
				.map(|node_id| {
					features.iter()
						.map(|feature| feature[node_id].to_vec())
						.collect::<Vec<_>>()
				})
				.collect::<Vec<_>>()
		})
	}
	fn __iter__(&self) -> () {
		todo!();
	}
	fn __and__<'py>(this: Bound<'py, Self>, py: Python<'py>, other: FeaturesLike<'py>) -> PyResult<IntersectFeatures> {
		IntersectFeatures::intersect(py, FeaturesLike::Intersection(this), other)
	}
	fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
		for feature in &self.features {
			visit.call(feature)?;
		}
		Ok(())
	}
	fn __clear__(&mut self) {
		self.features.clear();
	}
}

#[derive(FromPyObject)]
enum FeaturesLike<'py> {
	Features(Bound<'py, Features>),
	Intersection(Bound<'py, IntersectFeatures>),
}

#[pymethods]
impl Features {
	fn __bool__(&self) -> bool {
		self.len() > 0
	}
	fn __len__(&self) -> usize {
		self.len()
	}

	fn __repr__(&self) -> String {
		format!("{self:?}")
	}

	fn __getitem__(&self, key: u32) -> PyResult<Vec<u32>> {
		match self.as_ref().get(&key) {
			Some(values) => Ok(values.to_vec()),
			None => Err(PyErr::new::<PyKeyError, _>(format!("{key}")))
		}
	}

	fn get<'py>(&self, key: u32, default: Bound<'py, PyAny>) -> Either<Vec<u32>, Bound<'py, PyAny>> {
		match self.as_ref().get(&key) {
			Some(values) => Either::Left(values.to_vec()),
			None => Either::Right(default),
		}
	}

	/// Test if `key` is a valid key
	fn __contains__(&self, key: u32) -> bool {
		self.as_ref().contains_key(&key)
	}

	/// Keys view
	fn keys(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Features(this), mode: ViewMode::Keys, reversed: false }
	}

	/// Values view
	fn values(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Features(this), mode: ViewMode::Values, reversed: false }
	}

	/// Items view
	fn items(this: Py<Self>) -> PyDictView {
		PyDictView { base: AnyDict::Features(this), mode: ViewMode::Items, reversed: false }
	}

	fn __and__<'py>(this: Bound<'py, Self>, py: Python<'py>, other: FeaturesLike<'py>) -> PyResult<IntersectFeatures> {
		IntersectFeatures::intersect(py, FeaturesLike::Features(this), other)
	}

	/// Convert to native Python dict
	fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
		self
			.as_ref()
			.iter()
			.into_py_dict(py)
	}

	/// Hash value
	fn __hash__(&self) -> u64 {
		self.hash()
	}
}