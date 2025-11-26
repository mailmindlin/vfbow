use std::{borrow::Cow, collections::{BinaryHeap, HashMap}, num::NonZeroUsize, sync::Arc};

use ndarray::ArrayView2;

use crate::{util::{scoring::{Bhattacharyya, ChiSquare, DotProduct, ScoringMethods, L1, L2}, serde::{read_u32ish, write_u32ish}, Scoring}, vocabulary::TransformError, vocabulary_creator::VocabElement, Bow, Deserialize, Features, Serialize, Vocabulary};

//TODO: empirically determine this
const BINARY_SEARCH_THRESHOLD: usize = 64;

// Save a few bytes by packing this enum
enum DirectIndex {
	Direct(Vec<Features>),
	/// Number of entries in file
	CountId(usize),
}
impl DirectIndex {
	fn len(&self) -> usize {
		match self {
			DirectIndex::Direct(v) => v.len(),
			DirectIndex::CountId(c) => *c,
		}
	}
	fn clear(&mut self) {
		match self {
			Self::Direct(v) => {
				v.clear();
			},
			Self::CountId(id) => {
				*id = 0;
			}
		}
	}
}

/// A single entry in [InvertedEntries]
#[derive(Clone, Copy, Debug)]
struct InvertedEntry {
	entry_id: usize,
	word_weight: f32,
}


// Store the reverse lookup
/// Invariant: all inner vectors must be sorted by [entry_id](InvertedEntry::entry_id) ascending
#[derive(Clone, Debug)]
struct InvertedEntries(Vec<InvertedEntry>);

impl FromIterator<InvertedEntry> for InvertedEntries {
	fn from_iter<T: IntoIterator<Item = InvertedEntry>>(iter: T) -> Self {
		let inner = Vec::from_iter(iter);
		//TODO: should we always check this invariant?
		#[cfg(debug_assertions)] {
			debug_assert!(inner.iter().is_sorted_by_key(|entry| entry.entry_id), "Entries must be sorted");
		}
		Self(inner)
	}
}
impl<'a> IntoIterator for &'a InvertedEntries {
	type Item = &'a InvertedEntry;

	type IntoIter = core::slice::Iter<'a, InvertedEntry>;

	fn into_iter(self) -> Self::IntoIter {
		self.0.iter()
	}
}

impl InvertedEntries {
	/// Empty constructor
	fn new() -> Self { Self(Vec::new()) }
	/// Size of inverted index
	fn len(&self) -> usize {
		self.0.len()
	}
	fn get(&self, entry_id: usize) -> Option<&InvertedEntry> {
		if self.len() <= BINARY_SEARCH_THRESHOLD {
			for item in &self.0 {
				if item.entry_id == entry_id {
					return Some(item);
				} else if item.entry_id > entry_id {
					// Items are sorted, so we can exit early
					break;
				}
			}
			None
		} else {
			let idx = self.0.binary_search_by_key(&entry_id, |entry| entry.entry_id)
				.ok()?;
			Some(&self.0[idx])
		}
	}
	fn push(&mut self, entry_id: usize, word_weight: f32) {
		#[cfg(debug_assertions)] {
			if let Some(last) = self.0.last() {
				debug_assert!(last.entry_id < entry_id, "Entries must be in ascending order");
			}
		}
		self.0.push(InvertedEntry { entry_id, word_weight });
	}
	fn clear(&mut self) {
		self.0.clear();
	}
}

/// A database can be used to keep track of previously-seen features and query them
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow"))]
pub struct Database {
	/// Associated vocabulary
	vocabulary: Arc<Vocabulary>,
	/// Levels to go up the vocabulary tree to select nodes to store
	/// in the direct index
	levels: usize,
	/// Direct file (resized for allocation)
	direct: DirectIndex,
	/// Inverted file (must have size() == |words|)
	inverted: Vec<InvertedEntries>,
}

#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", get_all, set_all))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct QueryResult {
	/// Distance score
	pub score: f64,
	/// Database entry id
	pub id: usize, //TODO: make this u32?
}
impl QueryResult {
	/// Constructor
	fn new(id: usize, score: f64) -> Self {
		assert!(score.is_finite(), "Score must be finite");
		Self { id, score }
	}
}

// This is valid because of invariant
impl Eq for QueryResult {}
impl PartialOrd for QueryResult {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}
impl Ord for QueryResult {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		// Note order of arguments: sorts in reverse order
		let Some(result ) = other.score.partial_cmp(&self.score) else { unreachable!("Scores must be finite") };
		result
	}
}

impl Database {
	/// Creates a database with the given vocabulary
	/// 
	/// Parameters:
	/// - `voc` vocabulary
	/// - `use_direct` a direct index is used to store feature indexes
	/// - `levels` levels to go up the vocabulary tree to select the 
	///   node id to store in the direct index when adding images
	pub fn new(vocabulary: Arc<Vocabulary>, use_direct: bool, levels: usize) -> Self {
		let mut inverted = Vec::new();
		inverted.resize_with(vocabulary.num_features(), InvertedEntries::new);
		Self {
			vocabulary,
			levels,
			direct: if use_direct {
				DirectIndex::Direct(Vec::new())
			} else {
				DirectIndex::CountId(0)
			},
			inverted,
		}
	}

	/// Returns true if there are no entries in this database
	pub fn is_empty(&self) -> bool {
		match &self.direct {
			DirectIndex::Direct(vec) => vec.is_empty(),
			DirectIndex::CountId(c) => *c == 0,
		}
	}

	/// The number of entries
	pub fn len(&self) -> usize {
		self.direct.len()
	}

	/// The vocabulary used for this database
	pub fn vocabulary(&self) -> &Arc<Vocabulary> {
		&self.vocabulary
	}
  
	/// Checks if the direct index is being used
	pub fn using_direct_index(&self) -> bool {
		match self.direct {
			DirectIndex::CountId(..) => false,
			DirectIndex::Direct(..) => true,
		}
	}
	/// Returns the di levels when using direct index
	pub fn direct_index_levels(&self) -> Option<usize> {
		Some(self.levels)
	}

	/// Get the features for an entry
	pub fn features(&self, entry_id: usize) -> Option<&Features> {
		match &self.direct {
			DirectIndex::CountId(..) => None,
			DirectIndex::Direct(dfile) => dfile.get(entry_id),
		}
	}

	/// Clear all stored features
	/// 
	/// Doesn't mutate the vocabulary
	pub fn clear(&mut self) {
		self.direct.clear();
		for i in &mut self.inverted {
			i.clear();
		}
	}

	/// Transform some features and insert them in the database.
	/// 
	/// Returns the entry id, bag-of-words, the transformed features
	pub fn insert_transform<T: VocabElement>(&mut self, features: ArrayView2<T>) -> Result<(usize, Bow, Cow<'_, Features>), TransformError> {
		let (bow, bow2) = self.vocabulary.transform(features, Some(self.levels))?;
		let mut fv = Cow::Owned(bow2);
		let entry_id = self.insert(&bow, &mut fv);
		Ok((entry_id, bow, fv))
	}

	/// Insert some features into this database
	/// 
	/// If `fv` is a [`Cow::Owned`] a clone MAY be elided by taking it and replacing it with a reference
	pub fn insert<'a>(&'a mut self, bow: &Bow, fv: &mut Cow<'a, Features>) -> usize {
		// Update direct file
		let entry_id = match &mut self.direct {
			DirectIndex::Direct(vec) => {
				let id = vec.len();
				match fv {
					Cow::Owned(f) => {
						// We prevent cloning the features
						// Temporarily replaces it with an empty Features, but 
						vec.push(std::mem::take(f));
						*fv = Cow::Borrowed(vec.last().unwrap());
					},
					Cow::Borrowed(f) => {
						vec.push(f.clone());
					}
				}
				id
			},
			DirectIndex::CountId(next_id) => {
				let id = *next_id;
				*next_id += 1;
				id
			}
		};

		// Update inverted file
		for (word_id, word_weight) in bow.iter() {
			//TODO: should we do anything for OOB error here? Or is a panic justified?
			self.inverted[word_id as usize].push(entry_id, word_weight);
		}
		entry_id
	}
}

/// Limit results to at most `max_results`, and return them in ascending order
fn limit_results(scores: impl ExactSizeIterator<Item = (usize, f64)>, max_results: Option<NonZeroUsize>) -> Vec<QueryResult> {
	if let Some(max_results) = max_results && max_results.get() < scores.len() {
		// Take only top n
		println!("Take top {} of {}", max_results.get(), scores.len());
		// We allocate space for n+1 elements so we don't allocate when pushing
		let mut heap = BinaryHeap::with_capacity(max_results.get() + 1);
		let mut minimum = 0.; // Minimum score to consider adding to heap
		for (id, score) in scores.into_iter() {
			//TODO: handle NaN here?
			// Ignore scores that are too small
			if score <= minimum {
				continue;
			}
			heap.push(QueryResult::new(id, score));
			if heap.len() > max_results.get() {
				let min = heap.pop().unwrap();
				minimum = min.score;
			}
		}
		debug_assert!(heap.len() <= max_results.get());
		heap.into_sorted_vec()
	} else {
		println!("No limit {:?} / {}", max_results, scores.len());
		let mut results = scores
			.map(|(id, score)| QueryResult::new(id, score))
			.collect::<Vec<_>>();
		results.sort_unstable();
		results
	}
}

impl Database {
	fn query_generic<S: ScoringMethods>(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		let mut scores = HashMap::new();
		for (word_id, qvalue) in query.iter() {
			let Some(row) = self.inverted.get(word_id as usize) else {
				//TODO: maybe this is an error?
				continue;
			};
			// IFRows are sorted in ascending entry_id order
			for entry in row {
				if max_id.is_some_and(|max_id| max_id.get() <= entry.entry_id) {
					//TODO: break
					break;
				}
				let value = S::score(qvalue, entry.word_weight);
				match scores.entry(entry.entry_id) {
					std::collections::hash_map::Entry::Occupied(mut e) => {
						*e.get_mut() += value;
					},
					std::collections::hash_map::Entry::Vacant(e) => {
						e.insert(value);
					},
				}
			}
		}

		let mut results = limit_results(scores.into_iter(), max_results);

		for result in &mut results {
			result.score = S::finish(result.score);
		}
		results
	}

	fn query_l1(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		self.query_generic::<L1>(query, max_results, max_id)
	}

	fn query_l2(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		self.query_generic::<L2>(query, max_results, max_id)
	}

	fn query_chi_square(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		// In the current implementation, we suppose query is not normalized
		self.query_generic::<ChiSquare>(query, max_results, max_id)
	}
	  
	fn query_kl(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		let mut scores = HashMap::new();
		
		for (word_id, vi) in query.iter() {
			let Some(row) = self.inverted.get(word_id as usize) else { continue; };
	  
			// IFRows are sorted in ascending entry_id order
			for entry in row {
				if max_id.is_some_and(|max_id| max_id.get() <= entry.entry_id) {
					break;
				}
				
				let value = if vi != 0. && entry.word_weight != 0. {
					let vi = vi as f64;
					let wi = entry.word_weight as f64;
					vi * (vi / wi).ln()
				} else {
					0.
				};

				match scores.entry(entry.entry_id) {
					std::collections::hash_map::Entry::Occupied(mut e) => {
						*e.get_mut() += value;
					}
					std::collections::hash_map::Entry::Vacant(e) => {
						e.insert(value);
					}
				}
			}
		}

		// f64::ln() is not a const so this can't be
		#[allow(non_snake_case)]
		let LOG_EPS: f64 = f64::EPSILON.ln();
	  
		// resulting "scores" are now in [-X worst .. 0 best .. X worst]
		// but we cannot make sure which ones are better without calculating
		// the complete score
	  
		// complete scores and move to vector
		let scores = scores
			.into_iter()
			.map(|(eid, score)| {
				let value = query
					.iter()
					.filter_map(|(word_id, vi)| {
						if vi == 0. {
							return None;
						}
						self.inverted
							.get(word_id as usize)?
							.get(eid)?;
						
						let vi = vi as f64;
						Some(vi * (vi.ln() - LOG_EPS))
					})
					.sum::<f64>();
				(eid, score + value)
			});
		
		// cannot scale scores
		limit_results(scores, max_results)
	}
	
	fn query_bhattacharyya(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		// In the current implementation, we suppose query is not normalized
		self.query_generic::<Bhattacharyya>(query, max_results, max_id)
	}
	/*void Database::queryBhattacharyya(
		const BowVector &vec, QueryResults &ret, int max_results, int max_id) const
	  {
		BowVector::const_iterator vit;
	  
		//map<EntryId, double> pairs;
		//map<EntryId, double>::iterator pit;
	  
		std::map<EntryId, std::pair<double, int> > pairs; // <eid, <score, counter> >
		std::map<EntryId, std::pair<double, int> >::iterator pit;
	  
		for(vit = vec.begin(); vit != vec.end(); ++vit)
		{
		  const WordId word_id = vit->first;
		  const WordValue& qvalue = vit->second;
	  
		  const IFRow& row = m_ifile[word_id];
	  
		  // IFRows are sorted in ascending entry_id order
	  
		  for(auto rit = row.begin(); rit != row.end(); ++rit)
		  {
			const EntryId entry_id = rit->entry_id;
			const WordValue& dvalue = rit->word_weight;
	  
			if((int)entry_id < max_id || max_id == -1)
			{
			  double value = sqrt(qvalue * dvalue);
	  
			  pit = pairs.lower_bound(entry_id);
			  if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
			  {
				pit->second.first += value;
				pit->second.second += 1;
			  }
			  else
			  {
				pairs.insert(pit,
				  std::map<EntryId, std::pair<double, int> >::value_type(entry_id,
					std::make_pair(value, 1)));
			  }
			}
	  
		  } // for each inverted row
		} // for each query word
	  
		// move to vector
		ret.reserve(pairs.size());
		for(pit = pairs.begin(); pit != pairs.end(); ++pit)
		{
		  if(pit->second.second >= MIN_COMMON_WORDS)
		  {
			ret.push_back(Result(pit->first, pit->second.first));
			ret.back().nWords = pit->second.second;
			ret.back().bhatScore = pit->second.first;
		  }
		}
	  
		// scores are already in [0..1]
	  
		// sort vector in descending order
		std::sort(ret.begin(), ret.end(), Result::gt);
	  
		// cut vector
		if(max_results > 0 && (int)ret.size() > max_results)
		  ret.resize(max_results);
	}*/
	
	fn query_dot(&self, query: &Bow, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		// double value;
		// if(this->m_voc->getWeightingType() == BINARY)
		//   value = 1;
		// else
		//   value = qvalue * dvalue;
		self.query_generic::<DotProduct>(query, max_results, max_id)
	}

	pub fn query(&self, query: &Bow, scoring: Scoring, mut max_results: Option<NonZeroUsize>, mut max_id: Option<NonZeroUsize>) -> Vec<QueryResult> {
		let len = self.len();
		if len == 0 {
			return vec![];
		}
		// Clear filters if they're inoperable
		if max_results.as_ref().is_some_and(|r| len < r.get()) {
			println!("Clear limit max_results");
			max_results = None;
		}
		if max_id.as_ref().is_some_and(|r| len < r.get()) {
			println!("Clear limit max_id");
			max_id = None;
		}
		match scoring {
			Scoring::L1 => self.query_l1(query, max_results, max_id),
			Scoring::L2 => self.query_l2(query, max_results, max_id),
			Scoring::ChiSquare => self.query_chi_square(query, max_results, max_id),
			Scoring::Bhattacharyya => self.query_bhattacharyya(query, max_results, max_id),
			Scoring::KL => self.query_kl(query, max_results, max_id),
			Scoring::DotProduct => self.query_dot(query, max_results, max_id),
		}
	}

	pub fn query_transform<T: VocabElement>(&self, features: ArrayView2<T>, level: Option<usize>, scoring: Scoring, max_results: Option<NonZeroUsize>, max_id: Option<NonZeroUsize>) -> Result<Vec<QueryResult>, TransformError> {
		let (query, _) = self.vocabulary.transform(features, level)?;
		Ok(self.query(&query, scoring, max_results, max_id))
	}
}

/// Magic number to validate serialization version
const DB_SER_VERSION: u64 = 0x8a6f8867c2a89411;

impl Serialize for Database {
	fn write_to(&self, mut dst: impl std::io::Write) -> std::io::Result<()> {
		// Write header just in case we want to randomly read in the future
		dst.write_all(&DB_SER_VERSION.to_le_bytes())?;
		//TODO: allow sizeof(usize) > sizeof(u32)
		write_u32ish(self.len(), &mut dst)?;
		write_u32ish(self.levels, &mut dst)?;
		write_u32ish(self.vocabulary.num_features(), &mut dst)?; // Redundant, but it needs to be part of the header
		//TODO: preserve alignment?
		dst.write_all(&[if self.using_direct_index() { 1 } else { 0 }])?;
		for row in &self.inverted {
			write_u32ish(row.len(), &mut dst)?;
		}
		// Now write inverted index
		//TODO: bulk writes?
		for entry in self.inverted.iter().flatten() {
			write_u32ish(entry.entry_id, &mut dst)?;
			dst.write_all(&entry.word_weight.to_le_bytes())?;
		}
		// Now write direct index
		if let DirectIndex::Direct(direct) = &self.direct {
			for feat in direct {
				feat.write_to(&mut dst)?;
			}
		}
		// Now write vocabulary
		self.vocabulary.write_to(dst)
	}
}

impl Deserialize for Database {
	fn read_from(mut src: impl std::io::Read) -> std::io::Result<Self> {
		{
			let mut buf = [0; size_of::<usize>()];
			src.read_exact(&mut buf)?;
			if u64::from_le_bytes(buf) != DB_SER_VERSION {
				return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Database header mismatch"));
			}
		}
		let len = read_u32ish(&mut src)?;
		let levels = read_u32ish(&mut src)?;
		let num_features = read_u32ish(&mut src)?;
		let direct_index = {
			let mut buf = [0u8];
			src.read_exact(&mut buf)?;
			match buf[0] {
				0 => false,
				1 => true,
				_ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unexpected boolean value"))
			}
		};
		let inverted = {
			// Read lengths of inverted indices
			let inv_lengths = (0..num_features)
				.map(|_| read_u32ish(&mut src))
				.collect::<Result<Vec<_>, _>>()?;
			inv_lengths.into_iter()
				.map(|len| {
					//TODO: bulk reads
					(0..len)
						.map(|_| {
							let entry_id = read_u32ish(&mut src)?;
							let mut v_buf = [0; size_of::<f32>()];
							src.read_exact(&mut v_buf)?;
							let word_weight = f32::from_le_bytes(v_buf);
							Ok::<_, std::io::Error>(InvertedEntry { entry_id, word_weight })
						})
						.collect::<Result<InvertedEntries, _>>()
				})
				.collect::<Result<Vec<_>, _>>()?
		};
		let direct = if direct_index {
			let feats = (0..len)
				.map(|_| Features::read_from(&mut src))
				.collect::<Result<Vec<_>, _>>()?;
			DirectIndex::Direct(feats)
		} else {
			DirectIndex::CountId(len)
		};
		let vocabulary = Vocabulary::read_from(src, Default::default())?;
		Ok(Self {
			vocabulary: Arc::new(vocabulary),
			levels,
			direct,
			inverted,
		})
	}
}