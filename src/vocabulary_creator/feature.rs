use ndarray::{Array1, Array2};

use super::CreateVocabularyError;

/// Helper type for [`FeatureInfo::finfo`]
struct FeatureIndex {
	/// Index into vector of matrices
	midx: usize,
	/// Matrix row
	fidx: usize,
}

/// Struct to acces the features as a unique vector
pub(super) struct FeatureInfo<T> {
	/// Because we're pretending we squashed the arrays in `features`, we need a quick lookup structure to resolve get(i) -> features[midx][fidx].
	/// 
	/// We do this with a lot of memory by storing a table of `i -> (midx, fidx)`
	//TODO: I bet we can save some memory (and cache misses!) by converting all of this to a binary search
	finfo: Vec<FeatureIndex>,
	/// Feature arrays
	/// 
	/// Invariant: all the elements are non-empty and have the same number of columns
	features: Vec<Array2<T>>,
}

impl<T> FeatureInfo<T> {
	/// Create from a list of 2d arrays
	/// 
	/// The input format is available for ergonomics, but is effectively squashed into a single `Array2<T>` by removing axis 0 from each element
	pub(super) fn create(mut features: Vec<Array2<T>>) -> Result<Self, CreateVocabularyError> {
		// Ignore empty arrays
		features.retain(|feature| !feature.is_empty());

		// Pick the feature size from the first one
		let desc_cols = {
			let Some(feature0) = features.first() else {
				return Err(CreateVocabularyError::NoFeatures)
			};
			let desc_cols = feature0.ncols();
			if desc_cols == 0 {
				return Err(CreateVocabularyError::EmptyFeature);
			}
			desc_cols
		};

		let size = features.iter()
			.map(|feature| feature.nrows())
			.sum();
		let mut finfo = Vec::with_capacity(size);
		for (midx, feature) in features.iter().enumerate() {
			if feature.ncols() != desc_cols {
				return Err(CreateVocabularyError::ArrayDimMismatch);
			}

			for i in 0..feature.nrows() {
				finfo.push(FeatureIndex { midx, fidx: i });
			}
		}
		Ok(Self { finfo, features })
	}
	
	/// Feature length
	pub(super) fn feature_len(&self) -> usize {
		self.features[0].ncols()
	}

	/// Total number of rows
	pub(super) fn len(&self) -> usize {
		self.finfo.len()
	}
	/// Get the n<sup>th</sup> feature
	pub(super) fn get(&self, i: usize) -> ndarray::ArrayView1<'_, T> {
		let idx = &self.finfo[i];
		self.features[idx.midx].row(idx.fidx)
	}
}

impl FeatureInfo<f32> {
	/// Compute the mean of the features specified by `indices` (specialized for [f32] features)
	pub(super) fn mean_value(&self, indices: impl ExactSizeIterator<Item = usize>) -> Array1<f32> {
		let len = indices.len();
		assert_ne!(len, 0, "Empty indices");

		let mut mean = Array1::<f32>::zeros([self.feature_len()]);
		for idx in indices {
			let feature = self.get(idx);
			mean += &feature;
		}

		mean *= (len as f32).recip();

		mean
	}
}

impl FeatureInfo<u8> {
	/// Compute the mean of the features specified by `indices` (specialized for [u8] features)
	pub(super) fn mean_value(&self, indices: impl ExactSizeIterator<Item = usize>) -> Array1<u8> {
		let num_indices = indices.len();
		let feature_len = self.feature_len();

		// Threashold
		let threshold = (num_indices / 2 + num_indices % 2).try_into().unwrap();
		
		//TODO: shrink the size of the counters for smaller descriptors (reduce memory bandwidth)
		//determine number of bytes of the binary descriptor
		let mut sum = vec![([0u32; 8], 0xFF_u8); feature_len];
		// Track which bits we care about
		let mut mean = Array1::<u8>::zeros([self.feature_len()]);

		for idx in indices {
			//TODO: we can definately speed this up with SIMD
			let feature = self.get(idx);
			for (idx, (&p, (sum, mask))) in feature.iter().zip(sum.iter_mut()).enumerate() {
				let p = p & *mask;
				//TODO: is it worth using highest_one_bit? Or is that just less optimizable?
				for bit in 0..8 {
					let bit_mask = 1u8 << bit;
					if p & bit_mask != 0 {
						sum[bit] += 1;
						if sum[bit] > threshold {
							// We've exceeded the threshold, skip this one in future calculations
							*mask &= !bit_mask;
							mean[idx] |= bit_mask;
						}
					}
				}
			}
		}
		mean
	}
}