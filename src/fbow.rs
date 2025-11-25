
use std::{collections::{hash_map::Entry, HashMap}, fmt::Debug, hash::Hash, io, iter::FusedIterator};

use crate::util::{Deserialize, Scoring, SelfHash, Serialize, convert::convert_le, scoring::{LNorm, ScoringMethods}, serde::{read_u32, read_u32ish, write_u32, write_u32ish}};

/// Iterate over the shared keys of two HashMaps.
/// 
/// Full iteration is O(min(cap_a + len_a, cap_b + len_b))
struct ZipValues<'a, K, V> {
	items: std::collections::hash_map::Iter<'a, K, V>,
	lookup: &'a HashMap<K, V>,
}

impl<'a, K, V> ZipValues<'a, K, V> {
	fn new(a: &'a HashMap<K, V>, b: &'a HashMap<K, V>) -> Self {
		// Iterate over smaller map
		let (smol, big) = if a.capacity() + a.len() <= b.capacity() + b.len() {
			(a, b)
		} else {
			(b, a)
		};
		debug_assert!(smol.capacity() <= big.capacity());
		Self {
			items: smol.iter(),
			lookup: big,
		}
	}
}

impl<'a, K: Eq + Hash, V: Copy> Iterator for ZipValues<'a, K, V> {
	type Item = (V, V);

	fn size_hint(&self) -> (usize, Option<usize>) {
		let (_low, high) = self.items.size_hint();
		(0, high)
	}

	fn next(&mut self) -> Option<Self::Item> {
		while let Some((key, &value1)) = self.items.next() {
			if let Some(&value2) = self.lookup.get(key) {
				return Some((value1, value2));
			}
		}
		None
	}

	fn fold<B, F>(self, init: B, mut f: F) -> B where Self: Sized, F: FnMut(B, Self::Item) -> B, {
		// Iter specializes fold, so we might as well too
		self.items.fold(init, |acc, (key, &value1)| {
			match self.lookup.get(key) {
				Some(&value2) => f(acc, (value1, value2)),
				None => acc,
			}
		})
	}
}

fn values_left<'a, K: Eq + Hash, V: Copy>(a: &'a HashMap<K, V>, b: &'a HashMap<K, V>, filter: impl Fn(V) -> bool, mut f: impl FnMut(V, Option<V>)) {
	for (key, &value1) in a.iter() {
		if !filter(value1) {
			continue;
		}
		
		let value2 = b.get(key).copied();
		f(value1, value2);
	}
}

/// Bag of words
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen, module="vfbow", extends=pyo3::types::PyDict))]
#[derive(Clone, Debug, PartialEq)]
pub struct Bow(HashMap<u32, f32>);

impl Bow {
	pub fn with_capacity(capacity: usize) -> Self {
		Self(HashMap::with_capacity(capacity))
	}

	/// Number of items in this bag of words
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// Clear bag
	pub fn clear(&mut self) {
		self.0.clear();
	}

	/// Iterate over items
	pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + ExactSizeIterator + FusedIterator + Debug + Clone + '_ {
		self.0.iter()
			.map(|(k, v)| (*k, *v))
	}

	/// Remove key
	pub fn remove(&mut self, key: u32) -> Option<f32> {
		self.0.remove(&key)
	}

	/// Add weight to key
	pub fn update(&mut self, key: u32, weight: f32) {
		match self.0.entry(key) {
			Entry::Occupied(mut entry) => {
				*entry.get_mut() += weight;
			},
			Entry::Vacant(entry) => {
				entry.insert(weight);
			}
		}
	}

	fn zip<'a>(&'a self, other: &'a Self) -> ZipValues<'a, u32, f32> {
		ZipValues::new(&self.0, &other.0)
	}

	/// Returns the similitude score between to image descriptors
	pub fn score(&self, other: &Self, metric: impl Into<Scoring>) -> f64 {
		match metric.into() {
			Scoring::L1 => self.score_l1(other),
			Scoring::L2 => self.score_l2(other),
			Scoring::ChiSquare => self.score_chi_squared(other),
			Scoring::KL => self.score_kl(other),
			Scoring::Bhattacharyya => self.score_battacharyya(other),
			Scoring::DotProduct => self.score_dot(other),
		}
	}

	/// Compute norm
	pub fn norm(&self, norm: LNorm) -> f64 {
		 match norm {
			LNorm::L1 => self.norm_l1(),
			LNorm::L2 => self.norm_l2(),
		}
	}

	/// Compute L1 norm
	pub fn norm_l1(&self) -> f64 {
		self.0.values()
			.fold(0., |acc, &v| acc + v.abs() as f64)
	}

	/// Compute L2 norm
	pub fn norm_l2(&self) -> f64 {
		self.0.values()
			.fold(0., |acc, &v| acc + ((v * v) as f64).sqrt())
	}

	/// Scale all scores by `scalar`
	pub fn scale(&mut self, scalar: f32) {
		for value in self.0.values_mut() {
			*value *= scalar;
		}
	}

	fn score_generic<S: ScoringMethods>(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.map(|(u, v)| S::score(u, v))
			.sum::<f64>();
		S::finish(score)
	}

	/// Compute L1 score
	/// 
	/// Returns score in range [0..1]
	pub fn score_l1(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.map(|(v1, v2)| ((v1 - v2).abs() - v1.abs() - v2.abs()) as f64)
			.sum::<f64>();
		// ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
		//		for all i | v_i != 0 and w_i != 0 
		// (Nister, 2006)
		// scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
		let score = -score / 2.0;

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}

	/// Compute L2 score
	/// 
	/// Returns score in range [0..1]
	pub fn score_l2(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.map(|(v1, v2)| (v1 * v2) as f64)
			.sum::<f64>();

		// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
		//		for all i | v_i != 0 and w_i != 0 )
		// (Nister, 2006)
		let score = if score >= 1. { // rounding errors
			1.
		} else {
			1. - (1. - score).sqrt() // [0..1]
		};

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}

	pub fn score_chi_squared(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.fold(0., |score, (v1, v2)| {
				// (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
				// we move the -4 out
				if v1 + v2 != 0. {
					score + (v1 * v2 / (v1 + v2)) as f64
				} else {
					score
				}
			});

		// this takes the -4 into account
		let score = 2. * score; // [0..1]

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}

	pub fn score_kl(&self, other: &Self) -> f64 {
		let log_eps: f64 = f64::EPSILON.ln();

		let mut score = 0.;
		values_left(&self.0, &other.0,
			|v1| v1 != 0.,
			|v1, v2| {
				debug_assert_ne!(v1, 0.);
				match v2 {
					Some(0.) => {},
					Some(v2) => {
						score += v1 as f64 * ((v1 / v2) as f64).ln();
					},
					None => {
						let v1 = v1 as f64;
						score += v1 * (v1.ln() - log_eps);
					}
				}
			}
		);
		// Cannot be scaled
		score
	}

	pub fn score_battacharyya(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.fold(0., |score, (v1, v2)| score + ((v1 * v2) as f64).sqrt());
		score // already scaled
	}

	/// Compute dot product
	pub fn score_dot(&self, other: &Self) -> f64 {
		let score = self.zip(other)
			.fold(0., |score, (v1, v2)| score + ((v1 * v2) as f64));
		score // cannot scale
	}
}

impl AsRef<HashMap<u32, f32>> for Bow {
	fn as_ref(&self) -> &HashMap<u32, f32> {
		&self.0
	}
}

impl Serialize for Bow {
	fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
		write_u32ish(self.len(), &mut dst)?;
		let mut row_buffer = [0u8; size_of::<u32>() + size_of::<f32>()];
		for (key, value) in self.iter() {
			//TODO: is this worth it?
			row_buffer[..size_of::<u32>()].copy_from_slice(&key.to_le_bytes());
			row_buffer[size_of::<u32>()..].copy_from_slice(&value.to_le_bytes());
			dst.write_all(&row_buffer)?;
		}
		Ok(())
	}
}

impl Deserialize for Bow {
	fn read_from(mut src: impl io::Read) -> io::Result<Self> {
		let len = read_u32ish(&mut src)?;
		let mut hm = HashMap::with_capacity(len);

		for _ in 0..len {
			let key = read_u32(&mut src)?;
			let weight = f32::from_bits(read_u32(&mut src)?);
			let unique = hm.insert(key, weight).is_none();
			#[cfg(debug_assertions)]
			if !unique {
				return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Duplicate key {key}")));
			}
		}
		Ok(Self(hm))
	}
}

impl SelfHash for Bow {
	fn hash(&self) -> u64 {
		let mut seed = 0u64;
		for (key, value) in self.iter() {
			seed ^=
				(key as u64)
				+ ((value * 1000.) as u64)
				+ 0x9e3779b9
				+ (seed << 6)
				+ (seed >> 2);
		}
		seed
	}
}

/// Bag of words with augmented information
/// 
/// For each word, keeps information about the indices of the elements that have been classified into the word.
/// 
/// It is computed at the desired level
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen, module="vfbow", extends=pyo3::types::PyDict))]
#[derive(Clone, Debug, PartialEq)]
pub struct Features(HashMap<u32, Vec<u32>>);

impl Features {
	pub(crate) fn new() -> Self {
		Self(HashMap::new())
	}
	
	pub fn with_capacity(capacity: usize) -> Self {
		Self(HashMap::with_capacity(capacity))
	}

	pub(crate) fn insert(&mut self, key: u32, value: u32) {
		match self.0.entry(key) {
			std::collections::hash_map::Entry::Occupied(mut entry) => {
				entry.get_mut().push(value);
			},
			std::collections::hash_map::Entry::Vacant(entry) => {
				entry.insert(vec![value]);
			},
		}
	}
	
	pub fn len(&self) -> usize {
		self.0.len()
	}

	pub fn clear(&mut self) {
		self.0.clear();
	}
}

impl AsRef<HashMap<u32, Vec<u32>>> for Features {
	fn as_ref(&self) -> &HashMap<u32, Vec<u32>> {
		&self.0
	}
}

impl Serialize for Features {
	fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
		write_u32ish(self.len(), &mut dst)?;
		for (key, values) in self.0.iter() {
			write_u32(*key, &mut dst)?;
			// Now write values
			write_u32ish(values.len(), &mut dst)?;
			//TODO: maybe transmute to u8
			for value in values {
				write_u32(*value, &mut dst)?;
			}
		}
		Ok(())
	}
}

impl Deserialize for Features {
	fn read_from(mut src: impl io::Read) -> io::Result<Self> {
		let len = read_u32ish(&mut src)?;
		let mut result = Self::with_capacity(len);
		for _ in 0..len {
			let key = read_u32(&mut src)?;
			let values_len = read_u32ish(&mut src)?;

			// Bulk read
			//TODO: transmute from u8
			let values = {
				let mut values_bytes = vec![0u8; values_len * size_of::<u32>()];
				src.read_exact(&mut values_bytes)?;

				convert_le(&values_bytes)
					// Shouldn't be possible because valuse_bytes should be multiple of size_of::<u32>()
					.unwrap()
					.collect::<Vec<_>>()
			};
			if result.0.insert(key, values).is_some() {
				println!("Warning: duplicate key {key}");
			}
		}
		Ok(result)
	}
}

impl SelfHash for Features {
	fn hash(&self) -> u64 {
		let mut seed = 0;
		//TODO: I'm not 100% sure this is stable
		for (&key, values) in self.0.iter() {
			seed ^= (key as u64) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			for &value in values {
				seed ^= (value as u64) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
		}
		seed
	}
}