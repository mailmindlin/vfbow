//! Vocabulary data structure
mod builder;
mod serde;
mod node;

use std::{ffi::CStr, fmt::Debug, io::{self, ErrorKind, Read, Write}, str::FromStr, time::Instant};

use arrayvec::ArrayString;
pub use node::NodePath;

use crate::{fbow::{Bow, Features}, features::{DistanceQuery, FeatureType, FeaturesGeneric}, util::{DescriptorType, Deserialize, Serialize, serde::{read_u32, read_u32ish, read_u64, write_u32, write_u64, write_u32ish}}};
pub(crate) use builder::VocabularyBuilder;
pub use serde::{ParseValidationMode, VocabularyReadOptions};

/// A node in the vocabulary tree
pub(crate) struct Node {
	/// Node ID (feature index)
	base: u32,
	/// Number of children (branches + leaves)
	n: u32,
	/// Child branches
	children: Option<Box<[Node]>>,
}

impl Node {
	/// Create empty node
	const fn empty() -> Self {
		Self {
			base: 0,
			n: 0,
			children: None,
		}
	}

	/// Number of children nodes
	fn num_children(&self) -> usize {
		match &self.children {
			None => 0,
			Some(children) => children.len(),
		}
	}
}

/// Vocabulary parameters
#[derive(Debug)]
pub(crate) struct VocabularyParams {
	/// Descriptor name. May be empty
	desc_name: ArrayString<49>, // 49 bytes + null terminator
	/// Memory alignment of each feature
	alignment: usize,
	/// Total number of blocks
	nblocks: u32,
	total_size: u64,
	/// Descriptor type
	desc_type: DescriptorType,
	/// Descriptor size
	desc_size: usize,
	/// Number of children per node
	m_k: u32,
}

impl VocabularyParams {
	// const SER_LEN: usize = 50 + (3 * size_of::<u32>()) + (5 * size_of::<u64>());
	/// Create empty params
	pub(crate) const fn empty() -> Self {
		Self {
			desc_name: ArrayString::new_const(),
			alignment: 0,
			nblocks: 0,
			total_size: 0,
			desc_type: DescriptorType::Uint8,
			desc_size: 0,
			m_k: 0,
		}
	}

	/// Set vocabulary parameters
	pub(crate) fn set(&mut self, aligment: usize, k: u32, desc_type: DescriptorType, desc_size: usize, nblocks: u32, desc_name: &str) {
		self.set_name(desc_name);

		self.alignment = aligment;
		self.m_k = k;
		self.desc_type = desc_type;
		self.nblocks = nblocks;
		self.desc_size = desc_size;
		/*
		let desc_size_bytes_al: u64 = 0;
		let block_size_bytes_al: u64 = 0;
	
		//consider possible aligment of each descriptor adding offsets at the end
		self.params.desc_size_bytes_wp = self.params.desc_size;
		_desc_size_bytes_al= _params._desc_size_bytes_wp/ _params._aligment;
		if( _params._desc_size_bytes_wp% _params._aligment!=0)   _desc_size_bytes_al++;
		_params._desc_size_bytes_wp= _desc_size_bytes_al* _params._aligment;
	
	
		let foffnbytes_alg = sizeof(uint64_t)/_params._aligment;
		if(sizeof(uint64_t)%_params._aligment!=0) foffnbytes_alg++;
		_params._feature_off_start=foffnbytes_alg*_params._aligment;
		_params._child_off_start=_params._feature_off_start+_params._m_k*_params._desc_size_bytes_wp ;//where do children information start from the start of the block
	
		//block: nvalid|f0 f1 .. fn|ni0 ni1 ..nin
		_params._block_size_bytes_wp=_params._feature_off_start+  _params._m_k * ( _params._desc_size_bytes_wp + sizeof(Vocabulary::block_node_info));
		_block_size_bytes_al=_params._block_size_bytes_wp/_params._aligment;
		if (_params._block_size_bytes_wp%_params._aligment!=0) _block_size_bytes_al++;
		_params._block_size_bytes_wp= _block_size_bytes_al*_params._aligment;
	
		//give memory
		_params._total_size=_params._block_size_bytes_wp*_params._nblocks;
		_data = std::unique_ptr<char[], decltype(&AlignedFree)>((char*)AlignedAlloc(_params._aligment, _params._total_size), &AlignedFree);
	
		memset(_data.get(), 0, _params._total_size);*/
	
	}

	/// Set desccriptor name
	/// 
	/// Because it's statically allocated, we truncate the name to the first 49 bytes of unicode
	pub(crate) fn set_name(&mut self, name: &str) {
		self.desc_name.clear();
		let capacity = self.desc_name.capacity();

		let name = if name.len() > capacity {
			// Truncate name
			let len = name.floor_char_boundary(self.desc_name.capacity());
			&name[..len]
		} else {
			name
		};
		self.desc_name.push_str(name);
	}
}

impl Serialize for VocabularyParams {
	fn write_to(&self, mut dst: impl Write) -> std::io::Result<()> {
		{
			// First 50 bytes are name (implicit null terminator)
			let mut desc_bytes = [0u8; 50];
			desc_bytes[..self.desc_name.len()].copy_from_slice(self.desc_name.as_bytes());
			dst.write_all(&desc_bytes)?;
		}
		write_u32ish(self.alignment, &mut dst)?;
		write_u32(self.nblocks, &mut dst)?;
		// dst.write_all(&self.desc_size_bytes_wp.to_le_bytes())?;
		// dst.write_all(&self.block_size_bytes_wp.to_le_bytes())?;
		// dst.write_all(&self.feature_off_start.to_le_bytes())?;
		// dst.write_all(&self.child_off_start.to_le_bytes())?;
		write_u64(self.total_size, &mut dst)?;
		write_u32(self.desc_type.into(), &mut dst)?;
		write_u32ish(self.desc_size, &mut dst)?;
		write_u32(self.m_k, &mut dst)?;
		Ok(())
	}
}

impl Deserialize for VocabularyParams {
	fn read_from(mut src: impl Read) -> std::io::Result<Self> {
		let desc_name = {
			// We ensure there's at least one null terminator
			let mut desc_bytes = [0u8; 51];
			src.read_exact(&mut desc_bytes[..50])?;
			let cstr = CStr::from_bytes_until_nul(&desc_bytes).unwrap();
			let str = cstr.to_str()
				.map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
			ArrayString::from_str(str).unwrap() // I don't think we can overflow at this point
		};
		let alignment = read_u32ish(&mut src)?;
		let nblocks = read_u32(&mut src)?;
		// let desc_size_bytes_wp = read_u64(&mut src)?;
		// let block_size_bytes_wp = read_u64(&mut src)?;
		// let feature_off_start = read_u64(&mut src)?;
		// let child_off_start = read_u64(&mut src)?;
		let total_size = read_u64(&mut src)?;
		let desc_type = read_u32(&mut src)?.try_into()?;
		let desc_size = read_u32ish(&mut src)?;
		let m_k = read_u32(&mut src)?;

		Ok(Self {
			desc_name,
			alignment,
			nblocks,
			// desc_size_bytes_wp,
			// block_size_bytes_wp,
			// feature_off_start,
			// child_off_start,
			total_size,
			desc_type,
			desc_size,
			m_k,
		})
	}
}

/// Main class to represent a vocabulary of visual words
pub struct Vocabulary {
	/// Vocabulary parameters
	params: VocabularyParams,
	/// Root node
	root: Node,
	/// Features data
	features: FeaturesGeneric,
}

impl Debug for Vocabulary {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Vocabulary")
			.field("params", &self.params)
			// .field("root", &self.root)
			.field("features", &self.features)
			.finish()
	}
}

/// Error during feature transformation
/// 
/// See [Vocabulary::transform], [Vocabulary::transform_one]
#[derive(Clone, Debug, thiserror::Error)]
pub enum TransformError {
	/// No input data
	#[error("No input data")]
	NoInputData,
	/// Feature size does not match vocabulary
	#[error("Transform features are of different size ({feature_len}) than the vocabulary ones ({vocab_flen})")]
	SizeMismatch {
		/// Vocabulary feature length
		vocab_flen: usize,
		/// Input feature length
		feature_len: usize,
	},
	/// Feature type does not match vocabulary (e.g., u8 vs f32)
	#[error("Transform features are of different type than the vocabulary ones")]
	DTypeMismatch,
}

impl Vocabulary {
	/// Print tree structure to stdout
	pub fn print_tree(&self) {
		let mut stack = vec![vec![&self.root]];
		while let Some((current, s_prev)) = stack.split_last_mut() {
			let Some(top) = current.pop() else {
				// Pop empty level
				stack.pop();
				continue;
			};
			// Ignore first level
			for level in s_prev.iter().skip(1) {
				if level.is_empty() {
					print!(" ");
				} else {
					print!("│");
				}
			}

			if !s_prev.is_empty() {
				// Pick last line-drawing character
				if current.is_empty() {
					// Last in level
					print!("┗━ ");
				} else if top.children.is_none() {
					// Leaf
					print!("┣━ ");
				} else {
					// Branch
					print!("┡━ ");
				}
			}

			let num_children = top.num_children();
			println!("id {} ({} + {})", top.base, num_children, top.n as usize - num_children);
			if let Some(children) = top.children.as_ref() {
				let children = children.iter()
					.rev()
					.collect::<Vec<_>>();
				stack.push(children);
			}
		}
	}

	/// The descriptor name
	pub fn desc_name(&self) -> &str {
		&self.params.desc_name
	}

	/// Number of features
	pub fn num_features(&self) -> usize {
		self.features.len()
	}

	pub(crate) fn features(&self) -> &FeaturesGeneric {
		&self.features
	}

	/// Returns the descriptor type
	pub fn desc_type(&self) -> DescriptorType {
	    self.params.desc_type
	}
	/// Returns desc size in bytes or 0 if not set
	pub fn desc_size(&self) -> usize {
	    self.params.desc_size
	}

	/// Returns the branching factor (number of children per node)
	pub fn k(&self) -> u32 {
		self.params.m_k
	}

	/// Total number of blocks
	pub fn size(&self) -> u32 {
		self.params.nblocks
	}

	/// Transform a single feature, returning the node
	#[allow(private_bounds)]
	pub fn transform_one<'a: 'b, 'b, T: FeatureType>(&'a self, feature: ndarray::ArrayView1<'b, T>, max_level: Option<usize>) -> Result<NodePath<'a>, TransformError> {
		if feature.len() != self.params.desc_size {
			return Err(TransformError::SizeMismatch {
				feature_len: feature.len(),
				vocab_flen: self.params.desc_size,
			});
		}

		//TODO: maybe let features convert it?
		let mut path = Vec::with_capacity(max_level.unwrap_or(0));

		let q = self.features.query(feature);
		let mut block = &self.root;

		let mut cur_level = 0;//current level of recursion
		//copy to another structure and add padding with zeros
		let child_offset = loop {
			if max_level == Some(cur_level) {
				// if reached level,save
				break None;
			}

			// Find node with minimum distance
			//given the current block, finds the node with minimum distance
			let child_idx = q.min_index(block.base as _, block.n as _) as u32;

			assert!(child_idx < block.n);

			if let Some(children) = block.children.as_ref() && ((child_idx as usize) < children.len()) {
				// Child is a branch
				path.push(block);
				block = &children[child_idx as usize];
				cur_level += 1;
			} else {
				// Child is a leaf
				break Some(child_idx);
			}
		};
		Ok(NodePath::new(self, path, child_offset))
	}

	/// Transform multiple features, returning a Bag-of-Words and feature nodes
	#[allow(private_bounds)]
	pub fn transform<T: FeatureType>(&self, features: ndarray::ArrayView2<T>, max_level: Option<usize>) -> Result<(Bow, Features), TransformError> {
		if features.nrows() == 0 {
			return Err(TransformError::NoInputData);
		}
		if features.ncols() != self.params.desc_size {
			return Err(TransformError::SizeMismatch {
				feature_len: features.ncols(),
				vocab_flen: self.params.desc_size,
			});
		}

		let mut r = Bow::with_capacity(features.nrows());
		let mut r2 = Features::with_capacity(features.nrows());

		let start = Instant::now();

		//TODO: maybe let features convert it?

		for (idx, row) in features.rows().into_iter().enumerate() {
			// println!("Transform row {idx}");
			let q = self.features.query(row);
			let mut block = &self.root;
			let mut cur_level = 0;//current level of recursion
			//copy to another structure and add padding with zeros
			loop {
				// Find node with minimum distance
				//given the current block, finds the node with minimum distance
				let child_idx = q.min_index(block.base as _, block.n as _) as u32;
				if max_level == Some(cur_level) {
					// if reached level,save
					r2.insert(block.base, idx as _);
				}

				assert!(child_idx < block.n);

				if let Some(children) = block.children.as_ref() && ((child_idx as usize) < children.len()) {
					// println!("\tRecurse child {}", child_idx);
					// Child is a branch
					block = &children[child_idx as usize];
					cur_level += 1;
				} else {
					// println!("Found leaf {}", child_idx);
					// Child is a leaf -> add weight
					r.update(block.base + child_idx, 1.0);
					if max_level.is_none_or(|level| cur_level < level) {
						// store level not reached, save now
						r2.insert(block.base, idx as _);
					}
					break;
				}
			}
		}

		let dt = start.elapsed();
		#[cfg(debug_assertions)] // Don't print in release
		println!("Transform {} rows in {}ms ({}ns/row)", features.len(), dt.as_millis_f32(), (dt.as_nanos() as f64 / (features.len() as f64)));

		Ok((r, r2))
	}
}

/*impl SelfHash for Vocabulary {
	fn hash(&self) -> u64 {
		let mut seed = 0;
		for i in 0..self.params.total_size {
			seed ^= (self.data[i as usize] as u64) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		seed
	}
}*/