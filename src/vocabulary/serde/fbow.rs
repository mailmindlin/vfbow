//! Compatibility for parsing FBOW vocabulary files

use std::{collections::HashMap, ffi::CStr, io::{self, ErrorKind, Read}, str::Utf8Error};

use arrayvec::ArrayString;
use ndarray::ArrayView1;

use crate::{Vocabulary, util::convert::convert_le, vocabulary::{VocabularyBuilder, VocabularyParams}};

use super::{DescriptorType, ParseValidationMode, VocabularyReadOptions};

/// FBOW parameters
#[derive(Debug)]
struct FbowParams {
	/// Descriptor name. May be empty.
	/// 
	/// FBOW descriptors *shold* be 49 bytes long + null terminator, but this field is <= 50 bytes to enable
	/// liberal parsing with [ReadFbowOptions::descriptor_not_terminated]
	desc_name: ArrayString<50>,
	/// Memory alignment of each feature
	alignment: u32,
	/// Total number of blocks
	nblocks: u32,
	
	/// Size of the descriptor(includes padding)
	desc_size_bytes_wp: u64,
	/// Size of a block   (includes padding)
	block_size_bytes_wp: u64,
	/// Within a block, where the features start
	feature_off_start: u64,
	/// Within a block,where the children offset part starts
	child_off_start: u64,
	total_size: u64,
	/// original descriptor types and sizes (without padding)
	desc_type: DescriptorType,
	// desc_type: i32,
	desc_size: u32,
	/// Number of children per node
	m_k: u32,
}

trait DeserializeFixed: Sized {
	const SIZE: usize;
	fn read(src: impl Read) -> io::Result<Self>;
}

macro_rules! impl_desf_bytesle {
	($($ty:ty)*) => {
		$(
			impl DeserializeFixed for $ty {
				const SIZE: usize = size_of::<Self>();
			
				fn read(mut src: impl Read) -> io::Result<Self> {
					let mut buf = [0u8; size_of::<Self>()];
					src.read_exact(&mut buf)?;
					Ok(Self::from_le_bytes(buf))
				}
			}
		)*
	};
}
impl_desf_bytesle!(u16 u32 u64);

macro_rules! impl_desf_tryfrom {
	($($src:ty => $dst:ty),*) => {
		$(
			impl DeserializeFixed for $dst {
				const SIZE: usize = <$src as DeserializeFixed>::SIZE;
				fn read(src: impl Read) -> io::Result<Self> {
					let raw: $src = <$src as DeserializeFixed>::read(src)?;
					match <Self as TryFrom<$src>>::try_from(raw) {
						Ok(value) => Ok(value),
						Err(e) => Err(io::Error::new(ErrorKind::InvalidData, e)),
					}
				}
			}
		)*
	};
}

impl_desf_tryfrom!(
	u32 => DescriptorType
);

fn read_padding<const N: usize>(src: &mut impl Read) -> io::Result<()> {
	let mut buf = [0u8; N];
	src.read_exact(&mut buf)?;
	assert_eq!(buf, [0u8; N]);
	Ok(())
}

//TODO: generate a vectored read version
macro_rules! parse_fields {
	{($src:expr)} => {};
	{
		($src:expr)
		padding($len:literal);
		$($tt:tt)*
	} => {
		read_padding::<$len>(&mut $src)?;
		parse_fields! {
			($src)
			$($tt)*
		}
	};
	{
		($src:expr)
		$name:ident: $ty:ty;
		$($tt:tt)*
	} => {
		let $name: $ty = <$ty as DeserializeFixed>::read(&mut $src)?;
		parse_fields! {
			($src)
			$($tt)*
		}
	};
}

pub(super) struct ReadFbowOptions {
	/// Descriptor name was not terminated with a `\0` or invalid unicode
	invalid_name: ParseValidationMode,
	/// Padding bytes were not zero
	nonzero_padding: ParseValidationMode,
	too_many_children: ParseValidationMode,
	inconsistent_block: ParseValidationMode,
	invalid_weight: ParseValidationMode,
	block_cycle: ParseValidationMode,
	invalid_child: ParseValidationMode,
	//TODO: swallow unexpected desc type?
}

impl From<VocabularyReadOptions> for ReadFbowOptions {
	fn from(value: VocabularyReadOptions) -> Self {
		Self {
			invalid_name: value.inconsistent_block,
			nonzero_padding: value.inconsistent_block,
			inconsistent_block: value.inconsistent_block,
			too_many_children: value.too_many_children,
			invalid_weight: value.inconsistent_block,
			block_cycle: value.inconsistent_block,
			invalid_child: value.inconsistent_block,
		}
	}
}

macro_rules! warning {
	($kind:expr, $fmt:literal $(, $($arg:tt)+)?) => {
		if $kind >= ParseValidationMode::Warn {
			let message = format!($fmt $(, $($arg)+)?);
			
			match $kind {
				ParseValidationMode::Strict => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, message)),
				ParseValidationMode::Warn => println!("[warn] {message}"),
				ParseValidationMode::Ignore => unreachable!(),
			}
		}
	};
	($kind:expr, $error:expr) => {
		match $kind {
			ParseValidationMode::Strict => return Err(io::Error::new(io::ErrorKind::InvalidData, $error)),
			ParseValidationMode::Warn => println!("[warn] {}", $error),
			ParseValidationMode::Ignore => {},
		}
	}
}

macro_rules! validate {
	($cond:expr, $($arg:tt)*) => {
		if !$cond {
			return Err(io::Error::new(ErrorKind::InvalidData, format!($($arg)*)));
		}
	}
}

fn parse_desc_name(desc_bytes: &[u8; 50], options: &ReadFbowOptions) -> io::Result<ArrayString<50>> {
	let handle_utf8 = |e: Utf8Error| {
		warning!(options.invalid_name, e);
		// Return empty string
		Ok(ArrayString::new())
	};
	
	match CStr::from_bytes_until_nul(desc_bytes) {
		Ok(cstr) => match cstr.to_str() {
			Ok(str) => {
				let res = ArrayString::from(str)
					.unwrap(); // I don't think this can fail
				return Ok(res);
			},
			Err(e) => handle_utf8(e),
		},
		Err(..) => {
			// No null terminator
			warning!(options.invalid_name, "Descriptor name missing null terminator");
			ArrayString::from_byte_string(desc_bytes).or_else(handle_utf8)
		}
	}
}

impl FbowParams {
	fn read_from(mut src: impl Read, options: &ReadFbowOptions) -> std::io::Result<Self> {
		let desc_name = {
			// The field is 50 
			// We ensure there's at least one null terminator
			let mut desc_bytes = [0u8; 50];
			src.read_exact(&mut desc_bytes)?;
			parse_desc_name(&desc_bytes, options)?
		};

		/*#[repr(C)]
		struct FbowParamsC {
			//descriptor name. May be empty
			desc_name: [u8; 50],
			/// Memory alignment of each feature
			alignment: u32,
			/// Total number of blocks
			nblocks: u32,
			
			/// Size of the descriptor(includes padding)
			desc_size_bytes_wp: u64,
			/// Size of a block   (includes padding)
			block_size_bytes_wp: u64,
			/// Within a block, where the features start
			feature_off_start: u64,
			/// Within a block,where the children offset part starts
			child_off_start: u64,
			total_size: u64,
			/// original descriptor types and sizes (without padding)
			desc_type: i32,
			desc_size: i32,
			/// Number of children per node
			m_k: u32,
		}*/
		
		//TODO: use offset_of! to generate this
		parse_fields! {
			(src)
			// Because of padding, there should be an extra two bytes
			padding(2);
			alignment: u32;
			nblocks: u32;
			// Because of padding, there should be an extra four bytes
			padding(4);
			desc_size_bytes_wp: u64;
			block_size_bytes_wp: u64;
			feature_off_start: u64;
			child_off_start: u64;
			total_size: u64;
			desc_type: DescriptorType;
			desc_size: u32;
			m_k: u32;
			// Final 4 bytes of padding
			padding(4);
		}
		// println!("Alignment={alignment:#04x} {alignment}");
		// println!("nblocks={nblocks:#04x} {nblocks}");
		// println!("desc_size_bytes_wp={desc_size_bytes_wp:#010x} / {desc_size_bytes_wp}");
		// println!("block_size_bytes_wp={block_size_bytes_wp:#010x} / {block_size_bytes_wp}");
		// println!("feature_off_start={feature_off_start:#010x} / {feature_off_start}");
		// println!("child_off_start={child_off_start:#010x} / {child_off_start}");
		// println!("total_size={total_size:#010x} / {total_size}");
		// println!("desc_type={desc_type:?}");
		// println!("desc_size={desc_size:#06x} / {desc_size}");
		// println!("k={m_k:#06x} / {m_k}");

		// Total size should be consistent
		validate!((nblocks as u64) * block_size_bytes_wp == total_size, "Inconsistent header: nblocks ({nblocks}) * block_size_bytes_wp ({block_size_bytes_wp}) != total_size ({total_size})");
		validate!(desc_size_bytes_wp.is_multiple_of(alignment as _), "Inconsistent header: desc_size_bytes_wp must be multiple of alignment");
		// Block header is 8 bytes
		validate!(Block::HEADER_SIZE as u64 <= desc_size_bytes_wp, "Block size too small");
		// In theory it doesn't matter if children of features come first, but in practice features always do
		validate!(Block::HEADER_SIZE as u64 <= feature_off_start && feature_off_start < block_size_bytes_wp, "Invalid feature offset (actual: {feature_off_start}, valid: {}..{block_size_bytes_wp})", Block::HEADER_SIZE);
		validate!(feature_off_start <= child_off_start && child_off_start < block_size_bytes_wp, "Invalid child offset (actual: {child_off_start}, valid: {feature_off_start}..{block_size_bytes_wp})", );
		//TODO: ensure no overlap

		Ok(Self {
			desc_name,
			alignment,
			nblocks,
			desc_size_bytes_wp,
			block_size_bytes_wp,
			feature_off_start,
			child_off_start,
			total_size,
			desc_type,
			desc_size,
			m_k,
		})
	}
}

/// Structure represeting a information about node in a block
#[derive(Debug)]
struct BlockNodeInfo {
	id_or_childblock: u32,
	weight: f32,
}

impl BlockNodeInfo {
	fn is_leaf(&self) -> bool {
		self.id_or_childblock & 0x80000000 != 0
	}
}

/// A single FBOW block
#[derive(Debug)]
struct Block {
	/// Number of children
	/// 
	/// In fbow this was an `u16`, but we store it as an u32 for consistency
	n: u32,
	leaf: bool,
	parent_id: u32,
	children: Vec<(BlockNodeInfo, Vec<u8>)>,
}

fn take_bytes<'a, const N: usize>(data: &mut &'a [u8]) -> &'a [u8; N] {
	let (b, rest) = data.split_first_chunk::<N>().unwrap();
	*data = rest;
	b
}

#[inline]
fn convert_offset(offset: u64) -> usize {
	if cfg!(debug_assertions) {
		offset.try_into().unwrap()
	} else if size_of::<usize>() >= size_of::<u64>() {
		// Infallible conversion
		offset as usize
	} else {
		//TODO: should we cap it?
		offset as usize
	}
}

impl Block {
	const HEADER_SIZE: usize = 8;
	fn parse_header(mut data: &[u8], bi: usize, params: &FbowParams, options: &ReadFbowOptions) -> io::Result<(u32, bool, u32)> {
		assert_eq!(data.len(), Self::HEADER_SIZE);
		
		let n = {
			let n = u16::from_le_bytes(*take_bytes(&mut data)) as u32;
			// We only have space for k children per block, but it shows up in the ORB vocabulary
			if n > params.m_k {
				warning!(options.too_many_children, "Block #{bi} has too many children (actual: {n}, expected: < {})", params.m_k);
				params.m_k
			} else {
				n
			}
		};

		let leaf = {
			match u16::from_le_bytes(*take_bytes(&mut data)) {
				0 => false,
				1 => true,
				// Leaf should be 0 or 1
				v => {
					warning!(options.inconsistent_block, "Unexpected value for is_leaf: {v:#04x}");
					//TODO: what should this be?
					true
				}
			}
		};

		let parent_id = u32::from_le_bytes(*take_bytes(&mut data));
		Ok((n, leaf, parent_id))
	}

	fn parse(data: &[u8], bi: usize, params: &FbowParams, options: &ReadFbowOptions) -> std::io::Result<Self> {
		assert_eq!(data.len() as u64, params.block_size_bytes_wp, "Invalid block size");

		// Split into chunks
		let header = data.first_chunk::<{Self::HEADER_SIZE}>().unwrap();
		let (n, leaf, parent_id) = Self::parse_header(header, bi, params, options)?;

		let child_info = {
			// Each child is 8 bytes long
			const INFO_SIZE: usize = size_of::<u32>() + size_of::<f32>();
			let info = {
				// Select the relevant bytes
				let info_start = convert_offset(params.child_off_start);
				let info_len = INFO_SIZE * (n as usize);
				let info_end = info_start + info_len;
				&data[info_start..info_end]
			};

			// Convert to array of [[u8; 4]; 2]
			let info = {
				// In theory we should check u32/f32 alignment, but I think we're fine
				let (info, rem) = info.as_chunks::<4>();
				debug_assert!(rem.is_empty());
				let (info, rem) = info.as_chunks::<2>();
				debug_assert!(rem.is_empty());
				debug_assert_eq!(info.len(), n as usize);
				info
			};

			info.into_iter().map(|[id_or_childblock, weight]| {
				let id_or_childblock = u32::from_le_bytes(*id_or_childblock);
				let weight = f32::from_le_bytes(*weight);
				BlockNodeInfo { id_or_childblock, weight }
			})
		};

		let features = {
			// This is actually different from FBOW, but I think they got it wrong
			let features_start = convert_offset(params.feature_off_start);
			let feature_len_padded = convert_offset(params.desc_size_bytes_wp);
			let features_len = feature_len_padded * (n as usize);
			let features = (&data[features_start..features_start+features_len])
				.chunks_exact(feature_len_padded);
			assert_eq!(features.len(), n as usize);

			//TODO: maybe keep this as a bunch of slices / a single big vector to reduce allocations
			features.map(|feature| feature[..(params.desc_size as usize)].to_vec())
		};
		// println!("Block {bi} meta n={} of {} {:?} leaf={leaf} parent_id={parent_id}", n, params.m_k, n.cmp(&params.m_k));

		let children = child_info.zip(features).collect::<Vec<_>>();

		Ok(Self {
			n,
			leaf,
			parent_id,
			children,
		})
	}
}


impl Vocabulary {
	/// Convert from FBOW data
	/// 
	/// NB: We split this out of [Self::read_fbow] to limit specialization
	fn from_fbow(params: FbowParams, blocks: impl ExactSizeIterator<Item = Block>, options: ReadFbowOptions) -> io::Result<Self> {
		// TODO: template this
		match params.desc_type {
			DescriptorType::Uint8 => {
				let nblocks = blocks.len();
				let mut builder = VocabularyBuilder::<u8>::new(params.nblocks as _, params.desc_size as _);
				//TODO: we might be able to get rid of this if we had a better idea of the block traversal order
				let mut block_cache = HashMap::new();

				block_cache.insert(0, builder.root());
				for (block_id, block) in blocks.into_iter().enumerate() {
					if options.invalid_weight != ParseValidationMode::Ignore {
						for child in &block.children {
							if child.0.weight != 1.0 {
								warning!(options.invalid_weight, "invalid weight {} on block {block_id}+", child.0.weight);
							}
							//TODO: prune children with zero weight?
						}
					}
					let nb = block_cache.remove(&block_id)
						.expect("Missing block");

					// Reorder so leaves are at the end
					let (leaves, branches) = {
						enum LeafError {
							OutOfBounds(usize),
							Cycle(usize),
						}
						let is_leaf = |child: &BlockNodeInfo| {
							if child.is_leaf() {
								return Ok(true);
							}
							// Check that ID is in bounds (convert invalid children to leaves)
							let id = child.id_or_childblock as usize;
							if nblocks < id {
								Err(LeafError::OutOfBounds(id))
								// 
							} else if id < block_id {
								Err(LeafError::Cycle(id))
							} else {
								Ok(false)
							}
						};
						//TODO: is it worth making a fast path for when we know there won't be errors?
						let mut leaves = Vec::new();
						let mut branches = Vec::new();
						for item in block.children {
							let leaf = match is_leaf(&item.0) {
								Ok(leaf) => leaf,
								Err(LeafError::OutOfBounds(id)) => {
									warning!(options.invalid_child, "Child {block_id} -> {id} is out of bounds ({nblocks})");
									true
								},
								Err(LeafError::Cycle(id)) => {
									warning!(options.block_cycle, "Not a tree {block_id} -> {id}");
									true
								}
							};
							if leaf {
								leaves.push(item);
							} else {
								branches.push(item);
							}
						}
						(leaves, branches)
					};
					
					let children = nb.fill(branches, |(_, feat)| {
						ArrayView1::from(feat)
					}, leaves.iter().map(|(_info, feat)| ArrayView1::from(feat)));
					if let Some(children) = children {
						for ((c_info, _), cb) in children {
							block_cache.insert(c_info.id_or_childblock as usize, cb);
						}
					}
				}
				assert!(block_cache.is_empty());
				drop(block_cache);

				let mut v_params = VocabularyParams::empty();
				v_params.set_name(&params.desc_name);
				v_params.desc_type = params.desc_type;
				v_params.m_k = params.m_k;
				v_params.desc_size = params.desc_size as _;

				return Ok(builder.finish(v_params))
			},
			DescriptorType::Float32 => {
				let nblocks = blocks.len();
				let mut builder = VocabularyBuilder::<f32>::new(params.nblocks as _, params.desc_size as _);
				let mut block_cache = HashMap::new();

				block_cache.insert(0, builder.root());
				for (block_id, block) in blocks.into_iter().enumerate() {
					if options.invalid_weight != ParseValidationMode::Ignore {
						for child in &block.children {
							if child.0.weight != 1.0 {
								warning!(options.invalid_weight, "invalid weight {} on block {block_id}+", child.0.weight);
							}
							//TODO: prune children with zero weight?
						}
					}
					let Some(nb) = block_cache.remove(&block_id) else {
						return Err(io::Error::new(ErrorKind::InvalidData, format!("Missing block {block_id}")))
					};
					// Reorder so leaves are at the end
					let mut leaves = Vec::new();
					let mut branches = Vec::new();
					let child_is_leaf = |child: &BlockNodeInfo| {
						if child.is_leaf() {
							return true;
						}
						// Check that ID is in bounds (convert invalid children to leaves)
						let id = child.id_or_childblock as usize;
						if nblocks < id {
							println!("[warn] Child {block_id} -> {id} is out of bounds ({nblocks})");
							true
						} else if id < block_id {
							println!("[warn] Not a tree {block_id} -> {id}");
							true
						} else {
							false
						}
					};
					for (child, data) in block.children {
						// Convert to f32
						let data = convert_le(&data)
							.expect("TODO: good error")
							.collect::<Vec<_>>();
						(if child_is_leaf(&child) { &mut leaves } else { &mut branches }).push((child, data));
					}
					
					fn array_view<'a>((_, feat): &'a (BlockNodeInfo, Vec<f32>)) -> ArrayView1<'a, f32> {
						ArrayView1::from(feat)
					}
					let children = nb.fill(branches, array_view, leaves.iter().map(array_view));
					if let Some(children) = children {
						for ((c_info, _), cb) in children {
							block_cache.insert(c_info.id_or_childblock as usize, cb);
						}
					}
				}
				assert!(block_cache.is_empty());
				drop(block_cache);

				let mut v_params = VocabularyParams::empty();
				v_params.set_name(&params.desc_name);
				v_params.desc_type = params.desc_type;
				v_params.m_k = params.m_k;
				v_params.desc_size = params.desc_size as _;

				return Ok(builder.finish(v_params))
			},
		}
	}

	/// Read FBOW data 
	pub(super) fn read_fbow(mut src: impl Read, options: ReadFbowOptions) -> std::io::Result<Self> {
		// Parse FBOW vocabulary
		// println!("Reading as FBOW");
		let params = FbowParams::read_from(&mut src, &options)?;
		// println!("FBOW parameters: {params:?}");
		let blocks = {
			// We *could* read everything into memory and parse it, but I like this a bit better
			let mut block_data = vec![0u8; params.block_size_bytes_wp as usize];
			let mut blocks = Vec::with_capacity(params.nblocks as _);
			for bi in 0..params.nblocks {
				src.read_exact(&mut block_data)?;
				
				blocks.push(Block::parse(&block_data, bi as _, &params, &options)?);
			}

			// println!("Read blocks {blocks:?}");
			blocks
		};

		// Now convert to Vocabulary
		Self::from_fbow(params, blocks.into_iter(), options)
	}
}