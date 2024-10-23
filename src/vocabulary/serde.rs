use std::{cmp, collections::HashMap, ffi::CStr, io::{self, ErrorKind, Read, Write}, str::FromStr};

use arrayvec::ArrayString;
use ndarray::ArrayView1;

use crate::{traits::DescriptorType, vocabulary::VocabularyBuilder, Deserialize, Serialize};

use super::Vocabulary;

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

#[derive(Debug)]
struct Block {
	n: u16,
	leaf: bool,
	parent_id: u32,
	children: Vec<(BlockNodeInfo, Vec<u8>)>,
}
/*
	inline bool isleaf()const{return ( );}

	//if not leaf, returns the block where the children are
	//if leaf, returns the index of the feature it represents. In case of bagofwords it must be a invalid value
	inline uint32_t getId()const{return ( id_or_childblock&0x7FFFFFFF);}

	//sets as leaf, and sets the index of the feature it represents and its weight
	inline void setLeaf(uint32_t id,float Weight){
		assert(!(id & 0x80000000));//check msb is zero
		id_or_childblock=id;
		id_or_childblock|=0x80000000;//set the msb to one to distinguish from non leaf
		//now,set the weight too
		weight=Weight;
	}
	//sets as non leaf and sets the id of the block where the chilren are
	inline void setNonLeaf(uint32_t id){
		//ensure the msb is 0
		assert( !(id & 0x80000000));//32 bits 100000000...0.check msb is not set
		id_or_childblock=id;
	}
};*/

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

/// FBOW parameters
#[derive(Debug)]
// #[repr(C)]
struct FbowParams {
	//descriptor name. May be empty
	desc_name: ArrayString<49>, // 49 bytes + null terminator
	// desc_name: [u8; 50],
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

impl Deserialize for FbowParams {
	fn read_from(mut src: impl Read) -> std::io::Result<Self> {
		let desc_name = {
			// We ensure there's at least one null terminator
			let mut desc_bytes = [0u8; 51];
			src.read_exact(&mut desc_bytes[..50])?;
			println!("desc_buf {desc_bytes:?}");
			let cstr = CStr::from_bytes_until_nul(&desc_bytes).unwrap();
			let str = cstr.to_str()
				.map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
			ArrayString::from_str(str).unwrap() // I don't think we can overflow at this point
		};
		println!("Read name {desc_name}");
		fn read_padding<const N: usize>(src: &mut impl Read) -> io::Result<()> {
			let mut buf = [0u8; N];
			src.read_exact(&mut buf)?;
			assert_eq!(buf, [0u8; N]);
			Ok(())
		}


		// Because of padding, there should be an extra two bytes
		read_padding::<2>(&mut src)?;

		fn read_u32(src: &mut impl Read) -> io::Result<u32> {
			let mut buf = [0u8; size_of::<u32>()];
			src.read_exact(&mut buf)?;
			Ok(u32::from_le_bytes(buf))
		}
		fn read_u64(src: &mut impl Read) -> io::Result<u64> {
			let mut buf = [0u8; size_of::<u64>()];
			src.read_exact(&mut buf)?;
			Ok(u64::from_le_bytes(buf))
		}
		let alignment = read_u32(&mut src)?;
		let nblocks = read_u32(&mut src)?;
		println!("Alignment={alignment:#04x} {alignment}");
		println!("nblocks={nblocks:#04x} {nblocks}");
		// Because of padding, there should be an extra four bytes
		read_padding::<4>(&mut src)?;
		let desc_size_bytes_wp = read_u64(&mut src)?;
		println!("desc_size_bytes_wp={desc_size_bytes_wp:#010x} / {desc_size_bytes_wp}");
		let block_size_bytes_wp = read_u64(&mut src)?;
		println!("block_size_bytes_wp={block_size_bytes_wp:#010x} / {block_size_bytes_wp}");
		let feature_off_start = read_u64(&mut src)?;
		println!("feature_off_start={feature_off_start:#010x} / {feature_off_start}");
		let child_off_start = read_u64(&mut src)?;
		println!("child_off_start={child_off_start:#010x} / {child_off_start}");
		let total_size = read_u64(&mut src)?;
		println!("total_size={total_size:#010x} / {total_size}");
		let desc_type = match read_u32(&mut src)? {
			0 => DescriptorType::Uint8,
			5 => DescriptorType::Float32,
			dt => return Err(io::Error::new(ErrorKind::InvalidData, format!("Unexpected desc_type {dt}"))),
		};
		println!("desc_type={desc_type:?}");
		let desc_size = read_u32(&mut src)?;
		println!("desc_size={desc_size:#06x} / {desc_size}");
		let m_k = read_u32(&mut src)?;
		println!("k={m_k:#06x} / {m_k}");

		// Final 4 bytes of padding
		read_padding::<4>(&mut src)?;

		// Consistency check
		if (nblocks as u64) * block_size_bytes_wp != total_size {
			return Err(io::Error::new(ErrorKind::InvalidData, format!("Consistency check failed: nblocks ({nblocks}) * block_size_bytes_wp ({block_size_bytes_wp}) != total_size ({total_size})")));
		}

		Ok(Self {
			desc_name,
			alignment: alignment as _,
			nblocks,
			desc_size_bytes_wp,
			block_size_bytes_wp,
			feature_off_start,
			child_off_start,
			total_size,
			desc_type,
			desc_size: desc_size as _,
			m_k,
		})
	}
}

const FBOW_MAGIC: u64 = 55824124;
const VFBOW_MAGIC: u64 = 5546449913086866150;

// We write in VFBOW format
impl Serialize for Vocabulary {
	fn write_to(&self, mut dst: impl Write) -> std::io::Result<()> {
		//magic number
		dst.write_all(&VFBOW_MAGIC.to_le_bytes())?;
		//save string
		self.params.write_to(&mut dst)?;

		self.features.write_to(dst)
		
		// str.write((char*)&_params,sizeof(params));
		// str.write(_data.get(), _params._total_size);
	}
}

impl Deserialize for Vocabulary {
	fn read_from(mut src: impl Read) -> std::io::Result<Self> {
		let magic = {
			let mut sig_buf = [0u8; size_of::<u64>()];
			src.read_exact(&mut sig_buf)?;
			u64::from_le_bytes(sig_buf)
		};
		match magic {
			FBOW_MAGIC => {
				// Parse FBOW vocabulary
				println!("Reading as FBOW");
				let params = FbowParams::read_from(&mut src)?;
				println!("FBOW parameters: {params:?}");
				let blocks = {
					// We *could* read everything into memory and parse it, but I like this a bit better
					let mut block_data = vec![0u8; params.block_size_bytes_wp as usize];
					let mut blocks = Vec::with_capacity(params.nblocks as _);
					for bi in 0..params.nblocks {
						src.read_exact(&mut block_data)?;

						let n = {
							let b: [u8; 2] = block_data[0..2].try_into().unwrap();
							u16::from_le_bytes(b)
						};
						let n = cmp::min(params.m_k, n as u32);

						let leaf = {
							let b: [u8; 2] = block_data[2..4].try_into().unwrap();
							match u16::from_le_bytes(b) {
								0 => false,
								1 => true,
								v => return Err(io::Error::new(ErrorKind::InvalidData, format!("Unexpected value for is_leaf: {v:#04x}"))),
							}
						};

						let parent_id = {
							let b: [u8; 4] = block_data[4..8].try_into().unwrap();
							u32::from_le_bytes(b)
						};

						// println!("Block {bi} meta n={} of {} {:?} leaf={leaf} parent_id={parent_id}", n, params.m_k, n.cmp(&params.m_k));

						let mut children = Vec::with_capacity(n as _);
						for i in 0..n {
							let bni = {
								let offset = (params.child_off_start as usize) + (i as usize) * 8;
								let id_or_childblock = {
									let buf: [u8; 4] = block_data[offset..offset+4].try_into().unwrap();
									u32::from_le_bytes(buf)
								};
	
								let weight = {
									let buf: [u8; 4] = block_data[offset+4..offset+8].try_into().unwrap();
									f32::from_le_bytes(buf)
								};
								BlockNodeInfo { id_or_childblock, weight }
							};

							let feature = {
								// This is actually different from FBOW, but I think they got it wrong
								let offset = (params.feature_off_start as usize) + (i as usize) * (params.desc_size_bytes_wp as usize);
								block_data[offset..offset+(params.desc_size as usize)].to_vec()
							};

							children.push((bni, feature));
						}
						blocks.push(Block {
							n: n as _,
							leaf,
							parent_id,
							children,
						});
					}

					// println!("Read blocks {blocks:?}");
					blocks
				};
				// Now convert to Vocabulary

				match params.desc_type {
					DescriptorType::Uint8 => {
						let mut builder = VocabularyBuilder::<u8>::new(params.nblocks as _, params.desc_size as _);
						let mut block_cache = HashMap::new();

						block_cache.insert(0, builder.root());
						for (block_id, block) in blocks.into_iter().enumerate() {
							for child in &block.children {
								if child.0.weight != 1.0 {
									println!("Warning: invalid weight {}", child.0.weight);
								}
							}
							let nb = block_cache.remove(&block_id)
								.expect("Missing block");
							// Reorder so leaves are at the end
							let (leaves, branches) = block.children
								.into_iter()
								.partition::<Vec<_>, _>(|child| child.0.is_leaf());
							
							let children = nb.fill(branches, |(_, feat)| {
								ArrayView1::from(feat)
							}, leaves.iter().map(|(_info, feat)| ArrayView1::from(feat)));
							if let Some(children) = children {
								for ((c_info, _), cb) in children {

								}
							}
						}
					},
					_ => todo!(),
				}
				// _data = std::unique_ptr<char[], decltype(&AlignedFree)>((char*)AlignedAlloc(_params._aligment, _params._total_size), &AlignedFree);
				// if (_data.get() == nullptr) throw std::runtime_error("Vocabulary::fromStream Could not allocate data");
				// str.read(_data.get(), _params._total_size);
			},
			VFBOW_MAGIC => {
				todo!("Read VFBOW")
			},
			_ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid signature {magic:#08x}"))),
		}
	}
}