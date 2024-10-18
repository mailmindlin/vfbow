mod distance;
mod distance_l1;

use std::{ffi::CStr, io::{self, ErrorKind, Read, Write}, str::FromStr};

use arrayvec::ArrayString;

use crate::{fbow::{FBOW, FBOW2}, traits::{DescriptorType, Deserialize, NodeId, SelfHash, Serialize}};

/// a block represent all the child nodes of a parent, with its features and also information about where the child of these are in the data structure
/// 
/// a block structure is as follow: N|isLeaf|BlockParentId|p|F0...FN|C0W0 ... CNWN..
/// - N :16 bits : number of nodes in this block. Must be <=branching factor k. If N<k, then the block has empty spaces since block size is fixed
/// - isLeaf:16 bit inicating if all nodes in this block are leaf or not
/// - BlockParentId:31: id of the parent
/// - p :possible offset so that Fi is aligned
/// - Fi feature of the node i. it is aligned and padding added to the end so that F(i+1) is also aligned
/// - CiWi are the so called block_node_info (see structure up)
/// - Ci : either if the node is leaf (msb is set to 1) or not. If not leaf, the remaining 31 bits is the block where its children are. Else, it is the index of the feature that it represent
/// - Wi: float value empkoyed to know the weight of a leaf node (employed in cases of bagofwords)
pub(crate) struct Block<'a> {
	blockstart: &'a u8,
	// char *_blockstart;
	// uint64_t _desc_size_bytes=0;//size of the descriptor(without padding)
	// uint64_t _desc_size_bytes_wp=0;//size of the descriptor(includding padding)
	// uint64_t _feature_off_start=0;
	// uint64_t _child_off_start=0;//into the block,where the children offset part starts
}

impl<'a> Block<'a> {
	// Block(char * bsptr,uint64_t ds,uint64_t ds_wp,uint64_t fo,uint64_t co):_blockstart(bsptr),_desc_size_bytes(ds),_desc_size_bytes_wp(ds_wp),_feature_off_start(fo),_child_off_start(co){}
	// Block(uint64_t ds,uint64_t ds_wp,uint64_t fo,uint64_t co):_desc_size_bytes(ds),_desc_size_bytes_wp(ds_wp),_feature_off_start(fo),_child_off_start(co){}

	fn get_n(&self) -> u16 {
		todo!("*((uint16_t*)(_blockstart))")
	}
	pub(crate) fn set_n(&self, n: u16) {
		todo!("*((uint16_t*)(_blockstart))=n;")
	}

	fn is_leaf(&self) -> bool {
		todo!("*((uint16_t*)(_blockstart)+1)")
	}
	pub(super) fn set_leaf(&mut self, leaf: bool) {
		todo!("*((uint16_t*)(_blockstart)+1)=1;")
	}

	pub(super) fn set_parent(&mut self, parent_id: NodeId) {
		todo!("*(((uint32_t*)(_blockstart))+1)=pid;")
	}
	fn get_parent(&self) -> NodeId {
		todo!("*(((uint32_t*)(_blockstart))+1);")
	}

	pub(crate) fn block_node_info(&self, i: usize) -> &BlockNodeInfo {
		todo!()
	}

	pub(crate) fn block_node_info_mut(&mut self, i: usize) -> &mut BlockNodeInfo {
		todo!()
	}

	// inline  block_node_info * getBlockNodeInfo(int i){  return (block_node_info *)(_blockstart+_child_off_start+i*sizeof(block_node_info)); }
	// inline  void setFeature(int i,const cv::Mat &feature){memcpy( _blockstart+_feature_off_start+i*_desc_size_bytes_wp,feature.ptr<char>(0),feature.elemSize1()*feature.cols); }
	// inline  void getFeature(int i,cv::Mat  feature){    memcpy( feature.ptr<char>(0), _blockstart+_feature_off_start+i*_desc_size_bytes,_desc_size_bytes ); }
	// template<typename T> inline  T*getFeature(int i){return (T*) (_blockstart+_feature_off_start+i*_desc_size_bytes_wp);}
}

pub(crate) struct VocabularyParams {
	//descriptor name. May be empty
	desc_name: ArrayString<49>, // 49 bytes + null terminator
	/// Memory alignment
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
	// int32_t _desc_type=0,_desc_size=0;//original descriptor types and sizes (without padding)
	desc_type: u32,
	desc_size: u32,
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
			desc_size_bytes_wp: 0,
			block_size_bytes_wp: 0,
			feature_off_start: 0,
			child_off_start: 0,
			total_size: 0,
			desc_type: 0,
			desc_size: 0,
			m_k: 0,
		}
	}

	pub(crate) fn set(&mut self, aligment: usize, k: u32, desc_type: DescriptorType, desc_size: usize, nblocks: usize, desc_name: &str) {
		self.set_name(desc_name);
	
		todo!()
		/*self.params.alignment = aligment;
		self.params.m_k = k;
		self.params.desc_type=desc_type;
		self.params.desc_size=desc_size;
		self.params.nblocks = nblocks;
	
	
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
		dst.write_all(&self.alignment.to_le_bytes())?;
		dst.write_all(&self.nblocks.to_le_bytes())?;
		dst.write_all(&self.desc_size_bytes_wp.to_le_bytes())?;
		dst.write_all(&self.block_size_bytes_wp.to_le_bytes())?;
		dst.write_all(&self.feature_off_start.to_le_bytes())?;
		dst.write_all(&self.child_off_start.to_le_bytes())?;
		dst.write_all(&self.total_size.to_le_bytes())?;
		dst.write_all(&self.m_k.to_le_bytes())
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
		let desc_size_bytes_wp = read_u64(&mut src)?;
		let block_size_bytes_wp = read_u64(&mut src)?;
		let feature_off_start = read_u64(&mut src)?;
		let child_off_start = read_u64(&mut src)?;
		let total_size = read_u64(&mut src)?;
		let desc_type = read_u32(&mut src)?;
		let desc_size = read_u32(&mut src)?;
		let m_k = read_u32(&mut src)?;

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
struct BlockNodeInfo {
	/// if id, msb is 1.
	id_or_childblock: u32,
	weight: f32,
}

impl BlockNodeInfo {
	fn is_leaf(&self) -> bool {
		self.id_or_childblock & 0x80000000 != 0
	}

	// //if not leaf, returns the block where the children are
	// //if leaf, returns the index of the feature it represents. In case of bagofwords it must be a invalid value
	// inline uint32_t getId()const{return ( id_or_childblock&0x7FFFFFFF);}

	// //sets as leaf, and sets the index of the feature it represents and its weight
	pub(crate) fn set_leaf(&mut self, id: NodeId, weight: f32) {
	//     assert(!(id & 0x80000000));//check msb is zero
	//     id_or_childblock=id;
	//     id_or_childblock|=0x80000000;//set the msb to one to distinguish from non leaf
	//     //now,set the weight too
	//     weight=Weight;
		todo!()
	}
	// //sets as non leaf and sets the id of the block where the chilren are
	pub(crate) fn set_non_leaf(&mut self, id: NodeId){
		todo!()
	//     //ensure the msb is 0
	//     assert( !(id & 0x80000000));//32 bits 100000000...0.check msb is not set
	//     id_or_childblock=id;
	}
}

/// Main class to represent a vocabulary of visual words
#[cfg_attr(feature="python", pyo3::pyclass)]
pub struct Vocabulary {
	params: VocabularyParams,
	data: Vec<u8>,
	// /// information about the cpu so that mmx, sse, or avx extensions can be employed
	// cpu_info: CpuFeatures,
}

fn ilog2(x: u32) -> u32 {
	u32::BITS - x.next_power_of_two().leading_zeros()
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum TransformError {
	#[error("No input data")]
	NoInputData,
	#[error("Transform features are of different size than the vocabulary ones")]
	SizeMismatch,
}

impl Vocabulary {
	pub(crate) fn new(params: VocabularyParams) -> Self {
		todo!()
	}
	///returns the descriptor name
	pub fn desc_name(&self) -> &str {
		&self.params.desc_name
	}

	// /// Returns the descriptor type (CV_8UC1, CV_32FC1  )
	// pub fn desc_type(&self) -> u32 {
	//     self.params.desc_type
	// }
	// /// Returns desc size in bytes or 0 if not set
	// pub fn desc_size(&self) -> Option<NonZeroUsize> {
	//     self.params.desc_size
	// }

	/// Returns the branching factor (number of children per node)
	pub fn k(&self) -> u32 {
		self.params.m_k
	}

	/// indicates whether this object is valid
	pub fn is_valid(&self) -> bool {
		!self.data.is_empty()
	}
	/// total number of blocks
	pub fn size(&self) -> u32 {
		self.params.nblocks
	}

	/// removes all data
	pub fn clear(&mut self) {
		self.data.clear();
		self.params = VocabularyParams::empty();
	}

	//returns a block structure pointing at block b
	pub(crate) fn getBlock(&self, b: u32) -> Block {
		// assert(_data.get() != nullptr);
		assert!(self.is_valid());
		assert!(b < self.params.nblocks);
		todo!()
		// Block {
			
		// }
		// return Block(_data.get() + b * _params._block_size_bytes_wp, _params._desc_size, _params._desc_size_bytes_wp, _params._feature_off_start, _params._child_off_start);
	}

	pub fn transform_l1(&self, features: ndarray::ArrayView2<u8>, level: usize, result: &mut FBOW, result2: &mut FBOW2) -> Result<(), TransformError> {
		if features.nrows() == 0 {
			return Err(TransformError::NoInputData);
		}
		// if (features.type()!=_params._desc_type) throw std::runtime_error("Vocabulary::transform features are of different type than vocabulary");
		if features.ncols() != self.params.desc_size as _ {
			return Err(TransformError::SizeMismatch);
		}
		//decide the version to employ according to the type of features, aligment and cpu capabilities
		//orb
		
		todo!()
		/*if cfg!(target_pointer_width = "64") {
			if self.params.desc_size == 32 {
				_transform2<L1_32bytes>(features,level,result,result2);
			} else if self.params.desc_size == 61 && self.params.alignment.is_multiple_of(8) {
				// Full AKAZE
				_transform2<L1_61bytes>(features,level,result,result2);
			} else {
				// Generic
				_transform2<L1_x64>(features,level,result,result2);
			}
		} else {
			_transform2<L1_x32>(features,level,result,result2)
		}*/
	}

	pub fn transform_l2(&self, features: ndarray::ArrayView2<f32>, level: u32, result: &mut FBOW, result2: &mut FBOW2) {
		/*if is_x86_feature_detected!("avx") && self.params.alignment.is_multiple_of(32) {
			// AVX version
			if self.params.desc_size == 256 {
				// Specific for SURF 256 bytes
				self._transform2(features, level, result, result2, distance::l2_avx_array::<8>)
			} else {
				self._transform2(features, level, result, result2, distance::l2_avx_generic)
			}
		} else if is_x86_feature_detected!("sse") && self.params.alignment.is_multiple_of(16) {
			if self.params.desc_size == 256 {
				// Specific for SURF 256 bytes
				self._transform2_arr(features, level, result, result2, distance::l2_avx_array::<8>)
			} else {
				// Any other
				self._transform2(features, level, result, result2, distance::l2_avx_generic)
			}
		}

		// Generic version
		self._transform2(features, level, result, result2, distance::l2_generic)*/
		todo!()
	}

	fn _transform2_arr<T, D, F, const N: usize>(&self, features: ndarray::ArrayView2<T>, storeLevel: u32, r1: &mut FBOW, r2: &mut FBOW2, transform: impl Fn(&[F; N],&[F; N]) -> D) {
		todo!()
	}

	fn _transform2<T, D, F>(&self, features: ndarray::ArrayView2<T>, storeLevel: u32, r1: &mut FBOW, r2: &mut FBOW2, transform: impl Fn(&[F],&[F]) -> D) {
		// comp.setParams(_params._desc_size,_params._desc_size_bytes_wp);
		// using DType=typename Computer::DType;//distance type
		// using TData=typename Computer::TData;//data type
		let required_alignment = align_of::<F>();
		assert_eq!(self.params.alignment as usize % required_alignment, 0);
		//TODO: assert capacity?
		r1.clear();
		r2.clear();
		// Minimum distance found
		// let best_idx = None;
		// std::pair<DType,uint32_t> best_dist_idx(std::numeric_limits<uint32_t>::max(),0);//minimum distance found
		// block_node_info *bn_info;
		let nbits = ilog2(self.params.m_k);

		for cur_feature in 0..features.nrows() {
			/*comp.startwithfeature(features.ptr<TData>(cur_feature));
			//ensure feature is in a
			let c_block = self.getBlock(0);
			let level = 0;//current level of recursion
			let curNode = 0;//id of the current node of the tree
			//copy to another structure and add padding with zeros
			do{
				//given the current block, finds the node with minimum distance
				best_dist_idx.first=std::numeric_limits<uint32_t>::max();
				for cur_node in 0..c_block.getN() {
					DType d= comp.computeDist(c_block.getFeature<TData>(cur_node));
					if (d<best_dist_idx.first) best_dist_idx=std::make_pair(d,cur_node);
				}
				if( level==storeLevel)//if reached level,save
					r2[curNode].push_back( cur_feature);

				bn_info=c_block.getBlockNodeInfo(best_dist_idx.second);
				//if the node is leaf get weight,else go to its children
				if ( bn_info->isleaf()){
					r1[bn_info->getId()]+=bn_info->weight;
					if( level<storeLevel)//store level not reached, save now
						r2[curNode].push_back( cur_feature);
				break;
				}
				else setBlock(bn_info->getId(),c_block);//go to its children
				curNode= curNode<<nbits;
				curNode|=best_dist_idx.second;
				level++;
			}while( !bn_info->isleaf() && bn_info->getId()!=0);*/
			todo!()
		}
	}

	fn setParams(&mut self, aligment: usize, k: usize, desc_type: usize, desc_size: usize, nblocks: usize, desc_name: &str) {
		self.params.set_name(desc_name);
	
		todo!()
		/*self.params.alignment = aligment;
		self.params.m_k = k;
		self.params.desc_type=desc_type;
		self.params.desc_size=desc_size;
		self.params.nblocks = nblocks;
	
	
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
}

impl SelfHash for Vocabulary {
	fn hash(&self) -> u64 {
		let mut seed = 0;
		for i in 0..self.params.total_size {
			seed ^= (self.data[i as usize] as u64) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		seed
	}
}



const VOCABULARY_MAGIC: u64 = 55824124;
impl Serialize for Vocabulary {
	fn write_to(&self, mut dst: impl Write) -> std::io::Result<()> {
		//magic number
		dst.write_all(&VOCABULARY_MAGIC.to_le_bytes())?;
		//save string
		self.params.write_to(&mut dst)?;
		
		// str.write((char*)&_params,sizeof(params));
		// str.write(_data.get(), _params._total_size);
		todo!()
	}
}

impl Deserialize for Vocabulary {
	fn read_from(mut src: impl Read) -> std::io::Result<Self> {
		{
			let sig = {
				let mut sig_buf = [0u8; size_of::<u64>()];
				src.read_exact(&mut sig_buf)?;
				u64::from_le_bytes(sig_buf)
			};
			if VOCABULARY_MAGIC != sig {
				return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid signature"));
			}
		}

		//read string
		let params = VocabularyParams::read_from(&mut src)?;
		// _data = std::unique_ptr<char[], decltype(&AlignedFree)>((char*)AlignedAlloc(_params._aligment, _params._total_size), &AlignedFree);
		// if (_data.get() == nullptr) throw std::runtime_error("Vocabulary::fromStream Could not allocate data");
		// str.read(_data.get(), _params._total_size);
		todo!()
	}
}