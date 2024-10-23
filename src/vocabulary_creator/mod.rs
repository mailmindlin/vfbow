// mod exec;

use core::{error, range::Range};
use std::{arch::is_aarch64_feature_detected, collections::HashMap, marker::PhantomData, num::NonZeroUsize, ops::Index, process::id, sync::Mutex, u32};

use ndarray::{ArrayView1, CowArray, Ix1};
use rand::{rngs::StdRng, SeedableRng};

use crate::{traits::{DescriptorType, NodeId}, vocabulary::{Vocabulary, VocabularyParams}};


#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", get_all, set_all, eq))]
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct VocabularyCreatorParams {
	/// Braching factor
	pub k: u32,
	/// Maximum tree depth
	#[allow(non_snake_case)]
	pub L: Option<u32>,
	/// Number of threads to use while computing
	/// 
	/// 0 => system autodetect
	/// 1 => single-threaded
	pub nthreads: usize,
	pub max_iters: usize,
	pub verbose: bool,
}

impl Default for VocabularyCreatorParams {
	// Note: changes here should also be made to [VocabularyCreatorParams::new] for python FFI
	fn default() -> Self {
		Self {
			k: 32,
			L: None,
			nthreads: 1,
			max_iters: 11,
			verbose: false,
		}
	}
}

struct FeatureIndex {
	/// Index into vector of matrices
	midx: usize,
	/// Matrix row
	fidx: usize,
}

impl FeatureIndex {
	const fn new(midx: usize, fidx: usize) -> Self {
		Self {
			midx,
			fidx,
		}
	}
}

/// Struct to acces the features as a unique vector
struct FeatureInfo<T> {
	finfo: Vec<FeatureIndex>,
	features: Vec<ndarray::Array2<T>>,
}

impl<T> FeatureInfo<T> {
	fn create(features: Vec<ndarray::Array2<T>>) -> Self {
		let size = features.iter()
			.map(|feature| feature.nrows())
			.sum();
		let mut finfo = Vec::with_capacity(size);
		for (midx, feature) in features.iter().enumerate() {
			for i in 0..feature.nrows() {
				finfo.push(FeatureIndex::new(midx, i));
			}
		}
		Self { finfo, features }
	}

	fn info(&self, i: usize) -> &FeatureIndex {
		&self.finfo[i]
	}

	/// Total number of rows
	fn len(&self) -> usize {
		self.finfo.len()
	}
	/// Get the n<sup>th</sup> feature
	fn get(&self, i: usize) -> ndarray::ArrayView1<'_, T> {
		let idx = &self.finfo[i];
		self.features[idx.midx].row(idx.fidx)
	}
}

struct Node<'a, T> {
	/// id of this node in the tree
	id: NodeId,
	/// id of the parent node
	parent: NodeId,
	/// Feature of this node
	feature: CowArray<'a, T, Ix1>,
	//index of the feature this node represent(only if leaf and it stop because not enough points to create a new leave.
	//In case the node is a terminal point, but has many points beloging to its cluster, then, this is not set.
	//In other words, it is only used in nn search problems where L=-1
	feat_idx: Option<u32>,
	children: Vec<NodeId>,
	/// if leaf, its weight and the word id
	weight: f32,
}

/*impl<T> Default for Node<T> {
	fn default() -> Self {
		Self {
			id: u32::MAX,
			parent: u32::MAX,
			feature: (),
			feat_idx: u32::MAX,
			children: Vec::new(),
			weight: 1.
		}
	}
}*/

impl<'a, T> Node<'a, T> {
	fn is_leaf(&self) -> bool {
		self.children.is_empty()
	}

	fn new(id: NodeId, parent: NodeId, feature: CowArray<'a, T, Ix1>, feat_idx: Option<u32>) -> Self {
		Self {
			id,
			parent,
			feature,
			feat_idx,
			children: Vec::new(),
			weight: 1.,
		}
	}
}
struct Tree<'a, T>(HashMap<NodeId, Node<'a, T>>);

impl<'a, T> Tree<'a,T> {
	fn new() -> Self {
		/*let mut n = Node::default();
		n.id = 0;
		let mut nodes = HashMap::new();
		nodes.insert(0, n);
		Self(nodes)*/
		todo!()
	}

	fn add(&mut self, new_nodes: Vec<Node<'a,T>>, parent_id: NodeId) {
		let parent = self.0.get_mut(&parent_id)
			.expect("Invalid parent id");
		parent.children.extend(
			new_nodes
				.iter()
				.map(|node| node.id)
		);

		for node in new_nodes {
			let id = node.id;
			self.0.insert(id, node);
		}
	}

	fn len(&self) -> usize {
		self.0.len()
	}
}


trait VocabElement {
	const MIN_ALIGNMENT: usize;
	const TYPE: DescriptorType;
	fn prefer_alignment(_ncols: NonZeroUsize) -> usize {
		Self::MIN_ALIGNMENT
	}
}
impl VocabElement for u8 {
	const MIN_ALIGNMENT: usize = 8;
	const TYPE: DescriptorType = DescriptorType::Uint8;
	fn prefer_alignment(ncols: NonZeroUsize) -> usize {
		let ncols = ncols.get();
		// Prefer u128 alignment
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(64) && std::arch::is_x86_feature_detected!("avx512f") {
			return align_of::<core::arch::x86::__m512i>();
		}
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(32) && std::arch::is_x86_feature_detected!("avx") {
			return align_of::<core::arch::x86::__m256i>();
		}
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(32) && std::arch::is_x86_feature_detected!("sse") {
			return align_of::<core::arch::x86::__m128i>();
		}

		#[cfg(target_arch="aarch64")]
		if ncols.is_multiple_of(16) && std::arch::is_aarch64_feature_detected!("neon") {
			// NEON 
			return align_of::<core::arch::aarch64::uint8x16_t>();
		}

		// Try using u128
		// TODO does this have any performance benefit?
		if ncols.is_multiple_of(16) {
			return align_of::<u128>();
		} else if ncols.is_multiple_of(8) {
			return align_of::<u64>();
		} else {
			align_of::<u8>()
		}
	}
}
impl VocabElement for f32 {
	const MIN_ALIGNMENT: usize = 32;
	const TYPE: DescriptorType = DescriptorType::Float32;

	fn prefer_alignment(ncols: NonZeroUsize) -> usize {
		let ncols = ncols.get();
		// Prefer u128 alignment
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(16) && std::arch::is_x86_feature_detected!("avx512f") {
			return align_of::<core::arch::x86::__m512>();
		}
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(8) && std::arch::is_x86_feature_detected!("avx") {
			return align_of::<core::arch::x86::__m256>();
		}
		#[cfg(target_arch="x86")]
		if ncols.is_multiple_of(4) && std::arch::is_x86_feature_detected!("sse") {
			return align_of::<core::arch::x86::__m128>();
		}

		#[cfg(target_arch="aarch64")]
		if ncols.is_multiple_of(2) && std::arch::is_aarch64_feature_detected!("neon") {
			// NEON
			return if ncols.is_multiple_of(4) {
				align_of::<core::arch::aarch64::float32x4_t>()
			} else {
				align_of::<core::arch::aarch64::float32x2_t>()
			}
		}

		align_of::<f32>()
	}
}

struct InnerParams<T> {
	rng: StdRng,
	k: u32,
	#[allow(non_snake_case)]
	L: Option<u32>,
	features: FeatureInfo<T>,
	max_iters: usize,

	desc_cols: usize,
	// desc_type: u32,
	// desc_nbytes: usize,
}

impl<T> InnerParams<T> {
	fn child_node(&self, parent: NodeId, i: u32) -> NodeId {
		parent * self.k + 1 + u32::try_from(i).unwrap()
	}
	fn child_range(&self, parent: NodeId, n: u32) -> std::ops::Range<NodeId> {
		let start = self.child_node(parent, 0);
		let end = self.child_node(parent, n);
		start..end
	}
}

struct InnerResult<'a, T> {
	tree: HashMap<NodeId, Node<'a, T>>,
	/// for each node, its assigment vector
	id_assignments: HashMap<NodeId, Vec<u32>>,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum CreateVocabularyError {
	#[error("No features")]
	NoFeatures,
	#[error("Feature 0 had no columns")]
	EmptyFeature,
	#[error("Node had too many children")]
	TooManyChildren,
}

impl<'a, T: VocabElement> InnerResult<'a, T> {
	fn into_vocabulary(mut self, params: &'a InnerParams<T>, desc_name: &str) -> Result<Vocabulary, CreateVocabularyError> {
		//look for leafs and store
		//now, create the blocks

		let mut n_leaf_nodes = 0;
		let mut non_leaf_nodes = 0u32;
		let mut node_to_block = HashMap::new();

		for (id, node) in self.tree.iter_mut() {
			if node.is_leaf() {
				//assing an id if not set
				if node.feat_idx.is_none() {
					node.feat_idx = Some(n_leaf_nodes);
				}
				n_leaf_nodes += 1
			} else {
				node_to_block.insert(*id, non_leaf_nodes);
				non_leaf_nodes += 1;
			}
		}
		
		//determine the basic elements
		
		let v_params = {
			let mut v_params = VocabularyParams::empty();
			let alignment = T::MIN_ALIGNMENT;
			let desc_size = params.desc_cols * T::TYPE.element_size();
			v_params.set(T::MIN_ALIGNMENT, params.k, T::TYPE, desc_size, non_leaf_nodes as _, desc_name);
			v_params
		};

		let mut result = Vocabulary::new(v_params);

		//lets start
		for (id, node) in self.tree.iter() {
			if !node.is_leaf() {
				let block_id = node_to_block.get(id).unwrap();
				let mut binfo = result.getBlock(*block_id);

				let n = u16::try_from(node.children.len())
					.map_err(|e| CreateVocabularyError::TooManyChildren)?;
				binfo.set_n(n);
				binfo.set_parent(*id);
				/*let areAllChildrenLeaf = true;
				for (cidx, cid) in node.children.iter().enumerate() {
					let child = self.tree.get(cid).unwrap();
					binfo.set_feature(cidx, child.feature);
					//go to the end and set info
					if child.is_leaf() {
						binfo.block_node_info_mut(cidx).set_leaf(child.feat_idx, child.weight);
					} else {
						let child_block = node_to_block.get(&child.id).unwrap();
						binfo.block_node_info_mut(cidx).set_non_leaf(*child_block);
						areAllChildrenLeaf = false;
					}
				}
				binfo.set_leaf(areAllChildrenLeaf);*/
				todo!()
			}
		}

		Ok(result)
	}
}

/// This class creates the vocabulary
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow"))]
pub struct VocabularyCreator {
	params: VocabularyCreatorParams,
	// tree: Mutex<Tree>,
	// desc_cols: usize,
	// desc_type: u32,
	// desc_nbytes: usize,
	
	// features: FeatureInfo<T>,
	// /// for each node, its assigment vector
	// id_assignments: Mutex<HashMap<NodeId, Vec<u32>>>,
	// // ThreadSafeMap id_assigments;
	// // std::vector<std::thread> _Threads;
	// // std::atomic<bool> threadRunning[100];//do not  know how to create dinamically :S
}

impl VocabularyCreator {
	const MAX_THREADS: usize =100;

	pub fn new(params: VocabularyCreatorParams) -> Self {
		Self { params }
	}

	pub fn prefer_alignment<T: VocabElement>(&self, ncols: NonZeroUsize) -> usize {
		T::prefer_alignment(ncols)
	}
	
	/// create this from a set of features
	/// 
	/// Voc resulting vocabulary
	/// features: vector of features. Each matrix represents the features of an image.
	pub fn create<T: VocabElement>(&self, features: Vec<ndarray::Array2<T>>, desc_name: &str) -> Result<Vocabulary, CreateVocabularyError> {
		let feature0 = features.first()
			.ok_or(CreateVocabularyError::NoFeatures)?;
		let desc_cols = feature0.ncols();
		if desc_cols == 0 {
			return Err(CreateVocabularyError::EmptyFeature);
		}
		let desc_type = T::TYPE;
		// _descNBytes=features[0].cols* features[0].elemSize();

		// if(!(_descType==CV_8UC1|| _descType==CV_32FC1))
		//     throw std::runtime_error("Descriptors must be binary CV_8UC1 or float  CV_32FC1");
		// if (_descType==CV_8UC1){
		//     if (_descNBytes==32)
		//         dist_func=distance_hamming_32bytes;
		//     else
		//         dist_func=distance_hamming_generic;
		// }
		// else  dist_func=distance_float_generic;

		//create for later usage
		let features = FeatureInfo::create(features);

		//set all indices for the first level
		let mut id_assigments = HashMap::with_capacity(features.len());
		{
			let root_assign = (0..features.len())
				.map(|i| i as NodeId)
				.collect::<Vec<_>>();
			id_assigments.insert(0, root_assign);
		}

		let params = InnerParams {
			k: self.params.k,
			L: self.params.L,
			max_iters: self.params.max_iters,
			rng: StdRng::from_seed([0u8; 32]),
			features,
			desc_cols,
		};

		// Fix up nthreads
		let nthreads = {
			let max_threads = NonZeroUsize::new(Self::MAX_THREADS).unwrap();

			let nthreads = match NonZeroUsize::new(self.params.nthreads) {
				None => std::thread::available_parallelism()
					// Default to single-threaded
					.ok(),//TODO: warn?
				v => v,
			};
			match nthreads {
				None => None,
				Some(one) if one.get() == 1 => None, // Single-threaded
				Some(t) if t > max_threads => Some(max_threads),
				t => t,
			}
		};

		/*match nthreads {
			None => {
				// Single-threaded
				let stack = Vec::new();
				if let Some(iter) = self.createLevel(0, 0) {
					stack.push((iter, 0));
				}
			},
			Some(nthreads) => {
				let pool = rayon::ThreadPoolBuilder::new()
					.num_threads(nthreads.get())
					.build()
					.unwrap();

				if let Some(iter) = self.createLevel(0, 0) {
					pool.scope(|s| {
						s.spawn(body);
					})
				}
				//now, add threads
	
			}
		}*/
		todo!()
			// createLevel(0,0);

			// for(auto &t:threadRunning)t=false;
			// for(size_t i=0;i<_params.nthreads;i++)
			//     _Threads.push_back(std::thread(&VocabularyCreator::thread_consumer,this,i));
			// let mut ntimes=0;
			// while(ntimes < 10) {
			//     ntimes += 1;
			//     for(auto &t:threadRunning) if (t){ ntimes=0;break;}
			//     std::this_thread::sleep_for(std::chrono::microseconds(600));
			// }

			// //add exit info
			// for(size_t i=0;i<_Threads.size();i++) ParentDepth_ProcesQueue.push(std::make_pair(-1,-1));
			// for(std::thread &th:_Threads) th.join();

	//    std::cout<<TheTree.size()<<std::endl;
	//    for(auto &n:TheTree.getNodes())
	//        std::cout<<n.first<<" ";std::cout<<std::endl;

		//now, transform the tree into a vocabulary
		// convertIntoVoc(Voc,desc_name);
	}
	// void create(fbow::Vocabulary &Voc, const std::vector<cv::Mat> &features, const std::string &desc_name, Params params);
	// void create(fbow::Vocabulary &Voc, const cv::Mat &features, const std::string &desc_name, Params params);

	
// private:
	/*cv::Mat meanValue_binary( const std::vector<uint32_t>  &indices);
	cv::Mat meanValue_float( const std::vector<uint32_t>  &indices);

	void createLevel(const std::vector<uint32_t> &findices,  int parent=0, int curL=0);
	void createLevel(int parent=0, int curL=0, bool recursive=true);
	std::vector<uint32_t> getInitialClusterCenters(const std::vector<uint32_t> &findices);

	std::size_t vhash(const std::vector<std::vector<uint32_t> >& v_vec)  ;


	void thread_consumer(int idx);

	//for each pair of nodes, their distance
	//   std::map<uint64_t,float> features_distance;

	//


	void assignToClusters(const std::vector<uint32_t> &findices, const std::vector<cv::Mat> &center_features, std::vector<vector_sptr> &assigments, bool omp=false);
	std::vector<cv::Mat>  recomputeCenters(const std::vector<vector_sptr> &assigments, bool omp=false);
	std::size_t vhash(const std::vector<vector_sptr>& v_vec)  ;

	//------------
	void convertIntoVoc(Vocabulary &Voc, std::string dec_name);


	/**
	   * Calculates the distance between two descriptors
	   * @param a
	   * @param b
	   * @return distance
	   */
	static float distance_float_generic(const cv::Mat &a, const cv::Mat &b);
	static float distance_hamming_generic(const cv::Mat &a, const cv::Mat &b);
	static float distance_hamming_32bytes(const cv::Mat &a, const cv::Mat &b);
	std::function<float(const cv::Mat &a, const cv::Mat &b)> dist_func;*/
}