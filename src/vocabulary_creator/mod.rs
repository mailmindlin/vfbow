mod exec;
mod feature;
mod node;
mod specialization;

use std::{collections::VecDeque, mem, num::NonZeroUsize, sync::Mutex};

use feature::FeatureInfo;
use ndarray::{CowArray, Ix1};
use node::{Branch, Leaf, Node, TerminalBranch, TerminalLeaf};
use rand::{rngs::StdRng, SeedableRng};
use rayon::ScopeFifo;
pub(crate) use specialization::VocabElement;

use crate::vocabulary::{Vocabulary, VocabularyBuilder, VocabularyParams};


/// Parameters for [creating](VocabularyCreator) a [Vocabulary]
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", get_all, set_all, eq))]
#[derive(Clone, Debug, PartialEq, Hash)]
#[allow(non_snake_case)]
pub struct VocabularyCreatorParams {
	/// Braching factor
	pub k: u32,
	/// Maximum tree depth
	pub L: Option<u32>,
	/// Number of threads to use while computing
	/// 
	/// 0 => system autodetect
	/// 1 => single-threaded
	pub nthreads: usize,
	/// Maximum number of k-means iterations
	pub max_iters: usize,
	/// Print debug information
	pub verbose: bool,
}

impl Default for VocabularyCreatorParams {
	fn default() -> Self {
		// Note: changes here should also be made to [VocabularyCreatorParams::__init__] for python FFI
		Self {
			k: 32,
			L: None,
			nthreads: 1,
			max_iters: 11,
			verbose: false,
		}
	}
}

#[allow(non_snake_case)]
struct InnerParams<'a, T> {
	params: &'a VocabularyCreatorParams,
	rng: Mutex<StdRng>,
	features: FeatureInfo<T>,
}

fn empty_feature<'a, T: Clone>() -> CowArray<'a, T, Ix1> {
	ndarray::aview1(&[]).into()
}

/// Errors that can occur when creating a vocabulary
/// 
/// See [VocabularyCreator::create]
#[derive(Clone, Debug, thiserror::Error)]
pub enum CreateVocabularyError {
	/// No features (the input list was empty)
	#[error("No features")]
	NoFeatures,
	/// One of the provided arrays had zero columns
	#[error("Feature 0 had no columns")]
	EmptyFeature,

	// /// One of the provided arrays had a different number of columns than the others
	// #[error("Node had too many children")]
	// TooManyChildren,

	/// One of the provided arrays had a different number of columns than the others
	#[error("One of the provided arrays has a different dimension than the others")]
	ArrayDimMismatch,
}

/// This class creates the vocabulary
pub struct VocabularyCreator {
	params: VocabularyCreatorParams,
}

impl VocabularyCreator {
	const MAX_THREADS: usize = 100;

	/// Construct with given parameters
	pub fn new(params: VocabularyCreatorParams) -> Self {
		Self { params }
	}

	fn prefer_alignment<T: VocabElement>(&self, ncols: NonZeroUsize) -> usize {
		T::prefer_alignment(ncols)
	}
	
	/// Create vocabulary from a set of features. Can either be called with [u8] or [f32] features
	/// 
	/// # Parameters
	/// features: vector of features. Each matrix represents the features of an image.
	/// desc_name: Vocabulary descriptor name
	pub fn create<T: VocabElement + Send + Sync>(&self, features: Vec<ndarray::Array2<T>>, desc_name: &str) -> Result<Vocabulary, CreateVocabularyError> {
		// Create for later usage
		let features = FeatureInfo::create(features)?;

		// Set all indices for the first level
		let root_findices = (0..features.len())
			.map(|i| i as u32)
			.collect::<Vec<_>>();

		let params = InnerParams {
			params: &self.params,
			rng: Mutex::new(StdRng::from_seed([0u8; 32])),
			features,
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
				None => None, // Error => single-threaded
				Some(one) if one.get() == 1 => None, // Single-threaded
				Some(t) if t > max_threads => Some(max_threads),
				t => t,
			}
		};
		if self.params.verbose {
			println!("Using nthreads={nthreads:?}");
		}

		trait JobQueue<'f, 'n, T> {
			fn push(&mut self, depth: usize, node: &'n mut Node<'f, T>, params: &'f InnerParams<'f, T>);
		}

		impl<'f, 'n, T> JobQueue<'f, 'n, T> for Vec<(usize, &'n mut Node<'f, T>)> {
			fn push(&mut self, depth: usize, node: &'n mut Node<'f, T>, _: &InnerParams<'f, T>) {
				self.push((depth, node));
			}
		}

		impl<'f, 'n, T: VocabElement + Send + Sync> JobQueue<'f, 'n, T> for &ScopeFifo<'n> {
			fn push(&mut self, depth: usize, node: &'n mut Node<'f, T>, params: &'f InnerParams<'f, T>) {
				self.spawn_fifo(move |mut scope| {
					process(depth, node, params, &mut scope);
				});
			}
		}

		fn process<'f, 'n, T: VocabElement>(depth: usize, node: &'n mut Node<'f, T>, params: &'f InnerParams<'f, T>, queue: &mut impl JobQueue<'f, 'n, T>) {
			println!("create_level L={}", depth);
			let Node::Leaf(Leaf { feature, findices, .. }) = node else { unreachable!() };
			// Take feature out of leaf
			let feature = mem::replace(feature, empty_feature());

			match params.create_level(findices) {
				node::BranchNode::Terminal(children) => {
					println!("\tTerminal children {}", children.len());
					*node = Node::TerminalBranch(TerminalBranch {
						feature,
						children,
					});
				}
				node::BranchNode::Intermediate(children) => {
					*node = Node::Branch(Branch {
						feature,
						children: children.into_iter()
							.map(Node::Leaf)	
							.collect()
					});

					// Add children to queue
					if params.params.L.is_none_or(|max_depth| depth < max_depth as usize) {
						// Add to stack
						let Node::Branch(Branch { children, .. }) = node else { unreachable!() };

						println!("\tpush {} intermediate children", children.len());
						// Now add children to stack
						
						for child in children.iter_mut() {
							queue.push(depth+1, child, params);
						}
					} else {
						//TODO: convert to TerminalBranch, save some memory
					}
				}
			}
		}


		let mut root = Node::<T>::Leaf(Leaf {
			feature: empty_feature(),//TODO: skip this
			findices: root_findices,
		});

		let root_node = match nthreads {
			None => {
				// Single-threaded
				let mut queue = Vec::new();
				queue.push((0, &mut root));
				
				while let Some((depth, node)) = queue.pop() {
					process(depth, node, &params, &mut queue);
				}

				root
			},
			Some(nthreads) => {
				// Multi-threaded
				let pool = rayon::ThreadPoolBuilder::new()
					.num_threads(nthreads.get())
					.build()
					.unwrap();

				pool.scope_fifo(|mut scope| {
					process(0, &mut root, &params, &mut scope);
				});
				root
			}
		};

		//now, transform the tree into a vocabulary
		self.build_vocabulary(root_node, desc_name, params.features.feature_len())
	}

	fn build_vocabulary<T: VocabElement + Clone>(&self, mut root: Node<T>, desc_name: &str, desc_cols: usize) -> Result<Vocabulary, CreateVocabularyError> {
		//look for leafs and store
		//now, create the blocks

		let mut non_leaf_nodes = 0u32;
		let mut n_leaf_nodes = 0;

		// BFS
		// Simplify tree
		{
			let mut queue = VecDeque::new();
			queue.push_front(&mut root);
			while let Some(node) = queue.pop_front() {
				match node {
					Node::TerminalBranch(TerminalBranch { children, .. }) => {
						non_leaf_nodes += 1;
						n_leaf_nodes += children.len() as u32;
						// DO NOT recurse leaf nodes
					},
					Node::Terminal(..) => {
						n_leaf_nodes += 1;
					},
					Node::Leaf(Leaf { feature, .. }) => {
						// Rust doesn't make this nice
						let feature = mem::replace(feature, empty_feature());
						// Convert to terminal leaf
						*node = Node::Terminal(TerminalLeaf {
							feature,
							feat_idx: n_leaf_nodes,
						});
						n_leaf_nodes += 1;
					},
					Node::Branch(Branch { children, .. }) => {
						non_leaf_nodes += 1;
						//TODO: simplify some of these to TerminalBranch's?
						queue.extend(children);
					}
				}
			}
		}
		
		//determine the basic elements
		let v_params = {
			let mut v_params = VocabularyParams::empty();
			let alignment = T::MIN_ALIGNMENT;
			let desc_size = desc_cols * T::TYPE.element_size();
			v_params.set(alignment, self.params.k, T::TYPE, desc_size, non_leaf_nodes, desc_name);
			v_params
		};

		// TODO: check for overflow?
		let mut builder = VocabularyBuilder::new(n_leaf_nodes as usize + non_leaf_nodes as usize, desc_cols);

		//lets start
		{
			let mut queue = VecDeque::new();
			queue.push_front((root, builder.root()));
			while let Some((node, dst)) = queue.pop_front() {
				match node {
					Node::Branch(Branch { children, .. }) => {
						let (leaves, branches) = children
							.into_iter()
							.partition::<Vec<_>, _>(Node::is_leaf);
						
						let leaf_feats = leaves.iter()
							.map(|leaf| {
								let Node::Terminal(leaf) = leaf else { unreachable!("Unexpected leaf {leaf:?}") };
								leaf.feature.view()
							});
						
						let dst_children = dst.fill(branches, |branch| {
							match branch {
								Node::Branch(Branch { feature, .. }) => feature.view(),
								Node::TerminalBranch(TerminalBranch { feature, .. }) => feature.view(),
								_ => unreachable!("Unexpected leaf"),
							}
						}, leaf_feats);
						if let Some(dst_children) = dst_children {
							queue.extend(dst_children);
						}
					},
					Node::TerminalBranch(TerminalBranch { children, .. }) => {
						let leaf_feats = children.iter()
							.map(|leaf| leaf.feature.view());
						dst.fill_leaf(leaf_feats);
					},
					_ => unreachable!("Invalid node"),
				}
			}
			assert!(queue.is_empty());
		}

		Ok(builder.finish(v_params))
	}
}