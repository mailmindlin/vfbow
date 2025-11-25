use std::cell::{Cell, RefCell};

use ndarray::ArrayView1;

use crate::features::{FeatureType, Features, TypedFeatures};

use super::{Node, Vocabulary, VocabularyParams};

/// Data shared between all [NodeBuilder]s
struct BuilderShared<T: FeatureType> {
	/// Next node id
	next_id: Cell<u32>,
	/// Feature storage
	features: RefCell<TypedFeatures<T>>,
}

impl<T: FeatureType> BuilderShared<T> {
	fn next_id(&self, n: usize) -> u32 {
		let result = self.next_id.get();
		// Overflow checks
		let next_id = result.checked_add(
			n.try_into()
			// N is too big to fit in u32
			.unwrap()
		)
			// result + n overflows u32
		.unwrap();
		self.next_id.set(next_id);
		result
	}
}

pub(crate) struct NodeBuilder<'a, T: FeatureType> {
	shared: &'a BuilderShared<T>,
	node: Option<&'a mut Node>,
}

impl<'a, T: FeatureType> NodeBuilder<'a, T> {
	pub(crate) fn fill<'b, U, F>(mut self, branch_feats: Vec<U>, branch_feat: F, leaf_feats: impl ExactSizeIterator<Item = ArrayView1<'b, T>>) -> Option<impl Iterator<Item = (U, NodeBuilder<'a, T>)>>
		where
			T: 'b,
			F: for<'c> Fn(&'c U) -> ArrayView1<'c, T>
	{
		if branch_feats.is_empty() {
			self.fill_leaf(leaf_feats);
			return None;
		}

		let node = self.node.take().expect("Called fill twice");

		let n = branch_feats.len() + leaf_feats.len();
		node.base = self.shared.next_id(n);
		node.n = n as _;
		node.children = Some(
			branch_feats
				.iter()
				.map(|_| Node::empty())
				.collect()
		);
		{
			// Insert features
			let mut features = self.shared.features.borrow_mut();
			//TODO: is it worth chaining the iterators together here?
			features.insert(branch_feats.iter().map(branch_feat));
			features.insert(leaf_feats);
		}

		let shared = self.shared;
		let iter = node.children
			.as_mut()
			.unwrap()
			.iter_mut()
			.zip(branch_feats)
			.map(|(dst, src) | (src, NodeBuilder { node: Some(dst), shared }));
		Some(iter)
	}

	pub(crate) fn fill_leaf<'b>(mut self, leaf_feats: impl ExactSizeIterator<Item = ArrayView1<'b, T>>) where T: 'b {
		let node = self.node.take().expect("Called fill twice");

		node.base = self.shared.next_id(leaf_feats.len());
		node.n = leaf_feats.len() as _;
		node.children = None;
		//TODO: insert features
		let mut features = self.shared.features.borrow_mut();
		features.insert(leaf_feats);
	}
}

impl<'a, T: FeatureType> Drop for NodeBuilder<'a, T> {
	fn drop(&mut self) {
		if self.node.is_some() {
			panic!("Didn't fill node");
		}
	}
}

pub(crate) struct VocabularyBuilder<T: FeatureType> {
	shared: BuilderShared<T>,
	root: Node,
}

impl<T: FeatureType> VocabularyBuilder<T> {
	pub(crate) fn new(capacity: usize, feature_len: usize) -> Self {
		Self {
			shared: BuilderShared {
				next_id: Cell::new(0),
				features: RefCell::new(TypedFeatures::<T>::new(capacity, feature_len)),
			},
			root: Node { base: 0, n: 0, children: None },
		}
	}

	pub(crate) fn root(&mut self) -> NodeBuilder<'_, T> {
		NodeBuilder { shared: &self.shared, node: Some(&mut self.root) }
	}

	pub(crate) fn finish(self, params: VocabularyParams) -> Vocabulary {
		Vocabulary {
			params,
			root: self.root,
			features: self.shared.features.into_inner().into(),
		}
	}
}