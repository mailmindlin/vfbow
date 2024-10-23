use ndarray::{CowArray, Ix1};

#[derive(Debug)]
pub(super) struct TerminalLeaf<'a, T> {
    /// Feature of this node
	pub(super) feature: CowArray<'a, T, Ix1>,
	/// index of the feature this node represent(only if leaf and it stop because not enough points to create a new leave.
	/// In case the node is a terminal point, but has many points beloging to its cluster, then, this is not set.
	/// In other words, it is only used in nn search problems where L=-1
	pub(super) feat_idx: u32,
}

#[derive(Debug)]
pub(super) struct Branch<'a, T>{
	pub(super) feature: CowArray<'a, T, Ix1>,
	pub(super) children: Vec<Node<'a, T>>,
}

#[derive(Debug)]
pub(super) struct TerminalBranch<'a, T> {
	pub(super) feature: CowArray<'a, T, Ix1>,
	pub(super) children: Vec<TerminalLeaf<'a, T>>,
}


pub(super) enum BranchNode<'a, T> {
    Terminal(Vec<TerminalLeaf<'a, T>>),
    Intermediate(Vec<Leaf<'a, T>>),
}

/// Leaf that may be expanded
#[derive(Debug)]
pub(super) struct Leaf<'a, T> {
	/// Feature of this node
	pub(super) feature: CowArray<'a, T, Ix1>,
	/// if leaf, its weight and the word id
	pub(super) findices: Vec<u32>,
}

#[derive(Debug)]
pub(super) enum Node<'a, T> {
	/// Node with children
	Branch(Branch<'a, T>),
	/// Leaf that *could* be expanded in the future
	Leaf(Leaf<'a, T>),
	Terminal(TerminalLeaf<'a, T>),
	TerminalBranch(TerminalBranch<'a, T>),
}

impl<'a, T> Node<'a, T> {
	pub(super) const fn is_leaf(&self) -> bool {
		match self {
			Self::Branch(..) | Self::TerminalBranch(..) => false,
			Self::Leaf(..) | Self::Terminal(..) => true,
		}
	}
}