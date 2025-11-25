
use ndarray::CowArray;
use numpy::Ix1;

use super::{FeatureType, Node, Vocabulary};

#[derive(Clone)]
pub struct NodePath<'a> {
    vocab: &'a Vocabulary,
    path: Vec<&'a Node>,
    child_offset: Option<u32>,
}
impl<'a> NodePath<'a> {
    pub(super) fn new(vocab: &'a Vocabulary, path: Vec<&'a Node>, child_offset: Option<u32>) -> Self {
        Self { vocab, path, child_offset }
    }
    pub fn as_ref(&self) -> NodeRef<'_> {
        NodeRef {
            vocab: self.vocab,
            path: &self.path,
            child_offset: self.child_offset,
        }
    }
}

#[derive(Copy, Clone)]
pub struct NodeRef<'a> {
    vocab: &'a Vocabulary,
    // Invariant: must not be empty
    path: &'a [&'a Node],
    child_offset: Option<u32>,
}

impl<'a> NodeRef<'a> {
    pub fn parent(&self) -> Option<Self> {
        let path = match self.child_offset {
            Some(..) => self.path,
            None if self.path.len() == 1 => return None,
            None => &self.path[..self.path.len() - 2],
        };
        Some(Self { vocab: self.vocab, path, child_offset: None })
    }
    // pub fn ancestor(&self, n: usize) -> Option<Self> {
    //     todo!()
    // }
    pub fn word_ids(&self) -> Vec<u32> {
        let node = *self.path.last().unwrap();
        match self.child_offset {
            Some(child_offset) => {
                let child_idx = node.base + child_offset;
                vec![child_idx]
            },
            None => {
                // Branch node
                (0..node.n)
                    .map(|child_offset| node.base + child_offset)
                    .collect()
            }
        }
    }
    #[allow(private_bounds)]
    pub fn words<T: FeatureType>(&'_ self) -> Vec<CowArray<'_, T, Ix1>> {
        let node = *self.path.last().unwrap();
        match self.child_offset {
            Some(child_offset) => {
                let child_idx = node.base + child_offset;
                let feature = self.vocab.features.get(child_idx as usize).unwrap();
                vec![feature]
            },
            None => {
                // Branch node
                (0..node.n)
                    .map(|child_offset| node.base + child_offset)
                    .map(|child_idx| self.vocab.features.get(child_idx as usize).unwrap())
                    .collect()
            }
        }
    }
}