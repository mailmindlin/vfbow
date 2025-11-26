mod dtype;
pub(crate) mod serde;
pub(crate) mod scoring;
pub(crate) mod convert;
pub(crate) use dtype::DescriptorType;
pub use scoring::Scoring;
pub use serde::{Serialize, Deserialize};

/// A type that can hash itself
pub trait SelfHash {
	/// Returns an identifying hash
	fn hash(&self) -> u64;
}