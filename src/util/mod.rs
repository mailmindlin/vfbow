mod dtype;
pub(crate) mod serde;
pub(crate) mod scoring;
pub(crate) use dtype::DescriptorType;
pub use scoring::Scoring;
pub use serde::{Serialize, Deserialize};

pub trait SelfHash {
	/// returns a hash identifying this
	fn hash(&self) -> u64;
}