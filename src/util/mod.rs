mod dtype;
pub(crate) mod serde;
pub(crate) use dtype::DescriptorType;
pub use dtype::Scoring;
pub use serde::{Serialize, Deserialize};

pub trait SelfHash {
	/// returns a hash identifying this
	fn hash(&self) -> u64;
}