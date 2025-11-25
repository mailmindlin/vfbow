mod fbow;

use std::{collections::VecDeque, io::{self, Read, Write}};

use crate::{features::FeaturesGeneric, util::{serde::{write_u32, write_u32ish}, DescriptorType}, vocabulary::VocabularyParams, Deserialize, Serialize};

use super::Vocabulary;

const FBOW_MAGIC: u64 = 55824124;
const VFBOW_MAGIC: u64 = 5546449913086866150;

// We write in VFBOW format
impl Serialize for Vocabulary {
	fn write_to(&self, mut dst: impl Write) -> std::io::Result<()> {
		//magic number
		dst.write_all(&VFBOW_MAGIC.to_le_bytes())?;
		//save string
		self.params.write_to(&mut dst)?;

		self.features.write_to(&mut dst)?;

		// Write node data
		let mut queue = VecDeque::new();
		queue.push_back(&self.root);
		while let Some(node) = queue.pop_front() {
			write_u32(node.base, &mut dst)?;
			// Write variable-length values
			write_u32(node.n, &mut dst)?;
			if let Some(children) = &node.children {
				write_u32ish(children.len(), &mut dst)?;
				for child in children.iter() {
					todo!("Write child node reference");
				}
			} else {
				write_u32ish(0, &mut dst)?;
			}
		}
		Ok(())
		// str.write((char*)&_params,sizeof(params));
		// str.write(_data.get(), _params._total_size);
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", eq))]
pub enum ParseValidationMode {
	/// Silently accept invalid data
	Ignore,
	/// Print a message to stdout
	Warn,
	/// Return an error on invalid data
	Strict,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="python", derive(pyo3::FromPyObject))]
#[non_exhaustive]
pub struct VocabularyReadOptions {
	pub too_many_children: ParseValidationMode,
	pub inconsistent_block: ParseValidationMode,
}

impl VocabularyReadOptions {
	pub const fn all(mode: ParseValidationMode) -> Self {
		Self {
			too_many_children: mode,
			inconsistent_block: mode,
		}
	}
}

impl Default for VocabularyReadOptions {
	fn default() -> Self {
		Self::all(
			if cfg!(debug_assertions) {
				ParseValidationMode::Warn
			} else {
				ParseValidationMode::Ignore
			}
		)
	}
}

impl Vocabulary {
	/// Deserialize vocabulary
	/// 
	/// Supports vfbow and fbow formats
	pub fn read_from(mut src: impl Read, options: VocabularyReadOptions) -> std::io::Result<Self> {
		let magic = {
			let mut sig_buf = [0u8; size_of::<u64>()];
			src.read_exact(&mut sig_buf)?;
			u64::from_le_bytes(sig_buf)
		};
		match magic {
			FBOW_MAGIC => Self::read_fbow(src, options.into()),
			VFBOW_MAGIC => {
				//save string
				let params = VocabularyParams::read_from(&mut src)?;
				let features = FeaturesGeneric::read_from(src)?;
				//TODO: consistency check
				// let mut builder = VocabularyBuilder::<u8>::new(params.nblocks as _, params.desc_size as _);
				
				// Read nodes
				todo!("Parse nodes")
			},
			_ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid signature {magic:#08x}"))),
		}
	}
}