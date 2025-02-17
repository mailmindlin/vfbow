use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorType {
	Uint8 = 0,
	Float32 = 5,
}

impl Display for DescriptorType {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		f.write_str(if f.alternate() {
			self.numpy_name()
		} else {
			self.name()
		})
	}
}

impl From<DescriptorType> for u32 {
	fn from(value: DescriptorType) -> Self {
		match value {
			DescriptorType::Uint8 => 0,
			DescriptorType::Float32 => 5,
		}
	}
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Invalid dtype: {0}")]
pub struct InvalidDescriptorType(pub u32);

impl TryFrom<u32> for DescriptorType {
	type Error = InvalidDescriptorType;

	fn try_from(value: u32) -> Result<Self, Self::Error> {
		match value {
			0 => Ok(Self::Uint8),
			5 => Ok(Self::Float32),
			_ => Err(InvalidDescriptorType(value)),
		}
	}
}

impl DescriptorType {
	pub(crate) const fn element_size(&self) -> usize {
		match self {
			DescriptorType::Float32 => size_of::<f32>(),
			DescriptorType::Uint8 => size_of::<u8>(),
		}
	}

	/// Rust type name (e.g., `u8`)
	pub(crate) fn name(&self) -> &'static str {
		match self {
			Self::Uint8 => "u8",
			Self::Float32 => "f32",
		}
	}

	/// Numpy name (e.g., `np.uint8`)
	pub(crate) fn numpy_name(&self) -> &'static str {
		match self {
			Self::Uint8 => "np.uint8",
			Self::Float32 => "np.float32",
		}
	}
}