use std::io::{Read, Result as IOResult, Write};

pub(crate) type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorType {
    Uint8 = 0,
    Float32 = 5,
}

impl From<DescriptorType> for u32 {
    fn from(value: DescriptorType) -> Self {
        match value {
            DescriptorType::Uint8 => 0,
            DescriptorType::Float32 => 5,
        }
    }
}

impl TryFrom<u32> for DescriptorType {
    type Error = ();//TODO: better error type

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            5 => Ok(Self::Float32),
            _ => Err(()),
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
}
pub trait Serialize {
    /// Write to stream
    fn write_to(&self, dst: impl Write) -> IOResult<()>;
}

pub trait Deserialize: Sized {
    /// Read from stream
    fn read_from(src: impl Read) -> IOResult<Self>;
}

pub trait SelfHash {
    /// returns a hash identifying this
    fn hash(&self) -> u64;
}