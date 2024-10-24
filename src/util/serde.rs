use std::io::{Read, Write, self};

pub trait Serialize {
	/// Write to stream
	fn write_to(&self, dst: impl Write) -> io::Result<()>;
}

pub trait Deserialize: Sized {
	/// Read from stream
	fn read_from(src: impl Read) -> io::Result<Self>;
}

pub(crate) fn write_u32(value: u32, dst: &mut impl Write) -> std::io::Result<()> {
	let bytes = value.to_le_bytes();
	dst.write_all(&bytes)
}

pub(crate) fn write_u32ish(value: usize, dst: &mut impl Write) -> std::io::Result<()> {
	let value = value.try_into()
		.map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Value overflow"))?;
	write_u32(value, dst)
}

pub(crate) fn read_u32(src: &mut impl Read) -> std::io::Result<u32> {
	let mut bytes = [0u8; size_of::<u32>()];
	src.read_exact(&mut bytes)?;
	Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn read_u32ish(src: &mut impl Read) -> std::io::Result<usize> {
	let mut bytes = [0u8; size_of::<u32>()];
	src.read_exact(&mut bytes)?;
	let raw = u32::from_le_bytes(bytes);
	raw.try_into()
		.map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Integer overflow"))
}