//! Helpers for converting byte slices

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Source slice length is not a multiple of element size")]
pub(crate) struct InvalidChunkSizeError;


/// Reinterprets some bytes as a type by splitting into chunks of the size of the target type
/// 
/// Example:
// Something simiar to this test is below, but we can't use a doctest because it's private
/// ```ignore
/// # use vfbow::doctest_exports::convert_chunks;
/// let bytes = 0f32.to_le_bytes();
/// let result = convert_chunks::<f32, _>(&bytes, f32::from_le_bytes);
/// assert_eq!(&result, &[0f32]);
/// ```
//TODO: Use something like bytemuck to make this a no-op when possible
fn convert_chunks<'a, T, const N: usize, F: 'a + Fn([u8; N]) -> T>(src: &'a [u8], f: F) -> Result<impl Iterator<Item = T> + 'a, InvalidChunkSizeError> {
    assert_eq!(N, size_of::<T>());

    let (chunks, remainder) = src.as_chunks::<N>();
    if !remainder.is_empty() {
        return Err(InvalidChunkSizeError);
    }

    Ok(
        chunks
        .into_iter()
        .copied()
        .map(f)
    )
}


pub(crate) fn convert_le<'a, T: 'a, const N: usize>(src: &'a [u8]) -> Result<impl Iterator<Item = T> + 'a, InvalidChunkSizeError>
where
    T: FromLEBytes<Bytes = [u8; N]>,
{
    debug_assert_eq!(N, size_of::<T>());

    convert_chunks(src, T::from_le_bytes)
}

#[test]
fn test_convert_f32() {
    let bytes = [0, 0, 0, 0, 0, 0, 0, 64]; // two f32: 0.0 and 2.0
    let result: Vec<f32> = convert_le(&bytes).unwrap().collect();
    assert_eq!(&result, &[0.0f32, 2.0f32]);
}

#[test]
fn test_convert_u32() {
    let bytes = [1, 0, 0, 0, 2, 0, 0, 0]; // two u32: 1 and 2
    let result: Vec<u32> = convert_le(&bytes)
        .unwrap()
        .collect();
    assert_eq!(&result, &[1u32, 2u32]);
}

#[test]
fn test_convert_u64() {
    let bytes = [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0xFF]; // two u64: 1 and 2
    let result: Vec<u64> = convert_le(&bytes)
        .unwrap()
        .collect();
    assert_eq!(&result, &[1u64, 0xFF00000000000002u64]);
}

pub(crate) trait FromLEBytes {
    type Bytes;
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
}

impl FromLEBytes for u32 {
    type Bytes = [u8; 4];
    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        u32::from_le_bytes(bytes)
    }
}

impl FromLEBytes for f32 {
    type Bytes = [u8; 4];
    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        f32::from_le_bytes(bytes)
    }
}

impl FromLEBytes for u64 {
    type Bytes = [u8; 8];
    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        u64::from_le_bytes(bytes)
    }
}