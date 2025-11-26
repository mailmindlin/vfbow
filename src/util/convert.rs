//! Helpers for converting byte slices

/// Error from [`convert_le`] when the slice length is not a multiple of the target type size
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
        .iter()
        .copied()
        .map(f)
    )
}


/// Convert a byte slice to a iterator of values of type T, assuming little-endian byte order
/// 
/// Returns an error if the length of the slice is not a multiple of the size of T
pub(crate) fn convert_le<'a, T, const N: usize>(src: &'a [u8]) -> Result<impl Iterator<Item = T> + 'a, InvalidChunkSizeError>
where
    // Needed for the iterator lifetime
    T: 'a,
    // We can make the cleaner when generic_const_exprs is stable
    T: FromLEBytes<Bytes = [u8; N]>,
{
    // This is a static assertion that N == size_of::<T>()
    const {
        let () = assert!(N == size_of::<T>(), "N must equal size_of::<T>()");
    }
    // And this is the runtime equivalent, just to be safe
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
fn test_invalid_multiple() {
    let not_multiple_of_4 = [0, 1, 2, 3, 4, 5];
    assert!(convert_le::<u32, _>(&not_multiple_of_4).is_err(), "Should have returned an error");
    assert!(convert_le::<u32, _>(&[]).is_ok(), "Zero is valid");
}

/**
 * This tests the static assertion in [`convert_le`].
 * You should get a compile error if you enable this test.
 */
#[cfg(false)]
#[test]
fn test_invalid_args() {
    #[repr(transparent)]
    struct BadType(u32);
    impl FromLEBytes for BadType {
        type Bytes = [u8; 3];
        fn from_le_bytes(_: Self::Bytes) -> Self {
            panic!("should not be called")
        }
    }

    let _ = convert_le::<BadType, 3>(&[1,2,3,4]);
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

/// Helper trait for types that can be constructed from little-endian byte slices
pub(crate) trait FromLEBytes {
    /// Byte array type (should be `[u8; size_of::<Self>()]`)
    type Bytes;
    /// Read Self from little-endian bytes
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