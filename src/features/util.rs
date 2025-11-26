//! Utility functions

/// Generate pairs of functions, one that works on slices and one that works on arrays of a given size.
/// 
/// We want the array specialization because the compiler can often better optimize code when the size is known.
macro_rules! specialize_array {
    {
        $(#[$attr:meta])*
        $visibility:vis
        fn $fn_name:ident <$N:ident> ( $arg1:ident : &[$ty1:ty], $arg2:ident : &[$ty2:ty] ) -> $ret_type:ty $body:block
    } => {
        #[doc = "Specialized functions for slice and array inputs"]
        $visibility mod $fn_name {
            #[allow(unused_imports)]
            use super::*;

            $(#[$attr])*
            pub fn slice($arg1 : &[$ty1], $arg2 : &[$ty2]) -> $ret_type {
                assert_eq!($arg1.len(), $arg2.len());
                #[allow(non_snake_case, unused)]
                let $N: usize = $arg1.len();

                $body
            }
            $(#[$attr])*
            pub fn array<const $N: usize>( $arg1 : &[$ty1; $N], $arg2 : &[$ty2; $N] ) -> $ret_type {
                $body
            }
        }
    };
}

/// Constant assertion that `T` is not a ZST
pub(super) const fn assert_not_zst<T: Sized>() {
    let () = assert!(size_of::<T>() != 0, "ZST not allowed");
}

/// Constant assertion that `align_of<T>()` is greater than or equal to `align_of<U>()`
pub(super) const fn assert_alignment_geq<T: Sized, U: Sized>() {
    let () = assert!(align_of::<T>() >= align_of::<U>(), "Alignment must be greater");
}

/// Constant assertion that `size_of<T>()` is a multiple of `size_of<U>()`
pub(super) const fn assert_size_multiple<T: Sized, U: Sized>() {
    let () = assert!(size_of::<T>().is_multiple_of(size_of::<U>()), "size_of<T>() must be a multiple of size_of<U>()");
}

/// Constant assertion that any array `[T; N]` or slice `[T]` is packed
pub(super) const fn assert_array_packed<T: Sized>() {
    let () = assert!(super::shared::is_slice_packed::<T>(), "Array is not packed");
}

pub(super) use specialize_array;