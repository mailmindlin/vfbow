//! Helper macros for dispatching over enum variants

//TODO: Maybe just include this in 
/// Used by [_dispatch_one] to generate the 
macro_rules! _dispatch_case {
	{($base:expr) ($($variant:path),+) over $name:ident => $value:expr} => {
		match $base {
			$(
				$variant($name) => { $value }
			),+
		}
	};
}

/// Used by [dispatch] to generate a single dispatch method.
macro_rules! _dispatch_one {
	{
		($($variant:ident),+)
		fn $name:ident(&mut self $(, $($argn:ident: $argty:ty),+)?) -> $result:ty;
	} => {
		fn $name(&mut self $(, $($argn: $argty),+)?) -> $result {
			$crate::ffi::python::dispatch::_dispatch_case!{(self) ($(Self::$variant),+) over me => me.$name($($($argn),*)?)}
		}
	};
}

//TODO: Support methods not starting with `&mut self` (maybe convert to `self: &mut Self`?)
/// Generate enum dispatch implementations.
/// 
/// Only works on enums with 
/// 
/// ## Pattern
/// - Tuple of enum variant names
/// - Empty method signatures (like defining a trait)
/// 
/// ## Example
/// ```
/// use std::io;
/// enum DispatchWrite<A: io::Write, B: io::Write> { A(A), B(B) }
/// 
/// impl<A: io::Write, B: io::Write> io::Write for DispatchWrite<A, B> {
///     dispatch! {
///         (A, B)
///         fn write(&mut self, buf: &[u8]) -> io::Result<usize>;
///         fn flush(&mut self) -> io::Result<()>;
///     }
/// }
/// ```
macro_rules! dispatch {
	// Empty case
	{
		($($variant:ident),+)
	} => {};
	{
		($($variant:ident),+)
		fn $name:ident(&mut self $(, $($argn:ident: $argty:ty),+)?) -> $result:ty;
		$(fn $nameN:ident(&mut self $(, $($argnN:ident: $argtyN:ty),+)?) -> $resultN:ty;)*
	} => {
		$crate::ffi::python::dispatch::_dispatch_one! {
			($($variant),+)
			fn $name(&mut self $(, $($argn: $argty),+)?) -> $result;
		}
		$crate::ffi::python::dispatch::dispatch! {
			($($variant),+)
			$(
				fn $nameN(&mut self $(, $($argnN: $argtyN),+)?) -> $resultN;
			)*
		}
	};
}

/// 
/// impl<A: ToString, B: ToString> ToString for DispatchToString<A, B> {
///     dispatch! {
///         (A, B)
///         fn to_string(&self) -> String;
///     }
/// }

pub(super) use {dispatch, _dispatch_one, _dispatch_case};