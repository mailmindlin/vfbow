use std::{borrow::Cow, fs::File, io::{self, Read, Seek, SeekFrom, Write}, path::PathBuf};

use pyo3::{exceptions::PyTypeError, prelude::*, types::{PyBytes, PyString}};
use super::dispatch::dispatch;

mod consts {
	use pyo3::prelude::*;
	use pyo3::sync::GILOnceCell;
	use pyo3::types::PyString;
	use pyo3::{intern, Bound, Py, PyResult, Python};

	macro_rules! intern_strings {
		{$($name:ident = $value:literal),*} => {
			$(
				pub(super) fn $name<'py>(py: Python<'py>) -> &'py Bound<'py, PyString> {
					intern!(py, $value)
				}
			)*
		};
	}

	intern_strings! {
		fileno = "fileno",
		read = "read",
		write = "write",
		seek = "seek",
		tell = "tell",
		flush = "flush"
	}

	macro_rules! intern_types {
		{
			from $module:literal import (
				$(
					$item:literal as $name:ident
				),*
				$(,)?
			);
			$($tail:tt)*
		} => {

			$(
				pub(super) fn $name<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyAny>> {
					static INSTANCE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
			
					INSTANCE
						.get_or_try_init(py, || {
							const MODULE: &'static str = $module;
							const ITEM: &'static str = $item;
							let io = PyModule::import(py, MODULE)?;
							let cls = io.getattr(ITEM)?;
							Ok(cls.unbind())
						})
						.map(|x| x.bind(py))
				}
			)*
		};
		{
			from $module:literal import $item:literal as $name:ident;
			$($tail:tt)*
		} => {
			pub(super) fn $name<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyAny>> {
				static INSTANCE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
		
				INSTANCE
					.get_or_try_init(py, || {
						const MODULE: &'static str = $module;
						const ITEM: &'static str = $item;
						let io = PyModule::import_bound(py, MODULE)?;
						let cls = io.getattr(ITEM)?;
						Ok(cls.unbind())
					})
					.map(|x| x.bind(py))
			}
		};
		{} => {};
	}

	intern_types!{
		from "io" import (
			"IOBase" as io_base,
			"RawIOBase" as raw_io_base,
			"BufferedIOBase" as buffered_io_base,
			"FileIO" as file_io,
			"BytesIO" as bytes_io,
			"TextIOBase" as text_io_base,
			"TextIOWrapper" as text_io_wrapper,
			"StringIO" as string_io,
		);
	}

}

#[derive(Clone, Copy, Debug)]
enum KnownIOBase {
	Unknown,
	IOBase,
	RawIOBase,
	BufferedIOBase,
	FileIO,
	BytesIO,
	// BufferedReader,
	// BufferedWriter,
	// BufferedRandom,
	// BufferedRWPair,
	TextIOBase,
	TextIOWrapper,
	StringIO,
}

impl KnownIOBase {
	fn check(obj: &Bound<PyAny>) -> Self {
		macro_rules! check_instance {
			($obj:ident is $ty:ident) => {
				{
					let obj = $obj;
					let py = obj.py();
					consts::$ty(py)
						.and_then(|ty| obj.is_instance(ty))
				}
			};
			($obj:ident is! $ty:ident) => {
				{
					let obj = $obj;
					let py = obj.py();
					match consts::$ty(py) {
						Ok(ty) => obj.is_instance(ty).unwrap_or(false),
						Err(..) => false,
					}
				}
			};
		}
		if let Ok(false) | Err(..) = check_instance!(obj is io_base) {
			return Self::Unknown;
		}
		// We assume that there isn't any multiple inheritance
		if check_instance!(obj is! raw_io_base) {
			if check_instance!(obj is! file_io) {
				return Self::FileIO;
			}
		}
		todo!()
	}
}

enum PyCapability {
	Readable,
	Writable,
	Seekable,
}

#[derive(Clone, Copy, Debug)]
struct PyIOCapabilities {
	base: Option<KnownIOBase>,
	/// Is the inner type text-based
	text: bool,
}

/// PyIO that is bound to the GIL.
/// 
/// Faster than [PyIO] for multiple operations.
#[derive(Debug)]
struct PyIOBound<'py> {
	inner: Cow<'py, Bound<'py, PyAny>>,
	capabilities: PyIOCapabilities,
}

trait MapPyError {
	type Value;
	type Error: Into<PyErr>;
	fn map_py_err(self, py: Python<'_>, f: impl Fn(&Self::Error) -> PyErr) -> PyResult<Self::Value>;
}
impl<V, E: Into<PyErr>> MapPyError for Result<V, E> {
	type Value = V;
	type Error = E;

	fn map_py_err(self, py: Python<'_>, f: impl Fn(&Self::Error) -> PyErr) -> PyResult<Self::Value> {
		self.map_err(|e| {
			let e1 = f(&e);
			e1.set_cause(py, Some(e.into()));
			e1
		})
	}
}

impl<'py> PyIOBound<'py> {
	pub(super) fn require_attr(&self, attr_name: &Bound<PyString>) -> PyResult<()> {
		if !self.inner.hasattr(attr_name)? {
			return Err(PyTypeError::new_err(format!("Object does not have a .{attr_name}() method.")));
		}
		Ok(())
	}

	fn require_readable(&self) -> PyResult<()> {
		todo!()
	}

	fn require_writable(&self) -> PyResult<()> {
		todo!()
	}

	fn py(&self) -> Python<'py> {
		self.inner.py()
	}

	fn unbind(self) -> PyIO {
		PyIO {
			inner: self.inner
				.into_owned()
				.unbind(),
			capabilities: self.capabilities,
		}
	}

	fn py_read(&self, len: Option<usize>) -> io::Result<Bound<'py, PyBytes>> {
		let read = consts::read(self.py());
		if self.capabilities.text {
			todo!()
			// if buf.len() < 4 {
			// 	return Err(io::Error::new(
			// 		io::ErrorKind::InvalidInput,
			// 		"buffer size must be at least 4 bytes",
			// 	));
			// }
			// let res = inner.call_method1(consts::read(py), (buf.len() / 4,))?;
			// let rust_string = res.extract::<Cow<str>>()?;
			// let bytes = rust_string.as_bytes();
			// buf.write_all(bytes)?;
			// Ok(bytes.len())
		} else {
			self.inner.call_method1(read, (len,))?
				.downcast_into::<PyBytes>()
				.map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))
		}
	}

	fn py_write(&self, buf: &[u8]) -> io::Result<usize> {
		let py = self.py();
		let arg = if self.capabilities.text {
			let s = std::str::from_utf8(buf)
				.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Tried to write non-utf8 data to a TextIO object: {e}")))?;
			PyString::new_bound(py, s).into_any()
		} else {
			PyBytes::new_bound(py, buf).into_any()
		};

		let number_bytes_written = self.inner.call_method1(consts::write(py), (arg,))?;

		if number_bytes_written.is_none() {
			return Err(io::Error::new(
				io::ErrorKind::Other,
				"write() returned None, expected number of bytes written",
			));
		}

		number_bytes_written.extract().map_err(io::Error::from)
	}

	fn py_flush(&self) -> io::Result<()> {
		let py = self.py();
		self.inner.call_method0(consts::flush(py))?;
		Ok(())
	}

	#[cfg(unix)]
	fn py_as_raw_fd(&self) -> PyResult<std::os::fd::RawFd> {
		let py = self.py();
		self.inner.call_method0(consts::fileno(py))
			.map_py_err(py, |_| PyErr::new::<PyTypeError, _>("Object does not have a fileno() method."))?
			.extract()
			.map_py_err(py, |_| PyErr::new::<PyTypeError, _>("File descriptor is not an integer."))
	}

	// pub fn py_clone(&self, py: Python<'_>) -> PyFileLikeObject {
	// 	PyFileLikeObject {
	// 		inner: self.inner.clone_ref(py),
	// 		is_text_io: self.is_text_io,
	// 	}
	// }
}

impl<'py> Seek for PyIOBound<'py> {
	fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(offset) => (0, offset as i64),
            SeekFrom::End(offset) => (2, offset),
            SeekFrom::Current(offset) => (1, offset),
        };

        let res = self.inner.call_method1(consts::seek(self.py()), (offset, whence))?;
        res.extract().map_err(io::Error::from)
	}

	fn stream_position(&mut self) -> io::Result<u64> {
		let res = self.inner.call_method0(consts::tell(self.py()))?;
		Ok(res.extract()?)
	}
}

impl<'py> Read for PyIOBound<'py> {
	fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
		//TODO: maybe wrap buf in a memoryview and use readinto?
		let bytes = self.py_read(Some(buf.len()))?;
		let src = bytes.as_bytes();
		assert!(src.len() <= buf.len(), "Read overflow");
		buf[..src.len()].copy_from_slice(src);
		Ok(src.len())
	}

	fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
		let pybytes = self.py_read(None)?;
		let bytes = pybytes.as_bytes();
		buf.extend_from_slice(bytes);
		Ok(bytes.len())
	}
}
impl<'py> Write for PyIOBound<'py> {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
		//TODO: prevent copy with memoryview?
		self.py_write(buf)
	}

	fn flush(&mut self) -> io::Result<()> {
		self.py_flush()
	}
}

impl<'py> FromPyObject<'py> for PyIOBound<'py> {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		let text_io = consts::text_io_base(obj.py())?;
		let is_text_io = obj.is_instance(text_io)?;
		Ok(Self {
			inner: Cow::Owned(obj.clone()),
			capabilities: PyIOCapabilities {
				base: None,
				text: is_text_io,
			},
		})
	}
}

#[derive(Debug)]
pub(super) struct PyIO {
	inner: Py<PyAny>,
	capabilities: PyIOCapabilities,
}

impl PyIO {
	fn bind<'a>(&'a self, py: Python<'a>) -> PyIOBound<'a> {
		PyIOBound { inner: Cow::Borrowed(self.inner.bind(py)), capabilities: self.capabilities }
	}
}

impl Read for PyIO {
	fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
		Python::with_gil(|py| self.bind(py).read(buf))
	}
	fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
		Python::with_gil(|py| self.bind(py).read_vectored(bufs))
	}
	fn is_read_vectored(&self) -> bool {
		false
	}
	fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
		Python::with_gil(|py| self.bind(py).read_to_end(buf))
	}
	fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
		Python::with_gil(|py| self.bind(py).read_to_string(buf))
	}
	fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
		Python::with_gil(|py| self.bind(py).read_exact(buf))
	}
}
impl Write for PyIO {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
		Python::with_gil(|py| self.bind(py).write(buf))
	}
	fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
		Python::with_gil(|py| self.bind(py).write_all(buf))
	}
	fn flush(&mut self) -> io::Result<()> {
		Python::with_gil(|py| self.bind(py).flush())
	}
}

/// Represents either a path `Path` or a file-like object `FileLike`
#[derive(Debug)]
enum PyPathOrIO<'py> {
	Path(PathBuf),
	FileLike(PyIOBound<'py>),
}

impl<'py> PyPathOrIO<'py> {
	pub(super) fn from_bound(path_or_file_like: &Bound<'py, PyAny>, read: bool, write: bool) -> PyResult<Self> {
		// Check if it's a path
		let _e1 = match path_or_file_like.extract::<PathBuf>() {
			Ok(path) => return Ok(Self::Path(path)),
			Err(e) => e,
		};

		let _e2 = match path_or_file_like.downcast::<PyString>() {
			Ok(string_ref) => {
				let string = string_ref.to_string_lossy().to_string();
				return Ok(Self::Path(string.into()));
			}
			Err(e) => e,
		};

		let v = path_or_file_like.extract::<PyIOBound>()?;
		if read {
			v.require_readable()?;
		}
		if write {
			v.require_writable()?;
		}
		Ok(Self::FileLike(v))
	}
}

/// Python argument that we can adapt to [std::io::Read]
pub(super) enum PyRead {
	Native(File),
	Wrapped(PyIO),
}

impl Read for PyRead {
	dispatch!{
		(Native,Wrapped)
		fn read(&mut self,buf: &mut [u8]) -> std::io::Result<usize>;
		fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize>;
		fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize>;
		fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize>;
		fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()>;
	}
}

impl<'py> FromPyObject<'py> for PyRead {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		Ok(match PyPathOrIO::from_bound(ob, true, false)? {
			PyPathOrIO::Path(path_buf) => {
				// Allocate the buffer *first* so we don't affect the filesystem otherwise.
				let file = File::open(path_buf)?;
				Self::Native(file)
			},
			PyPathOrIO::FileLike(f) => Self::Wrapped(f.unbind()),
		})
	}
}

/// Python argument that we can adapt to [std::io::Write]
pub(super) enum PyWrite {
	Native(File),
	Wrapped(PyIO),
}

impl<'py> FromPyObject<'py> for PyWrite {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		Ok(match PyPathOrIO::from_bound(ob, false, true)? {
			PyPathOrIO::Path(path_buf) => {
				// Allocate the buffer *first* so we don't affect the filesystem otherwise.
				let file = File::open(path_buf)?;
				Self::Native(file)
			},
			PyPathOrIO::FileLike(f) => Self::Wrapped(f.unbind()),
		})
	}
}

impl Write for PyWrite {
	dispatch!{
		(Native,Wrapped)
		fn write(&mut self,buf: &[u8])->std::io::Result<usize>;
		fn flush(&mut self)->std::io::Result<()>;
	}
}