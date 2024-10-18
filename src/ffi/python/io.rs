use std::{fs::File, io::{Read, Write}, path::PathBuf};

use pyo3::{prelude::*, types::PyString};
use pyo3_file::PyFileLikeObject;
use super::dispatch::dispatch;

/// Represents either a path `Path` or a file-like object `FileLike`
#[derive(Debug)]
enum FileOrFileLike {
	Path(PathBuf),
	FileLike(PyFileLikeObject),
}

impl FileOrFileLike {
	pub(super) fn from_pyobject(path_or_file_like: PyObject, read: bool, write: bool) -> PyResult<FileOrFileLike> {
		Python::with_gil(|py| {
			// is a path
			if let Ok(string_ref) = path_or_file_like.downcast_bound::<PyString>(py) {
				let string = string_ref.to_string_lossy().to_string();
				return Ok(FileOrFileLike::Path(string.into()));
			}

			//TODO: support pathlib.Path

			// is a file-like
			match PyFileLikeObject::with_requirements(path_or_file_like, read, write, false, false) {
				Ok(f) => Ok(FileOrFileLike::FileLike(f)),
				Err(e) => Err(e)
			}
		})
	}

	pub(super) fn from_bound<'py>(path_or_file_like: &Bound<'py, PyAny>, read: bool, write: bool) -> PyResult<FileOrFileLike> {
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

		match PyFileLikeObject::py_with_requirements(path_or_file_like.clone(), read, write, false, false) {
			Ok(f) => return Ok(Self::FileLike(f)),
			Err(e) => {
				//TODO: do we do anything with e1/e2?
				Err(e)
			}
		}
	}
}

pub(super) enum PyRead {
	Native(File),
	Wrapped(PyFileLikeObject),
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
		Ok(match FileOrFileLike::from_bound(ob, true, false)? {
			FileOrFileLike::Path(path_buf) => {
				// Allocate the buffer *first* so we don't affect the filesystem otherwise.
				let file = File::open(path_buf)?;
				Self::Native(file)
			},
			FileOrFileLike::FileLike(f) => Self::Wrapped(f),
		})
	}
}

pub(super) enum PyWrite {
	Native(File),
	Wrapped(PyFileLikeObject),
}

impl<'py> FromPyObject<'py> for PyWrite {
	fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
		Ok(match FileOrFileLike::from_bound(ob, false, true)? {
			FileOrFileLike::Path(path_buf) => {
				// Allocate the buffer *first* so we don't affect the filesystem otherwise.
				let file = File::open(path_buf)?;
				Self::Native(file)
			},
			FileOrFileLike::FileLike(f) => Self::Wrapped(f),
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