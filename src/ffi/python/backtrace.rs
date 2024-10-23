use std::{panic::{self, BacktraceStyle, PanicHookInfo, UnwindSafe}, ptr, sync::{Arc, Mutex}};

use backtrace::Backtrace;
use pyo3::{PyResult, Python};

pub(super) struct PanicBacktrace {
	backtrace: Arc<Mutex<Option<Backtrace>>>,
	prev_hook: Option<Box<dyn Fn(&PanicHookInfo<'_>) + Send + Sync>>,
	handler_ptr: *const (dyn Fn(&PanicHookInfo<'_>) + Send + Sync + 'static),
}

impl PanicBacktrace  {
	pub(super) fn install() -> Self {
		let backtrace = Arc::new(Mutex::new(None));

		let target = Arc::downgrade(&backtrace);
		let handler: Box<dyn Fn(&PanicHookInfo<'_>) + Send + Sync + 'static> = Box::new(move |_| {
			let backtrace = Backtrace::new_unresolved();
			let target1 = match target.upgrade() {
				Some(t) => t,
				None => {
					println!("[bth] Warning: unable to access target");
					return;
				}
			};

			match target1.lock() {
				Ok(mut target) => {
					*target = Some(backtrace);
				},
				Err(_) => {
					println!("[bth] Warning: target poisoned");
				},
			};
		});
		let handler_ptr = Box::as_ptr(&handler);

		//TODO: can we tell if it was the default hook?

		let prev_hook = match panic::get_backtrace_style() {
			Some(BacktraceStyle::Short) | Some(BacktraceStyle::Full) => {
				let prev_hook = std::panic::take_hook();
				std::panic::set_hook(handler);

				Some(prev_hook)
			},
			// Disable handler
			_ => None,
		};

		
		Self {
			backtrace,
			prev_hook,
			handler_ptr,
		}
	}

	/*pub(super) fn take(&self) -> Option<Backtrace> {
		self.backtrace.lock().ok()?.take()
	}*/

	pub(super) fn catch_backtrace<F: Send + UnwindSafe + FnOnce() -> R, R: Send>(py: Python, callback: F) -> PyResult<R> {
		// Do this while we have the GIL, should be slightly safer
		// let hook = Self::install();
		match py.allow_threads(|| panic::catch_unwind(callback)) {
			Ok(r) => Ok(r),
			Err(payload) => {
				// let backtrace = hook.take();
				// drop(hook);
				let backtrace = Some(3);
				match backtrace {
					/*Some(mut backtrace) => {
						let err = if let Some(string) = payload.downcast_ref::<String>() {
							PanicException::new_err((string.clone(),))
						} else if let Some(s) = payload.downcast_ref::<&str>() {
							PanicException::new_err((s.to_string(),))
						} else {
							PanicException::new_err(("panic from Rust code",))
						};

						backtrace.resolve();;

						let frames = backtrace.frames();
						if let Some(tb) = err.traceback_bound(py) {
							println!("Got tb {tb}");
						} else {
							for frame in frames {
								let symbols = frame.symbols();
								if symbols.len() == 1 {
									// Common case

								}
							}
						}

						Err(err)
					},*/
					_ => panic::resume_unwind(payload)
				}
			}
		}
	}
}

impl Drop for PanicBacktrace {
	fn drop(&mut self) {
		let prev_hook = match self.prev_hook.take() {
			Some(prev_hook) => prev_hook,
			None => return, // Disabled
		};

		let hook = std::panic::take_hook();
		let is_current = ptr::addr_eq(Box::as_ptr(&hook), self.handler_ptr);
		if !is_current {
			// We weren't dropped
			debug_assert_eq!(Arc::weak_count(&self.backtrace), 0, "Panic hook replaced but still active");
			println!("Warning: panic handler replaced");

			// Let's put it back and hope for the best
			std::panic::set_hook(hook);
			return;
		}
		std::panic::set_hook(prev_hook);
	}
}