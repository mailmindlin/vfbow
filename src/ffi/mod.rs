//! FFI bindings (requires feature flag to enable)
//! 
//! | Binding | Feature Flag |
//! |---------|--------------|
//! | Python  | `python`     |
//! 
#[cfg(feature="python")]
mod python;
