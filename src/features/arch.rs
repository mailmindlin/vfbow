
/// Ensure that NEON is available at runtime (debug only)
/// 
/// This should be used as a safety check to limit unsafe behavior
#[inline(always)]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
#[track_caller]
pub(super) fn debug_ensure_neon() {
    use std::arch::is_aarch64_feature_detected;
    // This was constant-time available on my platform, but I'm not sure about all aarch64 platforms
    #[allow(clippy::assertions_on_constants)]
    { debug_assert!(is_aarch64_feature_detected!("neon")); }
}

#[inline(always)]
#[cfg(target_arch = "arm")]
#[track_caller]
pub(super) fn debug_ensure_neon() {
    use std::arm::is_arm_feature_detected;
    #[allow(clippy::assertions_on_constants)]
    { debug_assert!(is_arm_feature_detected!("neon")); }
}