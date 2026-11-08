// kevinotron/src/hip/ffi.rs
// M1 (2026-07-10): raw dynamic-loading FFI surface for the HIP backend.
//
// Design frozen in IGPU_CHANTIER_ARCH_2026-07-10.md §1: cudarc-style
// dlopen at runtime (libloading), NEVER a static link against libamdhip64.
// Only the ~22 C symbols covering the 14 logical operations of audit A1 are
// bound; anything else is an incremental addition later, not here.
//
// ABI notes (design §8 REJ-3): on Linux ROCm 7.x, hipDeviceptr_t is a plain
// `void*`; hipError_t / hiprtcResult are C ints with 0 == success. This
// module targets that ABI only (single machine: acemagic, ROCm 7.1).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::OnceLock;

pub type hipError_t = c_int;
pub type hiprtcResult = c_int;
pub const HIP_SUCCESS: hipError_t = 0;
pub const HIPRTC_SUCCESS: hiprtcResult = 0;

/// Linux ROCm: hipDeviceptr_t is `void*` (verified against the pilot build,
/// design §8 REJ-3; Windows would differ — out of scope).
pub type hipDeviceptr_t = *mut c_void;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hiprtcProgram = *mut c_void;

/// Resolved function pointers, one per bound symbol. Populated once by
/// `hiplib()` and memoized for the process lifetime (mirror of cudarc's
/// leaked library handles).
pub struct HipLib {
    // -- op 1: context / device --
    pub hipInit: unsafe extern "C" fn(c_uint) -> hipError_t,
    pub hipGetDeviceCount: unsafe extern "C" fn(*mut c_int) -> hipError_t,
    pub hipSetDevice: unsafe extern "C" fn(c_int) -> hipError_t,
    // -- op 3: streams (op 2 default stream = null handle, no symbol) --
    pub hipStreamCreate: unsafe extern "C" fn(*mut hipStream_t) -> hipError_t,
    pub hipStreamDestroy: unsafe extern "C" fn(hipStream_t) -> hipError_t,
    // -- op 13: sync --
    pub hipStreamSynchronize: unsafe extern "C" fn(hipStream_t) -> hipError_t,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> hipError_t,
    // -- op 15: async error query (per-thread runtime error flag). Both are
    // exported by libamdhip64.so.7.1 (verified nm -D, acemagic 2026-07-11).
    // Enables a post-launch hipPeekAtLastError in wrap.rs::launch (S2) so an
    // async fault is attributed to the FAULTING launch, not the next sync.
    pub hipGetLastError: unsafe extern "C" fn() -> hipError_t,
    pub hipPeekAtLastError: unsafe extern "C" fn() -> hipError_t,
    // -- ops 7/8: alloc / memset --
    pub hipMalloc: unsafe extern "C" fn(*mut hipDeviceptr_t, usize) -> hipError_t,
    pub hipFree: unsafe extern "C" fn(hipDeviceptr_t) -> hipError_t,
    pub hipMemsetD8Async:
        unsafe extern "C" fn(hipDeviceptr_t, u8, usize, hipStream_t) -> hipError_t,
    // -- ops 9-11: memcpy --
    pub hipMemcpyHtoDAsync:
        unsafe extern "C" fn(hipDeviceptr_t, *const c_void, usize, hipStream_t) -> hipError_t,
    pub hipMemcpyDtoHAsync:
        unsafe extern "C" fn(*mut c_void, hipDeviceptr_t, usize, hipStream_t) -> hipError_t,
    pub hipMemcpyDtoDAsync:
        unsafe extern "C" fn(hipDeviceptr_t, hipDeviceptr_t, usize, hipStream_t) -> hipError_t,
    // -- ops 5/6: module load / function lookup --
    pub hipModuleLoadData: unsafe extern "C" fn(*mut hipModule_t, *const c_void) -> hipError_t,
    pub hipModuleUnload: unsafe extern "C" fn(hipModule_t) -> hipError_t,
    pub hipModuleGetFunction:
        unsafe extern "C" fn(*mut hipFunction_t, hipModule_t, *const c_char) -> hipError_t,
    // -- op 12: launch --
    pub hipModuleLaunchKernel: unsafe extern "C" fn(
        hipFunction_t,
        c_uint, // gridDimX
        c_uint,
        c_uint,
        c_uint, // blockDimX
        c_uint,
        c_uint,
        c_uint, // sharedMemBytes
        hipStream_t,
        *mut *mut c_void, // kernelParams
        *mut *mut c_void, // extra
    ) -> hipError_t,
    // -- op 14: set_attribute (F4-only downstream, no-op tolerated elsewhere) --
    pub hipFuncSetAttribute: unsafe extern "C" fn(*const c_void, c_int, c_int) -> hipError_t,
    // -- errors --
    pub hipGetErrorString: unsafe extern "C" fn(hipError_t) -> *const c_char,
    // -- op 4: hiprtc runtime compilation (mirror of NVRTC) --
    pub hiprtcCreateProgram: unsafe extern "C" fn(
        *mut hiprtcProgram,
        *const c_char, // src
        *const c_char, // name
        c_int,         // numHeaders
        *const *const c_char,
        *const *const c_char,
    ) -> hiprtcResult,
    pub hiprtcCompileProgram:
        unsafe extern "C" fn(hiprtcProgram, c_int, *const *const c_char) -> hiprtcResult,
    pub hiprtcGetProgramLogSize:
        unsafe extern "C" fn(hiprtcProgram, *mut usize) -> hiprtcResult,
    pub hiprtcGetProgramLog: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult,
    pub hiprtcGetCodeSize: unsafe extern "C" fn(hiprtcProgram, *mut usize) -> hiprtcResult,
    pub hiprtcGetCode: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult,
    pub hiprtcDestroyProgram: unsafe extern "C" fn(*mut hiprtcProgram) -> hiprtcResult,
    pub hiprtcGetErrorString: unsafe extern "C" fn(hiprtcResult) -> *const c_char,
}

/// dlopen candidate names, in order. Ubuntu 26.04 ships the runtime under
/// /usr/lib/x86_64-linux-gnu (found by the plain soname); the /opt/rocm
/// entries cover an upstream-ROCm install (design §1).
const AMDHIP_CANDIDATES: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.7",
];
const HIPRTC_CANDIDATES: &[&str] = &[
    "libhiprtc.so",
    "libhiprtc.so.7",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
    "/opt/rocm/lib/libhiprtc.so",
    "/opt/rocm/lib/libhiprtc.so.7",
];

fn open_first(candidates: &[&str]) -> Result<&'static libloading::Library, String> {
    let mut errs = Vec::new();
    for name in candidates {
        match unsafe { libloading::Library::new(*name) } {
            // Leak on purpose: one handle per process lifetime (cudarc model).
            Ok(lib) => return Ok(Box::leak(Box::new(lib))),
            Err(e) => errs.push(format!("{name}: {e}")),
        }
    }
    Err(format!("dlopen failed for all candidates: {}", errs.join(" | ")))
}

macro_rules! sym {
    ($lib:expr, $name:ident) => {
        unsafe {
            *$lib
                .get(concat!(stringify!($name), "\0").as_bytes())
                .map_err(|e| format!("dlsym {}: {}", stringify!($name), e))?
        }
    };
}

fn load() -> Result<HipLib, String> {
    let hip = open_first(AMDHIP_CANDIDATES)?;
    let rtc = open_first(HIPRTC_CANDIDATES)?;
    Ok(HipLib {
        hipInit: sym!(hip, hipInit),
        hipGetDeviceCount: sym!(hip, hipGetDeviceCount),
        hipSetDevice: sym!(hip, hipSetDevice),
        hipStreamCreate: sym!(hip, hipStreamCreate),
        hipStreamDestroy: sym!(hip, hipStreamDestroy),
        hipStreamSynchronize: sym!(hip, hipStreamSynchronize),
        hipDeviceSynchronize: sym!(hip, hipDeviceSynchronize),
        hipGetLastError: sym!(hip, hipGetLastError),
        hipPeekAtLastError: sym!(hip, hipPeekAtLastError),
        hipMalloc: sym!(hip, hipMalloc),
        hipFree: sym!(hip, hipFree),
        hipMemsetD8Async: sym!(hip, hipMemsetD8Async),
        hipMemcpyHtoDAsync: sym!(hip, hipMemcpyHtoDAsync),
        hipMemcpyDtoHAsync: sym!(hip, hipMemcpyDtoHAsync),
        hipMemcpyDtoDAsync: sym!(hip, hipMemcpyDtoDAsync),
        hipModuleLoadData: sym!(hip, hipModuleLoadData),
        hipModuleUnload: sym!(hip, hipModuleUnload),
        hipModuleGetFunction: sym!(hip, hipModuleGetFunction),
        hipModuleLaunchKernel: sym!(hip, hipModuleLaunchKernel),
        hipFuncSetAttribute: sym!(hip, hipFuncSetAttribute),
        hipGetErrorString: sym!(hip, hipGetErrorString),
        hiprtcCreateProgram: sym!(rtc, hiprtcCreateProgram),
        hiprtcCompileProgram: sym!(rtc, hiprtcCompileProgram),
        hiprtcGetProgramLogSize: sym!(rtc, hiprtcGetProgramLogSize),
        hiprtcGetProgramLog: sym!(rtc, hiprtcGetProgramLog),
        hiprtcGetCodeSize: sym!(rtc, hiprtcGetCodeSize),
        hiprtcGetCode: sym!(rtc, hiprtcGetCode),
        hiprtcDestroyProgram: sym!(rtc, hiprtcDestroyProgram),
        hiprtcGetErrorString: sym!(rtc, hiprtcGetErrorString),
    })
}

/// Memoized library handle. `Err` is also memoized (a machine without ROCm
/// never re-probes).
pub fn hiplib() -> Result<&'static HipLib, String> {
    static LIB: OnceLock<Result<HipLib, String>> = OnceLock::new();
    match LIB.get_or_init(load) {
        Ok(lib) => Ok(lib),
        Err(e) => Err(e.clone()),
    }
}

/// Human-readable HIP error (falls back to the numeric code if the string
/// pointer is null).
pub fn hip_err_str(lib: &HipLib, code: hipError_t) -> String {
    unsafe {
        let p = (lib.hipGetErrorString)(code);
        if p.is_null() {
            format!("hipError {code}")
        } else {
            format!(
                "{} ({code})",
                std::ffi::CStr::from_ptr(p).to_string_lossy()
            )
        }
    }
}

pub fn hiprtc_err_str(lib: &HipLib, code: hiprtcResult) -> String {
    unsafe {
        let p = (lib.hiprtcGetErrorString)(code);
        if p.is_null() {
            format!("hiprtcResult {code}")
        } else {
            format!(
                "{} ({code})",
                std::ffi::CStr::from_ptr(p).to_string_lossy()
            )
        }
    }
}
