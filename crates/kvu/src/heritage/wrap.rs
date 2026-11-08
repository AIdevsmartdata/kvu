// kevinotron/src/hip/wrap.rs
// M1 (2026-07-10): thin RAII wrappers over hip/ffi.rs, mirroring the small
// cudarc surface the engine actually uses (IGPU_CHANTIER_ARCH §1). No
// business logic here; the first consumer is hip::su3_sweep_hip (M1 gate).

use super::ffi::{
    hip_err_str, hiplib, hiprtc_err_str, hipDeviceptr_t, hipFunction_t, hipModule_t,
    hipStream_t, HipLib, HIPRTC_SUCCESS, HIP_SUCCESS,
};
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::sync::Arc;

fn check(lib: &HipLib, code: i32, what: &str) -> Result<(), String> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", hip_err_str(lib, code)))
    }
}

/// hipInit + hipSetDevice — mirror of cudarc `CudaContext::new` (op 1).
pub struct HipContext {
    pub dev: i32,
    pub lib: &'static HipLib,
}

impl HipContext {
    pub fn new(dev: i32) -> Result<Arc<Self>, String> {
        let lib = hiplib()?;
        unsafe {
            check(lib, (lib.hipInit)(0), "hipInit")?;
            check(lib, (lib.hipSetDevice)(dev), "hipSetDevice")?;
        }
        Ok(Arc::new(HipContext { dev, lib }))
    }

    /// op 2: the default stream is the null handle — no symbol involved.
    pub fn default_stream(self: &Arc<Self>) -> HipStream {
        HipStream {
            raw: std::ptr::null_mut(),
            ctx: Arc::clone(self),
            owned: false,
        }
    }

    /// op 3: explicit stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<HipStream, String> {
        let mut raw: hipStream_t = std::ptr::null_mut();
        unsafe {
            check(self.lib, (self.lib.hipStreamCreate)(&mut raw), "hipStreamCreate")?;
        }
        Ok(HipStream {
            raw,
            ctx: Arc::clone(self),
            owned: true,
        })
    }

    /// op 4: runtime compilation via hiprtc (mirror of NVRTC
    /// `compile_ptx_with_opts`). Returns the loadable code object.
    ///
    /// A2 amendment (design §8): hiprtc does NOT infer the target arch —
    /// callers must pass `--offload-arch=<gfx>` in `opts` (helper:
    /// [`offload_arch_opt`]). On compile failure the hiprtc program log is
    /// returned verbatim in the error.
    pub fn compile_hip(&self, src: &str, name: &str, opts: &[&str]) -> Result<Vec<u8>, String> {
        let lib = self.lib;
        let c_src = CString::new(src).map_err(|e| format!("src NUL: {e}"))?;
        let c_name = CString::new(name).map_err(|e| format!("name NUL: {e}"))?;
        let c_opts: Vec<CString> = opts
            .iter()
            .map(|o| CString::new(*o).map_err(|e| format!("opt NUL: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let c_opt_ptrs: Vec<*const i8> = c_opts.iter().map(|o| o.as_ptr()).collect();
        unsafe {
            let mut prog: super::ffi::hiprtcProgram = std::ptr::null_mut();
            let r = (lib.hiprtcCreateProgram)(
                &mut prog,
                c_src.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if r != HIPRTC_SUCCESS {
                return Err(format!("hiprtcCreateProgram: {}", hiprtc_err_str(lib, r)));
            }
            let rc = (lib.hiprtcCompileProgram)(prog, c_opt_ptrs.len() as i32, c_opt_ptrs.as_ptr());
            // Always fetch the log (warnings matter for the port audit trail).
            let mut log_size: usize = 0;
            let mut log = String::new();
            if (lib.hiprtcGetProgramLogSize)(prog, &mut log_size) == HIPRTC_SUCCESS && log_size > 1
            {
                let mut buf = vec![0u8; log_size];
                if (lib.hiprtcGetProgramLog)(prog, buf.as_mut_ptr() as *mut i8) == HIPRTC_SUCCESS {
                    log = String::from_utf8_lossy(&buf)
                        .trim_end_matches('\0')
                        .to_string();
                }
            }
            if rc != HIPRTC_SUCCESS {
                (lib.hiprtcDestroyProgram)(&mut prog);
                return Err(format!(
                    "hiprtcCompileProgram({name}): {}\n--- hiprtc log ---\n{log}",
                    hiprtc_err_str(lib, rc)
                ));
            }
            let mut code_size: usize = 0;
            let r = (lib.hiprtcGetCodeSize)(prog, &mut code_size);
            if r != HIPRTC_SUCCESS {
                (lib.hiprtcDestroyProgram)(&mut prog);
                return Err(format!("hiprtcGetCodeSize: {}", hiprtc_err_str(lib, r)));
            }
            let mut code = vec![0u8; code_size];
            let r = (lib.hiprtcGetCode)(prog, code.as_mut_ptr() as *mut i8);
            (lib.hiprtcDestroyProgram)(&mut prog);
            if r != HIPRTC_SUCCESS {
                return Err(format!("hiprtcGetCode: {}", hiprtc_err_str(lib, r)));
            }
            if !log.trim().is_empty() {
                eprintln!("# hiprtc log ({name}):\n{log}");
            }
            Ok(code)
        }
    }

    /// op 5: load a hiprtc code object.
    pub fn load_module(self: &Arc<Self>, code: &[u8]) -> Result<HipModule, String> {
        let mut raw: hipModule_t = std::ptr::null_mut();
        unsafe {
            check(
                self.lib,
                (self.lib.hipModuleLoadData)(&mut raw, code.as_ptr() as *const c_void),
                "hipModuleLoadData",
            )?;
        }
        Ok(HipModule {
            raw,
            ctx: Arc::clone(self),
        })
    }
}

pub struct HipStream {
    raw: hipStream_t,
    pub ctx: Arc<HipContext>,
    owned: bool,
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            unsafe {
                let _ = (self.ctx.lib.hipStreamDestroy)(self.raw);
            }
        }
    }
}

pub struct HipModule {
    raw: hipModule_t,
    ctx: Arc<HipContext>,
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.ctx.lib.hipModuleUnload)(self.raw);
            }
        }
    }
}

impl HipModule {
    /// op 6: function lookup. Invariant A6 (design §8): kernels MUST be
    /// `extern "C" __global__` — no `hiprtcGetLoweredName`, plain-string
    /// lookup only.
    pub fn load_function(&self, name: &str) -> Result<HipFunction, String> {
        let c_name = CString::new(name).map_err(|e| format!("fn name NUL: {e}"))?;
        let mut raw: hipFunction_t = std::ptr::null_mut();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipModuleGetFunction)(&mut raw, self.raw, c_name.as_ptr()),
                &format!("hipModuleGetFunction({name})"),
            )?;
        }
        Ok(HipFunction { raw })
    }
}

#[derive(Clone, Copy)]
pub struct HipFunction {
    raw: hipFunction_t,
}

impl HipFunction {
    /// op 14 (F4-only downstream): hipFuncSetAttribute.
    pub fn set_attribute(&self, ctx: &HipContext, attr: i32, value: i32) -> Result<(), String> {
        unsafe {
            check(
                ctx.lib,
                (ctx.lib.hipFuncSetAttribute)(self.raw as *const c_void, attr, value),
                "hipFuncSetAttribute",
            )
        }
    }
}

/// Device buffer, RAII (Drop → hipFree). Mirror of cudarc `CudaSlice<T>`.
pub struct HipSlice<T> {
    ptr: hipDeviceptr_t,
    len: usize,
    ctx: Arc<HipContext>,
    _pd: PhantomData<T>,
}

impl<T> Drop for HipSlice<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.ctx.lib.hipFree)(self.ptr);
            }
        }
    }
}

impl<T> HipSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

/// Launch geometry (design §1).
#[derive(Clone, Copy, Debug)]
pub struct LaunchArgs {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shmem: u32,
}

/// Kernel argument. Each value is staged in an 8-byte little-endian slot;
/// `kernelParams` receives a pointer to the slot (the runtime reads the
/// per-parameter size from kernel metadata, so an i32/f32 in the low 4
/// bytes of an 8-byte slot is ABI-correct on x86_64 LE).
#[derive(Clone, Copy, Debug)]
pub enum KArg {
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    U64(u64),
    Ptr(*mut c_void),
}

impl KArg {
    fn bits(&self) -> u64 {
        match *self {
            KArg::I32(v) => v as u32 as u64,
            KArg::U32(v) => v as u64,
            KArg::F32(v) => v.to_bits() as u64,
            KArg::F64(v) => v.to_bits(),
            KArg::U64(v) => v,
            KArg::Ptr(p) => p as u64,
        }
    }
}

impl HipStream {
    pub fn raw(&self) -> hipStream_t {
        self.raw
    }

    /// op 7: hipMalloc + async memset-0 on this stream.
    pub fn alloc_zeros<T>(&self, len: usize) -> Result<HipSlice<T>, String> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                format!("alloc_zeros size overflow: {len} * {}", std::mem::size_of::<T>())
            })?;
        let mut ptr: hipDeviceptr_t = std::ptr::null_mut();
        unsafe {
            check(self.ctx.lib, (self.ctx.lib.hipMalloc)(&mut ptr, bytes), "hipMalloc")?;
            if let Err(e) = check(
                self.ctx.lib,
                (self.ctx.lib.hipMemsetD8Async)(ptr, 0, bytes, self.raw),
                "hipMemsetD8Async",
            ) {
                // Error path: hipMalloc already succeeded but the HipSlice (whose
                // Drop calls hipFree) is not yet built — free here or the device
                // allocation leaks.
                let _ = (self.ctx.lib.hipFree)(ptr);
                return Err(e);
            }
        }
        Ok(HipSlice {
            ptr,
            len,
            ctx: Arc::clone(&self.ctx),
            _pd: PhantomData,
        })
    }

    /// op 8.
    pub fn memset_zeros<T>(&self, dst: &mut HipSlice<T>) -> Result<(), String> {
        let bytes = dst.len * std::mem::size_of::<T>();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipMemsetD8Async)(dst.ptr, 0, bytes, self.raw),
                "hipMemsetD8Async",
            )
        }
    }

    /// op 9. NOTE A4 (design §8): host memory is not pinned, so this is
    /// synchronous de facto; never presume overlap.
    pub fn htod<T: Copy>(&self, src: &[T], dst: &mut HipSlice<T>) -> Result<(), String> {
        assert_eq!(src.len(), dst.len, "htod length mismatch");
        let bytes = src.len() * std::mem::size_of::<T>();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipMemcpyHtoDAsync)(
                    dst.ptr,
                    src.as_ptr() as *const c_void,
                    bytes,
                    self.raw,
                ),
                "hipMemcpyHtoDAsync",
            )?;
        }
        self.synchronize()
    }

    /// op 10 (synchronizes before returning — the host buffer must be full).
    pub fn dtoh<T: Copy>(&self, src: &HipSlice<T>, dst: &mut [T]) -> Result<(), String> {
        assert_eq!(src.len, dst.len(), "dtoh length mismatch");
        let bytes = dst.len() * std::mem::size_of::<T>();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipMemcpyDtoHAsync)(
                    dst.as_mut_ptr() as *mut c_void,
                    src.ptr,
                    bytes,
                    self.raw,
                ),
                "hipMemcpyDtoHAsync",
            )?;
        }
        self.synchronize()
    }

    /// op 11.
    pub fn dtod<T>(&self, src: &HipSlice<T>, dst: &mut HipSlice<T>) -> Result<(), String> {
        assert_eq!(src.len, dst.len, "dtod length mismatch");
        let bytes = src.len * std::mem::size_of::<T>();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipMemcpyDtoDAsync)(dst.ptr, src.ptr, bytes, self.raw),
                "hipMemcpyDtoDAsync",
            )
        }
    }

    /// op 12: kernel launch, cudarc-style `.arg().arg().launch()` collapsed
    /// into a slice of [`KArg`].
    pub fn launch(&self, f: &HipFunction, cfg: &LaunchArgs, args: &[KArg]) -> Result<(), String> {
        let mut slots: Vec<u64> = args.iter().map(KArg::bits).collect();
        let mut params: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipModuleLaunchKernel)(
                    f.raw,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shmem,
                    self.raw,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )
        }
    }

    /// op 13.
    pub fn synchronize(&self) -> Result<(), String> {
        unsafe {
            check(
                self.ctx.lib,
                (self.ctx.lib.hipStreamSynchronize)(self.raw),
                "hipStreamSynchronize",
            )
        }
    }
}

/// A2 amendment helper: the explicit hiprtc target-arch option. gfx1151
/// (acemagic Strix Halo) is the only deployed target; override via the
/// `KEVINOTRON_HIP_ARCH` env for a future second AMD box.
pub fn offload_arch_opt() -> String {
    let arch =
        std::env::var("KEVINOTRON_HIP_ARCH").unwrap_or_else(|_| "gfx1151".to_string());
    format!("--offload-arch={arch}")
}
