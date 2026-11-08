// Microbanc d'architecture : stockage S x arithmetique C, noyaux OR et HB (port K1), L^4, damier.
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemset hipMemset
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemset cudaMemset
#endif
#include <cstdio>
#include <chrono>
#include <cmath>
typedef unsigned int u32; typedef unsigned long long u64;
template<class T> struct Q4 { T x, y, z, w; };
template<class T> __device__ inline Q4<T> qm(Q4<T> a, Q4<T> b) { Q4<T> c;
  c.x = a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w; c.y = a.x*b.y + b.x*a.y - (a.z*b.w - a.w*b.z);
  c.z = a.x*b.z + b.x*a.z - (a.w*b.y - a.y*b.w); c.w = a.x*b.w + b.x*a.w - (a.y*b.z - a.z*b.y); return c; }
template<class T> __device__ inline Q4<T> qd(Q4<T> a) { return {a.x, -a.y, -a.z, -a.w}; }
template<class T> __device__ inline Q4<T> qa(Q4<T> a, Q4<T> b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
template<class C, class S> __device__ inline Q4<C> ld(const Q4<S>* U, long i) { Q4<S> q = U[i]; return {(C)q.x, (C)q.y, (C)q.z, (C)q.w}; }
struct U4 { u32 v[4]; };
__device__ inline U4 philox(U4 c, u32 k0, u32 k1) { for (int r = 0; r < 10; ++r) {
  u64 p0 = (u64)0xD2511F53u * c.v[0], p1 = (u64)0xCD9E8D57u * c.v[2]; U4 n;
  n.v[0] = (u32)(p1 >> 32) ^ c.v[1] ^ k0; n.v[1] = (u32)p1; n.v[2] = (u32)(p0 >> 32) ^ c.v[3] ^ k1; n.v[3] = (u32)p0; c = n;
  k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; } return c; }
template<class C> __device__ inline C u01(u32 a, u32 b) { u64 m = ((u64)(a >> 6) << 26) | (u64)(b >> 6); return (C)(((double)m + 0.5) * (1.0 / 4503599627370496.0)); }
template<class C, class S>
__global__ void k(Q4<S>* U, int L, int mu, int par, int hb, C beta, u32 sid) {
  long V = (long)L*L*L*L; long t = (long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= V/2) return;
  long st[4] = {(long)L*L*L, (long)L*L, (long)L, 1};
  long s = 2*t; int x[4]; { long r = s; x[3] = r % L; r /= L; x[2] = r % L; r /= L; x[1] = r % L; x[0] = r / L; }
  if (((x[0]+x[1]+x[2]+x[3]) & 1) != par) { s += 1; x[3] += 1; }
  auto up = [&](long ss, int d, int xd) { return xd == L-1 ? ss - (long)(L-1)*st[d] : ss + st[d]; };
  auto dn = [&](long ss, int d, int xd) { return xd == 0 ? ss + (long)(L-1)*st[d] : ss - st[d]; };
  Q4<C> A = {0,0,0,0}; long smu = up(s, mu, x[mu]);
  for (int nu = 0; nu < 4; ++nu) { if (nu == mu) continue;
    long snu = up(s, nu, x[nu]), sdn = dn(s, nu, x[nu]), smd = dn(smu, nu, x[nu]);
    Q4<C> a = qm(qm(ld<C>(U, nu*V+smu), qd(ld<C>(U, mu*V+snu))), qd(ld<C>(U, nu*V+s)));
    Q4<C> b = qm(qm(qd(ld<C>(U, nu*V+smd)), qd(ld<C>(U, mu*V+sdn))), ld<C>(U, nu*V+sdn)); A = qa(A, qa(a, b)); }
  C k = sqrt(A.x*A.x + A.y*A.y + A.z*A.z + A.w*A.w); C ik = (C)1 / k; Q4<C> Vv = {A.x*ik, A.y*ik, A.z*ik, A.w*ik};
  long li = (long)mu*V + s; Q4<C> Un;
  if (!hb) { Q4<C> Vd = qd(Vv); Un = qm(qm(Vd, qd(ld<C>(U, li))), Vd); }
  else { C al = beta * k; u32 dr = 0; C x0 = 1;
    for (int n = 0; n < 1000; ++n) { U4 r1 = philox({{(u32)li, sid, dr++, 7u}}, 1u, 2u), r2 = philox({{(u32)li, sid, dr++, 7u}}, 1u, 2u);
      C a1 = u01<C>(r1.v[0], r1.v[1]), a2 = u01<C>(r1.v[2], r1.v[3]), a3 = u01<C>(r2.v[0], r2.v[1]), a4 = u01<C>(r2.v[2], r2.v[3]);
      C c = cos((C)6.283185307179586 * a2); C l2 = -(log(a1) + c*c*log(a3)) / ((C)2 * al);
      if (a4*a4 <= (C)1 - l2) { x0 = (C)1 - (C)2*l2; break; } }
    U4 rd = philox({{(u32)li, sid, dr++, 7u}}, 1u, 2u); C ct = (C)2*u01<C>(rd.v[0], rd.v[1]) - (C)1, ph = (C)6.283185307179586*u01<C>(rd.v[2], rd.v[3]);
    C sn = sqrt(fmax((C)0, (C)1 - ct*ct)), rr = sqrt(fmax((C)0, (C)1 - x0*x0));
    Q4<C> W = {x0, rr*sn*cos(ph), rr*sn*sin(ph), rr*ct}; Un = qm(W, qd(Vv)); }
  C n = (C)1 / sqrt(Un.x*Un.x + Un.y*Un.y + Un.z*Un.z + Un.w*Un.w);
  U[li] = {(S)(Un.x*n), (S)(Un.y*n), (S)(Un.z*n), (S)(Un.w*n)}; }
template<class S> __global__ void kinit(Q4<S>* U, long n) { long i = (long)blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
  U4 r = philox({{(u32)i, 0u, 0u, 9u}}, 3u, 4u); double a = u01<double>(r.v[0], r.v[1]), b = u01<double>(r.v[2], r.v[3]);
  double q0 = 0.6 + 0.4*a, q1 = 0.5*b - 0.25, q2 = 0.1, q3 = -0.2; double nn = sqrt(q0*q0+q1*q1+q2*q2+q3*q3);
  U[i] = {(S)(q0/nn), (S)(q1/nn), (S)(q2/nn), (S)(q3/nn)}; }
template<class C, class S> double run(int L, int hb, int nsw) {
  long V = (long)L*L*L*L; Q4<S>* U; gpuMalloc(&U, sizeof(Q4<S>)*4*V);
  kinit<S><<<(4*V+255)/256, 256>>>(U, 4*V); gpuDeviceSynchronize();
  auto sweep = [&](u32 sid) { for (int mu = 0; mu < 4; ++mu) for (int p = 0; p < 2; ++p) k<C, S><<<(V/2+255)/256, 256>>>(U, L, mu, p, hb, (C)2.5, sid); };
  for (int i = 0; i < 3; ++i) sweep(i); gpuDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < nsw; ++i) sweep(100+i); gpuDeviceSynchronize();
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); gpuFree(U);
  return 4.0 * V * nsw / dt; }
int main(int argc, char** argv) {
  for (int L : {16, 32}) { int ns = L == 16 ? 200 : 40;
    for (int hb = 0; hb < 2; ++hb) {
      double a = run<double, double>(L, hb, ns), b = run<float, double>(L, hb, ns), c = run<double, float>(L, hb, ns), d = run<float, float>(L, hb, ns);
      printf("PROF L=%d %s liens/s : S64C64=%.3e S64C32=%.3e S32C64=%.3e S32C32=%.3e  (C32/C64 a S64 : x%.2f ; S32/S64 a C64 : x%.2f ; tout32 : x%.2f)\n",
             L, hb ? "HB" : "OR", a, b, c, d, b/a, c/a, d/a); fflush(stdout); } }
  return 0; }
