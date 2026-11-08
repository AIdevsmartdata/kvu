#!/usr/bin/env python3
"""ym4_noyaux.cb — noyau n°2 : chaîne C-b (bain thermique conditionnel « fin sachant dec » + mesure de fibre).

Références (lecture seule, copies à l'octet dans ref/, SHA dans ref/SHA_ref.txt) :
  lib_Cb.py 64967707… : sweep_cond_dec_fast, agrafes_at, plaq_values_grads_cached, run_chain (bloc mesure), block_sums ;
  lib_brancheC.py 2392599f… (gelé PREREG) : sweep_cond_dec, heatbath_su2, reunit, masks_dec, V_pairs, fiber_map.

Conventions (celles du projet) : C (4, L, L, L, L, 2, 2) complex128 C-contigu, lien U_mu(x) ; S_W = Σ_p (1 − ½ ReTr U_p) ;
base X_a = iσ_a/√2, ⟨X,Y⟩ = −Tr(XY), dérivées invariantes à droite U → e^{sX} U.

API :
  geo = Geometry(L)                                  masques dec, voisins, paires, plaquettes par lien (en cache)
  V_pairs(C, geo) -> (4, npair, 2, 2)                = lib_brancheC.V_pairs (produits de paires figés)
  agrafes(C, geo, mu, sites) -> (n, 2, 2)            = lib_Cb.agrafes_at (mêmes produits, même ordre)
  sweep(C, beta, rng, geo, Vfix, mode='ref_rng')     = sweep_cond_dec_fast : MÊMES tirages numpy (concordance ≤ 1e-12 / balayage)
  sweep(C, beta, rng, geo, Vfix, mode='rapide')      tout compilé, RNG par lien (splitmix64 à clé) : équivalence STATISTIQUE seulement
  mesure(C, geo, Wf) -> (F, Dcontrib, blk)           = (Wf@pv, Gf.T@(Gf/g), block_sums(pv).astype(f32)) de run_chain, sans matrice creuse
  chaine(C, beta, rng, Wf, ...)                      boucle thermalisation + segment de run_chain (sans kfib ni prolongation)

numba CPU, fastmath=False (float64 IEEE, pas de contraction FMA implicite). Fils : NUMBA_NUM_THREADS (≤ 8 cœurs E gamer ; 4 acemagic).
"""
import numpy as np
import numba as nb
from numba import njit, prange

__all__ = ['Geometry', 'V_pairs', 'agrafes', 'sweep', 'mesure', 'chaine', 'block_sums', 'masks_dec',
           'fourier_weights_sparse', 'plaquettes', 'mesure_libre']

_JIT = dict(cache=True, fastmath=False, nogil=True)
SQ2I = 1.0 / np.sqrt(2.0)

# =============================================================================================== géométrie
def masks_dec(L):
    """= lib_brancheC.masks_dec (recopiée : même définition ; testée à l'identique)."""
    X = np.indices((L,) * 4)
    pair1, pair2, free = [], [], []
    for mu in range(4):
        ortho_even = np.ones((L,) * 4, bool)
        for nu in range(4):
            if nu != mu: ortho_even &= (X[nu] % 2 == 0)
        pair1.append(ortho_even & (X[mu] % 2 == 0))
        pair2.append(ortho_even & (X[mu] % 2 == 1))
        free.append(~ortho_even)
    return np.array(pair1), np.array(pair2), np.array(free), (X.sum(0) % 2)


class Geometry:
    """Tables d'indices pour un réseau L^4 périodique (L pair ≥ 4). Sites aplatis en ordre C, liens = mu·L^4 + x."""

    def __init__(self, L):
        if L % 2 or L < 4:
            raise ValueError('L pair ≥ 4 requis')
        self.L = L; V = L ** 4; self.V = V
        self.M = masks_dec(L)
        pair1, pair2, free, par = self.M
        I = np.arange(V).reshape((L,) * 4)
        nbp = np.empty((4, V), np.int64); nbm = np.empty((4, V), np.int64)
        for d in range(4):
            nbp[d] = np.roll(I, -1, axis=d).ravel()      # x + d
            nbm[d] = np.roll(I, 1, axis=d).ravel()       # x - d
        self.nbp, self.nbm = nbp, nbm
        self.p1 = np.stack([np.flatnonzero(pair1[mu]) for mu in range(4)]).astype(np.int64)      # (4, npair)
        self.p1pm = np.stack([nbp[mu][self.p1[mu]] for mu in range(4)]).astype(np.int64)
        self.fr = [[np.flatnonzero(free[mu] & (par == p)).astype(np.int64) for p in (0, 1)] for mu in range(4)]
        self.free_all = [np.flatnonzero(free[mu]).astype(np.int64) for mu in range(4)]
        # plaquettes : ordre loops_plaquettes (plan (mu<nu) majeur puis site) ; liens e0..e3, signes + + − −
        planes = [(m, n) for m in range(4) for n in range(m + 1, 4)]
        self.planes = np.array(planes, np.int64)
        E = np.empty((6 * V, 4), np.int64)
        x = np.arange(V)
        for k, (m, n) in enumerate(planes):
            E[k * V:(k + 1) * V, 0] = m * V + x
            E[k * V:(k + 1) * V, 1] = n * V + nbp[m]
            E[k * V:(k + 1) * V, 2] = m * V + nbp[n]
            E[k * V:(k + 1) * V, 3] = n * V + x
        self.E = E
        # plaquettes contenant chaque lien (6 par lien en 4d), avec la position dans la plaquette
        flat = E.ravel()
        order = np.argsort(flat, kind='stable')
        cnt = np.bincount(flat, minlength=4 * V)
        assert (cnt == 6).all()
        self.lp_plaq = (order // 4).reshape(4 * V, 6).astype(np.int64)
        self.lp_pos = (order % 4).reshape(4 * V, 6).astype(np.int64)
        # fibre : liens libres (métrique 1) puis paires (métrique 2) — ordre de fiber_map
        self.fib_free = np.concatenate([mu * V + self.free_all[mu] for mu in range(4)]).astype(np.int64)
        self.fib_e1 = np.concatenate([mu * V + self.p1[mu] for mu in range(4)]).astype(np.int64)
        self.fib_e2 = np.concatenate([mu * V + self.p1pm[mu] for mu in range(4)]).astype(np.int64)
        self.n_fib = 3 * len(self.fib_free) + 3 * len(self.fib_e1)
        self._wf = None

    def wf_sparse(self, Wf):
        """Wf (nF, 6V), dense ou scipy.sparse -> CSR par plaquette (ptr, idx, val, nF). En cache sur l'objet Wf (identité)."""
        if self._wf is not None and self._wf[0] is Wf:
            return self._wf[1]
        import scipy.sparse as sp
        if sp.issparse(Wf):
            WfT = sp.csr_matrix(Wf.T); WfT.sum_duplicates(); WfT.sort_indices()
            if WfT.shape[0] != 6 * self.V:
                raise ValueError('Wf : %d colonnes attendues' % (6 * self.V))
            prep = (WfT.indptr.astype(np.int64), WfT.indices.astype(np.int64), WfT.data.astype(np.float64), WfT.shape[1])
        else:
            WfT = np.ascontiguousarray(np.asarray(Wf).T)
            if WfT.shape[0] != 6 * self.V:
                raise ValueError('Wf : %d colonnes attendues' % (6 * self.V))
            rows, cols = np.nonzero(WfT)
            ptr = np.zeros(WfT.shape[0] + 1, np.int64)
            np.cumsum(np.bincount(rows, minlength=WfT.shape[0]), out=ptr[1:])
            prep = (ptr, cols.astype(np.int64), WfT[rows, cols].copy(), WfT.shape[1])
        self._wf = (Wf, prep)
        return prep


def fourier_weights_sparse(L, pmax2=1):
    """= lib_brancheC.fourier_weights(L, pmax2) en scipy.sparse CSR (mêmes lignes, mêmes valeurs f(ph)/L², mêmes zéros
    structurels exclus) : évite la matrice dense (54 × 6L⁴ : 2,7 Go à L = 32)."""
    import itertools, scipy.sparse as sp
    X = np.indices((L,) * 4).reshape(4, -1).T
    ns = [n for n in itertools.product(range(-1, 2), repeat=4) if 0 < sum(np.abs(n) ** 2) <= pmax2 or sum(np.abs(n)) == 0]
    seen, moms = set(), []
    for n in ns:
        t = tuple(n); tm = tuple(-np.array(n))
        if tm in seen: continue
        seen.add(t); moms.append(np.array(n))
    V = L ** 4; rows, cols, vals = [], [], []; r = 0
    for pl in range(6):
        for n in moms:
            ph = 2 * np.pi * (X @ n) / L
            for f in ((np.cos,) if not n.any() else (np.cos, np.sin)):
                w = f(ph) / L ** 2; nz = np.flatnonzero(w)
                rows.append(np.full(len(nz), r)); cols.append(pl * V + nz); vals.append(w[nz]); r += 1
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, 6 * V))


def _flat(C, geo):
    if C.dtype != np.complex128 or not C.flags.c_contiguous or C.shape != (4,) + (geo.L,) * 4 + (2, 2):
        raise ValueError('C doit être (4,L,L,L,L,2,2) complex128 C-contigu')
    return C.reshape(4, geo.V, 2, 2)          # vue


# =============================================================================================== 2×2 complexes en scalaires
@njit(inline='always', **_JIT)
def _ld(U, m, s):
    return (U[m, s, 0, 0], U[m, s, 0, 1], U[m, s, 1, 0], U[m, s, 1, 1])

@njit(inline='always', **_JIT)
def _mm(A, B):
    return (A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
            A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3])

@njit(inline='always', **_JIT)
def _dg(A):
    return (A[0].conjugate(), A[2].conjugate(), A[1].conjugate(), A[3].conjugate())

@njit(inline='always', **_JIT)
def _add(A, B):
    return (A[0] + B[0], A[1] + B[1], A[2] + B[2], A[3] + B[3])

@njit(inline='always', **_JIT)
def _st(out, i, A):
    out[i, 0, 0] = A[0]; out[i, 0, 1] = A[1]; out[i, 1, 0] = A[2]; out[i, 1, 1] = A[3]

@njit(inline='always', **_JIT)
def _st4(U, m, s, A):
    U[m, s, 0, 0] = A[0]; U[m, s, 0, 1] = A[1]; U[m, s, 1, 0] = A[2]; U[m, s, 1, 1] = A[3]


@njit(inline='always', **_JIT)
def _staple(Cf, mu, s, nbp, nbm):
    """= agrafes_at pour un site : Σ_ν [U_ν(x+μ) U_μ(x+ν)† U_ν(x)†] + [U_ν(x+μ−ν)† U_μ(x−ν)† U_ν(x−ν)], même ordre."""
    z = 0j
    A = (z, z, z, z)
    for nu in range(4):
        if nu == mu:
            continue
        xm = nbp[mu, s]; xn = nbp[nu, s]; xmn = nbm[nu, xm]; xmi = nbm[nu, s]
        A = _add(A, _mm(_mm(_ld(Cf, nu, xm), _dg(_ld(Cf, mu, xn))), _dg(_ld(Cf, nu, s))))
        A = _add(A, _mm(_mm(_dg(_ld(Cf, nu, xmn)), _dg(_ld(Cf, mu, xmi))), _ld(Cf, nu, xmi)))
    return A


@njit(parallel=True, **_JIT)
def _k_agrafes(Cf, mu, sites, nbp, nbm, out):
    for i in prange(sites.shape[0]):
        _st(out, i, _staple(Cf, mu, sites[i], nbp, nbm))


@njit(parallel=True, **_JIT)
def _k_reunit(Cf):
    nl = Cf.shape[1]
    for t in prange(4 * nl):
        m = t // nl; s = t % nl
        a0 = 0.5 * (Cf[m, s, 0, 0] + Cf[m, s, 1, 1].conjugate())
        b = 0.5 * (Cf[m, s, 0, 1] - Cf[m, s, 1, 0].conjugate())
        n = np.sqrt(abs(a0) ** 2 + abs(b) ** 2)
        a0 = a0 / n; b = b / n
        Cf[m, s, 0, 0] = a0; Cf[m, s, 1, 1] = a0.conjugate(); Cf[m, s, 0, 1] = b; Cf[m, s, 1, 0] = -b.conjugate()


@njit(parallel=True, **_JIT)
def _k_vpairs(Cf, p1, p1pm, out):
    n = p1.shape[1]
    for t in prange(4 * n):
        m = t // n; i = t % n
        _st4(out, m, i, _mm(_ld(Cf, m, p1[m, i]), _ld(Cf, m, p1pm[m, i])))


# ----------------------------------------------------------------------------------------------- bain thermique : préparation / fin
@njit(inline='always', **_JIT)
def _prep1(A, beta, Vn, k, alpha, i):
    det = (A[0] * A[3] - A[1] * A[2]).real
    kk = np.sqrt(max(det, 1e-300))
    Vn[i, 0, 0] = A[0] / kk; Vn[i, 0, 1] = A[1] / kk; Vn[i, 1, 0] = A[2] / kk; Vn[i, 1, 1] = A[3] / kk
    k[i] = kk; alpha[i] = beta * kk


@njit(parallel=True, **_JIT)
def _k_pairs_prep(Cf, mu, p1, p1pm, Vfix, nbp, nbm, beta, Vn, k, alpha):
    """A_eff = A1 + A2† V† (agrafes aux deux liens de la paire), puis k, V = A/k, α = βk."""
    for i in prange(p1.shape[1]):
        A1 = _staple(Cf, mu, p1[mu, i], nbp, nbm)
        A2 = _staple(Cf, mu, p1pm[mu, i], nbp, nbm)
        V = _ld(Vfix, mu, i)
        Aeff = _add(A1, _mm(_dg(A2), _dg(V)))
        _prep1(Aeff, beta, Vn, k, alpha, i)


@njit(parallel=True, **_JIT)
def _k_free_prep(Cf, mu, sites, nbp, nbm, beta, Vn, k, alpha):
    for i in prange(sites.shape[0]):
        _prep1(_staple(Cf, mu, sites[i], nbp, nbm), beta, Vn, k, alpha, i)


@njit(inline='always', **_JIT)
def _hb_U(x0, r, v0, v1, v2, Vn, i):
    """W = x0 + i r v.σ (convention heatbath_su2) ; U = W V†."""
    W = (complex(x0, r * v2), complex(r * v1, r * v0), complex(-r * v1, r * v0), complex(x0, -r * v2))
    return _mm(W, _dg(_ld3(Vn, i)))

@njit(inline='always', **_JIT)
def _ld3(U, i):
    return (U[i, 0, 0], U[i, 0, 1], U[i, 1, 0], U[i, 1, 1])


@njit(parallel=True, **_JIT)
def _k_pairs_fin(Cf, mu, p1, p1pm, Vfix, x0, r, v, Vn):
    for i in prange(p1.shape[1]):
        U1 = _hb_U(x0[i], r[i], v[i, 0], v[i, 1], v[i, 2], Vn, i)
        _st4(Cf, mu, p1[mu, i], U1)
        _st4(Cf, mu, p1pm[mu, i], _mm(_dg(U1), _ld(Vfix, mu, i)))


@njit(parallel=True, **_JIT)
def _k_free_fin(Cf, mu, sites, x0, r, v, Vn):
    for i in prange(sites.shape[0]):
        _st4(Cf, mu, sites[i], _hb_U(x0[i], r[i], v[i, 0], v[i, 1], v[i, 2], Vn, i))


def _hb_numpy(alpha, rng):
    """Tirages de heatbath_su2 À L'IDENTIQUE (mêmes appels rng, même ordre, même arithmétique numpy)."""
    n = len(alpha); x0 = np.empty(n); todo = np.arange(n)
    while len(todo):
        a = alpha[todo]; u = rng.random(len(todo))
        x = 1.0 + np.log1p(-u * (-np.expm1(-2 * a))) / a
        ok = rng.random(len(todo)) < np.sqrt(np.maximum(1 - x * x, 0))
        x0[todo[ok]] = x[ok]; todo = todo[~ok]
    r = np.sqrt(np.maximum(1 - x0 ** 2, 0))
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return x0, r, v


# ----------------------------------------------------------------------------------------------- mode rapide : RNG compilé par lien
_GOLD = np.uint64(0x9E3779B97F4A7C15)

@njit(inline='always', **_JIT)
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

@njit(inline='always', **_JIT)
def _u01(st):
    """splitmix64 : st (uint64) -> (nouvel état, U[0,1) à 53 bits)."""
    st = st + np.uint64(0x9E3779B97F4A7C15)
    return st, float(np.int64(_mix(st) >> np.uint64(11))) * (1.0 / 9007199254740992.0)   # conversion signée (native)


@njit(inline='always', **_JIT)
def _hb_rapide(A, beta, key, lid):
    det = (A[0] * A[3] - A[1] * A[2]).real
    kk = np.sqrt(max(det, 1e-300))
    Vn = (A[0] / kk, A[1] / kk, A[2] / kk, A[3] / kk)
    a = beta * kk
    st = _mix(key ^ _mix(np.uint64(lid) + np.uint64(1)))
    if a >= 2.0:
        # Kennedy–Pendleton (1985) : δ = 1 − x ~ Γ(3/2, α) exact (−[ln u1 + cos²(2π u2) ln u3]/α), accepté avec √(1 − δ/2)
        # ⟹ x ∝ √(1−x²) e^{αx} sur [−1, 1] ; acceptation → 1 quand α croît (la proposition exponentielle tombe à ~1,25/√α).
        while True:
            st, u1 = _u01(st)
            st, u2 = _u01(st)
            st, u3 = _u01(st)
            st, u4 = _u01(st)
            cs = np.cos(2.0 * np.pi * u2)
            d = -(np.log(1.0 - u1) + cs * cs * np.log(1.0 - u3)) / a      # 1 − u ∈ (0, 1] : log fini
            if u4 * u4 <= 1.0 - 0.5 * d:
                x = 1.0 - d
                break
    else:
        em = -np.expm1(-2.0 * a)
        while True:
            st, u = _u01(st)
            if a < 1e-12:                       # limite α → 0 : proposition uniforme (e^{αx} = 1 + O(α))
                x = 2.0 * u - 1.0
            else:
                x = 1.0 + np.log1p(-u * em) / a
            st, u2 = _u01(st)
            if u2 < np.sqrt(max(1.0 - x * x, 0.0)):
                break
    r = np.sqrt(max(1.0 - x * x, 0.0))
    while True:                             # direction uniforme sur S² (Marsaglia 1972) : pas de trigonométrie
        st, u3 = _u01(st)
        st, u4 = _u01(st)
        y1 = 2.0 * u3 - 1.0; y2 = 2.0 * u4 - 1.0; q = y1 * y1 + y2 * y2
        if q < 1.0:
            break
    h = 2.0 * np.sqrt(1.0 - q)
    v0 = y1 * h; v1 = y2 * h; v2 = 1.0 - 2.0 * q
    W = (complex(x, r * v2), complex(r * v1, r * v0), complex(-r * v1, r * v0), complex(x, -r * v2))
    return _mm(W, _dg(Vn))


@njit(parallel=True, **_JIT)
def _k_hb_test(A, beta, key, out):
    """TEST : n tirages indépendants du bain thermique rapide pour UNE agrafe A (2×2) ; out[i] = ½ReTr(U A)/k = x₀.
    Loi exacte : x₀ ∝ √(1−x₀²) e^{αx₀}, α = βk, k = √det A."""
    T = (A[0, 0], A[0, 1], A[1, 0], A[1, 1])
    k = np.sqrt(max((T[0] * T[3] - T[1] * T[2]).real, 1e-300))
    for i in prange(out.shape[0]):
        U = _hb_rapide(T, beta, key, i)
        P = _mm(U, T)
        out[i] = 0.5 * (P[0] + P[3]).real / k


@njit(parallel=True, **_JIT)
def _k_pairs_rapide(Cf, mu, p1, p1pm, Vfix, nbp, nbm, beta, key):
    nl = Cf.shape[1]
    for i in prange(p1.shape[1]):
        s1 = p1[mu, i]
        A1 = _staple(Cf, mu, s1, nbp, nbm)
        A2 = _staple(Cf, mu, p1pm[mu, i], nbp, nbm)
        V = _ld(Vfix, mu, i)
        U1 = _hb_rapide(_add(A1, _mm(_dg(A2), _dg(V))), beta, key, mu * nl + s1)
        _st4(Cf, mu, s1, U1)
        _st4(Cf, mu, p1pm[mu, i], _mm(_dg(U1), V))


@njit(parallel=True, **_JIT)
def _k_free_rapide(Cf, mu, sites, nbp, nbm, beta, key):
    nl = Cf.shape[1]
    for i in prange(sites.shape[0]):
        s = sites[i]
        _st4(Cf, mu, s, _hb_rapide(_staple(Cf, mu, s, nbp, nbm), beta, key, mu * nl + s))


# =============================================================================================== API balayage
def V_pairs(C, geo):
    """(4, npair, 2, 2) : U_mu(x) U_mu(x+mu) aux x de pair1 (ordre flatnonzero) = lib_brancheC.V_pairs."""
    Cf = _flat(C, geo)
    out = np.empty((4, geo.p1.shape[1], 2, 2), np.complex128)
    _k_vpairs(Cf, geo.p1, geo.p1pm, out)
    return out


def agrafes(C, geo, mu, sites):
    Cf = _flat(C, geo)
    sites = np.ascontiguousarray(sites, dtype=np.int64)
    out = np.empty((len(sites), 2, 2), np.complex128)
    _k_agrafes(Cf, mu, sites, geo.nbp, geo.nbm, out)
    return out


def _vfix(Vfix, geo):
    Vf = np.ascontiguousarray(np.asarray(Vfix if not isinstance(Vfix, list) else np.stack(Vfix)), dtype=np.complex128)
    if Vf.shape != (4, geo.p1.shape[1], 2, 2):
        raise ValueError('Vfix : forme (4, npair, 2, 2) attendue')
    return Vf


def sweep(C, beta, rng, geo, Vfix, mode='ref_rng'):
    """Un balayage de la mesure conditionnelle Wilson(β) | dec(C) = Vfix. Modifie C en place et le renvoie.
    mode 'ref_rng' : rng = np.random.Generator, consommé EXACTEMENT comme heatbath_su2 (concordance ≤ 1e-12 / balayage).
    mode 'rapide'  : rng = np.random.Generator ; UN entier 64 bits tiré par balayage sert de clé ; tirages par lien compilés."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng : np.random.Generator attendu (une graine entière répétée rejouerait les mêmes tirages)')
    Cf = _flat(C, geo)
    Vf = _vfix(Vfix, geo)
    beta = float(beta)
    _k_reunit(Cf)
    if mode == 'rapide':
        key = np.uint64(rng.integers(0, 2 ** 63, dtype=np.int64))
        for mu in range(4):
            _k_pairs_rapide(Cf, mu, geo.p1, geo.p1pm, Vf, geo.nbp, geo.nbm, beta, np.uint64(key ^ np.uint64(mu + 1)))
        for mu in range(4):
            for p in (0, 1):
                _k_free_rapide(Cf, mu, geo.fr[mu][p], geo.nbp, geo.nbm, beta,
                               np.uint64(key ^ np.uint64((8 + 2 * mu + p) << 56)))
        return C
    if mode != 'ref_rng':
        raise ValueError(mode)
    if beta == 0.0:
        raise ValueError('β = 0 : heatbath_su2 de référence ne termine pas (α = 0 ⟹ x = NaN) ; utiliser mode="rapide"')
    n = geo.p1.shape[1]
    for mu in range(4):
        Vn = np.empty((n, 2, 2), np.complex128); k = np.empty(n); al = np.empty(n)
        _k_pairs_prep(Cf, mu, geo.p1, geo.p1pm, Vf, geo.nbp, geo.nbm, beta, Vn, k, al)
        x0, r, v = _hb_numpy(al, rng)
        _k_pairs_fin(Cf, mu, geo.p1, geo.p1pm, Vf, x0, r, v, Vn)
    for mu in range(4):
        for p in (0, 1):
            s = geo.fr[mu][p]; m = len(s)
            Vn = np.empty((m, 2, 2), np.complex128); k = np.empty(m); al = np.empty(m)
            _k_free_prep(Cf, mu, s, geo.nbp, geo.nbm, beta, Vn, k, al)
            x0, r, v = _hb_numpy(al, rng)
            _k_free_fin(Cf, mu, s, x0, r, v, Vn)
    return C


# =============================================================================================== mesure de fibre
@njit(parallel=True, **_JIT)
def _k_plaq(Uf, E, pv, gp):
    """pv[p] = ½ ReTr U_p ; gp[p, i, a] = d/ds ½ ReTr(U_p) pour e^{sX_a} sur le lien i de p (dérivée à droite)."""
    for p in prange(E.shape[0]):
        W0 = _ld3(Uf, E[p, 0]); W1 = _ld3(Uf, E[p, 1]); W2 = _dg(_ld3(Uf, E[p, 2])); W3 = _dg(_ld3(Uf, E[p, 3]))
        P1 = W0                         # préfixe du lien 1 (signe +)
        P2 = _mm(P1, W1)                # préfixe du lien 2 (signe − : inclut W2)
        P2 = _mm(P2, W2)
        Up = _mm(P2, W3)                # préfixe du lien 3 (signe −) = Up
        pv[p] = 0.5 * (Up[0] + Up[3]).real
        for i in range(4):
            if i == 0:
                Q = Up                   # préfixe = 1
                sg = 1.0
            elif i == 1:
                Q = _mm(_mm(_dg(P1), Up), P1); sg = 1.0
            elif i == 2:
                Q = _mm(_mm(_dg(P2), Up), P2); sg = -1.0
            else:
                Q = _mm(_mm(_dg(Up), Up), Up); sg = -1.0
            # ½ s Re Tr(X_a Q), X_a = iσ_a/√2
            c = 0.5 * sg * 0.7071067811865476
            gp[p, i, 0] = -c * (Q[2] + Q[1]).imag
            gp[p, i, 1] = c * (Q[2] - Q[1]).real
            gp[p, i, 2] = -c * (Q[0] - Q[3]).imag


@njit(inline='always', **_JIT)
def _grad_link(e, lp_plaq, lp_pos, gp, wptr, widx, wval, out):
    """out[a, :] += Σ_{plaquettes p ∋ e} gp[p, pos, a] · Wf[:, p]."""
    for j in range(6):
        p = lp_plaq[e, j]; pos = lp_pos[e, j]
        for q in range(wptr[p], wptr[p + 1]):
            w = wval[q]; c = widx[q]
            for a in range(3):
                out[a, c] += gp[p, pos, a] * w


@njit(parallel=True, **_JIT)
def _k_fibre_D(Uf, fib_free, fib_e1, fib_e2, lp_plaq, lp_pos, gp, wptr, widx, wval, nF, nch):
    nfree = fib_free.shape[0]; npr = fib_e1.shape[0]; ntot = nfree + npr
    Dl = np.zeros((nch, nF, nF))
    for c in prange(nch):
        G1 = np.zeros((3, nF)); G2 = np.zeros((3, nF)); row = np.zeros(nF); nz = np.empty(nF, np.int64)
        lo = (ntot * c) // nch; hi = (ntot * (c + 1)) // nch
        D = Dl[c]
        for t in range(lo, hi):
            G1[:, :] = 0.0
            if t < nfree:
                _grad_link(fib_free[t], lp_plaq, lp_pos, gp, wptr, widx, wval, G1)
                for a in range(3):
                    m = 0
                    for i in range(nF):
                        if G1[a, i] != 0.0:
                            nz[m] = i; m += 1
                    for ii in range(m):
                        xi = G1[a, nz[ii]]
                        for jj in range(m):
                            D[nz[ii], nz[jj]] += xi * G1[a, nz[jj]]
            else:
                q = t - nfree
                e1 = fib_e1[q]; e2 = fib_e2[q]
                G2[:, :] = 0.0
                _grad_link(e1, lp_plaq, lp_pos, gp, wptr, widx, wval, G1)
                _grad_link(e2, lp_plaq, lp_pos, gp, wptr, widx, wval, G2)
                U = _ld3(Uf, e1)
                Ud = _dg(U)
                # R_ba = ½ Re Tr(σ_b U† σ_a U) (= −Tr(X_b U† X_a U))
                for a in range(3):
                    if a == 0:
                        S = (0j, 1 + 0j, 1 + 0j, 0j)
                    elif a == 1:
                        S = (0j, -1j, 1j, 0j)
                    else:
                        S = (1 + 0j, 0j, 0j, -1 + 0j)
                    Y = _mm(_mm(Ud, S), U)
                    R0 = 0.5 * (Y[2] + Y[1]).real
                    R1 = 0.5 * (Y[2].imag - Y[1].imag)
                    R2 = 0.5 * (Y[0] - Y[3]).real
                    m = 0
                    for i in range(nF):
                        row[i] = G1[a, i] - (R0 * G2[0, i] + R1 * G2[1, i] + R2 * G2[2, i])
                        if row[i] != 0.0:
                            nz[m] = i; m += 1
                    for ii in range(m):
                        xi = 0.5 * row[nz[ii]]
                        for jj in range(m):
                            D[nz[ii], nz[jj]] += xi * row[nz[jj]]
    Dt = np.zeros((nF, nF))
    for c in range(nch):
        Dt += Dl[c]
    return Dt


@njit(inline='always', **_JIT)
def _Rcol(U, Ud, S):
    """(R_0a, R_1a, R_2a), R_ba = ½ Re Tr(σ_b U† σ_a U), σ_a = S."""
    Y = _mm(_mm(Ud, S), U)
    return 0.5 * (Y[2] + Y[1]).real, 0.5 * (Y[2].imag - Y[1].imag), 0.5 * (Y[0] - Y[3]).real


@njit(parallel=True, **_JIT)
def _k_rows(Uf, lfree, e1s, e2s, lp_plaq, lp_pos, gp, wptr, widx, wval, cmap, out):
    """Lignes de Gf (coordonnées de fibre) restreintes aux colonnes cmap ≥ 0 : 3 lignes par élément.
    Éléments 0..nfree-1 : liens libres (métrique 1) ; puis paires (e1, e2), ligne_a = ∇_{e1,a} − Σ_b R_ba ∇_{e2,b},
    R_ba = ½ Re Tr(σ_b U₁† σ_a U₁) = −Tr(X_b U₁† X_a U₁), divisée par √2 (métrique 2). out doit être nul en entrée."""
    nfree = lfree.shape[0]; ntot = nfree + e1s.shape[0]
    for t in prange(ntot):
        r0 = 3 * t
        if t < nfree:
            e = lfree[t]
            for j in range(6):
                p = lp_plaq[e, j]; pos = lp_pos[e, j]
                for q in range(wptr[p], wptr[p + 1]):
                    c = cmap[widx[q]]; w = wval[q]
                    for a in range(3):
                        out[r0 + a, c] += gp[p, pos, a] * w
        else:
            u = t - nfree; e1 = e1s[u]; e2 = e2s[u]
            U = _ld3(Uf, e1); Ud = _dg(U)
            R00, R10, R20 = _Rcol(U, Ud, (0j, 1 + 0j, 1 + 0j, 0j))          # colonne a = 0 (σ1)
            R01, R11, R21 = _Rcol(U, Ud, (0j, -1j, 1j, 0j))                 # a = 1 (σ2)
            R02, R12, R22 = _Rcol(U, Ud, (1 + 0j, 0j, 0j, -1 + 0j))         # a = 2 (σ3)
            for j in range(6):
                p = lp_plaq[e1, j]; pos = lp_pos[e1, j]
                for q in range(wptr[p], wptr[p + 1]):
                    c = cmap[widx[q]]; w = wval[q]
                    for a in range(3):
                        out[r0 + a, c] += gp[p, pos, a] * w
            for j in range(6):
                p = lp_plaq[e2, j]; pos = lp_pos[e2, j]
                for q in range(wptr[p], wptr[p + 1]):
                    c = cmap[widx[q]]; w = wval[q]
                    g0 = gp[p, pos, 0] * w; g1 = gp[p, pos, 1] * w; g2 = gp[p, pos, 2] * w
                    out[r0, c] -= R00 * g0 + R10 * g1 + R20 * g2
                    out[r0 + 1, c] -= R01 * g0 + R11 * g1 + R21 * g2
                    out[r0 + 2, c] -= R02 * g0 + R12 * g1 + R22 * g2
            for a in range(3):
                for c in range(out.shape[1]):
                    out[r0 + a, c] *= 0.7071067811865476


def _supports(geo, prep):
    """colonnes de Wf atteintes par les liens de direction mu (support des lignes de fibre), par mu."""
    ptr, idx, val, nF = prep
    prow = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    sup = []
    for mu in range(4):
        m = np.zeros(len(ptr) - 1, bool); m[geo.lp_plaq[mu * geo.V:(mu + 1) * geo.V].ravel()] = True
        sup.append(np.unique(idx[m[prow]]))
    return sup


CHUNK = 1 << 16
def _D_blas(Uf, geo, gp, prep, libre):
    """D = Σ_lignes ligne ligneᵀ : lignes remplies par numba (tranches de CHUNK éléments, colonnes du support de mu),
    produit Gᵀ G par BLAS (numpy détecte Gᵀ·G ⟹ syrk)."""
    ptr, idx, val, nF = prep
    if getattr(geo, '_sup', (None,))[0] is not prep[1]:      # supports en cache, liés au tableau idx du Wf préparé
        geo._sup = (prep[1], _supports(geo, prep))
    sup = geo._sup[1]
    D = np.zeros((nF, nF)); vide = np.zeros(0, np.int64)
    for mu in range(4):
        cols = sup[mu]
        if len(cols) == 0:
            continue
        cmap = np.full(nF, -1, np.int64); cmap[cols] = np.arange(len(cols))
        lf = (mu * geo.V + np.arange(geo.V)) if libre else (mu * geo.V + geo.free_all[mu])
        e1 = vide if libre else mu * geo.V + geo.p1[mu]; e2 = vide if libre else mu * geo.V + geo.p1pm[mu]
        items = [('f', lf[i:i + CHUNK]) for i in range(0, len(lf), CHUNK)] + \
                [('p', (e1[i:i + CHUNK], e2[i:i + CHUNK])) for i in range(0, len(e1), CHUNK)]
        Dm = np.zeros((len(cols), len(cols)))
        for kind, it in items:
            if kind == 'f':
                a, b1, b2 = np.ascontiguousarray(it, dtype=np.int64), vide, vide
            else:
                a, b1, b2 = vide, np.ascontiguousarray(it[0], dtype=np.int64), np.ascontiguousarray(it[1], dtype=np.int64)
            G = np.zeros((3 * (len(a) + len(b1)), len(cols)))
            _k_rows(Uf, a, b1, b2, geo.lp_plaq, geo.lp_pos, gp, ptr, idx, val, cmap, G)
            Dm += G.T @ G
        D[np.ix_(cols, cols)] += Dm
    return D


def block_sums(pv, L):
    """= lib_Cb.block_sums (même reshape/sum numpy ⟹ identique à l'octet)."""
    Lc = L // 2
    a = pv.reshape(6, L, L, L, L)
    a = a.reshape(6, Lc, 2, Lc, 2, Lc, 2, Lc, 2).sum(axis=(2, 4, 6, 8))
    return a.reshape(6, -1)


@njit(parallel=True, **_JIT)
def _k_F_neumaier(ptr, idx, val, pv, F):
    """F_i = Σ_q val_q pv_idx_q, sommation compensée de Neumaier (erreur ≈ 1 ulp de F, indépendante de L)."""
    for i in prange(F.shape[0]):
        s = 0.0; c = 0.0
        for q in range(ptr[i], ptr[i + 1]):
            x = val[q] * pv[idx[q]]
            t = s + x
            if abs(s) >= abs(x):
                c += (s - t) + x
            else:
                c += (x - t) + s
            s = t
        F[i] = s + c


_FPREP = [None, None]
def _F(Wf, pv):
    """dense : Wf @ pv (même produit que la référence) ; scipy.sparse : somme compensée par ligne."""
    import scipy.sparse as sp
    if not sp.issparse(Wf):
        return Wf @ pv
    if _FPREP[0] is not Wf:
        W = sp.csr_matrix(Wf)
        _FPREP[0] = Wf; _FPREP[1] = (W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64), W.shape[0])
    ptr, idx, val, n = _FPREP[1]
    F = np.empty(n)
    _k_F_neumaier(ptr, idx, val, pv, F)
    return F


def plaquettes(C, geo):
    """(pv (6V,), gp (6V, 4, 3)) : valeurs ½ReTr U_p et gradients par lien de plaquette (ordre loops_plaquettes)."""
    Uf = _flat(C, geo).reshape(-1, 2, 2)
    pv = np.empty(6 * geo.V); gp = np.empty((6 * geo.V, 4, 3))
    _k_plaq(Uf, geo.E, pv, gp)
    return pv, gp


def mesure(C, geo, Wf, nch=None, methode='blas'):
    """(F, Dcontrib, blk) = (Wf @ pv, Gf.T @ (Gf/g), block_sums(pv).astype(float32)) du bloc mesure de run_chain.
    Gf = T^T ∂P Wf^T n'est jamais formé : D est accumulé ligne de fibre par ligne de fibre (Wf creux par plaquette).
    Wf dense : F = Wf @ pv, MÊME produit numpy que la référence (concordance ≤ 1e-13 absolue).
    Wf scipy.sparse (fourier_weights_sparse, conseillé pour L ≥ 24) : F par somme compensée (Neumaier), ≈ 1 ulp."""
    Uf = _flat(C, geo).reshape(-1, 2, 2)
    pv, gp = plaquettes(C, geo)
    F = _F(Wf, pv)
    ptr, idx, val, nF = geo.wf_sparse(Wf)
    if nch is None:
        nch = 4 * nb.get_num_threads()
    if methode == 'blas':
        D = _D_blas(Uf, geo, gp, (ptr, idx, val, nF), False)
    else:
        D = _k_fibre_D(Uf, geo.fib_free, geo.fib_e1, geo.fib_e2, geo.lp_plaq, geo.lp_pos, gp, ptr, idx, val, nF, nch)
    return F, D, block_sums(pv, geo.L).astype(np.float32)


def mesure_libre(C, geo, Wf, nch=None, methode='blas'):
    """(F, D) SANS conditionnement (tous les liens libres, métrique 1) : D = Gf.T @ Gf avec Gf = Bg @ Wf.T,
    comme gate_G10_GG3b (témoin Haar G10)."""
    Uf = _flat(C, geo).reshape(-1, 2, 2)
    pv, gp = plaquettes(C, geo)
    F = _F(Wf, pv)
    ptr, idx, val, nF = geo.wf_sparse(Wf)
    if nch is None:
        nch = 4 * nb.get_num_threads()
    tous = np.arange(4 * geo.V, dtype=np.int64); vide = np.zeros(0, np.int64)
    if methode == 'blas':
        D = _D_blas(Uf, geo, gp, (ptr, idx, val, nF), True)
    else:
        D = _k_fibre_D(Uf, tous, vide, vide, geo.lp_plaq, geo.lp_pos, gp, ptr, idx, val, nF, nch)
    return F, D


# =============================================================================================== chaîne (boucle de run_chain)
def chaine(C, beta, rng, Wf, geo=None, ntherm=100, nsw=500, meas_every=2, binw=5, mode='ref_rng', Vfix=None, log=None):
    """Thermalisation + un segment de run_chain (sans kfib ni prolongation PREREG §3.3) ; modifie C.
    Renvoie dict(F (nmeas,nF), Db (nbin,nF,nF), blk (nmeas,6,Lc^4) f32, Vdev, unit, t_sweep, t_meas)."""
    import time
    geo = geo or Geometry(C.shape[1])
    Vf = V_pairs(C, geo) if Vfix is None else _vfix(Vfix, geo)
    t0 = time.time()
    for _ in range(ntherm):
        sweep(C, beta, rng, geo, Vf, mode)
    Fs, Db, blks = [], [], []
    Dacc = np.zeros((Wf.shape[0],) * 2); nm = 0; tm = 0.0
    for it in range(nsw):
        sweep(C, beta, rng, geo, Vf, mode)
        if (it + 1) % meas_every == 0:
            t1 = time.time()
            F, D, blk = mesure(C, geo, Wf)
            Fs.append(F); Dacc += D; blks.append(blk); nm += 1
            if nm % binw == 0:
                Db.append(Dacc / binw); Dacc = np.zeros_like(Dacc)
            tm += time.time() - t1
    Vn = V_pairs(C, geo)
    Cf = _flat(C, geo)
    uni = float(np.abs(np.einsum('mxij,mxkj->mxik', Cf, Cf.conj()) - np.eye(2)).max())
    return dict(F=np.array(Fs), Db=np.array(Db), blk=np.array(blks), Vdev=float(np.abs(Vn - Vf).max()), unit=uni,
                t_total=time.time() - t0, t_meas=tm)
