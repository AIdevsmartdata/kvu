#!/usr/bin/env python3
"""Concordance noyau cb ↔ références (lib_Cb 64967707…, lib_brancheC gelé 2392599f…). Tolérances : CIBLES.md §CIBLE 2.
Usage : test_concordance.py [--rapide-seulement] [--Lmax 16] [--sortie logs/concordance_<machine>.json]
RC = 0 si TOUTES les gates sont vertes, 1 sinon, 2 si une référence n'a pas le bon SHA."""
import os, sys, json, time, hashlib, socket, argparse, platform
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'ref', 'code'))
sys.dont_write_bytecode = True

ap = argparse.ArgumentParser()
ap.add_argument('--Lmax', type=int, default=16)
ap.add_argument('--sortie', default=None)
args = ap.parse_args()

# ---------------------------------------------------------------- SHA des références AVANT import
SHA_ATT = {l.split()[1]: l.split()[0] for l in open(os.path.join(HERE, 'ref', 'SHA_ref.txt')) if l.strip()}
def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()
for rel, s in SHA_ATT.items():
    if sha(os.path.join(HERE, rel)) != s:
        print('ECHEC SHA', rel); sys.exit(2)
SHA_CFG = {l.split()[1]: l.split()[0] for l in open(os.path.join(HERE, 'data', 'SHA_cfg.txt')) if l.strip()}

import lib_Cb as R, lib_brancheC as B
import numba, scipy, scipy.sparse as sp
import cb

TOL = dict(agrafes=1e-14, vpairs=1e-14, sweep=1e-12, vdev=1e-12, unit=1e-12, F=1e-13, D=1e-12, gc1=1e-14)
res = dict(machine=socket.gethostname(), date=time.strftime('%Y-%m-%dT%H:%M:%S%z'), python=platform.python_version(),
           numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, threads=numba.get_num_threads(),
           sha_refs=SHA_ATT, tol=TOL, cas=[], gates={})
ok_all = True
def gate(nom, val, tol, info=None):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    g = res['gates'].setdefault(nom, dict(max=0.0, tol=tol, n=0, ok=True))
    g['max'] = max(g['max'], float(val)); g['n'] += 1; g['ok'] &= ok
    if not ok:
        print('  ROUGE', nom, val, '>', tol, info or '', flush=True)
    return ok
def gbool(nom, ok, info=None):
    global ok_all
    ok_all &= bool(ok)
    g = res['gates'].setdefault(nom, dict(n=0, ok=True)); g['n'] += 1; g['ok'] &= bool(ok)
    if not ok: print('  ROUGE', nom, info or '', flush=True)

# ---------------------------------------------------------------- configurations
def from_q(q):
    C = np.empty(q.shape[:-1] + (2, 2), complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]; C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]; C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return C
def cfg_flat(L):
    return np.ascontiguousarray(np.broadcast_to(np.eye(2, dtype=complex), (4,) + (L,) * 4 + (2, 2)))
def cfg_pi(L):
    """= flot_su2.cfg_frustre : liens ±1, toutes les plaquettes = −1."""
    x = np.indices((L,) * 4)
    eps = [np.ones((L,) * 4), (-1.0) ** x[0], (-1.0) ** (x[0] + x[1]), (-1.0) ** (x[0] + x[1] + x[2])]
    C = cfg_flat(L).copy()
    for mu in range(4):
        C[mu] = eps[mu][..., None, None] * np.eye(2)
    return C
def cfg_haar(L, seed):
    q = np.random.default_rng(1000 + seed).standard_normal((4,) + (L,) * 4 + (4,))
    return from_q(q / np.linalg.norm(q, axis=-1, keepdims=True))
def cfg_prod(L):
    f = f'L{L}_b2.3.npy'; p = os.path.join(HERE, 'data', f)
    assert sha(p) == SHA_CFG[f], 'SHA cfg ' + f
    return np.ascontiguousarray(np.load(p))

def plaquette_moy(C):
    E, s = B.loops_plaquettes(C.shape[1]); pv, _ = R.plaq_values_grads_cached(C); return float(pv.mean())

# plan de test : (L, nom, fabrique, betas, graines)
PLAN = []
for L in (4, 8):
    PLAN += [(L, 'plat', lambda L=L: cfg_flat(L), (2.3, 2.8, 16.0), (0, 1, 2)),
             (L, 'pi_flux', lambda L=L: cfg_pi(L), (2.3, 2.8, 16.0), (0, 1, 2))]
    PLAN += [(L, f'haar{s}', lambda L=L, s=s: cfg_haar(L, s), (2.3, 2.8, 16.0), (0, 1, 2)) for s in range(3)]
PLAN += [(12, 'haar0', lambda: cfg_haar(12, 0), (2.8,), (0,)),
         (12, 'prod_b2.3', lambda: cfg_prod(12), (2.3, 16.0), (0, 1, 2)),
         (16, 'prod_b2.3', lambda: cfg_prod(16), (2.3,), (0, 1)),
         (16, 'plat', lambda: cfg_flat(16), (16.0,), (0,))]
PLAN = [p for p in PLAN if p[0] <= args.Lmax]

geos = {}
t00 = time.time()
for (L, nom, fab, betas, seeds) in PLAN:
    t0 = time.time()
    C0 = fab(); geo = geos.setdefault(L, cb.Geometry(L)); M = B.masks_dec(L)
    cas = dict(L=L, cfg=nom, plaquette=plaquette_moy(C0))
    # --- géométrie : masques, plaquettes (plaq_order_check), fibre
    if nom in ('plat', 'haar0'):
        gbool('masks_dec_identiques', all(np.array_equal(a, b) for a, b in zip(M, geo.M)))
        E, s = B.loops_plaquettes(L)
        gbool('plaquettes_identiques(plaq_order_check)', np.array_equal(E, geo.E) and R.plaq_order_check(L))
        T, g = B.fiber_map(C0, M)
        gbool('n_fib', T.shape[1] == geo.n_fib)
    # --- V_pairs
    Vr = B.V_pairs(C0, M); Vm = cb.V_pairs(C0, geo)
    gate('V_pairs', max(np.abs(Vr[m] - Vm[m]).max() for m in range(4)), TOL['vpairs'])
    # --- agrafes, tous les ensembles de sites du balayage
    nbr, p1, p1pm, fr = R._sweep_cache(L, M)
    Cf = C0.reshape(4, -1, 2, 2)
    da = 0.0
    for mu in range(4):
        for s in (p1[mu], p1pm[mu], fr[mu][0], fr[mu][1]):
            da = max(da, np.abs(R.agrafes_at(Cf, mu, s, nbr) - cb.agrafes(C0, geo, mu, s)).max())
    gate('agrafes', da, TOL['agrafes']); cas['agrafes'] = da
    # --- un balayage ref_rng (GS1 : contre lib_Cb ET contre le gelé sweep_cond_dec aux petits L)
    cas['sweep'] = []
    for beta in betas:
        for seed in seeds:
            Ca = C0.copy(); Cm = C0.copy()
            R.sweep_cond_dec_fast(Ca, beta, np.random.default_rng(seed), M, Vr)
            cb.sweep(Cm, beta, np.random.default_rng(seed), geo, Vr, mode='ref_rng')
            d = float(np.abs(Ca - Cm).max())
            gate('sweep_ref_rng_1_balayage', d, TOL['sweep'], (L, nom, beta, seed))
            dg = None
            if L <= 8 or (L == 12 and seed == 0):
                Cg = C0.copy(); B.sweep_cond_dec(Cg, beta, np.random.default_rng(seed), M, Vr)
                dg = float(np.abs(Cg - Cm).max()); gate('GS1_contre_gele_sweep_cond_dec', dg, TOL['sweep'], (L, nom, beta, seed))
            Vn = cb.V_pairs(Cm, geo); vdev = float(np.abs(Vn - np.stack(Vr)).max())
            uni = float(max(np.abs(Cm[mu] @ B.dag(Cm[mu]) - np.eye(2)).max() for mu in range(4)))
            gate('Vdev_ref_rng', vdev, TOL['vdev']); gate('unitarite_ref_rng', uni, TOL['unit'])
            cas['sweep'].append(dict(beta=beta, seed=seed, dU=d, dU_gele=dg, Vdev=vdev, unit=uni))
    # --- mode rapide : invariants sur 5 balayages (+ β = 0)
    for beta in (betas[0], 0.0):
        Cm = C0.copy(); rng = np.random.default_rng(7)
        for _ in range(5):
            cb.sweep(Cm, beta, rng, geo, Vr, mode='rapide')
        vdev = float(np.abs(cb.V_pairs(Cm, geo) - np.stack(Vr)).max())
        uni = float(max(np.abs(Cm[mu] @ B.dag(Cm[mu]) - np.eye(2)).max() for mu in range(4)))
        gate('Vdev_rapide_5_balayages', vdev, TOL['vdev']); gate('unitarite_rapide_5_balayages', uni, TOL['unit'])
        gbool('rapide_fini', np.isfinite(Cm).all())
    try:
        cb.sweep(C0.copy(), 0.0, np.random.default_rng(0), geo, Vr, mode='ref_rng'); gbool('ref_rng_beta0_refuse', False)
    except ValueError:
        gbool('ref_rng_beta0_refuse', True)
    # --- mesure : bloc de run_chain
    Wf = B.fourier_weights(L, 1); WfT = np.ascontiguousarray(Wf.T)
    T, g = B.fiber_map(C0, M); pv, Bg = R.plaq_values_grads_cached(C0)
    Fr = Wf @ pv; Gf = T.T @ (Bg @ WfT); Dr = Gf.T @ (Gf / g[:, None]); br = R.block_sums(pv, L).astype(np.float32)
    F, D, blk = cb.mesure(C0, geo, Wf)
    nDr = float(np.linalg.norm(Dr)); den = nDr if nDr > 0 else 1.0     # plat / π-flux : gradients nuls ⟹ D_ref = 0, écart ABSOLU
    dF = float(np.abs(F - Fr).max()); dD = float(np.linalg.norm(D - Dr) / den)
    gate('mesure_F_abs', dF, TOL['F']); gate('mesure_D_frob_rel', dD, TOL['D']); gbool('mesure_blk_identique', np.array_equal(blk, br))
    Ws = cb.fourier_weights_sparse(L)
    gbool('fourier_sparse_identique', np.array_equal(Ws.toarray(), Wf))
    Fs, Ds, _ = cb.mesure(C0, geo, Ws)
    # F creux (somme compensée) : comparé à la somme EXACTEMENT arrondie (math.fsum) ; l'écart à la référence dense doit
    # s'expliquer par l'erreur d'arrondi de la référence elle-même : |F_sp − F_ref| ≤ |F_ref − F_exact| + 1e-14·max(1,|F|)
    import math
    Fx = np.array([math.fsum(Wf[i] * pv) for i in range(Wf.shape[0])])
    dFx = float(np.max(np.abs(Fs - Fx) / np.maximum(1, np.abs(Fx))))
    gate('mesure_sparse_F_vs_somme_exacte (rel. 1e-14)', dFx, 1e-14)
    dFs = float(np.abs(Fs - Fr).max())
    marge = float(np.max(np.abs(Fs - Fr) - (np.abs(Fr - Fx) + 1e-14 * np.maximum(1, np.abs(Fx)))))
    gate('mesure_sparse_F_ecart_ref_explique_par_arrondi_ref', max(marge, 0.0), 0.0)
    cas['F_ref_err_vs_exact'] = float(np.abs(Fr - Fx).max())
    dDs = float(np.linalg.norm(Ds - Dr) / den)
    gate('mesure_sparse_D_frob_rel', dDs, TOL['D'])
    Fb, Db_, _ = cb.mesure(C0, geo, Wf, methode='boucle')
    gate('mesure_voie_boucle_D_frob_rel', float(np.linalg.norm(Db_ - Dr) / den), TOL['D'])
    Gl = Bg @ WfT; Dlr = Gl.T @ Gl; Fl, Dl = cb.mesure_libre(C0, geo, Wf); nl = float(np.linalg.norm(Dlr)) or 1.0
    gate('mesure_libre_D_frob_rel (G10)', float(np.linalg.norm(Dl - Dlr) / nl), TOL['D']); gate('mesure_libre_F_abs', float(np.abs(Fl - Fr).max()), TOL['F'])
    cas.update(dF=dF, dD=dD, dF_sparse=dFs, dD_sparse=dDs, Dnorm=float(np.linalg.norm(Dr)), Fmax=float(np.abs(Fr).max()))
    # --- GC1 : valeurs et gradients de plaquettes contre le gelé plaq_values_grads
    if L <= 12:
        pvg, Bgg, Eg = B.plaq_values_grads(C0)
        pvm, gp = cb.plaquettes(C0, geo)
        npl = len(Eg)
        rows = np.concatenate([3 * Eg[:, i] + a for i in range(4) for a in range(3)])
        cols = np.concatenate([np.arange(npl)] * 12)
        vals = np.concatenate([gp[:, i, a] for i in range(4) for a in range(3)])
        Bm = sp.coo_matrix((vals, (rows, cols)), shape=Bgg.shape).tocsr()
        d = max(float(np.abs(pvm - pvg).max()), float(abs(Bm - Bgg).max()))
        gate('GC1_plaq_values_grads_gele', d, TOL['gc1']); cas['GC1'] = d
    cas['t'] = round(time.time() - t0, 2)
    res['cas'].append(cas)
    print(f'L={L:2d} {nom:10s} plaq={cas["plaquette"]:+.4f} agr={da:.1e} dU={max(s["dU"] for s in cas["sweep"]):.1e} '
          f'F={dF:.1e} D={dD:.1e} Fsp={dFs:.1e} ({cas["t"]} s)', flush=True)

res['ok'] = bool(ok_all); res['t_total'] = round(time.time() - t00, 1)
out = args.sortie or os.path.join(HERE, 'logs', f'concordance_{socket.gethostname()}.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, 'w'), indent=1, default=float)
for k, g in res['gates'].items():
    print(f'{"VERT " if g["ok"] else "ROUGE"} {k:55s} n={g["n"]:3d}' + (f' max={g["max"]:.2e} tol={g["tol"]:.0e}' if 'max' in g else ''))
print('CONCORDANCE', 'VERTE' if ok_all else 'ROUGE', out)
sys.exit(0 if ok_all else 1)
