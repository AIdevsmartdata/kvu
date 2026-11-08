#!/usr/bin/env python3
"""Analyse GPU-1 (PLAN.md) — tourne SUR ACEMAGIC, sur les fichiers CUDA rapatriés.

P2 identité des tirages ; P3 plaquette CUDA contre AMD ; P4 C1-C4 ; P5 KS plaquettes locales
(permutation par configuration) ; P6 intégrité des dumps CUDA ; P8 débit.
Estimateurs : stat()/combine() importés de la COPIE SHA de analyse_plaquette.py (src_origine/).
Usage : python3 analyse_gpu1.py [--sans-ks]   -> analyse/resultats_gpu1.json
"""
import os, sys, glob, json, re, argparse
import numpy as np
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, 'src_origine'))
import analyse_plaquette as AP   # copie sha256 ec5fef48…

SRC = '/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926'
CUDA = os.path.join(ROOT, 'cuda', 'ens')
AMDN = os.path.join(ROOT, 'ref_amd', 'ens')
BETAS = ['2.3', '2.8', '3.5']


def refs_amd(L, b):
    """Références AMD par cellule : dict nom -> liste de dossiers de chaînes."""
    R = {}
    if b == '2.3':
        R['validation_G2'] = sorted(glob.glob(f'{SRC}/validation/validation/L{L}_b2.3_c*'))
    if L == 6 and b == '2.8':
        R['production_L6'] = sorted(glob.glob(f'{SRC}/ensembles/production_L6/L6_b2.8_c*'))
    n = sorted(glob.glob(f'{AMDN}/L{L}_b{b}_c*'))
    if n:
        R['neuve'] = n
    return R


def chaines(ds):
    out = []
    for d in ds:
        if not os.path.exists(os.path.join(d, 'COMPLETE')):
            raise SystemExit(f'INCOMPLETE {d}')
        out.append((os.path.basename(d), AP.analyse_chaine(d)))
    return out


def cellule(chs):
    """C1-C4 + combinaison, pour 3 chaînes c1, c2 (chaudes), c3 (froide)."""
    d = dict(chs)
    k = {n.split('_')[-1]: v for n, v in d.items()}
    m, e = AP.combine([v for v in d.values()])
    res = dict(moy=m, err=e, chaines={n: dict(moy=v['plaquette']['moy'], err=v['plaquette']['err'], tau=v['plaquette']['tau'],
                                               z_moities=v['z_moities'], C1_ok=v['C1_ok'], start=v['start'],
                                               tau_polyakov_max=v['tau_polyakov_max']) for n, v in d.items()})
    res['C1_ok'] = all(v['C1_ok'] for v in d.values())
    if all(c in k for c in ('c1', 'c2', 'c3')):
        mh, eh = AP.combine([k['c1'], k['c2']])
        z2 = (k['c3']['plaquette']['moy'] - mh) / np.hypot(k['c3']['plaquette']['err'], eh)
        z3 = (k['c1']['plaquette']['moy'] - k['c2']['plaquette']['moy']) / np.hypot(k['c1']['plaquette']['err'], k['c2']['plaquette']['err'])
        res.update(z_C2=float(z2), C2_ok=bool(abs(z2) < 3), z_C3=float(z3), C3_ok=bool(abs(z3) < 3))
    return res


def p3_p4():
    out = {}
    for L in (6, 8):
        for b in BETAS:
            key = f'L{L}_b{b}'
            cu = cellule(chaines(sorted(glob.glob(f'{CUDA}/L{L}_b{b}_c*'))))
            cu['C4_ok'] = all(1000 >= 5 * v['tau'] for v in cu['chaines'].values())
            r = dict(cuda=cu, amd={}, z={})
            allamd = []
            for nom, ds in refs_amd(L, b).items():
                chs = chaines(ds)
                allamd += [v for _, v in chs]
                a = cellule(chs)
                de = json.load(open(os.path.join(ds[0], 'manifest.json')))['dump_every']
                a['C4_ok'] = all(de >= 5 * v['tau'] for v in a['chaines'].values()) if de else None
                a['dump_every'] = de
                r['amd'][nom] = a
                z = (cu['moy'] - a['moy']) / np.hypot(cu['err'], a['err'])
                r['z'][nom] = dict(z=float(z), ok=bool(abs(z) < 3), delta=float(cu['moy'] - a['moy']))
            if not allamd:
                r['z_chi2'] = None; r['amd_combine'] = None; out[key] = r
                print(f"{key}: CUDA {cu['moy']:.7f}({cu['err']*1e7:.0f}) SANS REFERENCE AMD | C1 {cu['C1_ok']} C2 {cu.get('z_C2', 0):+.2f} C3 {cu.get('z_C3', 0):+.2f} C4 {cu['C4_ok']}", flush=True)
                continue
            m, e = AP.combine(allamd)
            r['z_chi2'] = float((cu['moy'] - m) / np.hypot(cu['err'], e))
            r['amd_combine'] = dict(moy=m, err=e, n_chaines=len(allamd))
            out[key] = r
            print(f"{key}: CUDA {cu['moy']:.7f}({cu['err']*1e7:.0f}) " +
                  ' '.join(f"{n} {a['moy']:.7f}({a['err']*1e7:.0f}) z={r['z'][n]['z']:+.2f}" for n, a in r['amd'].items()) +
                  f" | CUDA C1 {cu['C1_ok']} C2 {cu.get('z_C2', 0):+.2f} C3 {cu.get('z_C3', 0):+.2f} C4 {cu['C4_ok']}", flush=True)
    zs = np.array([v['z_chi2'] for v in out.values() if v['z_chi2'] is not None])
    chi2 = float((zs ** 2).sum()); p = float(sst.chi2.sf(chi2, len(zs)))
    gate = all(all(x['ok'] for x in v['z'].values()) for v in out.values()) and p > 0.01
    therm = {k: dict(cuda=all(v['cuda'].get(c, True) for c in ('C1_ok', 'C2_ok', 'C3_ok', 'C4_ok')),
                     amd={n: all(a.get(c, True) is not False for c in ('C1_ok', 'C2_ok', 'C3_ok', 'C4_ok')) for n, a in v['amd'].items()})
             for k, v in out.items()}
    return dict(cellules=out, chi2=chi2, ndf=len(zs), p_chi2=p, n_abs_z_gt2=int((np.abs(zs) > 2).sum()),
                P3_ok=bool(gate), P4=therm, P4_cuda_ok=all(t['cuda'] for t in therm.values()))


# ---------------------------------------------------------------- plaquettes locales
def load_U(f):
    return np.load(f)          # (4, L, L, L, L, 2, 2) complex128


def dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def plaq_locales(U):
    """½ ReTr U_mu(x) U_nu(x+mu) U_mu(x+nu)† U_nu(x)†, 6V valeurs."""
    out = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            A = U[mu] @ np.roll(U[nu], -1, axis=mu)
            B = U[nu] @ np.roll(U[mu], -1, axis=nu)          # (U_nu(x) U_mu(x+nu))† = U_mu(x+nu)† U_nu(x)†
            out.append(0.5 * np.real(np.einsum('...ij,...ji->...', A, dag(B))))
    return np.stack(out).ravel()


def cfgs(d):
    return sorted(glob.glob(os.path.join(d, 'cfg_*.npy')))


def p5(B=2000, graine=27092026, nb=20000):
    res = {}
    for L in (6, 8):
        jeux = {'cuda': [c for d in sorted(glob.glob(f'{CUDA}/L{L}_b2.8_c*')) for c in cfgs(d)],
                'amd_neuve': [c for d in sorted(glob.glob(f'{AMDN}/L{L}_b2.8_c*')) for c in cfgs(d)]}
        if L == 6:
            jeux['amd_production_L6'] = [c for d in sorted(glob.glob(f'{SRC}/ensembles/production_L6/L6_b2.8_c*')) for c in cfgs(d)]
        edges = np.linspace(-1, 1, nb + 1)
        H, X = {}, {}
        for k, fs in jeux.items():
            X[k] = [plaq_locales(load_U(f)) for f in fs]
            H[k] = np.array([np.histogram(x, edges)[0] for x in X[k]], dtype=np.int64)
        rL = {}
        for ref in [k for k in jeux if k != 'cuda']:
            a = np.concatenate(X['cuda']); b = np.concatenate(X[ref])
            ks = sst.ks_2samp(a, b)
            Hc, Hr = H['cuda'], H[ref]
            allH = np.vstack([Hc, Hr]); n1 = len(Hc)

            def Dgrid(h1, h2):
                c1 = np.cumsum(h1) / h1.sum(); c2 = np.cumsum(h2) / h2.sum()
                return float(np.max(np.abs(c1 - c2)))
            D0 = Dgrid(Hc.sum(0), Hr.sum(0))
            rng = np.random.default_rng(graine)
            Dp = np.empty(B)
            for i in range(B):
                p = rng.permutation(len(allH))
                Dp[i] = Dgrid(allH[p[:n1]].sum(0), allH[p[n1:]].sum(0))
            pperm = float((1 + np.sum(Dp >= D0)) / (B + 1))
            rL[ref] = dict(n_cfg_cuda=len(Hc), n_cfg_ref=len(Hr), n_plaq_cuda=int(a.size), n_plaq_ref=int(b.size),
                           D_exact=float(ks.statistic), p_asympt_naive=float(ks.pvalue), D_grille=D0, p_perm=pperm,
                           D_perm_q50=float(np.median(Dp)), D_perm_q99=float(np.quantile(Dp, 0.99)),
                           moy_cuda=float(a.mean()), moy_ref=float(b.mean()), sd_cuda=float(a.std()), sd_ref=float(b.std()),
                           ok=bool(pperm > 0.01) if ref == 'amd_neuve' else None)
            print(f"P5 L={L} cuda vs {ref}: D={ks.statistic:.2e} p_naif={ks.pvalue:.3g} D_grille={D0:.2e} "
                  f"p_perm={pperm:.3f} (q50 D_perm={np.median(Dp):.2e})", flush=True)
        res[f'L{L}'] = rL
    res['P5_ok'] = all(res[f'L{L}']['amd_neuve']['ok'] for L in (6, 8))
    return res


def p6():
    """Intégrité des dumps CUDA : forme, unitarité, det, plaquette numpy = CSV."""
    mx = dict(unit=0.0, det=0.0, plaq=0.0); n = 0; forme = True
    for d in sorted(glob.glob(f'{CUDA}/L*_b*_c*')):
        L = int(json.load(open(os.path.join(d, 'manifest.json')))['l'])
        meas = {}
        for line in open(os.path.join(d, 'measurements.csv')).read().split('\n')[1:]:
            if line:
                t = line.split(','); meas[int(t[0])] = float(t[3])
        for f in cfgs(d):
            U = load_U(f); n += 1
            forme &= U.shape == (4, L, L, L, L, 2, 2) and U.dtype == np.complex128
            I = U @ dag(U) - np.eye(2)
            mx['unit'] = max(mx['unit'], float(np.abs(I).max()))
            mx['det'] = max(mx['det'], float(np.abs(np.linalg.det(U) - 1).max()))
            idx = int(re.search(r'cfg_(\d+)', f).group(1))
            mx['plaq'] = max(mx['plaq'], abs(float(plaq_locales(U).mean()) - meas[idx]))
    ok = forme and mx['unit'] < 1e-13 and mx['det'] < 1e-13 and mx['plaq'] < 1e-12
    print('P6', n, 'dumps', mx, 'forme', forme, 'ok', ok, flush=True)
    return dict(n_dumps=n, forme_ok=bool(forme), max_dev=mx, P6_ok=bool(ok))


def p2():
    res = {}
    for L in (6, 8):
        a = os.path.join(ROOT, 'ref_amd', 'p2', f'L{L}'); c = os.path.join(ROOT, 'cuda', 'p2', f'L{L}')
        dev = []
        for f in cfgs(a):
            g = os.path.join(c, os.path.basename(f))
            dev.append(float(np.abs(load_U(f) - load_U(g)).max()))
        rd = lambda d: [float(l.split(',')[3]) for l in open(os.path.join(d, 'measurements.csv')).read().split('\n')[1:] if l]
        pa, pc = rd(a), rd(c)
        res[f'L{L}'] = dict(max_dU_initial=dev[0], max_dU_par_pas=dev, dplaq_par_pas=[x - y for x, y in zip(pc, pa)],
                            P2_ok=bool(dev[0] < 1e-13))
        print(f'P2 L={L} initial {dev[0]:.2e} ; pas 1..20 : ' + ' '.join(f'{x:.1e}' for x in dev[1:]), flush=True)
    res['P2_ok'] = all(res[f'L{L}']['P2_ok'] for L in (6, 8))
    return res


def p8():
    pat = re.compile(r'(\w+)=([-\d.e+]+)')
    res = {}
    for L in (6, 8, 12, 16, 24, 32):
        r = {}
        for nom, f in (('cuda', f'{ROOT}/cuda/logs/bench_L{L}.log'), ('amd_26_09', f'{SRC}/bench/gpu_L{L}.log'),
                       ('amd_27_09', f'{ROOT}/logs/amd_bench_L{L}.log')):
            if os.path.exists(f):
                ln = [l for l in open(f) if l.startswith('BENCH')]
                if ln:
                    r[nom] = {k: float(v) for k, v in pat.findall(ln[0])}
        if 'cuda' in r:
            c = r['cuda']
            s_pas = lambda x: x['hb_s_par_sweep'] + 4 * x['or_s_par_sweep']
            r['s_par_pas_cuda'] = s_pas(c)
            for ref in ('amd_26_09', 'amd_27_09'):
                if ref in r:
                    r[f'gain_hb_vs_{ref}'] = r[ref]['hb_s_par_sweep'] / c['hb_s_par_sweep']
                    r[f'gain_or_vs_{ref}'] = r[ref]['or_s_par_sweep'] / c['or_s_par_sweep']
                    r[f'gain_pas_vs_{ref}'] = s_pas(r[ref]) / s_pas(c)
        res[f'L{L}'] = r
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sans-ks', action='store_true'); ap.add_argument('--partiel', action='store_true'); a = ap.parse_args()
    R = dict(P3_P4=p3_p4(), P8=p8())
    if a.partiel:
        json.dump(R, open(os.path.join(ROOT, 'analyse', 'resultats_partiels_cuda.json'), 'w'), indent=1); sys.exit(0)
    R['P2'] = p2(); R['P6'] = p6()
    if not a.sans_ks:
        R['P5'] = p5()
    os.makedirs(os.path.join(ROOT, 'analyse'), exist_ok=True)
    json.dump(R, open(os.path.join(ROOT, 'analyse', 'resultats_gpu1.json'), 'w'), indent=1)
    print('P2', R['P2']['P2_ok'], 'P3', R['P3_P4']['P3_ok'], 'chi2', round(R['P3_P4']['chi2'], 2), 'ndf', R['P3_P4']['ndf'],
          'p', round(R['P3_P4']['p_chi2'], 3), 'P4cuda', R['P3_P4']['P4_cuda_ok'], 'P6', R['P6']['P6_ok'],
          'P5', R.get('P5', {}).get('P5_ok'))
