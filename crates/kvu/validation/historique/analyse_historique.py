#!/usr/bin/env python3
"""Analyses P2-P6 de gates_gpu1 (gpu_port/src/analyse_gpu1.py) rejouées sur les ensembles du kevinotron
UNIFIÉ. Tourne SUR ACEMAGIC (numpy + scipy), sur les dumps rapatriés par `kevinotron fetch`.

Gates (ARCHITECTURE_UNIFIEE.md §6.6) :
  p2     G-H08  départs chauds HIP et CUDA de mêmes graines : max|ΔU| < 1e-13 sur la configuration initiale
  p3     G-H09  plaquette CUDA unifié contre les références AMD, 6 cellules : |z| < 3 et χ² p > 0,01
  p4cuda G-H10  critères C1-C4 de PREREG_CRITERES sur les runs CUDA unifiés
  p4hip  G-H11  critères C1-C4 sur les runs HIP unifiés
  p5     G-H12  KS par permutation, plaquettes locales, β = 2,8, L = 6 et 8 : p > 0,01
  p6     G-H13  intégrité des dumps : |UU† − 1| ≤ 1e-13, |det − 1| ≤ 1e-13, plaquette numpy = CSV ≤ 1e-12

Les SEUILS sont ceux de PREREG_CRITERES.md et de gates_gpu1.sh, GELÉS ; ils ne sont pas ajustés ici.
Les estimateurs (stat, combine) viennent de la COPIE sha256-vérifiée de analyse_plaquette.py.
Sortie standard : une ligne `PASS|FAIL <gate> <détail>` ; le détail complet va sur stderr et dans --json.
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

import numpy as np
from scipy import stats as sst

sys.dont_write_bytecode = True

GPU_PORT = os.environ.get('KVU_GPU_PORT', '/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port')
SRC = os.environ.get('KVU_WILSON_GPU', '/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926')
AMDN = f'{GPU_PORT}/ref_amd/ens'
SHA_AP = 'ec5fef4837591443b4f8b4c52fa42ef956c6971f1ec12f585ddd89989baede45'
BETAS = ['2.3', '2.8', '3.5']
SEUIL_Z, SEUIL_P, SEUIL_UNIT, SEUIL_PLAQ, SEUIL_P2 = 3.0, 0.01, 1e-13, 1e-12, 1e-13


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def charger_ap():
    """analyse_plaquette.py de la copie gpu_port (sha256 vérifié avant import)."""
    chemin = os.path.join(GPU_PORT, 'src_origine', 'analyse_plaquette.py')
    h = hashlib.sha256(open(chemin, 'rb').read()).hexdigest()
    if h != SHA_AP:
        raise SystemExit(f'analyse_plaquette.py : sha256 {h} != {SHA_AP} (copie altérée)')
    spec = importlib.util.spec_from_file_location('analyse_plaquette', chemin)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AP = charger_ap()


# ---------------------------------------------------------------------------------------------
# Lecture tolérante du manifeste (su2_hip d'origine OU manifeste unifié kevinotron-unifie/1)
# ---------------------------------------------------------------------------------------------
def aplatir(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(aplatir(v, pre))
        else:
            out.setdefault(k, v)
    return out


def manifeste_normalise(d):
    """Paramètres de la chaîne du répertoire `d`. Trois dispositions sont lues :
    - su2_hip / su2_cuda d'origine : manifest.json dans le répertoire de la chaîne (clés l, beta, seed, start…) ;
    - unifié, une chaîne : manifest.json dans `d`, avec `config_run.lot.chaines[0]` ;
    - unifié, lot de K chaînes : manifest.json dans le répertoire du LOT (le parent), chaîne = sous-répertoire cNNN."""
    man_dir, idx = d, 0
    if not os.path.exists(os.path.join(d, 'manifest.json')):
        m_idx = re.fullmatch(r'c(\d+)', os.path.basename(d.rstrip('/')))
        parent = os.path.dirname(d.rstrip('/'))
        if not m_idx or not os.path.exists(os.path.join(parent, 'manifest.json')):
            raise KeyError(f'{d} : ni manifest.json, ni lot parent')
        man_dir, idx = parent, int(m_idx.group(1))
    brut = json.load(open(os.path.join(man_dir, 'manifest.json')))
    cr = brut.get('config_run')
    if cr:   # manifeste unifié (kevinotron-unifie/1)
        ch = cr['lot']['chaines'][idx]
        return dict(l=int(cr['lot']['l']), beta=float(ch['beta']), seed=int(ch['graine']),
                    start='cold' if str(ch['depart']).lower() in ('froid', 'cold') else 'hot',
                    therm_steps=int(cr['therm']), statut=brut.get('statut', 'COMPLETE'), dump_every=int(cr.get('dump_every', 0) or 0))
    m = aplatir(brut)

    def pick(*cles):
        for c in cles:
            if c in m and m[c] is not None:
                return m[c]
        raise KeyError(f"{man_dir}/manifest.json : aucune des clés {cles}")

    start = str(pick('start', 'depart', 'départ')).lower()
    return dict(l=int(pick('l', 'L', 'ls')), beta=float(pick('beta', 'bêta')), seed=int(pick('seed', 'graine')),
                start='cold' if 'cold' in start or 'froid' in start else 'hot',
                therm_steps=int(pick('therm_steps', 'therm')), statut=m.get('statut', 'COMPLETE'),
                dump_every=int(m.get('dump_every', 0) or 0))


def read_gpu(d):
    """Même contrat que analyse_plaquette.read_gpu, manifeste normalisé."""
    rows = [r for r in csv.DictReader(open(os.path.join(d, 'measurements.csv'))) if r['phase'] == 'sample']
    P = np.array([float(r['plaquette']) for r in rows])
    pol = np.array([[float(r[f'P{m}']) for m in range(4)] for r in rows])
    th = list(csv.DictReader(open(os.path.join(d, 'therm.csv'))))
    thP = np.array([float(r['plaquette']) for r in th])
    thpol = np.array([[float(r[f'P{m}']) for m in range(4)] for r in th])
    return P, pol, thP, thpol, manifeste_normalise(d)


AP.read_gpu = read_gpu


def chaines_dans(racine):
    """Tous les répertoires de chaîne sous `racine` : ceux qui contiennent measurements.csv."""
    out = []
    for dp, _, fn in os.walk(racine):
        if 'measurements.csv' in fn and 'therm.csv' in fn:
            out.append(dp)
    return sorted(out)


def est_complete(d):
    """COMPLETE dans le répertoire de la chaîne (une chaîne) ou dans celui du lot (parent)."""
    return os.path.exists(os.path.join(d, 'COMPLETE')) or os.path.exists(os.path.join(os.path.dirname(d.rstrip('/')), 'COMPLETE'))


def exiger_complete(ds):
    for d in ds:
        if not est_complete(d):
            raise SystemExit(f'INCOMPLETE {d}')


def regrouper(ds):
    """{(L, 'β'): [dossiers]} ; dans une cellule, chaudes triées par graine = c1, c2 ; froide = c3."""
    cel = {}
    for d in ds:
        m = manifeste_normalise(d)
        cel.setdefault((m['l'], f"{m['beta']:.1f}"), []).append((m['start'] != 'cold', m['seed'], d))
    out = {}
    for k, v in cel.items():
        chaudes = sorted((s, d) for h, s, d in v if h)
        froides = sorted((s, d) for h, s, d in v if not h)
        noms = [f'L{k[0]}_b{k[1]}_c{i + 1}' for i in range(len(chaudes) + len(froides))]
        out[k] = dict(zip(noms, [d for _, d in chaudes] + [d for _, d in froides]))
    return out


def analyser_cellule(chaines):
    """C1-C4 + combinaison pour {nom: dossier} (c1, c2 chaudes ; c3 froide), comme analyse_gpu1.cellule."""
    chs = [(n, AP.analyse_chaine(d)) for n, d in sorted(chaines.items())]
    d = dict(chs)
    k = {n.split('_')[-1]: v for n, v in d.items()}
    m, e = AP.combine(list(d.values()))
    res = dict(moy=m, err=e, chaines={n: dict(moy=v['plaquette']['moy'], err=v['plaquette']['err'], tau=v['plaquette']['tau'],
                                               z_moities=v['z_moities'], C1_ok=v['C1_ok'], start=v['start'],
                                               tau_polyakov_max=v['tau_polyakov_max']) for n, v in d.items()})
    res['C1_ok'] = all(v['C1_ok'] for v in d.values())
    if all(c in k for c in ('c1', 'c2', 'c3')):
        mh, eh = AP.combine([k['c1'], k['c2']])
        z2 = (k['c3']['plaquette']['moy'] - mh) / np.hypot(k['c3']['plaquette']['err'], eh)
        z3 = (k['c1']['plaquette']['moy'] - k['c2']['plaquette']['moy']) / np.hypot(
            k['c1']['plaquette']['err'], k['c2']['plaquette']['err'])
        res.update(z_C2=float(z2), C2_ok=bool(abs(z2) < SEUIL_Z), z_C3=float(z3), C3_ok=bool(abs(z3) < SEUIL_Z))
    else:
        res.update(C2_ok=False, C3_ok=False, manque='il faut c1, c2 (chaudes) et c3 (froide)')
    return res


def refs_amd(L, b):
    """Références AMD par cellule (identiques à analyse_gpu1.refs_amd)."""
    R = {}
    if b == '2.3':
        R['validation_G2'] = sorted(glob.glob(f'{SRC}/validation/validation/L{L}_b2.3_c*'))
    if L == 6 and b == '2.8':
        R['production_L6'] = sorted(glob.glob(f'{SRC}/ensembles/production_L6/L6_b2.8_c*'))
    n = sorted(glob.glob(f'{AMDN}/L{L}_b{b}_c*'))
    if n:
        R['neuve'] = n
    return R


def chaines_nommees(ds):
    exiger_complete(ds)
    return [(os.path.basename(d), AP.analyse_chaine(d)) for d in ds]


def cellule_amd(ds):
    exiger_complete(ds)
    return analyser_cellule({os.path.basename(d): d for d in ds})


def p4(racine):
    """C1-C4 de chaque cellule d'un ensemble unifié (gates G-H10, G-H11)."""
    ds = chaines_dans(racine)
    if not ds:
        raise SystemExit(f'aucune chaîne sous {racine}')
    exiger_complete(ds)
    cel = regrouper(ds)
    attendu = {(L, b) for L in (6, 8) for b in BETAS}
    manquantes = sorted(attendu - set(cel))
    res, ko = {}, []
    for (L, b), chs in sorted(cel.items()):
        c = analyser_cellule(chs)
        c['C4_ok'] = all(1000 >= 5 * v['tau'] for v in c['chaines'].values())
        res[f'L{L}_b{b}'] = c
        bad = [x for x in ('C1_ok', 'C2_ok', 'C3_ok', 'C4_ok') if not c.get(x, False)]
        log(f"L{L} b{b}: moy {c['moy']:.7f}({c['err'] * 1e7:.0f}) C1 {c['C1_ok']} C2 {c.get('z_C2', float('nan')):+.2f} "
            f"C3 {c.get('z_C3', float('nan')):+.2f} C4 {c['C4_ok']}")
        if bad:
            ko.append(f'L{L}b{b}:{"/".join(bad)}')
    ok = not manquantes and not ko and len(res) == 6
    detail = f"{len(ds)} chaînes, {len(res)} cellules ; C1-C4 " + ('tous verts' if ok else
             'ROUGES : ' + ', '.join(ko + [f'cellule absente L{L}b{b}' for L, b in manquantes]))
    return ok, detail, res


def p3(racine_cuda):
    """Plaquette CUDA unifiée contre les références AMD (gate G-H09)."""
    ds = chaines_dans(racine_cuda)
    exiger_complete(ds)
    cel = regrouper(ds)
    out = {}
    for L in (6, 8):
        for b in BETAS:
            chs = cel.get((L, b))
            if not chs:
                out[f'L{L}_b{b}'] = dict(erreur='cellule absente')
                continue
            cu = analyser_cellule(chs)
            r = dict(cuda=cu, z={})
            allamd = []
            for nom, dd in refs_amd(L, b).items():
                chs_a = chaines_nommees(dd)
                allamd += [v for _, v in chs_a]
                a = cellule_amd(dd)
                z = (cu['moy'] - a['moy']) / np.hypot(cu['err'], a['err'])
                r['z'][nom] = dict(z=float(z), ok=bool(abs(z) < SEUIL_Z))
            if allamd:
                m, e = AP.combine(allamd)
                r['z_chi2'] = float((cu['moy'] - m) / np.hypot(cu['err'], e))
            out[f'L{L}_b{b}'] = r
            log(f"L{L} b{b}: CUDA {cu['moy']:.7f}({cu['err'] * 1e7:.0f}) " +
                ' '.join(f'{n} z={x["z"]:+.2f}' for n, x in r['z'].items()))
    cellules = [v for v in out.values() if 'z_chi2' in v]
    zs = np.array([v['z_chi2'] for v in cellules])
    chi2 = float((zs ** 2).sum()) if len(zs) else float('nan')
    p = float(sst.chi2.sf(chi2, len(zs))) if len(zs) else float('nan')
    z_ok = all(all(x['ok'] for x in v['z'].values()) for v in cellules)
    ok = len(cellules) == 6 and z_ok and p > SEUIL_P
    detail = f'{len(cellules)} cellules ; max|z|={np.abs(zs).max() if len(zs) else float("nan"):.2f} ; chi2={chi2:.2f}/{len(zs)} p={p:.3f}'
    return ok, detail, dict(cellules=out, chi2=chi2, ndf=len(zs), p_chi2=p)


# ---------------------------------------------------------------------------------------------
# Plaquettes locales (P5, P6)
# ---------------------------------------------------------------------------------------------
def load_U(f):
    return np.load(f)  # (4, L, L, L, L, 2, 2) complex128


def dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def plaq_locales(U):
    out = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            A = U[mu] @ np.roll(U[nu], -1, axis=mu)
            B = U[nu] @ np.roll(U[mu], -1, axis=nu)
            out.append(0.5 * np.real(np.einsum('...ij,...ji->...', A, dag(B))))
    return np.stack(out).ravel()


def cfgs(d):
    return sorted(glob.glob(os.path.join(d, 'cfg_*.npy')))


def p5(racine_cuda, B=2000, graine=27092026, nb=20000):
    """KS par permutation (par configuration) des plaquettes locales à β = 2,8 (gate G-H12)."""
    cel = regrouper(chaines_dans(racine_cuda))
    res, ok_all = {}, True
    for L in (6, 8):
        chs = cel.get((L, '2.8'))
        if not chs:
            return False, f'cellule L{L} b2.8 absente', {}
        cu_files = [c for d in chs.values() for c in cfgs(d)]
        ref_files = [c for d in sorted(glob.glob(f'{AMDN}/L{L}_b2.8_c*')) for c in cfgs(d)]
        edges = np.linspace(-1, 1, nb + 1)
        X = {'cuda': [plaq_locales(load_U(f)) for f in cu_files], 'ref': [plaq_locales(load_U(f)) for f in ref_files]}
        H = {k: np.array([np.histogram(x, edges)[0] for x in v], dtype=np.int64) for k, v in X.items()}
        a, b = np.concatenate(X['cuda']), np.concatenate(X['ref'])
        ks = sst.ks_2samp(a, b)
        Hc, Hr = H['cuda'], H['ref']
        allH = np.vstack([Hc, Hr])
        n1 = len(Hc)

        def Dgrid(h1, h2):
            return float(np.max(np.abs(np.cumsum(h1) / h1.sum() - np.cumsum(h2) / h2.sum())))
        D0 = Dgrid(Hc.sum(0), Hr.sum(0))
        rng = np.random.default_rng(graine)
        Dp = np.empty(B)
        for i in range(B):
            p = rng.permutation(len(allH))
            Dp[i] = Dgrid(allH[p[:n1]].sum(0), allH[p[n1:]].sum(0))
        pperm = float((1 + np.sum(Dp >= D0)) / (B + 1))
        ok = pperm > SEUIL_P
        ok_all &= ok
        res[f'L{L}'] = dict(n_cfg_cuda=len(Hc), n_cfg_ref=len(Hr), D_exact=float(ks.statistic), D_grille=D0, p_perm=pperm, ok=bool(ok))
        log(f'P5 L={L}: D={ks.statistic:.2e} D_grille={D0:.2e} p_perm={pperm:.3f} ({len(Hc)} + {len(Hr)} configs)')
    return ok_all, ' ; '.join(f"L{L} p_perm={res[f'L{L}']['p_perm']:.3f}" for L in (6, 8)), res


def p6(racines):
    """Intégrité des dumps (gate G-H13) : forme, unitarité, déterminant, plaquette numpy = CSV."""
    mx = dict(unit=0.0, det=0.0, plaq=0.0)
    n, forme = 0, True
    for racine in racines:
        for d in chaines_dans(racine):
            man = manifeste_normalise(d)
            L = man['l']
            meas = {}
            for line in open(os.path.join(d, 'measurements.csv')).read().split('\n')[1:]:
                if line:
                    t = line.split(',')
                    meas[int(t[0])] = float(t[3])
            for f in cfgs(d):
                U = load_U(f)
                n += 1
                forme &= U.shape == (4, L, L, L, L, 2, 2) and U.dtype == np.complex128
                mx['unit'] = max(mx['unit'], float(np.abs(U @ dag(U) - np.eye(2)).max()))
                mx['det'] = max(mx['det'], float(np.abs(np.linalg.det(U) - 1).max()))
                idx = int(re.search(r'cfg_(\d+)', f).group(1))
                if idx in meas:
                    mx['plaq'] = max(mx['plaq'], abs(float(plaq_locales(U).mean()) - meas[idx]))
    ok = n > 0 and forme and mx['unit'] <= SEUIL_UNIT and mx['det'] <= SEUIL_UNIT and mx['plaq'] <= SEUIL_PLAQ
    return ok, f"{n} dumps ; max|UU†-1|={mx['unit']:.2e} max|det-1|={mx['det']:.2e} max|plaq-csv|={mx['plaq']:.2e}", dict(n=n, **mx)


def dossiers_par_L(racine):
    """{L: dossier de chaîne} sous `racine` (une seule chaîne par L attendue : les runs P2 sont mono-chaîne)."""
    out = {}
    for d in chaines_dans(racine):
        out.setdefault(manifeste_normalise(d)['l'], d)
    return out


def p2(dir_hip, dir_cuda):
    """Départs chauds HIP et CUDA de mêmes graines (gate G-H08)."""
    res, ok_all = {}, True
    dh, dc = dossiers_par_L(dir_hip), dossiers_par_L(dir_cuda)
    for L in (6, 8):
        if L not in dh or L not in dc:
            return False, f'L{L} : run absent (HIP {sorted(dh)}, CUDA {sorted(dc)})', {}
        fa, fc = cfgs(dh[L]), cfgs(dc[L])
        if not fa or len(fa) != len(fc):
            return False, f'L{L} : {len(fa)} dumps HIP contre {len(fc)} CUDA', {}
        dev = [float(np.abs(load_U(x) - load_U(os.path.join(dc[L], os.path.basename(x)))).max()) for x in fa]
        ok = dev[0] < SEUIL_P2
        ok_all &= ok
        res[f'L{L}'] = dict(max_dU_initial=dev[0], max_dU_par_pas=dev, ok=bool(ok))
        log(f'P2 L={L} initial {dev[0]:.2e} ; pas 1..{len(dev) - 1} : ' + ' '.join(f'{x:.1e}' for x in dev[1:]))
    return ok_all, ' ; '.join(f"L{L} initial {res[f'L{L}']['max_dU_initial']:.2e}" for L in (6, 8)), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gate', choices=['p2', 'p3', 'p4cuda', 'p4hip', 'p5', 'p6'])
    ap.add_argument('--cuda', help='racine des chaînes CUDA unifiées (gh06 rapatrié)')
    ap.add_argument('--hip', help='racine des chaînes HIP unifiées (gh07)')
    ap.add_argument('--p2-hip'), ap.add_argument('--p2-cuda')
    ap.add_argument('--json', help='écrit le détail complet')
    a = ap.parse_args()
    nom = dict(p2='G-H08', p3='G-H09', p4cuda='G-H10', p4hip='G-H11', p5='G-H12', p6='G-H13')[a.gate]
    try:
        if a.gate == 'p2':
            ok, detail, res = p2(a.p2_hip, a.p2_cuda)
        elif a.gate == 'p3':
            ok, detail, res = p3(a.cuda)
        elif a.gate == 'p4cuda':
            ok, detail, res = p4(a.cuda)
        elif a.gate == 'p4hip':
            ok, detail, res = p4(a.hip)
        elif a.gate == 'p5':
            ok, detail, res = p5(a.cuda)
        else:
            ok, detail, res = p6([x for x in (a.cuda, a.hip) if x])
    except SystemExit as e:
        ok, detail, res = False, str(e.code), {}
    if a.json:
        json.dump(dict(gate=nom, ok=bool(ok), detail=detail, resultats=res), open(a.json, 'w'), indent=1, default=float)
    print(('PASS ' if ok else 'FAIL ') + nom + ' ' + detail)


if __name__ == '__main__':
    main()
