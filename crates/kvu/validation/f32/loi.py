#!/usr/bin/env python3
"""Bibliothèque de jugement du chemin FP32 (tâche T4) : gates LOI-0 à LOI-4 et LOI-τ.

SEUILS GELÉS (ARCHITECTURE_UNIFIEE.md §6.2) : ils sont écrits ici AVANT toute donnée du lot et ne se
règlent jamais. Un écart se rapporte tel quel, avec les nombres. Les choix de protocole que §6.2 ne
fixe pas (définitions de σ pour LOI-τ, amincissement des séries de Polyakov, nombre de permutations de
LOI-3) sont eux aussi figés dans ce fichier avant le premier run.

Références FP64 (lecture seule, jamais copiées) :
  - gpu_port/ref_amd/ens : su2_hip de production, β ∈ {2,8 ; 3,5}, L ∈ {6, 8}, 3 chaînes par cellule ;
  - wilson_gpu_grandL/validation/validation : su2_hip, β = 2,3, L ∈ {6, 8}, 3 chaînes (même protocole :
    therm 2000, nmeas 20000, skip 1, nor 4, deux départs chauds + un froid), pour la cellule β = 2,3 que
    ref_amd/ens ne contient pas (comme le fait analyse_gpu1.py, refs_amd) ;
  - estimateurs : stat(), tau_int() (Sokal c = 6) et combine() de analyse_plaquette.py du chantier
    wilson_gpu_grandL_20260926, importé en lecture (aucun fichier .pyc écrit).
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sst

sys.dont_write_bytecode = True

CHANTIER = '/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926'
GPU_PORT = '/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port'
REF_ENS = os.path.join(GPU_PORT, 'ref_amd', 'ens')
REF_G2 = os.path.join(CHANTIER, 'validation', 'validation')

# ---------------------------------------------------------------------------------------------
# Seuils gelés
# ---------------------------------------------------------------------------------------------
SEUILS = dict(
    loi0_defaut_norme=2e-6,          # max | |q| - 1 | après 10^4 pas, L = 8, β = 2,3 ; plafond = 0
    loi1_z=4.0,                      # |z| < 4 pour m1 et m2 à chacun des 7 α ; plafond 0
    loi1_ks_p=1e-4,                  # KS contre la CDF exacte, 4e5 tirages : p > 1e-4
    loi1_alphas=[0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0],
    loi1_n=20_000_000,
    loi1_nks=400_000,
    loi2_z=3.0,                      # 6 cellules : |z| < 3 par cellule
    loi2_chi2_p=0.01,                # χ²(6) : p > 0,01
    loi2_l16_z=4.0,                  # L = 16, β ∈ {2,5 ; 16} : |z| < 4
    loi3_p=0.01,                     # KS par permutation (P5) et KS de |P| (Polyakov) : p > 0,01
    loi3_permutations=2000,          # protocole P5 de analyse_gpu1.py
    loi3_graine=27092026,
    loi3_bins=20000,
    loi4a_z=3.0,                     # β x 1,002 : ROUGE si |z| > 3 dans >= 4 cellules sur 6 ...
    loi4a_ncellules=4,
    loi4a_chi2_p=1e-3,               # ... ou p(χ²) < 1e-3
    loi4b_alpha_min=5.0,             # mutant metro : LOI-1 ROUGE à chaque α >= 5
    loitau_rel=0.2,                  # |Δτ| / τ <= 0,2 + 2σ
    loitau_nsigma=2.0,
)
CELLULES = [(L, b) for L in (6, 8) for b in ('2.3', '2.8', '3.5')]
FACTEUR_MUTANT_BETA = 1.002


def charger_ap():
    """analyse_plaquette.py du chantier, importé en lecture seule."""
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if CHANTIER not in sys.path:
        sys.path.insert(0, CHANTIER)
    import analyse_plaquette as ap  # noqa: E402
    return ap


# ---------------------------------------------------------------------------------------------
# LOI-1 : loi exacte du tirage de x0
# ---------------------------------------------------------------------------------------------
def moments_exacts(alpha, n=4000):
    """Quadrature de Gauss-Tchebychev de 2e espèce à n nœuds (comme test_heatbath_cuda.py) : E[x^p], p = 1..4."""
    k = np.arange(1, n + 1)
    x = np.cos(k * np.pi / (n + 1))
    w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1)) ** 2
    f = np.exp(alpha * (x - 1))
    z = np.sum(w * f)
    return [float(np.sum(w * f * x ** p) / z) for p in (1, 2, 3, 4)]


def cdf_exacte(alpha, ngrille=2_000_001):
    """CDF de p(x) ∝ sqrt(1-x²) exp(α x) sur [-1,1] : x = cos θ, ∫ sin²θ exp(α cos θ) dθ (trapèzes fins)."""
    th = np.linspace(0.0, np.pi, ngrille)
    f = np.sin(th) ** 2 * np.exp(alpha * (np.cos(th) - 1.0))
    c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(th))])
    z = c[-1]
    xs = np.cos(th)[::-1]            # croissant
    fs = (z - c)[::-1] / z           # F(x) = P(X <= x) = ∫_{θ_x}^{π} / Z
    return lambda v: np.interp(v, xs, fs)


def juge_loi1_alpha(ligne, dossier):
    """Juge un α : renvoie un dict (z1, z2, ks_p, plafond, ok)."""
    a = ligne['alpha']                       # α effectivement utilisé (arrondi float promu en double)
    n = ligne['n']
    e1, e2, _, e4 = moments_exacts(a)
    sd1 = np.sqrt(e2 - e1 ** 2)
    sd2 = np.sqrt(e4 - e2 ** 2)
    z1 = (ligne['m1'] - e1) / (sd1 / np.sqrt(n))
    z2 = (ligne['m2'] - e2) / (sd2 / np.sqrt(n))
    tir = np.fromfile(os.path.join(dossier, ligne['ks_fichier']), dtype='<f4').astype(np.float64)
    ks = sst.kstest(tir, cdf_exacte(a))
    ok = (abs(z1) < SEUILS['loi1_z'] and abs(z2) < SEUILS['loi1_z'] and ligne['plafond'] == 0
          and ks.pvalue > SEUILS['loi1_ks_p'])
    return dict(alpha_demande=ligne.get('alpha_demande'), alpha=a, n=n, z1=float(z1), z2=float(z2),
                m1=ligne['m1'], e1=e1, m2=ligne['m2'], e2=e2, ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                n_ks=int(tir.size), plafond=int(ligne['plafond']), minimum=ligne.get('min'),
                maximum=ligne.get('max'), ok=bool(ok))


def juge_loi1(dossier):
    lignes = [json.loads(l) for l in open(os.path.join(dossier, 'res.jsonl')) if l.strip()]
    return [juge_loi1_alpha(l, dossier) for l in lignes]


# ---------------------------------------------------------------------------------------------
# LOI-2 : plaquette moyenne contre FP64
# ---------------------------------------------------------------------------------------------
def dossiers_chaines(base, L, b):
    d = sorted(glob.glob(os.path.join(base, f'L{L}_b{b}_c*')))
    for x in d:
        if not os.path.exists(os.path.join(x, 'COMPLETE')):
            raise SystemExit(f'INCOMPLET {x}')
    return d


def reference_fp64(L, b):
    """Dossiers FP64 de la cellule (L, b) : voir l'en-tête."""
    base = REF_G2 if b == '2.3' else REF_ENS
    return dossiers_chaines(base, L, b)


def analyse(ap, dossiers):
    return [ap.analyse_chaine(d) for d in dossiers]


def comparer(ap, a32, a64):
    """z de la plaquette combinée F32 contre FP64 (erreurs de Sokal)."""
    m32, e32 = ap.combine(a32)
    m64, e64 = ap.combine(a64)
    z = (m32 - m64) / np.hypot(e32, e64)
    return dict(m32=m32, e32=e32, m64=m64, e64=e64, z=float(z), delta=float(m32 - m64),
                n32=len(a32), n64=len(a64))


def juge_loi2(ap, base32, refs_l16=None):
    """6 cellules (L ∈ {6,8} × β ∈ {2,3 ; 2,8 ; 3,5}) + χ²(6) ; L = 16 si `refs_l16` (dossier FP64 unifié)."""
    cel, brut32, brut64 = {}, {}, {}
    for L, b in CELLULES:
        a32 = analyse(ap, dossiers_chaines(base32, L, b))
        a64 = analyse(ap, reference_fp64(L, b))
        brut32[f'L{L}_b{b}'], brut64[f'L{L}_b{b}'] = a32, a64
        cel[f'L{L}_b{b}'] = dict(comparer(ap, a32, a64), tau32=[c['plaquette']['tau'] for c in a32],
                                 tau64=[c['plaquette']['tau'] for c in a64])
    zs = np.array([c['z'] for c in cel.values()])
    chi2 = float((zs ** 2).sum())
    p = float(sst.chi2.sf(chi2, len(zs)))
    ok6 = bool(all(abs(c['z']) < SEUILS['loi2_z'] for c in cel.values()) and p > SEUILS['loi2_chi2_p'])
    res = dict(cellules=cel, chi2=chi2, ndl=len(zs), p_chi2=p, ok_6cellules=ok6, _a32=brut32, _a64=brut64)
    if refs_l16 is not None:
        l16 = {}
        for b in ('2.5', '16'):
            d32 = dossiers_chaines(base32, 16, b)
            d64 = dossiers_chaines(refs_l16, 16, b)
            l16[f'L16_b{b}'] = comparer(ap, analyse(ap, d32), analyse(ap, d64))
        res['L16'] = l16
        res['ok_L16'] = bool(all(abs(c['z']) < SEUILS['loi2_l16_z'] for c in l16.values()))
    return res


def juge_loi4a(res_mutant):
    """Le mutant β x 1,002 doit être ROUGE : |z| > 3 dans >= 4 cellules sur 6, ou p(χ²) < 1e-3."""
    n_rouges = sum(abs(c['z']) > SEUILS['loi4a_z'] for c in res_mutant['cellules'].values())
    rouge = bool(n_rouges >= SEUILS['loi4a_ncellules'] or res_mutant['p_chi2'] < SEUILS['loi4a_chi2_p'])
    return dict(n_cellules_rouges=int(n_rouges), p_chi2=res_mutant['p_chi2'], rouge=rouge)


def juge_loi4b(res_mutant_loi1):
    """Le mutant metro doit être ROUGE (échec de LOI-1) à chaque α >= 5."""
    cibles = [r for r in res_mutant_loi1 if r['alpha'] >= SEUILS['loi4b_alpha_min'] - 1e-6]
    return dict(alphas=[r['alpha'] for r in cibles], rouges=[not r['ok'] for r in cibles],
                rouge=bool(cibles and all(not r['ok'] for r in cibles)))


# ---------------------------------------------------------------------------------------------
# LOI-τ : temps d'autocorrélation intégré
# ---------------------------------------------------------------------------------------------
def juge_loitau(res_loi2, ap, cle='L8_b2.3'):
    """τ_int(P) F32 contre FP64 à L = 8, β = 2,3.

    τ = moyenne des τ_int des chaînes ; σ(τ) = sqrt(Σ δτ_i²)/n avec δτ de Sokal (analyse_plaquette.tau_int).
    Critère gelé : |τ32 − τ64| / τ64 <= 0,2 + 2 σ_rel, σ_rel = hypot(σ32, σ64) / τ64.
    """
    a32 = res_loi2['_a32'][cle]
    a64 = res_loi2['_a64'][cle]
    t32 = np.array([c['plaquette']['tau'] for c in a32])
    t64 = np.array([c['plaquette']['tau'] for c in a64])
    d32 = np.array([c['plaquette']['dtau'] for c in a32])
    d64 = np.array([c['plaquette']['dtau'] for c in a64])
    tau32, tau64 = float(t32.mean()), float(t64.mean())
    s32 = float(np.sqrt((d32 ** 2).sum()) / len(d32))
    s64 = float(np.sqrt((d64 ** 2).sum()) / len(d64))
    sigma_rel = float(np.hypot(s32, s64) / tau64)
    ecart = float(abs(tau32 - tau64) / tau64)
    seuil = SEUILS['loitau_rel'] + SEUILS['loitau_nsigma'] * sigma_rel
    return dict(tau32=tau32, tau64=tau64, sigma32=s32, sigma64=s64, sigma_rel=sigma_rel, ecart_rel=ecart,
                seuil=float(seuil), ok=bool(ecart <= seuil), tau32_chaines=t32.tolist(), tau64_chaines=t64.tolist())


# ---------------------------------------------------------------------------------------------
# LOI-3 : KS par permutation sur les plaquettes locales (protocole P5 de analyse_gpu1.py) + |P| (Polyakov)
# ---------------------------------------------------------------------------------------------
def quat_vers_c16(q, L):
    """(4, L, L, L, L, 4) quaternions (a0, a1, a2, a3) -> (4, L, L, L, L, 2, 2) complex128 :
    [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]]."""
    a0, a1, a2, a3 = (q[..., k] for k in range(4))
    u = np.empty(q.shape[:-1] + (2, 2), dtype=np.complex128)
    u[..., 0, 0] = a0 + 1j * a3
    u[..., 0, 1] = a2 + 1j * a1
    u[..., 1, 0] = -a2 + 1j * a1
    u[..., 1, 1] = a0 - 1j * a3
    return u


def dag(m):
    return np.conj(np.swapaxes(m, -1, -2))


def plaq_locales(u):
    """½ ReTr U_mu(x) U_nu(x+mu) U_mu(x+nu)† U_nu(x)†, 6V valeurs (copie de analyse_gpu1.plaq_locales)."""
    out = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            a = u[mu] @ np.roll(u[nu], -1, axis=mu)
            b = u[nu] @ np.roll(u[mu], -1, axis=nu)
            out.append(0.5 * np.real(np.einsum('...ij,...ji->...', a, dag(b))))
    return np.stack(out).ravel()


def cfgs_npy(d):
    return sorted(glob.glob(os.path.join(d, 'cfg_*.npy')))


def cfgs_q64(d):
    return sorted(glob.glob(os.path.join(d, 'cfg_*.q64')))


def plaq_de_q64(fichier, L):
    q = np.fromfile(fichier, dtype='<f8').reshape(4, L, L, L, L, 4)
    return plaq_locales(quat_vers_c16(q, L))


def plaq_de_npy(fichier):
    return plaq_locales(np.load(fichier))


def ks_permutation(x32, x64, B, graine, nb):
    """Copie du protocole P5 : histogrammes par configuration sur nb bins de [-1,1], statistique D sur la
    grille, permutation des configurations entre les deux jeux."""
    edges = np.linspace(-1, 1, nb + 1)
    h32 = np.array([np.histogram(x, edges)[0] for x in x32], dtype=np.int64)
    h64 = np.array([np.histogram(x, edges)[0] for x in x64], dtype=np.int64)
    a = np.concatenate(x32)
    b = np.concatenate(x64)
    ks = sst.ks_2samp(a, b)

    def dgrid(u, v):
        c1 = np.cumsum(u) / u.sum()
        c2 = np.cumsum(v) / v.sum()
        return float(np.max(np.abs(c1 - c2)))

    tout = np.vstack([h32, h64])
    n1 = len(h32)
    d0 = dgrid(h32.sum(0), h64.sum(0))
    rng = np.random.default_rng(graine)
    dp = np.empty(B)
    for i in range(B):
        p = rng.permutation(len(tout))
        dp[i] = dgrid(tout[p[:n1]].sum(0), tout[p[n1:]].sum(0))
    pperm = float((1 + np.sum(dp >= d0)) / (B + 1))
    return dict(n_cfg_f32=len(h32), n_cfg_ref=len(h64), D_exact=float(ks.statistic), p_asympt_naive=float(ks.pvalue),
                D_grille=d0, p_perm=pperm, D_perm_q50=float(np.median(dp)), D_perm_q99=float(np.quantile(dp, 0.99)),
                moy_f32=float(a.mean()), moy_ref=float(b.mean()), sd_f32=float(a.std()), sd_ref=float(b.std()))


def juge_loi3_plaquettes(base32, B=None):
    B = B or SEUILS['loi3_permutations']
    res = {}
    for L in (6, 8):
        d32 = dossiers_chaines(base32, L, '2.8')
        dref = dossiers_chaines(REF_ENS, L, '2.8')
        x32 = [plaq_de_q64(f, L) for d in d32 for f in cfgs_q64(d)]
        x64 = [plaq_de_npy(f) for d in dref for f in cfgs_npy(d)]
        r = ks_permutation(x32, x64, B, SEUILS['loi3_graine'], SEUILS['loi3_bins'])
        r['ok'] = bool(r['p_perm'] > SEUILS['loi3_p'])
        res[f'L{L}'] = r
    res['ok'] = all(res[f'L{L}']['ok'] for L in (6, 8))
    return res


def series_polyakov(ap, dossiers, direction=3):
    """|P_direction| par chaîne (phase 'sample') et τ_int de |P| de chaque chaîne."""
    series, taus = [], []
    for d in dossiers:
        P, pol, _, _, _ = ap.read_gpu(d)
        s = np.abs(pol[:, direction])
        series.append(s)
        taus.append(ap.tau_int(s)[0])
    return series, taus


def juge_loi3_polyakov(ap, base32, direction=3):
    """Distribution de |P_3| à L = 8, β = 2,3 : KS à deux échantillons, séries amincies d'un pas
    k = ceil(5 · max τ_int(|P|)) (critère C4 : espacement >= 5 τ) pour les rendre quasi indépendantes."""
    s32, t32 = series_polyakov(ap, dossiers_chaines(base32, 8, '2.3'), direction)
    s64, t64 = series_polyakov(ap, reference_fp64(8, '2.3'), direction)
    k = int(np.ceil(5 * max(max(t32), max(t64))))
    a = np.concatenate([s[::k] for s in s32])
    b = np.concatenate([s[::k] for s in s64])
    ks = sst.ks_2samp(a, b)
    return dict(pas_amincissement=k, n32=int(a.size), n64=int(b.size), D=float(ks.statistic), p=float(ks.pvalue),
                tau32=t32, tau64=t64, moy32=float(a.mean()), moy64=float(b.mean()), ok=bool(ks.pvalue > SEUILS['loi3_p']))
