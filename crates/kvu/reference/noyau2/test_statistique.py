#!/usr/bin/env python3
"""Équivalence STATISTIQUE du mode 'rapide' (RNG compilé) avec la chaîne de référence, + gates PREREG G9 / G10 rejouées.
Données : data/L8 = copies (SHA, SHA_L8.txt) des configurations V et des sorties de production de la chaîne de RÉFÉRENCE
(Cb/runL8, run_one.py → lib_Cb.run_chain, acemagic) ; β₀ ∈ {2,3 ; 16}, 10 V chacun, L = 8.
Protocole rapide = run_chain (100 + 500 balayages, mesure tous les 2, bins de 5) ; comparaison à N égal (250 mesures).
Usage : test_statistique.py <sortie.json>   RC 0 = tout VERT."""
import os, sys, io, json, time, hashlib, socket
import numpy as np, scipy.linalg as sla
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'ref', 'code'))
sys.dont_write_bytecode = True
import lib_Cb as R, lib_brancheC as B
import numba, cb

D8 = os.path.join(HERE, 'data', 'L8')
SHA = {l.split()[1]: l.split()[0] for l in open(os.path.join(D8, 'SHA_L8.txt'))}
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
for f, s in SHA.items():
    assert sha(os.path.join(D8, f)) == s, 'SHA ' + f
for f, s in {l.split()[1]: l.split()[0] for l in open(os.path.join(HERE, 'ref', 'SHA_ref.txt'))}.items():
    assert sha(os.path.join(HERE, f)) == s, 'SHA ' + f
man = json.load(open(os.path.join(D8, 'MANIFEST_L8.json')))
out = sys.argv[1]
res = dict(machine=socket.gethostname(), threads=numba.get_num_threads(), date=time.strftime('%Y-%m-%dT%H:%M:%S%z'), gates={}, V=[])
ok_all = True
def gate(nom, ok, **info):
    global ok_all
    ok_all &= bool(ok); res['gates'][nom] = dict(ok=bool(ok), **info)
    print(('VERT ' if ok else 'ROUGE'), nom, json.dumps(info, default=float), flush=True)

def load_cfg(e):
    a = np.load(os.path.join(D8, e['cfg_local']))
    if a.ndim == 6:
        C = np.empty(a.shape[:5] + (2, 2), dtype=complex)
        C[..., 0, :] = a; C[..., 1, 0] = -np.conj(a[..., 1]); C[..., 1, 1] = np.conj(a[..., 0])
    else:
        C = np.array(a)
    bio = io.BytesIO(); np.save(bio, C)
    assert hashlib.sha256(bio.getvalue()).hexdigest() == e['sha'], 'sha cfg complète ' + e['nom']
    return np.ascontiguousarray(C)

def gl(C, D):
    return sla.eigh(0.5 * (C + C.T), 0.5 * (D + D.T))

def moy_err(x):
    """moyenne, erreur avec τ_int (lib_Cb.tau_int), par colonne."""
    x = np.atleast_2d(x.T).T
    m = x.mean(0); n = len(x)
    tau = np.array([R.tau_int(x[:, i]) for i in range(x.shape[1])])
    return m, np.sqrt(x.var(0, ddof=1) * 2 * tau / n), tau

def cpcv(F, Db):
    """= analyse_Cb.chain_stats (validation croisée par moitiés) : Rayleigh du vecteur appris sur une moitié, évalué sur l'autre."""
    n = len(F); nb_bins = len(Db); h = n // 2; hb = nb_bins // 2; r = []
    for (a, b, da, db) in ((slice(0, h), slice(h, n), slice(0, hb), slice(hb, nb_bins)), (slice(h, n), slice(0, h), slice(hb, nb_bins), slice(0, hb))):
        l2, W2 = gl(np.cov(F[a].T), Db[da].mean(0)); ww = W2[:, -1]
        r.append(float(ww @ np.cov(F[b].T) @ ww / (ww @ Db[db].mean(0) @ ww)))
    return float(np.mean(r))

def med_jk(v):
    v = np.asarray(v); N = len(v); jk = np.array([np.median(np.delete(v, i)) for i in range(N)])
    return float(np.median(v)), float(np.sqrt((N - 1) / N * ((jk - jk.mean()) ** 2).sum()))

L = 8; geo = cb.Geometry(L); Wf = B.fourier_weights(L, 1)
per_beta = {}
t0 = time.time()
for e in man:
    beta = float(e['beta']); C = load_cfg(e)
    zr = np.load(os.path.join(D8, e['nom'] + '_ref.npz'))
    Fr, Dbr = zr['F'][:250], zr['Db'][:50]                       # N égal (la réf. prolongée garde ses 250 premières mesures)
    rng = np.random.default_rng(int(e['seed']) + 7919)          # graine distincte de la production
    t = time.time()
    r = cb.chaine(C, beta, rng, Wf, geo=geo, ntherm=100, nsw=500, meas_every=2, binw=5, mode='rapide')
    tch = time.time() - t
    Fm, Dbm = r['F'], r['Db']
    mr, sr, taur = moy_err(Fr); mm, sm, taum = moy_err(Fm)
    z = (mm - mr) / np.sqrt(sr ** 2 + sm ** 2)
    # plaquette moyenne = moyenne des 6 composantes n = 0 (cos) × L²/L⁴ … on prend directement les F_n=0 (6 plans)
    cp_r = float(gl(np.cov(Fr.T), Dbr.mean(0))[0][-1]); cp_m = float(gl(np.cov(Fm.T), Dbm.mean(0))[0][-1])
    trr = float(gl(np.cov(Fr.T), Dbr.mean(0))[0].mean()); trm = float(gl(np.cov(Fm.T), Dbm.mean(0))[0].mean())
    Dr_moy = Dbr.mean(0); Dm_moy = Dbm.mean(0)
    dD = float(np.linalg.norm(Dm_moy - Dr_moy) / np.linalg.norm(Dr_moy))
    rec = dict(nom=e['nom'], beta=beta, t_chaine_rapide=round(tch, 2), t_chaine_ref_prod=float(json.load(open(os.path.join(D8, e['nom'] + '_ref.json')))['t_total']),
               z_max=float(np.abs(z).max()), n_z_gt3=int((np.abs(z) > 3).sum()), z_moy=float(z.mean()), z2_moy=float((z ** 2).mean()),
               cp_ref=cp_r, cp_rapide=cp_m, cpcv_ref=cpcv(Fr, Dbr), cpcv_rapide=cpcv(Fm, Dbm), tr54_ref=trr, tr54_rapide=trm, dD_rel=dD, Vdev=r['Vdev'], unit=r['unit'],
               tau_max_rapide=float(taum.max()), tau_max_ref=float(taur.max()))
    res['V'].append(rec); per_beta.setdefault(beta, []).append((rec, z))
    print(f"{e['nom']:12s} rapide {tch:5.1f} s (réf. prod {rec['t_chaine_ref_prod']:.0f} s)  |z|max {rec['z_max']:.2f}  "
          f"CP réf {cp_r:.4f} rapide {cp_m:.4f}  ΔD {dD:.3f}  Vdev {r['Vdev']:.1e}", flush=True)

gref = json.load(open(os.path.join(D8, 'gauss_L8.json')))
for beta, lst in sorted(per_beta.items()):
    zs = np.concatenate([z for _, z in lst]); recs = [r for r, _ in lst]
    nz = len(zs)
    gate(f'b{beta:g}_F54_z : |z|max < 4,5 et fraction |z|>3 ≤ 1 %', np.abs(zs).max() < 4.5 and (np.abs(zs) > 3).mean() <= 0.01,
         n=nz, z_max=float(np.abs(zs).max()), frac_gt3=float((np.abs(zs) > 3).mean()), z_moy=float(zs.mean()), z2_moy=float((zs ** 2).mean()))
    # moyenne des z sur 540 valeurs : sous H0 et en négligeant les corrélations entre F_i, σ ≈ 1/√n ; on borne |z̄| < 4/√(n/54)
    # (corrélations entre composantes : on prend un seul degré de liberté effectif par V, borne prudente)
    gate(f'b{beta:g}_biais_moyen_z', abs(zs.mean()) < 4 / np.sqrt(len(lst)), z_moy=float(zs.mean()), borne=4 / np.sqrt(len(lst)))
    d = np.array([r['cp_rapide'] - r['cp_ref'] for r in recs])
    tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    gate(f'b{beta:g}_CPvar_apparie_|t|<3', abs(tstat) < 3, t=tstat, d_moy=float(d.mean()), cp_ref_moy=float(np.mean([r['cp_ref'] for r in recs])))
    mr_, sr_ = med_jk([r['cp_ref'] for r in recs]); mm_, sm_ = med_jk([r['cp_rapide'] for r in recs])
    gate(f'b{beta:g}_CPvar_mediane_V_3sigma', abs(mm_ - mr_) < 3 * np.hypot(sr_, sm_), med_ref=mr_, s_ref=sr_, med_rapide=mm_, s_rapide=sm_)
    dtr = np.array([r['tr54_rapide'] - r['tr54_ref'] for r in recs]); ttr = float(dtr.mean() / (dtr.std(ddof=1) / np.sqrt(len(dtr))))
    gate(f'b{beta:g}_tr54_apparie_|t|<3', abs(ttr) < 3, t=ttr)
    gate(f'b{beta:g}_Vdev_unit_1e-12', max(max(r['Vdev'], r['unit']) for r in recs) < 1e-12, max=max(max(r['Vdev'], r['unit']) for r in recs))
    if beta == 16:
        # G9 (PREREG, analyse_Cb) : r_P(β = 16) ∈ [0,8 ; 1,25], forme 'full' (λ_max) et forme 'cv_cv' (validation croisée).
        # La référence (test/analyse_L8.json du chantier, copiée avec SHA) donne full ROUGE (biais MP à N = 250), cv VERT.
        # Reproduction = (a) notre recalcul des chaînes de réf. retrouve ses nombres (≤ 1e-10 rel.), (b) mêmes verdicts en rapide,
        # (c) r_P rapide compatible avec r_P réf. à 3σ (σ = jackknife de la médiane sur V, les deux côtés).
        aref = json.load(open(os.path.join(D8, 'analyse_L8_ref.json')))['gates']['G9']['8']
        cg = gref['fam_med_full'] / beta; cgc = gref['fam_med_cv'] / beta
        mcr, scr = med_jk([r['cpcv_ref'] for r in recs]); mcm, scm = med_jk([r['cpcv_rapide'] for r in recs])
        rf_ref, rf_rap = mr_ / cg, mm_ / cg; rc_ref, rc_rap = mcr / cgc, mcm / cgc
        a_ok = abs(rf_ref / aref['r_P_full'] - 1) <= 1e-10 and abs(rc_ref / aref['r_P_cv_cv'] - 1) <= 1e-10
        vf = lambda x: 0.8 <= x <= 1.25
        b_ok = vf(rf_rap) == aref['ok_full'] and vf(rc_rap) == aref['ok_cv']
        c_ok = abs(rf_rap - rf_ref) < 3 * np.hypot(sr_, sm_) / cg and abs(rc_rap - rc_ref) < 3 * np.hypot(scr, scm) / cgc
        gate('G9_L8_reproduite (recalcul réf. = analyse_L8 ; mêmes verdicts full/cv ; rapide ≈ réf. à 3σ)', a_ok and b_ok and c_ok,
             r_P_full_ref=rf_ref, r_P_full_ref_chantier=aref['r_P_full'], r_P_full_rapide=rf_rap, verdict_full_ref=aref['ok_full'],
             r_P_cv_ref=rc_ref, r_P_cv_ref_chantier=aref['r_P_cv_cv'], r_P_cv_rapide=rc_rap, verdict_cv_ref=aref['ok_cv'],
             verdict_full_rapide=vf(rf_rap), verdict_cv_rapide=vf(rc_rap), a=a_ok, b=b_ok, c=c_ok)
res['t_stat'] = round(time.time() - t0, 1)

# ---------------------------------------------------------------- loi du bain thermique rapide (intrinsèque, une agrafe)
# x₀ ∝ √(1−x²) e^{αx} : E[x₀] = I₂(α)/I₁(α) ; KS contre la CDF exacte (quadrature). Couvre les deux branches (α < 2 : proposition
# exponentielle ; α ≥ 2 : Kennedy–Pendleton) et la limite α → 0 (β = 0).
from scipy.special import ive
from scipy import integrate, stats as sst
t = time.time(); res['hb'] = []
A0 = np.array([[0.3 + 1.1j, 0.7 - 0.2j], [-0.7 - 0.2j, 0.3 - 1.1j]])      # multiple de SU(2), k = √det
kA = float(np.sqrt(np.linalg.det(A0).real))
for alpha in (0.0, 0.5, 1.9, 2.0, 5.0, 30.0, 200.0):
    n = 400000; xs = np.empty(n)
    cb._k_hb_test(A0, alpha / kA, np.uint64(int(alpha * 1000) + 17), xs)
    if alpha == 0.0:
        mu_ex = 0.0
        cdf = lambda x: (np.arcsin(np.clip(x, -1, 1)) + np.clip(x, -1, 1) * np.sqrt(1 - np.clip(x, -1, 1) ** 2)) / np.pi + 0.5
    else:
        mu_ex = float(ive(2, alpha) / ive(1, alpha))
        grid = np.linspace(-1, 1, 200001); dens = np.sqrt(1 - grid ** 2) * np.exp(alpha * (grid - 1))
        cum = integrate.cumulative_trapezoid(dens, grid, initial=0); cum /= cum[-1]
        cdf = lambda x, g=grid, c=cum: np.interp(x, g, c)
    se = xs.std() / np.sqrt(n); z = (xs.mean() - mu_ex) / se
    ks = sst.kstest(xs, cdf)
    rec = dict(alpha=alpha, moy=float(xs.mean()), moy_exacte=mu_ex, z=float(z), ks_p=float(ks.pvalue), ks_D=float(ks.statistic))
    res['hb'].append(rec)
    gate(f'bain_thermique_rapide_loi_alpha{alpha:g} (|z|<4, KS p>1e-4)', abs(z) < 4 and ks.pvalue > 1e-4, **rec)
res['t_hb'] = round(time.time() - t, 1)

# ---------------------------------------------------------------- G10 rejouée À L'IDENTIQUE (mêmes tirages que gate_G10_GG3b)
def jk_lam(F, D, nb=25, f=lambda l: l[-1]):
    n = len(F); idx = np.array_split(np.arange(n), nb); vals = []
    for b in idx:
        m = np.ones(n, bool); m[b] = False
        vals.append(f(sla.eigh(np.cov(F[m].T), D[m].mean(0), eigvals_only=True)))
    vals = np.array(vals); return float(np.sqrt((nb - 1) / nb * ((vals - vals.mean()) ** 2).sum()))
t = time.time()
rng = np.random.default_rng(27092026); N = 2000; Fs = []; Ds = []
for s in range(N):
    q = rng.standard_normal((4,) + (L,) * 4 + (4,)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    C = np.empty((4,) + (L,) * 4 + (2, 2), dtype=complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]; C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]; C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]; C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    F, D = cb.mesure_libre(C, geo, Wf); Fs.append(F); Ds.append(D)
Fs = np.array(Fs); Ds = np.array(Ds)
G10ref = json.load(open(os.path.join(D8, 'gate_G10_GG3b_ref.json')))
for n in (250, 2000):
    lam = sla.eigh(np.cov(Fs[:n].T), Ds[:n].mean(0), eigvals_only=True)
    s_max = jk_lam(Fs[:n], Ds[:n]); s_tr = jk_lam(Fs[:n], Ds[:n], f=lambda l: l.mean())
    lit = abs(lam[-1] - 1 / 6) < 3 * s_max; nor = abs(lam.mean() - 1 / 6) < 3 * s_tr
    rl, rn = G10ref[f'G10_litteral_N{n}'], G10ref[f'G10_normalisation_N{n}']
    d = max(abs(lam[-1] - rl['info']['lam_max']) / rl['info']['lam_max'], abs(s_max - rl['info']['sigma']) / rl['info']['sigma'],
            abs(lam.mean() - rn['info']['tr_sur_54']) / rn['info']['tr_sur_54'], abs(s_tr - rn['info']['sigma']) / rn['info']['sigma'])
    gate(f'G10_N{n}_reproduite (mêmes verdicts que la réf. ; écart rel. ≤ 1e-9)', lit == rl['ok'] and nor == rn['ok'] and d <= 1e-9,
         litteral=bool(lit), litteral_ref=rl['ok'], normalisation=bool(nor), normalisation_ref=rn['ok'], lam_max=float(lam[-1]),
         tr_sur_54=float(lam.mean()), ecart_rel_max=float(d))
res['t_G10'] = round(time.time() - t, 1)
res['G11'] = 'NON rejouée : exige kfib = λ_min de la Hessienne de fibre (cible 3, noyau hess) ; hors périmètre du noyau 2'
res['ok'] = bool(ok_all)
json.dump(res, open(out, 'w'), indent=1, default=float)
print('STATISTIQUE', 'VERTE' if ok_all else 'ROUGE', out, f"({res['t_stat']} s chaînes, {res['t_G10']} s G10)")
sys.exit(0 if ok_all else 1)
