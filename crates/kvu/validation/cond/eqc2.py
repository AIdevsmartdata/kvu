#!/usr/bin/env python3
"""EQC-2 : équivalence STATISTIQUE du mode `rapide` (kvu cond chaine, GPU) avec les chaînes de production de référence.

Protocole et seuils : noyau2/test_statistique.py (gelés avant tout run ; voir ARCHITECTURE_UNIFIEE.md §6.3).
Données (lues EN PLACE, jamais copiées) : /chemin/vers/projet/outils/ym4_noyaux/noyau2/data/L8 : 10 V × β₀ ∈ {2,3 ; 16}, L = 8,
250 mesures (100 + 500 balayages, mesure tous les 2, D à CHAQUE mesure puis bins de 5), N égal à la référence.

Gates par β₀ :
  F54_z      : |z|max < 4,5 et fraction |z| > 3 ≤ 1 % (540 z)
  biais_z    : |z̄| < 4/√nV
  CPvar      : C_P^var apparié sur V, |t| < 3
  mediane    : médianes sur V compatibles à 3σ (jackknife)
  tr54       : tr/54 apparié, |t| < 3
  Vdev_unit  : Vdev, unitarité < 1e-12
Loi du tirage (intrinsèque) pour α ∈ {0 ; 0,5 ; 1,9 ; 2 ; 5 ; 30 ; 200} : |z| < 4 (moyenne = I₂/I₁) et KS p > 1e-4 (4·10⁵ tirages).
Mutant : le même protocole avec β × 1,01 doit être ROUGE (au moins une gate violée) ; sinon la gate n'a pas de puissance.

Usage : eqc2.py --bin <binaire> [--backend auto|cpu|hip|cuda] [--mutant] [--loi] [--sortie out.json] [--tmp DIR] [--nv 10]
  <binaire> : soit un exécutable `kevinotron` (appelé `<bin> cond …`), soit le binaire de test `cond_cli-*` (pont, voir
  tests/cond_cli.rs). RC 0 si tout est conforme (nominal VERT et mutant ROUGE si --mutant).
"""
import os, sys, json, time, shutil, subprocess, argparse, tempfile
import numpy as np
import scipy.linalg as sla

NOYAU = '/chemin/vers/projet/outils/ym4_noyaux/noyau2'
sys.path.insert(0, os.path.join(NOYAU, 'ref', 'code'))
sys.dont_write_bytecode = True
import lib_Cb as R  # noqa: E402  (tau_int de la référence)

D8 = os.path.join(NOYAU, 'data', 'L8')

ap = argparse.ArgumentParser()
ap.add_argument('--bin', required=True)
ap.add_argument('--backend', default='auto')
ap.add_argument('--mutant', action='store_true', help='rejoue le protocole avec beta x 1,01 (doit être ROUGE)')
ap.add_argument('--loi', action='store_true', help='ajoute la gate de loi du tirage (7 α)')
ap.add_argument('--sortie', default=None)
ap.add_argument('--tmp', default=None)
ap.add_argument('--nv', type=int, default=10, help='nombre de V par β (10 = protocole complet)')
ap.add_argument('--mode', default='rapide', choices=['rapide', 'ref_rng'],
                help="mode des chaînes : rapide (la gate) ou ref_rng (CALIBRATION : loi exacte de la référence, mesure le taux de fausses alarmes de la gate)")
ap.add_argument('--decalage-graine', type=int, default=7919,
                help="décalage de graine du protocole de test_statistique (7919). Toute autre valeur = RÉPLICAT d'information, non gaté")
args = ap.parse_args()

TMP = args.tmp or tempfile.mkdtemp(prefix='eqc2_', dir=os.environ.get('TMPDIR', '/tmp'))
os.makedirs(TMP, exist_ok=True)


def cond(*a, check=True):
    """Exécute `cond <a…>` ; rend (rc, stdout)."""
    if os.path.basename(args.bin).startswith('cond_cli-'):
        env = dict(os.environ, KVU_COND_ARGV='\x1f'.join(a))
        cmd = [args.bin, '--ignored', '--exact', 'pont_cli', '--nocapture', '--test-threads=1']
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        out = '\n'.join(l for l in p.stdout.splitlines() if l.startswith('{'))
    else:
        p = subprocess.run([args.bin, 'cond', *a], capture_output=True, text=True)
        out = p.stdout.strip()
    if check and p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f'cond {a[0]} : rc = {p.returncode}')
    return p.returncode, out


# ------------------------------------------------------------------------------------------ outils statistiques (test_statistique.py)
def gl(C, D):
    return sla.eigh(0.5 * (C + C.T), 0.5 * (D + D.T))


def moy_err(x):
    """moyenne, erreur avec τ_int (lib_Cb.tau_int), par colonne."""
    x = np.atleast_2d(x.T).T
    m = x.mean(0); n = len(x)
    tau = np.array([R.tau_int(x[:, i]) for i in range(x.shape[1])])
    return m, np.sqrt(x.var(0, ddof=1) * 2 * tau / n), tau


def med_jk(v):
    v = np.asarray(v); N = len(v); jk = np.array([np.median(np.delete(v, i)) for i in range(N)])
    return float(np.median(v)), float(np.sqrt((N - 1) / N * ((jk - jk.mean()) ** 2).sum()))


def to_q(C):
    q = np.empty(C.shape[:-2] + (4,))
    q[..., 0] = 0.5 * (C[..., 0, 0].real + C[..., 1, 1].real)
    q[..., 3] = 0.5 * (C[..., 0, 0].imag - C[..., 1, 1].imag)
    q[..., 2] = 0.5 * (C[..., 0, 1].real - C[..., 1, 0].real)
    q[..., 1] = 0.5 * (C[..., 0, 1].imag + C[..., 1, 0].imag)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def load_cfg(e):
    a = np.load(os.path.join(D8, e['cfg_local']))
    if a.ndim == 6:
        C = np.empty(a.shape[:5] + (2, 2), dtype=complex)
        C[..., 0, :] = a; C[..., 1, 0] = -np.conj(a[..., 1]); C[..., 1, 1] = np.conj(a[..., 0])
    else:
        C = np.array(a)
    return C


def une_chaine(e, facteur_beta):
    """Une chaîne rapide du protocole de run_chain ; rend (F (250,54), Db (50,54,54), controles)."""
    cfg = os.path.join(TMP, f"{e['nom']}.q64.npy")
    np.save(cfg, to_q(load_cfg(e)))
    out = os.path.join(TMP, f"run_{e['nom']}_{facteur_beta:g}")
    if os.path.exists(out):
        shutil.rmtree(out)
    cond('chaine', '--cfg', cfg, '--beta', repr(float(e['beta']) * facteur_beta), '--graine', str(int(e['seed']) + args.decalage_graine),
         '--mode', args.mode, '--ntherm', '100', '--nsweep', '500', '--every', '2', '--deach', '1', '--ckpt', '0',
         '--out', out, '--backend', args.backend, '--sans-journal')
    F = np.load(os.path.join(out, 'F.npy')); D = np.load(os.path.join(out, 'D.npy'))
    ctl = json.load(open(os.path.join(out, 'controles.json')))
    shutil.rmtree(out); os.remove(cfg)
    assert F.shape == (250, 54) and D.shape == (250, 54, 54), (F.shape, D.shape)
    return F, D.reshape(50, 5, 54, 54).mean(1), ctl


def protocole(facteur_beta):
    man = json.load(open(os.path.join(D8, 'MANIFEST_L8.json')))
    per_beta = {}
    gates = {}
    t0 = time.time()
    for e in man:
        if int(e['iV']) >= args.nv:
            continue
        beta = float(e['beta'])
        zr = np.load(os.path.join(D8, e['nom'] + '_ref.npz'))
        Fr, Dbr = zr['F'][:250], zr['Db'][:50]
        t = time.time()
        Fm, Dbm, ctl = une_chaine(e, facteur_beta)
        mr, sr, _ = moy_err(Fr); mm, sm, _ = moy_err(Fm)
        z = (mm - mr) / np.sqrt(sr ** 2 + sm ** 2)
        cp_r = float(gl(np.cov(Fr.T), Dbr.mean(0))[0][-1]); cp_m = float(gl(np.cov(Fm.T), Dbm.mean(0))[0][-1])
        trr = float(gl(np.cov(Fr.T), Dbr.mean(0))[0].mean()); trm = float(gl(np.cov(Fm.T), Dbm.mean(0))[0].mean())
        rec = dict(nom=e['nom'], beta=beta, z_max=float(np.abs(z).max()), cp_ref=cp_r, cp_rapide=cp_m, tr54_ref=trr, tr54_rapide=trm,
                   Vdev=ctl['Vdev'], unit=ctl['unitarite'], t_chaine=round(time.time() - t, 1))
        per_beta.setdefault(beta, []).append((rec, z))
        print(f"{e['nom']:12s} {rec['t_chaine']:6.1f} s  |z|max {rec['z_max']:.2f}  CP réf {cp_r:.4f} rapide {cp_m:.4f}  Vdev {ctl['Vdev']:.1e}", flush=True)

    def gate(nom, ok, **info):
        gates[nom] = dict(ok=bool(ok), **info)
        print(('VERT ' if ok else 'ROUGE'), nom, json.dumps(info, default=float), flush=True)

    for beta, lst in sorted(per_beta.items()):
        zs = np.concatenate([z for _, z in lst]); recs = [r for r, _ in lst]
        gate(f'b{beta:g}_F54_z |z|max<4,5 et frac(|z|>3)<=1%', np.abs(zs).max() < 4.5 and (np.abs(zs) > 3).mean() <= 0.01,
             n=len(zs), z_max=float(np.abs(zs).max()), n_gt3=int((np.abs(zs) > 3).sum()), frac_gt3=float((np.abs(zs) > 3).mean()),
             z2_moy=float((zs ** 2).mean()))
        gate(f'b{beta:g}_biais_moyen_z', abs(zs.mean()) < 4 / np.sqrt(len(lst)), z_moy=float(zs.mean()), borne=4 / np.sqrt(len(lst)))
        d = np.array([r['cp_rapide'] - r['cp_ref'] for r in recs])
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        gate(f'b{beta:g}_CPvar_apparie_|t|<3', abs(tstat) < 3, t=tstat)
        mr_, sr_ = med_jk([r['cp_ref'] for r in recs]); mm_, sm_ = med_jk([r['cp_rapide'] for r in recs])
        gate(f'b{beta:g}_CPvar_mediane_3sigma', abs(mm_ - mr_) < 3 * np.hypot(sr_, sm_), med_ref=mr_, med_rapide=mm_)
        dtr = np.array([r['tr54_rapide'] - r['tr54_ref'] for r in recs]); ttr = float(dtr.mean() / (dtr.std(ddof=1) / np.sqrt(len(dtr))))
        gate(f'b{beta:g}_tr54_apparie_|t|<3', abs(ttr) < 3, t=ttr)
        mx = max(max(r['Vdev'], r['unit']) for r in recs)
        gate(f'b{beta:g}_Vdev_unit_1e-12', mx < 1e-12, max=mx)
    return gates, round(time.time() - t0, 1)


def loi_du_tirage():
    from scipy.special import ive
    from scipy import integrate, stats as sst
    gates = {}
    for alpha in (0.0, 0.5, 1.9, 2.0, 5.0, 30.0, 200.0):
        n = 400000
        f = os.path.join(TMP, f'x0_{alpha:g}.npy')
        cond('tirage', '--alpha', repr(alpha), '--n', str(n), '--graine', str(int(alpha * 1000) + 17), '--out', f, '--backend', args.backend)
        xs = np.load(f); os.remove(f)
        if alpha == 0.0:
            mu_ex = 0.0
            cdf = lambda x: (np.arcsin(np.clip(x, -1, 1)) + np.clip(x, -1, 1) * np.sqrt(1 - np.clip(x, -1, 1) ** 2)) / np.pi + 0.5
        else:
            mu_ex = float(ive(2, alpha) / ive(1, alpha))
            grid = np.linspace(-1, 1, 200001); dens = np.sqrt(1 - grid ** 2) * np.exp(alpha * (grid - 1))
            cum = integrate.cumulative_trapezoid(dens, grid, initial=0); cum /= cum[-1]
            cdf = lambda x, g=grid, c=cum: np.interp(x, g, c)
        z = (xs.mean() - mu_ex) / (xs.std() / np.sqrt(n)); ks = sst.kstest(xs, cdf)
        ok = abs(z) < 4 and ks.pvalue > 1e-4
        gates[f'loi_alpha{alpha:g} |z|<4, KS p>1e-4'] = dict(ok=bool(ok), z=float(z), ks_p=float(ks.pvalue), moy=float(xs.mean()), moy_exacte=mu_ex)
        print(('VERT ' if ok else 'ROUGE'), f'loi alpha={alpha:g}', f'z={z:.2f} KS p={ks.pvalue:.3f}', flush=True)
    return gates


res = dict(date=time.strftime('%Y-%m-%dT%H:%M:%S%z'), backend=args.backend, bin=args.bin, nv=args.nv, decalage_graine=args.decalage_graine, mode=args.mode)
ok_global = True
if args.loi:
    g = loi_du_tirage(); res['loi'] = g; ok_global &= all(v['ok'] for v in g.values())
if not (args.loi and args.nv == 0):
    g, dt = protocole(1.0)
    res['nominal'] = g; res['t_nominal_s'] = dt
    nominal_vert = all(v['ok'] for v in g.values()) and len(g) > 0
    print('EQC-2 nominal :', 'VERT' if nominal_vert else 'ROUGE')
    ok_global &= nominal_vert
    if args.mutant:
        gm, dtm = protocole(1.01)
        res['mutant'] = gm; res['t_mutant_s'] = dtm
        rouges = [k for k, v in gm.items() if not v['ok']]
        res['mutant_rouges'] = rouges
        print('EQC-2 mutant (beta x 1,01) :', 'ROUGE comme exigé (%d gates rouges)' % len(rouges) if rouges else 'VERT : PAS DE PUISSANCE')
        ok_global &= bool(rouges)
if args.sortie:
    json.dump(res, open(args.sortie, 'w'), indent=1, default=float)
shutil.rmtree(TMP, ignore_errors=True)
print('EQC-2', ('PASS' if ok_global else 'FAIL') if (args.decalage_graine == 7919 and args.mode == 'rapide') else 'REPLICAT (non gaté, décalage %d) : %s' % (args.decalage_graine, 'conforme' if ok_global else 'écart'))
sys.exit(0 if ok_global else 1)
