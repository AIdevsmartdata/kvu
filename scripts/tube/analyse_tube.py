#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# analyse_tube.py -- rapports de Creutz, borne elementaire, vortex, Polyakov de fibre a partir des tube.csv de `kvu su2 gen --cotes`.
# usage : python3 -I analyse_tube.py RUN_DIR [--therm-obs N] [--mutant-normalisation]  -> JSON sur stdout
# Erreurs : jackknife sur 20 blocs de la serie. Statut : [numerique].
import sys, json, math, os, glob, warnings
import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)

NB = 20

class Refus(Exception):
    """Analyse impossible : refus explicite (message sur stderr, code 2), jamais de résultat non fini."""

def lire(p):
    with open(p) as f:
        h = f.readline().strip().split(',')
    a = np.loadtxt(p, delimiter=',', skiprows=1, ndmin=2)
    return h, a

def blocs(x, nb=NB):
    n = len(x) // nb
    return np.array([x[k * n:(k + 1) * n].mean(axis=0) for k in range(nb)])

def jack(bm, f):
    """bm : moyennes de blocs (nb, ...) ; f : fonction des moyennes -> valeur et erreur jackknife."""
    nb = len(bm)
    tot = bm.mean(axis=0)
    v = f(tot)
    js = np.array([f((tot * nb - bm[k]) / (nb - 1)) for k in range(nb)])
    e = np.sqrt((nb - 1) / nb * np.sum((js - js.mean(axis=0)) ** 2, axis=0))
    return v, e

def analyser(dirc, beta, cotes, plan, rmax, tmax, mutant=False, ecarter=0.0):
    h, a = lire(os.path.join(dirc, 'tube.csv'))
    a = a[int(ecarter * a.shape[0]):]
    col = {k: i for i, k in enumerate(h)}
    n = a.shape[0]
    if n < NB:
        raise Refus(f"{dirc} : {n} observation(s) restante(s) (après --ecarter) : au moins {NB} sont requises "
                    f"(jackknife sur {NB} blocs), estimation impossible, refus")
    W = np.zeros((n, rmax + 1, tmax + 1))
    W[:, 0, 0] = 1.0
    for r in range(1, rmax + 1):
        W[:, r, 0] = 1.0
    for t in range(1, tmax + 1):
        W[:, 0, t] = 1.0
    for r in range(1, rmax + 1):
        for t in range(1, tmax + 1):
            W[:, r, t] = a[:, col[f'Wt_{r}x{t}']]
    if mutant:
        W = 2.0 * W  # MUTANT de la porte de borne : normalisation Tr au lieu de 1/2 Tr
    bm = blocs(W)
    out = {'n_obs': int(n), 'beta': beta}
    # Creutz carres
    chi = {}
    for R in range(1, min(rmax, tmax) + 1):
        def f(w, R=R):
            return -np.log(w[R, R] * w[R - 1, R - 1] / (w[R, R - 1] * w[R - 1, R]))
        try:
            v, e = jack(bm, f)
            chi[R] = (float(v), float(e))
        except Exception:
            chi[R] = (float('nan'), float('nan'))
    out['chi'] = chi
    # borne elementaire (version tore) pour les boucles du plan long
    fib = 1
    for d in range(4):
        if d not in plan:
            fib *= cotes[d]
    N2 = cotes[plan[0]] * cotes[plan[1]]
    t = math.tanh(beta * fib)
    viol = []; viol_s = []
    marge_min = float('inf'); marge_min_s = float('inf')
    for r in range(1, rmax + 1):
        for tt in range(1, tmax + 1):
            A = r * tt
            B = (t ** A + t ** (N2 - A)) / (1 - t ** N2) if t < 1 else float('inf')   # version tore exacte (rigoureuse)
            Bs = t ** A                                                               # version simple (enonce)
            m, e = jack(bm, lambda w, r=r, tt=tt: w[r, tt])
            e = max(e, 1e-300)
            marge_min = min(marge_min, (B - m) / e)
            marge_min_s = min(marge_min_s, (Bs - m) / e)
            if m > B + 3 * e:
                viol.append({'R': r, 'T': tt, 'W': float(m), 'err': float(e), 'borne': B})
            if m > Bs + 3 * e:
                viol_s.append({'R': r, 'T': tt, 'W': float(m), 'err': float(e), 'borne': Bs})
    out['borne'] = {'t': t, 'moins_log_t': -math.log(t) if t < 1 else 0.0, 'violations_3sigma': viol,
                    'marge_min_en_sigma': marge_min, 'violations_3sigma_simple': viol_s,
                    'marge_min_en_sigma_simple': marge_min_s, 'fibre_sites': fib}
    # vortex, Polyakov de fibre, plaquettes
    fv = [k for k in h if k.startswith('frac_vortex')][0]
    m, e = jack(blocs(a[:, col[fv]]), lambda x: x)
    out['frac_vortex'] = (float(m), float(e))
    for d in range(4):
        if d in plan:
            continue
        for k in (f'P{d}', f'absP{d}', f'P{d}sq'):
            m, e = jack(blocs(a[:, col[k]]), lambda x: x)
            out[k] = (float(m), float(e))
    for k in ('plaq01', 'plaq23', 'plaq02', 'V'):
        m, e = jack(blocs(a[:, col[k]]), lambda x: x)
        out[k] = (float(m), float(e))
    # plateau (critere C2 du PREREG)
    Rs = sorted(chi)
    best = None
    for i0 in range(len(Rs)):
        R0 = Rs[i0]
        if R0 < 3:
            continue
        for i1 in range(i0 + 2, len(Rs)):
            seg = Rs[i0:i1 + 1]
            v = np.array([chi[R][0] for R in seg]); e = np.array([chi[R][1] for R in seg])
            if not np.all(np.isfinite(v)) or not np.all(e > 0):
                continue
            w = 1 / e ** 2
            mu = float(np.sum(w * v) / np.sum(w))
            if np.all(np.abs(v - mu) <= 2 * e):
                if best is None or len(seg) > len(best['R']) or (len(seg) == len(best['R']) and seg[0] < best['R'][0]):
                    best = {'R': seg, 'sigma': mu, 'err_stat': float(1 / math.sqrt(np.sum(w))),
                            'demi_ecart': float((v.max() - v.min()) / 2)}
    if best:
        best['err'] = max(best['err_stat'], best['demi_ecart'])
    out['plateau'] = best
    return out

def main():
    d = sys.argv[1]
    mutant = '--mutant-normalisation' in sys.argv
    ecarter = 0.0
    if '--ecarter' in sys.argv:
        ecarter = float(sys.argv[sys.argv.index('--ecarter') + 1])
    if os.path.exists(os.path.join(d, 'manifest.json')):
        cfg = json.load(open(os.path.join(d, 'manifest.json')))['tube']   # manifeste commun (champ « tube »)
    else:
        cfg = json.load(open(os.path.join(d, 'tube.json')))               # sorties antérieures au manifeste commun
    cotes = cfg['cotes']
    plan = cfg['tube_obs']['plan']; rmax = cfg['tube_obs']['rmax']; tmax = cfg['tube_obs']['tmax']
    betas = cfg['betas']
    res = []
    nch = len(betas)
    for c, b in enumerate(betas):
        dirc = d if nch == 1 else os.path.join(d, f'c{c:03d}')
        if not os.path.exists(os.path.join(dirc, 'tube.csv')):
            continue
        attendu = cfg['nmeas'] // max(cfg['tube_obs'].get('every', 1) or 1, 1)
        with open(os.path.join(dirc, 'tube.csv')) as f:
            n_lignes = sum(1 for _ in f) - 1
        if n_lignes < attendu and '--partiel' not in sys.argv:
            continue   # chaine encore en cours : ignoree (sauf --partiel)
        try:
            r = analyser(dirc, b, cotes, plan, rmax, tmax, mutant, ecarter)
        except Refus as e:
            print(f'analyse_tube.py : {e}', file=sys.stderr)
            sys.exit(2)
        r['chaine'] = c
        res.append(r)
    print(json.dumps({'run': d, 'cotes': cotes, 'mutant_normalisation': mutant, 'ecarter': ecarter, 'resultats': res}, indent=1))

if __name__ == '__main__':
    main()
