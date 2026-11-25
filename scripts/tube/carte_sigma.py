#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# carte_sigma.py -- agrege les analyses (analyse_tube.py) d'une production en table markdown (PREREG §4, critères C1-C4).
# usage : python3 -I carte_sigma.py A_L2.json A_L3.json A_L4.json [--derive D_L2.json D_L3.json D_L4.json] > carte.md
import sys, json, math
import numpy as np
# S_v(L) = S/beta du vortex du secteur doublement tordu : CANDIDATS minimums a l = 12 (volume 12 x 12 x L x L, torsion 01,23),
# porte G1 du module (`kvu su2 tube smin --cotes 12,12,L,L`, references du socle §3.1 retrouvees a moins de 1e-9 pres ; RAPPORT
# du module §2). Ce ne sont PAS des valeurs l -> infini : a L = 4, l = 16 donne 9.471232 (ecart 2.8e-4, fiche
# ERR-socle-l-infini-1e-4 du carnet du projet) ; incertitude en l de quelques 1e-4, non chiffree a L = 2 et 3.
SV = {2: 6.673485560, 3: 9.072532509, 4: 9.471515838}   # l = 12
# incertitude en l : chiffree seulement a L = 4 (l = 16 : 9.471232, ecart 2.8e-4) ; NON chiffree a L = 2 et 3
INCERTITUDE_L = {4: 'écart à ℓ = 16 : 2,8e-4 ; ℓ → ∞ non établi'}
args = sys.argv[1:]
der = []
if '--derive' in args:
    i = args.index('--derive'); der = args[i + 1:]; args = args[:i]
D = {}
for f in der:
    j = json.load(open(f)); D[j['cotes'][2]] = {r['chaine']: r for r in j['resultats']}
print('| L | β | n | χ(1,1) | χ(2,2) | χ(3,3) | χ(4,4) | χ(5,5) | χ(6,6) | χ(8,8) | σ plateau (R) | −ln tanh(βL²) | f_v | σ/(2f_v) | ⟨abs P₂⟩ | C1 viol. | dérive |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
pentes = {}
for f in args:
    j = json.load(open(f)); L = j['cotes'][2]; pts = []
    for r in j['resultats']:
        c = r['chi']
        def fx(R):
            v, e = c.get(str(R), c.get(R, (float('nan'), float('nan'))))
            return '%.4f(%.4f)' % (v, e) if np.isfinite(v) else '—'
        pl = r['plateau']
        if pl:
            sig = '%.5f(%.5f) R=%d-%d' % (pl['sigma'], pl['err'], pl['R'][0], pl['R'][-1]); pts.append((r['beta'], pl['sigma'], pl['err']))
        else:
            sig = 'non extrait'
        b = r['borne']; fv = r['frac_vortex'][0]
        rap = '%.2f' % (pl['sigma'] / (2 * fv)) if (pl and fv > 0) else '—'
        nv = len(b['violations_3sigma_simple'])
        dr = '—'
        if L in D and r['chaine'] in D[L]:
            r2 = D[L][r['chaine']]; worst = 0.0
            for R in range(2, 9):
                a1 = c.get(str(R)); a2 = r2['chi'].get(str(R))
                if a1 and a2 and np.isfinite(a1[0]) and np.isfinite(a2[0]) and a1[1] > 0:
                    worst = max(worst, abs(a1[0] - a2[0]) / math.hypot(a1[1], a2[1]))
            dr = '%.1fσ' % worst
        print('| %d | %.1f | %d | %s | %s | %s | %s | %s | %s | %s | %s | %.3g | %.2e | %s | %.3f | %d | %s |' % (
            L, r['beta'], r['n_obs'], fx(1), fx(2), fx(3), fx(4), fx(5), fx(6), fx(8), sig, b['moins_log_t'], fv, rap,
            r['absP2'][0], nv, dr))
    pts = [p for p in pts if p[1] > 0]
    if len(pts) >= 3:
        p3 = sorted(pts)[-3:]
        x = np.array([p[0] for p in p3]); y = np.log([p[1] for p in p3]); w = np.array([(p[1] / p[2]) ** 2 for p in p3])
        A = np.vstack([np.ones(3), x]).T; Wm = np.diag(w)
        cov = np.linalg.inv(A.T @ Wm @ A); coef = cov @ A.T @ Wm @ y
        pentes[L] = (float(coef[1]), float(math.sqrt(cov[1, 1])), [float(v) for v in x])
print()
for L, (s, e, xs) in pentes.items():
    print('- L = %d : pente d ln σ / dβ (trois plus grands β extraits %s) = %.3f ± %.3f ; −S_v(L) à ℓ = 12 = %.6f (%s)'
          % (L, xs, s, e, -SV[L], INCERTITUDE_L.get(L, 'incertitude en ℓ non chiffrée')))
