#!/usr/bin/env python3
"""Tests de lecture_spectro.py. Python du venv certif (numpy + scipy).
  python3 test_lecture_spectro.py [DOSSIER_REDUCTIONS_R6]     (sans argument : tests synthetiques seulement)
Les reductions R6 s'obtiennent par `lecture_spectro.py reduire REP X_r<i>_p<k>.npz --canal k` (18 repliques : 8 acemagic, 10 loue).
Valeurs de reference : kvu_boite/RESULTAT_R6_COMBINE.md (20/11/2026)."""
import sys, os, glob, math, tempfile, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lecture_spectro as ls
ech = []
def ok(nom, cond, detail=''):
    ech.append(bool(cond)); print(('OK    ' if cond else 'ECHEC ') + nom + (' : ' + detail if detail else ''))

# --- synthetique : energies connues (0,9 ; 1,7 ; 2,6), 5 operateurs dont un doublon
with tempfile.TemporaryDirectory() as tmp:
    d = os.path.join(tmp, 's'); ls.synthetique(d)
    red = ls.reduire(d, 0, 10); red['man'] = ls.lire_manifeste(d)
    # flux = lecture directe (memes blocs)
    r = ls.lecture(red, 0, faire_multi=False)
    p0 = r['principale']['etat0']['plateau']; p1 = r['principale']['etat1']['plateau']
    ok('GEVP : E0 = 0,90 retrouvee', p0 and abs(p0['valeur'] - 0.9) < max(4 * p0['erreur'], 0.005), str(p0 and (round(p0['valeur'], 4), round(p0['erreur'], 4))))
    ok('GEVP : E1 = 1,70 retrouvee', p1 and abs(p1['valeur'] - 1.7) < max(4 * p1['erreur'], 0.005), str(p1 and (round(p1['valeur'], 4), round(p1['erreur'], 4))))
    ix = r['elagage']['idx']
    ok('elagage : jamais les deux copies ensemble', ix is None or not (0 in ix and 4 in ix))
    ok('bornes diagonales >= E0 (3 sigma)', all(v + 3 * e >= 0.9 for dg in r['diagonaux'].values() for t, v, e, k in dg['E_eff'][1:3] if k))
    # sans soustraction du vide (mutant) : l'energie doit etre fausse
    Cj = ls.echantillons(red['Cb'], red['Mb'], False); rm = ls.analyser(Cj, 5, 16, 1, [0, 1, 2])
    pm = rm['etat0']['plateau'] if 'etat0' in rm else None
    ok('mutant « vide oublie » detecte', pm is None or abs(pm['valeur'] - 0.9) > 5 * max(pm['erreur'], 1e-3))
    # multi-exponentielle sur donnees sans bruit : 2 exponentielles exactes
    T = 16; taus = np.arange(1, 6); E1, E2, A1, A2 = 0.8, 1.9, 1.0, 0.3
    f = lambda e: np.cosh(e * (T / 2 - taus)) / math.cosh(e * (T / 2 - 1))
    y = A1 * f(E1) + A2 * f(E2); W = np.diag(1 / (1e-6 * y) ** 2)
    b = ls.ajuste_n(y, taus, T, W, 2)
    ok('ajustement 2 exponentielles exact : E1 = 0,8', b and abs(b[0] - 0.8) < 1e-4, str(b and round(b[0], 6)))
    # canal d'impulsion synthetique : meme lecteur, pas de soustraction
    d2 = os.path.join(tmp, 'p'); ls.synthetique(d2, impulsion=True, graine=5)
    rp = ls.reduire(d2, 1, 10)
    ok('canal d\'impulsion lu (neutre_p1.bin)', rp['ncfg'] == 40000 and rp['Cb'].shape[0] == 4000)

# --- reel : R6 combine (18 repliques)
if len(sys.argv) > 1:
    D = sys.argv[1]
    REF = {1: dict(el=(13, 1.3121, 0.0016), diag=((1.3320, .0017), (1.3212, .0062)), b1=(1.3235, .0069)),
           2: dict(B15=(1.5659, .0028), el=(13, 1.5664, .0025), diag=((1.5970, .0030), (1.5591, .0141)), b1=(0.80, .94), b2=(1.16, .82), b2c=(1.12, .71)),
           3: dict(B15=(1.362, .319), el=(15, 1.362, .319), diag=((1.9410, .0064), (1.8788, .0418)), b1=(0.91, 1.91))}
    for k in (1, 2, 3):
        fs = sorted(glob.glob('%s/*_p%d.npz' % (D, k)))
        if len(fs) != 18: ok('R6 k=%d : 18 reductions' % k, False, str(len(fs))); continue
        red = ls.combiner([ls.charger_reduit(f) for f in fs]); red['man'] = ls.lire_manifeste(sorted(glob.glob('/chemin/vers/R6_b22986/r0'))[0])
        r = ls.lecture(red, k); R = REF[k]
        ok('R6 k=%d : 108000 blocs' % k, r['nb_blocs'] == 108000)
        m, v, e = R['el']
        el = r['elaguee']['etat0']['E_eff'][1] if 'elaguee' in r else (0, float('nan'), 0)
        ok('R6 k=%d : base elaguee m=%d, E_eff(1) = %.4f(%.4f)' % (k, m, v, e), len(r['elagage']['idx'] or []) == m and abs(el[1] - v) < 2 * e + 5e-4, '%.4f(%.4f)' % (el[1], el[2]))
        dg = r['diagonaux'][r['pre_designe']]['E_eff']
        for t, (rv, re_) in enumerate(R['diag'], 1):
            ok('R6 k=%d : borne diagonale tau=%d = %.4f(%.4f)' % (k, t, rv, re_), abs(dg[t][1] - rv) < 1.5 * re_ + 5e-4 and abs(dg[t][2] - re_) < 0.2 * re_ + 5e-4, '%.4f(%.4f)' % (dg[t][1], dg[t][2]))
        if 'B15' in R:
            p = r['principale']; x = p['etat0']['E_eff'][1]
            ok('R6 k=%d : B15 E_eff(1) = %.4f(%.4f)' % (k, *R['B15']), p['nom'] == 'B15' and abs(x[1] - R['B15'][0]) < 1.5 * R['B15'][1] + 5e-4, '%.4f(%.4f)' % (x[1], x[2]))
        for nm, key in (('b1_1-4', 'b1'), ('b2_1-5', 'b2'), ('b2c_1-5', 'b2c')):
            if key in R:
                x = (r['multi_exp'] if key != 'b2c' else r['multi_exp_cov'])[nm]
                ok('R6 k=%d : %s E1 = %.2f(%.2f)' % (k, nm, *R[key]), x and abs(x['E1'] - R[key][0]) < 0.03 and abs(x['err'] - R[key][1]) < 0.03, 'E1 = %.4f +- %.4f' % (x['E1'], x['err']))
print('\n%d/%d OK' % (sum(ech), len(ech))); sys.exit(0 if all(ech) else 1)
