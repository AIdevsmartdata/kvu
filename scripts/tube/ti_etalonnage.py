#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ti_etalonnage.py -- integration thermodynamique en beta pour le cas d'etalonnage n = 2 (PREREG_ETALONNAGE.md).
# usage : python3 -I ti_etalonnage.py DIR_G0 DIR_G1 DIR_G2 DIR_G3 DIR_G4  (chacun contient apparie.json de `kvu su2 tube apparie`)
# F(G, b) = log Z(G, b), Z(G, 0) = 1 (Haar) : F = -int_0^b <V>_G ;  p_j = 2F(G_j) - F(G_{j+1}) ; Delta = 8p0 + 4p1 + 2p2 + p3
#   = 16 F(G_0) - F(G_4) ;  log R(G) = log Z_tw/Z = -int_0^b (<V>_1 - <V>_0).
# Quadrature : Simpson composite sur la grille uniforme (b = 0 exact : <V> = 6|G|, difference de secteurs nulle) ;
# controle : Simpson au pas double (ecart rendu). Erreurs : points independants, sum w_i^2 sigma_i^2.
# Reference couplage fort (ordre dominant) : t = I2(b)/I1(b) ; Delta_SC(2) = 384 t^4 + 1536 t^8 ;
#   log R_SC(G, plan (a,b)) = -2 (|G| / (l_a l_b)) t^(l_a l_b), (a,b) = plan AJOUTE par le secteur 1.  Troncature a priori :
#   relative ~ 20 t^4.
# Protocole (PREREG_ETALONNAGE) : secteur 0 sans torsion (--fibre aucune), secteur 1 = un seul plan tordu (--long PLAN) ;
#   tout autre couple de secteurs est REFUSE (code 1, message « hors protocole ») : la reference ci-dessus ne vaut que la.
# P4 (8p0 + 4p1 + 2p2 + p3 = Delta) est une IDENTITE ALGEBRIQUE sur les memes donnees et les memes poids : controle de
#   l'arithmetique du script, sans valeur de validation (elle tient aussi pour des donnees fausses, p. ex. un decalage
#   commun des beta). Le mutant --mutant-decalage n'est juge que par les references independantes (P1, P1_abs, P2).
import sys, json, math
import numpy as np

def bessel_i(n, x, kmax=60):
    return sum((x / 2) ** (2 * k + n) / (math.factorial(k) * math.factorial(k + n)) for k in range(kmax))

def t_sc(b):
    return bessel_i(2, b) / bessel_i(1, b)

def chi2_repliques(j):
    """chi2/ddl de la dispersion des repliques (V0 et V1) autour de leur moyenne ponderee, par beta."""
    out = []
    for r in j['resultats']:
        pr = r.get('par_replique')
        if not pr:
            continue
        for s in ('V0', 'V1'):
            m = np.array([x[0] for x in pr[s]]); e = np.array([x[1] for x in pr[s]])
            w = 1 / e ** 2; mu = np.sum(w * m) / np.sum(w)
            out.append(float(np.sum(w * (m - mu) ** 2) / max(len(m) - 1, 1)))
    return out

def plans(s):
    return set() if s == 'aucune' else set(s.split(','))

def plan_ajoute(j, d):
    """Plan tordu AJOUTE par le secteur 1 ; refus (sortie, code 1) de tout couple de secteurs hors protocole."""
    s0, s1 = plans(j['secteur0']), plans(j['secteur1'])
    if s0 or len(s1) != 1:
        sys.exit("ti_etalonnage : %s : secteurs hors protocole d'etalonnage (secteur0 = %s, secteur1 = %s ; attendu : "
                 "secteur0 = aucune et secteur1 = un seul plan, soit `su2 tube apparie --fibre aucune --long PLAN`)"
                 % (d, j['secteur0'], j['secteur1']))
    return s1.pop()

def charger(d, decalage=False):
    """d : un repertoire, ou plusieurs joints par '+' (grilles fusionnees, triees par beta)."""
    js = [json.load(open(x + '/apparie.json')) for x in d.split('+')]
    j = dict(js[0])
    j['resultats'] = sorted([r for jj in js for r in jj['resultats']], key=lambda r: r['beta'])
    for jj in js[1:]:
        assert jj['cotes'] == j['cotes'] and jj['secteur0'] == j['secteur0'] and jj['secteur1'] == j['secteur1'], \
            'fusion de grilles incompatibles'
    plan = plan_ajoute(j, d)

    cotes = j['cotes']; vol = int(np.prod(cotes))
    b = [0.0] + [r['beta'] for r in j['resultats']]
    v0 = [6.0 * vol] + [r['V0'][0] for r in j['resultats']]
    e0 = [0.0] + [r['V0'][1] for r in j['resultats']]
    dv = [0.0] + [r['DV'][0] for r in j['resultats']]
    edv = [0.0] + [r['DV'][1] for r in j['resultats']]
    if decalage:  # MUTANT : integrande attribue au point de grille precedent (etiquetage de beta faux)
        for a in (v0, e0, dv, edv):
            a[1:-1] = a[2:]
    return dict(cotes=cotes, vol=vol, b=np.array(b), v0=np.array(v0), e0=np.array(e0), dv=np.array(dv), edv=np.array(edv),
                plan=plan, chi2=chi2_repliques(j))

def simpson_cumul(y, e, h):
    """integrales de 0 a b_k pour k pair, et erreurs (points independants)."""
    n = len(y); out = {}
    for k in range(2, n, 2):
        w = np.zeros(n)
        for i in range(0, k, 2):
            w[i] += h / 3; w[i + 1] += 4 * h / 3; w[i + 2] += h / 3
        out[k] = (float(np.dot(w, y)), float(math.sqrt(np.dot(w ** 2, e ** 2))))
    return out

def poids_simpson(n, k, h, pas=1):
    w = np.zeros(n)
    for i in range(0, k, 2 * pas):
        w[i] += pas * h / 3; w[i + pas] += 4 * pas * h / 3; w[i + 2 * pas] += pas * h / 3
    return w

def simpson_pas_double(y, h, k, e=None):
    """Simpson au pas 2h jusqu'a b_k (k multiple de 4) ; avec e : aussi l'erreur statistique de l'ECART S(h) - S(2h)."""
    if k % 4 != 0:
        return float('nan') if e is None else (float('nan'), float('nan'))
    w2 = poids_simpson(len(y), k, h, 2)
    v = float(np.dot(w2, y))
    if e is None:
        return v
    dw = poids_simpson(len(y), k, h, 1) - w2
    return v, float(math.sqrt(np.dot(dw ** 2, e ** 2)))

def main():
    mutant = '--mutant-decalage' in sys.argv
    G = [charger(d, mutant) for d in sys.argv[1:6]]
    b = G[0]['b']; h = b[1] - b[0]
    for g in G:
        assert np.allclose(g['b'], b), 'grilles differentes'
        assert np.allclose(np.diff(g['b']), h), 'grille non uniforme'
    res = {'grille': list(map(float, b)), 'geometries': [g['cotes'] for g in G], 'points': []}
    # integrandes : -(...) ; Delta = -int (16 V_G0 - V_G4) ; p_j = -int (2 V_Gj - V_Gj+1)
    integ = {}
    integ['Delta'] = (-(16 * G[0]['v0'] - G[4]['v0']), np.sqrt((16 * G[0]['e0']) ** 2 + G[4]['e0'] ** 2))
    for j in range(4):
        integ['p%d' % j] = (-(2 * G[j]['v0'] - G[j + 1]['v0']), np.sqrt((2 * G[j]['e0']) ** 2 + G[j + 1]['e0'] ** 2))
    for j in range(5):
        integ['logR_G%d' % j] = (-G[j]['dv'], G[j]['edv'])
    cum = {k: simpson_cumul(y, e, h) for k, (y, e) in integ.items()}
    for k in sorted(cum['Delta']):
        bb = float(b[k]); t = t_sc(bb)
        pt = {'beta': bb, 't': t}
        for name in integ:
            v, e = cum[name][k]
            v2, ed = simpson_pas_double(integ[name][0], h, k, integ[name][1])
            pt[name] = [v, e, v2, ed]
        pt['somme_p'] = 8 * pt['p0'][0] + 4 * pt['p1'][0] + 2 * pt['p2'][0] + pt['p3'][0]
        dsc = 384 * t ** 4 + 1536 * t ** 8
        pt['Delta_SC'] = dsc
        pt['rapport_Delta'] = pt['Delta'][0] / dsc
        pt['err_rapport_Delta'] = pt['Delta'][1] / dsc
        pt['troncature_relative_a_priori'] = 20 * t ** 4
        lr = []
        for j, g in enumerate(G):
            # plan AJOUTE par le secteur 1 (protocole verifie au chargement)
            pl = g['plan']; a, c = int(pl[0]), int(pl[1])
            A = g['cotes'][a] * g['cotes'][c]
            lr.append(-2 * (g['vol'] / A) * t ** A)
        pt['logR_SC'] = lr
        res['points'].append(pt)
    # criteres du PREREG_ETALONNAGE (P1-P4, repliques)
    verdicts = []
    for pt in res['points']:
        bb = pt['beta']; t = pt['t']; tol = 20 * t ** 4
        d, e, d2, ed = pt['Delta']
        if abs(bb - 0.75) < 1e-9 or abs(bb - 1.0) < 1e-9:
            ok = abs(d / pt['Delta_SC'] - 1) <= tol + 3 * e / pt['Delta_SC']
            verdicts.append(('P1', bb, 'VERT' if ok else 'ROUGE', d / pt['Delta_SC'], e / pt['Delta_SC'], tol))
            for j in range(3):
                v, ee, _, _ = pt['logR_G%d' % j]; sc = pt['logR_SC'][j]
                okr = abs(v / sc - 1) <= tol + 3 * ee / abs(sc)
                verdicts.append(('P2_G%d' % j, bb, 'VERT' if okr else 'ROUGE', v / sc, ee / abs(sc), tol))
        if abs(bb - 0.25) < 1e-9 or abs(bb - 0.5) < 1e-9:
            ok = abs(d - pt['Delta_SC']) <= 3 * e
            verdicts.append(('P1_abs', bb, 'VERT' if ok else 'ROUGE', d - pt['Delta_SC'], e, 0.0))
        if not math.isnan(d2):
            # regle gelee : |S(h) - S(2h)| < sigma/3 ; « err » affiche ici l'erreur statistique propre de l'ecart (ajout E1)
            verdicts.append(('P3', bb, 'VERT' if abs(d - d2) < e / 3 else 'ROUGE', d - d2, ed, e / 3))
        # identite algebrique (controle de l'arithmetique), pas une validation : statut EXACTE / FAUSSE, jamais VERT
        verdicts.append(('P4_identite_algebrique', bb, 'EXACTE' if abs(pt['somme_p'] - d) < 1e-9 * max(1, abs(d)) else 'FAUSSE',
                         pt['somme_p'] - d, 0, 0))
    chi2 = [c for g in G for c in g['chi2']]
    res['verdicts'] = verdicts
    res['chi2_repliques_max'] = max(chi2) if chi2 else None
    res['chi2_repliques_moyen'] = float(np.mean(chi2)) if chi2 else None
    res['mutant_decalage'] = mutant
    res['P4_nature'] = ("identité algébrique 8p0 + 4p1 + 2p2 + p3 = Delta sur les mêmes données et les mêmes poids : contrôle de "
                        "l'arithmétique, sans valeur de validation (vraie aussi pour des données fausses, p. ex. un décalage "
                        "commun des β)")
    # le mutant n'est juge que par les references independantes (couplage fort) : P1, P1_abs, P2
    res['mutant_juge_par'] = ['P1', 'P1_abs', 'P2']
    res['mutant_rouge'] = (any(v[2] == 'ROUGE' for v in verdicts if v[0] in ('P1', 'P1_abs') or v[0].startswith('P2_'))
                           if mutant else None)
    print(json.dumps(res, indent=1))
    for v in verdicts:
        print('# %s beta=%.3f %s valeur=%.5g err=%.3g tol=%.3g' % v, file=sys.stderr)
    print('# chi2 repliques : max %.2f moyen %.2f' % (res['chi2_repliques_max'] or float('nan'), res['chi2_repliques_moyen'] or float('nan')), file=sys.stderr)
    for pt in res['points']:
        print('# beta=%.3f t=%.4f Delta=%.5f(%.5f) [pas double %.5f] SC=%.5f rapport=%.4f(%.4f) tronc=%.3f | logR_G0=%.6f(%.6f) SC=%.6f' % (
            pt['beta'], pt['t'], pt['Delta'][0], pt['Delta'][1], pt['Delta'][2], pt['Delta_SC'], pt['rapport_Delta'],
            pt['err_rapport_Delta'], pt['troncature_relative_a_priori'], pt['logR_G0'][0], pt['logR_G0'][1], pt['logR_SC'][0]),
            file=sys.stderr)

if __name__ == '__main__':
    main()
