#!/usr/bin/env python3
"""WP11 / FIB-DERIV (formule générale des chemins) : jouets à chemins riches (docs/FIBRE_COAIRE.md §3.3).

Jouet B : 2 liens fins, 1 lien grossier ; chemin de référence G0 = (1+, 2+) ; famille de 4 chemins avec liens à rebours et
          répétitions : (1+,2+), (1+,1+,2+,1-), (2-,1+,2+,2+), (1+,2-,2+,2+) ; poids 0,4 / 0,25 / 0,2 / 0,15.
Jouet C : 3 liens fins, 2 liens grossiers partageant des liens fins (J de taille 6 x 9).
Gate : |J analytique - DF centrées (références génériques, 50 chiffres)| <= 1e-8, sur 12 configurations par jouet
       (amplitudes 0,3 / 0,8 / Haar). On rapporte aussi det(J J^T) (positif : rang plein) et l'invariance de det(J J^T)
       par trivialisation à droite (J_droite = blocs R(U_l)^T, cf. §2.3).
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
import su2_mp as S

mp.mp.dps = 50
ok = True
def ligne(etat, ident, detail):
    global ok
    if etat == 'FAIL':
        ok = False
    print(f'{etat} {ident} {detail}', flush=True)

rng = random.Random(26102614)
JOUET_B = dict(liens=[1, 2], cartes=[((
    [(1, 1), (2, 1)],
    [[(1, 1), (2, 1)], [(1, 1), (1, 1), (2, 1), (1, -1)], [(2, -1), (1, 1), (2, 1), (2, 1)], [(1, 1), (2, -1), (2, 1), (2, 1)]],
    [mp.mpf('0.4'), mp.mpf('0.25'), mp.mpf('0.2'), mp.mpf('0.15')]))])
JOUET_C = dict(liens=[1, 2, 3], cartes=[
    ([(1, 1), (2, 1)], [[(1, 1), (2, 1)], [(3, 1), (2, -1), (1, 1), (2, 1), (2, 1)], [(1, 1), (3, -1), (3, 1), (2, 1)]],
     [mp.mpf('0.5'), mp.mpf('0.3'), mp.mpf('0.2')]),
    ([(2, 1), (3, 1)], [[(2, 1), (3, 1)], [(2, 1), (1, -1), (1, 1), (3, 1)], [(1, -1), (2, 1), (3, 1), (1, 1)]],
     [mp.mpf('0.6'), mp.mpf('0.25'), mp.mpf('0.15')])])

for nom, jouet in (('B', JOUET_B), ('C', JOUET_C)):
    emax = mp.mpf(0); dets = []; edroite = mp.mpf(0); n = 0
    for amp in ('0.3', '0.8', None):
        for k in range(4):
            U = {l: S.su2_aleatoire(rng, amp) for l in jouet['liens']}
            Jlig = []
            for ref, chem, poids in jouet['cartes']:
                Ja = S.jac_carte(U, ref, chem, poids, jouet['liens'])
                Jd = S.jac_dfc(U, ref, chem, poids, jouet['liens'])
                emax = max(emax, max(S.max_abs(Ja[l] - Jd[l]) for l in jouet['liens']))
                Jlig.append(Ja)
            nr, nc = 3 * len(jouet['cartes']), 3 * len(jouet['liens'])
            J = mp.matrix(nr, nc); Jr = mp.matrix(nr, nc)
            for c, Ja in enumerate(Jlig):
                for j, l in enumerate(jouet['liens']):
                    Rl = S.Rad(U[l])
                    for a in range(3):
                        for b in range(3):
                            J[3 * c + a, 3 * j + b] = Ja[l][a, b]
                    # trivialisation à droite sur la source : U_l -> U_l exp(i e'.s) = exp(i (R(U_l) e').s) U_l  => colonnes J R(U_l)
                    B = Ja[l] * Rl
                    for a in range(3):
                        for b in range(3):
                            Jr[3 * c + a, 3 * j + b] = B[a, b]
            d = mp.det(J * J.T); dr = mp.det(Jr * Jr.T)
            dets.append(d); edroite = max(edroite, abs(dr / d - 1)); n += 1
    ligne('PASS' if emax < mp.mpf('1e-8') else 'FAIL', f'FIB-DERIV-CHEMINS-{nom}', f'jouet {nom} : max |J analytique - DF centrées| = {mp.nstr(emax, 3)} sur {n} configurations (amplitudes 0,3 / 0,8 / Haar ; seuil 1e-8)')
    ligne('PASS' if edroite < mp.mpf('1e-40') else 'FAIL', f'FIB-TRIVIALISATION-{nom}', f'jouet {nom} : det(J J^T) identique en trivialisation à droite (source) : max écart relatif {mp.nstr(edroite, 3)} ; det(J J^T) dans [{mp.nstr(min(dets), 4)}, {mp.nstr(max(dets), 4)}]')
print('BILAN FIB-CHEMINS', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
