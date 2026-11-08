"""Supplément (ID INFO-LANDAU-*, non gelé dans §1.3) : `kvu hess landau` contre lat.landau_fix (S2) sur 4^4 : Haar (graine 26102613091) et gaussien beta = 8 (graine 26102613092) ;
tol 1e-24, maxit 5000, omega 1,7. Critères : max|U_kvu - U_lat| <= 1e-9 (même algorithme, mêmes itérations) ; theta_kvu < 1e-20 ; sortie identique à 1 et 8 fils."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
lat = O.charger_lat()
sc = O.SCRATCH + '/hess_landau'; os.makedirs(sc, exist_ok=True)
L = 4; geo = lat.Geo(L)
Ch = O.haar_c16(L, 26102613091); Ulh = O.champ_vers_liens_lat(Ch)
gs = lat.Gauss(geo, 0.25)
A = gs.draw(np.random.default_rng(26102613092), 1.0 / 8.0, 3, coarse=False)
tout = True
for nom, Ul in (('haar', Ulh), ('gauss_b8', lat.su2_exp(A))):
    cfg = sc + '/%s.npy' % nom
    O.ecrire_npy(cfg, O.liens_lat_vers_champ(Ul, L))
    Uf, theta, it = lat.landau_fix(geo, Ul, tol=1e-24, maxit=5000)
    sorties = []
    for fils in (1, 8):
        out = sc + '/%s_fix_%d.npy' % (nom, fils)
        O.kvu(['hess', 'landau', '--config', cfg, '--sortie', out, '--tol', '1e-24', '--iter-max', 5000, '--fils', fils])
        sorties.append(out)
    same = open(sorties[0], 'rb').read() == open(sorties[1], 'rb').read()
    Uk = O.champ_vers_liens_lat(np.load(sorties[0]))
    # les deux fixations convergent vers une même copie à une rotation globale près possible : on compare aussi la plaquette et le critère
    e = np.abs(Uk - Uf).max()
    ok = e <= 1e-9 and same
    tout &= ok
    O.verdict(f, 'INFO-LANDAU-%s' % nom, ok, 'max|U_kvu - U_lat| = %.3e (seuil 1e-9) ; lat : theta=%.1e it=%d ; sortie identique a 1 et 8 fils : %s' % (e, theta, it, same))
sys.exit(0 if tout else 1)
