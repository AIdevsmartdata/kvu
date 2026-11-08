"""HESS-EXACT : 4^4, base s2 ; H assemblée colonne par colonne par `kvu hess hv` contre lat.py
(a) coordonnées gauche : lat.hess_su2_left x beta ; (b) additives : lat.hess_su2_additive x beta.
Fonds : Haar graine 26102613011 ; gaussien lat.Gauss beta = 8, graine 26102613012. Critère max|dH|/max|H| <= 1e-10."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
lat = O.charger_lat()
sc = O.SCRATCH + '/hess_exact'; os.makedirs(sc, exist_ok=True)
L = 4; geo = lat.Geo(L); n = 12 * L ** 4
np.save(sc + '/id.npy', np.eye(n))

def fonds():
    Ch = O.haar_c16(L, 26102613011)
    Ul = O.champ_vers_liens_lat(Ch)
    yield 'haar', 2.3, Ul, lat.su2_log(Ul)
    gs = lat.Gauss(geo, 0.25)
    rng = np.random.default_rng(26102613012)
    beta = 8.0
    A = gs.draw(rng, 1.0 / beta, 3, coarse=False)
    yield 'gauss_b8', beta, lat.su2_exp(A), A

ok_tout = True
for nom, beta, Ul, A in fonds():
    cfg = sc + '/%s.npy' % nom
    O.ecrire_npy(cfg, O.liens_lat_vers_champ(Ul, L))
    for coord in ('gauche', 'additive'):
        O.kvu(['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/id.npy', '--sortie', sc + '/%s_%s.npy' % (nom, coord),
               '--base', 's2', '--coord', coord, '--beta', beta, '--fils', O.FILS])
        Hk = np.load(sc + '/%s_%s.npy' % (nom, coord)).T   # colonne i = H e_i
        if coord == 'gauche':
            Href, _ = lat.hess_su2_left(geo, Ul)
        else:
            Href, _ = lat.hess_su2_additive(geo, A)
        Href = Href * beta
        e = np.abs(Hk - Href).max() / np.abs(Href).max()
        sym = np.abs(Hk - Hk.T).max() / np.abs(Hk).max()
        ok = e <= 1e-10
        ok_tout &= ok
        O.verdict(f, 'HESS-EXACT-%s-%s' % ('a' if coord == 'gauche' else 'b', nom), ok, 'max|dH|/max|H| = %.3e (seuil 1e-10) ; symetrie de H assemblee %.3e ; max|H| = %.4f' % (e, sym, np.abs(Href).max()))
O.verdict(f, 'HESS-EXACT', ok_tout, '(a) gauche et (b) additives, 2 fonds')
sys.exit(0 if ok_tout else 1)
