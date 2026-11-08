"""HESS-LMIN : `kvu hess lmin` (LOBPCG, tol 1e-10, graine fixée) contre scipy.linalg.eigh(P^T H P, P^T Q P) dense, 4^4,
espaces libre, kerB (coordonnées additives, référence q0) et tcov (gauche, référence qcov), 3 fonds.
Critères : |dlambda_min| <= 1e-8 max(1,|lambda_min|) ; residu projete ||P^T(H v - lam Q v)|| / (||P^T H P||_2 ||P^T v||) <= 1e-9 (v^T Q v = 1),
calculé ICI avec les matrices denses de lat.py (indépendant de kvu) sur le vecteur rendu par kvu."""
import os, sys, json, numpy as np, scipy.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
lat = O.charger_lat()
sc = O.SCRATCH + '/hess_lmin'; os.makedirs(sc, exist_ok=True)
L = 4; geo = lat.Geo(L); n = 12 * L ** 4
K = (geo.R.T @ geo.R).toarray()

def fonds():
    Ch = O.haar_c16(L, 26102613041)
    Ul = O.champ_vers_liens_lat(Ch)
    yield 'haar', 2.3, Ul, lat.su2_log(Ul)
    for nom, beta, graine in (('gauss_b8', 8.0, 26102613042), ('gauss_b64', 64.0, 26102613043)):
        gs = lat.Gauss(geo, 0.25)
        A = gs.draw(np.random.default_rng(graine), 1.0 / beta, 3, coarse=False)
        yield nom, beta, lat.su2_exp(A), A

ok_tout = True
base_cache = {}
for nom, beta, Ul, A in fonds():
    cfg = sc + '/%s.npy' % nom
    O.ecrire_npy(cfg, O.liens_lat_vers_champ(Ul, L))
    for espace in ('libre', 'kerB', 'tcov'):
        if espace == 'tcov':
            P = lat.basis_cov(geo, Ul)
            H, _ = lat.hess_su2_left(geo, Ul); H = H * beta
            Q = lat.qcov_su2(geo, Ul) * beta
            coord, ref = 'gauche', 'qcov'
        else:
            P, _ = lat.basis_fix(geo, fluct=(espace == 'kerB'), ncol=3)
            H, _ = lat.hess_su2_additive(geo, A); H = H * beta
            Q = 0.25 * beta * np.kron(K, np.eye(3))
            coord, ref = 'additive', 'q0'
        Hr = P.T @ H @ P; Qr = P.T @ Q @ P
        Hr = (Hr + Hr.T) / 2; Qr = (Qr + Qr.T) / 2
        w = sla.eigh(Hr, Qr, eigvals_only=True)
        lam_d = w[0]
        vec = sc + '/%s_%s_v.npy' % (nom, espace)
        rj = sc + '/%s_%s.json' % (nom, espace)
        O.kvu(['hess', 'lmin', '--config', cfg, '--espace', espace, '--coord', coord, '--reference', ref, '--beta', beta, '--graine', 1,
               '--tol', '1e-10', '--vecteur-sortie', vec, '--sortie', rj, '--fils', O.FILS])
        R = O.rapport_lmin(rj); lam = R['lambda_min']
        v = np.load(vec)
        vqv_brut = float(v @ (Q @ v))
        v = v / np.sqrt(vqv_brut)
        r = P.T @ (H @ v - lam * (Q @ v))
        nH = np.linalg.norm(Hr, 2)
        nPv = np.linalg.norm(P.T @ v)
        res = np.linalg.norm(r) / (nH * nPv)
        # le vecteur doit être dans X : ||(1 - P P^T) v||
        hors = np.linalg.norm(v - P @ (P.T @ v)) / np.linalg.norm(v)
        dl = abs(lam - lam_d); seuil = 1e-8 * max(1.0, abs(lam_d))
        assert all(np.isfinite([lam, lam_d, res, hors, vqv_brut]))
        ok = dl <= seuil and res <= 1e-9 and hors <= 1e-9 and abs(vqv_brut - 1) <= 1e-9
        ok_tout &= ok
        dimok = (R['dim'] == P.shape[1])
        ok_tout &= dimok
        O.verdict(f, 'HESS-LMIN-%s-%s' % (espace, nom), ok and dimok,
                  'lam_kvu=%.12f lam_dense=%.12f |d|=%.2e (seuil %.1e) ; residu projete=%.2e (seuil 1e-9) ; hors-X=%.1e (<=1e-9) ; |vQv-1|=%.1e (<=1e-9) ; dim kvu=%s dense=%d ; iter=%s' %
                  (lam, lam_d, dl, seuil, res, hors, abs(vqv_brut - 1), R['dim'], P.shape[1], R['solveur']['iterations']))
O.verdict(f, 'HESS-LMIN', ok_tout, '3 espaces x 3 fonds, 4^4')
sys.exit(0 if ok_tout else 1)
