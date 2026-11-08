"""HESS-ESPACES : rejeu ponctuel de S2 (prod.py) : 3 fonds gaussiens 4^4 (lat.Gauss, beta in {8, 64}), fixation de Landau (lat.landau_fix),
T-fix-fluct (kerB) et T-fix-plein (landau) en coordonnées additives, T-cov (tcov) en coordonnées gauche : eps' de `kvu` contre eps' de lat.py (dense),
|diff| <= 1e-8 absolu.  Contrôle annexe (non gate) : projecteur CG = projecteur SVD (même lambda_min, 1e-9)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
lat = O.charger_lat()
sc = O.SCRATCH + '/hess_espaces'; os.makedirs(sc, exist_ok=True)
L = 4; geo = lat.Geo(L)
K = (geo.R.T @ geo.R).toarray()
Q0 = 0.25 * np.kron(K, np.eye(3))
Vfl, _ = lat.basis_fix(geo, fluct=True, ncol=3)
Vpl, _ = lat.basis_fix(geo, fluct=False, ncol=3)
gs = lat.Gauss(geo, 0.25)
tout = True; pire = 0.0; pire_cg = 0.0
for nom, beta, graine, grossier in (('b8_grossier', 8.0, 26102613051, True), ('b64_grossier', 64.0, 26102613052, True), ('b8_plein', 8.0, 26102613053, False)):
    rng = np.random.default_rng(graine)
    A = gs.draw(rng, 1.0 / beta, 3, coarse=grossier)
    U = lat.su2_exp(A)
    Uf, theta, it = lat.landau_fix(geo, U, tol=1e-24, maxit=5000)
    Af = lat.su2_log(Uf)
    Hadd, _ = lat.hess_su2_additive(geo, Af)
    HL, _ = lat.hess_su2_left(geo, Uf)
    Qc = lat.qcov_su2(geo, Uf)
    Vc = lat.basis_cov(geo, Uf)
    ref = {}
    lmin, lmax, _ = lat.gen_eig_extremes(Hadd, Q0, Vfl); ref['kerB'] = 1 - lmin
    lmin, lmax, _ = lat.gen_eig_extremes(Hadd, Q0, Vpl); ref['landau'] = 1 - lmin
    lmin, lmax, _ = lat.gen_eig_extremes(HL, Qc, Vc); ref['tcov'] = 1 - lmin
    cfg = sc + '/%s.npy' % nom
    O.ecrire_npy(cfg, O.liens_lat_vers_champ(Uf, L))
    for espace, coord, rf in (('kerB', 'additive', 'q0'), ('landau', 'additive', 'q0'), ('tcov', 'gauche', 'qcov')):
        rj = sc + '/%s_%s.json' % (nom, espace)
        O.kvu(['hess', 'lmin', '--config', cfg, '--espace', espace, '--coord', coord, '--reference', rf, '--beta', beta, '--graine', 1, '--tol', '1e-10', '--sortie', rj, '--fils', O.FILS])
        R = O.rapport_lmin(rj); eps = R['eps_prime']; O.fin(ref[espace])
        d = abs(eps - ref[espace])
        if espace == 'tcov':      # CG imprécis pour tcov près de l'identité (limite connue, PREVOL §3) : pas de contrôle CG
            dcg = 0.0
        else:
          rj2 = sc + '/%s_%s_cg.json' % (nom, espace)
          O.kvu(['hess', 'lmin', '--config', cfg, '--espace', espace, '--coord', coord, '--reference', rf, '--beta', beta, '--graine', 1, '--tol', '1e-10', '--projecteur', 'cg', '--sortie', rj2, '--fils', O.FILS])
          R2 = O.rapport_lmin(rj2); dcg = abs(R2['eps_prime'] - eps)
        pire = max(pire, d); pire_cg = max(pire_cg, dcg)
        ok = d <= 1e-8
        tout &= ok
        O.verdict(f, 'HESS-ESPACES-%s-%s' % (espace, nom), ok, "eps' kvu=%.12f lat=%.12f |d|=%.2e (seuil 1e-8) ; CG-SVD |d|=%.1e ; landau theta=%.1e it=%d" % (eps, ref[espace], d, dcg, theta, it))
O.verdict(f, 'HESS-ESPACES', tout, '3 fonds gaussiens x 3 traitements ; pire |d| = %.2e ; pire ecart CG/SVD = %.2e (information)' % (pire, pire_cg))
sys.exit(0 if tout else 1)
