"""HESS-KERDU, X = kerdu, L = 6 : (i) à U = 1, eps' = 0 (<= 1e-10) ; (ii) DŪ·x = 0 pour des vecteurs projetés sur X par `kvu hess espace --projeter`
(20 vecteurs gaussiens, Haar graine 26102613071 et froid) : ||DŪ x||/||x|| <= 1e-10 ; (iii) dimension de X de `kvu` = dimension du script indépendant
`ref_dim_kerdu.py` (SVD dense numpy, JSON haché AVANT le run, chemin en argument 2).
Contrôles supplémentaires (PREVOL) : (ii') les vecteurs projetés (20 gaussiens + 200 vecteurs de base tirés au hasard) sont aussi testés avec le Jacobien ANALYTIQUE indépendant
(kerdu_ref.py, forme fermée de WP11) et orthogonaux aux générateurs de la jauge résiduelle de ce même module : ||J_py x|| / ||x|| <= 1e-10 et ||G_py x|| / ||x|| <= 1e-10."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O, kerdu_ref as KR

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
ref = json.load(open(sys.argv[2]))
sc = O.SCRATCH + '/hess_kerdu'; os.makedirs(sc, exist_ok=True)
L = 6; n = 12 * L ** 4
froid = np.zeros((4, L, L, L, L, 2, 2), complex); froid[..., 0, 0] = 1; froid[..., 1, 1] = 1
fonds = {'froid': froid, 'haar': O.haar_c16(L, 26102613071)}
tout = True
for nom, C in fonds.items():
    cfg = sc + '/%s.npy' % nom
    O.ecrire_npy(cfg, C)
    rng = np.random.default_rng(26102613072)
    Z = rng.standard_normal((20, n))
    idx = rng.choice(n, size=200, replace=False)
    Eb = np.zeros((200, n)); Eb[np.arange(200), idx] = 1.0
    Z = np.vstack([Z, Eb])                      # 20 gaussiens puis 200 vecteurs de base
    NZ = len(Z)
    O.ecrire_npy(sc + '/Z.npy', Z)
    O.kvu(['hess', 'espace', '--config', cfg, '--espace', 'kerdu', '--projeter', sc + '/Z.npy', '--sortie', sc + '/PZ_%s.npy' % nom, '--rapport', sc + '/espace_%s.json' % nom, '--fils', O.FILS])
    R = json.load(open(sc + '/espace_%s.json' % nom))
    PZ = np.load(sc + '/PZ_%s.npy' % nom)
    O.kvu(['hess', 'jvp', '--config', cfg, '--vecteurs', sc + '/PZ_%s.npy' % nom, '--sortie', sc + '/JPZ_%s.npy' % nom, '--fils', O.FILS])
    JP = np.load(sc + '/JPZ_%s.npy' % nom)
    O.fin(PZ, JP)
    assert all(np.linalg.norm(PZ[k]) > 1e-8 for k in range(NZ)), 'vecteur projeté nul : ratio indéfini'
    e2 = max(np.linalg.norm(JP[k]) / np.linalg.norm(PZ[k]) for k in range(NZ))
    Jpy, Gpy = KR.construire(C)
    e2p = max(np.linalg.norm(Jpy @ PZ[k]) / np.linalg.norm(PZ[k]) for k in range(NZ))
    e2g = max(np.linalg.norm(Gpy @ PZ[k]) / np.linalg.norm(PZ[k]) for k in range(NZ))
    # idempotence (information) : Π(Π z) = Π z
    O.fin(e2p, e2g)
    O.ecrire_npy(sc + '/PZ_in.npy', PZ)
    O.kvu(['hess', 'espace', '--config', cfg, '--espace', 'kerdu', '--projeter', sc + '/PZ_in.npy', '--sortie', sc + '/PPZ_%s.npy' % nom, '--rapport', sc + '/espace2_%s.json' % nom, '--fils', O.FILS])
    PPZ = np.load(sc + '/PPZ_%s.npy' % nom)
    idem = max(np.linalg.norm(PPZ[k] - PZ[k]) / np.linalg.norm(PZ[k]) for k in range(NZ))
    ok2 = e2 <= 1e-10 and e2p <= 1e-10 and e2g <= 1e-10; tout &= ok2
    O.verdict(f, 'HESS-KERDU-ii-%s' % nom, ok2, 'max ||DU x||/||x|| = %.3e (DU dual de kvu) ; %.3e (J analytique independant) ; ||G_jauge x||/||x|| = %.3e (generateurs independants) ; seuil 1e-10 ; %d vecteurs projetes (20 gaussiens + 200 de base) ; idempotence du projecteur %.2e (info)' % (e2, e2p, e2g, NZ, idem))
    ok3 = (R['dim'] == ref[nom]['dim']); tout &= ok3
    O.verdict(f, 'HESS-KERDU-iii-%s' % nom, ok3, 'dim kvu = %s ; dim independante = %s (rang %s sur %s lignes ; marge des sigma %s) ; Cholesky de la jauge : %s' %
              (R['dim'], ref[nom]['dim'], ref[nom]['rang'], ref[nom]['lignes'], ref[nom]['sigma_rel_derniere_gardee_et_premiere_ecartee'], R['jauge_residuelle']['gram']))
# (i) U = 1
rj = sc + '/lmin_froid.json'
O.kvu(['hess', 'lmin', '--config', sc + '/froid.npy', '--espace', 'kerdu', '--coord', 'gauche', '--reference', 'q0', '--beta', 1, '--graine', 1, '--tol', '1e-10', '--sortie', rj, '--fils', O.FILS])
R = O.rapport_lmin(rj); e = abs(R['eps_prime']); ok = e <= 1e-10; tout &= ok
O.verdict(f, 'HESS-KERDU-i', ok, "|eps'| a U = 1 = %.3e (seuil 1e-10) ; dim = %s ; lambda_min(Q|X)/(6 beta) = %s" % (e, R['dim'], R.get('q_restreinte', {}).get('rapport_borne')))
O.verdict(f, 'HESS-KERDU', tout, 'L = 6')
sys.exit(0 if tout else 1)
