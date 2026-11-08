#!/usr/bin/env python3
"""WP11 / FIB-JAC-BALABAN : la VRAIE carte de Bałaban (L_b = 3, 10 473 chemins par lien grossier), en F64.

(a) FIB-CARTE-REG : l'implémentation indépendante de balaban_np.py redonne les Ubar des fixtures de WP6
    (crates/kvu/validation/rg/fixtures, Ubar de g_nl.py ; 3 configurations 9^4, amplitude 0,6), max |dq| <= 1e-12 ;
(b) FIB-DERIV-BALABAN-DF : jacobien en forme fermée = différences finies (centrées, extrapolation de Richardson h, h/2) sur
    3 liens grossiers de chacune des 3 fixtures, 48 liens fins par lien grossier (les 3 de la référence + 45 tirés au hasard
    parmi les liens touchés) ; seuil 1e-8 (absolu, blocs 3x3) ;
(c) FIB-DOMAINE-FIXTURES (INFO) : th max et fraction th > pi/2 des 10 473 logs.
Usage : t4_balaban.py <dossier des fixtures>
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import balaban_np as B

ok = True
def ligne(etat, ident, detail):
    global ok
    if etat == 'FAIL':
        ok = False
    print(f'{etat} {ident} {detail}', flush=True)

fx = sys.argv[1]
man = json.load(open(os.path.join(fx, 'manifeste.json')))
bloc = B.Bloc(9)
ligne('PASS' if bloc.npaths == 10473 else 'FAIL', 'FIB-CARTE-CHEMINS', f'{bloc.npaths} chemins par lien grossier (attendu 10 473), longueur max {bloc.lmax} (+ 3 de la référence = 14)')
rng = np.random.default_rng(26102615)
emax = 0.0; ejac = 0.0; nb = 0; info = []
for graine in (7, 11, 13):
    fq = os.path.join(fx, f'cfg_s{graine}_q64.npy'); fu = os.path.join(fx, f'ubar_s{graine}_q64.npy')
    for f in (fq, fu):
        h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
        if h != man['fichiers'][os.path.basename(f)]:
            ligne('FAIL', 'FIB-FIXTURES-SHA', f'{os.path.basename(f)} : sha256 {h[:16]}... != manifeste')
    q = np.load(fq).reshape(-1, 4); ub = np.load(fu)
    thmax = 0.0; nhors = 0; ntot = 0
    for mu in range(4):
        for z in np.ndindex(3, 3, 3, 3):
            tab = bloc.tables(z, mu)
            u, th = bloc.ubar(q, z, mu, tab)
            emax = max(emax, float(np.abs(u - ub[(mu,) + z]).max()))
            thmax = max(thmax, float(th.max())); nhors += int((th > np.pi / 2).sum()); ntot += th.size
    info.append(f's{graine} : th max {thmax:.3f}, fraction th > pi/2 = {nhors / ntot:.2e}')
    # jacobien contre différences finies
    for (z, mu) in [((0, 0, 0, 0), 0), ((1, 2, 0, 1), 2), ((2, 1, 2, 0), 3)]:
        tab = bloc.tables(z, mu)
        liens, J, _ = bloc.jac(q, z, mu, tab)
        Iref = tab[3]
        autres = rng.choice([l for l in liens if l not in set(Iref)], size=45, replace=False)
        choix = list(Iref) + list(autres)
        u0, _ = bloc.ubar(q, z, mu, tab)
        def delta(qq):
            u, _ = bloc.ubar(qq, z, mu, tab)
            w, _ = B.qlog(B.qmul(u, B.qdag(u0)))
            return w
        for l in choix:
            jl = J[np.searchsorted(liens, l)]
            Jd = np.zeros((3, 3))
            for a in range(3):
                def dc(hh):
                    e = np.zeros(3); e[a] = hh
                    qp = q.copy(); qp[l] = B.qmul(B.qexp(e), q[l])
                    qm = q.copy(); qm[l] = B.qmul(B.qexp(-e), q[l])
                    return (delta(qp) - delta(qm)) / (2 * hh)
                h = 1e-3
                Jd[:, a] = (4 * dc(h / 2) - dc(h)) / 3          # Richardson : O(h^4)
            ejac = max(ejac, float(np.abs(Jd - jl).max())); nb += 1
ligne('PASS' if emax <= 1e-12 else 'FAIL', 'FIB-CARTE-REG', f'Ubar (balaban_np, énumération indépendante) = fixtures WP6 (g_nl.py) : max |dq| = {emax:.2e} sur 3 x 324 liens grossiers (seuil 1e-12)')
ligne('PASS' if ejac <= 1e-8 else 'FAIL', 'FIB-DERIV-BALABAN-DF', f'jacobien forme fermée - DF (Richardson, h = 1e-3) : max |dJ| = {ejac:.2e} sur {nb} blocs 3x3 (3 fixtures x 3 liens grossiers x 48 liens fins ; seuil 1e-8)')
ligne('INFO', 'FIB-DOMAINE-FIXTURES', ' ; '.join(info))
print('BILAN FIB-JAC-BALABAN', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
