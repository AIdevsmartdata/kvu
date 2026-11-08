#!/usr/bin/env python3
"""WP11 / FIB-JOUET : bloc jouet à DEUX liens, Ubar = U1 exp(t log(U1^-1 U2)) (docs/FIBRE_COAIRE.md §4).

(a) jacobien analytique (formule générale des chemins, §3.3) = différences finies centrées (références génériques, 50 chiffres),
    10 configurations de Haar x t in {0,5 ; 0,3 ; 0,8} : seuil de la gate FIB-DERIV 1e-8 (atteint ~1e-20) ;
(b) forme fermée du jouet J = [1 - A, A], A = t R1 Eg(t w) Lg(w) R1^T, et det(J J^T) en th, t ;
(c) densité riemannienne de la fibre (différences finies du plongement w -> (V e^{-i t w.s}, V e^{i (1-t) w.s})) = forme fermée ;
(d) COAIRE : sqrt(det g) det(J J^T)^{-1/2} = sin^2 th / th^2 = densité de Haar de W en coordonnées exponentielles (exact, SymPy + 60 points) ;
(e) argument direct indépendant (invariance de Haar) contrôlé par Monte Carlo numpy (INFO) ;
(f) valeurs gelées (mpmath 30 chiffres, deux canaux : Bessel et quadrature) pour la loi conditionnelle et le MUTANT sans jacobien ;
(g) jouet U(1).
Sortie : lignes PASS/FAIL/INFO, valeurs gelées dans valeurs_jouet.json ; code retour 0 ssi tout PASS.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SORTIES = os.environ.get('FIBRE_SORTIES', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sorties'))
os.makedirs(SORTIES, exist_ok=True)
import sympy as sp
import mpmath as mp
import numpy as np
import su2_mp as S

ok = True
def ligne(etat, ident, detail):
    global ok
    if etat == 'FAIL':
        ok = False
    print(f'{etat} {ident} {detail}', flush=True)

def carte_jouet(t):
    return [(1, 1)], [[(1, 1)], [(2, 1)]], [1 - t, t]

# ---------------------------------------------------------------- (a), (b) jacobien
mp.mp.dps = 50
rng = random.Random(26102612)
eA = mp.mpf(0); eB = mp.mpf(0); eD = mp.mpf(0); n_cfg = 0
for t in (mp.mpf(1) / 2, mp.mpf(3) / 10, mp.mpf(8) / 10):
    ref, chem, poids = carte_jouet(t)
    for k in range(10):
        U = {1: S.su2_aleatoire(rng), 2: S.su2_aleatoire(rng)}
        Ja = S.jac_carte(U, ref, chem, poids, [1, 2])
        Jd = S.jac_dfc(U, ref, chem, poids, [1, 2])
        eA = max(eA, max(S.max_abs(Ja[l] - Jd[l]) for l in (1, 2)))
        # forme fermée du jouet
        w = S.log_cf(S.dag(U[1]) * U[2]); th = S.norme(w)
        R1 = S.Rad(U[1])
        A = t * R1 * S.Eg([t * x for x in w]) * S.Lg(w) * R1.T
        eB = max(eB, S.max_abs(Ja[1] - (mp.eye(3) - A)), S.max_abs(Ja[2] - A))
        J = mp.matrix(3, 6)
        for i in range(3):
            for j in range(3):
                J[i, j] = Ja[1][i, j]; J[i, 3 + j] = Ja[2][i, j]
        dJ = mp.det(J * J.T)
        dcf = ((1 - t)**2 + t**2) * ((mp.sin(t * th)**2 + mp.sin((1 - t) * th)**2) / mp.sin(th)**2) ** 2
        eD = max(eD, abs(dJ / dcf - 1))
        n_cfg += 1
ligne('PASS' if eA < mp.mpf('1e-8') else 'FAIL', 'FIB-DERIV-JOUET-DF', f'jacobien analytique (formule des chemins) - DF centrées génériques : max |dJ| = {mp.nstr(eA, 3)} sur {n_cfg} configurations de Haar, t in {{1/2, 3/10, 8/10}} (seuil 1e-8)')
ligne('PASS' if eB < mp.mpf('1e-40') else 'FAIL', 'FIB-JOUET-J-FERME', f'J = [1 - A, A], A = t R1 Eg(t w) Lg(w) R1^T : max écart {mp.nstr(eB, 3)}')
ligne('PASS' if eD < mp.mpf('1e-40') else 'FAIL', 'FIB-JOUET-DETJJT', f'det(J J^T) = ((1-t)^2 + t^2) ((sin^2(t th) + sin^2((1-t) th))/sin^2 th)^2 : max écart relatif {mp.nstr(eD, 3)}')

# ---------------------------------------------------------------- (c), (d) métrique de la fibre et coaire
eG = mp.mpf(0); eC = mp.mpf(0); h = mp.mpf('1e-20'); n_pt = 0
for t in (mp.mpf(1) / 2, mp.mpf(3) / 10):
    for k in range(30):
        V = S.su2_aleatoire(rng)
        thk = mp.mpf(rng.uniform(0.02, 3.1))
        n = [mp.mpf(rng.gauss(0, 1)) for _ in range(3)]; nn = S.norme(n)
        w = [thk * x / nn for x in n]
        def plonge(w_):
            return V * S.expm_gen([-t * x for x in w_]), V * S.expm_gen([(1 - t) * x for x in w_])
        P0 = plonge(w)
        # contrôle : le point est sur la fibre
        Ub = S.carte({1: P0[0], 2: P0[1]}, *carte_jouet(t), exp=S.expm_gen, log=S.logm_gen)[0]
        assert S.max_abs(Ub - V) < mp.mpf('1e-40')
        tang = []
        for i in range(3):
            e = [mp.mpf(0)] * 3; e[i] = h
            Pp = plonge([a + b for a, b in zip(w, e)]); Pm = plonge([a - b for a, b in zip(w, e)])
            # vecteur tangent trivialisé à gauche, lien par lien (coordonnées orthonormées) : log(U(w+h) U(w)^-1)/h
            v = []
            for lk in range(2):
                dp = S.logm_gen(Pp[lk] * S.dag(P0[lk])); dm = S.logm_gen(Pm[lk] * S.dag(P0[lk]))
                v += [(a - b) / (2 * h) for a, b in zip(dp, dm)]
            tang.append(v)
        g = mp.matrix(3, 3)
        for i in range(3):
            for j in range(3):
                g[i, j] = sum(a * b for a, b in zip(tang[i], tang[j]))
        sg = mp.sqrt(mp.det(g))
        sg_cf = mp.sqrt(t**2 + (1 - t)**2) * (mp.sin(t * thk)**2 + mp.sin((1 - t) * thk)**2) / thk**2
        eG = max(eG, abs(sg / sg_cf - 1))
        dJ_cf = ((1 - t)**2 + t**2) * ((mp.sin(t * thk)**2 + mp.sin((1 - t) * thk)**2) / mp.sin(thk)**2) ** 2
        eC = max(eC, abs(sg / mp.sqrt(dJ_cf) / (mp.sin(thk)**2 / thk**2) - 1))
        n_pt += 1
ligne('PASS' if eG < mp.mpf('1e-15') else 'FAIL', 'FIB-JOUET-METRIQUE', f'sqrt(det g_fibre) (DF du plongement) = sqrt(t^2+(1-t)^2) (sin^2(t th)+sin^2((1-t) th))/th^2 : max écart relatif {mp.nstr(eG, 3)} sur {n_pt} points')
ligne('PASS' if eC < mp.mpf('1e-15') else 'FAIL', 'FIB-COAIRE-JOUET-NUM', f'sqrt(det g) det(J J^T)^(-1/2) / (sin^2 th/th^2) - 1 : max {mp.nstr(eC, 3)} sur {n_pt} points (coaire = densité de Haar de W)')
# SymPy exact
th_, t_ = sp.symbols('theta t', positive=True)
al = sp.sin(t_ * th_) * sp.exp(sp.I * (1 - t_) * th_) / sp.sin(th_)
# identités utilisées : 1 - alpha = sin((1-t)th) e^{-i t th}/sin th ; |alpha|^2 + |1 - alpha|^2
r1 = sp.simplify(sp.expand_complex(sp.expand_trig((1 - al) - sp.sin((1 - t_) * th_) * sp.exp(-sp.I * t_ * th_) / sp.sin(th_))))
somme = sp.sin(t_ * th_)**2 / sp.sin(th_)**2 + sp.sin((1 - t_) * th_)**2 / sp.sin(th_)**2
detJJ = ((1 - t_)**2 + t_**2) * somme**2
sg_s = sp.sqrt(t_**2 + (1 - t_)**2) * (sp.sin(t_ * th_)**2 + sp.sin((1 - t_) * th_)**2) / th_**2
r2 = sp.simplify(sg_s / sp.sqrt(detJJ) - sp.sin(th_)**2 / th_**2)
# t = 1/2 : det(J J^T) = 1/(8 cos^4(th/2))
r3 = sp.simplify(sp.trigsimp(detJJ.subs(t_, sp.Rational(1, 2)) - 1 / (8 * sp.cos(th_ / 2)**4)))
ligne('PASS' if (r1 == 0 and r2 == 0 and r3 == 0) else 'FAIL', 'FIB-COAIRE-JOUET-SYMPY',
      f'1 - alpha = sin((1-t)th) e^(-i t th)/sin th : reste {r1} ; sqrt(det g)/sqrt(det JJ^T) - sin^2th/th^2 = {r2} ; t = 1/2 : det JJ^T - 1/(8 cos^4(th/2)) = {r3}')

# ---------------------------------------------------------------- (e) argument direct par Monte Carlo (INFO)
rs = np.random.default_rng(26102613)
def haar_np(n):
    q = rs.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
    return q
def qmat(q):
    a0, a1, a2, a3 = q.T
    M = np.empty((len(q), 2, 2), complex)
    M[:, 0, 0] = a0 + 1j * a3; M[:, 0, 1] = a2 + 1j * a1; M[:, 1, 0] = -a2 + 1j * a1; M[:, 1, 1] = a0 - 1j * a3
    return M
nmc = 200000
U1 = qmat(haar_np(nmc)); U2 = qmat(haar_np(nmc))
W = np.conj(np.swapaxes(U1, 1, 2)) @ U2
x = np.real(np.trace(W, axis1=1, axis2=2)) / 2
th = np.arccos(np.clip(x, -1, 1))
v = np.stack([np.real(np.trace(W @ s, axis1=1, axis2=2) / 2j) for s in (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))], axis=1)
nv = np.linalg.norm(v, axis=1); f = np.where(nv > 0, th / np.where(nv > 0, nv, 1), 1.0); w = f[:, None] * v
PA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
demi = np.cos(th / 2)[:, None, None] * np.eye(2) + (np.sin(th / 2) / np.where(th > 0, th, 1))[:, None, None] * 1j * np.einsum('na,aij->nij', w, PA)
Ub = U1 @ demi          # Ubar = U1 exp(1/2 log W)
xb = np.real(np.trace(Ub, axis1=1, axis2=2)) / 2
m1, m2 = x.mean(), (x**2).mean(); se1 = x.std() / np.sqrt(nmc); se2 = (x**2).std() / np.sqrt(nmc)
corr = np.corrcoef(x, xb)[0, 1]; corr2 = np.corrcoef(x**2, xb**2)[0, 1]
mb1, mb2 = xb.mean(), (xb**2).mean()
ligne('INFO', 'FIB-JOUET-MC', f'Haar x Haar, {nmc} tirages, t = 1/2 : <x_W> = {m1:.5f} +- {se1:.5f} (0) ; <x_W^2> = {m2:.5f} +- {se2:.5f} (1/4) ; <x_Ubar> = {mb1:.5f}, <x_Ubar^2> = {mb2:.5f} (0, 1/4) ; corr(x_W, x_Ubar) = {corr:+.4f}, corr(x_W^2, x_Ubar^2) = {corr2:+.4f} (indépendance : 0 +- {1/np.sqrt(nmc):.4f})')

# ---------------------------------------------------------------- (f) valeurs gelées
mp.mp.dps = 30
BRK = [0, mp.mpf('0.005'), mp.mpf('0.02'), mp.mpf('0.08'), mp.mpf('0.3'), mp.pi / 2, mp.pi]
def moments_radial(dens, beta):
    """<cos th>, <cos^2 th> pour la densité radiale dens(th) e^{beta cos th} sur [0, pi] (quadrature)"""
    b = mp.mpf(beta)
    # facteur e^{-beta} pour la stabilité à grand beta
    Z = mp.quad(lambda u: dens(u) * mp.exp(b * (mp.cos(u) - 1)), BRK)
    M1 = mp.quad(lambda u: dens(u) * mp.cos(u) * mp.exp(b * (mp.cos(u) - 1)), BRK)
    M2 = mp.quad(lambda u: dens(u) * mp.cos(u)**2 * mp.exp(b * (mp.cos(u) - 1)), BRK)
    return M1 / Z, M2 / Z
gel = {'convention': 'x = 1/2 Tr(U1^-1 U2) = cos th ; loi conditionnelle sachant Ubar = V (indépendante de V et de t) ; mutant = sans facteur de coaire',
       'betas': {}, 'u1': {}}
eBes = mp.mpf(0); eMut = mp.mpf(0)
for beta in ('2', '2.8', '8', '1000'):
    b = mp.mpf(beta)
    c1, c2 = moments_radial(lambda u: mp.sin(u)**2, beta)
    bes = mp.besseli(2, b) / mp.besseli(1, b)
    # <cos^2> = (I1/b)''/(I1/b) = (I3 + I2/b)/I1  (récurrences I1' = I2 + I1/b, I2' = I3 + 2 I2/b)
    d2 = (mp.besseli(3, b) + mp.besseli(2, b) / b) / mp.besseli(1, b)
    eBes = max(eBes, abs(c1 - bes), abs(c2 - d2))
    m1, m2 = moments_radial(lambda u: mp.sin(u / 2)**2, beta)        # mutant t = 1/2
    mbes = (mp.besseli(1, b) - (mp.besseli(0, b) + mp.besseli(2, b)) / 2) / (mp.besseli(0, b) - mp.besseli(1, b))
    eMut = max(eMut, abs(m1 - mbes))
    t3 = mp.mpf(3) / 10
    n1, n2 = moments_radial(lambda u: mp.sin(t3 * u)**2 + mp.sin((1 - t3) * u)**2, beta)   # mutant t = 0,3
    gel['betas'][beta] = {
        'x_moy': mp.nstr(c1, 20), 'x2_moy': mp.nstr(c2, 20), 'un_moins_x': mp.nstr(1 - c1, 20),
        'mutant_t_demi': {'x_moy': mp.nstr(m1, 20), 'x2_moy': mp.nstr(m2, 20), 'un_moins_x': mp.nstr(1 - m1, 20)},
        'mutant_t_0.3': {'x_moy': mp.nstr(n1, 20), 'x2_moy': mp.nstr(n2, 20), 'un_moins_x': mp.nstr(1 - n1, 20)},
        'ecart_relatif_mutant_t_demi_sur_1_moins_x': mp.nstr((1 - m1) / (1 - c1) - 1, 8),
        'ecart_relatif_mutant_t_0.3_sur_1_moins_x': mp.nstr((1 - n1) / (1 - c1) - 1, 8)}
    # U(1) : <cos psi> = I1/I0, mutant identique (jacobien constant)
    u1 = mp.besseli(1, b) / mp.besseli(0, b)
    u1q = mp.quad(lambda u: mp.cos(u) * mp.exp(b * (mp.cos(u) - 1)), BRK) / mp.quad(lambda u: mp.exp(b * (mp.cos(u) - 1)), BRK)
    eBes = max(eBes, abs(u1 - u1q))
    gel['u1'][beta] = {'cos_psi_moy': mp.nstr(u1, 20)}
    ligne('INFO', f'FIB-JOUET-GEL-b{beta}', f'<x> = {mp.nstr(c1, 15)} (I2/I1) ; <x^2> = {mp.nstr(c2, 15)} ; mutant t=1/2 : <x> = {mp.nstr(m1, 15)} ; mutant t=0,3 : <x> = {mp.nstr(n1, 15)} ; écart relatif de <1-x> (mutant t=1/2) : {mp.nstr((1 - m1) / (1 - c1) - 1, 6)} ; U(1) <cos psi> = {mp.nstr(u1, 15)}')
ligne('PASS' if eBes < mp.mpf('1e-20') else 'FAIL', 'FIB-JOUET-BESSEL', f'deux canaux (Bessel exact / quadrature) : max écart {mp.nstr(eBes, 3)} (<x> = I2/I1, <x^2> = (I3 + I2/b)/I1, U(1) I1/I0)')
ligne('PASS' if eMut < mp.mpf('1e-20') else 'FAIL', 'FIB-JOUET-MUTANT-BESSEL', f'mutant t = 1/2 : quadrature = (I1 - (I0+I2)/2)/(I0 - I1) : max écart {mp.nstr(eMut, 3)}')
json.dump(gel, open(os.path.join(SORTIES, 'valeurs_jouet.json'), 'w'), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- (g) jouet U(1) : jacobien constant
eU = 0.0
for k in range(20):
    a1, a2 = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi); t = rng.uniform(0.05, 0.95)
    def phibar(x1, x2):
        d = (x2 - x1 + np.pi) % (2 * np.pi) - np.pi
        return x1 + t * d
    h = 1e-6
    J1 = (phibar(a1 + h, a2) - phibar(a1 - h, a2)) / (2 * h); J2 = (phibar(a1, a2 + h) - phibar(a1, a2 - h)) / (2 * h)
    if abs(((a2 - a1 + np.pi) % (2 * np.pi)) - np.pi) < np.pi - 1e-3:
        eU = max(eU, abs(J1 - (1 - t)), abs(J2 - t))
ligne('PASS' if eU < 1e-8 else 'FAIL', 'FIB-JOUET-U1-J', f'U(1) : J = [1 - t, t] constant hors coupure (DF) : max écart {eU:.2e}')
print('BILAN FIB-JOUET', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
