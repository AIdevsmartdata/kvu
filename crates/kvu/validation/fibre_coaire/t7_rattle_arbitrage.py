#!/usr/bin/env python3
"""WP11 / FIB-ARB-RATTLE : arbitrage du désaccord D1 (docs/FIBRE_COAIRE.md §7) — quel potentiel donner à RATTLE (route B) ?

Opus : RATTLE (énergie cinétique 1/2 p^T M^-1 p, M constante) échantillonne e^{-V} sigma_M (volume riemannien de la fibre pour la
       métrique M, mesure de Liouville de T*F projetée) ; pour cibler la désintégration e^{-S} det(JJ^T)^{-1/2} sigma il faut
       V = S + 1/2 log det(J M^-1 J^T)   (M = 1 : S + 1/2 log det(J J^T), comme ARCHITECTURE §5).
Kimi : RATTLE échantillonne e^{-V} sqrt(det M)/sqrt(det(J M^-1 J^T) det(J J^T)) sigma ; V = S + 1/2 log det M - 1/2 log det(J M^-1 J^T) ;
       « pour M = 1 aucun potentiel additionnel n'est nécessaire » (le texte et la formule encadrée se contredisent pour M = 1).
Arbitre :
 (a) SymPy : courbe-graphe y = f(x) de R^2, contrainte xi = y - f(x) ; coaire exacte (delta(xi) dx dy -> dx) ; marginale de Liouville
     sur T*F : int exp(-p_x^2/(2 g)) dp_x = sqrt(2 pi g), g = m1 + m2 f'^2 ; et la prescription Opus donne exactement la cible ;
 (b) expérience numérique : RATTLE + Metropolis (projection de Newton, contrôle de réversibilité, impulsions tirées dans N(0, M) et
     projetées M-orthogonalement) sur f(x) = 1,2 sin x, V = 0,5 cos x + y^2/2, pour M = 1 et M = diag(1, 4), trois potentiels :
     A (aucun terme), B (Opus : + 1/2 log A), C (Kimi : - 1/2 log A), A = grad xi^T M^-1 grad xi ; la cible est la loi conditionnelle
     de x sachant xi = 0 sous e^{-V} dx dy, soit ∝ e^{-V(x, f(x))} dx. On compare <cos x>, <cos 2x> aux prédictions des deux thèses.
"""
import os, sys, math
import numpy as np
import sympy as sp

ok = True
def ligne(etat, ident, detail):
    global ok
    if etat == 'FAIL':
        ok = False
    print(f'{etat} {ident} {detail}', flush=True)

# ---------------------------------------------------------------- (a) SymPy
x, y, px, m1, m2, a_ = sp.symbols('x y p_x m1 m2 a', real=True)
pos = sp.symbols('g', positive=True)
f = a_ * sp.sin(x); fp = sp.diff(f, x)
g = m1 + m2 * fp**2                                        # métrique induite par M sur la courbe (paramètre x)
Aq = fp**2 / m1 + 1 / m2                                   # grad xi^T M^-1 grad xi, grad xi = (-f', 1)
I_p = sp.integrate(sp.exp(-px**2 / (2 * pos)), (px, -sp.oo, sp.oo))
r1 = sp.simplify(I_p - sp.sqrt(2 * sp.pi * pos))
r2 = sp.simplify(g / Aq - m1 * m2)                         # Opus : e^{-V - 1/2 log A} sqrt(g) = e^{-V} sqrt(m1 m2) : constante
ligne('PASS' if r1 == 0 and r2 == 0 else 'FAIL', 'FIB-ARB-SYMPY',
      f'int e^(-p^2/2g) dp - sqrt(2 pi g) = {r1} (marginale de Liouville ∝ e^(-V) sqrt(g) dx = e^(-V) sigma_M) ; g/A - m1 m2 = {r2} (prescription +1/2 log A : densité ∝ e^(-V) dx = cible exacte)')

# ---------------------------------------------------------------- (b) expérience RATTLE
A_F, B_V = 1.2, 0.5
def fx(xx): return A_F * np.sin(xx)
def fpx(xx): return A_F * np.cos(xx)
def fppx(xx): return -A_F * np.sin(xx)

def lance(Mdiag, variante, graine, K=4000, T=500, nst=6, h=0.15):
    rng = np.random.default_rng(graine)
    Mi = 1.0 / np.asarray(Mdiag, float); Md = np.asarray(Mdiag, float)
    c = {'A': 0.0, 'B': 0.5, 'C': -0.5}[variante]
    def Aq(xx): return fpx(xx)**2 * Mi[0] + Mi[1]
    def Veff(q):
        return B_V * np.cos(q[:, 0]) + 0.5 * q[:, 1]**2 + c * np.log(Aq(q[:, 0]))
    def gVeff(q):
        gx = -B_V * np.sin(q[:, 0]) + c * (2 * fpx(q[:, 0]) * fppx(q[:, 0]) * Mi[0]) / Aq(q[:, 0])
        return np.stack([gx, q[:, 1]], 1)
    def gxi(q):                                             # xi = y - f(x)
        return np.stack([-fpx(q[:, 0]), np.ones(len(q))], 1)
    def xi(q): return q[:, 1] - fx(q[:, 0])
    def proj_p(q, p):                                       # p tangent : grad xi^T M^-1 p = 0, projection M-orthogonale
        n = gxi(q); An = np.sum(n * n * Mi, 1)
        lam = np.sum(n * p * Mi, 1) / An
        return p - lam[:, None] * n
    def pas(q, p):
        """un pas RATTLE ; renvoie (q', p', convergé)"""
        n0 = gxi(q)
        qt = q + h * Mi * (p - 0.5 * h * gVeff(q))
        lam = np.zeros(len(q)); ok_ = np.ones(len(q), bool)
        for it in range(30):
            qn = qt - 0.5 * h * h * lam[:, None] * Mi * n0
            r = xi(qn)
            dr = np.sum(gxi(qn) * (-0.5 * h * h * Mi * n0), 1)
            lam = lam - r / dr
        qn = qt - 0.5 * h * h * lam[:, None] * Mi * n0
        ok_ &= np.abs(xi(qn)) < 1e-12
        ph = (qn - q) * Md / h
        pn = proj_p(qn, ph - 0.5 * h * gVeff(qn))
        return qn, pn, ok_
    x0 = rng.uniform(0, 2 * np.pi, K); q = np.stack([x0, fx(x0)], 1)
    som = np.zeros((K, 2)); nacc = 0; nrev = 0; ntot = 0
    for tr in range(T):
        z = rng.normal(size=(K, 2)) * np.sqrt(Md)
        p = proj_p(q, z)
        H0 = Veff(q) + 0.5 * np.sum(p * p * Mi, 1)
        qq, pp = q.copy(), p.copy(); good = np.ones(K, bool)
        for k in range(nst):
            qq, pp, okk = pas(qq, pp); good &= okk
        # réversibilité : on repart de (qq, -pp)
        qr, pr = qq.copy(), -pp
        for k in range(nst):
            qr, pr, okk = pas(qr, pr); good &= okk
        rev = good & (np.abs(qr - q).max(1) < 1e-8)
        H1 = Veff(qq) + 0.5 * np.sum(pp * pp * Mi, 1)
        acc = rev & (rng.uniform(size=K) < np.exp(np.minimum(0, H0 - H1)))
        q = np.where(acc[:, None], qq, q)
        nacc += acc.sum(); nrev += (~rev).sum(); ntot += K
        if tr >= 50:
            som += np.stack([np.cos(q[:, 0]), np.cos(2 * q[:, 0])], 1)
    moy_ch = som / (T - 50)
    return moy_ch.mean(0), moy_ch.std(0) / np.sqrt(K), nacc / ntot, nrev / ntot

def predit(dens):
    xs = np.linspace(0, 2 * np.pi, 200001)[:-1]
    w = dens(xs); w /= w.sum()
    return np.array([np.sum(w * np.cos(xs)), np.sum(w * np.cos(2 * xs))])

Vc = lambda xs: np.exp(-(B_V * np.cos(xs) + 0.5 * fx(xs)**2))
cible = predit(Vc)
print(f'INFO FIB-ARB-CIBLE loi conditionnelle exacte de x sachant xi = 0 (quadrature, 2e5 points) : <cos x> = {cible[0]:.10f}, <cos 2x> = {cible[1]:.10f}', flush=True)
for Mdiag in ((1.0, 1.0), (1.0, 4.0)):
    m1_, m2_ = Mdiag
    g_ = lambda xs: m1_ + m2_ * fpx(xs)**2
    A_ = lambda xs: fpx(xs)**2 / m1_ + 1 / m2_
    G_ = lambda xs: 1 + fpx(xs)**2
    # thèse Opus : RATTLE échantillonne e^{-Veff} sqrt(g) dx
    opus = {v: predit(lambda xs, c=c: Vc(xs) * np.exp(-c * np.log(A_(xs))) * np.sqrt(g_(xs))) for v, c in (('A', 0.0), ('B', 0.5), ('C', -0.5))}
    # thèse Kimi : RATTLE échantillonne e^{-Veff} sqrt(det M)/(sqrt(A) sqrt(G)) sigma, sigma = sqrt(G) dx
    kimi = {v: predit(lambda xs, c=c: Vc(xs) * np.exp(-c * np.log(A_(xs))) / np.sqrt(A_(xs))) for v, c in (('A', 0.0), ('B', 0.5), ('C', -0.5))}
    for v in ('A', 'B', 'C'):
        m, e, acc, nrev = lance(Mdiag, v, 26102618 + int(10 * m2_) + ord(v))
        zc = (m - cible) / e; zo = (m - opus[v]) / e; zk = (m - kimi[v]) / e
        lab = f'M = diag{Mdiag}, potentiel {v}'
        print(f'INFO FIB-ARB-RATTLE {lab} : <cos x>, <cos 2x> = {m[0]:.5f} +- {e[0]:.5f}, {m[1]:.5f} +- {e[1]:.5f} ; acceptation {acc:.3f}, rejets de réversibilité {nrev:.1e} ; '
              f'z(cible) = {zc[0]:+.1f}, {zc[1]:+.1f} ; z(thèse Opus) = {zo[0]:+.1f}, {zo[1]:+.1f} ; z(thèse Kimi) = {zk[0]:+.1f}, {zk[1]:+.1f}', flush=True)
        if v == 'B':
            ligne('PASS' if np.all(np.abs(zc) < 4) else 'FAIL', f'FIB-ARB-B-{int(m2_)}', f'{lab} (Opus, +1/2 log det(J M^-1 J^T)) retrouve la cible : |z| = {np.abs(zc).max():.1f} < 4')
        if v == 'A' and Mdiag == (1.0, 1.0):
            ligne('PASS' if np.any(np.abs(zc) > 10) else 'FAIL', 'FIB-ARB-A-1', f'{lab} (Kimi : « rien à ajouter pour M = 1 ») est loin de la cible : |z| max = {np.abs(zc).max():.1f} > 10')
        if v == 'C':
            ligne('PASS' if np.any(np.abs(zc) > 10) else 'FAIL', f'FIB-ARB-C-{int(m2_)}', f'{lab} (formule encadrée de Kimi, -1/2 log det(J M^-1 J^T)) est loin de la cible : |z| max = {np.abs(zc).max():.1f} > 10')
        ligne('PASS' if np.all(np.abs(zo) < 4) else 'FAIL', f'FIB-ARB-LIOUVILLE-{v}{int(m2_)}', f'{lab} : la loi échantillonnée = prédiction « e^(-V) sigma_M » (Liouville) à |z| = {np.abs(zo).max():.1f} < 4 (thèse Kimi : |z| = {np.abs(zk).max():.1f})')
print('BILAN FIB-ARB-RATTLE', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
