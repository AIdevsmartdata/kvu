#!/usr/bin/env python3
"""Outils SU(2) en précision arbitraire (mpmath) pour WP11 (fibre de Bałaban, jacobien de la coaire).

Conventions (docs/FIBRE_COAIRE.md §1) :
  W = exp(i w.sigma) = cos(th) + i sin(th) n.sigma, th = |w| in [0, pi), n = w/th  (coordonnées « demi-angle » w,
  identiques à rg::bloc::log_q / exp_q de kvu) ; perturbation À GAUCHE U -> exp(i e.sigma) U ;
  R(A)_ab = 1/2 Re Tr(sigma_a A sigma_b A^dag)  (A (i e.sigma) A^-1 = i (R(A) e).sigma).
Deux familles de fonctions, volontairement indépendantes :
  * formes FERMÉES (exp_cf, log_cf, Lg, Ld, Eg, Ed, jac_carte) : ce que l'on vérifie ;
  * références GÉNÉRIQUES (expm_gen, logm_gen = mpmath.expm / mpmath.logm, algorithmes matriciels généraux) : ce qui sert
    aux différences finies. Aucune forme fermée n'entre dans une différence finie.
"""
import mpmath as mp

I2 = lambda: mp.eye(2)
SIG = [mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, -1j], [1j, 0]]), mp.matrix([[1, 0], [0, -1]])]


def dag(A):
    return A.transpose_conj()


def alg(w):
    """i w.sigma (matrice 2x2)"""
    return 1j * (w[0] * SIG[0] + w[1] * SIG[1] + w[2] * SIG[2])


def norme(w):
    return mp.sqrt(sum(x * x for x in w))


def vec(x):
    return [mp.mpf(v) for v in x]


# ---------------------------------------------------------------- références génériques (différences finies)
def expm_gen(w):
    return mp.expm(alg(w))


def logm_gen(W):
    """coordonnées w de la branche principale de log W, par diagonalisation générique (mpmath.eig) et logarithme principal
    de chaque valeur propre (pas de forme fermée). NB : mpmath.logm n'est PAS utilisé : près de -1 il renvoie un logarithme
    de trace 2 pi i (valeur propre e^{-i th} prise sur l'autre feuillet), constaté le 25/10 (dbg1.py)."""
    E, ER = mp.eig(W)
    X = ER * mp.diag([mp.log(x) for x in E]) * mp.inverse(ER)
    # X = i w.sigma  =>  w_a = Tr(X sigma_a) / (2i)
    return [mp.re(sum((X * SIG[a])[k, k] for k in range(2)) / 2j) for a in range(3)]


# ---------------------------------------------------------------- formes fermées
def exp_cf(w):
    th = norme(w)
    s = mp.sinc(th) if th != 0 else mp.mpf(1)   # mp.sinc(x) = sin(x)/x
    return mp.cos(th) * I2() + s * alg(w)


def log_cf(W):
    a0 = mp.re((W[0, 0] + W[1, 1]) / 2)
    v = [mp.re(sum((W * SIG[a])[k, k] for k in range(2)) / 2j) for a in range(3)]   # sin(th) n
    s = norme(v)
    th = mp.atan2(s, a0)
    f = th / s if s != 0 else mp.mpf(1)
    return [f * x for x in v]


def Rad(A):
    """matrice 3x3 de l'action adjointe : A (i e.sigma) A^dag = i (R e).sigma"""
    R = mp.matrix(3, 3)
    Ad = dag(A)
    for a in range(3):
        for b in range(3):
            M = SIG[a] * A * SIG[b] * Ad
            R[a, b] = mp.re((M[0, 0] + M[1, 1]) / 2)
    return R


def croix(w):
    """[w x] : [w x] e = w x e"""
    return mp.matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def nnT(w):
    th = norme(w)
    M = mp.matrix(3, 3)
    if th == 0:
        return M
    for a in range(3):
        for b in range(3):
            M[a, b] = w[a] * w[b] / th**2
    return M


def _f_cot(th):
    """th cot th (série près de 0)"""
    if th < mp.mpf('1e-8'):
        return 1 - th**2 / 3 - th**4 / 45
    return th * mp.cos(th) / mp.sin(th)


def _f_sc(th):
    """sin th cos th / th"""
    if th < mp.mpf('1e-8'):
        return 1 - 2 * th**2 / 3 + 2 * th**4 / 15
    return mp.sin(th) * mp.cos(th) / th


def _f_s2(th):
    """sin^2 th / th^2"""
    if th < mp.mpf('1e-8'):
        return 1 - th**2 / 3 + 2 * th**4 / 45
    return (mp.sin(th) / th) ** 2


def Lg(w):
    """d log à gauche : log(exp(i e.s) W) = i (w + Lg(w) e + O(e^2)).s"""
    th = norme(w); P = nnT(w)
    return P + _f_cot(th) * (mp.eye(3) - P) + croix(w)


def Ld(w):
    """d log à droite : log(W exp(i e.s)) = i (w + Ld(w) e + O(e^2)).s"""
    th = norme(w); P = nnT(w)
    return P + _f_cot(th) * (mp.eye(3) - P) - croix(w)


def Eg(w):
    """d exp à gauche : exp(i(w + d).s) = exp(i (Eg(w) d + O(d^2)).s) exp(i w.s)  ;  Eg = Lg^-1"""
    th = norme(w); P = nnT(w)
    return P + _f_sc(th) * (mp.eye(3) - P) - _f_s2(th) * croix(w)


def Ed(w):
    """d exp à droite : exp(i(w + d).s) = exp(i w.s) exp(i (Ed(w) d + O(d^2)).s)  ;  Ed = Ld^-1"""
    th = norme(w); P = nnT(w)
    return P + _f_sc(th) * (mp.eye(3) - P) + _f_s2(th) * croix(w)


# ---------------------------------------------------------------- carte de Bałaban générique (un lien grossier)
def holonomie(U, chemin):
    """chemin = liste de (lien, signe) ; U = dict/list de matrices ; produit dans l'ordre du chemin"""
    P = I2()
    for l, s in chemin:
        P = P * (U[l] if s == 1 else dag(U[l]))
    return P


def carte(U, ref, chemins, poids, exp=exp_cf, log=log_cf):
    """Ubar = U0 exp( sum_G w_G log(U0^-1 U(G)) ) ; renvoie (Ubar, U0, liste des w_G, wbar)"""
    U0 = holonomie(U, ref); U0i = dag(U0)
    ws = [log(U0i * holonomie(U, G)) for G in chemins]
    wbar = [sum(p * w[a] for p, w in zip(poids, ws)) for a in range(3)]
    return U0 * exp(wbar), U0, ws, wbar


def transport(U, chemin, lien):
    """T_G(l) = sum_{k : l_k = l} s_k R(P_k*) ; P_k* = transport du début du chemin à la QUEUE du lien l_k le long du chemin"""
    T = mp.matrix(3, 3)
    P = I2()
    for l, s in chemin:
        if s == 1:
            if l == lien:
                T += Rad(P)
            P = P * U[l]
        else:
            P = P * dag(U[l])
            if l == lien:
                T -= Rad(P)
    return T


def jac_carte(U, ref, chemins, poids, liens):
    """blocs 3x3 J_{c,l} (forme fermée, docs/FIBRE_COAIRE.md §3.3) pour les liens fins demandés"""
    Ub, U0, ws, wbar = carte(U, ref, chemins, poids)
    R0 = Rad(U0)
    A = R0 * Eg(wbar)
    out = {}
    for l in liens:
        T0 = transport(U, ref, l)
        S = mp.matrix(3, 3)
        for p, G, w in zip(poids, chemins, ws):
            TG = transport(U, G, l)
            S += p * (Lg(w) * (R0.T * (TG - T0)))
        out[l] = T0 + A * S
    return out


def jac_dfc(U, ref, chemins, poids, liens, h=None):
    """différences finies centrées, références génériques uniquement (expm_gen, logm_gen)"""
    if h is None:
        h = mp.mpf(10) ** (-(mp.mp.dps // 2 - 2))
    Ub0 = carte(U, ref, chemins, poids, exp=expm_gen, log=logm_gen)[0]
    Ub0i = dag(Ub0)
    out = {}
    for l in liens:
        B = mp.matrix(3, 3)
        for a in range(3):
            e = [mp.mpf(0)] * 3; e[a] = h
            em = [-x for x in e]
            Up = dict(U); Up[l] = expm_gen(e) * U[l]
            Um = dict(U); Um[l] = expm_gen(em) * U[l]
            dp = logm_gen(carte(Up, ref, chemins, poids, exp=expm_gen, log=logm_gen)[0] * Ub0i)
            dm = logm_gen(carte(Um, ref, chemins, poids, exp=expm_gen, log=logm_gen)[0] * Ub0i)
            for b in range(3):
                B[b, a] = (dp[b] - dm[b]) / (2 * h)
        out[l] = B
    return out


def max_abs(M):
    return max(abs(M[i, j]) for i in range(M.rows) for j in range(M.cols))


def su2_aleatoire(rng, amplitude=None):
    """élément de SU(2) : Haar (amplitude None, quaternion gaussien normalisé) ou exp(i a g.sigma), g gaussien"""
    if amplitude is None:
        q = [mp.mpf(rng.gauss(0, 1)) for _ in range(4)]
        nq = mp.sqrt(sum(x * x for x in q)); q = [x / nq for x in q]
        return q[0] * I2() + 1j * (q[1] * SIG[0] + q[2] * SIG[1] + q[3] * SIG[2])
    return expm_gen([mp.mpf(amplitude) * mp.mpf(rng.gauss(0, 1)) for _ in range(3)])
