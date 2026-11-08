#!/usr/bin/env python3
"""Prototype numpy du gradient de Phi = 1/2 log det(J J^T) (J = D Ubar, carte de Balaban) par PASSE ARRIERE à la main.

Dérivation : voir le brief de spécification (WP12a) ; h = sum_c <W_c, J_c> à W = G^-1 J FIGÉ ; grad Phi = grad h.
Vérifié contre des différences finies de log det(J J^T) le long de directions aléatoires.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import balaban_np as B

I3 = np.eye(3)

def av(X):
    """a(X)_c = sum_ab eps_abc X_ab"""
    return np.array([X[1, 2] - X[2, 1], X[2, 0] - X[0, 2], X[0, 1] - X[1, 0]])

def cr(e):
    return np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])

# ------------------------------------------------------------- dérivées de Lg, Eg (formes fermées, WP11 §3.2)
def coeffs_L(th):
    """f, h, f'/th, h'/th (L_g = f 1 + h w w^T + [w x])"""
    if th < 0.3:
        t2 = th * th
        f = 1 - t2 / 3 - t2**2 / 45 - 2 * t2**3 / 945 - t2**4 / 4725 - 2 * t2**5 / 93555
        h = 1 / 3 + t2 / 45 + 2 * t2**2 / 945 + t2**3 / 4725 + 2 * t2**4 / 93555
        fp = -2 / 3 - 4 * t2 / 45 - 12 * t2**2 / 945 - 8 * t2**3 / 4725 - 20 * t2**4 / 93555
        hp = 2 / 45 + 8 * t2 / 945 + 6 * t2**2 / 4725 + 16 * t2**3 / 93555
        return f, h, fp, hp
    f = th / np.tan(th)
    h = (1 - f) / th**2
    fpp = (1 / np.tan(th) - th / np.sin(th)**2)       # f'
    fp = fpp / th
    hp = (-fpp / th**2 - 2 * (1 - f) / th**3) / th
    return f, h, fp, hp

def coeffs_E(th):
    """s, k, m, s'/th, k'/th, m'/th (E_g = s 1 + k w w^T - m [w x])"""
    if th < 0.3:
        t2 = th * th
        s = 1 - 2 * t2 / 3 + 2 * t2**2 / 15 - 4 * t2**3 / 315 + 2 * t2**4 / 2835
        m = 1 - t2 / 3 + 2 * t2**2 / 45 - t2**3 / 315 + 2 * t2**4 / 14175
        k = -(s - 1) / t2 if th > 1e-3 else None
        # séries de k = (1-s)/th^2 = 2/3 - 2 th^2/15 + 4 th^4/315 - 2 th^6/2835 + ...
        k = 2 / 3 - 2 * t2 / 15 + 4 * t2**2 / 315 - 2 * t2**3 / 2835 + 4 * t2**4 / 155925
        sp = -4 / 3 + 8 * t2 / 15 - 24 * t2**2 / 315 + 16 * t2**3 / 2835
        kp = -4 / 15 + 16 * t2 / 315 - 12 * t2**2 / 2835 + 32 * t2**3 / 155925
        mp_ = -2 / 3 + 8 * t2 / 45 - 6 * t2**2 / 315 + 16 * t2**3 / 14175
        return s, k, m, sp, kp, mp_
    s = np.sin(2 * th) / (2 * th)
    m = (np.sin(th) / th)**2
    k = (1 - s) / th**2
    sdp = (np.cos(2 * th) / th - np.sin(2 * th) / (2 * th**2))   # s'
    mdp = 2 * np.sin(th) * (np.cos(th) * th - np.sin(th)) / th**3 # m'
    kdp = -sdp / th**2 - 2 * (1 - s) / th**3                      # k'
    return s, k, m, sdp / th, kdp / th, mdp / th

def dLg(w):
    """3 matrices dLg/dw_a"""
    th = np.linalg.norm(w)
    f, h, fp, hp = coeffs_L(th)
    out = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1
        out.append(fp * w[a] * I3 + hp * w[a] * np.outer(w, w) + h * (np.outer(e, w) + np.outer(w, e)) + cr(e))
    return out

def dEg(w):
    th = np.linalg.norm(w)
    s, k, m, sp, kp, mp_ = coeffs_E(th)
    out = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1
        out.append(sp * w[a] * I3 + kp * w[a] * np.outer(w, w) + k * (np.outer(e, w) + np.outer(w, e)) - mp_ * w[a] * cr(w) - m * cr(e))
    return out

def Lg1(w):
    return B.Lg(w[None])[0]

def Eg1(w):
    return B.Eg(w[None])[0]

# ------------------------------------------------------------- une ligne de bloc : J (original), Lambda, et gradient à W donné
def donnees_lien(bloc, q, z, mu):
    I, Sg, Wt, Iref = bloc.tables(z, mu)
    U0 = np.array([1.0, 0, 0, 0]); T0 = []
    for l in Iref:
        T0.append(U0.copy()); U0 = B.qmul(U0, q[l])
    P, T = bloc.produits(q, I, Sg, avec_transports=True)
    w, th = B.qlog(B.qmul(B.qdag(U0)[None], P))
    wbar = np.einsum('p,pa->a', Wt, w)
    return dict(I=I, Sg=Sg, Wt=Wt, Iref=Iref, U0=U0, T0=T0, P=P, T=T, w=w, th=th, wbar=wbar)

def jac_bloc(bloc, q, z, mu):
    """(liens, J (nl,3,3) original frame, Lam (nl,3,3), N0 (nl,3,3)) ; J = R0 (N0 + E Lam)"""
    d = donnees_lien(bloc, q, z, mu)
    I, Sg, Wt, Iref, U0 = d['I'], d['Sg'], d['Wt'], d['Iref'], d['U0']
    R0 = B.Rad(U0); E = Eg1(d['wbar'])
    Lw = B.Lg(d['w'])                                  # (P,3,3)
    liens, inv = np.unique(np.concatenate([I[Sg != 0], Iref]), return_inverse=True)
    nl = len(liens)
    msk = Sg != 0
    pos = inv[:msk.sum()]; pos0 = inv[msk.sum():]
    Yq = B.qmul(B.qdag(U0)[None, None], d['T'])        # (P, lm, 4) relatifs
    N = B.Rad(Yq)                                      # (P, lm, 3, 3)
    contrib = Sg[:, :, None, None] * np.einsum('p,pij,pkjl->pkil', Wt, Lw, N)
    Lam = np.zeros((nl, 3, 3)); np.add.at(Lam, pos, contrib[msk])
    N0 = np.zeros((nl, 3, 3))
    for j, p0 in enumerate(pos0):
        N0[p0] += B.Rad(B.qmul(B.qdag(U0), d['T0'][j]))
    Lwsum = np.einsum('p,pij->ij', Wt, Lw)
    Lam = Lam - np.einsum('ij,ljk->lik', Lwsum, N0)
    J = np.einsum('ij,ljk->lik', R0 @ E, Lam) + np.einsum('ij,ljk->lik', R0, N0)
    return liens, J, Lam, N0, d

def grad_bloc(bloc, q, z, mu, Wblk, g):
    """ajoute à g (N,3) la contribution du bloc (z, mu) à grad h, Wblk (nl,3,3) = blocs de W = G^-1 J (original frame)"""
    liens, J, Lam, N0, d = jac_bloc(bloc, q, z, mu)
    I, Sg, Wt, Iref, U0 = d['I'], d['Sg'], d['Wt'], d['Iref'], d['U0']
    R0 = B.Rad(U0); E = Eg1(d['wbar'])
    nl = len(liens)
    V = np.einsum('ji,ljk->lik', R0, Wblk)             # R0^T W
    Um = np.einsum('ji,ljk->lik', E, V)                # E^T V
    # Psi' = sum_l Lam_l V_l^T
    Psi = np.einsum('lij,lkj->ik', Lam, V)
    dE = dEg(d['wbar'])
    e = np.array([np.trace(dE[a] @ Psi) for a in range(3)])
    Lw = B.Lg(d['w'])
    Lwsum = np.einsum('p,pij->ij', Wt, Lw)
    Yq = B.qmul(B.qdag(U0)[None, None], d['T'])
    N = B.Rad(Yq)                                      # (P, lm, 3, 3)
    msk = Sg != 0
    inv = np.searchsorted(liens, np.concatenate([I[Sg != 0], Iref]))
    Pn, lm = Sg.shape
    # slot de chaque (p,k)
    slot = np.zeros((Pn, lm), dtype=int); slot[msk] = inv[:msk.sum()]
    slot0 = inv[msk.sum():]
    # N0 par occurrence
    N0occ = [B.Rad(B.qmul(B.qdag(U0), d['T0'][j])) for j in range(len(Iref))]
    # F_G = sum_k s_k N_k U_{l_k}^T - sum_k0 N0_k0 U_{l0}^T
    F0 = sum(N0occ[j] @ Um[slot0[j]].T for j in range(len(Iref)))
    Fg = np.einsum('pk,pkij,pklj->pil', Sg.astype(float), N, Um[slot]) - F0   # Um[slot]: (P,lm,3,3) ; sum over k of s N U^T
    # d_G,a = tr(dL_a F_G)
    c_end = np.zeros((Pn, 3))
    cstep = np.zeros((Pn, lm, 3))
    for p in range(Pn):
        dL = dLg(d['w'][p])
        dvec = np.array([np.trace(dL[a] @ Fg[p]) for a in range(3)])
        c_end[p] = Wt[p] * (Lw[p].T @ (e + dvec))
    # c_step = 2 w s a(L^T U N^T)
    M = np.einsum('pji,pkjl->pkil', Lw, Um[slot])      # L^T U  (P, lm, 3, 3)
    X = np.einsum('pkij,pklj->pkil', M, N)             # (L^T U) N^T
    for p in range(Pn):
        for k in range(lm):
            if Sg[p, k] != 0:
                cstep[p, k] = 2 * Wt[p] * Sg[p, k] * av(X[p, k])
    # ref
    c0 = []
    for j in range(len(Iref)):
        Xv = V[slot0[j]] @ N0occ[j].T
        Xl = Lwsum.T @ Um[slot0[j]] @ N0occ[j].T
        c0.append(2 * av(Xv) - 2 * av(Xl))
    Ctot = c_end.sum(0) + cstep.sum((0, 1)) + sum(c0)
    # suffix sums per loop : C_{G,m} = sum_{k>m} c_k + [s_m = -1] c_m + c_end
    for p in range(Pn):
        suf = c_end[p].copy()
        for k in range(lm - 1, -1, -1):
            if Sg[p, k] == 0:
                continue
            Cm = suf + (cstep[p, k] if Sg[p, k] == -1 else 0)
            g[liens[slot[p, k]]] += Sg[p, k] * (N[p, k].T @ Cm)
            suf = suf + cstep[p, k]
    # ref : N0^T (sum_{k0>m0} c0 - Ctot)
    for m0 in range(len(Iref)):
        s = sum(c0[k0] for k0 in range(m0 + 1, len(Iref)))
        g[Iref[m0]] += N0occ[m0].T @ (s - Ctot)

def build_J(bloc, q, coarse, cidx, nN):
    n = len(coarse)
    Jf = np.zeros((3 * n, 3 * nN))
    for (z, mu) in coarse:
        c = cidx(z, mu); liens, J, Lam, N0, d = jac_bloc(bloc, q, z, mu)
        for i, l in enumerate(liens):
            Jf[3 * c:3 * c + 3, 3 * l:3 * l + 3] = J[i]
    return Jf

def main(L=6, seed=1, amp=0.8):
    bloc = B.Bloc(L); Lc = L // 3; Vc = Lc**4; Nn = 4 * L**4
    coarse = [(z, mu) for mu in range(4) for z in np.ndindex(Lc, Lc, Lc, Lc)]
    def cidx(z, mu): return mu * Vc + ((((z[0] * Lc + z[1]) * Lc + z[2]) * Lc) + z[3])
    rng = np.random.default_rng(seed)
    q = B.qexp(amp * rng.normal(size=(Nn, 3)) / 1.7)
    t0 = time.time()
    Jf = build_J(bloc, q, coarse, cidx, Nn)
    G = Jf @ Jf.T
    Phi0 = 0.5 * np.linalg.slogdet(G)[1]
    Wf = np.linalg.solve(G, Jf)
    g = np.zeros((Nn, 3))
    for (z, mu) in coarse:
        c = cidx(z, mu); liens, J, Lam, N0, d = jac_bloc(bloc, q, z, mu)
        Wblk = np.stack([Wf[3 * c:3 * c + 3, 3 * l:3 * l + 3] for l in liens])
        grad_bloc(bloc, q, z, mu, Wblk, g)
    print(f'gradient analytique : {time.time() - t0:.1f} s, Phi = {Phi0:.8f}')
    # différences finies le long de directions aléatoires
    def Phi(qq):
        J2 = build_J(bloc, qq, coarse, cidx, Nn)
        return 0.5 * np.linalg.slogdet(J2 @ J2.T)[1]
    for t in range(4):
        d = rng.normal(size=(Nn, 3)); d /= np.linalg.norm(d)
        def phi_s(s): return Phi(np.array([B.qmul(B.qexp(s * d[i]), q[i]) for i in range(Nn)]))
        h = 1e-3
        dc = lambda hh: (phi_s(hh) - phi_s(-hh)) / (2 * hh)
        fd = (4 * dc(h / 2) - dc(h)) / 3
        an = np.sum(g * d)
        print(f'direction {t} : analytique {an:+.10f}  DF {fd:+.10f}  écart relatif {abs(an - fd) / abs(fd):.2e}')

if __name__ == '__main__':
    main(*(float(x) if '.' in x else int(x) for x in sys.argv[1:]))
