#!/usr/bin/env python3
"""Opérateurs linéaires en espace direct (disposition kvu) pour le cas U(1) exact et la limite beta -> infini.

lien (nu, x) : nu*V + site(x) ; plaquette (k, x) : k*V + site(x), k = indice de (mu, nu) dans combinations(range(4), 2) ;
theta_p = th_mu(x) + th_nu(x + mu) - th_mu(x + nu) - th_nu(x) ; (d lambda)_(nu, x) = lambda(x + nu) - lambda(x) ;
Q = moyenne linéaire (0.4) : Q[c, l] = sum_G w_G sum_k s_k [l_k = l] (mêmes chemins que balaban_np) ;
S_U(1) gaussienne = beta theta^T M theta = (beta/2) sum_p theta_p^2, M = 1/2 rot^T rot.
"""
import itertools
import numpy as np
import balaban_np as B

PAIRES = list(itertools.combinations(range(4), 2))

class Ops:
    def __init__(self, L):
        self.L = L; self.V = L**4; self.N = 4 * self.V; self.Lc = L // 3; self.Vc = self.Lc**4; self.n = 4 * self.Vc
        self.bloc = B.Bloc(L)
        L_ = L
        X = np.array(list(np.ndindex(L_, L_, L_, L_)))           # sites en ordre C
        self.X = X
        def site(x):
            x = np.mod(x, L_)
            return ((x[..., 0] * L_ + x[..., 1]) * L_ + x[..., 2]) * L_ + x[..., 3]
        self.site = site
        e = np.eye(4, dtype=int)
        # plaquettes : 4 liens et signes
        P = []
        for k, (mu, nu) in enumerate(PAIRES):
            s0 = site(X)
            P.append(np.stack([mu * self.V + s0, nu * self.V + site(X + e[mu]), mu * self.V + site(X + e[nu]), nu * self.V + s0], 1))
        self.plaq = np.concatenate(P, 0)                          # (6V, 4)
        self.psig = np.array([1.0, 1.0, -1.0, -1.0])
        # centres des blocs
        self.centres = np.where(np.all(X % 3 == 1, axis=1))[0]
        # Q dense (n x N)
        Q = np.zeros((self.n, self.N))
        for mu in range(4):
            for z in np.ndindex(self.Lc, self.Lc, self.Lc, self.Lc):
                c = mu * self.Vc + ((z[0] * self.Lc + z[1]) * self.Lc + z[2]) * self.Lc + z[3]
                I, Sg, W, Iref = self.bloc.tables(z, mu)
                np.add.at(Q[c], I[Sg != 0], (W[:, None] * Sg)[Sg != 0])
        self.Q = Q

    def rot(self, X):
        """rot @ X (X : (N, ...))"""
        out = self.psig[0] * X[self.plaq[:, 0]]
        for k in range(1, 4):
            out += self.psig[k] * X[self.plaq[:, k]]
        return out

    def rotT(self, Y):
        out = np.zeros((self.N,) + Y.shape[1:])
        for k in range(4):
            np.add.at(out, self.plaq[:, k], self.psig[k] * Y)
        return out

    def M(self):
        """M = 1/2 rot^T rot (dense)"""
        R = self.rot(np.eye(self.N))
        return 0.5 * R.T @ R

    def d(self):
        """gradient de jauge (N x V)"""
        D = np.zeros((self.N, self.V)); e = np.eye(4, dtype=int)
        s0 = self.site(self.X)
        for nu in range(4):
            D[nu * self.V + s0, self.site(self.X + e[nu])] += 1.0
            D[nu * self.V + s0, s0] -= 1.0
        return D

def null_space(A, tol=1e-10):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    r = int((s > tol * max(1.0, s[0] if len(s) else 1.0)).sum())
    return vh[r:].T

class Gauss:
    """loi conditionnelle gaussienne exacte (U(1) non compact, ou chaque couleur de SU(2) à beta -> infini)"""
    def __init__(self, ops):
        self.o = ops
        M = ops.M(); self.Mmat = M
        Z = null_space(ops.Q)                                     # ker Q (N x (N - n))
        D = ops.d()
        non_centres = np.setdiff1d(np.arange(ops.V), ops.centres)
        DL = D[:, non_centres]                                    # jauge résiduelle (lambda = 0 aux centres) : dans ker Q
        self.err_jauge_dans_kerQ = float(np.abs(ops.Q @ DL).max())
        C = Z.T @ DL
        Bs = Z @ null_space(C.T)                                  # tranche : ker Q ∩ (jauge résiduelle)^perp
        self.dimS = Bs.shape[1]
        MV = Bs.T @ M @ Bs
        self.MVi = np.linalg.inv(MV)
        self.Bs = Bs
        self.Gam = Bs @ self.MVi @ Bs.T                           # Gamma (picov : Gam) ; cov(theta) = Gam/(2 beta)
        Qp = ops.Q.T @ np.linalg.inv(ops.Q @ ops.Q.T)
        self.H = Qp - self.Gam @ (M @ Qp)                         # extension harmonique (KKT) : theta* = H phibar
        self.Mc = self.H.T @ M @ self.H                           # S_min = beta phibar^T Mc phibar
        self.PG = DL @ np.linalg.pinv(DL)                         # projecteur sur la jauge résiduelle
        self.DL = DL

    def cov_F_beta(self, lignes=None):
        """beta * cov(F) = 1/2 rot Gam rot^T (restreint aux plaquettes demandées)"""
        RG = self.o.rot(self.Gam)                                 # (6V, N)
        if lignes is None:
            return 0.5 * np.einsum('pi,pi->p', RG, self.o.rot(np.eye(self.o.N)))   # diagonale seulement
        return 0.5 * self.o.rot(RG[lignes].T).T

    def penalite(self, kappa):
        """route A gaussienne, kappa = beta eps^2 : beta * cov(theta) = (2 M + Q^T Q / kappa + P_G)^-1 (partie invariante)"""
        K = 2 * self.Mmat + self.o.Q.T @ self.o.Q / kappa + self.PG
        return np.linalg.inv(K)
