#!/usr/bin/env python3
"""Carte de Bałaban L_b = 3 (CMP 109 (0.4), forme covariante de WP6) et son jacobien en forme fermée, en numpy (F64).

Implémentation INDÉPENDANTE (énumération des chemins réécrite ici, pas d'import de geom.py ni de kvu) ; elle est validée
contre les fixtures de WP6 (`crates/kvu/validation/rg/fixtures`, Ubar de g_nl.py) par t4_balaban.py.
Disposition kvu : champ q de forme (4, L, L, L, L, 4), lien (nu, x) d'indice nu*L^4 + site(x), site C-order ;
quaternion [a0, a1, a2, a3] <-> U = a0 + i (a1 s1 + a2 s2 + a3 s3).
Lien grossier (z, mu) : centre c = 3z + (1,1,1,1), c' = c + 3 e_mu ; Ubar = U0 exp( sum_G w_G log(U0^-1 U(G)) ),
G = P . [x, x + 3e_mu] . P'^-1, P in G(c, x), P' in G(c', x'), w = 1/(81 |G(c,x)|^2), U0 = segment droit c -> c'.
"""
import itertools, math
import numpy as np

# ---------------------------------------------------------------- quaternions (vectorisés sur les axes de tête)
def qmul(a, b):
    a0, av = a[..., 0], a[..., 1:]; b0, bv = b[..., 0], b[..., 1:]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a0 * b0 - np.sum(av * bv, axis=-1)
    out[..., 1:] = a0[..., None] * bv + b0[..., None] * av - np.cross(av, bv)
    return out

def qdag(a):
    out = a.copy(); out[..., 1:] *= -1; return out

def qlog(q):
    """w (demi-angle) : q = (cos th, sin th n) -> w = th n ; branche principale th in [0, pi]"""
    v = q[..., 1:]; s = np.linalg.norm(v, axis=-1)
    th = np.arctan2(s, q[..., 0])
    f = np.where(s > 1e-6, th / np.where(s > 0, s, 1.0), 1 + s**2 / 6 + 7 * s**4 / 360)
    return f[..., None] * v, th

def qexp(w):
    th = np.linalg.norm(w, axis=-1)
    f = np.where(th > 1e-6, np.sin(th) / np.where(th > 0, th, 1.0), 1 - th**2 / 6 + th**4 / 120)
    out = np.empty(w.shape[:-1] + (4,)); out[..., 0] = np.cos(th); out[..., 1:] = f[..., None] * w
    return out

def croix(w):
    Z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([Z, -w[..., 2], w[..., 1]], -1), np.stack([w[..., 2], Z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], Z], -1)], -2)

def Rad(q):
    """R(A) : A (i e.s) A^dag = i (R e).s ; R = (q0^2 - |v|^2) 1 - 2 q0 [v x] + 2 v v^T"""
    q0, v = q[..., 0], q[..., 1:]
    return ((q0**2 - np.sum(v * v, -1))[..., None, None] * np.eye(3) - 2 * q0[..., None, None] * croix(v)
            + 2 * v[..., :, None] * v[..., None, :])

def _proj(w):
    th = np.linalg.norm(w, axis=-1)
    n = w / np.where(th > 0, th, 1.0)[..., None]
    return th, n[..., :, None] * n[..., None, :]

def Lg(w):
    th, P = _proj(w)
    fc = np.where(th > 1e-4, th * np.cos(th) / np.where(np.sin(th) != 0, np.sin(th), 1.0), 1 - th**2 / 3 - th**4 / 45)
    return P + fc[..., None, None] * (np.eye(3) - P) + croix(w)

def Eg(w):
    th, P = _proj(w)
    sc = np.where(th > 1e-4, np.sin(th) * np.cos(th) / np.where(th > 0, th, 1.0), 1 - 2 * th**2 / 3 + 2 * th**4 / 15)
    s2 = np.where(th > 1e-4, (np.sin(th) / np.where(th > 0, th, 1.0))**2, 1 - th**2 / 3 + 2 * th**4 / 45)
    return P + sc[..., None, None] * (np.eye(3) - P) - s2[..., None, None] * croix(w)

# ---------------------------------------------------------------- géométrie (énumération indépendante)
def _chemins_centre(delta):
    """tous les chemins les plus courts du centre vers centre + delta (delta in {-1,0,1}^4) : listes de (dep, nu, signe),
    dep = déplacement du site de départ du lien par rapport au centre"""
    pas = [(m, d) for m, d in enumerate(delta) if d != 0]
    out = []
    for perm in itertools.permutations(pas):
        cur = [0, 0, 0, 0]; ch = []
        for m, d in perm:
            if d == 1:
                ch.append((tuple(cur), m, 1)); cur[m] += 1
            else:
                cur[m] -= 1; ch.append((tuple(cur), m, -1))
        out.append(ch)
    return out

def gabarit(mu):
    """chemins du lien grossier (z = 0, mu) relatifs au coin 3z : liste (poids, [(dep, nu, signe)]), et le chemin de référence"""
    ctr = (1, 1, 1, 1); emu = tuple(3 if m == mu else 0 for m in range(4))
    out = []; ref = None
    for r in itertools.product(range(3), repeat=4):
        delta = tuple(r[m] - 1 for m in range(4))
        Ps = _chemins_centre(delta)
        w = 1.0 / (81 * len(Ps)**2)
        x = r
        mid = [(tuple(x[m] + (k if m == mu else 0) for m in range(4)), mu, 1) for k in range(3)]
        for P in Ps:
            for P2 in Ps:
                g = [(tuple(ctr[m] + d[m] for m in range(4)), nu, s) for d, nu, s in P]
                g += mid
                g += [(tuple(ctr[m] + emu[m] + d[m] for m in range(4)), nu, -s) for d, nu, s in reversed(P2)]
                out.append((w, g))
        if delta == (0, 0, 0, 0):
            ref = mid
    return out, ref

class Bloc:
    def __init__(self, L):
        assert L % 3 == 0
        self.L = L; self.Lc = L // 3; self.V = L**4
        self.gab = [gabarit(mu) for mu in range(4)]
        self.lmax = max(len(g) for w, g in self.gab[0][0])
        self.npaths = len(self.gab[0][0])

    def idx_lien(self, x, nu):
        L = self.L
        x = [v % L for v in x]
        return nu * self.V + ((x[0] * L + x[1]) * L + x[2]) * L + x[3]

    def tables(self, z, mu):
        """indices et signes (npaths, lmax) (signe 0 = bourrage), poids, chemin de référence"""
        gab, ref = self.gab[mu]
        o = [3 * v for v in z]
        I = np.zeros((len(gab), self.lmax), dtype=np.int64); Sg = np.zeros((len(gab), self.lmax), dtype=np.int64)
        W = np.empty(len(gab))
        for p, (w, g) in enumerate(gab):
            W[p] = w
            for k, (d, nu, s) in enumerate(g):
                I[p, k] = self.idx_lien([o[m] + d[m] for m in range(4)], nu); Sg[p, k] = s
        Iref = np.array([self.idx_lien([o[m] + d[m] for m in range(4)], nu) for d, nu, s in ref])
        return I, Sg, W, Iref

    @staticmethod
    def produits(q, I, Sg, avec_transports=False):
        """produits de chemins ; si demandé, transports P* (début du chemin -> queue du lien) pour chaque pas"""
        n, lm = I.shape
        P = np.zeros((n, 4)); P[:, 0] = 1.0
        T = np.zeros((n, lm, 4)) if avec_transports else None
        for k in range(lm):
            u = q[I[:, k]]
            s = Sg[:, k]
            ud = np.where((s == -1)[:, None], qdag(u), u)
            ud = np.where((s == 0)[:, None], np.array([1.0, 0, 0, 0]), ud)
            if avec_transports:
                T[:, k] = P                                  # signe +1 : prefixe avant le lien
            P = qmul(P, ud)
            if avec_transports:
                T[:, k] = np.where((s == -1)[:, None], P, T[:, k])   # signe -1 : prefixe APRES U^-1
        return P, T

    def ubar(self, q, z, mu, tab=None):
        I, Sg, W, Iref = tab if tab is not None else self.tables(z, mu)
        U0 = np.array([1.0, 0, 0, 0])
        for l in Iref:
            U0 = qmul(U0, q[l])
        P, _ = self.produits(q, I, Sg)
        w, th = qlog(qmul(qdag(U0)[None], P))
        wbar = np.einsum('p,pa->a', W, w)
        return qmul(U0, qexp(wbar)), th

    def jac(self, q, z, mu, tab=None):
        """blocs J_{c,l} (forme fermée §3.3) : renvoie (liens, blocs (nl, 3, 3))"""
        I, Sg, W, Iref = tab if tab is not None else self.tables(z, mu)
        U0 = np.array([1.0, 0, 0, 0]); T0 = []
        for l in Iref:
            T0.append(U0.copy()); U0 = qmul(U0, q[l])
        P, T = self.produits(q, I, Sg, avec_transports=True)
        w, th = qlog(qmul(qdag(U0)[None], P))
        wbar = np.einsum('p,pa->a', W, w)
        R0 = Rad(U0); A = R0 @ Eg(wbar)
        B = W[:, None, None] * np.einsum('ij,pjk,lk->pil', A, Lg(w), R0)      # B_G = w_G R0 Eg(wbar) Lg(w_G) R0^T
        Btot = B.sum(0)
        liens, inv = np.unique(np.concatenate([I[Sg != 0], Iref]), return_inverse=True)
        J = np.zeros((len(liens), 3, 3))
        # sum_G sum_k s_k B_G R(P*_{G,k})
        RT = Rad(T)                                             # (n, lm, 3, 3)
        contrib = Sg[:, :, None, None] * np.einsum('pij,pkjl->pkil', B, RT)
        msk = Sg != 0
        pos = inv[:msk.sum()]
        np.add.at(J, pos, contrib[msk])
        # (1 - Btot) T0(l) pour les 3 liens de la référence (signes +1)
        pos0 = inv[msk.sum():]
        for j, p0 in enumerate(pos0):
            J[p0] += (np.eye(3) - Btot) @ Rad(T0[j])
        return liens, J, th
