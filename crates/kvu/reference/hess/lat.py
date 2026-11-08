"""lat.py — S2 (chantier vision, phase 6) : réseau 4D périodique, Hessienne EXACTE de l'action de Wilson
SU(2) (coordonnées additives de l'algèbre, ou coordonnées à gauche), U(1), mutant SU(2) abélianisé.

Conventions (PREREG §1) :
  SU(2) : S = beta * sum_p (1 - 1/2 Re tr U_p), beta = 2 C_A / g^2 = 4/g^2 (C_A = 2),
          U = exp(i A.sigma/2), A in R^3.  Toutes les Hessiennes sont rendues PAR UNITE DE beta (H/beta).
  U(1)  : S = beta_U * sum_p (1 - cos F_p), beta_U = 1/g^2 ; rendu H/beta_U.
  mutant: S = beta * sum_p (1 - cos(|F_p|/2)), F_p = A1+A2-A3-A4 in R^3 (commutateurs retirés).
Indices : site idx = ((x0*N+x1)*N+x2)*N+x3 ; lien l = 4*idx+mu ; ddl SU(2) = 3*l+c.
Plaquette (x,mu<nu) : l1=(x,mu), l2=(x+mu,nu), l3=(x+nu,mu), l4=(x,nu) ; U_p = U1 U2 U3^+ U4^+.
"""
import itertools
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

SIG = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ----------------------------------------------------------------------------- géométrie
class Geo:
    def __init__(self, N, Lb=2):
        self.N, self.d = N, 4
        self.V = N ** 4
        self.nl = 4 * self.V
        c = np.array(list(itertools.product(range(N), repeat=4)))
        self.coords = c
        self.idx = lambda cc: ((cc[..., 0] * N + cc[..., 1]) * N + cc[..., 2]) * N + cc[..., 3]
        self.sh = np.zeros((4, 2, self.V), dtype=np.int64)  # sh[mu,0]=x+mu, sh[mu,1]=x-mu
        for mu in range(4):
            e = np.zeros(4, dtype=int); e[mu] = 1
            self.sh[mu, 0] = self.idx((c + e) % N)
            self.sh[mu, 1] = self.idx((c - e) % N)
        P = []
        for x in range(self.V):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    P.append((4 * x + mu, 4 * self.sh[mu, 0, x] + nu, 4 * self.sh[nu, 0, x] + mu, 4 * x + nu))
        self.plaq = np.array(P, dtype=np.int64)  # (np,4)
        self.sgn = np.array([1, 1, -1, -1])
        self.np = len(P)
        # rot (scalaire) : (np, nl)
        rows = np.repeat(np.arange(self.np), 4)
        self.R = sp.csr_matrix((np.tile(self.sgn, self.np).astype(float), (rows, self.plaq.ravel())),
                               shape=(self.np, self.nl))
        # div (scalaire) : (div z)_x = sum_mu z_{x,mu} - z_{x-mu,mu}
        r, cc, v = [], [], []
        for x in range(self.V):
            for mu in range(4):
                r += [x, x]; cc += [4 * x + mu, 4 * self.sh[mu, 1, x] + mu]; v += [1.0, -1.0]
        self.div = sp.csr_matrix((v, (r, cc)), shape=(self.V, self.nl))
        # bloc de coin Lb=2 : B z_{X,mu} = 1/8 sum_{y perp} (z_{x0+y,mu} + z_{x0+y+mu,mu})
        assert N % Lb == 0 and Lb == 2
        self.Nc = N // Lb
        self.block_terms = []  # (row, x0, mu, yvec list) pour la version covariante
        r, cc, v = [], [], []
        row = 0
        for X in itertools.product(range(self.Nc), repeat=4):
            x0 = np.array(X) * Lb
            for mu in range(4):
                perp = [nu for nu in range(4) if nu != mu]
                for ys in itertools.product(range(2), repeat=3):
                    y = np.zeros(4, dtype=int)
                    for k, nu in enumerate(perp):
                        y[nu] = ys[k]
                    xa = self.idx((x0 + y) % N)
                    xb = self.sh[mu, 0, xa]
                    r += [row, row]; cc += [4 * xa + mu, 4 * xb + mu]; v += [1 / 8, 1 / 8]
                    self.block_terms.append((row, int(self.idx(x0 % N)), mu, y.copy(), int(xa), int(xb)))
                row += 1
        self.nB = row
        self.B = sp.csr_matrix((v, (r, cc)), shape=(row, self.nl))
        # plaquettes contenant chaque lien
        self.plaq_of_link = [[] for _ in range(self.nl)]
        for p in range(self.np):
            for k in range(4):
                self.plaq_of_link[self.plaq[p, k]].append((p, k))


# ----------------------------------------------------------------------------- SU(2) utilitaires
def su2_exp(A):
    """A (...,3) -> exp(i A.sigma/2) (...,2,2), exact (formule fermée)."""
    th = np.linalg.norm(A, axis=-1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(th > 1e-300, np.sin(th) / np.where(th > 0, th, 1), 1.0) / 2  # sin(th)/(2 th)... *A
    M = np.cos(th)[..., None, None] * I2 + 1j * np.einsum("...a,aij->...ij", A * s[..., None] , SIG)
    return M


def su2_log(U):
    """U (...,2,2) in SU(2) -> A (...,3) avec U = exp(i A.sigma/2), |A| < 2 pi."""
    a0 = np.real(U[..., 0, 0] + U[..., 1, 1]) / 2
    av = np.stack([np.imag(U[..., 0, 1] + U[..., 1, 0]) / 2,
                   np.real(U[..., 0, 1] - U[..., 1, 0]) / 2,
                   np.imag(U[..., 0, 0] - U[..., 1, 1]) / 2], axis=-1)
    nv = np.linalg.norm(av, axis=-1)
    th = np.arctan2(nv, np.clip(a0, -1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(nv > 1e-300, 2 * th / np.where(nv > 0, nv, 1), 2.0)
    return av * f[..., None]


def adj(V):
    """R(V)_{ab} = 1/2 tr(sigma_a V sigma_b V^+)  (...,3,3)."""
    Vd = np.conj(np.swapaxes(V, -1, -2))
    return 0.5 * np.real(np.einsum("aij,...jk,bkl,...li->...ab", SIG, V, SIG, Vd))


def exp_derivs(A, s=1.0):
    """Pour E(A) = exp(i s A.sigma/2) : (E, dE[a], d2E[a,b]) exacts via exponentielle de matrices par blocs.
    dE[a]   = d/dA_a E,  d2E[a,b] = d^2/dA_a dA_b E."""
    n = A.shape[0]
    X = 1j * s * np.einsum("na,aij->nij", A, SIG) / 2
    Y = [1j * s * SIG[a] / 2 for a in range(3)]
    E = sla.expm(X)
    dE = np.zeros((n, 3, 2, 2), complex)
    for a in range(3):
        M = np.zeros((n, 4, 4), complex)
        M[:, :2, :2] = X; M[:, 2:, 2:] = X; M[:, :2, 2:] = Y[a]
        dE[:, a] = sla.expm(M)[:, :2, 2:]
    blk = np.zeros((n, 3, 3, 2, 2), complex)
    for a in range(3):
        for b in range(3):
            M = np.zeros((n, 6, 6), complex)
            for k in range(3):
                M[:, 2 * k:2 * k + 2, 2 * k:2 * k + 2] = X
            M[:, :2, 2:4] = Y[a]; M[:, 2:4, 4:6] = Y[b]
            blk[:, a, b] = sla.expm(M)[:, :2, 4:6]
    d2E = blk + np.swapaxes(blk, 1, 2)
    return E, dE, d2E


def plaq_hess(F, D, DD):
    """F (np,4,2,2), D (np,4,3,2,2), DD (np,4,3,3,2,2) facteurs de P = F1F2F3F4 et leurs dérivées.
    Rend h (np,4,3,4,3) = Hessienne de (1 - 1/2 Re tr P) et la valeur (np,)."""
    npl = F.shape[0]
    h = np.zeros((npl, 4, 3, 4, 3))
    for k in range(4):
        for m in range(4):
            acc = np.broadcast_to(I2, (npl, 1, 1, 2, 2))
            for i in range(4):
                if i == k and i == m:
                    fac = DD[:, i]                          # (np,3,3,2,2)
                elif i == k:
                    fac = D[:, i][:, :, None]               # (np,3,1,2,2)
                elif i == m:
                    fac = D[:, i][:, None, :]               # (np,1,3,2,2)
                else:
                    fac = F[:, i][:, None, None]
                acc = acc @ fac
            acc = np.broadcast_to(acc, (npl, 3, 3, 2, 2))
            h[:, k, :, m, :] = -0.5 * np.real(np.trace(acc, axis1=-2, axis2=-1))
    P = F[:, 0] @ F[:, 1] @ F[:, 2] @ F[:, 3]
    val = 1 - 0.5 * np.real(np.trace(P, axis1=-2, axis2=-1))
    return h, val


def assemble(geo, h, ncol=3, dense=True):
    """h (np,4,ncol,4,ncol) -> Hessienne globale (nl*ncol)^2."""
    dof = (ncol * geo.plaq[:, :, None] + np.arange(ncol)[None, None, :]).reshape(geo.np, 4 * ncol)
    rows = np.repeat(dof[:, :, None], 4 * ncol, axis=2).ravel()
    cols = np.repeat(dof[:, None, :], 4 * ncol, axis=1).ravel()
    vals = h.reshape(geo.np, 4 * ncol, 4 * ncol).ravel()
    n = geo.nl * ncol
    M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return M.toarray() if dense else M


# ----------------------------------------------------------------------------- Hessiennes des modèles
def hess_su2_additive(geo, A, dense=True, mutant_drop_same_link=False):
    """H/beta en coordonnées additives : A_l -> A_l + zeta_l.  A (nl,3)."""
    Ep, dEp, d2Ep = exp_derivs(A, +1.0)
    Em, dEm, d2Em = exp_derivs(A, -1.0)
    L = geo.plaq
    F = np.stack([Ep[L[:, 0]], Ep[L[:, 1]], Em[L[:, 2]], Em[L[:, 3]]], axis=1)
    D = np.stack([dEp[L[:, 0]], dEp[L[:, 1]], dEm[L[:, 2]], dEm[L[:, 3]]], axis=1)
    DD = np.stack([d2Ep[L[:, 0]], d2Ep[L[:, 1]], d2Em[L[:, 2]], d2Em[L[:, 3]]], axis=1)
    if mutant_drop_same_link:
        DD = np.zeros_like(DD)
    h, val = plaq_hess(F, D, DD)
    return assemble(geo, h, 3, dense), val


def hess_su2_left(geo, U, dense=True):
    """H_L/beta en coordonnées à gauche : U_l -> exp(i zeta_l.sigma/2) U_l.  U (nl,2,2)."""
    L = geo.plaq
    Ud = np.conj(np.swapaxes(U, -1, -2))
    Yp = 1j * SIG / 2
    F = np.stack([U[L[:, 0]], U[L[:, 1]], Ud[L[:, 2]], Ud[L[:, 3]]], axis=1)
    D = np.zeros(F.shape[:2] + (3, 2, 2), complex)
    DD = np.zeros(F.shape[:2] + (3, 3, 2, 2), complex)
    for k in range(4):
        for a in range(3):
            if k < 2:
                D[:, k, a] = Yp[a] @ F[:, k]
                DD[:, k, a, a] = -0.25 * F[:, k]
            else:
                D[:, k, a] = F[:, k] @ (-Yp[a])
                DD[:, k, a, a] = -0.25 * F[:, k]
    h, val = plaq_hess(F, D, DD)
    return assemble(geo, h, 3, dense), val


def qcov_su2(geo, U, dense=True):
    """Q_cov/beta = 1/4 sum_p J_p^T J_p, X_p = z1 + R(U1) z2 - R(U1U2U3^+) z3 - R(U_p) z4."""
    L = geo.plaq
    Ud = np.conj(np.swapaxes(U, -1, -2))
    U1, U2, U3d, U4d = U[L[:, 0]], U[L[:, 1]], Ud[L[:, 2]], Ud[L[:, 3]]
    W2 = U1; W3 = U1 @ U2 @ U3d; W4 = W3 @ U4d
    J = np.stack([np.broadcast_to(np.eye(3), (geo.np, 3, 3)), adj(W2), -adj(W3), -adj(W4)], axis=1)  # (np,4,3,3)
    h = 0.25 * np.einsum("pkca,pmcb->pkamb", J, J)
    return assemble(geo, h, 3, dense)


def hess_mutant(geo, A, dense=True):
    """mutant abélianisé : 1 - cos(|F|/2), F = A1+A2-A3-A4 (R^3).  H/beta."""
    L = geo.plaq
    Fp = (A[L[:, 0]] + A[L[:, 1]] - A[L[:, 2]] - A[L[:, 3]])
    E, dE, d2E = exp_derivs(Fp, 1.0)
    hF = -0.5 * np.real(np.trace(d2E, axis1=-2, axis2=-1))   # (np,3,3) Hessienne de 1-1/2 tr E(F)=1-cos(|F|/2)
    s = geo.sgn.astype(float)
    h = np.einsum("k,m,pab->pkamb", s, s, hF)
    val = 1 - np.cos(np.linalg.norm(Fp, axis=1) / 2)
    return assemble(geo, h, 3, dense), val


def hess_u1(geo, A1, dense=True):
    """U(1) : H/beta_U = R^T diag(cos F) R ; A1 (nl,)."""
    Fp = geo.R @ A1
    M = geo.R.T @ sp.diags(np.cos(Fp)) @ geo.R
    return (M.toarray() if dense else M.tocsr()), 1 - np.cos(Fp)


# ----------------------------------------------------------------------------- actions (valeurs exactes)
def action_su2_plaqs(geo, A, plist=None):
    E = su2_exp(A)
    Ed = np.conj(np.swapaxes(E, -1, -2))
    L = geo.plaq if plist is None else geo.plaq[plist]
    P = E[L[:, 0]] @ E[L[:, 1]] @ Ed[L[:, 2]] @ Ed[L[:, 3]]
    return 1 - 0.5 * np.real(np.trace(P, axis1=-2, axis2=-1))


# ----------------------------------------------------------------------------- jauge
def gauge_transform(geo, U, Om):
    """U_{x,mu} -> Om_x U_{x,mu} Om_{x+mu}^+ ; U (nl,2,2), Om (V,2,2)."""
    Uv = U.reshape(geo.V, 4, 2, 2)
    out = np.empty_like(Uv)
    for mu in range(4):
        out[:, mu] = Om @ Uv[:, mu] @ np.conj(np.swapaxes(Om[geo.sh[mu, 0]], -1, -2))
    return out.reshape(geo.nl, 2, 2)


def landau_fix(geo, U, tol=1e-26, maxit=20000, omega=1.7):
    """Fixation de Landau réseau (max sum Re tr U) par sur-relaxation, damier.  Rend (U_fixe, theta, it)."""
    Uv = U.reshape(geo.V, 4, 2, 2).copy()
    par = geo.coords.sum(1) % 2
    theta = np.inf
    for it in range(maxit):
        for p in (0, 1):
            xs = np.where(par == p)[0]
            h = np.zeros((len(xs), 2, 2), complex)
            for mu in range(4):
                h += Uv[xs, mu] + np.conj(np.swapaxes(Uv[geo.sh[mu, 1, xs], mu], -1, -2))
            # maximiser Re tr(g h) : g = h^+/sqrt(det h)
            g = np.conj(np.swapaxes(h, -1, -2))
            det = np.real(np.linalg.det(g))
            g = g / np.sqrt(np.maximum(det, 1e-300))[:, None, None]
            # sur-relaxation g^omega
            Ag = su2_log(g)
            g = su2_exp(omega * Ag)
            gd = np.conj(np.swapaxes(g, -1, -2))
            for mu in range(4):
                Uv[xs, mu] = g @ Uv[xs, mu]
                xm = geo.sh[mu, 1, xs]
                Uv[xm, mu] = Uv[xm, mu] @ gd
        if it % 20 == 0 or it == maxit - 1:
            Aal = su2_log(Uv.reshape(-1, 2, 2)).reshape(geo.V, 4, 3)
            # condition de Landau réseau : partie anti-hermitienne ~ sin(|A|/2) Ahat
            nA = np.linalg.norm(Aal, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                ah = np.where(nA > 0, np.sin(nA / 2) * Aal / np.where(nA > 0, nA, 1), 0.0)
            Dl = np.zeros((geo.V, 3))
            for mu in range(4):
                Dl += ah[:, mu] - ah[geo.sh[mu, 1], mu]
            theta = float(np.mean(np.sum(Dl ** 2, axis=1)))
            if theta < tol:
                break
    return Uv.reshape(geo.nl, 2, 2), theta, it


# ----------------------------------------------------------------------------- sous-espaces
def null_basis(Cmat, rtol=1e-10):
    C = Cmat.toarray() if sp.issparse(Cmat) else Cmat
    return sla.null_space(C, rcond=rtol)


def basis_fix(geo, fluct=True, ncol=3):
    """Base orthonormée (nl*ncol, m) de {div z = 0, z ⊥ constantes, [B z = 0]} (diag en couleur)."""
    rows = [geo.div.toarray()]
    if fluct:
        rows.append(geo.B.toarray())
    else:
        cst = np.zeros((4, geo.nl))
        for mu in range(4):
            cst[mu, mu::4] = 1.0
        rows.append(cst)
    V1 = null_basis(np.vstack(rows))
    return V1 if ncol == 1 else np.kron(V1, np.eye(ncol)), V1


def basis_cov(geo, U):
    """Base de {tau_U^T z = 0 (tranche covariante), B_U z = 0 (bloc covariant)}, coordonnées à gauche."""
    nl = geo.nl
    Uv = U.reshape(geo.V, 4, 2, 2)
    Radj = adj(U).reshape(geo.V, 4, 3, 3)
    T = np.zeros((3 * geo.V, 3 * nl))  # lignes = (x,c) ; tau(omega)_{x,mu} = omega_x - R(U_{x,mu}) omega_{x+mu}
    for x in range(geo.V):
        for mu in range(4):
            l = 4 * x + mu
            T[3 * x:3 * x + 3, 3 * l:3 * l + 3] += np.eye(3)
            xp = geo.sh[mu, 0, x]
            T[3 * xp:3 * xp + 3, 3 * l:3 * l + 3] += -Radj[x, mu].T
    Bm = np.zeros((3 * geo.nB, 3 * nl))
    for (row, x0, mu, y, xa, xb) in geo.block_terms:
        # transporteur de x0 à x0+y le long des axes croissants
        W = I2.copy(); x = x0
        for nu in range(4):
            if y[nu]:
                W = W @ Uv[x, nu]; x = geo.sh[nu, 0, x]
        assert x == xa
        RW = adj(W); RWU = adj(W @ Uv[xa, mu])
        la, lb = 4 * xa + mu, 4 * xb + mu
        Bm[3 * row:3 * row + 3, 3 * la:3 * la + 3] += RW / 8
        Bm[3 * row:3 * row + 3, 3 * lb:3 * lb + 3] += RWU / 8
    return null_basis(np.vstack([T, Bm]))


def gen_eig_extremes(H, Q, V):
    """lambda_min, lambda_max de (V^T H V, V^T Q V)."""
    Hr = V.T @ (H @ V)
    Qr = V.T @ (Q @ V)
    Hr = (Hr + Hr.T) / 2; Qr = (Qr + Qr.T) / 2
    w = sla.eigh(Hr, Qr, eigvals_only=True)
    return w[0], w[-1], w


# ----------------------------------------------------------------------------- fonds gaussiens
class Gauss:
    """Tirage A ~ N(0, (beta Q0)^{-1}) sur les modes transverses non constants, par couleur (scalaire).
    Q0/beta = c0 * K, K = R^T R ; c0 = 1/4 (SU(2), mutant), 1 (U(1)).  Variance par composante ∝ g^2."""
    def __init__(self, geo, c0):
        self.geo = geo
        K = (geo.R.T @ geo.R).toarray()
        Vt, _ = basis_fix(geo, fluct=False, ncol=1)
        Kt = Vt.T @ K @ Vt
        w, P = np.linalg.eigh(c0 * (Kt + Kt.T) / 2)
        assert w.min() > 1e-8
        self.Vt, self.w, self.P = Vt, w, P
        # espérance conditionnelle étant donné la moyenne de bloc : A_c = Kt^{-1} Bt^T (Bt Kt^{-1} Bt^T)^{-1} Bt A
        Bt = geo.B.toarray() @ Vt
        Ki = P @ np.diag(1 / w) @ P.T
        S = Bt @ Ki @ Bt.T
        self.Mcond = Ki @ Bt.T @ np.linalg.pinv(S, rcond=1e-12) @ Bt

    def draw(self, rng, invbeta, ncol, coarse):
        """cov(A) = (beta Q0/beta)^{-1} = invbeta * (c0 K)^{-1} ; SU(2): invbeta=g^2/4 ; U(1): invbeta=g^2."""
        cols = []
        for c in range(ncol):
            z = rng.standard_normal(len(self.w))
            at = self.P @ (z / np.sqrt(self.w)) * np.sqrt(invbeta)  # coordonnées transverses
            if coarse:
                at = self.Mcond @ at
            cols.append(self.Vt @ at)
        return np.stack(cols, axis=1)  # (nl, ncol)
