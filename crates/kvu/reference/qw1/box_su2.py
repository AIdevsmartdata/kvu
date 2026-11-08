#!/usr/bin/env python3
"""QW1 -- SU(2) Wilson, boite de Dirichlet [0,l]^4 a liens de bord FIXES (CPU, numpy).

Conventions : S = beta * sum_p (1 - 1/2 Re Tr U_p), beta = 4/g^2 = 2 C_A / g^2 (C_A = 2 pour SU(2)).
U in SU(2) code en quaternion (a0,a1,a2,a3) <-> a0 + i a.sigma.
Lien U_mu(x) : de x a x+mu. Lien FIXE (bord) : ses deux extremites sont sur le bord du cube
(une coordonnee = 0 ou l). Tous les autres liens de la boite sont mis a jour (heat-bath
Kennedy-Pendleton + overrelaxation, damier). Mode periodique (--periodic) : aucun lien fixe,
memes routines, sert au gate G1 contre le generateur publie et a tirer des bords thermiques.

Mutants (variable d'environnement QW1_MUTANT) : M1 signe du heat-bath inverse ; M2 liens de bord
aussi mis a jour ; M3 agrafe arriere omise.
"""
import os, sys, json, time, argparse
import numpy as np

MUT = os.environ.get("QW1_MUTANT", "")


def qmul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.stack([a0*b0 - a1*b1 - a2*b2 - a3*b3,
                     a0*b1 + b0*a1 - (a2*b3 - a3*b2),
                     a0*b2 + b0*a2 - (a3*b1 - a1*b3),
                     a0*b3 + b0*a3 - (a1*b2 - a2*b1)])


def qdag(a):
    return np.stack([a[0], -a[1], -a[2], -a[3]])


def sh(a, nu, s):
    """valeur en x + s*nu (a de forme (4, n,n,n,n))."""
    return np.roll(a, -s, axis=nu + 1)


def staple(U, mu):
    A = np.zeros_like(U[mu])
    for nu in range(4):
        if nu == mu:
            continue
        # avant : U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
        A += qmul(qmul(sh(U[nu], mu, 1), qdag(sh(U[mu], nu, 1))), qdag(U[nu]))
        if MUT != "M3":
            # arriere : U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)
            Unm = sh(U[nu], nu, -1)
            A += qmul(qmul(qdag(sh(Unm, mu, 1)), qdag(sh(U[mu], nu, -1))), Unm)
    return A


def haar_rej_x0(alpha, rng):
    """meme densite, par rejet depuis Haar (x0 = 1re composante d'un point uniforme de S^3),
    acceptation exp(alpha (x0 - 1)) : exact pour tout alpha >= 0, efficace pour alpha petit (y compris 0)."""
    out = np.empty_like(alpha)
    todo = np.arange(alpha.size)
    while todo.size:
        v = rng.normal(size=(4, todo.size))
        x0 = v[0] / np.linalg.norm(v, axis=0)
        ok = rng.random(todo.size) <= np.exp(alpha[todo] * (x0 - 1.0))
        out[todo[ok]] = x0[ok]
        todo = todo[~ok]
    return out


ALPHA_SMALL = 1.0


def kp_x0(alpha, rng):
    """x0 de densite ~ sqrt(1-x0^2) exp(alpha x0), vectorise : Kennedy-Pendleton pour alpha >= 1,
    rejet depuis Haar pour alpha < 1 (KP diverge quand alpha -> 0 : agrafe nulle, cas reel du bord flux)."""
    out = np.empty_like(alpha)
    small = alpha < ALPHA_SMALL
    if small.any():
        out[small] = haar_rej_x0(alpha[small], rng)
    todo = np.nonzero(~small)[0]
    rounds = 0
    while todo.size:
        rounds += 1
        if rounds > 100000:
            raise RuntimeError("plafond heat-bath atteint (alpha ~ 0)")
        al = alpha[todo]
        r1 = 1.0 - rng.random(todo.size)
        r2 = rng.random(todo.size)
        r3 = 1.0 - rng.random(todo.size)
        r4 = rng.random(todo.size)
        lam2 = -(np.log(r1) + np.cos(2*np.pi*r2)**2 * np.log(r3)) / (2.0*al)
        ok = r4**2 <= 1.0 - lam2
        out[todo[ok]] = 1.0 - 2.0*lam2[ok]
        todo = todo[~ok]
    return out


def heatbath_draw(alpha, rng):
    x0 = kp_x0(alpha, rng)
    if MUT == "M1":
        x0 = -x0
    v = rng.normal(size=(3, alpha.size))
    v /= np.linalg.norm(v, axis=0)
    s = np.sqrt(np.clip(1 - x0**2, 0, None))
    return np.vstack([x0, s*v])


class Box:
    def __init__(self, l, beta, periodic=False):
        self.l, self.beta, self.per = l, beta, periodic
        n = l if periodic else l + 1
        self.n = n
        g = np.indices((n,)*4)               # (4, n,n,n,n)
        self.coords = g
        par = g.sum(0) % 2
        self.upd = []
        for mu in range(4):
            if periodic:
                m = np.ones((n,)*4, bool)
            else:
                inside = g[mu] <= l - 1       # le lien existe dans la boite
                onb_x = ((g == 0) | (g == l)).any(0)
                gp = g.copy(); gp[mu] += 1
                onb_y = ((gp == 0) | (gp == l)).any(0)
                fixed = onb_x & onb_y
                m = inside & ~fixed
                if MUT == "M2":
                    m = inside.copy()
            self.upd.append([m & (par == p) for p in (0, 1)])
        self.U = np.zeros((4, 4) + (n,)*4)
        self.U[:, 0] = 1.0

    def nupd(self):
        return sum(int(m[0].sum() + m[1].sum()) for m in self.upd)

    def sweep(self, rng, nor=2):
        b = self.beta
        for mu in range(4):
            for p in (0, 1):
                m = self.upd[mu][p]
                A = staple(self.U, mu)[:, m]
                k = np.linalg.norm(A, axis=0)
                k = np.maximum(k, 1e-300)  # k = 0 : alpha = 0 -> Haar exact, What arbitraire
                What = A / k
                X = heatbath_draw(b * k, rng)
                self.U[mu][:, m] = qmul(X, qdag(What))
        for _ in range(nor):
            for mu in range(4):
                for p in (0, 1):
                    m = self.upd[mu][p]
                    A = staple(self.U, mu)[:, m]
                    k = np.maximum(np.linalg.norm(A, axis=0), 1e-300)
                    Wd = qdag(A / k)
                    self.U[mu][:, m] = qmul(qmul(Wd, qdag(self.U[mu][:, m])), Wd)
        # renormalisation (arrondi)
        # (seulement les liens tires : les liens de bord ne doivent pas bouger d'un ulp, gate G2)
        for mu in range(4):
            m = self.upd[mu][0] | self.upd[mu][1]
            V = self.U[mu][:, m]
            self.U[mu][:, m] = V / np.linalg.norm(V, axis=0)

    def plaq_all(self):
        """P[mu,nu] = 1/2 Tr U_p(x) pour mu<nu ; tableau (6, n,n,n,n)."""
        U = self.U
        out = []
        for mu in range(4):
            for nu in range(mu + 1, 4):
                q = qmul(qmul(U[mu], sh(U[nu], mu, 1)), qmul(qdag(sh(U[mu], nu, 1)), qdag(U[nu])))
                out.append(q[0])
        return np.array(out)

    def wloop(self, x, mu, nu, a):
        """boucle a x a de coin x dans le plan (mu,nu) ; 1/2 Tr."""
        U = self.U
        q = np.array([1.0, 0, 0, 0])
        y = list(x)
        def link(y, d, fwd):
            if fwd:
                return U[d][:, y[0], y[1], y[2], y[3]]
            z = list(y); z[d] -= 1
            return qdag(U[d][:, z[0], z[1], z[2], z[3]])
        for d, fwd in [(mu, 1)]*a + [(nu, 1)]*a + [(mu, 0)]*a + [(nu, 0)]*a:
            q = qmul(q, link(y, d, fwd))
            y[d] += 1 if fwd else -1
        assert y == list(x)
        return q[0]


PLANES = [(mu, nu) for mu in range(4) for nu in range(mu + 1, 4)]


def set_boundary(box, kind, rng, thermal_src=None):
    """Fixe les liens de bord (mode boite). Les liens interieurs partent froids (=1)."""
    l, n = box.l, box.n
    g = box.coords
    U = np.zeros((4, 4) + (n,)*4); U[:, 0] = 1.0
    if kind == "cold":
        pass
    elif kind == "gauge":          # jauge pure aleatoire : g(x) g(x+mu)^+  (equivalent a cold)
        G = rng.normal(size=(4,) + (n,)*4); G /= np.linalg.norm(G, axis=0)
        for mu in range(4):
            U[mu] = qmul(G, qdag(sh(G, mu, 1)))
    elif kind == "z2full":         # toutes les plaquettes de bord = -1
        for mu in range(4):
            s = (-1.0) ** (g[:mu].sum(0) if mu > 0 else 0 * g[0])
            U[mu, 0] = s
    elif kind == "z2_01":          # torsion de centre dans les plans (0,1)
        U[1, 0] = (-1.0) ** g[0]
    elif kind == "flux":           # rotations fixees non commutantes, phi = pi/2
        phi = np.pi / 2
        U[1, 0] = np.cos(phi * g[0]); U[1, 3] = np.sin(phi * g[0])
        U[3, 0] = np.cos(phi * g[2]); U[3, 1] = np.sin(phi * g[2])
    elif kind.startswith("haar"):
        U = rng.normal(size=(4, 4) + (n,)*4); U /= np.linalg.norm(U, axis=1, keepdims=True)
    elif kind.startswith("therm"):
        src = np.load(thermal_src)   # (4,4,L,L,L,L) periodique
        sl = (slice(None), slice(None)) + (slice(0, n),)*4
        U = src[sl].copy()
    else:
        raise ValueError(kind)
    # interieur froid, bord impose
    for mu in range(4):
        m = box.upd[mu][0] | box.upd[mu][1]
        U[mu][:, m] = np.array([1.0, 0, 0, 0])[:, None]
    box.U = U


def measure(box, P=None):
    l = box.l
    if P is None:
        P = box.plaq_all()
    c = l // 2
    vals = []
    for ip, (mu, nu) in enumerate(PLANES):
        for a in (0, 1):
            for b in (0, 1):
                x = [c]*4; x[mu] -= a; x[nu] -= b
                vals.append(P[ip, x[0], x[1], x[2], x[3]])
    pc = float(np.mean(vals))
    w2 = np.nan
    if l >= 4:
        ws = []
        for mu, nu in PLANES:
            x = [c]*4; x[mu] -= 1; x[nu] -= 1
            ws.append(box.wloop(x, mu, nu, 2))
        w2 = float(np.mean(ws))
    return pc, w2


def depth_classes(l):
    """profondeur d'une plaquette (base x, plan mu,nu) = min sur ses coins de la distance au bord."""
    n = l + 1
    g = np.indices((n,)*4)
    dist = np.minimum(g, l - g).min(0)
    D = np.full((6,) + (n,)*4, -1)
    for ip, (mu, nu) in enumerate(PLANES):
        ok = (g[mu] <= l - 1) & (g[nu] <= l - 1)
        d = np.minimum.reduce([dist, sh(dist[None], mu, 1)[0], sh(dist[None], nu, 1)[0],
                               sh(sh(dist[None], mu, 1), nu, 1)[0]])
        D[ip][ok] = d[ok]
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--bc", default="cold")
    ap.add_argument("--thermal-src", default=None)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--therm", type=int, default=300)
    ap.add_argument("--nmeas", type=int, default=1000)
    ap.add_argument("--nor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None, help="periodique : sauver la config finale (.npy)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    box = Box(a.l, a.beta, periodic=a.periodic)
    if not a.periodic:
        set_boundary(box, a.bc, np.random.default_rng(1000 + a.seed), a.thermal_src)
        # masque des liens de bord calcule independamment de box.upd (sinon le mutant M2 est invisible a G2)
        g = box.coords
        onb = ((g == 0) | (g == a.l)).any(0)
        fixed_mask = []
        for mu in range(4):
            gp = g.copy(); gp[mu] += 1
            fixed_mask.append((g[mu] <= a.l - 1) & onb & ((gp == 0) | (gp == a.l)).any(0))
        U0fix = np.concatenate([box.U[mu][:, fixed_mask[mu]].ravel() for mu in range(4)])
        D = depth_classes(a.l)
        nd = a.l // 2 + 1
    t0 = time.time()
    for _ in range(a.therm):
        box.sweep(rng, a.nor)
    rows = []
    for _ in range(a.nmeas):
        box.sweep(rng, a.nor)
        P = box.plaq_all()
        if a.periodic:
            rows.append([float(P.mean())])
        else:
            pc, w2 = measure(box, P)
            prof = [float(P[D == d].mean()) for d in range(nd)]
            rows.append([pc, w2] + prof)
    res = {"args": vars(a), "mutant": MUT, "seconds": time.time() - t0, "nupd": box.nupd(),
           "rows": rows}
    if not a.periodic:
        U1fix = np.concatenate([box.U[mu][:, fixed_mask[mu]].ravel() for mu in range(4)])
        res["boundary_max_change"] = float(np.abs(U1fix - U0fix).max())
        res["depth_counts"] = [int((D == d).sum()) for d in range(nd)]
    if a.dump:
        np.save(a.dump, box.U)
    with open(a.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
