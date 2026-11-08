"""Références INDÉPENDANTES de kvu pour ker DŪ (L = 6) : J analytique (reference/hess/balaban_np.py, forme fermée de WP11) et générateurs de la jauge résiduelle
τ_U(λ)_{x,μ} = λ_x − R(U_{x,μ}) λ_{x+μ}, λ = e_{x,c}, x hors des centres 3z + (1,1,1,1). Disposition s2 : lien 4x + μ, indice 3 lien + a."""
import os, sys, numpy as np
d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, d); sys.path.insert(0, os.path.join(d, '..', '..', 'reference', 'hess'))
import balaban_np as B
L = 6; V = L ** 4; nl = 4 * V; n = 3 * nl; Lc = 2; Vc = Lc ** 4

def quat_champ(C):
    return np.stack([C[..., 0, 0].real, C[..., 0, 1].imag, C[..., 0, 1].real, C[..., 0, 0].imag], -1).reshape(nl, 4)   # lien champ mu*V + x

def construire(C):
    q = quat_champ(C)
    bl = B.Bloc(L)
    # index s2 : l = 4x + mu ; colonne 3 l + a
    J = np.zeros((3 * 4 * Vc, n))
    for mu in range(4):
        for zs in range(Vc):
            z = [(zs // Lc ** (3 - m)) % Lc for m in range(4)]
            liens, Jb, th = bl.jac(q, z, mu)
            ls = 4 * zs + mu
            for i, e in enumerate(liens):
                m2, x2 = divmod(int(e), V)
                l2 = 4 * x2 + m2
                J[3 * ls:3 * ls + 3, 3 * l2:3 * l2 + 3] += Jb[i]
    coord = np.array([[(s // L ** (3 - m)) % L for m in range(4)] for s in range(V)])
    centre = np.all(coord % 3 == 1, axis=1)
    def site(c): return ((c[0] % L * L + c[1] % L) * L + c[2] % L) * L + c[3] % L
    G = []
    for x in range(V):
        if centre[x]:
            continue
        for c in range(3):
            g = np.zeros(n)
            for mu in range(4):
                l = 4 * x + mu
                g[3 * l + c] += 1.0                       # λ_x en début de lien
            for mu in range(4):                           # liens qui finissent en x : −R(U_{x−μ,μ})^T... : τ(λ)_{x',μ} = λ_{x'} − R λ_{x'+μ}
                cx = coord[x].copy(); cx[mu] -= 1
                xm = site(cx); l = 4 * xm + mu
                R = B.Rad(q[mu * V + xm])
                g[3 * l:3 * l + 3] += -R[:, c]            # −R λ avec λ = e_c : colonne c
            G.append(g)
    G = np.array(G)
    return J, G

