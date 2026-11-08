#!/usr/bin/env python3
"""WP11 / FIB-MUT-PUISSANCE (INFO, pour WP12b) : taille de l'effet du facteur de coaire sur <S> = beta sum_p <1 - P_p> à l'ordre
dominant en 1/beta, SU(2), 6^4, V = 1 (docs/FIBRE_COAIRE.md §6.4).

Avec F = 1/2 log det JJ^T ~ 1/4 A^T H_L A (pas de terme linéaire, t6) et cov(A) = Gam/(2 beta) (x) 1_3 (tranche) :
    <S>_correct - <S>_mutant = - Cov(S, F) = - Tr((Gam (x) 1_3) H_L)/(8 beta) + O(beta^-2),   <S> -> 3 dim(tranche)/2 = 5760.
Tr((Gam (x) 1) H_L) estimée par Hutchinson : a = (Gam^1/2 (x) 1) z, a^T H_L a = d^2/ds^2 log det JJ^T(exp(s a)) (différences centrées).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import balaban_np as B
import operateurs as O

t0 = time.time()
ops = O.Ops(6); bloc = ops.bloc
coarse = [(z, mu) for mu in range(4) for z in np.ndindex(2, 2, 2, 2)]
TAB = {(z, mu): bloc.tables(z, mu) for z, mu in coarse}
def cidx(z, mu): return mu * ops.Vc + ((z[0] * 2 + z[1]) * 2 + z[2]) * 2 + z[3]
def logdet(q):
    J = np.zeros((3 * ops.n, 3 * ops.N))
    for z, mu in coarse:
        c = cidx(z, mu); liens, Jb, _ = bloc.jac(q, z, mu, TAB[(z, mu)])
        for i, l in enumerate(liens):
            J[3 * c:3 * c + 3, 3 * l:3 * l + 3] = Jb[i]
    return np.linalg.slogdet(J @ J.T)[1]
G = O.Gauss(ops)
Lc = np.linalg.cholesky(G.MVi)                       # MVi = L L^T  =>  Gam = (Bs L)(Bs L)^T
Rt = G.Bs @ Lc
rng = np.random.default_rng(26102619)
f0 = logdet(np.tile([1.0, 0, 0, 0], (ops.N, 1)))
s = 0.01
vals = []
for k in range(16):
    a = np.stack([Rt @ rng.normal(size=Rt.shape[1]) for _ in range(3)], 1)   # (N, 3) : a ~ N(0, Gam (x) 1)
    fp = logdet(B.qexp(s * a)); fm = logdet(B.qexp(-s * a))
    vals.append((fp + fm - 2 * f0) / s**2)
vals = np.array(vals)
tr, etr = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
print(f'INFO FIB-MUT-HUTCHINSON Tr((Gam (x) 1) H_L) = {tr:.1f} +- {etr:.1f} (16 vecteurs, s = {s})', flush=True)
for beta in (1000.0, 16.0, 8.0, 2.8):
    d = -tr / (8 * beta); rel = d / 5760.0
    print(f'INFO FIB-MUT-EFFET beta = {beta:g} : <S>_correct - <S>_mutant ~ {d:+.3f} +- {etr/(8*beta):.3f} sur <S> = 5760 : relatif {100*rel:+.3f} % (+- {100*etr/(8*beta*5760):.3f} %) [ordre dominant en 1/beta ; non fiable à beta ~ 3]', flush=True)
print(f'INFO FIB-MUT-DUREE {time.time() - t0:.0f} s')
