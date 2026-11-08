#!/usr/bin/env python3
"""Boucles de Wilson n x n SU(2) sur un reseau periodique 4D, par paires complexes (a, b) : U = [[a, b], [-b*, a*]].

Entrees : .npy c16 (complex128, (4,L,L,L,L,2,2)) ou q64 (float64, (4,L,L,L,L,4), quaternion (a0,a1,a2,a3),
convention su2_hb::quat_to_mat de kvu : U = [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]]).
Les 4 premiers indices : direction du lien mu, puis x0..x3 (periodique).

x = (1/2) Tr W  (reel), alpha = arccos(x) in [0, pi] (valeur propre e^{+-i alpha}), Theta = 2 alpha in [0, 2 pi].
Seuils (PREREG) : voir SEUILS.
"""
import json
import math
import sys

import numpy as np

PLANS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Seuils sur x = 1/2 Tr W (angle de flux Theta = 2 arccos x ; Theta > T  <=>  x < cos(T/2))
SEUILS = {
    "T1_pi": 0.0,                         # Theta > pi        (Phi0 = 0.5 * 2pi/9, flux par face 9 Phi0 = pi)  <=> x < 0
    "T2_3pi2": -math.sqrt(0.5),           # Theta > 3 pi / 2  (Phi0 = 0.75 * 2pi/9)                           <=> x < -1/sqrt(2)
    "C32": -math.cos(math.pi / 32.0),     # Theta >= 2 pi (1 - 1/32)  (fermeture de Phi0 = 2pi/9, kappa = 1/32)  <=> x <= -cos(pi/32)
    "C16": -math.cos(math.pi / 16.0),     # kappa = 1/16
    "C8": -math.cos(math.pi / 8.0),       # kappa = 1/8
}
NBINS_ALPHA = 3600  # histogramme de alpha dans [0, pi], pas 0.05 degre


def lire_cfg(chemin):
    """Renvoie (a, b, L) : deux tableaux complex128 de forme (4, L, L, L, L)."""
    U = np.load(chemin, mmap_mode="r")
    if U.dtype == np.complex128 and U.ndim == 7 and U.shape[0] == 4 and U.shape[5:] == (2, 2):
        a = np.ascontiguousarray(U[..., 0, 0])
        b = np.ascontiguousarray(U[..., 0, 1])
    elif U.dtype == np.float64 and U.ndim == 6 and U.shape[0] == 4 and U.shape[5] == 4:
        q = np.asarray(U)
        a = q[..., 0] + 1j * q[..., 3]
        b = q[..., 2] + 1j * q[..., 1]
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
    else:
        raise ValueError(f"{chemin}: format inattendu {U.shape} {U.dtype}")
    L = a.shape[1]
    if a.shape[1:] != (L, L, L, L):
        raise ValueError(f"{chemin}: reseau non isotrope {a.shape}")
    return a, b, L


def mul(p, q):
    """(a1,b1)(a2,b2) = (a1 a2 - b1 conj(b2), a1 b2 + b1 conj(a2))."""
    a1, b1 = p
    a2, b2 = q
    return (a1 * a2 - b1 * np.conj(b2), a1 * b2 + b1 * np.conj(a2))


def dag(p):
    a, b = p
    return (np.conj(a), -b)


def shift(p, mu, n):
    """Valeur au site x + n e_mu : roll de -n selon l'axe mu."""
    return (np.roll(p[0], -n, axis=mu), np.roll(p[1], -n, axis=mu))


def lignes(a, b, n):
    """S^{(n)}_mu(x) = U_mu(x) U_mu(x+mu) ... U_mu(x+(n-1)mu), pour mu = 0..3."""
    S = []
    for mu in range(4):
        U = (a[mu], b[mu])
        cur = U
        for j in range(1, n):
            cur = mul(cur, shift(U, mu, j))
        S.append(cur)
    return S


def x_boucles(a, b, n=3):
    """x = 1/2 Tr W_{n x n}(x ; mu nu) pour les 6 plans et tous les sites : tableau (6, L^4) float64."""
    S = lignes(a, b, n)
    out = []
    for (mu, nu) in PLANS:
        P = mul(S[mu], shift(S[nu], mu, n))          # S_mu(x) S_nu(x + n mu)
        P = mul(P, dag(shift(S[mu], nu, n)))         # ... S_mu(x + n nu)^dagger
        P = mul(P, dag(S[nu]))                       # ... S_nu(x)^dagger
        out.append(P[0].real.reshape(-1))
    return np.stack(out)


def stats_x(x):
    """Statistiques d'un tableau x (6, N) de 1/2 Tr W : comptes aux seuils, somme, somme des carres, histogramme de alpha."""
    xf = np.clip(x.reshape(-1), -1.0, 1.0)
    N = xf.size
    comptes = {k: int(np.count_nonzero(xf < s)) if k in ("T1_pi", "T2_3pi2") else int(np.count_nonzero(xf <= s))
               for k, s in SEUILS.items()}
    alpha = np.arccos(xf)
    h, _ = np.histogram(alpha, bins=NBINS_ALPHA, range=(0.0, math.pi))
    # moments de l'angle Theta = 2 alpha (pour la comparaison au champ libre)
    theta = 2.0 * alpha
    return {
        "N": int(N),
        "comptes": comptes,
        "sx": float(xf.sum()),
        "sx2": float((xf * xf).sum()),
        "sTheta2": float((theta * theta).sum()),
        "sTheta4": float((theta ** 4).sum()),
        "hist_alpha": h.astype(np.int64),
        "xmin": float(xf.min()),
    }


def stats_cfg(chemin, ns=(1, 2, 3)):
    a, b, L = lire_cfg(chemin)
    res = {"L": L}
    for n in ns:
        x = x_boucles(a, b, n)
        res[f"n{n}"] = stats_x(x)
    return res


if __name__ == "__main__":
    r = stats_cfg(sys.argv[1], ns=tuple(int(t) for t in sys.argv[2:]) or (3,))
    for k, v in r.items():
        if isinstance(v, dict):
            v = dict(v)
            v.pop("hist_alpha", None)
        print(k, json.dumps(v))
