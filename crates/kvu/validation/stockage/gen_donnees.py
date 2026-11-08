#!/usr/bin/env python3
"""Génère les petits jeux de test NPY de la tâche T8 avec NUMPY (référence externe du format).

Écrit dans crates/kvu/tests/donnees/stockage/ :
  c16_L2.npy, q64_L2.npy       champ SU(2) L = 2 (64 liens), unités de Hurwitz exactes (aucun arrondi)
  f64_3x2.npy, f32_5.npy, u32_2x2x2.npy, f64_scalaire.npy
Le test Rust refait les mêmes données et exige des fichiers IDENTIQUES À L'OCTET (en-têtes de numpy compris).
Régénération : python3 gen_donnees.py   (numpy >= 1.20)
"""
import os

import numpy as np

SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "donnees", "stockage")


def hurwitz():
    """Les 24 unités de Hurwitz : ±1 sur un axe (8) et (±1/2, ±1/2, ±1/2, ±1/2) (16). Normes exactes."""
    u = []
    for slot in range(4):
        for s in (1.0, -1.0):
            v = [0.0] * 4
            v[slot] = s
            u.append(v)
    for b in range(16):
        u.append([0.5 if not (b >> k) & 1 else -0.5 for k in range(4)])
    return u


def champ_quaternions(l):
    n = 4 * l ** 4
    u = hurwitz()
    return np.array([u[(i * 7 + 3) % 24] for i in range(n)], dtype=np.float64)


def vers_c16(q, l):
    # matrice [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] -> (re00, im00, re01, im01, re10, im10, re11, im11)
    m = np.empty((q.shape[0], 8), dtype=np.float64)
    m[:, 0], m[:, 1] = q[:, 0], q[:, 3]
    m[:, 2], m[:, 3] = q[:, 2], q[:, 1]
    m[:, 4], m[:, 5] = -q[:, 2], q[:, 1]
    m[:, 6], m[:, 7] = q[:, 0], -q[:, 3]
    return m.view(np.complex128).reshape(4, l, l, l, l, 2, 2)


def main():
    os.makedirs(SORTIE, exist_ok=True)
    l = 2
    q = champ_quaternions(l)
    np.save(os.path.join(SORTIE, "c16_L2.npy"), vers_c16(q, l))
    np.save(os.path.join(SORTIE, "q64_L2.npy"), q.reshape(4, l, l, l, l, 4))
    np.save(os.path.join(SORTIE, "f64_3x2.npy"), np.array([0.5, -1.25, 3.0, 4e-10, -0.0, 1e300]).reshape(3, 2))
    np.save(os.path.join(SORTIE, "f32_5.npy"), np.array([1.5, -2.25, 0.1, 3e38, -0.0], dtype=np.float32))
    np.save(os.path.join(SORTIE, "u32_2x2x2.npy"), (np.arange(8, dtype=np.uint32) * 123456789 % 4294967291).astype(np.uint32).reshape(2, 2, 2))
    np.save(os.path.join(SORTIE, "f64_scalaire.npy"), np.array(2.5))
    print("écrit dans", os.path.abspath(SORTIE), "avec numpy", np.__version__)


if __name__ == "__main__":
    main()
