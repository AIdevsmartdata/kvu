#!/usr/bin/env python3
"""Fixtures de la lecture tordue (saturation v3) : Ft et Dtw = (D0, Dm, Dp) par torsion.py sur les cas de fibre.

Pour chaque cas `tests/donnees/fibre/<cas>/` qui a un `q.npy` (configurations Haar L = 4 et 8), calcule avec
`reference/saturation_v3/torsion.py` (copie à l'octet de celui du chantier) et `noyau2/cb.py` :
  tw_Ft.npy   (18,)          = torsion.F_tw(pv, L, psi), pv = cb.plaquettes(C, geo)[0]
  tw_Dtw.npy  (3, 18, 18)    = torsion.D_tw(Uf, geo, gp, prep, sup, psi)
Usage : /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python validation/cond/gen_torsion.py [--cas L4_haar0,...]
"""
import argparse, hashlib, json, os, sys

ICI = os.path.dirname(os.path.abspath(__file__))
KVU = os.path.abspath(os.path.join(ICI, '..', '..'))
NOYAU2 = os.environ.get('KVU_NOYAU2', '/chemin/vers/projet/outils/ym4_noyaux/noyau2')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'kevinotron', 'numba_noyau2'))
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
sys.dont_write_bytecode = True
sys.path.insert(0, NOYAU2)
sys.path.insert(0, os.path.join(KVU, 'reference', 'saturation_v3'))

import numpy as np
import cb
import torsion as T


def sha256(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def from_q(q):
    C = np.empty(q.shape[:-1] + (2, 2), complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]; C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]; C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return C


ap = argparse.ArgumentParser()
ap.add_argument('--cas', default='L4_haar0,L4_haar1,L4_haar2,L8_haar0,L8_haar1')
args = ap.parse_args()
attendu = open(os.path.join(KVU, 'reference', 'saturation_v3', 'SHA256SUMS')).read().split()[0]
assert sha256(T.__file__) == attendu, 'torsion.py différent de la copie de référence'
for cas in args.cas.split(','):
    d = os.path.join(KVU, 'tests', 'donnees', 'fibre', cas)
    q = np.load(os.path.join(d, 'q.npy'))
    L = q.shape[1]
    C = np.ascontiguousarray(from_q(q))
    geo = cb.Geometry(L)
    psi = T.psi_sites(L)
    pv, gp = cb.plaquettes(C, geo)
    Ft = T.F_tw(pv, L, psi)
    prep, sup = T.prep_tw(geo, T.poids_tw(L))
    Uf = cb._flat(C, geo).reshape(-1, 2, 2)
    Dtw = np.stack(T.D_tw(Uf, geo, gp, prep, sup, psi))
    np.save(os.path.join(d, 'tw_Ft.npy'), Ft)
    np.save(os.path.join(d, 'tw_Dtw.npy'), Dtw)
    print(json.dumps({'cas': cas, 'L': L, 'Ft0': float(Ft[0]), 'normes': [float(np.linalg.norm(x)) for x in Dtw],
                      'sha_torsion': attendu[:16], 'sha_cb': sha256(os.path.join(NOYAU2, 'cb.py'))[:16]}))
