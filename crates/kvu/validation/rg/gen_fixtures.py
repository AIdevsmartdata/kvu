#!/usr/bin/env python3
"""Fixtures RG-REG-PY (WP6) : 3 configurations SU(2) 9^4 (amplitude 0,6, graines numpy 7, 11, 13) et leur variable grossière
Ubar (81 sites x 4 directions = 324 liens grossiers) calculée par `g_nl.ubar` (S1, LECTURE SEULE).

Usage (venv épinglé, une seule fois) :
    /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python3 -I gen_fixtures.py <dossier_sortie>

`g_nl.py` exécute toutes ses gates à l'import et écrit dans V/res : on ne l'importe PAS. On exécute, depuis le fichier lu en lecture seule,
ses seules instructions de tête (jusqu'à `res = {}` exclu : imports, rng(7), U, SEQ, glob_seq, prod, ubar), avec les mêmes octets de source ;
aucun fichier n'est écrit dans V (bytecode désactivé).

Sorties : cfg_s<graine>_q64.npy (4,9,9,9,9,4) float64, quaternions [a0,a1,a2,a3] (U = a0 + i(a1 s1 + a2 s2 + a3 s3)),
ubar_s<graine>_q64.npy (4,3,3,3,3,4), et manifeste.json (sha256 de chaque fichier, de g_nl.py, de geom.py, versions).
Les indices : g_nl range lid(x, nu) = site(x)*4 + nu avec site C-order ; kvu range (nu, x0, x1, x2, x3) : on réordonne.
"""
import ast, hashlib, json, os, sys
sys.dont_write_bytecode = True
import numpy as np

V = '/chemin/vers/projet/chantiers/vision_20261025/calculs/S1_bloc_symetrique_L3'
GNL = os.path.join(V, 'calculs', 'g_nl.py')
GRAINES = [7, 11, 13]
NF = 9
LC = 3


def sha(chemin):
    return hashlib.sha256(open(chemin, 'rb').read()).hexdigest()


def charger_ubar():
    """Exécute les instructions de tête de g_nl.py (avant `res = {}`) dans un espace de noms isolé."""
    src = open(GNL).read()
    arbre = ast.parse(src)
    morceaux = []
    for noeud in arbre.body:
        if isinstance(noeud, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'res' for t in noeud.targets):
            break
        morceaux.append(noeud)
    ns = {'__file__': GNL, '__name__': 'g_nl_tete'}
    exec(compile(ast.Module(body=morceaux, type_ignores=[]), GNL, 'exec'), ns)
    return ns


def quat(M):
    """matrices (...,2,2) -> quaternions (...,4) : même formule que su2_hb::mat_to_quat."""
    a0 = (M[..., 0, 0].real + M[..., 1, 1].real) / 2
    a1 = (M[..., 0, 1].imag + M[..., 1, 0].imag) / 2
    a2 = (M[..., 0, 1].real - M[..., 1, 0].real) / 2
    a3 = (M[..., 0, 0].imag - M[..., 1, 1].imag) / 2
    return np.stack([a0, a1, a2, a3], axis=-1)


def ecrire(chemin, a):
    np.save(chemin, np.ascontiguousarray(a, dtype='<f8'))


def main():
    sortie = sys.argv[1]
    os.makedirs(sortie, exist_ok=True)
    ns = charger_ubar()
    QL = ns['QL']
    ubar = ns['ubar']
    # contrôle : la graine 7 de g_nl est celle de l'exécution de tête (mêmes octets)
    U7 = ns['U'].copy()
    manifeste = {'format': 'kvu-rg-fixtures/1', 'g_nl_sha256': sha(GNL), 'geom_sha256': sha(os.path.join(V, 'src', 'geom.py')),
                 'q3_lib_sha256': sha(os.path.join(V, 'bundle_q3', 'q3_lib.py')),
                 'numpy': np.__version__, 'python': sys.version.split()[0], 'amplitude': 0.6, 'tore_fin': NF, 'fichiers': {}}
    NLR = NF ** 4 * 4
    for graine in GRAINES:
        rng = np.random.default_rng(graine)
        U = QL.expm_su2(QL.alg(0.6 * rng.normal(size=(NLR, 3))))
        if graine == 7:
            assert np.array_equal(U, U7), 'graine 7 : U différent de celui de g_nl.py'
        ns['U'] = U
        # configuration : lid = site*4 + nu -> (nu, site)
        q = quat(U).reshape(NF, NF, NF, NF, 4, 4)           # (x0,x1,x2,x3,nu,comp)
        cfg = np.transpose(q, (4, 0, 1, 2, 3, 5))           # (nu,x0,x1,x2,x3,comp)
        # variable grossière
        ub = np.zeros((4, LC, LC, LC, LC, 4))
        for z0 in range(LC):
            for z1 in range(LC):
                for z2 in range(LC):
                    for z3 in range(LC):
                        for mu in range(4):
                            M = ubar(U, (z0, z1, z2, z3), mu)
                            ub[mu, z0, z1, z2, z3] = quat(M)
        nom_c = 'cfg_s%d_q64.npy' % graine
        nom_u = 'ubar_s%d_q64.npy' % graine
        ecrire(os.path.join(sortie, nom_c), cfg)
        ecrire(os.path.join(sortie, nom_u), ub)
        manifeste['fichiers'][nom_c] = sha(os.path.join(sortie, nom_c))
        manifeste['fichiers'][nom_u] = sha(os.path.join(sortie, nom_u))
        print('graine', graine, 'ok', 'norme max |q|-1 =', float(np.abs(np.linalg.norm(cfg, axis=-1) - 1).max()))
    json.dump(manifeste, open(os.path.join(sortie, 'manifeste.json'), 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
