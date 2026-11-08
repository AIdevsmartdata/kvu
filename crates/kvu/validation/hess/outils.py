"""Outils communs des validations WP13 (python3 -I du venv épinglé /chemin/vers/projet/outils/ym4_noyaux/venv).
Aucune gate ne se juge ici : ce module fabrique les fonds, appelle `kvu` et écrit les verdicts bruts."""
import json, os, subprocess, sys, hashlib, time
import numpy as np

KVU = os.environ.get('KVU_BIN', '/chemin/vers/projet/chantiers/kevinotron_unification/bin/wp13/acemagic/kvu')
YM4 = '/chemin/vers/projet/outils/ym4_noyaux/noyau3'
LATDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reference', 'hess')  # copie hachée de S2/lat.py (pas de substitution possible)
GRANDL = '/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production'
SCRATCH = os.environ.get('KVU_SCRATCH', '/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp13')
FILS = os.environ.get('KVU_FILS', '4')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def quat_vers_c16(q):
    """q (..., 4) (a0,a1,a2,a3) -> U (...,2,2) : [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] (convention kvu/ym4)."""
    U = np.zeros(q.shape[:-1] + (2, 2), complex)
    U[..., 0, 0] = q[..., 0] + 1j * q[..., 3]
    U[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    U[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]
    U[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return U


def haar_c16(L, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(4, L, L, L, L, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return quat_vers_c16(q)


def champ_vers_liens_lat(C):
    """C (4,L,L,L,L,2,2) -> (nl,2,2), lien lat l = 4*idx + mu."""
    L = C.shape[1]
    nl = 4 * L ** 4
    out = np.zeros((nl, 2, 2), complex)
    for mu in range(4):
        out[mu::4] = C[mu].reshape(-1, 2, 2)
    return out


def liens_lat_vers_champ(Ul, L):
    C = np.zeros((4, L, L, L, L, 2, 2), complex)
    for mu in range(4):
        C[mu] = Ul[mu::4].reshape(L, L, L, L, 2, 2)
    return C


def perm_ym4_vers_s2(L):
    """perm[i_s2] = i_ym4 : composante s2 3(4x+mu)+a = composante ym4 3(mu V + x)+a."""
    V = L ** 4
    perm = np.empty(12 * V, int)
    for mu in range(4):
        for x in range(V):
            for a in range(3):
                perm[3 * (4 * x + mu) + a] = 3 * (mu * V + x) + a
    return perm


def ecrire_npy(chemin, arr):
    np.save(chemin, np.ascontiguousarray(arr))


def kvu(args, **kw):
    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': FILS, 'RAYON_NUM_THREADS': FILS})
    cmd = ['nice', '-n', '10', KVU] + [str(a) for a in args]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError('kvu %s : rc=%d\n%s\n%s' % (' '.join(map(str, args)), r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    return r


def verdict(f, ident, ok, detail, statut=None):
    """Écrit une ligne PASS|FAIL <ID> <détail> sur f et la sortie standard."""
    s = ('PASS' if ok else 'FAIL') if statut is None else statut
    ligne = '%s %s %s' % (s, ident, detail)
    print(ligne, flush=True)
    if f is not None:
        f.write(ligne + '\n'); f.flush()
    return ok


def charger_ym4():
    sys.path.insert(0, YM4)
    import ym4_hess as H
    return H


def charger_lat():
    sys.path.insert(0, LATDIR)
    import lat
    return lat


def fin(*valeurs):
    """Assertion de finitude AVANT toute agrégation (avenant A6) : scalaires, tableaux ou listes."""
    for v in valeurs:
        a = np.asarray(v, dtype=float)
        assert np.isfinite(a).all(), 'valeur non finie : %r' % (v,)
    return True


def rapport_lmin(chemin):
    """Charge un rapport JSON de `kvu hess lmin` et exige solveur.converge is True (méthode lobpcg) et des valeurs finies."""
    R = json.load(open(chemin))
    sol = R['solveur']
    assert sol.get('methode') != 'lobpcg' or sol.get('converge') is True, 'LOBPCG non convergé : %s' % chemin
    fin(R['lambda_min'], R['eps_prime'], R['residu_projete_rel'])
    return R
