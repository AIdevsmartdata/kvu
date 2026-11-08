#!/usr/bin/env python3
"""Fixtures de référence de cb.py (noyau2) pour les gates du bain conditionnel (T5).

Usage : gen_fixtures.py <répertoire_de_sortie> [--niveau petit|complet]
  petit    : géométrie L = 4 et 8, séquences numpy, hb_numpy, balayages ref_rng à L = 4 (≤ 5 Mo : commitables)
  complet  : petit + L = 8 (configs {plate, π-flux, Haar×3} et 2 configs de production L8) : ≥ 100 cas (TEMPORAIRE, jamais commité)

Environnement : venv /chemin/vers/projet/outils/ym4_noyaux/venv, NUMBA_NUM_THREADS=4, nice -n 10. cb.py est lu EN PLACE
(/chemin/vers/projet/outils/ym4_noyaux/noyau2), son sha256 est vérifié avant tout calcul et consigné dans index.json.
Format : tableaux .npy float64 / int64 (lus par kvu::cond::npy_min), index.json pour les métadonnées.
Convention de quaternion : q = (a0, a1, a2, a3) pour U = [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] ;
champ q64 (4, L, L, L, L, 4), lien li = mu·V + s.
"""
import os, sys, json, hashlib, argparse
import numpy as np

NOYAU = '/chemin/vers/projet/outils/ym4_noyaux/noyau2'
sys.path.insert(0, NOYAU)
sys.dont_write_bytecode = True

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--niveau', default='petit', choices=['petit', 'complet'])
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)


def sha256_fichier(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# cb.py doit être celui du code v3 (SHA_code_v3.txt)
SHA_CB = sha256_fichier(os.path.join(NOYAU, 'cb.py'))
ATTENDU = open(os.path.join(NOYAU, 'logs', 'SHA_code_v3.txt')).read().split('cb.py')[0].split()[-1]
assert SHA_CB == ATTENDU, f'cb.py {SHA_CB} != code v3 {ATTENDU}'
import cb  # noqa: E402


def sauve(nom, a):
    np.save(os.path.join(OUT, nom), a)


# ------------------------------------------------------------------------------------------ conversions q <-> C
def from_q(q):
    C = np.empty(q.shape[:-1] + (2, 2), complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]
    C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]
    C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return C


def to_q(C):
    q = np.empty(C.shape[:-2] + (4,))
    q[..., 0] = 0.5 * (C[..., 0, 0].real + C[..., 1, 1].real)
    q[..., 3] = 0.5 * (C[..., 0, 0].imag - C[..., 1, 1].imag)
    q[..., 2] = 0.5 * (C[..., 0, 1].real - C[..., 1, 0].real)
    q[..., 1] = 0.5 * (C[..., 0, 1].imag + C[..., 1, 0].imag)
    return q


def cfg_flat(L):
    q = np.zeros((4,) + (L,) * 4 + (4,)); q[..., 0] = 1.0
    return q


def cfg_pi(L):
    """= flot_su2.cfg_frustre : liens ±1, toutes les plaquettes = −1."""
    x = np.indices((L,) * 4)
    eps = [np.ones((L,) * 4), (-1.0) ** x[0], (-1.0) ** (x[0] + x[1]), (-1.0) ** (x[0] + x[1] + x[2])]
    q = np.zeros((4,) + (L,) * 4 + (4,))
    for mu in range(4):
        q[mu, ..., 0] = eps[mu]
    return q


def cfg_haar(L, seed):
    q = np.random.default_rng(1000 + seed).standard_normal((4,) + (L,) * 4 + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def cfg_prod_L8(nom):
    """Configuration de production L8 (data/L8, SHA vérifié par test_statistique) → quaternions."""
    D8 = os.path.join(NOYAU, 'data', 'L8')
    a = np.load(os.path.join(D8, nom + '_cfg.npy'))
    if a.ndim == 6:   # stockage (4,L,L,L,L,2) : première ligne de la matrice
        C = np.empty(a.shape[:5] + (2, 2), dtype=complex)
        C[..., 0, :] = a; C[..., 1, 0] = -np.conj(a[..., 1]); C[..., 1, 1] = np.conj(a[..., 0])
    else:
        C = np.array(a)
    q = to_q(C)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


index = dict(cb_sha256=SHA_CB, numpy=np.__version__, niveau=args.niveau, geo={}, rng=[], hb=[], configs={}, cas=[])

# ------------------------------------------------------------------------------------------ géométrie
for L in (4, 8):
    g = cb.Geometry(L)
    info = dict(L=L, n_fib=int(g.n_fib), npair=int(g.p1.shape[1]))
    sauve(f'geo_L{L}_p1.npy', g.p1.astype(np.int64))
    sauve(f'geo_L{L}_p1pm.npy', g.p1pm.astype(np.int64))
    for mu in range(4):
        sauve(f'geo_L{L}_free_all_{mu}.npy', g.free_all[mu].astype(np.int64))
        for p in (0, 1):
            sauve(f'geo_L{L}_fr_{mu}_{p}.npy', g.fr[mu][p].astype(np.int64))
    sauve(f'geo_L{L}_fib_free.npy', g.fib_free.astype(np.int64))
    sauve(f'geo_L{L}_fib_e1.npy', g.fib_e1.astype(np.int64))
    sauve(f'geo_L{L}_fib_e2.npy', g.fib_e2.astype(np.int64))
    index['geo'][str(L)] = info

# ------------------------------------------------------------------------------------------ séquences numpy
SEEDS_RNG = (0, 1, 2709, 2 ** 63 - 1)
N_SEQ = 100000
for seed in SEEDS_RNG:
    rng = np.random.default_rng(seed)
    st0 = rng.bit_generator.state['state']
    ss = np.random.SeedSequence(seed)
    rec = dict(seed=str(seed), pool=[int(v) for v in ss.pool], gen_state_u64=[str(int(v)) for v in ss.generate_state(4, np.uint64)],
               etat0=str(st0['state']), inc0=str(st0['inc']))
    u = rng.random(N_SEQ)
    st = rng.bit_generator.state['state']
    rec.update(sha_random=hashlib.sha256(u.tobytes()).hexdigest(), etat_random=str(st['state']), inc_random=str(st['inc']))
    sauve(f'rng_{seed}_random_1000.npy', u[:1000])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N_SEQ)
    st = rng.bit_generator.state['state']
    rec.update(sha_normal=hashlib.sha256(z.tobytes()).hexdigest(), etat_normal=str(st['state']), inc_normal=str(st['inc']))
    sauve(f'rng_{seed}_normal_1000.npy', z[:1000])
    # séquence mixte : random() et standard_normal() alternés par blocs (états mélangés)
    rng = np.random.default_rng(seed)
    mix = []
    for i in range(2000):
        mix.append(rng.random() if i % 3 else rng.standard_normal())
    sauve(f'rng_{seed}_mixte_2000.npy', np.array(mix))
    rec['etat_mixte'] = str(rng.bit_generator.state['state']['state'])
    index['rng'].append(rec)

# ------------------------------------------------------------------------------------------ hb_numpy
HB = [('un', np.array([3.7])),
      ('log300', np.geomspace(0.01, 200.0, 300)),
      ('bas400', np.linspace(0.01, 2.0, 400)),
      ('alea2000', np.random.default_rng(77).uniform(0.01, 200.0, 2000)),
      # au-delà du seuil de calcul parallèle de kvu (4096) : exerce les sauts de PCG et le tampon de ziggurat
      ('grand8192', np.random.default_rng(78).uniform(0.01, 200.0, 8192))]
for nom, alpha in HB:
    rng = np.random.default_rng(5)
    x0, r, v = cb._hb_numpy(alpha, rng)
    sauve(f'hb_{nom}_alpha.npy', alpha); sauve(f'hb_{nom}_x0.npy', x0); sauve(f'hb_{nom}_r.npy', r); sauve(f'hb_{nom}_v.npy', v)
    st = rng.bit_generator.state['state']
    index['hb'].append(dict(nom=nom, n=len(alpha), graine=5, etat=str(st['state']), inc=str(st['inc'])))

# ------------------------------------------------------------------------------------------ balayages ref_rng
BETAS = (2.3, 2.8, 16.0)
GRAINES = (0, 1, 2)


def cas_balayages(L, nom, q):
    geo = cb.Geometry(L)
    C0 = np.ascontiguousarray(from_q(q))
    Vr = cb.V_pairs(C0, geo)
    f = f'cfg_L{L}_{nom}.npy'
    sauve(f, q)
    sauve(f'vp_L{L}_{nom}.npy', to_q(Vr))
    index['configs'][f'L{L}_{nom}'] = f
    for beta in BETAS:
        for seed in GRAINES:
            C = C0.copy(); rng = np.random.default_rng(seed)
            cb.sweep(C, beta, rng, geo, Vr, mode='ref_rng')
            st1 = rng.bit_generator.state['state']
            fo = f'out_L{L}_{nom}_b{beta}_s{seed}.npy'
            sauve(fo, to_q(C))
            for _ in range(4):
                cb.sweep(C, beta, rng, geo, Vr, mode='ref_rng')
            st5 = rng.bit_generator.state['state']
            sauve(f'out5_L{L}_{nom}_b{beta}_s{seed}.npy', to_q(C)) if L == 4 else None
            index['cas'].append(dict(L=L, cfg=nom, beta=beta, seed=seed, cfg_file=f, out=fo, vp=f'vp_L{L}_{nom}.npy',
                                     etat1=str(st1['state']), inc1=str(st1['inc']), etat5=str(st5['state']), inc5=str(st5['inc'])))


for nom, q in (('plat', cfg_flat(4)), ('pi', cfg_pi(4)), ('haar0', cfg_haar(4, 0)), ('haar1', cfg_haar(4, 1)), ('haar2', cfg_haar(4, 2))):
    cas_balayages(4, nom, q)
if args.niveau == 'complet':
    for nom, q in (('plat', cfg_flat(8)), ('pi', cfg_pi(8)), ('haar0', cfg_haar(8, 0)), ('haar1', cfg_haar(8, 1)), ('haar2', cfg_haar(8, 2)),
                   ('prod_b2.3', cfg_prod_L8('L8_b2.3_v0')), ('prod_b16', cfg_prod_L8('L8_b16_v0'))):
        cas_balayages(8, nom, q)

json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print(f'fixtures : {len(index["cas"])} cas de balayage, {tot / 1e6:.2f} Mo dans {OUT}')
