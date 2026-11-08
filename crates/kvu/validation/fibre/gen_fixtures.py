#!/usr/bin/env python3
"""Génère les fixtures de la mesure de fibre (T6) à partir de noyau2/cb.py : cas de test_concordance.py.

Cas (14, plan de test_concordance.py) : L ∈ {4, 8} × {plat, pi_flux, haar0, haar1, haar2} ; L = 12 : haar0 et
production β = 2,3 ; L = 16 : production β = 2,3 et plat. Bonus : deux configurations de production L = 8
(β = 2,3 et 16). Les configurations de production sont LUES EN PLACE dans noyau2/data (SHA vérifiés).

Chaque cas écrit un répertoire <sortie>/<cas>/ avec :
  q.npy                 configuration (4,L,L,L,L,4) quaternions float64 (a0, a1, a2, a3)
  cb_pv.npy, cb_gp.npy  plaquettes (6V) et gradients (6V,4,3)                       = cb.plaquettes
  cb_F.npy, cb_D.npy, cb_blk.npy   (F dense Wf@pv, D, blk float32)                 = cb.mesure (Wf dense)
  cb_Fs.npy             F par somme compensée (Wf creux)                           = cb.mesure (Wf creux)
  cb_Fx.npy             somme exacte math.fsum de Wf[i]*pv
  cb_Flibre.npy, cb_Dlibre.npy                                                     = cb.mesure_libre
  ref_F.npy, ref_D.npy, ref_blk.npy, ref_Dlibre.npy   références lib_Cb (--avec-ref, défaut oui)
  meta.json
et un fichier <sortie>/Wf_L<L>.npz équivalent en .npy : wf_L<L>_{ptr,idx,val}.npy (CSR de fourier_weights_sparse).
`index.json` liste les cas. Mode --petit : jeu commité (L ≤ 8, ≤ 5 Mo) : plat et pi_flux sans q (reconstruits côté
Rust), gradients sous-échantillonnés à L = 8 (une plaquette sur 8), Wf de L = 8 sous-échantillonné (1 entrée sur 64).

Usage : python gen_fixtures.py --sortie DIR [--petit] [--Lmax 16] [--cas nom,...] [--sans-ref]
Environnement : venv ym4_noyaux ; KVU_NOYAU2 (défaut /chemin/vers/projet/outils/ym4_noyaux/noyau2).
"""
import argparse, hashlib, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--sortie', required=True)
ap.add_argument('--petit', action='store_true')
ap.add_argument('--Lmax', type=int, default=16)
ap.add_argument('--cas', default=None)
ap.add_argument('--sans-ref', action='store_true')
args = ap.parse_args()

NOYAU2 = os.environ.get('KVU_NOYAU2', '/chemin/vers/projet/outils/ym4_noyaux/noyau2')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CB = os.path.join(HERE, '..', '..', 'reference', 'noyau2', 'cb.py')
# cache numba hors du dépôt (cb.py a cache=True : sinon __pycache__ apparaît dans l'arbre de travail)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'kevinotron', 'numba_noyau2'))
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
sys.dont_write_bytecode = True
sys.path.insert(0, NOYAU2)
sys.path.insert(0, os.path.join(NOYAU2, 'ref', 'code'))

import numpy as np


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for bloc in iter(lambda: f.read(1 << 20), b''):
            h.update(bloc)
    return h.hexdigest()


# la copie de référence du dépôt et le cb.py lu doivent être identiques à l'octet
sha_cb = sha256(os.path.join(NOYAU2, 'cb.py'))
if os.path.exists(REPO_CB) and sha_cb != sha256(REPO_CB):
    sys.exit('ECHEC : cb.py lu (%s) différent de reference/noyau2/cb.py du dépôt' % NOYAU2)

import cb
import lib_Cb as R
import lib_brancheC as B

SHA_CFG = {l.split()[1]: l.split()[0] for l in open(os.path.join(NOYAU2, 'data', 'SHA_cfg.txt')) if l.strip()}


# ------------------------------------------------------------------------------- configurations
def from_q(q):
    C = np.empty(q.shape[:-1] + (2, 2), complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]; C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]; C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return C


def to_q(C):
    """Inverse de from_q ; sur une matrice SU(2) exacte : identité ; sinon projection par moyennes (comme mat_to_quat)."""
    q = np.empty(C.shape[:-2] + (4,))
    q[..., 0] = (C[..., 0, 0].real + C[..., 1, 1].real) / 2
    q[..., 1] = (C[..., 0, 1].imag + C[..., 1, 0].imag) / 2
    q[..., 2] = (C[..., 0, 1].real - C[..., 1, 0].real) / 2
    q[..., 3] = (C[..., 0, 0].imag - C[..., 1, 1].imag) / 2
    return q


def cfg_flat(L):
    return np.ascontiguousarray(np.broadcast_to(np.eye(2, dtype=complex), (4,) + (L,) * 4 + (2, 2)))


def cfg_pi(L):
    x = np.indices((L,) * 4)
    eps = [np.ones((L,) * 4), (-1.0) ** x[0], (-1.0) ** (x[0] + x[1]), (-1.0) ** (x[0] + x[1] + x[2])]
    C = cfg_flat(L).copy()
    for mu in range(4):
        C[mu] = eps[mu][..., None, None] * np.eye(2)
    return C


def cfg_haar(L, seed):
    q = np.random.default_rng(1000 + seed).standard_normal((4,) + (L,) * 4 + (4,))
    return from_q(q / np.linalg.norm(q, axis=-1, keepdims=True))


def cfg_prod(L):
    f = 'L%d_b2.3.npy' % L
    p = os.path.join(NOYAU2, 'data', f)
    assert sha256(p) == SHA_CFG[f], 'SHA cfg ' + f
    return np.ascontiguousarray(np.load(p)), {'source': p, 'sha256': SHA_CFG[f]}


def cfg_prod8(nom):
    p = os.path.join(NOYAU2, 'data', 'L8', nom)
    return np.ascontiguousarray(np.load(p)), {'source': p, 'sha256': sha256(p)}


PLAN = []
for L in (4, 8):
    PLAN += [(L, 'plat', lambda L=L: (cfg_flat(L), {})), (L, 'pi_flux', lambda L=L: (cfg_pi(L), {}))]
    PLAN += [(L, 'haar%d' % s, lambda L=L, s=s: (cfg_haar(L, s), {})) for s in range(3)]
PLAN += [(12, 'haar0', lambda: (cfg_haar(12, 0), {})),
         (12, 'prod_b2.3', lambda: cfg_prod(12)),
         (16, 'prod_b2.3', lambda: cfg_prod(16)),
         (16, 'plat', lambda: (cfg_flat(16), {}))]
BONUS = [(8, 'prod8_b2.3_v0', lambda: cfg_prod8('L8_b2.3_v0_cfg.npy')),
         (8, 'prod8_b16_v0', lambda: cfg_prod8('L8_b16_v0_cfg.npy'))]
PLAN += BONUS
if args.petit:
    # jeu commité : L ≤ 8, Haar limités à 3 (L = 4) et 2 (L = 8) ; pas de production
    gardes = {(4, n) for n in ('plat', 'pi_flux', 'haar0', 'haar1', 'haar2')} | \
             {(8, n) for n in ('plat', 'pi_flux', 'haar0', 'haar1')}
    PLAN = [p for p in PLAN if (p[0], p[1]) in gardes]
PLAN = [p for p in PLAN if p[0] <= args.Lmax]
if args.cas:
    voulus = set(args.cas.split(','))
    PLAN = [p for p in PLAN if '%d_%s' % (p[0], p[1]) in voulus or ('L%d_%s' % (p[0], p[1])) in voulus]


def sauve(d, nom, a):
    np.save(os.path.join(d, nom + '.npy'), np.ascontiguousarray(a))


os.makedirs(args.sortie, exist_ok=True)
index, wf_ecrit = [], set()
t_tot = time.time()
for (L, nom, fab) in PLAN:
    t0 = time.time()
    C0, src = fab()
    q = to_q(C0)
    C0q = from_q(q)                       # cb et Rust voient EXACTEMENT les mêmes matrices
    ecart_q = float(np.abs(C0q - C0).max())
    C = np.ascontiguousarray(C0q)
    geo = cb.Geometry(L)
    V = L ** 4
    cas = 'L%d_%s' % (L, nom)
    d = os.path.join(args.sortie, cas)
    os.makedirs(d, exist_ok=True)
    constructible = nom in ('plat', 'pi_flux')
    if not (args.petit and constructible):
        sauve(d, 'q', q)
    # poids de Fourier : creux (cb) et dense (lib_brancheC) ; identité des deux vérifiée
    Ws = cb.fourier_weights_sparse(L)
    Ws.sort_indices()
    wf_dense_ok = L <= 16
    Wf = B.fourier_weights(L, 1) if wf_dense_ok else None
    if wf_dense_ok:
        assert np.array_equal(Ws.toarray(), Wf), 'fourier_weights_sparse != fourier_weights'
    if L not in wf_ecrit:
        wf_ecrit.add(L)
        ptr, idx, val = Ws.indptr.astype(np.int64), Ws.indices.astype(np.int64), Ws.data.astype(np.float64)
        pas = 64 if (args.petit and L == 8) else 1
        if pas > 1:
            idx, val = idx[::pas], val[::pas]
        sauve(args.sortie, 'wf_L%d_ptr' % L, ptr)
        sauve(args.sortie, 'wf_L%d_idx' % L, idx)
        sauve(args.sortie, 'wf_L%d_val' % L, val)
        json.dump({'L': L, 'pas': pas, 'nnz': int(ptr[-1])}, open(os.path.join(args.sortie, 'wf_L%d.json' % L), 'w'))
    # cb : plaquettes, mesure (Wf dense), mesure (Wf creux), mesure libre
    pv, gp = cb.plaquettes(C, geo)
    F, D, blk = cb.mesure(C, geo, Wf if wf_dense_ok else Ws)
    Fs, Ds, _ = cb.mesure(C, geo, Ws)
    Fl, Dl = cb.mesure_libre(C, geo, Wf if wf_dense_ok else Ws)
    Fx = np.array([math.fsum(Ws[i].data * pv[Ws[i].indices]) for i in range(Ws.shape[0])])
    pas_gp = 8 if (args.petit and L == 8) else 1
    if args.petit and ((L == 8 and nom == 'haar1') or constructible):
        pas_gp = 0                         # pas de gradients (taille du jeu commité ; plat / pi_flux : tous nuls)
    if not (args.petit and constructible):
        sauve(d, 'cb_pv', pv)
    if pas_gp:
        sauve(d, 'cb_gp', gp.reshape(6 * V, 12)[::pas_gp])
    sauve(d, 'cb_F', F); sauve(d, 'cb_D', D); sauve(d, 'cb_blk', blk)
    sauve(d, 'cb_Fs', Fs); sauve(d, 'cb_Fx', Fx)
    sauve(d, 'cb_Flibre', Fl); sauve(d, 'cb_Dlibre', Dl)
    # références lib_Cb (concordance)
    ref_ok = False
    if not args.sans_ref and wf_dense_ok:
        M = B.masks_dec(L)
        T, g = B.fiber_map(C, M)
        pvr, Bg = R.plaq_values_grads_cached(C)
        WfT = np.ascontiguousarray(Wf.T)
        Fr = Wf @ pvr
        Gf = T.T @ (Bg @ WfT)
        Dr = Gf.T @ (Gf / g[:, None])
        br = R.block_sums(pvr, L).astype(np.float32)
        Gl = Bg @ WfT
        sauve(d, 'ref_F', Fr); sauve(d, 'ref_D', Dr); sauve(d, 'ref_blk', br); sauve(d, 'ref_Dlibre', Gl.T @ Gl)
        ref_ok = True
    meta = dict(cas=cas, L=L, nom=nom, plaquette=float(pv.mean()), q_construit_cote_rust=bool(args.petit and constructible),
                gp_pas=pas_gp, ecart_q_vs_cfg_origine=ecart_q, ref_libCb=ref_ok, nF=int(Ws.shape[0]),
                nnz_wf=int(Ws.nnz), Dnorm=float(np.linalg.norm(D)), t=round(time.time() - t0, 2),
                numpy=np.__version__, sha_cb=sha_cb, **src)
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    index.append(cas)
    print('%-18s L=%2d plaq=%+.4f |D|=%.3e (%.1f s)' % (cas, L, meta['plaquette'], meta['Dnorm'], meta['t']), flush=True)

json.dump({'cas': index, 'petit': args.petit, 'cb_sha256': sha_cb}, open(os.path.join(args.sortie, 'index.json'), 'w'), indent=1)
print('FIXTURES %d cas dans %s (%.0f s)' % (len(index), args.sortie, time.time() - t_tot))
