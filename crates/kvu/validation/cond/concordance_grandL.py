#!/usr/bin/env python3
"""Concordance à plus grand L : 1 balayage `ref_rng` de `kvu cond sweep` contre `cb.sweep(mode='ref_rng')` (L = 12, 16 :
configurations de production de noyau2/data, β ∈ {2,3 ; 16}, graines {0, 1}), max|ΔC| ≤ 1e-12 et état PCG64 final identique.
Complète EQC-1 (L = 4 et 8). Usage : concordance_grandL.py --bin <cond_cli-*|kevinotron> [--backend auto|cpu|hip|cuda] [--L 12 16]
RC 0 = tout VERT."""
import os, sys, json, argparse, subprocess, tempfile, shutil
import numpy as np

NOYAU = '/chemin/vers/projet/outils/ym4_noyaux/noyau2'
sys.path.insert(0, NOYAU); sys.dont_write_bytecode = True
import cb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--bin', required=True)
ap.add_argument('--backend', default='auto')
ap.add_argument('--L', type=int, nargs='+', default=[12, 16])
args = ap.parse_args()
TMP = tempfile.mkdtemp(prefix='conc_', dir=os.environ.get('TMPDIR', '/tmp'))


def cond(*a):
    if os.path.basename(args.bin).startswith('cond_cli-'):
        p = subprocess.run([args.bin, '--ignored', '--exact', 'pont_cli', '--nocapture', '--test-threads=1'],
                           env=dict(os.environ, KVU_COND_ARGV='\x1f'.join(a)), capture_output=True, text=True)
    else:
        p = subprocess.run([args.bin, 'cond', *a], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1500:])
    return [json.loads(l[l.index('{'):]) for l in p.stdout.splitlines() if '{' in l][-1]


def from_q(q):
    C = np.empty(q.shape[:-1] + (2, 2), complex)
    C[..., 0, 0] = q[..., 0] + 1j * q[..., 3]; C[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    C[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]; C[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    return C


ok = True; n = 0; dmax = 0.0
for L in args.L:
    C0 = np.ascontiguousarray(np.load(os.path.join(NOYAU, 'data', f'L{L}_b2.3.npy')))
    geo = cb.Geometry(L); Vr = cb.V_pairs(C0, geo)
    for beta in (2.3, 16.0):
        for seed in (0, 1):
            rng = np.random.default_rng(seed); C = C0.copy()
            cb.sweep(C, beta, rng, geo, Vr, mode='ref_rng')
            etat = rng.bit_generator.state['state']
            out = os.path.join(TMP, 'o.npy')
            rep = cond('sweep', '--cfg', os.path.join(NOYAU, 'data', f'L{L}_b2.3.npy'), '--beta', repr(beta), '--graine', str(seed),
                       '--mode', 'ref_rng', '--n', '1', '--out', out, '--format', 'q64', '--backend', args.backend)
            q = np.load(out); Cm = from_q(q)
            d = float(np.abs(C - Cm).max()); dmax = max(dmax, d); n += 1
            st = rep['alea']['Numpy']
            sok = (int(st['etat']) == etat['state'] and int(st['inc']) == etat['inc'])
            good = d <= 1e-12 and sok and rep['vdev'] <= 1e-12 and rep['unitarite'] <= 1e-12
            ok &= good
            print(('VERT ' if good else 'ROUGE'), f'L={L} beta={beta} graine={seed} : max|dC| = {d:.2e}, etat PCG64 {"identique" if sok else "DIFFERENT"}, Vdev {rep["vdev"]:.1e}')
shutil.rmtree(TMP, ignore_errors=True)
print(('PASS' if ok else 'FAIL'), f'concordance grand L ({args.backend}) : {n} cas, max|dC| = {dmax:.2e}')
sys.exit(0 if ok else 1)
