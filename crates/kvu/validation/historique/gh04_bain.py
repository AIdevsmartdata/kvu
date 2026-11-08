#!/usr/bin/env python3
"""G-H04 : bain thermique exact (ancienne gate P1 de gates_gpu1), moteur unifié, F64, HIP ou CUDA.

Reprise de reference/su2_cuda/test_heatbath_cuda.py, SANS numpy (le gamer n'en a pas) :
densité cible p(x) ∝ sqrt(1-x²) exp(α x) sur [-1,1] ; moments exacts m1..m4 par quadrature de
Gauss-Tchebychev de 2ᵉ espèce (n = 4000) ; empiriques lus sur la ligne TESTHB de
`kevinotron su2 test-hb ALPHA N --graine S`. Seuil GELÉ : |z| < 4 pour m1 et m2 à chaque α,
plafond (tirages ayant atteint 10⁶ essais) = 0. N = 2·10⁷, α ∈ {0,3 ; 1 ; 1,999 ; 2 ; 5 ; 20 ; 100},
graines 9000 + i. Les deux branches sont couvertes : Creutz (α < 2) et Kennedy-Pendleton (α ≥ 2).

Usage : gh04_bain.py BINAIRE [--n N] [--precision f64]
Sortie : PASS|FAIL G-H04 <détail>. Code de sortie 0 (le statut est dans la ligne).
"""
import math
import subprocess
import sys

ALPHAS = [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0]
SEUIL_Z = 4.0


def exact(alpha, n=4000):
    """(E[x], E[x²], E[x³], E[x⁴]) par Gauss-Tchebychev de 2ᵉ espèce."""
    xs, ws = [], []
    for k in range(1, n + 1):
        th = k * math.pi / (n + 1)
        xs.append(math.cos(th))
        ws.append(math.pi / (n + 1) * math.sin(th) ** 2)
    f = [math.exp(alpha * (x - 1)) for x in xs]
    z = sum(w * g for w, g in zip(ws, f))
    return [sum(w * g * x ** p for w, g, x in zip(ws, f, xs)) / z for p in (1, 2, 3, 4)]


def ligne_testhb(sortie):
    for l in sortie.splitlines():
        if l.startswith('TESTHB'):
            return dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
    return None


def main():
    if len(sys.argv) < 2:
        print('FAIL G-H04 usage : gh04_bain.py BINAIRE [--n N]')
        return
    binaire = sys.argv[1]
    n = 20_000_000
    precision = 'f64'
    if '--n' in sys.argv:
        n = int(sys.argv[sys.argv.index('--n') + 1])
    if '--precision' in sys.argv:
        precision = sys.argv[sys.argv.index('--precision') + 1]
    pires, backend, mauvais = 0.0, '?', []
    try:   # backend réellement choisi, lu dans doctor --json (la ligne TESTHB ne le porte pas)
        d = subprocess.run([binaire, 'doctor', '--json'], capture_output=True, text=True, timeout=60).stdout
        import re
        m = re.search(r'"backend"\s*:\s*"(\w+)"', d)
        if m:
            backend = m.group(1)
    except Exception:
        pass
    for i, a in enumerate(ALPHAS):
        p = subprocess.run([binaire, 'su2', 'test-hb', repr(a), str(n), '--graine', str(9000 + i), '--precision', precision],
                           capture_output=True, text=True)
        kv = ligne_testhb(p.stdout)
        if p.returncode != 0 or kv is None:
            print(f'FAIL G-H04 exécution à alpha={a} : rc={p.returncode} {p.stderr.strip()[-200:]}')
            return
        backend = kv.get('backend', backend)
        m1, m2 = float(kv['m1']), float(kv['m2'])
        e1, e2, _, e4 = exact(a)
        sd1, sd2 = math.sqrt(e2 - e1 ** 2), math.sqrt(e4 - e2 ** 2)
        z1, z2 = (m1 - e1) / (sd1 / math.sqrt(n)), (m2 - e2) / (sd2 / math.sqrt(n))
        pires = max(pires, abs(z1), abs(z2))
        plafond = int(kv.get('plafond', '-1'))
        print(f"  alpha={a:8.3f} {'Creutz' if a < 2 else 'KP':6s} z1={z1:+.2f} z2={z2:+.2f} plafond={plafond}", file=sys.stderr)
        if abs(z1) >= SEUIL_Z or abs(z2) >= SEUIL_Z or plafond != 0:
            mauvais.append(f'alpha={a} z1={z1:+.2f} z2={z2:+.2f} plafond={plafond}')
    if mauvais:
        print('FAIL G-H04 ' + '; '.join(mauvais))
    else:
        print(f'PASS G-H04 bain thermique exact, 7 alpha, N={n:.0e}, max|z|={pires:.2f} < 4, plafond 0 (backend {backend})')


if __name__ == '__main__':
    main()
