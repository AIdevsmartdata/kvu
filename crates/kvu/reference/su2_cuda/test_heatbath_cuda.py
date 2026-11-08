#!/usr/bin/env python3
"""Test unitaire EXACT du tirage de x0 (bain thermique SU(2)) de su2_hip.

Densité cible : p(x) ∝ sqrt(1-x^2) exp(alpha x) sur [-1,1] (x = 1/2 Tr(U V)).
Moments exacts par quadrature de Gauss–Tchebychev de 2e espèce (poids sqrt(1-x^2), n=4000 nœuds),
comparés aux moments empiriques GPU (--test-hb). Seuil GELÉ avant exécution : |z| < 4 pour
m1 et m2 à chaque alpha (erreur standard = écart-type exact / sqrt(n)).
Les deux branches sont testées : Creutz (alpha < 2) et Kennedy–Pendleton (alpha >= 2).
"""
import subprocess, json, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'bin', 'su2_cuda')
N = 20_000_000
ALPHAS = [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0]


def exact(alpha, n=4000):
    k = np.arange(1, n + 1)
    x = np.cos(k * np.pi / (n + 1))
    # Gauss–Tchebychev 2e espèce : ∫ sqrt(1-x^2) f dx ≈ Σ π/(n+1) sin^2 θ_k f(x_k)
    w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1)) ** 2
    f = np.exp(alpha * (x - 1))
    Z = np.sum(w * f)
    m = [np.sum(w * f * x ** p) / Z for p in (1, 2, 3, 4)]
    return m


res = {'n': N, 'seuil_z': 4.0, 'alphas': {}}
ok_all = True
for i, a in enumerate(ALPHAS):
    out = subprocess.run(['taskset', '-c', '12-19', BIN, '--test-hb', repr(a), str(N), '--seed', str(9000 + i)], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith('TESTHB')]
    if out.returncode != 0 or not line:
        print('ECHEC execution', a, out.returncode, out.stderr[-500:]); sys.exit(1)
    kv = dict(t.split('=') for t in line[0].split()[1:])
    m1, m2 = float(kv['m1']), float(kv['m2'])
    e1, e2, e3, e4 = exact(a)
    sd1 = np.sqrt(e2 - e1 ** 2); sd2 = np.sqrt(e4 - e2 ** 2)
    z1 = (m1 - e1) / (sd1 / np.sqrt(N)); z2 = (m2 - e2) / (sd2 / np.sqrt(N))
    ok = abs(z1) < 4 and abs(z2) < 4 and int(kv['plafond']) == 0
    ok_all &= ok
    res['alphas'][repr(a)] = dict(branche='Creutz' if a < 2 else 'Kennedy-Pendleton', m1_gpu=m1, m1_exact=e1, z1=z1,
                                  m2_gpu=m2, m2_exact=e2, z2=z2, min=float(kv['min']), max=float(kv['max']),
                                  plafond=int(kv['plafond']), ok=bool(ok))
    print(f"alpha={a:8.3f} {res['alphas'][repr(a)]['branche']:18s} m1={m1:.8f} exact={e1:.8f} z1={z1:+.2f} "
          f"m2={m2:.8f} exact={e2:.8f} z2={z2:+.2f} min={float(kv['min']):.4f} ok={ok}", flush=True)
res['ok'] = bool(ok_all)
# contrôle de la quadrature : E[x0] = I2(alpha)/I1(alpha) ; vérifié par la série de Bessel à alpha=5
def besseli(nu, x, K=200):
    from math import factorial, lgamma, exp, log
    return sum(exp((2 * k + nu) * log(x / 2) - lgamma(k + 1) - lgamma(k + nu + 1)) for k in range(K))
res['controle_quadrature_alpha5'] = dict(quad=exact(5.0)[0], bessel=besseli(2, 5.0) / besseli(1, 5.0))
print('controle quadrature alpha=5 :', res['controle_quadrature_alpha5'])
json.dump(res, open(os.path.join(HERE, 'test_heatbath_cuda.json'), 'w'), indent=1)
print('TEST_HEATBATH', 'PASS' if ok_all else 'FAIL')
