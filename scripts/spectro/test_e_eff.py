"""Test de e_eff (analyse_spectro.py) : periodes 16, 32, 48, 64 ; energie connue retrouvee a 1e-6 ;
periodes <= 32 : identique au bit a l'ancienne garde min(., 700). Lancer : python3 scripts/spectro/test_e_eff.py"""
import math, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_spectro as A

def e_eff_ancien(lam, T):
    ntau, n = lam.shape
    E = np.full((ntau - 1, n), np.nan)
    for t in range(ntau - 1):
        a = T / 2 - t
        for k in range(n):
            r = lam[t, k] / lam[t + 1, k] if lam[t + 1, k] != 0 else np.nan
            if not np.isfinite(r) or r <= 1 or lam[t, k] <= 0 or lam[t + 1, k] <= 0:
                continue
            lo, hi = 1e-9, 40.0
            f = lambda e: math.cosh(min(e * a, 700)) / math.cosh(min(e * (a - 1), 700))
            if f(hi) < r:
                continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < r: lo = mid
                else: hi = mid
            E[t, k] = 0.5 * (lo + hi)
    return E

def main():
    for T in (16, 32, 48, 64):
        for E0 in (0.05, 0.3, 0.9):
            tau = np.arange(0, T // 2)
            lam = np.cosh(E0 * (T / 2 - tau))[:, None]
            if not np.all(np.isfinite(lam)): continue
            E = A.e_eff(lam, T)[:, 0]
            assert np.all(np.isfinite(E)), (T, E0, E)
            assert np.max(np.abs(E - E0)) < 1e-6, (T, E0, np.max(np.abs(E - E0)))
    rng = random.Random(1)
    for T in (16, 24, 32):
        lam = np.array([[math.exp(-rng.uniform(0.01, 1.2) * t) * (1 + 0.01 * rng.random()) for _ in range(3)] for t in range(T // 2 + 1)])
        a, b = A.e_eff(lam, T), e_eff_ancien(lam, T)
        assert np.array_equal(a, b, equal_nan=True) and a.tobytes() == b.tobytes(), T
    print("OK : periodes 16/32/48/64 a 1e-6 ; <= 32 identique au bit")
main()
