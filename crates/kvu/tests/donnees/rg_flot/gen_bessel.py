import json, mpmath, sys
sys.dont_write_bytecode = True
mpmath.mp.dps = 60
xs = [mpmath.mpf(10) ** (mpmath.mpf(e) / 10) for e in range(-80, 31)]   # 1e-8 ... 1e3, 111 points
xs += [mpmath.mpf(v) for v in (0.5, 1, 2, 2.9, 5, 10, 25, 48, 96, 150, 400, 700)]
out = []
for x in sorted(set(xs)):
    r21 = mpmath.besseli(2, x) / mpmath.besseli(1, x)
    r10 = mpmath.besseli(1, x) / mpmath.besseli(0, x)
    out.append({"x": float(x), "r21": float(r21), "r10": float(r10)})
json.dump({"genere_par": "mpmath 60 chiffres, I2/I1 et I1/I0", "points": out}, open("bessel_ratios.json", "w"), indent=0)
print(len(out))
