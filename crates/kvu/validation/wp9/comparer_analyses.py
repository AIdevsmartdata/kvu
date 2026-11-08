#!/usr/bin/env python3
"""Compare deux analyse.json (gamer contre acemagic) : tout nombre à ≤ 1e-12 en relatif (p-values MC : égalité exacte attendue,
écart toléré ≤ 2/NMC signalé), textes identiques hors chemins et versions. Code 0 ssi conforme."""
import json, sys
a, b = (json.load(open(x)) for x in sys.argv[1:3])
TOL = 1e-12
pire = [0.0, "-"]; ok = True; n = 0; mc = []
def marche(x, y, ch):
    global ok, n
    if ch.endswith(".versions") or ch.endswith(".courbes"):
        return
    if isinstance(x, dict):
        if set(x) != set(y): ok = False; print("CLES", ch, set(x) ^ set(y)); return
        for k in x: marche(x[k], y[k], f"{ch}.{k}")
    elif isinstance(x, list):
        if len(x) != len(y): ok = False; print("LONG", ch); return
        for i, (u, v) in enumerate(zip(x, y)): marche(u, v, f"{ch}[{i}]")
    elif isinstance(x, bool) or x is None or isinstance(x, str):
        if x != y and not ch.endswith(".flot"):
            if isinstance(x, str) and ("p₀(MC)" in x or "p_R(MC)" in x or "p_δ(MC)" in x):
                mc.append(ch); return
            ok = False; print("DIFF", ch, str(x)[:120], "|", str(y)[:120])
    else:
        n += 1
        r = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if "_mc" in ch:
            if r > 0: mc.append((ch, x, y))
            return
        if r > pire[0]: pire[0], pire[1] = r, ch
        if r > TOL: ok = False; print("DIFF", ch, x, y, r)
marche(a, b, "analyse")
if mc:
    ok = False
    print("p-values MC différentes :", mc[:5])
print(f"{'conforme' if ok else 'NON CONFORME'} : {n} nombres, pire écart relatif {pire[0]:.3e} ({pire[1]}) ; p-values MC différentes : {len(mc)}")
sys.exit(0 if ok else 1)
