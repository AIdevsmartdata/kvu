#!/usr/bin/env python3
"""Compare deux flot.json (même ensemble, backends différents) : valeurs numériques des estimateurs ≤ 1e-9 relatif,
compteurs de domaine identiques, θ max ≤ 1e-12. Code 0 ssi conforme. (WP9-FLOT-GPU et WP9-2MACH)"""
import json, sys, math
a, b = (json.load(open(x)) for x in sys.argv[1:3])
TOL, TOLT = 1e-9, 1e-12
pire = [0.0, "-"]; ok = True; n = 0
def cmp_num(x, y, chemin):
    global ok, n
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None:
        if x != y: ok = False; print("DIFF", chemin, x, y)
        return
    n += 1
    if "theta_max" in chemin:
        if abs(x - y) > TOLT: ok = False; print("DIFF θ", chemin, x, y)
        return
    if "fraction_hors" in chemin or "n_ambigues" in chemin or "n_boucles" in chemin or chemin.endswith("n_groupes") or chemin.endswith("taille"):
        if x != y: ok = False; print("DIFF compteur", chemin, x, y)
        return
    r = abs(x - y) / max(abs(x), abs(y), 1e-300)
    if r > pire[0]: pire[0], pire[1] = r, chemin
    if r > TOL: ok = False; print("DIFF", chemin, x, y, r)
def marche(x, y, chemin):
    global ok
    if isinstance(x, dict):
        if set(x) != set(y): ok = False; print("CLES", chemin, set(x) ^ set(y)); return
        for k in x: marche(x[k], y[k], f"{chemin}.{k}")
    elif isinstance(x, list):
        if len(x) != len(y): ok = False; print("LONG", chemin); return
        for i, (u, v) in enumerate(zip(x, y)): marche(u, v, f"{chemin}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        cmp_num(x, y, chemin)
    elif x != y and not any(s in chemin for s in ("ensemble", "fichier", "version", "dossier")):
        ok = False; print("DIFF texte", chemin, x, y)
ca, cb = a["cellules"], b["cellules"]
if len(ca) != len(cb): print("nombre de cellules différent"); sys.exit(1)
for x, y in zip(ca, cb): marche(x, y, f"cellule[{x['beta']}]")
print(f"{'conforme' if ok else 'NON CONFORME'} : {n} nombres comparés, pire écart relatif {pire[0]:.3e} ({pire[1]}) ; compteurs de domaine identiques ; seuils {TOL:g} / θ {TOLT:g}")
sys.exit(0 if ok else 1)
