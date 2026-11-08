#!/usr/bin/env python3
"""Supports exacts des liens grossiers (WP22), calcul INDÉPENDANT de kvu (parcours en largeur en Python pur).

Usage : python3 -I support_py.py [L] [--json FICHIER]

Distance d_P entre liens : graphe dont les sommets sont les 4 L⁴ liens du tore et où deux liens distincts sont voisins
ssi ils appartiennent à une même plaquette (18 voisins par lien pour L ≥ 3). Pour le lien grossier V(X = 0, μ) des
opérateurs à décimation, le chemin droit est {(kμ̂, μ) : k = 0, 1, 2}. Supports (docs/BLOCAGE_QUASI_LOCAL.md §5) :
  dec           : le chemin droit (3 liens) ;
  ape:α:n       : boule d_P de rayon n autour du chemin droit ;
  flot:t:ε      : boule d_P de rayon 3N autour du chemin droit, N = t/ε pas RK3 ;
  agr:c         : chemin droit + 6 agrafes de longueur 3 (33 liens).
Rend, pour chaque rayon R ≤ R_max, le nombre de liens de la boule (cumulé) et de la couche d_P = R.
"""
import json
import sys
from collections import deque


def voisins(lien, L):
    """Les 18 liens qui partagent une plaquette avec `lien` = (x, mu), x tuple de 4 entiers modulo L."""
    x, mu = lien
    out = []

    def dep(y, m, d):
        y = list(y)
        y[m] = (y[m] + d) % L
        return tuple(y)

    for nu in range(4):
        if nu == mu:
            continue
        # plaquette (x ; μ, ν) : (x, ν), (x+ν, μ), (x+μ, ν)
        out += [(x, nu), (dep(x, nu, 1), mu), (dep(x, mu, 1), nu)]
        # plaquette (x−ν ; μ, ν) : (x−ν, ν), (x−ν, μ), (x−ν+μ, ν)
        xm = dep(x, nu, -1)
        out += [(xm, nu), (xm, mu), (dep(xm, mu, 1), nu)]
    return out


def boule(depart, L, rmax):
    dist = {l: 0 for l in depart}
    file = deque(depart)
    while file:
        l = file.popleft()
        if dist[l] == rmax:
            continue
        for v in voisins(l, L):
            if v not in dist:
                dist[v] = dist[l] + 1
                file.append(v)
    return dist


def chemin_droit(mu, L):
    return [(tuple((k if m == mu else 0) % L for m in range(4)), mu) for k in range(3)]


def agr(mu, L):
    def dep(y, m, d):
        y = list(y)
        y[m] = (y[m] + d) % L
        return tuple(y)

    x = (0, 0, 0, 0)
    s = set(chemin_droit(mu, L))
    for nu in range(4):
        if nu == mu:
            continue
        for sg in (1, -1):
            base = x if sg == 1 else dep(x, nu, -1)
            s.add((base, nu))  # U(x, ±ν) : (x, ν) ou (x − ν, ν)
            s.add((dep(base, mu, 3), nu))  # U(x + 3μ, ±ν) : (x + 3μ, ν) ou (x − ν + 3μ, ν)
            xs = dep(x, nu, sg)
            for k in range(3):
                s.add((dep(xs, mu, k), mu))  # S(x ± ν)
    return s


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 12
    sortie = None
    if "--json" in sys.argv:
        sortie = sys.argv[sys.argv.index("--json") + 1]
    res = {"L": L, "lien": "V(X = 0, μ = 0)", "n_liens_tore": 4 * L**4, "boules": {}, "couches": {}}
    rmax = 3 * L  # au-delà, la boule a saturé le tore pour L ≤ 24
    d = boule(chemin_droit(0, L), L, rmax)
    for r in range(rmax + 1):
        res["boules"][r] = sum(1 for v in d.values() if v <= r)
        res["couches"][r] = sum(1 for v in d.values() if v == r)
    res["agr_33"] = len(agr(0, L))
    res["rayon_saturation"] = max(d.values())
    # symétrie : la taille de la boule ne dépend pas de μ
    for mu in (1, 2, 3):
        dm = boule(chemin_droit(mu, L), L, 13)
        assert sum(1 for v in dm.values() if v <= 13) == res["boules"][13], mu
    print(json.dumps({k: res[k] for k in ("L", "n_liens_tore", "agr_33", "rayon_saturation")}))
    for r in (0, 1, 2, 3, 6, 7, 12, 13, 14):
        print(f"R = {r:2d} : boule {res['boules'][r]:6d} liens, couche {res['couches'][r]:5d}")
    if sortie:
        with open(sortie, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
