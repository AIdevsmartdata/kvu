# SPDX-License-Identifier: Apache-2.0
"""Répartition de la campagne WP10 entre l'acemagic et le gamer : mesures (bench_wp10.sh) → durées prévues → makespan des scénarios.

Usage : python3 -I wp10_allocation.py --ace bench_ace.tsv --gam bench_gam.tsv [--off-ace S] [--off-gam S] [--cable-mo-s X] [--json FICHIER]
Les durées sont des PRÉVISIONS [ESTIMATION] calculées à partir des temps mesurés ; la simulation de la file dynamique (plus longues d'abord,
affinités du plan) reproduit la règle de `wp10_orch.py`. Rapporté, jamais une gate.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp10_plan  # noqa: E402

FENETRE_S = 540.0


def lire_tsv(chemin):
    t = {}
    for ligne in Path(chemin).read_text().splitlines():
        n, v, rc = ligne.split()
        t[n] = float(v)
    return t


def mesures(t):
    """Débits et coûts déduits des temps mesurés (mêmes formules que bench_wp10.sh)."""
    m = {}
    for L, n_pas, n_f, n_mes in ((54, 100, 12, 4), (36, 300, 60, 10)):
        m[f"g{L}"] = n_pas / (t[f"g{L}_b"] - t[f"g{L}_a"])          # pas/s F32
        m[f"f{L}"] = n_f / (t[f"f{L}_b"] - t[f"f{L}_a"])            # pas/s F64
        m[f"b{L}"] = (t[f"b{L}_avec"] - t[f"b{L}_sans"]) / n_mes    # s par bloc en ligne
        m[f"s{L}"] = t[f"g{L}_a"]                                   # mise en route (1 pas)
    m["fenetre_s"] = max(0.0, t["w54_reprise"] - 0.2)                # reprise coupée : relecture + 1 pas + point de reprise
    return m


def duree_tache(tache, m):
    """Durée prévue (s) d'une tâche sur une machine de mesures m (fenêtres de 540 s comprises)."""
    L = tache["L"]
    pas_par_s = m[f"g{L}"] if tache["precision"] == "f32" else m[f"f{L}"]
    bloc = m[f"b{L}"]
    calc = tache["therm"] / pas_par_s + tache["nmeas"] * (tache["skip"] / pas_par_s + bloc)
    utile = max(1.0, FENETRE_S - m["fenetre_s"])
    n_fen = max(1, math.ceil(calc / utile))
    return calc + n_fen * m["fenetre_s"], n_fen


def simuler(taches, m_ace, m_gam, machines=("ace", "gam")):
    """File dynamique : à chaque libération, la machine prend la première tâche du plan compatible avec son affinité."""
    restantes = list(taches)
    libre = {mm: 0.0 for mm in machines}
    occupe = {mm: 0.0 for mm in machines}
    plan = {mm: [] for mm in machines}
    mesures_m = {"ace": m_ace, "gam": m_gam}
    nom_aff = {"ace": "acemagic", "gam": "gamer"}
    while restantes:
        mm = min(libre, key=libre.get)
        choix = next((t for t in restantes if t["affinite"] in ("any", nom_aff[mm])), None)
        if choix is None:
            libre[mm] = float("inf")
            if all(v == float("inf") for v in libre.values()):
                break
            continue
        d, _ = duree_tache(choix, mesures_m[mm])
        libre[mm] += d
        occupe[mm] += d
        plan[mm].append((choix["id"], round(d / 60, 1)))
        restantes.remove(choix)
    return {"makespan_min": max([v for v in libre.values() if v != float("inf")] or [0.0]) / 60, "occupe_min": {k: v / 60 for k, v in occupe.items()}, "plan": plan}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace", required=True)
    p.add_argument("--gam", required=True)
    p.add_argument("--off-ace", type=float, default=float("nan"), help="s : rg bloc hors ligne sur l'acemagic (dump L = 54 q64, page cache chaud)")
    p.add_argument("--off-gam", type=float, default=float("nan"))
    p.add_argument("--cable-mo-s", type=float, default=270.0)
    p.add_argument("--json")
    a = p.parse_args(argv)
    ma, mg = mesures(lire_tsv(a.ace)), mesures(lire_tsv(a.gam))
    taches = wp10_plan.plan()
    sortie = {"mesures": {"acemagic": ma, "gamer": mg}}
    out = ["## Mesures (temps mur ; débits par différence de deux longueurs)", "",
           "| quantité | acemagic (HIP) | gamer (CUDA) | gamer / acemagic |", "|---|---|---|---|"]
    noms = {"g54": "pas/s F32 L = 54", "f54": "pas/s F64 L = 54", "b54": "s par bloc en ligne L = 54 (F32 promu)",
            "g36": "pas/s F32 L = 36", "f36": "pas/s F64 L = 36", "b36": "s par bloc en ligne L = 36", "fenetre_s": "s par fenêtre (relecture + 1 pas + point de reprise de 1,09 Go)"}
    for k, nom in noms.items():
        out.append(f"| {nom} | {ma[k]:.3f} | {mg[k]:.3f} | {mg[k] / ma[k]:.2f} |")
    out += ["", "## Durées prévues d'une tâche (min, fenêtres de 540 s comprises) [ESTIMATION]", "", "| tâche | acemagic | gamer | gamer / acemagic |", "|---|---|---|---|"]
    types = {}
    for t in taches:
        types.setdefault((t["genre"], t["L"], t["precision"]), t)
    duree = {}
    for (genre, L, prec), t in types.items():
        da, na = duree_tache(t, ma)
        dg, ng = duree_tache(t, mg)
        duree[(genre, L)] = (da, dg)
        out.append(f"| {genre} L = {L} {prec} (ex. {t['id']}) | {da / 60:.1f} ({na} fen.) | {dg / 60:.1f} ({ng} fen.) | {dg / da:.2f} |")
    sortie["durees_min"] = {f"{g}{L}": [v[0] / 60, v[1] / 60] for (g, L), v in duree.items()}
    # scénarios
    sa = simuler([{**t, "affinite": "any"} for t in taches], ma, mg, machines=("ace",))
    sg = simuler([{**t, "affinite": "any"} for t in taches], ma, mg, machines=("gam",))
    sd = simuler(taches, ma, mg)
    sc = {k: v for k, v in sd.items()}
    out += ["", "## Scénarios (makespan = fin de la dernière tâche, GPU libres, hors attente du verrou) [ESTIMATION]", "", "| scénario | makespan (h) | occupation acemagic (h) | occupation gamer (h) |", "|---|---|---|---|"]
    out.append(f"| acemagic seule (affinités ignorées) | {sa['makespan_min'] / 60:.2f} | {sa['occupe_min']['ace'] / 60:.2f} | 0 |")
    out.append(f"| gamer seul (affinités ignorées) | {sg['makespan_min'] / 60:.2f} | 0 | {sg['occupe_min']['gam'] / 60:.2f} |")
    out.append(f"| **deux files indépendantes, tirage dynamique (retenu)** | **{sd['makespan_min'] / 60:.2f}** | {sd['occupe_min']['ace'] / 60:.2f} | {sd['occupe_min']['gam'] / 60:.2f} |")
    # dumps : le gamer génère, l'acemagic bloque hors ligne
    n_mes = 200
    b_off = a.off_ace if not math.isnan(a.off_ace) else float("nan")
    transfert = 1088391296 / 1e6 / a.cable_mo_s
    par_mesure_dump = max(b_off, transfert, 10 / mg["g54"] + 4.0)    # acemagic (bloc hors ligne) ; câble ; gamer (10 pas + écriture du dump q64, 4 s supposées)
    par_chaine_dump = (2000 / mg["g54"]) / 1.0 + n_mes * par_mesure_dump
    out.append(f"| dumps : le gamer génère, l'acemagic bloque hors ligne (pipeline parfait, par chaîne L = 54) | {par_chaine_dump / 3600:.2f} par chaîne (une chaîne à la fois) | {n_mes * b_off / 3600:.2f} par chaîne | — |")
    sortie["dumps"] = {"s_par_mesure": par_mesure_dump, "min_par_chaine": par_chaine_dump / 60, "transfert_s": transfert, "bloc_hors_ligne_ace_s": b_off, "bloc_hors_ligne_gam_s": a.off_gam}
    # F64 partout (option b du pilote) : pas de contrôle, bloc F64 résident sans promotion
    f64 = []
    for t in taches:
        if t["genre"] in ("C",):
            continue
        f64.append({**t, "precision": "f64", "options": "compat-prod"})
    sf = simuler(f64, ma, mg)
    out.append(f"| génération F64 de toutes les chaînes (option b : 23 chaînes, sans les 4 contrôles F64) | {sf['makespan_min'] / 60:.2f} | {sf['occupe_min']['ace'] / 60:.2f} | {sf['occupe_min']['gam'] / 60:.2f} |")
    sortie["scenarios"] = {"ace_seule": sa["makespan_min"], "gamer_seul": sg["makespan_min"], "dynamique": sd, "f64_partout": sf}
    out += ["", "### Tirage dynamique retenu : prévision par machine", ""]
    for mm, nom in (("ace", "acemagic"), ("gam", "gamer")):
        out.append(f"- {nom} ({sd['occupe_min'][mm] / 60:.2f} h) : " + ", ".join(f"{i} ({d} min)" for i, d in sd["plan"][mm]))
    texte = "\n".join(out) + "\n"
    print(texte)
    if a.json:
        Path(a.json).write_text(json.dumps(sortie, indent=1, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
