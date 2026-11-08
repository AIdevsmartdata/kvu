# SPDX-License-Identifier: Apache-2.0
"""Plan de la campagne WP10 (PREREG v1) : liste fermée des tâches, graines et protocoles.

Source unique pour l'orchestrateur, les scripts de post-traitement et l'analyse. Aucune entrée n'est tirée au hasard :
les graines sont écrites ici, figées par le hachage du PREREG avant tout run de production.
Usage : python3 -I wp10_plan.py [--json FICHIER]   (écrit le plan, ou l'affiche).
"""

import json
import sys

BETAS = (4, 8, 16)
FACTEUR_MUTANT = 1.002          # chaîne F32 simulée à 1,002 β, analysée au β nominal (WP10-LOI2-MUT)
FACTEUR_FAIBLE = 1.0002         # mutant faible (INFO) : plancher de détection de LOI-2, relecture de Sol (R2)
L_GRAND = 54                    # 54 -> 18 -> 6
L_PETIT = 36                    # 36 -> 12 -> 4 (27 est impair : refusé par le générateur, voir PREREG §0)
GRAINE_BASE = 26102610000
PROTO = {"therm": 2000, "nmeas": 200, "skip": 10, "nor": 4, "start": "hot"}


def beta_txt(b):
    return f"b{int(b):02d}"


def cellule(l, beta):
    return f"L{l}_{beta_txt(beta)}"


def _tache(ident, genre, l, beta, graine, precision, proto, affinite, garder, beta_run=None):
    return {
        "id": ident,
        "genre": genre,                      # P production F32, R recertification F32 (acemagic), C controle F64, M mutant F32 (1,002 β), N mutant faible F32 (1,0002 β, INFO)
        "cellule": cellule(l, beta),
        "L": l,
        "beta": float(beta),
        "beta_run": float(beta if beta_run is None else beta_run),
        "graine": graine,
        "precision": precision,
        "options": "rapide" if precision == "f32" else "compat-prod",
        "affinite": affinite,                # "any" | "acemagic" | "gamer"
        "garder": garder,                    # "tout" | {"un_sur": N} (elagage apres G-H13 et analyse)
        "mesures": "bloc:3:1",
        **proto,
    }


def plan():
    taches = []
    # --- recertification (acemagic, classe A) puis production du grand réseau : les plus longues d'abord (LPT)
    for c in (1, 2, 3):
        taches.append(_tache(f"R54-b16-c{c}", "R", L_GRAND, 16, GRAINE_BASE + 700 + c, "f32", PROTO, "acemagic", {"un_sur": 100}))
    for k, b in ((3, 16), (1, 4), (2, 8)):
        for c in (1, 2, 3):
            aff = "gamer" if b == 16 else "any"      # la cellule β = 16 du grand réseau est produite côté gamer (classe B) : c'est elle que R54 recertifie
            taches.append(_tache(f"P54-{beta_txt(b)}-c{c}", "P", L_GRAND, b, GRAINE_BASE + 100 * k + c, "f32", PROTO, aff, "tout"))
    # --- contrôle F64 du grand réseau
    taches.append(_tache("C54-b16", "C", L_GRAND, 16, GRAINE_BASE + 1101, "f64", PROTO, "any", {"un_sur": 100}))   # protocole complet (relecture de Sol, point 5)
    # --- petit réseau : production, contrôles F64, mutant
    for k, b in ((4, 4), (5, 8), (6, 16)):
        for c in (1, 2, 3):
            taches.append(_tache(f"P36-{beta_txt(b)}-c{c}", "P", L_PETIT, b, GRAINE_BASE + 100 * k + c, "f32", PROTO, "any", "tout"))
    for k, b in ((8, 4), (9, 8), (10, 16)):
        taches.append(_tache(f"C36-{beta_txt(b)}", "C", L_PETIT, b, GRAINE_BASE + 100 * k + 1, "f64", PROTO, "any", {"un_sur": 100}))
    taches.append(_tache("M36-b08", "M", L_PETIT, 8, GRAINE_BASE + 1201, "f32", PROTO, "any", {"un_sur": 100}, beta_run=round(8 * FACTEUR_MUTANT, 6)))
    taches.append(_tache("N36-b08", "N", L_PETIT, 8, GRAINE_BASE + 1301, "f32", PROTO, "any", {"un_sur": 100}, beta_run=round(8 * FACTEUR_FAIBLE, 6)))
    ids = [t["id"] for t in taches]
    graines = [t["graine"] for t in taches]
    assert len(set(ids)) == len(ids) and len(set(graines)) == len(graines)
    return taches


def plan_test():
    """Plan d'ESSAI de l'orchestrateur (L = 18 → 6 → 2, quelques secondes par tâche), graines 26102629xxx : jamais utilisé pour la campagne."""
    proto = {"therm": 30, "nmeas": 12, "skip": 3, "nor": 4, "start": "hot"}
    base = 26102629000
    t = []
    t.append(_tache("R18-b04-c1", "R", 18, 4, base + 1, "f32", proto, "acemagic", {"un_sur": 4}))
    t.append(_tache("P18-b04-c1", "P", 18, 4, base + 2, "f32", proto, "gamer", "tout"))
    t.append(_tache("P18-b04-c2", "P", 18, 4, base + 3, "f32", proto, "any", "tout"))
    t.append(_tache("P18-b04-c3", "P", 18, 4, base + 4, "f32", proto, "any", "tout"))
    t.append(_tache("C18-b04", "C", 18, 4, base + 5, "f64", proto, "any", {"un_sur": 4}))
    t.append(_tache("M18-b04", "M", 18, 4, base + 6, "f32", proto, "any", {"un_sur": 4}, beta_run=round(4 * FACTEUR_MUTANT, 6)))
    for i, x in enumerate(t):
        x["verrou"] = "partage" if i % 2 else "exclusif"      # les deux modes de verrou sont exercés
    return t


def main(argv):
    if len(argv) >= 2 and argv[1] == "--test":
        texte = json.dumps({"schema": "wp10-plan/1", "taches": plan_test()}, indent=1, ensure_ascii=False) + "\n"
        sys.stdout.write(texte)
        return 0
    p = plan()
    texte = json.dumps({"schema": "wp10-plan/1", "taches": p}, indent=1, ensure_ascii=False) + "\n"
    if len(argv) >= 3 and argv[1] == "--json":
        with open(argv[2], "w", encoding="utf-8") as f:
            f.write(texte)
    else:
        sys.stdout.write(texte)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
