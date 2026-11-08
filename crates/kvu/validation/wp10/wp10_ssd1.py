# SPDX-License-Identifier: Apache-2.0
"""SSD-1 : projection disque de la campagne WP10, acceptée AVANT le départ (PREREG v1 §5). Lecture seule, aucune écriture hors de la sortie standard.

Tailles exactes : un fichier q64 de côté L pèse 128·L⁴ + 128 octets (en-tête NPY de 128 octets, vérifié sur un dump de L = 54 : 1 088 391 296 octets).
Sorties conservées = configurations grossières de chaque tâche (toutes pour la production, un indice sur N pour les tâches de contrôle après élagage),
plus une provision de 1,5 Mo de CSV, manifestes et JSON par tâche. Pic transitoire = points de reprise (2 × 1,09 Go par file) et configurations des tâches
de contrôle avant élagage. Verdicts : PASS|FAIL SSD-1-… ; code de sortie = nombre de FAIL.
Usage : python3 -I wp10_ssd1.py --plan plan.json --racine E [--gam-racine D] [--ssh gamer] [--plafond-go 30] [--garde-ace-go 100] [--garde-gam-go 50]
"""

import argparse
import json
import os
import subprocess
import sys

ENTETE = 128
METADONNEES_PAR_TACHE = 1_500_000


def taille_q64(l):
    return 128 * l ** 4 + ENTETE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--racine", required=True)
    ap.add_argument("--gam-racine", default="")
    ap.add_argument("--ssh", default="gamer")
    ap.add_argument("--plafond-go", type=float, default=30.0)
    ap.add_argument("--garde-ace-go", type=float, default=100.0)
    ap.add_argument("--garde-gam-go", type=float, default=50.0)
    a = ap.parse_args(argv)
    taches = json.load(open(a.plan, encoding="utf-8"))["taches"]
    gardes = transitoire = meta = 0
    lignes = []
    for t in taches:
        l1, l2 = t["L"] // 3, t["L"] // 9
        n = t["nmeas"]
        if t["garder"] == "tout":
            k = n
        else:
            k = sum(1 for m in range(1, n + 1) if m % t["garder"]["un_sur"] == 0)
        octets = k * (taille_q64(l1) + taille_q64(l2))
        gardes += octets
        transitoire += (n - k) * (taille_q64(l1) + taille_q64(l2)) if t["garder"] != "tout" else 0
        meta += METADONNEES_PAR_TACHE
        lignes.append((t["id"], k, n, octets))
    total = gardes + meta
    nfail = 0

    def verdict(ok, ident, detail):
        nonlocal nfail
        print(f"{'PASS' if ok else 'FAIL'} {ident} {detail}")
        nfail += 0 if ok else 1

    for ident, k, n, o in lignes:
        print(f"INFO SSD-1-DETAIL {ident} {k}/{n} configurations conservées, {o / 1e6:.1f} Mo")
    verdict(total / 1e9 <= a.plafond_go, "SSD-1-PLAFOND",
            f"sorties conservées projetées {total / 1e9:.4f} Go (configurations {gardes / 1e9:.4f} Go + métadonnées {meta / 1e9:.3f} Go) pour un plafond de {a.plafond_go:.0f} Go (D6) ; marge {(a.plafond_go * 1e9 - total) / 1e9:.3f} Go")
    # pic transitoire : un point de reprise de L = 54 en vol par file (deux copies + lien de sauvegarde), plus les configurations des tâches de contrôle avant élagage
    pic = 3 * taille_q64(54) * 2 + transitoire
    print(f"INFO SSD-1-PIC pic transitoire projeté {pic / 1e9:.2f} Go (points de reprise : 2 files × 3 × 1,09 Go ; configurations de contrôle avant élagage : {transitoire / 1e9:.2f} Go)")
    base = a.racine
    while not os.path.exists(base) and base != "/":
        base = os.path.dirname(base.rstrip("/")) or "/"
    st = os.statvfs(base)
    libre = st.f_bavail * st.f_frsize
    verdict(libre - total - pic >= a.garde_ace_go * 1e9, "SSD-1-ACEMAGIC",
            f"libre {libre / 1e9:.0f} Go ; après sorties conservées et pic transitoire : {(libre - total - pic) / 1e9:.0f} Go (≥ {a.garde_ace_go:.0f} Go exigés par la garde de l'outil ; ≥ 30 Go par D6)")
    if a.gam_racine:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", a.ssh, f"df --output=avail -B1 {a.gam_racine} 2>/dev/null || df --output=avail -B1 $(dirname {a.gam_racine})"],
                           capture_output=True, text=True)
        try:
            libre_g = int(r.stdout.strip().splitlines()[-1])
            # le gamer garde un original de chaque chaîne qu'il a produite (au pire : tout)
            verdict(libre_g - total - pic >= a.garde_gam_go * 1e9, "SSD-1-GAMER",
                    f"libre {libre_g / 1e9:.0f} Go ; au pire (tout produit côté gamer) : {(libre_g - total - pic) / 1e9:.0f} Go (≥ {a.garde_gam_go:.0f} Go exigés par la garde de l'outil)")
        except (ValueError, IndexError):
            verdict(False, "SSD-1-GAMER", f"libre du gamer illisible : {r.stdout!r} {r.stderr!r}")
    return nfail


if __name__ == "__main__":
    sys.exit(main())
