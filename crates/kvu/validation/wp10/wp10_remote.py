# SPDX-License-Identifier: Apache-2.0
"""Aides d'exécution de la campagne WP10, bibliothèque standard seule : lancées localement (acemagic) ou par ssh (gamer).

  info DIR                  JSON : le répertoire de la chaîne existe, COMPLETE, point de reprise (pas faits, mesures faites), nombre de configurations grossières, octets
  reparer DIR               remet les sorties en ligne (bloc3.csv, bloc3_hist.csv, bloc3_cfg_*.npy) à l'état du point de reprise courant (après un arrêt brutal)
  sauvegarder DIR K         copie le point de reprise courant en `reprise.champs.wK` (lien dur) et `reprise.champs.json.wK`
  restaurer DIR K           remet la sauvegarde K comme point de reprise courant (puis `reparer`)
  nettoyer DIR              retire les fichiers de reprise (`reprise.*`) d'une chaîne COMPLETE
  noyer DIR                 supprime les sauvegardes `reprise.champs.w*`
  libre CHEMIN              octets libres du système de fichiers de CHEMIN (ou de son premier ancêtre existant)
  memoire                   MemAvailable en octets
Les fichiers de reprise portent le préfixe `reprise.champs.` ou le nom `reprise.json` : `rg::en_ligne::est_transitoire` les écarte du manifeste kvu-regen/1.
"""

import json
import os
import re
import shutil
import sys

CFG = re.compile(r"bloc3_cfg_(\d+)\.npy\Z")


def lire_json(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def info(rep):
    res = {"existe": os.path.isdir(rep), "complete": os.path.isfile(os.path.join(rep, "COMPLETE")), "reprise": None,
           "n_cfg": 0, "octets": 0}
    if not res["existe"]:
        return res
    rj = os.path.join(rep, "reprise.json")
    if os.path.isfile(rj):
        try:
            e = lire_json(rj)
            res["reprise"] = {"pas_faits": e["pas_faits"], "mesures_faites": e["mesures_faites"]}
        except (OSError, ValueError, KeyError):
            res["reprise"] = {"illisible": True}
    for nom in os.listdir(rep):
        p = os.path.join(rep, nom)
        if os.path.isfile(p):
            res["octets"] += os.path.getsize(p)
            if CFG.fullmatch(nom):
                res["n_cfg"] += 1
    return res


def tronquer(chemin, n_lignes):
    """Garde l'en-tête et n_lignes lignes de données."""
    if not os.path.isfile(chemin):
        return 0
    with open(chemin, "rb") as f:
        lignes = f.read().split(b"\n")
    # lignes[-1] est vide si le fichier se termine par un saut de ligne : lignes de données = len − 2
    donnees = len(lignes) - 2
    if n_lignes >= donnees:
        return 0
    avec_fin = b"\n".join(lignes[:1 + n_lignes]) + b"\n"
    part = chemin + ".part"
    with open(part, "wb") as f:
        f.write(avec_fin)
    os.replace(part, chemin)
    return donnees - n_lignes


def reparer(rep):
    """Sorties en ligne ramenées au point de reprise : mesures m = 1…m_faites conservées, le reste retiré (il sera refait, de façon déterministe)."""
    cfg = lire_json(os.path.join(rep, "config_run.json"))
    e = lire_json(os.path.join(rep, "reprise.json"))
    therm, skip = cfg["therm"], cfg["skip"]
    pas_faits = e["pas_faits"]
    m_faites = max(0, e["mesures_faites"] - 1)
    retires = 0
    for nom in sorted(os.listdir(rep)):
        m = CFG.fullmatch(nom)
        if m and int(m.group(1)) > pas_faits:
            os.remove(os.path.join(rep, nom))
            retires += 1
    surplus = [tronquer(os.path.join(rep, f), m_faites) for f in ("bloc3.csv", "bloc3_hist.csv")]
    # fichiers écrits par `Ecrivain` avec .part ? rien à faire : écritures atomiques
    return {"pas_faits": pas_faits, "mesures_faites": m_faites, "cfg_retirees": retires, "lignes_retirees": surplus}


def sauvegarder(rep, k):
    src, dst = os.path.join(rep, "reprise.champs"), os.path.join(rep, f"reprise.champs.w{k}")
    if not os.path.isfile(src):
        return {"sauvegarde": False}
    for p in (dst, os.path.join(rep, f"reprise.champs.json.w{k}")):
        if os.path.exists(p):
            os.remove(p)
    os.link(src, dst)
    shutil.copy2(os.path.join(rep, "reprise.json"), os.path.join(rep, f"reprise.champs.json.w{k}"))
    return {"sauvegarde": True}


def restaurer(rep, k):
    src, js = os.path.join(rep, f"reprise.champs.w{k}"), os.path.join(rep, f"reprise.champs.json.w{k}")
    if not (os.path.isfile(src) and os.path.isfile(js)):
        return {"restaure": False}
    cour = os.path.join(rep, "reprise.champs")
    if os.path.exists(cour):
        os.remove(cour)
    os.link(src, cour)
    shutil.copy2(js, os.path.join(rep, "reprise.json"))
    return {"restaure": True}


def nettoyer(rep):
    n = 0
    for nom in os.listdir(rep):
        if nom == "reprise.json" or nom.startswith("reprise.champs"):
            os.remove(os.path.join(rep, nom))
            n += 1
    return {"retires": n}


def noyer(rep):
    n = 0
    for nom in os.listdir(rep):
        if re.fullmatch(r"reprise\.champs(\.json)?\.w\d+", nom):
            os.remove(os.path.join(rep, nom))
            n += 1
    return {"retires": n}


def libre(chemin):
    p = chemin
    while not os.path.exists(p) and p != "/":
        p = os.path.dirname(p)
    u = shutil.disk_usage(p)
    return {"libre_octets": u.free, "total_octets": u.total}


def memoire():
    with open("/proc/meminfo") as f:
        for ligne in f:
            if ligne.startswith("MemAvailable:"):
                return {"memavailable_octets": int(ligne.split()[1]) * 1024}
    return {"memavailable_octets": None}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    c, a = argv[1], argv[2:]
    try:
        if c == "info":
            r = info(a[0])
        elif c == "reparer":
            r = reparer(a[0])
        elif c == "sauvegarder":
            r = sauvegarder(a[0], int(a[1]))
        elif c == "restaurer":
            r = restaurer(a[0], int(a[1]))
        elif c == "nettoyer":
            r = nettoyer(a[0])
        elif c == "noyer":
            r = noyer(a[0])
        elif c == "libre":
            r = libre(a[0])
        elif c == "memoire":
            r = memoire()
        else:
            print(__doc__)
            return 2
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"erreur": str(exc)}))
        return 1
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
