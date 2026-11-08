"""WP14 : ordonnanceur de travaux. Il ne décide de RIEN : il lit un fichier de travaux (TSV) et lance chaque travail par le LANCEUR UNIQUE
(`scripts/executer.py lancer`, branche procedure-allegee-2026-11-01) ; le manifeste de chaque travail est écrit par le lanceur, jamais par ce script.

Usage (python3 -I -B) :
    driver.py TRAVAUX.tsv --executer EXECUTER.py --depot DEPOT --sha SHA --manifestes DIR --creneaux N [--nice 10] [--ram-min-go 20] [--disque-min-go 100]

Format du TSV (une ligne par travail, # = commentaire) : nom <TAB> apres <TAB> duree_max_s <TAB> mem_max <TAB> entrees <TAB> sorties <TAB> commande
  apres   : noms (séparés par des virgules) de travaux qui doivent être TERMINÉS avec rc 0 avant celui-ci, ou « - » ;
  entrees : fichiers à hacher avant et après (séparés par « ; ») ; sorties : fichiers produits à hacher (séparés par « ; ») ;
  commande : argv découpé par shlex.
Règles : un travail dont le manifeste existe déjà (jamais réécrit) et porte rc 0 est SAUTÉ ; un travail déjà tenté avec un autre rc n'est pas relancé ici ;
un refus AVANT lancement (rc 75 mémoire, 76 garde, 77 disque, 78 échéance, 125 lanceur) est réessayé au plus 2 fois, sous un nouveau nom de manifeste
(« nom.a2 », « nom.a3 ») : rien n'avait démarré. Un travail DÉMARRÉ (tout autre rc) n'est JAMAIS relancé. Un travail dont une dépendance a échoué n'est pas lancé.
Chaque créneau a son propre verrou (/tmp/kvu_wp14_creneau<k>.lock) : au plus N travaux simultanés (≤ 12, un fil chacun).
Sortie standard : une ligne par événement (DEBUT, FIN rc, SAUT, ABANDON)."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time


def lire_travaux(chemin):
    travaux, noms = [], set()
    for ligne in open(chemin, encoding="utf-8"):
        if not ligne.strip() or ligne.lstrip().startswith("#"):
            continue
        c = ligne.rstrip("\n").split("\t")
        if len(c) != 7:
            raise SystemExit(f"driver : ligne mal formée ({len(c)} champs) : {ligne[:80]}")
        nom, apres, duree, mem, entrees, sorties, cmd = c
        if nom in noms:
            raise SystemExit(f"driver : nom en double : {nom}")
        noms.add(nom)
        travaux.append({"nom": nom, "apres": [] if apres == "-" else apres.split(","), "duree": int(duree), "mem": mem,
                        "entrees": [x for x in entrees.split(";") if x and x != "-"], "sorties": [x for x in sorties.split(";") if x and x != "-"],
                        "argv": shlex.split(cmd)})
    for t in travaux:
        for d in t["apres"]:
            if d not in noms:
                raise SystemExit(f"driver : dépendance inconnue {d} (travail {t['nom']})")
    return travaux


def etat_manifeste(chemin):
    try:
        d = json.load(open(chemin))
        return d.get("statut"), d.get("rc")
    except (OSError, ValueError):
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("travaux")
    ap.add_argument("--executer", required=True)
    ap.add_argument("--depot", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--manifestes", required=True)
    ap.add_argument("--creneaux", type=int, required=True)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--ram-min-go", type=int, default=20)
    ap.add_argument("--disque-min-go", type=int, default=100)
    ap.add_argument("--attente-verrou", type=int, default=86400)
    a = ap.parse_args()
    if not 1 <= a.creneaux <= 12:
        raise SystemExit("driver : --creneaux entre 1 et 12")
    os.makedirs(a.manifestes, exist_ok=True)
    travaux = lire_travaux(a.travaux)
    par_nom = {t["nom"]: t for t in travaux}
    verrou_io = threading.Lock()
    etat = {t["nom"]: "attente" for t in travaux}   # attente | en_cours | ok | echec | abandon
    libres = list(range(a.creneaux))
    cond = threading.Condition()

    def log(msg):
        with verrou_io:
            print(time.strftime("%F %T ") + msg, flush=True)

    def lancer_travail(t, k):
        tentative = 1
        while True:
            nom_manif = t["nom"] if tentative == 1 else f"{t['nom']}.a{tentative}"
            manif = os.path.join(a.manifestes, nom_manif + ".json")
            statut, rc_prec = etat_manifeste(manif)
            if statut == "TERMINE" and rc_prec == 0:
                log(f"SAUT {t['nom']} (manifeste TERMINE rc 0 déjà présent)")
                return "ok"
            if statut is not None:   # manifeste existant autre que rc 0 : jamais réécrit ; on passe à la tentative suivante seulement si rien n'avait démarré
                if statut == "REFUSE" and tentative < 3:
                    tentative += 1
                    continue
                log(f"ABANDON {t['nom']} : manifeste existant {statut} rc {rc_prec} (jamais relancé)")
                return "echec"
            cmd = [sys.executable, "-I", "-B", a.executer, "lancer", "--manifeste", manif, "--etiquette", "wp14:" + t["nom"], "--cwd", a.depot, "--depot", a.depot,
                   "--sha", a.sha, "--propre", "--verrou", f"/tmp/kvu_wp14_creneau{k}.lock", "--attente-verrou", str(a.attente_verrou), "--nice", str(a.nice),
                   "--mem-max", t["mem"], "--duree-max", str(t["duree"]), "--ram-min-go", str(a.ram_min_go), "--ram-essais", "3", "--ram-intervalle", "60",
                   "--disque-min-go", str(a.disque_min_go), "--disque-chemin", a.manifestes]
            for e in t["entrees"]:
                cmd += ["--entree", e]
            for s in t["sorties"]:
                cmd += ["--sortie", s]
            cmd += ["--"] + t["argv"]
            log(f"DEBUT {t['nom']} (créneau {k}, tentative {tentative})")
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            statut, rc_man = etat_manifeste(manif)
            log(f"FIN {t['nom']} rc lanceur {rc}, manifeste {statut} rc {rc_man}")
            if statut == "TERMINE" and rc_man == 0:
                return "ok"
            if statut == "REFUSE" and tentative < 3:   # rien n'a démarré : nouvelle tentative sous un nouveau nom
                tentative += 1
                time.sleep(30)
                continue
            return "echec"

    def travailleur(t, k):
        res = lancer_travail(t, k)
        with cond:
            etat[t["nom"]] = res
            libres.append(k)
            cond.notify_all()

    fils = []
    with cond:
        while True:
            for t in travaux:
                if etat[t["nom"]] != "attente":
                    continue
                if any(etat[d] in ("echec", "abandon") for d in t["apres"]):
                    etat[t["nom"]] = "abandon"
                    log(f"ABANDON {t['nom']} : dépendance en échec")
                    continue
                if all(etat[d] == "ok" for d in t["apres"]) and libres:
                    k = libres.pop(0)
                    etat[t["nom"]] = "en_cours"
                    th = threading.Thread(target=travailleur, args=(t, k), daemon=False)
                    th.start()
                    fils.append(th)
            if all(v in ("ok", "echec", "abandon") for v in etat.values()):
                break
            cond.wait(timeout=5.0)
    for th in fils:
        th.join()
    bilan = {v: sum(1 for x in etat.values() if x == v) for v in ("ok", "echec", "abandon")}
    log(f"BILAN {bilan}")
    return 0 if bilan["echec"] == 0 and bilan["abandon"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
