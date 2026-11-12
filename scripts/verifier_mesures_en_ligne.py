#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Vérifier, en lecture seule, les fichiers des mesures en ligne de runs `kvu su2 gen --mesures …`.

Usage : python3 scripts/verifier_mesures_en_ligne.py DOSSIER [DOSSIER…]

Chaque DOSSIER est un run (répertoire contenant `manifest.json`) ou un répertoire qui en contient plusieurs. Pour chaque
run TERMINÉ (fichier `COMPLETE`) lancé avec `--mesures`, le script relit la commande de création dans `manifest.json`
(`--mesures SPEC` ou `--mesures=SPEC`), en déduit les pas où chaque mesure devait être faite, puis contrôle chaque fichier
de chaque chaîne (le run lui-même, ou ses sous-répertoires `c000`, `c001`… pour un lot) :

    wilson:R<r>xT<t>[:toutes]:<P>              wilson.csv         clé (pas, r, t), r·t lignes par pas
    polyakov_corr:<P>                          polyakov_corr.csv  clé (pas, r), L/2 + 1 lignes par pas
    flot:<dt>:<tmax>:<P>[:cible=…][:chaque=N]  flot.csv           clé (pas, t), même nombre de lignes à chaque pas
                                               flot_t0.csv        clé (pas), une ligne par pas (`nan` si la cible n'est pas atteinte)
    bloc:3:<P>                                 bloc3.csv, bloc3_hist.csv   clé (pas), une ligne par pas

Une mesure de période P > 0 est faite aux pas `therm + m·skip` pour m = P, 2P, … ≤ nmeas ; avec P = 0 ou P > nmeas, aucun
fichier n'est attendu. Verdicts : OK ; ABSENT (fichier attendu inexistant : cas de `--bord --mesures` en 0.1.0) ; TRONQUÉ (pas
attendus manquants : reprise sans `--mesures` en 0.1.0) ; DOUBLON (clé répétée : reprise après un arrêt brutal en 0.1.0) ;
INCOMPLET (nombre de lignes d'un pas inattendu, ligne partielle ou champ manquant) ; INATTENDU (pas hors calendrier, ou
en-tête inconnu). Les runs sans `COMPLETE`
sont signalés « non terminé » et ne sont pas jugés. La chaîne elle-même (`measurements.csv`, `therm.csv`, dumps) n'est pas
concernée par ces défauts et n'est pas relue.

Code de retour : 0 si tous les runs terminés sont sains, 1 si au moins un fichier pose problème, 2 en cas d'erreur d'usage.
"""

import json
import os
import re
import sys

# fichier -> (en-tête exact, nombre de colonnes de la clé, dernier champ vide toléré)
FORMATS = {
    "wilson.csv": ("pas,r,t,W", 3, False),
    "polyakov_corr.csv": ("pas,r,C", 2, False),
    "flot.csv": ("pas,t,E_plaq,E_trefle,t2E,Q", 2, False),
    "flot_t0.csv": ("pas,t0,w0", 1, False),
    # `sha256_cfg` est vide quand la configuration grossière n'est pas écrite
    "bloc3.csv": ("pas,l_grossier,n_boucles,fraction_hors,theta_max,n_ambigues,plaq_grossiere,sha256_cfg", 1, True),
    "bloc3_hist.csv": ("pas," + ",".join("h%d" % i for i in range(64)), 1, False),
}
ELAGUER = {"target", ".git", "node_modules", "__pycache__"}


def spec_de(argv):
    """Spécification `--mesures` de la commande de création (les deux syntaxes), ou None."""
    for i, a in enumerate(argv):
        if a == "--mesures" and i + 1 < len(argv):
            return argv[i + 1]
        if isinstance(a, str) and a.startswith("--mesures="):
            return a.split("=", 1)[1]
    return None


def analyser_element(element, l):
    """Rend (fichiers [(nom, lignes par pas ou None)], période) ou None si l'élément est inconnu."""
    c = [x.strip() for x in element.strip().split(":")]
    try:
        if c[0] == "wilson":
            m = re.fullmatch(r"[Rr](\d+)[xX][Tt](\d+)", c[1])  # mêmes formes que `obs/enligne.rs::rxt`
            reste = [x for x in c[2:] if x != "toutes"]
            return [("wilson.csv", int(m.group(1)) * int(m.group(2)))], int(reste[0])
        if c[0] == "polyakov_corr":
            return [("polyakov_corr.csv", (l // 2 + 1) if l else None)], int(c[1])
        if c[0] == "flot":
            return [("flot.csv", None), ("flot_t0.csv", 1)], int(c[3])
        if c[0] == "bloc":
            return [("bloc3.csv", 1), ("bloc3_hist.csv", 1)], int(c[2])
    except (AttributeError, IndexError, ValueError):
        return None
    return None


def controler(chemin, nom, attendus, par_pas):
    """Rend (verdict, détail) pour un fichier de mesure."""
    entete, ncle, dernier_vide = FORMATS[nom]
    ncol = entete.count(",") + 1
    if not os.path.exists(chemin):
        return "ABSENT", "%d pas attendus" % len(attendus)
    cles = set()
    compte = {}
    doublons = 0
    try:
        with open(chemin, "r", encoding="utf-8", errors="replace", newline="") as f:
            premiere = f.readline()
            if premiere.rstrip("\n") != entete:
                return "INATTENDU", "en-tête « %s » au lieu de « %s »" % (premiere.rstrip("\n")[:40], entete[:40])
            for ligne in f:
                if not ligne.endswith("\n"):
                    return "INCOMPLET", "ligne finale partielle (écriture interrompue) : « %s »" % ligne[:40]
                champs = ligne[:-1].split(",")
                if not champs[0].strip().isdigit():
                    return "INATTENDU", "ligne illisible : « %s »" % ligne[:40].strip()
                vides = [x for x in (champs[:-1] if dernier_vide else champs) if not x.strip()]
                if len(champs) != ncol or vides:
                    return "INCOMPLET", "ligne à %d champ(s) au lieu de %d, ou champ vide : « %s »" % (len(champs), ncol, ligne[:40].strip())
                cle = tuple(x.strip() for x in champs[:ncle])
                if cle in cles:
                    doublons += 1
                cles.add(cle)
                p = int(champs[0])
                compte[p] = compte.get(p, 0) + 1
    except OSError as e:
        return "INATTENDU", str(e)
    if doublons:
        return "DOUBLON", "%d ligne(s) dont la clé est déjà présente" % doublons
    hors = sorted(p for p in compte if p not in set(attendus))
    if hors:
        return "INATTENDU", "pas hors calendrier : %s" % hors[:5]
    manquants = [p for p in attendus if p not in compte]
    if manquants:
        dernier = max(compte) if compte else "aucun"
        return "TRONQUÉ", "%d pas présents sur %d (premier manquant : %d ; dernier présent : %s)" % (
            len(compte), len(attendus), manquants[0], dernier)
    tailles = sorted(set(compte.values()))
    if par_pas is not None and tailles != [par_pas]:
        return "INCOMPLET", "lignes par pas : %s au lieu de %d" % (tailles[:6], par_pas)
    if par_pas is None and len(tailles) > 1:
        return "INCOMPLET", "nombre de lignes variable d'un pas à l'autre : %s" % tailles[:6]
    return "OK", "%d pas" % len(attendus)


def auditer(manifeste):
    """Rend None si ce n'est pas un run `su2 gen` avec mesures, sinon (termine, [(fichier, verdict, détail)])."""
    try:
        with open(manifeste, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    argv = m.get("argv") if isinstance(m, dict) else None
    if not isinstance(argv, list) or "su2" not in argv or "gen" not in argv:
        return None
    spec = spec_de(argv)
    if spec is None or not spec.replace(",", "").strip():
        return None
    rep = os.path.dirname(manifeste)
    cfg = m.get("config_run")
    if not isinstance(cfg, dict):
        try:
            with open(os.path.join(rep, "config_run.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
    termine = os.path.exists(os.path.join(rep, "COMPLETE"))
    try:
        therm, nmeas, skip = int(cfg["therm"]), int(cfg["nmeas"]), int(cfg["skip"])
    except (KeyError, TypeError, ValueError):
        return termine, [("config_run", "INATTENDU", "therm, nmeas ou skip illisibles")]
    l = m.get("l") if isinstance(m.get("l"), int) else None
    chaines = sorted(d for d in os.listdir(rep) if re.fullmatch(r"c\d+", d) and os.path.isdir(os.path.join(rep, d)))
    dossiers = [(d, os.path.join(rep, d)) for d in chaines] or [("", rep)]
    resultats = []
    for element in spec.split(","):
        if not element.strip():
            continue
        a = analyser_element(element, l)
        if a is None:
            resultats.append((element.strip(), "INATTENDU", "élément de --mesures inconnu de ce script : non vérifié"))
            continue
        fichiers, periode = a
        attendus = [therm + k * skip for k in range(periode, nmeas + 1, periode)] if periode > 0 else []
        if not attendus:
            continue
        for prefixe, dossier in dossiers:
            for nom, par_pas in fichiers:
                verdict, detail = controler(os.path.join(dossier, nom), nom, attendus, par_pas)
                resultats.append((os.path.join(prefixe, nom), verdict, detail))
    return termine, resultats


def main():
    racines = sys.argv[1:]
    if not racines or racines[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    runs = problemes = non_termines = 0
    for racine in racines:
        if not os.path.isdir(racine):
            print("erreur : « %s » n'est pas un répertoire" % racine, file=sys.stderr)
            return 2
        for dossier, sous, fichiers in os.walk(racine):
            sous[:] = sorted(d for d in sous if d not in ELAGUER)
            if "manifest.json" not in fichiers:
                continue
            r = auditer(os.path.join(dossier, "manifest.json"))
            if r is None:
                continue
            termine, resultats = r
            if not termine:
                non_termines += 1
                print("NON TERMINÉ %s : pas de COMPLETE, non jugé" % dossier)
                continue
            runs += 1
            mauvais = [x for x in resultats if x[1] != "OK"]
            if mauvais:
                problemes += 1
                for fichier, verdict, detail in mauvais:
                    print("PROBLÈME %s : %s : %s (%s)" % (dossier, fichier, verdict, detail))
            else:
                print("OK %s : %d fichier(s) de mesure" % (dossier, len(resultats)))
    print("bilan : %d run(s) terminé(s) avec mesures en ligne, %d avec problème, %d non terminé(s)" % (runs, problemes, non_termines))
    return 1 if problemes else 0


if __name__ == "__main__":
    sys.exit(main())
