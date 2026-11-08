#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Planifier et exécuter des lots de boîtes.

Verrou GPU : `kvu` prend lui-même celui de sa machine, mais un `kvu` lancé par ssh sans
`kvu.toml` ne reconnaît pas le poste et prend le verrou par défaut (/tmp/kevinotron_gpu.lock)
au lieu du sien (/tmp/gamer_gpu.lock). Chaque cible ssh doit donc déclarer son verrou
(`--verrou NOM=CHEMIN`, ou `NOM=aucun`) et le pilote enveloppe chaque lot dans `flock`.

Mode de test documenté : une cible local: peut désigner tout exécutable,
notamment un faux kvu qui produit les fichiers du contrat. Le pilote transmet
l'environnement (durée et échecs du faux commandés par le test).
Seul le chemin ssh utilise une commande distante échappée par shlex.quote.
"""

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid


BORDS = {"cold", "gauge", "z2full", "z2_01", "flux", "haar", "therm"}
OPTIONS = {"ref-k1", "compat-prod"}
PARAMETRES = ("therm", "nmeas", "skip", "nor", "options", "profil_tous")
CELLULE = ("ell", "beta") + PARAMETRES
CHAMPS_TACHE = ("ligne", "ell", "beta", "bord", "graine") + PARAMETRES
# Code de sortie de `flock -E` quand le verrou n'est pas obtenu dans le délai (kvu n'en rend jamais : 2, 3, 5, 6, 7, 8).
CODE_VERROU = 75


class ErreurUsage(Exception):
    """Erreur d'entrée ou de planification (code 2)."""


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        traductions = {
            "unrecognized arguments:": "arguments inconnus :",
            "expected one argument": "une valeur est attendue",
            "not allowed with argument": "incompatible avec l'argument",
        }
        for anglais, francais in traductions.items():
            message = message.replace(anglais, francais)
        self.print_usage(sys.stderr)
        self.exit(2, f"erreur : {message}\n")

    def format_usage(self):
        return super().format_usage().replace("usage:", "utilisation :", 1)

    def format_help(self):
        return super().format_help().replace("usage:", "utilisation :", 1)


def arguments():
    p = Arguments(description="Pilote de balayage des boîtes à bord fixé.",
                  add_help=False, allow_abbrev=False,
                  epilog="Mode de test : local: accepte un faux exécutable kvu ; "
                  "son environnement est transmis sans modification. Le verrou GPU "
                  "est celui de --verrou (flock) ou, à défaut, celui que kvu prend. Mémoire en Go décimaux (10⁹ octets). "
                  "Les numéros de lots continuent entre cellules de mêmes ell et beta "
                  "afin d'éviter les collisions. Une reprise exige le même plan.")
    p._positionals.title = "arguments positionnels"
    p._optionals.title = "options"
    p.add_argument("taches", nargs="?", metavar="TACHES.csv", help="fichier CSV des chaînes")
    p.add_argument("--sortie", metavar="DIR", help="répertoire neuf des résultats")
    p.add_argument("--cible", action="append", default=[], metavar="NOM=TYPE:CHEMIN",
                   help="répétable : NOM=local:KVU ou NOM=ssh:HOTE:KVU")
    p.add_argument("--debit", action="append", default=[], metavar="NOM=VALEUR",
                   help="liens libres/s de la cible (défaut : 5e7)")
    p.add_argument("--verrou", action="append", default=[], metavar="NOM=CHEMIN|aucun",
                   help="verrou GPU de la cible (chemin absolu ou ~/…) : chaque lot est lancé sous `flock` sur ce fichier. "
                   "Obligatoire pour une cible ssh ; pour local:, « aucun » (défaut) laisse `kvu` prendre le verrou de sa machine")
    for nom, valeur, aide in (
        ("therm", "1000", "pas de thermalisation"),
        ("nmeas", "30000", "nombre de mesures (au moins 1)"),
        ("skip", "1", "pas entre mesures"),
        ("nor", "2", "nombre de surrelaxations"),
        ("profil-tous", "10", "période du profil"),
        ("chaines-max", "128", "nombre maximal de chaînes par lot"),
        ("mem-gpu-go", "6.0", "mémoire maximale par GPU en Go"),
        ("fenetre-s", "600", "fenêtre maximale par lot en secondes"),
        ("attente-verrou-s", "7200", "attente maximale du verrou GPU (par `flock` si --verrou, sinon par `kvu`), ajoutée au délai de chaque lot"),
    ):
        p.add_argument("--" + nom, default=valeur, metavar="N",
                       help=f"{aide} (défaut : {valeur})")
    p.add_argument("--options", default="ref-k1", metavar="CHOIX",
                   help="ref-k1 ou compat-prod (défaut : ref-k1)")
    p.add_argument("--rep-distant", default="~/kvu_pilote", metavar="DIR",
                   help="racine distante des sessions (défaut : ~/kvu_pilote)")
    p.add_argument("--reprendre", action="store_true", help="autoriser une sortie existante")
    p.add_argument("--reprendre-incomplets", action="store_true",
                   help="nouvelle tentative suffixée _r2, _r3… ; conserver les anciennes")
    p.add_argument("--dry-run", action="store_true",
                   help="imprimer un plan LPT déterministe, sans écriture ni exécution")
    p.add_argument("--aide", "-h", action="help", help="afficher cette aide et quitter")
    a = p.parse_args()
    if not a.taches or not a.sortie or not a.cible:
        p.error("TACHES.csv, --sortie et au moins une --cible sont obligatoires")
    for champ in ("therm", "nmeas", "skip", "nor", "profil_tous", "chaines_max"):
        minimum = 1 if champ in ("nmeas", "chaines_max") else 0
        setattr(a, champ, entier(getattr(a, champ), champ, minimum))
    for champ in ("mem_gpu_go", "fenetre_s", "attente_verrou_s"):
        setattr(a, champ, positif(getattr(a, champ), champ))
    if a.options not in OPTIONS:
        raise ErreurUsage("--options : choisir ref-k1 ou compat-prod")
    if not a.rep_distant or "\0" in a.rep_distant:
        raise ErreurUsage("--rep-distant : chemin invalide")
    return a


def entier(texte, champ, minimum=0):
    if not re.fullmatch(r"\+?[0-9]+", str(texte)):
        raise ErreurUsage(f"{champ} : entier ≥ {minimum} attendu")
    try:
        valeur = int(texte)
    except ValueError:
        raise ErreurUsage(f"{champ} : entier trop grand") from None
    if valeur < minimum:
        raise ErreurUsage(f"{champ} : entier ≥ {minimum} attendu")
    return valeur


def positif(texte, champ):
    try:
        valeur = float(texte)
    except (ValueError, OverflowError):
        raise ErreurUsage(f"{champ} : nombre fini strictement positif attendu") from None
    if not math.isfinite(valeur) or valeur <= 0:
        raise ErreurUsage(f"{champ} : nombre fini strictement positif attendu")
    return valeur


def lire_cibles(a):
    cibles = {}
    for texte in a.cible:
        nom, sep, definition = texte.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", nom) or nom in cibles:
            raise ErreurUsage(f"cible invalide ou répétée : {texte}")
        genre, sep, reste = definition.partition(":")
        hote = None
        if genre == "ssh":
            hote, sep, reste = reste.partition(":")
            if not re.fullmatch(r"[A-Za-z0-9_@][A-Za-z0-9_.@-]*", hote):
                raise ErreurUsage(f"hôte ssh invalide : {hote}")
        if genre not in ("local", "ssh") or not sep or not reste or "\0" in reste:
            raise ErreurUsage(f"cible invalide : {texte}")
        cibles[nom] = dict(type=genre, kvu=reste, hote=hote, debit=5e7)
    vus = set()
    for texte in a.debit:
        nom, sep, valeur = texte.partition("=")
        if not sep or nom not in cibles or nom in vus:
            raise ErreurUsage(f"débit : cible inconnue ou répétée : {texte}")
        cibles[nom]["debit"] = positif(valeur, f"débit de {nom}")
        vus.add(nom)
    lire_verrous(a, cibles)
    return cibles


def lire_verrous(a, cibles):
    """`--verrou NOM=CHEMIN` : le pilote enveloppe chaque lot de la cible dans `flock` ; `NOM=aucun` : pas d'enveloppe.
    Une cible ssh DOIT déclarer son verrou : essai réel du 25/10, un lot ssh est parti 7 s avant la libération du verrou
    du gamer parce que le `kvu` distant, sans kvu.toml, prenait /tmp/kevinotron_gpu.lock."""
    for nom in cibles:
        cibles[nom]["verrou"] = None
    vus = set()
    for texte in a.verrou:
        nom, sep, valeur = texte.partition("=")
        if not sep or nom not in cibles or nom in vus:
            raise ErreurUsage(f"verrou : cible inconnue ou répétée : {texte}")
        vus.add(nom)
        if valeur == "aucun":
            continue
        if not (valeur.startswith("/") or valeur.startswith("~/")) or len(valeur) < 3 or "\0" in valeur or "\n" in valeur:
            raise ErreurUsage(f"verrou : chemin absolu ou ~/… attendu, ou « aucun » : {texte}")
        cibles[nom]["verrou"] = valeur
    for nom, cible in cibles.items():
        if cible["type"] == "ssh" and nom not in vus:
            raise ErreurUsage(f"cible ssh « {nom} » : verrou GPU non déclaré ; ajouter --verrou {nom}=CHEMIN "
                              f"(par exemple /tmp/gamer_gpu.lock) ou --verrou {nom}=aucun")
        if cible["type"] == "local" and cible["verrou"] and shutil.which("flock") is None:
            raise ErreurUsage(f"cible locale « {nom} » : --verrou demande `flock`, absent de ce poste")


def mots_verrou(cible, a, citer):
    """Arguments `flock` qui précèdent la commande `kvu` (liste vide sans verrou) ; `citer` protège le chemin du verrou."""
    if not cible.get("verrou"):
        return []
    return ["flock", "-E", str(CODE_VERROU), "-w", f"{a.attente_verrou_s:g}", citer(cible["verrou"])]


def lire_taches(chemin, a):
    with open(chemin, encoding="utf-8-sig", newline="") as fichier:
        lignes = [(n, s) for n, s in enumerate(fichier, 1)
                  if s.strip() and not s.lstrip().startswith("#")]
    if not lignes:
        raise ErreurUsage("ligne 1 : en-tête CSV manquant")
    lecteur = csv.reader((s for _, s in lignes), strict=True)
    numero = lignes[0][0]
    try:
        entete = [s.strip() for s in next(lecteur)]
        obligatoires = {"ell", "beta", "bord", "graine"}
        if len(set(entete)) != len(entete) or not obligatoires.issubset(entete):
            raise ErreurUsage(f"ligne {numero} : en-tête requis : ell,beta,bord,graine, sans doublon")
        taches, vues = [], set()
        while lecteur.line_num < len(lignes):
            numero = lignes[lecteur.line_num][0]
            valeurs = next(lecteur)
            if len(valeurs) > len(entete):
                raise ErreurUsage("trop de colonnes")
            brut = dict(zip(entete, (s.strip() for s in valeurs)))
            if any(not brut.get(c) for c in obligatoires):
                raise ErreurUsage("valeur obligatoire manquante")
            t = dict(ligne=numero, ell=entier(brut["ell"], "ell", 2),
                     beta=positif(brut["beta"], "beta"), bord=brut["bord"],
                     graine=entier(brut["graine"], "graine"))
            if t["ell"] % 2:
                raise ErreurUsage("ell doit être pair et ≥ 2")
            if t["bord"] not in BORDS:
                raise ErreurUsage(f"bord inconnu : {t['bord']}")
            for champ in PARAMETRES:
                valeur = brut.get(champ) or getattr(a, champ)
                if champ == "options":
                    if valeur not in OPTIONS:
                        raise ErreurUsage("options : choisir ref-k1 ou compat-prod")
                    t[champ] = valeur
                else:
                    t[champ] = entier(valeur, champ, 1 if champ == "nmeas" else 0)
            cle = tuple(t[c] for c in ("ell", "beta", "bord", "graine"))
            if cle in vues:
                raise ErreurUsage("doublon (ell, beta, bord, graine)")
            vues.add(cle)
            taches.append(t)
    except (ErreurUsage, csv.Error) as e:
        raise ErreurUsage(f"ligne {numero} : {e}") from None
    if not taches:
        raise ErreurUsage(f"ligne {numero} : aucune tâche")
    return taches


def temps_estime(ell, therm, nmeas, skip, k, nor, debit):
    """Fonction pure : secondes, avec 30 s fixes et 4·ell·(ell−1)³ liens."""
    try:
        return 30.0 + ((therm + nmeas * skip) * k * 4 * ell * (ell - 1) ** 3
                       * (1 + nor)) / debit
    except OverflowError:
        return math.inf


def estimer(t, k, debit):
    return temps_estime(t["ell"], t["therm"], t["nmeas"], t["skip"], k, t["nor"], debit)


def planifier(taches, cibles, a):
    cellules = {}
    for t in taches:
        cellules.setdefault(tuple(t[c] for c in CELLULE), []).append(t)
    lent = min(c["debit"] for c in cibles.values())
    lots, numeros = [], {}
    for chaines in cellules.values():
        t = chaines[0]
        # Comparaison entière en dixièmes d'octet pour le facteur 1,3.
        memoire_dix = 16 * (t["ell"] + 2) ** 4 * 8 * 13
        budget_dix = int(a.mem_gpu_go * 1_000_000_000) * 10
        temps = estimer(t, 1, lent)
        if temps > a.fenetre_s or memoire_dix > budget_dix:
            raise ErreurUsage(f"cellule ell={t['ell']}, beta={t['beta']!r} : "
                              f"une chaîne, temps estimé {temps:.9g} s ; "
                              f"fenêtre {a.fenetre_s:g} s ou mémoire dépassée")
        haut = min(len(chaines), a.chaines_max, budget_dix // memoire_dix)
        bas = 1
        while bas < haut:
            milieu = (bas + haut + 1) // 2
            if estimer(t, milieu, lent) <= a.fenetre_s:
                bas = milieu
            else:
                haut = milieu - 1
        nombre = (len(chaines) + bas - 1) // bas
        taille, reste = divmod(len(chaines), nombre)
        position = 0
        paire = (t["ell"], t["beta"])
        for n in range(nombre):
            k = taille + (n < reste)
            rang = numeros.get(paire, 0)
            numeros[paire] = rang + 1
            ident = f"L{t['ell']:02d}_b{t['beta']!r}_{rang:03d}"
            membres = chaines[position:position + k]
            lots.append(dict(id=ident, base=ident, membres=membres, k=k,
                             ell=t["ell"], beta=t["beta"], t_est_s=estimer(t, k, lent)))
            position += k
    return sorted(lots, key=lambda lot: (-lot["t_est_s"], lot["id"]))


def simulation(lots, cibles):
    charges = {nom: 0.0 for nom in cibles}
    previsions = {}
    for lot in lots:
        durees = {nom: estimer(lot["membres"][0], lot["k"], c["debit"])
                  for nom, c in cibles.items()}
        nom = min(cibles, key=lambda n: (charges[n] + durees[n], n))
        charges[nom] += durees[nom]
        previsions[lot["id"]] = (nom, durees[nom])
    return previsions, charges


def afficher_plan(lots, cibles):
    previsions, charges = simulation(lots, cibles)
    print("identifiant cible K ell beta t_est_s")
    for lot in lots:
        nom, duree = previsions[lot["id"]]
        print(f"{lot['id']} {nom} {lot['k']} {lot['ell']} {lot['beta']!r} {duree:.12g}")
    for nom in sorted(charges):
        print(f"TOTAL {nom} {charges[nom]:.12g} s")
    print("PLAN OK")


def chemin_shell(chemin):
    """Développer seulement ~/ sur le distant, en protégeant tout le suffixe."""
    if chemin.startswith("~/"):
        return '"$HOME"/' + shlex.quote(chemin[2:])
    return shlex.quote(chemin)


def chemin_rsync(chemin):
    """Chemin distant pour rsync : `-s` (arguments protégés) évite tout shell distant, donc ni `$HOME` ni guillemets ;
    `~/…` devient un chemin relatif au répertoire personnel distant (défaut de rsync), un chemin absolu reste tel quel."""
    return chemin[2:] if chemin.startswith("~/") else chemin


def commande(lot, cible, sortie, session, a):
    t = lot["membres"][0]
    distant = a.rep_distant.rstrip("/") + "/" + session + "/" + lot["id"]
    repertoire = str(sortie / "lots" / lot["id"]) if cible["type"] == "local" else distant
    cmd = [cible["kvu"], "su2", "boite", "--ell", str(t["ell"]), "--beta", repr(t["beta"]),
           "--chaines", ",".join(f"{c['bord']}:{c['graine']}" for c in lot["membres"])]
    for champ in PARAMETRES:
        cmd.extend(["--" + champ.replace("_", "-"), str(t[champ])])
    cmd.extend(["--backend", "auto", "--outdir", repertoire])
    if cible["type"] == "ssh":
        # Le parent de --outdir doit exister, mais le lot reste neuf (créé par kvu).
        parent = distant.rsplit("/", 1)[0]
        mots = mots_verrou(cible, a, chemin_shell) + [chemin_shell(cmd[0])] + [shlex.quote(x) for x in cmd[1:-1]]
        mots.append(chemin_shell(repertoire))
        texte = "mkdir -p -- " + chemin_shell(parent) + " && exec " + " ".join(mots)
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", cible["hote"], texte]
    else:
        cmd = mots_verrou(cible, a, os.path.expanduser) + cmd
    return cmd, distant


def existe(chemin):
    return os.path.lexists(chemin)


def lire_json(chemin):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def preparer_reprise(lots, taches, sortie, a):
    ancien = None
    if existe(sortie):
        if not a.reprendre:
            raise ErreurUsage("le dossier de sortie existe : --reprendre est exigé")
        if not sortie.is_dir():
            raise ErreurUsage("la sortie existante n'est pas un répertoire")
        if (sortie / "index.json").exists():
            try:
                ancien = lire_json(sortie / "index.json")
                anciennes_taches = [{c: t[c] for c in CHAMPS_TACHE} for t in ancien["taches"]]
                if ancien["schema"] != "pilote-boites/1" or anciennes_taches != taches:
                    raise ValueError()
                groupes = {}
                for t in ancien["taches"]:
                    base = re.sub(r"_r[0-9]+$", "", t["lot"])
                    groupes.setdefault(base, []).append((t["ligne"], t["chaine"]))
                attendu = {lot["base"]: [(t["ligne"], i) for i, t in enumerate(lot["membres"])]
                           for lot in lots}
                if groupes != attendu:
                    raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise ErreurUsage("index existant incompatible : la reprise exige le même fichier et le même plan") from None
    precedents = {re.sub(r"_r[0-9]+$", "", l["id"]): l for l in ancien["lots"]} if ancien else {}
    for lot in lots:
        precedent = precedents.get(lot["base"], {})
        ident = precedent.get("id", lot["base"])
        chemin = sortie / "lots" / ident
        statut = "attente"
        if (chemin / "COMPLETE").is_file():
            statut = "deja_fait"
        elif existe(chemin) or existe(sortie / "lots" / (ident + ".log")):
            statut = "incomplet_existant"
            if a.reprendre_incomplets:
                n = 2
                while (existe(sortie / "lots" / f"{lot['base']}_r{n}") or
                       existe(sortie / "lots" / f"{lot['base']}_r{n}.log")):
                    n += 1
                ident = f"{lot['base']}_r{n}"
                statut = "attente"
        lot["id"] = ident
        lot.update(cible=None, commande=[], statut=statut, rc=None, debut_unix=None,
                   fin_unix=None, duree_s=None, perf={}, verrou="non_determine", sortie="lots/" + ident)
        if statut != "attente":
            for champ in ("cible", "commande", "rc", "debut_unix", "fin_unix", "duree_s", "perf", "verrou"):
                if champ in precedent:
                    lot[champ] = precedent[champ]
    return ancien


def lire_resultats(lot, sortie, journal):
    try:
        with open(journal, encoding="utf-8", errors="replace") as fichier:
            for ligne in fichier:
                if ligne.startswith("PERF boite "):
                    lot["perf"] = dict(mot.split("=", 1) for mot in ligne.split()[2:] if "=" in mot)
    except OSError:
        pass
    try:
        manifeste = lire_json(sortie / lot["sortie"] / "manifest.json")
        if isinstance(manifeste, dict):
            if manifeste.get("verrou") in ("pris_ici", "ancetre", "partage", "ignore", "non_determine"):
                lot["verrou"] = manifeste["verrou"]
            for champ in ("debut_unix", "fin_unix"):
                v = manifeste.get(champ)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    lot[champ] = v
    except (OSError, ValueError):
        pass


class Execution:
    def __init__(self, lots, taches, cibles, sortie, session, a):
        self.lots, self.taches, self.cibles = lots, taches, cibles
        self.sortie, self.session, self.a = sortie, session, a
        self.mutex = threading.Lock()
        self.arret = threading.Event()
        self.interruption = False
        self.erreurs = []
        self.compteurs = {nom: 0 for nom in cibles}
        self.file = queue.Queue()
        for lot in lots:
            if lot["statut"] == "attente":
                self.file.put(lot)

    def evenement(self, message):
        date = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with open(self.sortie / "pilote.log", "a", encoding="utf-8") as fichier:
            fichier.write(f"{date} {message}\n")

    def index(self):
        document = dict(schema="pilote-boites/1", session=self.session,
                        taches=[{**t, "lot": lot["id"], "chaine": i}
                                for lot in self.lots for i, t in enumerate(lot["membres"])],
                        lots=[{k: v for k, v in lot.items() if k not in ("membres", "base")}
                              for lot in self.lots],
                        cibles={nom: {"type": c["type"], "verrou": c.get("verrou"), "lots": self.compteurs[nom]}
                                for nom, c in self.cibles.items()})
        document["taches"].sort(key=lambda t: t["ligne"])
        # Le temporaire reste dans la sortie, sur le même système de fichiers.
        fd, temporaire = tempfile.mkstemp(prefix=".index-", suffix=".json", dir=self.sortie)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fichier:
                json.dump(document, fichier, ensure_ascii=False, indent=2, allow_nan=False)
                fichier.write("\n")
            os.replace(temporaire, self.sortie / "index.json")
        finally:
            if existe(temporaire):
                os.unlink(temporaire)

    def processus(self, cmd, journal, delai):
        proc = subprocess.Popen(cmd, stdout=journal, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        limite = time.monotonic() + delai
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    return rc, None
                if self.arret.is_set() or time.monotonic() >= limite:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    cause = "interruption" if self.arret.is_set() else "délai maximal dépassé"
                    return proc.returncode, cause
                try:
                    return proc.wait(timeout=min(0.2, max(0.01, limite - time.monotonic()))), None
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def traiter(self, lot, nom):
        cible = self.cibles[nom]
        debut = time.time()
        horloge = time.monotonic()
        cmd, distant = commande(lot, cible, self.sortie, self.session, self.a)
        with self.mutex:
            lot.update(cible=nom, commande=cmd, statut="en_cours", debut_unix=debut,
                       t_est_s=estimer(lot["membres"][0], lot["k"], cible["debit"]))
            self.compteurs[nom] += 1
            self.evenement(f"démarrage {lot['id']} cible={nom}")
        journal = self.sortie / "lots" / (lot["id"] + ".log")
        rc, erreur = None, None
        try:
            with open(journal, "x", encoding="utf-8") as flux:
                # le délai compte aussi l'attente du verrou GPU (7200 s au plus), prise par flock (--verrou) ou par kvu
                rc, erreur = self.processus(cmd, flux, max(1800, 3 * lot["t_est_s"]) + self.a.attente_verrou_s)
                if rc == CODE_VERROU and cible.get("verrou") and erreur is None:
                    erreur = f"verrou GPU {cible['verrou']} non obtenu en {self.a.attente_verrou_s:g} s"
                if cible["type"] == "ssh" and not self.arret.is_set():
                    # Pas de suppression distante. rsync reçoit un répertoire neuf.
                    local = self.sortie / lot["sortie"]
                    local.mkdir()
                    transfert = ["rsync", "-a", "-s", "-e", "ssh -o BatchMode=yes -o ServerAliveInterval=15",
                                 "--", cible["hote"] + ":" + chemin_rsync(distant) + "/", str(local) + "/"]
                    rc_transfert, cause = self.processus(transfert, flux, max(1800, 3 * lot["t_est_s"]))
                    if rc_transfert or cause:
                        erreur = f"rapatriement en échec (rc={rc_transfert})" + (f" : {cause}" if cause else "")
        except OSError as e:
            erreur = f"exécution impossible : {e}"
        with self.mutex:
            lot.update(rc=rc, fin_unix=time.time(), duree_s=time.monotonic() - horloge)
            lire_resultats(lot, self.sortie, journal)
            complet = (self.sortie / lot["sortie"] / "COMPLETE").is_file()
            lot["statut"] = "ok" if rc == 0 and complet and erreur is None else "echec"
            if erreur:
                lot["erreur"] = erreur
            self.evenement(f"{'fin' if lot['statut'] == 'ok' else 'échec'} {lot['id']} "
                           f"cible={nom} rc={rc} statut={lot['statut']}" + (f" : {erreur}" if erreur else ""))
            self.index()

    def travail(self, nom):
        try:
            while not self.arret.is_set():
                try:
                    lot = self.file.get_nowait()
                except queue.Empty:
                    return
                self.traiter(lot, nom)
        except Exception as e:
            with self.mutex:
                self.erreurs.append(str(e))
                self.arret.set()

    def lancer(self):
        def interrompre(_signal, _cadre):
            self.interruption = True
            self.arret.set()

        ancien_signal = signal.signal(signal.SIGINT, interrompre)
        ancien_term = signal.signal(signal.SIGTERM, interrompre)
        fils = []
        try:
            self.index()
            for lot in self.lots:
                if lot["statut"] == "incomplet_existant":
                    self.evenement(f"échec {lot['id']} : incomplet_existant, conservé sans relance")
            for nom in self.cibles:
                fil = threading.Thread(target=self.travail, args=(nom,), name="cible-" + nom)
                fils.append(fil)
                fil.start()
            for fil in fils:
                while fil.is_alive():
                    fil.join(timeout=0.2)
            if self.interruption:
                with self.mutex:
                    self.evenement("interruption : arrêt des processus lancés par le pilote")
                    self.index()
                return 130
            if self.erreurs:
                print("erreur d'exécution : " + "; ".join(self.erreurs), file=sys.stderr)
                return 1
            return int(any(l["statut"] not in ("ok", "deja_fait") for l in self.lots))
        finally:
            self.arret.set()
            for fil in fils:
                fil.join()
            signal.signal(signal.SIGINT, ancien_signal)
            signal.signal(signal.SIGTERM, ancien_term)


def principal():
    try:
        a = arguments()
        cibles = lire_cibles(a)
        taches = lire_taches(a.taches, a)
        lots = planifier(taches, cibles, a)
        sortie = Path(a.sortie).absolute()
        ancien = preparer_reprise(lots, taches, sortie, a)
        if a.dry_run:
            afficher_plan(lots, cibles)
            return 0
        sortie.mkdir(parents=True, exist_ok=a.reprendre)
        (sortie / "lots").mkdir(exist_ok=True)
        session = ancien["session"] if ancien else "s_" + uuid.uuid4().hex
        return Execution(lots, taches, cibles, sortie, session, a).lancer()
    except (ErreurUsage, OSError, UnicodeError, ValueError, OverflowError) as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(principal())
