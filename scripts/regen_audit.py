# SPDX-License-Identifier: Apache-2.0
"""Auditer les ensembles WP4 et leur intégrité, exclusivement en lecture."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import time


CLASSES = ("A", "B", "C")
BLOC = 8 * 1024 * 1024


class ErreurUsage(ValueError):
    """Signaler des arguments incompatibles ou une sortie interdite."""


class Analyseur(argparse.ArgumentParser):
    """Présenter aussi les diagnostics automatiques en français."""

    def error(self, message):
        for anglais, francais in (
            ("unrecognized arguments:", "arguments inconnus :"),
            ("expected one argument", "une valeur est attendue"),
            ("argument ", "option "),
        ):
            message = message.replace(anglais, francais)
        raise ErreurUsage(message)


def _fils(valeur):
    try:
        nombre = int(valeur)
        if nombre > 0:
            return nombre
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("le nombre de fils doit être un entier strictement positif")


def _arguments(arguments):
    analyseur = Analyseur(
        description="Audit WP4 et contrôle d'intégrité sans écriture dans les ensembles.",
        add_help=False, usage="%(prog)s [options]",
    )
    analyseur._optionals.title = "options"
    analyseur.add_argument("-h", "--help", action="help", help="afficher cette aide")
    for option, aide in (
        ("--kvu", "binaire kvu à exécuter"),
        ("--ensembles", "liste nom|chemins|note des ensembles"),
        ("--sortie-json", "rapport JSON à écrire hors des ensembles"),
        ("--sortie-md", "rapport Markdown à écrire hors des ensembles"),
        ("--preuve-gbit", "preuve G-BIT transmise à kvu"),
        ("--integrite", "ensemble dont les fichiers doivent être hachés"),
    ):
        analyseur.add_argument(option, help=aide, metavar="CHEMIN")
    analyseur.add_argument("--fils", type=_fils, default=None, help="nombre de fils (défaut : 4)")
    options = analyseur.parse_args(arguments)
    if options.integrite is not None:
        if any((options.kvu, options.ensembles, options.sortie_md, options.preuve_gbit)):
            raise ErreurUsage("--integrite est incompatible avec les options du mode audit")
        options.fils = options.fils or 4
    else:
        if not all((options.kvu, options.ensembles, options.sortie_json, options.sortie_md)):
            raise ErreurUsage("l'audit exige --kvu, --ensembles, --sortie-json et --sortie-md")
        if options.fils is not None:
            raise ErreurUsage("--fils exige --integrite")
    return options


def _date():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sous(chemin, racine):
    return os.path.commonpath((chemin, racine)) == racine


def _verifier_sorties(sorties, racines):
    """Contrôler tous les chemins avant d'ouvrir la première sortie.

    Vérifier les chemins lexicaux ET résolus empêche les contournements par
    liens symboliques. Refuser les liens physiques protège aussi leurs cibles.
    Aucun répertoire de sortie n'est créé automatiquement.
    """
    vues = set()
    for sortie in filter(None, sorties):
        absolu = os.path.abspath(sortie)
        reel = os.path.realpath(sortie)
        for racine in racines:
            if (_sous(absolu, os.path.abspath(racine))
                    or _sous(reel, os.path.realpath(racine))):
                raise ErreurUsage(f"sortie interdite sous un ensemble : {sortie}")
        if reel in vues:
            raise ErreurUsage("les sorties JSON et Markdown doivent être distinctes")
        vues.add(reel)
        if os.path.exists(sortie) and os.stat(sortie).st_nlink > 1:
            raise ErreurUsage(f"sortie interdite avec plusieurs liens physiques : {sortie}")


def _json(nom):
    with open(nom, encoding="utf-8") as fichier:
        return json.load(fichier)


def _ecrire_json(nom, rapport):
    with open(nom, "w", encoding="utf-8") as fichier:
        json.dump(rapport, fichier, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        fichier.write("\n")


def _sha256(nom):
    hachage = hashlib.sha256()
    octets = 0
    with open(nom, "rb") as fichier:
        while True:
            bloc = fichier.read(BLOC)
            if not bloc:
                break
            hachage.update(bloc)
            octets += len(bloc)
    return hachage.hexdigest(), octets


def _ensembles(nom):
    ensembles = []
    with open(nom, encoding="utf-8") as fichier:
        for numero, ligne in enumerate(fichier, 1):
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            champs = ligne.split("|", 2)
            if len(champs) < 2 or not champs[0].strip():
                raise ErreurUsage(f"{nom}:{numero} : nom|chemin1;chemin2|note attendu")
            chemins = [p.strip() for p in champs[1].split(";")]
            if not all(chemins):
                raise ErreurUsage(f"{nom}:{numero} : chemin vide")
            ensembles.append({"nom": champs[0].strip(), "chemins": chemins,
                              "note": champs[2].strip() if len(champs) == 3 else ""})
    return ensembles


def _indices_reunion(chemins):
    """Ne compter que les racines les plus hautes, une fois par chemin réel."""
    reels = [os.path.realpath(p) for p in chemins]
    return {i for i, chemin in enumerate(reels)
            if not any(j != i and _sous(chemin, autre)
                       and (chemin != autre or j < i)
                       for j, autre in enumerate(reels))}


def _executer(arguments):
    resultat = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if resultat.returncode != 0:
        raise ValueError(
            f"{arguments[0]} : code {resultat.returncode} ; stderr : {resultat.stderr.strip()}"
        )
    return resultat.stdout


def _entier(objet, cle):
    valeur = objet[cle]
    if type(valeur) is not int or valeur < 0:
        raise ValueError(f"{cle} : entier positif ou nul attendu")
    return valeur


def _ecart(octets, du_sb):
    if du_sb is None or (du_sb == 0 and octets != 0):
        return None
    return abs(octets - du_sb) / du_sb if du_sb else 0.0


def _pourcentage(ecart):
    return "indisponible" if ecart is None else f"{100 * ecart:.4f} %"


def _go(octets):
    return round(octets / 2**30, 2)


def _regenerable(classes):
    presentes = [c for c in CLASSES if classes[c]["octets"] or classes[c]["n"]]
    if presentes == ["A"]:
        return "oui, bit-identique après rejeu de contrôle (REGEN-2)"
    if presentes == ["B"]:
        return "statistiquement (nouvelle réalisation)"
    if presentes in (["C"], []):
        return "non : à garder"
    return ", ".join(f"{c} : {_go(classes[c]['octets']):.2f} Go" for c in CLASSES)


def _fusionner(ensemble, rapport, raisons):
    """Ajouter un rapport validé sans recompter les octets hors chaînes."""
    ensemble["fichiers"] += _entier(rapport, "total_fichiers")
    ensemble["octets"] += _entier(rapport, "total_octets")
    for classe in CLASSES:
        valeurs = rapport["classes"][classe]
        ensemble["chaines"][classe]["n"] += _entier(valeurs, "chaines")
        ensemble["chaines"][classe]["octets"] += _entier(valeurs, "octets")
    for cle in ("fichiers", "octets"):
        ensemble["hors_chaines"][cle] += _entier(rapport["hors_chaines"], cle)
    for chaine in rapport["chaines"]:
        classe = chaine["classe"]
        if classe not in CLASSES or not isinstance(chaine["raisons"], list):
            raise ValueError("classe ou raisons de chaîne invalides")
        source = chaine["source"]
        source = source if source in ("su2_hip/1", "su2_gen/1") else "(non reconnu)"
        ensemble["sources"][source] += 1
        if any(not isinstance(r, str) for r in chaine["raisons"]):
            raise ValueError("raison de chaîne non textuelle")
        # Une raison répétée dans une chaîne ne concerne toujours qu'une chaîne.
        for raison in set(chaine["raisons"]):
            raisons[classe][raison] = raisons[classe].get(raison, 0) + 1


def auditer(options):
    ensembles = _ensembles(options.ensembles)
    racines = [p for e in ensembles for p in e["chemins"]]
    _verifier_sorties((options.sortie_json, options.sortie_md), racines)
    binaire = options.kvu
    if os.path.dirname(binaire):
        binaire = os.path.abspath(binaire)
    else:
        # Résoudre le PATH pour hacher exactement le binaire qui sera lancé.
        binaire = next((os.path.abspath(os.path.join(p, binaire))
                        for p in os.get_exec_path()
                        if os.path.isfile(os.path.join(p, binaire))
                        and os.access(os.path.join(p, binaire), os.X_OK)), None)
        if binaire is None:
            raise ErreurUsage(f"binaire introuvable : {options.kvu}")
    sha, _ = _sha256(binaire)
    erreurs = []
    preuves = set()
    for ensemble in ensembles:
        ensemble.update({
            "fichiers": 0, "octets": 0, "du_sb": 0,
            "chaines": {c: {"n": 0, "octets": 0} for c in CLASSES},
            "hors_chaines": {"fichiers": 0, "octets": 0},
            "sources": {"su2_hip/1": 0, "su2_gen/1": 0, "(non reconnu)": 0},
        })
        raisons = {c: {} for c in CLASSES}
        retenus = _indices_reunion(ensemble["chemins"])
        for indice, chemin in enumerate(ensemble["chemins"]):
            commande = [binaire, "store", "classer", "--chemin", chemin, "--json"]
            if options.preuve_gbit:
                commande.extend(("--preuve-gbit", options.preuve_gbit))
            try:
                rapport = json.loads(_executer(commande))
                if rapport["simulation"] is not True or not rapport["ensembles"]:
                    raise ValueError("rapport de simulation vide ou invalide")
                if (not isinstance(rapport["gbit_prouvees"], list)
                        or any(not isinstance(p, str) for p in rapport["gbit_prouvees"])):
                    raise ValueError("liste des preuves G-BIT invalide")
                preuves.update(rapport["gbit_prouvees"])
                if indice in retenus:
                    for partie in rapport["ensembles"]:
                        _fusionner(ensemble, partie, raisons)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                erreurs.append(f"{ensemble['nom']} ({chemin}) : classement impossible : {exc}")
            # Cette mesure reste indépendante, même si le classement échoue.
            try:
                mesure = int(_executer(["du", "-sb", "--", chemin]).split()[0])
                if mesure < 0:
                    raise ValueError("mesure négative")
                if indice in retenus and ensemble["du_sb"] is not None:
                    ensemble["du_sb"] += mesure
            except (OSError, ValueError, IndexError) as exc:
                ensemble["du_sb"] = None
                erreurs.append(f"{ensemble['nom']} ({chemin}) : mesure du impossible : {exc}")
        ensemble["raisons"] = {
            c: [[texte, n] for texte, n in sorted(raisons[c].items(), key=lambda r: (-r[1], r[0]))]
            for c in CLASSES
        }
        # max conserve la première classe en cas d'égalité : C, puis B, puis A.
        ensemble["classe_dominante"] = max(
            ("C", "B", "A"), key=lambda c: ensemble["chaines"][c]["octets"]
        )
        ensemble["go"] = _go(ensemble["octets"])
        ensemble["ecart_relatif"] = _ecart(ensemble["octets"], ensemble["du_sb"])
        ensemble["regenerable"] = _regenerable(ensemble["chaines"])
    total = sum(e["octets"] for e in ensembles)
    total_du = (sum(e["du_sb"] for e in ensembles)
                if all(e["du_sb"] is not None for e in ensembles) else None)
    ecart_total = _ecart(total, total_du)
    details = [f"{len(ensembles)}/7 ensembles", f"écart total {_pourcentage(ecart_total)}"]
    conforme = len(ensembles) == 7 and ecart_total is not None and ecart_total <= 0.01
    for ensemble in ensembles:
        ecart = ensemble["ecart_relatif"]
        details.append(f"{ensemble['nom']} : {_pourcentage(ecart)}")
        conforme = conforme and ecart is not None and ecart <= 0.01
        print(f"INFO {ensemble['nom']} classe={ensemble['classe_dominante']} "
              f"go={ensemble['go']:.2f} ecart={_pourcentage(ecart)}")
    conforme = conforme and not erreurs
    details.extend(erreurs)
    verdict = "PASS" if conforme else "FAIL"
    detail = " ; ".join(details)
    rapport = {
        "schema": "kvu-audit-ensembles/1", "date": _date(), "binaire_kvu": binaire,
        "binaire_sha256": sha, "preuves_gbit": sorted(preuves), "ensembles": ensembles,
        "total_octets": total, "total_du_sb": total_du, "ecart_total_relatif": ecart_total,
        "total_go": _go(total),
        "go_par_classe": {c: _go(sum(e["chaines"][c]["octets"] for e in ensembles)) for c in CLASSES},
        "verdict": {"AUDIT-7": verdict, "detail": detail},
    }
    _verifier_sorties((options.sortie_json, options.sortie_md), racines)
    _ecrire_json(options.sortie_json, rapport)
    with open(options.sortie_md, "w", encoding="utf-8") as fichier:
        fichier.write(_markdown(rapport))
    print(f"{verdict} AUDIT-7 {detail}")
    return 0 if conforme else 8


def _texte_md(texte):
    return texte.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _markdown(rapport):
    preuves = ", ".join(rapport["preuves_gbit"]) or "aucune"
    lignes = ["# Audit des ensembles WP4", "", f"Date UTC : {rapport['date']}", "",
              f"Binaire : {_texte_md(rapport['binaire_kvu'])} "
              f"(SHA-256 : {rapport['binaire_sha256'][:12]})", "",
              f"Preuves G-BIT prises en compte : {_texte_md(preuves)}", "",
              "| ensemble | classe | Go mesurés | chaînes A/B/C | régénérable | raison |",
              "| --- | --- | ---: | --- | --- | --- |"]
    for ensemble in rapport["ensembles"]:
        raisons = ensemble["raisons"][ensemble["classe_dominante"]]
        raison = raisons[0][0] if raisons else "aucun manifeste de chaîne : zone hors chaînes, non régénérable (classe C)"
        raison = raison if len(raison) <= 160 else raison[:159] + "…"
        if ensemble["note"]:
            raison += " ; " + ensemble["note"]
        nombres = "/".join(str(ensemble["chaines"][c]["n"]) for c in CLASSES)
        cellules = (ensemble["nom"], ensemble["classe_dominante"], f"{ensemble['go']:.2f}",
                    nombres, ensemble["regenerable"], raison)
        lignes.append("| " + " | ".join(_texte_md(c) for c in cellules) + " |")
    repartition = ", ".join(f"{c} : {rapport['go_par_classe'][c]:.2f} Go" for c in CLASSES)
    lignes.extend(["", f"Total : {rapport['total_go']:.2f} Go ; {repartition}.", "",
                   f"{rapport['verdict']['AUDIT-7']} AUDIT-7 : {_texte_md(rapport['verdict']['detail'])}"])
    for ensemble in rapport["ensembles"]:
        lignes.extend(["", f"## {_texte_md(ensemble['nom'])}", "",
                       f"Chemins : {_texte_md('; '.join(ensemble['chemins']))}", "",
                       f"Mesure du -sb : {ensemble['du_sb']} octets ; "
                       f"écart : {_pourcentage(ensemble['ecart_relatif'])}."])
        for classe in CLASSES:
            lignes.extend(["", f"### Classe {classe}", ""])
            lignes.extend(f"- {n} chaîne(s) : {_texte_md(texte)}"
                          for texte, n in ensemble["raisons"][classe])
            if not ensemble["raisons"][classe]:
                lignes.append("Aucune raison de chaîne enregistrée.")
    return "\n".join(lignes) + "\n"


def _priorite_minimale():
    try:
        os.nice(19)
    except OSError as exc:
        print(f"Priorité minimale indisponible : {exc}", file=os.sys.stderr)
    try:
        _executer(["ionice", "-c3", "-p", str(os.getpid())])
    except (OSError, ValueError) as exc:
        print(f"Priorité d'entrée-sortie minimale indisponible (ionice) : {exc}", file=os.sys.stderr)


def _erreur_parcours(erreur):
    raise erreur


def _trouver_chaines(racine):
    chaines = []
    for dossier, repertoires, fichiers in os.walk(racine, onerror=_erreur_parcours):
        repertoires.sort()
        if "manifest.json" in fichiers:
            chaines.append(dossier)
            repertoires[:] = []
    return chaines


def _verifier_fichier(racine, nom, attendu):
    resultat = {"nom": nom, "sha256_attendu": attendu, "sha256": None,
                "statut": "DIFFERENT", "octets_lus": 0}
    chemin = os.path.join(racine, nom)
    if os.path.isabs(nom) or not _sous(os.path.realpath(chemin), os.path.realpath(racine)):
        resultat["erreur"] = "chemin de référence hors de la chaîne"
        return resultat
    try:
        # Compter aussi les octets déjà lus si une lecture échoue ensuite.
        hachage = hashlib.sha256()
        with open(chemin, "rb") as fichier:
            while True:
                bloc = fichier.read(BLOC)
                if not bloc:
                    break
                hachage.update(bloc)
                resultat["octets_lus"] += len(bloc)
        resultat["sha256"] = hachage.hexdigest()
        resultat["statut"] = "OK" if resultat["sha256"] == attendu.lower() else "DIFFERENT"
    except FileNotFoundError:
        resultat["statut"] = "ABSENT"
    except OSError as exc:
        resultat["erreur"] = f"lecture impossible : {exc}"
    return resultat


def _integrite_chaine(racine, executeur):
    resultat = {"chemin": racine, "fichiers": [], "non_listes": [], "erreurs": []}
    references = {}
    try:
        manifeste = _json(os.path.join(racine, "manifest.json"))
        if not isinstance(manifeste, dict):
            raise ValueError("le manifeste doit être un objet")
        for table in ("sorties_sha256", "csv_sha256"):
            valeurs = manifeste.get(table, {})
            if not isinstance(valeurs, dict):
                resultat["erreurs"].append(f"{table} : table nom → sha256 attendue")
                continue
            for nom, sha in valeurs.items():
                if (not nom or not isinstance(sha, str) or len(sha) != 64
                        or any(c not in "0123456789abcdefABCDEF" for c in sha)):
                    resultat["erreurs"].append(f"{table}[{nom}] : référence sha256 invalide")
                    continue
                if nom in references and references[nom].lower() != sha.lower():
                    resultat["erreurs"].append(f"{nom} : références sha256 contradictoires")
                references[nom] = sha
        if not references:
            resultat["erreurs"].append("aucune référence de hachage")
    except (OSError, ValueError) as exc:
        resultat["erreurs"].append(f"manifeste illisible : {exc}")
    taches = [executeur.submit(_verifier_fichier, racine, nom, sha)
              for nom, sha in sorted(references.items())]
    resultat["fichiers"] = [tache.result() for tache in taches]
    listes = {os.path.normpath(nom) for nom in references}
    try:
        for dossier, repertoires, fichiers in os.walk(racine, onerror=_erreur_parcours):
            repertoires.sort()
            for fichier in sorted(fichiers):
                relatif = os.path.relpath(os.path.join(dossier, fichier), racine)
                if relatif not in listes and relatif not in ("manifest.json", "COMPLETE"):
                    resultat["non_listes"].append(relatif)
    except OSError as exc:
        resultat["erreurs"].append(f"parcours incomplet : {exc}")
    return resultat


def integrite(options):
    _priorite_minimale()
    debut = time.monotonic()
    racine = os.path.abspath(options.integrite)
    _verifier_sorties((options.sortie_json,), (racine,))
    if not os.path.isdir(racine):
        raise ErreurUsage(f"ensemble introuvable : {racine}")
    ecarts = []
    try:
        chemins = _trouver_chaines(racine)
    except OSError as exc:
        chemins = []
        ecarts.append(f"parcours impossible : {exc}")
    chaines = []
    n_ok = n_total = octets = 0
    if not chemins and not ecarts:
        ecarts.append("aucune chaîne avec manifest.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.fils) as executeur:
        for i, chemin in enumerate(chemins, 1):
            chaine = _integrite_chaine(chemin, executeur)
            chaines.append(chaine)
            for fichier in chaine["fichiers"]:
                n_total += 1
                octets += fichier["octets_lus"]
                if fichier["statut"] == "OK":
                    n_ok += 1
                else:
                    ecarts.append(f"{chemin}/{fichier['nom']} : {fichier['statut']}"
                                  + (f" ; {fichier['erreur']}" if "erreur" in fichier else ""))
            ecarts.extend(f"{chemin} : {erreur}" for erreur in chaine["erreurs"])
            for nom in chaine["non_listes"]:
                print(f"INFO {chemin}/{nom} : fichier non listé", file=os.sys.stderr)
            if i % 10 == 0 or i == len(chemins):
                duree = max(time.monotonic() - debut, 1e-9)
                print(f"chaîne {i}/{len(chemins)}, {octets} octets lus, "
                      f"débit {octets / duree / 10**6:.2f} Mo/s", file=os.sys.stderr)
    duree = time.monotonic() - debut
    conforme = not ecarts and n_ok == n_total
    verdict = "PASS" if conforme else "FAIL"
    if options.sortie_json:
        _verifier_sorties((options.sortie_json,), (racine,))
        _ecrire_json(options.sortie_json, {
            "schema": "kvu-integrite-ensembles/1", "date": _date(), "ensemble": racine,
            "chaines": chaines, "totaux": {"n_ok": n_ok, "n_total": n_total,
                "n_chaines": len(chaines), "octets_lus": octets, "duree_s": duree},
            "verdict": {"REGEN-INTEG": verdict, "ecarts": ecarts},
        })
    for ecart in ecarts[:20]:
        print(f"FAIL {ecart}")
    print(f"{verdict} REGEN-INTEG {n_ok}/{n_total} fichiers de {len(chaines)} chaînes, "
          f"{octets} octets lus en {duree:.2f} s")
    return 0 if conforme else 8


def main(arguments=None):
    """Renvoyer 0 pour PASS, 8 pour FAIL, 2 pour une erreur d'usage."""
    try:
        options = _arguments(arguments)
        return integrite(options) if options.integrite is not None else auditer(options)
    except (ErreurUsage, OSError, UnicodeError) as exc:
        print(f"Erreur d'usage : {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
