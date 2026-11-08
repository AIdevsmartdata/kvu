# SPDX-License-Identifier: Apache-2.0
"""Recalcul indépendant de kvu-source/1, avec les listes du dépôt examiné."""

import hashlib
import json
import os
import re
import sys


RECETTE = "kvu-source/1"
USAGE = "Utilisation : source_hash_kvu.py RACINE [--liste] [--json]"


def lire_listes(racine):
    chemin = os.path.join(racine, "crates/kvu/src/stockage/recette.rs")
    with open(chemin, encoding="utf-8") as fichier:
        texte = fichier.read()
    listes = []
    for nom in ("DOSSIERS_KVU_SOURCE_1", "FICHIERS_KVU_SOURCE_1"):
        declaration = re.search(
            r"\bpub\s+const\s+" + nom + r"\s*:\s*&\s*\[\s*&str\s*\]"
            r"\s*=\s*&\s*\[(.*?)\]\s*;",
            texte,
            flags=re.DOTALL,
        )
        if declaration is None:
            raise ValueError(f"constante {nom} absente dans {chemin}")
        # Les constantes attendues sont des tableaux littéraux de chaînes simples.
        contenu = re.sub(r",\s*$", "", declaration.group(1))
        try:
            valeurs = json.loads("[" + contenu + "]")
        except ValueError as erreur:
            raise ValueError(f"constante {nom} illisible dans {chemin}") from erreur
        if not valeurs or not all(isinstance(v, str) and v for v in valeurs):
            raise ValueError(f"constante {nom} invalide dans {chemin}")
        listes.append(valeurs)
    return listes


def propager_erreur(erreur):
    raise erreur


def calculer(racine):
    if not os.path.isdir(racine):
        raise ValueError(f"racine du dépôt invalide : {racine}")
    dossiers, obligatoires = lire_listes(racine)
    chemins = set()
    for relatif in dossiers:
        dossier = os.path.join(racine, relatif)
        if not os.path.isdir(dossier):
            raise ValueError(f"dossier requis absent ou invalide : {relatif}")
        # os.walk ne suit pas les liens vers des dossiers. isfile suit ceux
        # vers des fichiers et écarte les autres entrées (dont les liens cassés).
        for courant, _, noms in os.walk(dossier, followlinks=False, onerror=propager_erreur):
            for nom in noms:
                chemin = os.path.join(courant, nom)
                if os.path.isfile(chemin):
                    chemins.add(os.path.relpath(chemin, racine).replace(os.sep, "/"))
    for relatif in obligatoires:
        if not os.path.isfile(os.path.join(racine, relatif)):
            raise ValueError(f"fichier requis absent ou invalide : {relatif}")
        chemins.add(relatif)
    chemins = sorted(chemins)
    empreintes = {}
    for relatif in chemins:
        with open(os.path.join(racine, relatif), "rb") as fichier:
            empreintes[relatif] = hashlib.file_digest(fichier, "sha256").hexdigest()
    contenu = json.dumps(empreintes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(contenu).hexdigest(), chemins


def main(arguments):
    if arguments in (["--help"], ["-h"]):
        print(USAGE)
        print("--liste : chemins triés ; --json : recette, empreinte et nombre de fichiers.")
        return 0
    try:
        if not arguments or arguments[0].startswith("--"):
            raise ValueError(USAGE)
        options = set(arguments[1:])
        if options - {"--liste", "--json"}:
            raise ValueError("option inconnue ; " + USAGE)
        empreinte, chemins = calculer(os.path.abspath(arguments[0]))
        if "--json" in options:
            resultat = {
                "recette": RECETTE,
                "source_hash": empreinte,
                "nfichiers": len(chemins),
            }
            if "--liste" in options:
                resultat["fichiers"] = chemins
            print(json.dumps(resultat))
        elif "--liste" in options:
            print("\n".join(chemins))
        else:
            print(empreinte)
        print(f"{len(chemins)} fichiers", file=sys.stderr)
    except (OSError, ValueError, UnicodeError) as erreur:
        print(f"Erreur kvu-source/1 : {erreur}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
