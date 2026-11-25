# SPDX-License-Identifier: Apache-2.0
"""Valider les manifestes de régénération sans dépendance ni écriture."""

import copy
import json
from pathlib import Path
import re
import sys


class ErreurUsage(ValueError):
    """Signaler un schéma ou des arguments non pris en charge."""


_MOTS_CLES = {
    "$schema", "$id", "title", "description", "type", "required",
    "properties", "additionalProperties", "items", "enum", "const",
    "minimum", "minItems", "minLength", "format", "if", "then",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_SCHEMA_DEFAUT = Path(__file__).resolve().parents[1] / "schemas/kvu-regen-1.schema.json"


def _chemin(elements):
    """Encoder un chemin selon les échappements des pointeurs JSON."""
    return "/" + "/".join(str(e).replace("~", "~0").replace("/", "~1") for e in elements)


def _nombre(valeur):
    return type(valeur) in (int, float)


def _verifier_schema(schema, chemin=()):
    """Contrôler tout le schéma, même ses branches non appliquées."""
    def erreur(raison):
        raise ErreurUsage(f"{_chemin(chemin)} : {raison}")

    if not isinstance(schema, dict):
        erreur("un schéma doit être un objet")
    for mot in schema:
        if mot not in _MOTS_CLES:
            erreur(f"mot-clé inconnu : {mot}")
    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        if not types or any(not isinstance(t, str) or t not in _TYPES for t in types):
            erreur("déclaration de type invalide")
    if "required" in schema:
        requis = schema["required"]
        if (not isinstance(requis, list)
                or any(not isinstance(nom, str) for nom in requis)
                or len(set(requis)) != len(requis)):
            erreur("required doit être une liste de noms distincts")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        erreur("additionalProperties doit être un booléen")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        erreur("enum doit être une liste non vide")
    if "minimum" in schema and not _nombre(schema["minimum"]):
        erreur("minimum doit être un nombre")
    for mot in ("minItems", "minLength"):
        if mot in schema and (type(schema[mot]) is not int or schema[mot] < 0):
            erreur(f"{mot} doit être un entier positif ou nul")
    if "format" in schema and schema["format"] != "sha256":
        erreur("seul le format sha256 est pris en charge")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            erreur("properties doit être un objet")
        for nom, enfant in schema["properties"].items():
            if not isinstance(nom, str):
                erreur("un nom de propriété doit être une chaîne")
            _verifier_schema(enfant, chemin + ("properties", nom))
    for mot in ("items", "if", "then"):
        if mot in schema:
            _verifier_schema(schema[mot], chemin + (mot,))


def _du_type(valeur, nom):
    return {
        "object": isinstance(valeur, dict),
        "array": isinstance(valeur, list),
        "string": isinstance(valeur, str),
        "integer": type(valeur) is int,
        "number": _nombre(valeur),
        "boolean": type(valeur) is bool,
        "null": valeur is None,
    }[nom]


def _egaux(gauche, droite):
    """Comparer des valeurs JSON sans assimiler les booléens aux nombres."""
    if _nombre(gauche) and _nombre(droite):
        return gauche == droite
    if type(gauche) is not type(droite):
        return False
    if isinstance(gauche, dict):
        return gauche.keys() == droite.keys() and all(
            _egaux(gauche[cle], droite[cle]) for cle in gauche
        )
    if isinstance(gauche, list):
        return len(gauche) == len(droite) and all(
            _egaux(g, d) for g, d in zip(gauche, droite)
        )
    return gauche == droite


def _valider(schema, instance, chemin):
    erreurs = []

    def erreur(raison, position=chemin):
        erreurs.append(f"{_chemin(position)} : {raison}")

    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        if not any(_du_type(instance, nom) for nom in types):
            erreur("type incompatible avec le schéma")
    if "enum" in schema and not any(_egaux(instance, v) for v in schema["enum"]):
        erreur("valeur absente des valeurs autorisées")
    if "const" in schema and not _egaux(instance, schema["const"]):
        erreur("valeur différente de la constante attendue")
    if isinstance(instance, dict):
        for nom in schema.get("required", []):
            if nom not in instance:
                erreur("champ obligatoire absent", chemin + (nom,))
        proprietes = schema.get("properties", {})
        for nom, valeur in instance.items():
            if nom in proprietes:
                erreurs.extend(_valider(proprietes[nom], valeur, chemin + (nom,)))
            elif schema.get("additionalProperties", True) is False:
                erreur("propriété supplémentaire interdite", chemin + (nom,))
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            erreur("nombre d'éléments inférieur au minimum")
        if "items" in schema:
            for indice, valeur in enumerate(instance):
                erreurs.extend(_valider(schema["items"], valeur, chemin + (indice,)))
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            erreur("longueur inférieure au minimum")
        if schema.get("format") == "sha256" and re.fullmatch(r"[0-9a-f]{64}", instance) is None:
            erreur("sha256 attendu : exactement 64 caractères hexadécimaux minuscules")
    if _nombre(instance) and "minimum" in schema and instance < schema["minimum"]:
        erreur(f"nombre inférieur au minimum {schema['minimum']}")
    if "if" in schema and not _valider(schema["if"], instance, chemin):
        if "then" in schema:
            erreurs.extend(_valider(schema["then"], instance, chemin))
    return erreurs


def valider(schema: dict, instance) -> list[str]:
    """Renvoyer les erreurs du manifeste, ou lever ErreurUsage pour le schéma."""
    _verifier_schema(schema)
    return _valider(schema, instance, ())


def mutants_champ_retire(schema: dict, instance: dict) -> list[tuple[str, dict]]:
    """Retirer chaque champ requis présent, sans parcourir les conditions."""
    _verifier_schema(schema)
    mutants = []

    def parcourir(sous_schema, valeur, chemin):
        if isinstance(valeur, dict):
            for nom in sous_schema.get("required", []):
                if nom in valeur:
                    copie = copy.deepcopy(instance)
                    parent = copie
                    for element in chemin:
                        parent = parent[element]
                    del parent[nom]
                    mutants.append((_chemin(chemin + (nom,)), copie))
            for nom, enfant in sous_schema.get("properties", {}).items():
                if nom in valeur:
                    parcourir(enfant, valeur[nom], chemin + (nom,))
        if isinstance(valeur, list) and valeur and "items" in sous_schema:
            parcourir(sous_schema["items"], valeur[0], chemin + (0,))

    parcourir(schema, instance, ())
    return mutants


def _constante_interdite(valeur):
    raise ValueError(f"constante JSON interdite : {valeur}")


def _lire_json(nom):
    try:
        with open(nom, encoding="utf-8") as fichier:
            return json.load(fichier, parse_constant=_constante_interdite)
    except (OSError, ValueError) as exc:
        raise ErreurUsage(f"{nom} : fichier illisible ou JSON malformé") from exc


def _arguments(arguments):
    """Lire les options en conservant tous les diagnostics en français."""
    schema = _SCHEMA_DEFAUT
    mutant = None
    fichiers = []
    indice = 0
    while indice < len(arguments):
        argument = arguments[indice]
        if argument == "--":
            fichiers.extend(arguments[indice + 1:])
            break
        if argument in ("-h", "--help"):
            print("Utilisation : regen_valider.py [--schema S] M.json [M2.json…]")
            print("              regen_valider.py [--schema S] --mutants M.json")
            print("Valider des manifestes ou vérifier le rejet des champs retirés.")
            return None
        if argument in ("--schema", "--mutants"):
            indice += 1
            if indice == len(arguments) or arguments[indice].startswith("--"):
                raise ErreurUsage(f"{argument} : chemin de fichier attendu")
            if argument == "--schema":
                schema = arguments[indice]
            else:
                if mutant is not None:
                    raise ErreurUsage("l'option --mutants ne peut apparaître qu'une fois")
                mutant = arguments[indice]
        elif argument.startswith("-"):
            raise ErreurUsage(f"option inconnue : {argument}")
        else:
            fichiers.append(argument)
        indice += 1
    if mutant is not None and fichiers:
        raise ErreurUsage("le mode --mutants accepte un seul manifeste")
    if mutant is None and not fichiers:
        raise ErreurUsage("au moins un manifeste est nécessaire")
    return schema, mutant, fichiers


def main(arguments=None):
    """Renvoyer 0 en cas de succès, 8 pour un rejet et 2 pour une erreur d'usage."""
    try:
        options = _arguments(sys.argv[1:] if arguments is None else arguments)
        if options is None:
            return 0
        nom_schema, nom_mutants, fichiers = options
        schema = _lire_json(nom_schema)
        _verifier_schema(schema)
        if nom_mutants is not None:
            instance = _lire_json(nom_mutants)
            erreurs = valider(schema, instance)
            if erreurs:
                for message in erreurs[:20]:
                    print(f"FAIL {nom_mutants} : {message}")
                print("FAIL REGEN-SCHEMA-MUTANTS 0/0")
                return 8
            mutants = mutants_champ_retire(schema, instance)
            rejetes = 0
            for chemin, mutant in mutants:
                if valider(schema, mutant):
                    rejetes += 1
                    print(f"PASS mutant {chemin}")
                else:
                    print(f"FAIL mutant {chemin} : accepté à tort")
            succes = rejetes == len(mutants)
            print(f"{'PASS' if succes else 'FAIL'} REGEN-SCHEMA-MUTANTS {rejetes}/{len(mutants)}")
            return 0 if succes else 8
        invalide = False
        for nom in fichiers:
            erreurs = valider(schema, _lire_json(nom))
            if erreurs:
                invalide = True
                for message in erreurs[:20]:
                    print(f"FAIL {nom} : {message}")
            else:
                print(f"PASS {nom}")
        return 8 if invalide else 0
    except ErreurUsage as exc:
        print(f"Erreur d'usage : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
