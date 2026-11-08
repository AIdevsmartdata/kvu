"""GATES v1 : contrôle WP2-EMISES des identifiants et de la couverture SKIP.

La comparaison porte sur les ensembles de noms, indépendamment des compteurs
globaux. Les identifiants supplémentaires émis sont seulement informatifs.
Ce module ne dépend que de la bibliothèque standard, y compris sous python3 -I.
"""

import argparse
import json
import sys


EXTRAS_PAR_DEFAUT = ("MACH-1", "VERROU-1", "FILE-1")
STATUTS = frozenset(("PASS", "FAIL", "SKIP", "INFO", "FAIL-DOCUMENTE"))


def comparer(ref_gates, emises_gates, extra, equiv):
    """Retourne les trois listes triées et le verdict, sans modifier les entrées.

    Les gates sont des listes de dictionnaires. ``extra`` est un itérable de
    noms (ou None pour les défauts). ``equiv`` est un dictionnaire ancien/nouveau
    ou un itérable de couples, permettant plusieurs remplacements du même nom.
    Les équivalences sont directes, sans résolution transitive.
    """
    attendus = {gate["nom"] for gate in ref_gates}
    attendus.update(nom for nom in (
        EXTRAS_PAR_DEFAUT if extra is None else extra
    ) if nom)
    emis = {gate["nom"] for gate in emises_gates}
    equivalences = list(equiv.items() if hasattr(equiv, "items") else equiv)

    presents = set(emis)
    presents.update(ancien for ancien, nouveau in equivalences if nouveau in emis)
    autorises = attendus | {nouveau for _, nouveau in equivalences}

    # Chaque machine portant un SKIP doit avoir un PASS ailleurs pour ce nom.
    machines_pass = {}
    machines_skip = {}
    for gate in emises_gates:
        if gate["statut"] == "PASS":
            machines_pass.setdefault(gate["nom"], set()).add(gate["machine"])
        elif gate["statut"] == "SKIP":
            machines_skip.setdefault(gate["nom"], set()).add(gate["machine"])
    skips_non_couverts = sorted(
        nom for nom, machines in machines_skip.items()
        if any(not (machines_pass.get(nom, set()) - {machine})
               for machine in machines)
    )
    manquants = sorted(attendus - presents)
    return {
        "manquants": manquants,
        "en_trop": sorted(emis - autorises),
        "skips_non_couverts": skips_non_couverts,
        "ok": not manquants and not skips_non_couverts,
    }


def _lire_gates(chemin):
    """Charge et valide les lignes utiles du document GATES v1."""
    with open(chemin, encoding="utf-8") as fichier:
        document = json.load(fichier)
    if not isinstance(document, dict) or "gates" not in document:
        raise ValueError('clé "gates" absente')
    gates = document["gates"]
    if not isinstance(gates, list):
        raise ValueError('la clé "gates" doit contenir une liste')
    for numero, gate in enumerate(gates, start=1):
        if not isinstance(gate, dict) or any(
            not isinstance(gate.get(cle), str)
            for cle in ("nom", "machine", "statut", "detail")
        ):
            raise ValueError("gate {} : champs texte requis invalides".format(numero))
        if gate["statut"] not in STATUTS:
            raise ValueError("gate {} : statut inconnu".format(numero))
    return gates


def _equivalence(texte):
    """Valide une option ANCIEN=NOUVEAU avant tout accès aux fichiers."""
    ancien, separateur, nouveau = texte.partition("=")
    if not separateur or not ancien or not nouveau:
        raise argparse.ArgumentTypeError("équivalence attendue : ANCIEN=NOUVEAU")
    return ancien, nouveau


def main(argv):
    """Exécute la commande et retourne 0, 8 ou 2 ; argv exclut le nom du script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", metavar="REFERENCE.json")
    parser.add_argument("emises", metavar="EMISES.json")
    parser.add_argument("--extra", action="append", metavar="ID",
                        help="remplace les extras par défaut ; répétable")
    parser.add_argument("--equiv", action="append", type=_equivalence, default=[],
                        metavar="ANCIEN=NOUVEAU")
    parser.add_argument("--json", action="store_true", dest="sortie_json")
    # argparse écrit lui-même les erreurs d'usage sur stderr. main reste appelable.
    try:
        args = parser.parse_args(argv)
    except SystemExit as erreur:
        return erreur.code

    gates = []
    for chemin in (args.reference, args.emises):
        try:
            gates.append(_lire_gates(chemin))
        except (OSError, ValueError, UnicodeError) as erreur:
            print("Erreur : {} : {}".format(chemin, erreur), file=sys.stderr)
            return 2
    resultat = comparer(gates[0], gates[1], args.extra, args.equiv)
    if args.sortie_json:
        print(json.dumps(resultat, ensure_ascii=False))
    else:
        for cle, titre in (("manquants", "manquants"),
                           ("en_trop", "en trop"),
                           ("skips_non_couverts", "skips non couverts")):
            print("{} ({}) :".format(titre, len(resultat[cle])))
            for nom in resultat[cle]:
                print("  " + nom)
    return 0 if resultat["ok"] else 8


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
