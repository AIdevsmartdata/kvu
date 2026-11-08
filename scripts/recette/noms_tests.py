#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""noms_tests.py : référence des tests de kvu TELS QUE CARGO LES COMPTE, et contrôle qu'aucun test n'a disparu.

Pourquoi. « N tests verts » ne dit rien si des tests ont été retirés, désactivés par un `#[cfg]` ou rendus `#[ignore]` en
silence. Ce script compare la suite que cargo liste (`cargo test -- --list` et `-- --list --ignored`) à une référence
versionnée (`scripts/recette/noms_tests/`), et vérifie que chaque test actif listé a bien été exécuté (sortie de
`cargo test`). La logique est celle de `inventaire_cargo.py` (outillage de recette interne, repris tel quel).

Usage :
  noms_tests.py reference <liste.out> <ignores.out> --commit <sha> --etape <nom> --source <texte>  > reference.txt
  noms_tests.py verifier  <reference.txt> <liste.out> <ignores.out> [<tests.out>]

`verifier` rend 0 si le candidat ne fait qu'ajouter ou réactiver des tests (et, avec `tests.out`, si chaque test actif a été
exécuté et que rien d'autre ne l'a été), 1 sinon, 2 sur une erreur d'usage ou une sortie de cargo incohérente.
Python 3.8 ou plus, bibliothèque standard seulement.
"""

import argparse
import importlib.util
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))


def _inventaire():
    """Module inventaire_cargo.py, chargé depuis le dossier de ce script (le mode isolé `python3 -I` ne l'y cherche pas)."""
    spec = importlib.util.spec_from_file_location("inventaire_cargo", os.path.join(ICI, "inventaire_cargo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ic = _inventaire()


def _lire(chemin):
    with open(chemin, encoding="utf-8", errors="replace") as f:
        return f.read()


def suite_listee(liste, ignores):
    """Suite {cible: {nom: actif|ignore}} d'après les deux sorties `--list`."""
    return ic.suite_de(ic.parser_liste(_lire(liste)), ic.parser_liste(_lire(ignores)))


def cmd_reference(args):
    suite = suite_listee(args.liste, args.ignores)
    sys.stdout.write(ic.ecrire_reference(suite, args.commit, args.etape, args.source))
    total, nb_actifs, nb_ignores, nb_cibles = ic.resume(suite)
    print(f"référence : {total} tests ({nb_actifs} actifs, {nb_ignores} ignorés) dans {nb_cibles} exécutables", file=sys.stderr)
    return 0


def cmd_verifier(args):
    base, meta = ic.lire_reference(_lire(args.reference))
    candidat = suite_listee(args.liste, args.ignores)
    retraits = ic.comparer(base, candidat)
    anomalies = []
    if args.execution:
        vus, _ = ic.parser_execution(_lire(args.execution))
        anomalies = ic.anomalies_execution(candidat, vus)
    total, nb_actifs, nb_ignores, nb_cibles = ic.resume(candidat)
    b_total, b_actifs, _, _ = ic.resume(base)
    print(f"base ({meta.get('commit', '?')[:12]}, {meta.get('etape', '?')}) : {b_total} tests dont {b_actifs} actifs ; "
          f"candidat : {total} tests dont {nb_actifs} actifs, {nb_ignores} ignorés, {nb_cibles} exécutables")
    if retraits:
        print(f"ÉCHEC : {len(retraits)} test(s) de la référence retiré(s) ou désactivé(s) : {ic.formater(retraits, 10)}")
    if anomalies:
        print(f"ÉCHEC : {len(anomalies)} incohérence(s) entre la liste et l'exécution : {ic.formater(anomalies, 10)}")
    if retraits or anomalies:
        return 1
    print("OK : aucun test de la référence retiré ni désactivé" + (" ; chaque test actif a été exécuté" if args.execution else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sous = p.add_subparsers(dest="commande", required=True)
    r = sous.add_parser("reference", help="écrit la référence canonique d'une suite listée par cargo")
    r.add_argument("liste")
    r.add_argument("ignores")
    r.add_argument("--commit", required=True)
    r.add_argument("--etape", required=True)
    r.add_argument("--source", required=True)
    v = sous.add_parser("verifier", help="compare une suite listée par cargo à une référence")
    v.add_argument("reference")
    v.add_argument("liste")
    v.add_argument("ignores")
    v.add_argument("execution", nargs="?")
    args = p.parse_args(argv)
    try:
        return cmd_reference(args) if args.commande == "reference" else cmd_verifier(args)
    except (ic.SortieIncoherente, OSError) as e:
        print(f"ERREUR : {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
