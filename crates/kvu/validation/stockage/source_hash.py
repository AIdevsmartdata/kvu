#!/usr/bin/env python3
"""Reproduit en Python la recette `source_hash` du kevinotron (K8 et Unifiee).

    sha256(json.dumps({chemin_relatif: sha256(fichier)}, sort_keys=True))

Recettes (identiques à src/manifest.rs, `fichiers_recette`) :
  build_sh  src/**/*.rs + Cargo.toml, Cargo.lock, {e6,e7,f4}_generators.bin
  complete  src/** (tous fichiers réguliers) + Cargo.toml, Cargo.lock, build.rs, {e6,e7,f4}_generators.bin
  unifiee   complete + crates/kvu/src/** + crates/kvu/noyaux/** + crates/kvu/Cargo.toml + crates/kvu/build.rs

Usage :  source_hash.py [--recette unifiee] [--racine DIR] [--liste] [--json]
Sortie : « <sha256> <nombre de fichiers> » (ou la liste des fichiers, ou un objet JSON).
Aucune dépendance hors bibliothèque standard.
"""
import argparse
import hashlib
import json
import os
import sys

RACINE_BUILD_SH = ["Cargo.toml", "Cargo.lock", "e6_generators.bin", "e7_generators.bin", "f4_generators.bin"]
RACINE_COMPLETE = ["Cargo.toml", "Cargo.lock", "build.rs", "e6_generators.bin", "e7_generators.bin", "f4_generators.bin"]
RACINE_UNIFIEE = RACINE_COMPLETE + ["crates/kvu/Cargo.toml", "crates/kvu/build.rs"]
DOSSIERS_UNIFIEE = ["crates/kvu/src", "crates/kvu/noyaux"]


def _lister(racine, rel, seulement_rs, sortie):
    """Tous les fichiers réguliers sous racine/rel (récursif) ; un lien vers un fichier est haché par son contenu."""
    base = os.path.join(racine, rel)
    for nom in sorted(os.listdir(base)):
        r = rel + "/" + nom
        chemin = os.path.join(racine, r)
        if os.path.isdir(chemin) and not os.path.islink(chemin):
            _lister(racine, r, seulement_rs, sortie)
        elif os.path.isfile(chemin):  # fichier régulier ou lien vers un fichier (comme manifest.rs)
            if not seulement_rs or nom.endswith(".rs"):
                sortie.append(r)


def fichiers_recette(racine, recette):
    v = []
    _lister(racine, "src", recette == "build_sh", v)
    if recette == "unifiee":
        for d in DOSSIERS_UNIFIEE:
            _lister(racine, d, False, v)
    v += {"build_sh": RACINE_BUILD_SH, "complete": RACINE_COMPLETE, "unifiee": RACINE_UNIFIEE}[recette]
    return sorted(v)


def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def hash_arbre(racine, recette):
    fichiers = fichiers_recette(racine, recette)
    blob = json.dumps({f: sha256_fichier(os.path.join(racine, f)) for f in fichiers}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), fichiers


def main():
    ici = os.path.dirname(os.path.abspath(__file__))
    defaut = os.path.abspath(os.path.join(ici, "..", "..", "..", ".."))  # crates/kvu/validation/stockage -> racine
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recette", choices=["build_sh", "complete", "unifiee"], default="unifiee")
    ap.add_argument("--racine", default=defaut)
    ap.add_argument("--liste", action="store_true", help="imprime la liste triée des fichiers couverts")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    h, fichiers = hash_arbre(os.path.abspath(a.racine), a.recette)
    if a.liste:
        print("\n".join(fichiers))
    elif a.json:
        print(json.dumps({"recette": a.recette, "source_hash": h, "nfichiers": len(fichiers)}))
    else:
        print(h, len(fichiers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
