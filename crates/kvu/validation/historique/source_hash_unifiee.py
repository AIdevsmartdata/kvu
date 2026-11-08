#!/usr/bin/env python3
"""Recalcule INDÉPENDAMMENT (Python, hashlib) le source_hash de la recette `Unifiee` (T8) :
sha256(json.dumps({chemin: sha256(fichier)}, sort_keys=True)) sur
  src/** (tous fichiers) + crates/kvu/src/** + crates/kvu/noyaux/** + les fichiers racine listés.
Les listes (FICHIERS_RACINE_UNIFIEE, DOSSIERS_KVU_UNIFIEE) sont LUES dans src/manifest.rs de l'arbre :
elles définissent la recette ; le hachage, le tri et la sérialisation sont refaits ici.

Usage : source_hash_unifiee.py RACINE_DU_DEPOT   ->  imprime l'empreinte (64 hex)
"""
import hashlib
import json
import os
import re
import sys


def liste_constante(texte, nom):
    m = re.search(nom + r'\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];', texte, re.S)
    if not m:
        raise SystemExit(f'constante {nom} introuvable dans src/manifest.rs')
    return re.findall(r'"([^"]+)"', m.group(1))


def fichiers_sous(racine, rel):
    out = []
    base = os.path.join(racine, rel)
    for dp, _, fn in os.walk(base):
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.isfile(p):
                out.append(os.path.relpath(p, racine).replace(os.sep, '/'))
    return out


def empreinte(racine):
    txt = open(os.path.join(racine, 'src', 'manifest.rs'), encoding='utf-8').read()
    racine_f = liste_constante(txt, 'FICHIERS_RACINE_UNIFIEE')
    dossiers = liste_constante(txt, 'DOSSIERS_KVU_UNIFIEE')
    fichiers = set(fichiers_sous(racine, 'src'))
    for d in dossiers:
        fichiers.update(fichiers_sous(racine, d))
    fichiers.update(racine_f)
    recs = {f: hashlib.sha256(open(os.path.join(racine, f), 'rb').read()).hexdigest() for f in fichiers}
    blob = json.dumps(recs, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), len(recs)


if __name__ == '__main__':
    h, n = empreinte(sys.argv[1])
    print(h)
    print(n, 'fichiers', file=sys.stderr)
