#!/usr/bin/env python3
"""Gate WP3-GREP : plus aucun nom de machine, chemin ou hote en dur dans le code de src/.

Commande de reference (GATES v1) : grep -rnE 'acemagic|gamer|10\\.42\\.0|/chemin/vers/projet|remondiere' crates/kvu/src
Regle de comptage (GATES v1 : "les tests et kvu.toml sont exclus du comptage") appliquee ici de facon mecanique :
 - les modules de tests en ligne sont exclus : tout ce qui suit la ligne `#[cfg(test)]` qui introduit le module `mod tests`
   (verifie : ce module est le dernier element du fichier, rien d'autre ne le suit a la colonne 0) ;
 - exclusions explicites, listees dans EXCLUSIONS (rapportees, jamais silencieuses) :
     src/heritage/**                        copies gelees a l'octet par WP1-AUTONOME (commentaires seulement) ;
     src/**/compat_tests_existants.rs       constantes importees par les tests existants, qui ne doivent pas changer.
Usage : python3 -I wp3_grep.py <racine du depot>   -> imprime PASS|FAIL WP3-GREP ... ; rc 0 si PASS, 1 sinon.

Casse (reserve 2 de T0-VERIF, lot t1-prelude) : le motif gele de GATES v1 est SENSIBLE a la casse, et la gate reste
calculee avec lui (aucun seuil change). La meme recherche INSENSIBLE a la casse, avec les memes exclusions, est
imprimee en plus sur une ligne `INFO WP3-GREP-CASSE code=N ...` (jamais un verdict) : elle montre l'identite des
profils qui reste cablee dans le type `Machine` (variantes `Acemagic`, `Gamer`, noms de profil fabriques a la
compilation, `pub use` de compatibilite). Voir docs/MACHINES.md, section « WP3-GREP et la casse ».
"""
import os
import re
import sys

MOTIF = re.compile(r"acemagic|gamer|10\.42\.0|/chemin/vers/projet|remondiere")
MOTIF_CASSE = re.compile(MOTIF.pattern, re.IGNORECASE)
EXCLUSIONS = ("src/heritage/", "compat_tests_existants.rs")


def partie_code(lignes):
    """Rend (lignes de code, anomalie) : coupe au module de tests en ligne s'il est le dernier element."""
    for i, l in enumerate(lignes):
        if l.strip() == "#[cfg(test)]" and not l.startswith(" "):
            suivante = next((x for x in lignes[i + 1:] if not x.lstrip().startswith("#[")), "")
            if re.match(r"(pub(\([a-z]+\))? )?mod \w+ *\{", suivante):
                # le module de tests doit aller jusqu'a la fin du fichier : aucun element a la colonne 0 apres sa fermeture
                reste = lignes[i + 1:]
                fermetures = [k for k, x in enumerate(reste) if x.startswith("}")]
                if fermetures:
                    apres = [x for x in reste[fermetures[0] + 1:] if x.strip() and not x.startswith("//")]
                    if any(re.match(r"\S", x) for x in apres):
                        return lignes, "module de tests suivi de code (comptage sans coupe)"
                return lignes[:i], None
    return lignes, None


def main():
    racine = sys.argv[1]
    src = os.path.join(racine, "crates/kvu/src")
    code, exclus, tests_seuls, anomalies, brut = [], [], 0, [], 0
    casse = {}
    for d, _, fs in os.walk(src):
        for f in sorted(fs):
            if not f.endswith(".rs"):
                continue
            chemin = os.path.join(d, f)
            rel = os.path.relpath(chemin, racine)
            with open(chemin, encoding="utf-8") as h:
                lignes = h.read().split("\n")
            tous = [(n + 1, l) for n, l in enumerate(lignes) if MOTIF.search(l)]
            brut += len(tous)
            if any(e in rel for e in EXCLUSIONS):
                exclus.append((rel, len(tous)))
                continue
            gardees, anomalie = partie_code(lignes)
            if anomalie:
                anomalies.append((rel, anomalie))
            vus = [(n + 1, l) for n, l in enumerate(gardees) if MOTIF.search(l)]
            n_casse = sum(1 for l in gardees if MOTIF_CASSE.search(l))
            if n_casse:
                casse[rel] = n_casse
            tests_seuls += len(tous) - len(vus)
            code += [(rel, n, l.strip()[:110]) for n, l in vus]
    for rel, n, l in sorted(code):
        print(f"  {rel}:{n}: {l}")
    for rel, n in sorted(exclus):
        print(f"  EXCLU {rel} ({n} lignes)")
    for rel, a in anomalies:
        print(f"  ANOMALIE {rel} : {a}")
    print(f"INFO WP3-GREP-CASSE code={sum(casse.values())} (insensible a la casse, memes exclusions ; information, "
          f"pas un verdict) " + " ".join(f"{r}:{n}" for r, n in sorted(casse.items())))
    ok = not code and not anomalies
    print(f"{'PASS' if ok else 'FAIL'} WP3-GREP code={len(code)} tests_en_ligne_exclus={tests_seuls} "
          f"fichiers_exclus={len(exclus)} grep_brut={brut}")
    return 0 if ok else 1


sys.exit(main())
