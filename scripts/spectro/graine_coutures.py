# SPDX-License-Identifier: Apache-2.0
"""Graine de M2 : a3 = 2 ln Z(A) - ln Z(B) = 6 V_B * Int_0^beta (<p>_A - <p>_B) db
(V_B = 2 V_A ; les constantes N_p*beta se compensent).

Usage : graine_coutures.py DIR_A DIR_B [--blocs N] [--sortie FICHIER.json]

Les dossiers DIR_A et DIR_B sont ceux ecrits par
`kvu su2 gen --cotes ... --beta b1,b2,... --graine ... --outdir DIR` : un
sous-dossier de chaine par beta (cNNN/measurements.csv, cNNN/geometrie.json)
et un manifest.json dont la cle `argv` porte `--cotes` et `--beta`.

<p> est la moyenne de la colonne `plaquette` des lignes de phase `sample`
(les lignes `initial` sont ignorees) ; quadrature trapeze en beta.

L'erreur estimee tient compte de l'autocorrelation de la chaine de Markov :
jackknife par blocs contigus de chaque chaine (N blocs, mesures en trop a la
fin ignorees), les chaines etant independantes entre elles ; la variance de a3
est la somme des variances des moyennes de chaines ponderees par le carre des
poids de la quadrature (poids trapeze x 6 V_B).

Refus (code de sortie 2, message sur stderr, aucun fichier ecrit) : manifeste
illisible, option --cotes/--beta absente, V_B != 2 V_A, betas differents entre
A et B ou non strictement croissants ou moins de 2, nombre de chaines
different du nombre de betas, --blocs < 2, chaine plus courte que le nombre de
blocs, valeur non finie, mesures illisibles.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np


class Refus(Exception):
    """Erreur « propre » : message sur stderr, code de sortie 2, pas de traceback."""


def lire_manifeste(dossier):
    """Lit dossier/manifest.json et renvoie le dictionnaire ; Refus si illisible."""
    chemin = os.path.join(dossier, "manifest.json")
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        raise Refus("manifeste illisible: %s (%s)" % (chemin, e))
    if not isinstance(m, dict):
        raise Refus("manifeste illisible: %s n'est pas un objet JSON" % chemin)
    return m


def option_argv(manifeste, nom):
    """Valeur de l'option `nom` dans manifeste["argv"] ; Refus si absente."""
    argv = manifeste.get("argv")
    if not isinstance(argv, list):
        raise Refus("option %s absente du manifeste (cle argv absente ou invalide)" % nom)
    for i, a in enumerate(argv):
        if a == nom:
            if i + 1 >= len(argv):
                raise Refus("option %s absente du manifeste (valeur manquante)" % nom)
            return argv[i + 1]
    raise Refus("option %s absente du manifeste" % nom)


def cotes_de(manifeste):
    """Cotes (4 entiers > 0) lues dans argv."""
    brut = option_argv(manifeste, "--cotes")
    try:
        cotes = tuple(int(x) for x in str(brut).split(","))
    except (TypeError, ValueError):
        raise Refus("option --cotes invalide: %r" % (brut,))
    if len(cotes) != 4 or any(c <= 0 for c in cotes):
        raise Refus("option --cotes invalide: %r" % (brut,))
    return cotes


def betas_de(manifeste):
    """Betas (floats finis) lus dans argv."""
    brut = option_argv(manifeste, "--beta")
    try:
        betas = [float(x) for x in str(brut).split(",")]
    except (TypeError, ValueError):
        raise Refus("option --beta invalide: %r" % (brut,))
    if not betas:
        raise Refus("option --beta invalide: %r" % (brut,))
    for b in betas:
        if not math.isfinite(b):
            raise Refus("beta non fini: %r" % (b,))
    return betas


def lire_mesures(chemin):
    """Valeurs de plaquette des lignes de phase `sample` de measurements.csv."""
    try:
        with open(chemin, "r", encoding="utf-8", newline="") as f:
            lignes = list(csv.reader(f))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise Refus("mesures illisibles: %s (%s)" % (chemin, e))
    if not lignes:
        raise Refus("mesures illisibles: %s est vide" % chemin)
    entete = [c.strip() for c in lignes[0]]
    try:
        i_phase = entete.index("phase")
        i_plaq = entete.index("plaquette")
    except ValueError:
        raise Refus("mesures illisibles: en-tete sans colonne phase/plaquette dans %s" % chemin)
    valeurs = []
    for num, ligne in enumerate(lignes[1:], start=2):
        if not ligne or all(not c.strip() for c in ligne):
            continue
        if len(ligne) <= max(i_phase, i_plaq):
            raise Refus("mesures illisibles: ligne %d trop courte dans %s" % (num, chemin))
        if ligne[i_phase].strip() != "sample":
            continue
        try:
            v = float(ligne[i_plaq])
        except ValueError:
            raise Refus("mesures illisibles: plaquette non numerique ligne %d de %s" % (num, chemin))
        if not math.isfinite(v):
            raise Refus("valeur non finie (plaquette) ligne %d de %s" % (num, chemin))
        valeurs.append(v)
    if not valeurs:
        raise Refus("mesures illisibles: aucune ligne sample dans %s" % chemin)
    return np.asarray(valeurs, dtype=float)


def lire_chaines(dossier, betas):
    """Liste des chaines du dossier, triees par nom de sous-dossier."""
    chemins = sorted(glob.glob(os.path.join(dossier, "c*", "measurements.csv")))
    chaines = []
    for chemin in chemins:
        ch = os.path.dirname(chemin)
        geo = os.path.join(ch, "geometrie.json")
        cotes = None
        try:
            with open(geo, "r", encoding="utf-8") as f:
                g = json.load(f)
            if isinstance(g, dict) and "cotes" in g:
                cotes = tuple(int(x) for x in g["cotes"])
        except (OSError, ValueError, KeyError, TypeError):
            cotes = None
        # Repli : certaines geometries ne portent pas la cle `cotes`
        # (ex. {"schema": "kvu-geometrie/1"}) ; la source de verite est alors
        # `--cotes` du manifeste du dossier (cf. CONCEPTION.md).
        if cotes is None:
            cotes = cotes_de(lire_manifeste(dossier))
        mesures = lire_mesures(chemin)
        chaines.append({"dir": ch, "cotes": cotes, "mesures": mesures})
    return chaines


def moyennes_chaines(chaines):
    """Moyenne de chaque chaine."""
    return np.asarray([float(np.mean(c["mesures"])) for c in chaines], dtype=float)


def jackknife_blocs(x, n_blocs):
    """(moyenne, variance de la moyenne) par jackknife par blocs contigus."""
    x = np.asarray(x, dtype=float)
    n = x.size
    L = n // n_blocs
    if L < 1:
        raise Refus("chaine trop courte: %d mesures < %d blocs" % (n, n_blocs))
    moy_bloc = np.asarray([np.mean(x[j * L:(j + 1) * L]) for j in range(n_blocs)], dtype=float)
    moy_tot = float(np.mean(moy_bloc))
    pseudo = (n_blocs * moy_tot - moy_bloc) / (n_blocs - 1)
    var = (n_blocs - 1) / n_blocs * float(np.sum((pseudo - np.mean(pseudo)) ** 2))
    return moy_tot, var


def integrale_trapeze(betas, d):
    """Integrale de d(beta) par la methode des trapezes."""
    total = 0.0
    for i in range(len(betas) - 1):
        total += (betas[i + 1] - betas[i]) * (d[i] + d[i + 1]) / 2.0
    return total


def poids_trapeze(betas):
    """Poids de la quadrature trapeze (sans le facteur 6 V_B)."""
    n = len(betas)
    poids = []
    for i in range(n):
        if i == 0:
            poids.append((betas[1] - betas[0]) / 2.0)
        elif i == n - 1:
            poids.append((betas[n - 1] - betas[n - 2]) / 2.0)
        else:
            poids.append((betas[i + 1] - betas[i - 1]) / 2.0)
    return poids


def calculer(dir_a, dir_b, n_blocs):
    """Orchestration complete : renvoie le dictionnaire de resultats."""
    if n_blocs < 2:
        raise Refus("--blocs doit valoir au moins 2 (recu %d)" % n_blocs)
    ma = lire_manifeste(dir_a)
    mb = lire_manifeste(dir_b)
    cotes_a = cotes_de(ma)
    cotes_b = cotes_de(mb)
    va = cotes_a[0] * cotes_a[1] * cotes_a[2] * cotes_a[3]
    vb = cotes_b[0] * cotes_b[1] * cotes_b[2] * cotes_b[3]
    if vb != 2 * va:
        raise Refus("V_B != 2*V_A (V_A=%d, V_B=%d)" % (va, vb))
    betas_a = betas_de(ma)
    betas_b = betas_de(mb)
    if len(betas_a) < 2:
        raise Refus("il faut au moins 2 beta (recu %d)" % len(betas_a))
    if betas_a != betas_b:
        raise Refus("betas differents entre A et B: %r vs %r" % (betas_a, betas_b))
    for i in range(len(betas_a) - 1):
        if not betas_a[i + 1] > betas_a[i]:
            raise Refus("betas non strictement croissants: %r" % (betas_a,))
    betas = betas_a
    chaines_a = lire_chaines(dir_a, betas)
    chaines_b = lire_chaines(dir_b, betas)
    if len(chaines_a) != len(betas):
        raise Refus("nombre de chaines de %s (%d) different du nombre de betas (%d)"
                    % (dir_a, len(chaines_a), len(betas)))
    if len(chaines_b) != len(betas):
        raise Refus("nombre de chaines de %s (%d) different du nombre de betas (%d)"
                    % (dir_b, len(chaines_b), len(betas)))
    for ch in chaines_a + chaines_b:
        if ch["mesures"].size < n_blocs:
            raise Refus("chaine %s : %d mesures < %d blocs"
                        % (ch["dir"], ch["mesures"].size, n_blocs))
    moy_a = np.empty(len(betas))
    var_a = np.empty(len(betas))
    moy_b = np.empty(len(betas))
    var_b = np.empty(len(betas))
    for i in range(len(betas)):
        moy_a[i], var_a[i] = jackknife_blocs(chaines_a[i]["mesures"], n_blocs)
        moy_b[i], var_b[i] = jackknife_blocs(chaines_b[i]["mesures"], n_blocs)
    d = moy_a - moy_b
    integ = integrale_trapeze(betas, d)
    a3 = 6.0 * vb * integ
    poids = poids_trapeze(betas)
    var_a3 = 0.0
    for i in range(len(betas)):
        p = 6.0 * vb * poids[i]
        var_a3 += p * p * (var_a[i] + var_b[i])
    sigma = math.sqrt(var_a3)
    for nom, v in (("a3", a3), ("integrale", integ), ("sigma_jackknife", sigma)):
        if not math.isfinite(v):
            raise Refus("resultat non fini: %s = %r" % (nom, v))
    n_mesures = min(int(ch["mesures"].size) for ch in chaines_a + chaines_b)
    return {
        "a3": float(a3),
        "sigma_jackknife": float(sigma),
        "integrale": float(integ),
        "n_blocs": int(n_blocs),
        "n_mesures": int(n_mesures),
        "betas": [float(b) for b in betas],
        "d": [float(x) for x in d],
        "vb": int(vb),
    }


def ecrire_json(chemin, res):
    """Ecrit les 6 cles exigees (apres tous les calculs)."""
    contenu = {
        "a3": res["a3"],
        "sigma_jackknife": res["sigma_jackknife"],
        "integrale": res["integrale"],
        "n_blocs": res["n_blocs"],
        "n_mesures": res["n_mesures"],
        "betas": res["betas"],
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main(argv=None):
    """Point d'entree ; renvoie le code de sortie."""
    p = argparse.ArgumentParser(
        description="Graine de M2 a3 = 6 V_B * integrale(<pA>-<pB>) dbeta, "
                    "erreur par jackknife par blocs.")
    p.add_argument("DIR_A")
    p.add_argument("DIR_B")
    p.add_argument("--blocs", type=int, default=20)
    p.add_argument("--sortie", default=None)
    args = p.parse_args(argv)
    try:
        res = calculer(args.DIR_A, args.DIR_B, args.blocs)
        betas = res["betas"]
        d = res["d"]
        print("betas=%s" % (betas,))
        print("<pA>-<pB>=%s" % (d,))
        print("graine a3 = 6*V_B*integrale = %s  (integrale trapeze %s)"
              % (format(6 * res["vb"] * res["integrale"], ".6g"),
                 format(res["integrale"], ".6g")))
        print("a3 = %s +- %s (jackknife par blocs, %d blocs)"
              % (format(res["a3"], ".6g"), format(res["sigma_jackknife"], ".6g"),
                 res["n_blocs"]))
        if args.sortie:
            ecrire_json(args.sortie, res)
    except Refus as e:
        print("erreur: %s" % e, file=sys.stderr)
        return 2
    except Exception as e:  # filet de securite : jamais de traceback
        print("erreur: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
