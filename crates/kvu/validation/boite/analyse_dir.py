#!/usr/bin/env python3
"""Analyses indépendantes des gates statistiques des boîtes de Dirichlet."""

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import math
from pathlib import Path
import struct

import numpy as np

REFERENCE = Path(__file__).resolve().parents[2] / "reference" / "qw1"
sys.path.insert(0, str(REFERENCE))
from gamma import uwerr


class Parseur(argparse.ArgumentParser):
    """Aide et diagnostics d'arguments en français."""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "arguments positionnels"
        self._optionals.title = "options"
        self.add_argument("-h", "--help", action="help",
                          help="afficher cette aide et terminer")

    def error(self, message):
        for anglais, francais in (
            ("the following arguments are required", "arguments obligatoires"),
            ("unrecognized arguments", "arguments inconnus"),
            ("expected one argument", "une valeur est attendue"),
            ("invalid choice", "choix invalide"),
            ("choose from", "choisir parmi"),
            ("argument command", "sous-commande"),
        ):
            message = message.replace(anglais, francais)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog} : erreur : {message}\n")


def entier(texte):
    """Convertit un argument entier avec un diagnostic français."""
    try:
        return int(texte)
    except ValueError:
        raise argparse.ArgumentTypeError("un entier est attendu") from None


def reel(texte):
    """Convertit un argument réel fini."""
    try:
        valeur = float(texte)
    except ValueError:
        raise argparse.ArgumentTypeError("un réel est attendu") from None
    if not math.isfinite(valeur):
        raise argparse.ArgumentTypeError("un réel fini est attendu")
    return valeur


def _ell(valeur):
    if isinstance(valeur, bool) or not isinstance(valeur, int) or valeur < 2 or valeur % 2:
        raise ValueError("ell doit être un entier pair supérieur ou égal à 2")
    return valeur


def _csv(chemin, entete, vide_ok=False):
    """Lit un CSV de séries ; `vide_ok` admet un fichier réduit à son en-tête (profil par couche jamais mesuré : --profil-tous 0)."""
    with chemin.open(encoding="utf-8", newline="") as fichier:
        lecteur = csv.reader(fichier)
        if next(lecteur, None) != entete:
            raise ValueError(f"en-tête incorrect : {chemin}")
        lignes = list(lecteur)
    if not lignes and vide_ok:
        return np.empty(0, dtype=np.int64), np.empty((0, len(entete) - 1))
    if not lignes or any(len(ligne) != len(entete) for ligne in lignes):
        raise ValueError(f"série vide ou ligne incomplète : {chemin}")
    try:
        pas = np.array([int(ligne[0]) for ligne in lignes], dtype=np.int64)
        valeurs = np.array([list(map(float, ligne[1:])) for ligne in lignes])
    except (ValueError, OverflowError):
        raise ValueError(f"valeur numérique incorrecte : {chemin}") from None
    if np.any(pas < 0) or np.any(pas[1:] <= pas[:-1]):
        raise ValueError(f"pas négatifs ou non croissants : {chemin}")
    return pas, valeurs


def lire_run(dossier):
    """Relit un manifeste et toutes ses chaînes, sans concaténer les séries."""
    dossier = Path(dossier)
    with (dossier / "manifest_regen.json").open(encoding="utf-8") as fichier:
        parametres = json.load(fichier)["parametres"]
    boite = parametres["boite"]
    ell = _ell(boite["ell"] if "ell" in boite else parametres["L"])
    beta = float(parametres["beta"])
    if not math.isfinite(beta):
        raise ValueError("beta doit être fini")
    run = {"ell": ell, "beta": beta, "chaines": [], "profils": {}}
    for description in boite["chaines"]:
        c = description["chaine"]
        if isinstance(c, bool) or not isinstance(c, int) or c < 0 or c in run["profils"]:
            raise ValueError("numéro de chaîne invalide ou dupliqué")
        chemin = dossier / f"c{c:03d}"
        pas, mesures = _csv(chemin / "mesures.csv", ["pas", "pc", "w2"])
        _, profil = _csv(chemin / "profil.csv", ["pas"] + [f"d{d}" for d in range(ell // 2)], vide_ok=True)
        if not np.isfinite(mesures[:, 0]).all() or not np.isfinite(profil).all():
            raise ValueError(f"mesure ou profil non fini : chaîne {c}")
        if ell >= 4 and not np.isfinite(mesures[:, 1]).all():
            raise ValueError(f"W2 non fini : chaîne {c}")
        run["chaines"].append({"chaine": c, "bord": description["bord"],
                               "graine": description["graine"], "pas": pas,
                               "pc": mesures[:, 0], "w2": mesures[:, 1]})
        run["profils"][c] = profil
    if not run["chaines"]:
        raise ValueError("le run ne contient aucune chaîne")
    return run


def cellule(run, bord, obs):
    """Moyenne des moyennes et erreur Γ de M chaînes indépendantes (S = 1,5)."""
    if obs not in {"pc", "w2"}:
        raise ValueError("observable attendue : pc ou w2")
    details = []
    for chaine in run["chaines"]:
        if chaine["bord"] != bord:
            continue
        serie = np.asarray(chaine[obs], dtype=float)
        if serie.ndim != 1 or not serie.size or not np.isfinite(serie).all():
            raise ValueError(f"série {obs} vide ou non finie pour le bord {bord}")
        moyenne, erreur, tau, _ = uwerr(serie, S=1.5)
        details.append((float(moyenne), float(erreur), float(tau)))
    if not details:
        raise ValueError(f"aucune chaîne pour le bord {bord}")
    m = len(details)
    return (math.fsum(d[0] for d in details) / m,
            math.sqrt(math.fsum(d[1] ** 2 for d in details)) / m, details)


def z(m1, s1, m2, s2):
    """Écart signé en unités d'erreur combinée, y compris à erreur nulle."""
    erreur = math.hypot(s1, s2)
    if erreur == 0:
        return 0.0 if m1 == m2 else math.copysign(math.inf, m1 - m2)
    return (m1 - m2) / erreur


def delta(run, a, b, obs):
    """Différence de deux cellules indépendantes."""
    m1, s1, _ = cellule(run, a, obs)
    m2, s2, _ = cellule(run, b, obs)
    return m1 - m2, math.hypot(s1, s2)


def reference_qw1():
    """Charge les constantes gelées de QW1 sans les recopier."""
    with (REFERENCE / "qw1_reference.json").open(encoding="utf-8") as fichier:
        return json.load(fichier)


def _runs(arguments):
    runs = {}
    for argument in arguments:
        texte, separateur, dossier = argument.partition("=")
        if not separateur or not dossier:
            raise ValueError("--run attend ELL=DOSSIER")
        try:
            ell = _ell(int(texte))
        except ValueError:
            raise ValueError("--run attend un ell pair supérieur ou égal à 2") from None
        if ell in runs:
            raise ValueError(f"ell fourni plusieurs fois : {ell}")
        run = lire_run(dossier)
        if run["ell"] != ell:
            raise ValueError(f"ell={ell} diffère du manifeste ({run['ell']})")
        runs[ell] = run
    return runs


def _verdict(identifiant, reussi, detail="", mutant=False):
    if mutant:
        identifiant += "-MUTANT"
        if not reussi:
            detail = "ECHEC_OUTIL " + detail
    print(f"{'PASS' if reussi else 'FAIL'} {identifiant} {detail}".rstrip())
    return 0 if reussi else 1


def _qw1(args):
    reference = reference_qw1()
    runs = _runs(args.run)
    comparaisons = {}
    differences = {}
    for ell, run in runs.items():
        if str(ell) not in reference["cellules"]:
            raise ValueError(f"aucune référence QW1 pour ell={ell}")
        ref = reference["cellules"][str(ell)]
        for obs in (["pc", "w2"] if ell >= 4 else ["pc"]):
            nom = "d" + obs
            d, erreur = delta(run, "cold", "z2full", obs)
            r, sr = ref[nom]
            ecart = z(d, erreur, r, sr)
            comparaisons[ell, nom] = ecart
            differences[ell, nom] = d
            print(f"INFO G-DIR-QW1 ell={ell} obs={nom} delta={d:.17g} "
                  f"err={erreur:.17g} ref={r:.17g} err_ref={sr:.17g} z={ecart:.17g}")
            for bord in dict.fromkeys(c["bord"] for c in run["chaines"]):
                if bord in ref[obs]:
                    m, s, _ = cellule(run, bord, obs)
                    mr, se = ref[obs][bord]
                    print(f"INFO G-DIR-QW1 ell={ell} obs={obs} bord={bord} "
                          f"moyenne={m:.17g} err={s:.17g} ref={mr:.17g} "
                          f"err_ref={se:.17g} z={z(m, s, mr, se):.17g}")
            cle = f"{nom}_ell{ell}"
            if cle in reference["independant_verification"]:
                mr, se = reference["independant_verification"][cle]
                print(f"INFO G-DIR-QW1 ell={ell} obs={nom} reference=independante "
                      f"z={z(d, erreur, mr, se):.17g}")
    echecs = []
    if args.mutant:
        if 8 in runs and not abs(comparaisons[8, "dpc"]) >= 3:
            echecs.append("ell=8,dpc : |z| < 3")
        if 4 in runs and not abs(differences[4, "dpc"]) < 0.02:
            echecs.append("ell=4,dpc : |delta| >= 0,02")
    else:
        if 8 not in runs:
            echecs.append("ell=8,dpc et ell=8,dw2 manquants")
        echecs.extend(f"ell={ell},{obs} : z={ecart:.17g}"
                      for (ell, obs), ecart in comparaisons.items() if not abs(ecart) < 3)
    return _verdict("G-DIR-QW1", not echecs,
                    "; ".join(echecs) if echecs else "critères satisfaits", args.mutant)


def _jauge(args):
    ecarts = {}
    for ell, run in _runs(args.run).items():
        m, s, _ = cellule(run, "haar" if args.mutant else "gauge", "pc")
        mc, sc, _ = cellule(run, "cold", "pc")
        ecarts[ell] = z(m, s, mc, sc)
        print(f"INFO G-DIR-JAUGE ell={ell} z={ecarts[ell]:.17g}")
    reussi = (any(abs(v) > 3 for ell, v in ecarts.items() if ell in {2, 4})
              if args.mutant else all(abs(v) <= 3 for v in ecarts.values()))
    return _verdict("G-DIR-JAUGE", reussi, "seuil de jauge évalué", args.mutant)


def _agit(args):
    ell, run = next(iter(_runs([args.run]).items()))
    m, s, _ = cellule(run, "z2full", "pc")
    mc, sc, _ = cellule(run, "cold", "pc")
    ecart = z(m, s, mc, sc)
    reussi = abs(ecart) <= 5 if args.mutant else abs(ecart) > 5
    return _verdict("G-DIR-AGIT", reussi, f"ell={ell} z={ecart:.17g}", args.mutant)


def bessel_i(nu, x):
    """Série de I_nu(x), arrêt à 1e-18 relatif, sans bibliothèque spéciale."""
    if nu < 0 or not math.isfinite(x):
        raise ValueError("ordre ou argument de Bessel invalide")
    terme = (x / 2) ** nu / math.factorial(nu)
    somme = terme
    if terme == 0:
        return somme
    for k in range(1, 100000):
        terme *= (x / 2) ** 2 / (k * (k + nu))
        somme += terme
        if not math.isfinite(somme):
            raise ValueError("argument trop grand pour la série de Bessel")
        if abs(terme) <= 1e-18 * abs(somme):
            return somme
    raise ValueError("la série de Bessel ne converge pas")


def attendu_sc(beta):
    """Plaquette au couplage fort, avec sa limite en beta nul."""
    u = bessel_i(2, beta) / bessel_i(1, beta) if beta != 0 else 0.0
    return u + 4 * u ** 5


def _sc(args):
    if args.mutant and args.beta_nominal is None:
        raise ValueError("--beta-nominal est obligatoire avec --mutant")
    if not args.mutant and args.beta_nominal is not None:
        raise ValueError("--beta-nominal est réservé au contrôle --mutant")
    run = lire_run(args.run)
    attendu = attendu_sc(args.beta_nominal if args.mutant else run["beta"])
    ecarts = []
    for bord in dict.fromkeys(c["bord"] for c in run["chaines"]):
        m, s, _ = cellule(run, bord, "pc")
        ecart = z(m, s, attendu, 0)
        ecarts.append(ecart)
        print(f"INFO G-DIR-SC bord={bord} moyenne={m:.17g} err={s:.17g} "
              f"attendu={attendu:.17g} z={ecart:.17g}")
    reussi = any(abs(v) > 3 for v in ecarts) if args.mutant else all(abs(v) < 3 for v in ecarts)
    return _verdict("G-DIR-SC", reussi, "seuil du couplage fort évalué", args.mutant)


def _observables(chemin):
    with Path(chemin).open(encoding="utf-8") as fichier:
        valeurs = json.load(fichier)
    pc, w2, profil = valeurs["pc"], valeurs["w2"], valeurs["profil"]
    if not isinstance(profil, list):
        raise ValueError("le profil doit être une liste")
    for v in [pc] + ([] if w2 is None else [w2]) + profil:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("observable numérique non finie ou invalide")
    return pc, w2, profil


def _obs_cmp(args):
    if args.tol < 0:
        raise ValueError("la tolérance doit être positive ou nulle")
    pa, wa, da = _observables(args.a)
    pb, wb, db = _observables(args.b)
    structure = len(da) == len(db) and (wa is None) == (wb is None)
    couples = [(pa, pb)] + list(zip(da, db))
    if wa is not None and wb is not None:
        couples.append((wa, wb))
    ecart = max(abs(a - b) for a, b in couples) if structure else math.inf
    identique = all(struct.pack("!d", a) == struct.pack("!d", b) for a, b in couples)
    reussi = structure and (identique if args.identique else ecart <= args.tol)
    return _verdict("G-DIR-OBS-CMP", reussi,
                    f"ecart={ecart:.17g} tol={args.tol:.17g}"
                    + (" identique=bits" if args.identique else ""))


def _stats(args):
    run = lire_run(args.dossier)
    for bord in dict.fromkeys(c["bord"] for c in run["chaines"]):
        if args.obs == "w2" and run["ell"] < 4:
            print(f"SKIP G-DIR-STATS bord={bord} W2 indéfini pour ell < 4")
            continue
        m, s, details = cellule(run, bord, args.obs)
        sortie = {"bord": bord, "obs": args.obs, "M": len(details), "moyenne": m,
                  "erreur": s, "tau": math.fsum(d[2] for d in details) / len(details)}
        print("INFO G-DIR-STATS " + json.dumps(sortie, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv=None):
    parseur = Parseur(description=__doc__)
    commandes = parseur.add_subparsers(dest="commande", required=True, title="sous-commandes")
    for nom, aide, fonction in (
        ("qw1", "comparer aux références gelées de QW1", _qw1),
        ("jauge", "évaluer le témoin de jauge", _jauge),
        ("agit", "vérifier que le bord agit", _agit),
        ("sc", "vérifier la série de couplage fort", _sc),
        ("obs-cmp", "comparer deux fichiers d'observables", _obs_cmp),
        ("stats", "afficher les statistiques par bord", _stats),
    ):
        p = commandes.add_parser(nom, help=aide, description=aide)
        p.set_defaults(fonction=fonction)
        if nom in {"qw1", "jauge", "agit", "sc"}:
            p.add_argument("--run", required=True, action="append" if nom in {"qw1", "jauge"} else "store",
                           metavar="DOSSIER" if nom == "sc" else "ELL=DOSSIER",
                           help="répertoire du run à analyser")
            p.add_argument("--mutant", action="store_true", help="évaluer le contrôle négatif")
        if nom == "sc":
            p.add_argument("--beta-nominal", type=reel, metavar="BETA",
                           help="couplage nominal obligatoire pour le mutant")
        if nom == "obs-cmp":
            p.add_argument("a", metavar="A.json", help="premier fichier")
            p.add_argument("b", metavar="B.json", help="second fichier")
            p.add_argument("--tol", type=reel, default=1e-12, help="tolérance absolue (défaut : 1e-12)")
            p.add_argument("--identique", action="store_true", help="exiger les mêmes bits flottants")
        if nom == "stats":
            p.add_argument("dossier", metavar="DOSSIER", help="répertoire du run")
            p.add_argument("--obs", choices=["pc", "w2"], default="pc", help="observable (défaut : pc)")
    args = parseur.parse_args(argv)
    try:
        return args.fonction(args)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as erreur:
        parseur.error(f"entrée invalide ({type(erreur).__name__}) : {erreur}")


if __name__ == "__main__":
    sys.exit(main())
