# SPDX-License-Identifier: Apache-2.0
"""Recalcul HORS LIGNE de `corr_flux.csv` depuis `ops_flux.bin` (secteur de flux (1,1,0)).

Usage :
    python3 -I -B scripts/spectro/corr_flux_hl.py DOSSIER [--tau-max T] [--out FICHIER.csv]

`DOSSIER` doit contenir `manifeste.json` et `ops_flux.bin` (f64 petit-boutiste,
tableau [configuration][t][opérateur], opérateur = i_niveau*7 + k, 7 opérateurs
bruts par niveau). Le nombre de configurations est déduit de la taille du fichier :
cfg = taille / (tt * nlev * 7 * 8).

La formule reproduit exactement le code Rust (`flux.rs::base_corr`, `flux.rs::corr_flux`,
`correlateur.rs::paires`, `correlateur.rs::correlateur` avec `Mutant::Aucun`) :
  - base de corrélation par niveau : [a+ + a-, b+ + b-, c, T1, T2] (5 opérateurs,
    indice ia = niveau*5 + k) ;
  - C_ab(tau) = (1 / (2*tt)) * somme_t [ x_a(t) x_b(t+tau) + x_b(t) x_a(t+tau) ],
    les décalages de temps étant pris modulo tt ;
  - paires (ia, ib) avec ia <= ib, parcourues ia croissant puis ib croissant.

Le CSV écrit a pour en-tête `cfg,px,py,c<ia>_<ib>_<tau>,...` (tau = 0..tau_max),
une ligne par configuration, chaque valeur écrite `%.17e`. Une ligne JSON de
résumé est imprimée sur la sortie standard.

Refus (code de sortie 2, message sur stderr, aucun fichier de sortie écrit) :
manifeste ou binaire absent, manifeste sans `secteur_flux.niveaux`, taille du
binaire nulle ou non multiple de tt*nlev*7*8, valeur non finie, `--tau-max` négatif.
"""

import argparse
import json
import os
import sys

import numpy as np

NOPS_BRUTS = 7
NOPS_BASE = 5


class Refus(Exception):
    """Erreur d'entrée : provoque un code de sortie 2 et aucun fichier écrit."""


def lire_manifeste(dossier):
    """Charge `DOSSIER/manifeste.json` et renvoie le dictionnaire."""
    chemin = os.path.join(dossier, "manifeste.json")
    if not os.path.isfile(chemin):
        raise Refus("manifeste.json absent : %s" % chemin)
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        raise Refus("manifeste.json illisible (%s) : %s" % (chemin, e))
    if not isinstance(man, dict):
        raise Refus("manifeste.json : objet JSON attendu")
    return man


def parametres(man):
    """Extrait (tt, nlev, tau_max_du_manifeste) du manifeste."""
    sf = man.get("secteur_flux")
    if not isinstance(sf, dict):
        raise Refus("manifeste sans entrée secteur_flux")
    niveaux = sf.get("niveaux")
    if not isinstance(niveaux, list):
        raise Refus("secteur_flux sans liste niveaux")
    nlev = len(niveaux)
    if nlev == 0:
        raise Refus("secteur_flux : liste niveaux vide")
    cotes = man.get("cotes")
    temps = man.get("temps")
    if not isinstance(cotes, list) or not cotes:
        raise Refus("manifeste sans liste cotes")
    if not isinstance(temps, int) or isinstance(temps, bool) or not (0 <= temps < len(cotes)):
        raise Refus("manifeste : indice temps invalide")
    tt = cotes[temps]
    if not isinstance(tt, int) or isinstance(tt, bool) or tt < 1:
        raise Refus("manifeste : cote de temps invalide (%r)" % (tt,))
    tau_max = man.get("tau_max", sf.get("tau_max"))
    if tau_max is None:
        raise Refus("manifeste sans tau_max")
    if not isinstance(tau_max, int) or isinstance(tau_max, bool) or tau_max < 0:
        raise Refus("manifeste : tau_max invalide (%r)" % (tau_max,))
    return tt, nlev, tau_max


def lire_ops(dossier, tt, nlev):
    """Lit `DOSSIER/ops_flux.bin` et renvoie un tableau (cfg, tt, nlev*7)."""
    chemin = os.path.join(dossier, "ops_flux.bin")
    if not os.path.isfile(chemin):
        raise Refus("ops_flux.bin absent : %s" % chemin)
    taille = os.path.getsize(chemin)
    if taille == 0:
        raise Refus("ops_flux.bin vide : %s" % chemin)
    bloc = tt * nlev * NOPS_BRUTS * 8
    if taille % bloc != 0:
        raise Refus(
            "taille de ops_flux.bin (%d octets) non multiple de tt*nlev*7*8 = %d" % (taille, bloc)
        )
    ops = np.fromfile(chemin, dtype="<f8")
    if ops.size * 8 != taille:
        raise Refus("ops_flux.bin : lecture incomplète (%d valeurs pour %d octets)" % (ops.size, taille))
    if not np.isfinite(ops).all():
        raise Refus("valeur non finie dans ops_flux.bin")
    return ops.reshape(-1, tt, nlev * NOPS_BRUTS)


def base_corr(ops):
    """Port de `flux.rs::base_corr` : (cfg, tt, nlev*7) -> (cfg, tt, nlev*5)."""
    cfg, tt, n = ops.shape
    nlev = n // NOPS_BRUTS
    r = ops.reshape(cfg, tt, nlev, NOPS_BRUTS)
    b = np.stack(
        [r[..., 0] + r[..., 1], r[..., 2] + r[..., 3], r[..., 4], r[..., 5], r[..., 6]],
        axis=-1,
    )
    return b.reshape(cfg, tt, nlev * NOPS_BASE)


def paires(n):
    """Port de `correlateur.rs::paires` : (0,0), (0,1), ..., (1,1), ..."""
    return [(a, b) for a in range(n) for b in range(a, n)]


def correlateur(base, tau_max):
    """Port de `correlateur.rs::correlateur` (Mutant::Aucun) : (cfg, npaires*(tau_max+1))."""
    cfg, tt, n = base.shape
    t = np.arange(tt)
    colonnes = []
    for a, b in paires(n):
        x = base[:, :, a]
        y = base[:, :, b]
        for tau in range(tau_max + 1):
            t2 = (t + tau) % tt
            acc = (x * y[:, t2] + y * x[:, t2]).sum(axis=1) / (2.0 * tt)
            colonnes.append(acc)
    if not colonnes:
        return np.zeros((cfg, 0), dtype=np.float64)
    return np.stack(colonnes, axis=1)


def entete(n, tau_max):
    """En-tête du CSV : cfg, px, py puis c<ia>_<ib>_<tau>."""
    cols = ["cfg", "px", "py"]
    for a, b in paires(n):
        for tau in range(tau_max + 1):
            cols.append("c%d_%d_%d" % (a, b, tau))
    return cols


def ecrire_csv(chemin, C, cols):
    """Écrit le CSV de façon atomique (fichier temporaire puis renommage)."""
    tmp = chemin + ".tmp"
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write(",".join(cols) + "\n")
            for i in range(C.shape[0]):
                f.write("%d,0,0," % i + ",".join("%.17e" % v for v in C[i]) + "\n")
        os.replace(tmp, chemin)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    """Point d'entrée : renvoie le code de sortie."""
    p = argparse.ArgumentParser(
        description="Recalcule corr_flux.csv hors ligne depuis ops_flux.bin (secteur de flux)."
    )
    p.add_argument("dossier", metavar="DOSSIER", help="dossier contenant manifeste.json et ops_flux.bin")
    p.add_argument("--tau-max", type=int, default=None, help="décalage maximal (défaut : manifeste)")
    p.add_argument("--out", default=None, help="fichier CSV de sortie (défaut : DOSSIER/corr_flux_hl.csv)")
    args = p.parse_args(argv)
    try:
        man = lire_manifeste(args.dossier)
        tt, nlev, tau_max_man = parametres(man)
        if args.tau_max is None:
            tau_max = tau_max_man
        else:
            if args.tau_max < 0:
                raise Refus("--tau-max négatif (%d)" % args.tau_max)
            tau_max = args.tau_max
        ops = lire_ops(args.dossier, tt, nlev)
        cfg = ops.shape[0]
        base = base_corr(ops)
        C = correlateur(base, tau_max)
        cols = entete(nlev * NOPS_BASE, tau_max)
        sortie = args.out if args.out is not None else os.path.join(args.dossier, "corr_flux_hl.csv")
        ecrire_csv(sortie, C, cols)
    except Refus as e:
        sys.stderr.write("corr_flux_hl: %s\n" % e)
        return 2
    except OSError as e:
        sys.stderr.write("corr_flux_hl: erreur d'entrée/sortie : %s\n" % e)
        return 2
    print(
        json.dumps(
            {
                "cfg": int(cfg),
                "tt": int(tt),
                "nlev": int(nlev),
                "tau_max": int(tau_max),
                "colonnes": int(C.shape[1]),
                "sortie": sortie,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
