# SPDX-License-Identifier: Apache-2.0
"""Lecteur direct des séries de torsion écrites par `kvu su2 thermo mesure --series`.

Lit, pour une β donnée, les fichiers `series/bras<a>_g<ggg>_fe.bin` (f64 petit-boutiste,
tableau [mesure][chaîne du groupe], chaîne = k·R + i) et calcule log R avec l'estimateur
multi-histogramme biné `Histogramme` (copie exacte de l'ancien estimateur, non modifiée).

Aucune conversion vers l'ancienne disposition `su2twist`, aucun champ `seed` fabriqué :
les séries sont lues telles quelles et le bootstrap rééchantillonne les répliques.

Bootstrap et appariement : le mode est choisi d'après le champ `appariement` du manifeste
et reporté dans la clé de sortie `bootstrap_mode`.
  - `appariement` égal EXACTEMENT à la chaîne du producteur
    (« la graine de la chaîne (nœud k, réplique i) est G + k·R + i pour tous les bras ») :
    mode `conjoint` — à chaque tirage UN SEUL vecteur d'indices est tiré et appliqué à tous
    les bras ; les covariances entre bras sont conservées, `sigma_bootstrap` est correcte.
  - `appariement` commençant par `indépendant` ou `independant` : mode `independant` —
    un tirage par bras (les covariances entre bras sont ignorées).
  - toute autre valeur (ou champ absent / non chaîne) : refus (code 2).

Usage :
    python3 -I -B scripts/thermo/lire_series.py DOSSIER --beta B
        [--bin-width H] [--bootstrap N] [--graine-bootstrap S] [--maxit M] [--out FICHIER.json]

Codes de sortie : 0 succès ; 2 refus (entrée invalide) ; 3 Newton non convergé sur les
données complètes (le JSON est tout de même produit, avec newton_ok = false).
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Avec `python3 -I`, le dossier du script n'est pas dans sys.path : on l'ajoute nous-mêmes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from histogramme import Histogramme  # noqa: E402

TOL = 1e-12

APPARIEMENT_CONJOINT = ('la graine de la chaîne (nœud k, réplique i) est G + k·R + i '
                        'pour tous les bras')
PREFIXES_INDEPENDANT = ('indépendant', 'independant')


class Refus(Exception):
    """Entrée invalide : refus avec code de sortie non nul et aucun fichier écrit."""


def _sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, 'rb') as f:
        for bloc in iter(lambda: f.read(1 << 20), b''):
            h.update(bloc)
    return h.hexdigest()


def _entier_positif(d, cle, contexte):
    """Entier JSON strictement positif (bool, float, chaîne, null refusés)."""
    if cle not in d:
        raise Refus(f"{contexte} : champ « {cle} » absent")
    v = d[cle]
    if isinstance(v, bool) or not isinstance(v, int):
        raise Refus(f"{contexte} : « {cle} » doit être un entier JSON (reçu {v!r})")
    if v <= 0:
        raise Refus(f"{contexte} : « {cle} » doit être > 0 (reçu {v!r})")
    return v


def _poids(man):
    poids = man.get('poids')
    if not isinstance(poids, list) or not poids:
        raise Refus("manifeste : champ « poids » absent ou vide")
    vals = []
    for x in poids:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise Refus(f"poids invalide : chaque poids doit être un nombre fini (reçu {x!r})")
        fx = float(x)
        if not math.isfinite(fx):
            raise Refus(f"poids invalide : chaque poids doit être un nombre fini (reçu {x!r})")
        vals.append(fx)
    K = len(vals)
    attendu = np.linspace(-1.0, 1.0, K)
    if np.max(np.abs(np.asarray(vals, float) - attendu)) > TOL:
        raise Refus(f"poids inattendus : attendu linspace(-1, 1, {K})")
    return vals


def lire_manifeste(dossier):
    dossier = Path(dossier).resolve()
    chemin = dossier / 'manifest_thermo.json'
    if not chemin.is_file():
        raise Refus(f"manifeste introuvable : {chemin}")
    try:
        man = json.loads(chemin.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise Refus(f"manifeste illisible : {e}")
    if not isinstance(man, dict):
        raise Refus("manifeste : objet JSON attendu")
    if 'series' not in man or not isinstance(man['series'], dict):
        raise Refus("manifeste : pas de champ « series » (dossier écrit sans --series)")
    return dossier, man, _sha256(chemin)


def indices_beta(man, beta):
    betas = man.get('betas')
    if not isinstance(betas, list) or not betas:
        raise Refus("manifeste : champ « betas » absent ou vide")
    try:
        vals = [float(v) for v in betas]
    except (TypeError, ValueError):
        raise Refus(f"manifeste : betas invalides ({betas!r})")
    exacts = [i for i, v in enumerate(vals) if v == beta]
    if exacts:
        return exacts[0]
    proches = [i for i, v in enumerate(vals) if abs(v - beta) <= TOL]
    if len(proches) == 1:
        return proches[0]
    if len(proches) > 1:
        raise Refus(f"β = {beta!r} ambiguë : nœuds {[betas[i] for i in proches]} "
                    f"à moins de {TOL}")
    raise Refus(f"β = {beta!r} absente du manifeste (betas = {betas})")


def mode_bootstrap(man):
    if 'appariement' not in man:
        raise Refus("manifeste : champ « appariement » absent")
    v = man['appariement']
    if not isinstance(v, str):
        raise Refus("manifeste : « appariement » doit être une chaîne")
    if v == APPARIEMENT_CONJOINT:
        return 'conjoint'
    if v.startswith(PREFIXES_INDEPENDANT):
        return 'independant'
    raise Refus(f"manifeste : valeur d'appariement inconnue : {v!r}")


def plafonds(man):
    if 'plafonds_du_bain' not in man:
        return
    v = man['plafonds_du_bain']
    if v is None:
        return
    if isinstance(v, bool):
        if v:
            raise Refus("acquisition invalide : plafonds_du_bain non nul")
        return
    if isinstance(v, (int, float)):
        if v != 0:
            raise Refus("acquisition invalide : plafonds_du_bain non nul")
        return
    if isinstance(v, (list, dict, str)):
        if len(v) != 0:
            raise Refus("acquisition invalide : plafonds_du_bain non nul")
        return
    raise Refus("acquisition invalide : plafonds_du_bain non nul")


def _dimension(man, ser, cle):
    valeur = _entier_positif(man, cle, 'manifeste')
    if cle in ser:
        valeur_series = _entier_positif(ser, cle, 'manifeste : series')
        if valeur_series != valeur:
            raise Refus(f"dimensions contradictoires : series.{cle} = {valeur_series} ≠ {cle} = {valeur}")
    return valeur


def lire_series(dossier, man, b, beta):
    """Renvoie (series, infos) : series[a] = matrice (nmeas, R) des S_F du bras a."""
    ser = man['series']
    G = _entier_positif(ser, 'betas_par_groupe', 'manifeste : series')
    R = _dimension(man, ser, 'repliques')
    nmeas = _dimension(man, ser, 'nmeas')
    poids = _poids(man)
    K = len(poids)
    g = b // G
    k = b % G
    nbetas_groupe = min(G, len(man['betas']) - g * G)
    ncol = nbetas_groupe * R
    taille = nmeas * ncol * 8
    series = []
    for a in range(K):
        chemin = dossier / 'series' / f'bras{a}_g{g:03d}_fe.bin'
        if not chemin.is_file():
            raise Refus(f"fichier de série manquant : {chemin}")
        reel = chemin.stat().st_size
        if reel != taille:
            raise Refus(f"taille inattendue pour {chemin} : {reel} octets, attendu {taille} "
                        f"({nmeas} mesures × {ncol} chaînes × 8 octets)")
        raw = np.fromfile(chemin, dtype='<f8')
        if raw.size != nmeas * ncol:
            raise Refus(f"lecture incomplète de {chemin}")
        if not np.all(np.isfinite(raw)):
            raise Refus(f"valeur non finie dans {chemin}")
        mat = raw.reshape(nmeas, ncol)[:, k * R:(k + 1) * R]
        series.append(mat)
    infos = dict(K=K, R=R, nmeas=nmeas, groupe=g, rang_dans_groupe=k, poids=poids)
    return series, infos


def calculer(series, poids, beta, h, nboot, graine, maxit, mode):
    lo = min(float(s.min()) for s in series)
    hi = max(float(s.max()) for s in series)
    edges = np.arange(math.floor(lo / h) * h - h, math.ceil(hi / h) * h + 2 * h, h)
    centres = (edges[:-1] + edges[1:]) / 2
    C = np.array([[np.histogram(s[:, r], bins=edges)[0] for r in range(s.shape[1])]
                  for s in series])
    H = Histogramme(beta * np.asarray(poids, float), centres, C)
    f, ok = H.solve(maxit=maxit)
    logR = float(-f[-1])
    boots = []
    non_ok = 0
    if nboot > 0:
        rng = np.random.default_rng(graine)
        K, R = C.shape[0], C.shape[1]
        for _ in range(nboot):
            if mode == 'conjoint':
                idx = rng.integers(R, size=R)
                Cb = H.counts[:, idx, :]
            else:
                # Un tirage INDÉPENDANT par bras : chaque bras a son propre flux de
                # nombres aléatoires (graine dérivée de `graine`), sinon les deux bras
                # recevraient exactement les mêmes indices et le mode « indépendant »
                # se réduirait au mode conjoint (σ nulle sur des bras miroirs constants).
                Cb = np.stack([H.counts[a, rng.integers(R, size=R)] for a in range(K)])
            fb, okb = H.solve(Cb, maxit=maxit)
            if not okb:
                non_ok += 1
                continue
            boots.append(-(fb[-1] - fb[0]))
    sigma = float(np.std(np.asarray(boots, float), ddof=1)) if len(boots) >= 2 else None
    return dict(logR=logR, newton_ok=bool(ok), newton_iterations=int(H.it),
                nbins=int(len(centres)), n_bootstrap_ok=len(boots),
                n_bootstrap_non_converge=non_ok, sigma_bootstrap=sigma,
                bootstrap_mode=mode)


def main(argv=None):
    p = argparse.ArgumentParser(description="Lecteur direct des séries de torsion kvu.")
    p.add_argument('dossier')
    p.add_argument('--beta', type=float, required=True)
    p.add_argument('--bin-width', type=float, default=0.02)
    p.add_argument('--bootstrap', type=int, default=100)
    p.add_argument('--graine-bootstrap', type=int, default=17)
    p.add_argument('--maxit', type=int, default=2000)
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    try:
        if not (a.bin_width > 0) or not math.isfinite(a.bin_width):
            raise Refus(f"largeur de bin invalide : {a.bin_width}")
        if a.bootstrap < 0:
            raise Refus(f"nombre de bootstraps invalide : {a.bootstrap}")
        if a.maxit < 1:
            raise Refus(f"maxit invalide : {a.maxit}")
        dossier, man, sha = lire_manifeste(a.dossier)
        plafonds(man)
        b = indices_beta(man, a.beta)
        mode = mode_bootstrap(man)
        series, infos = lire_series(dossier, man, b, a.beta)
        res = calculer(series, infos['poids'], a.beta, a.bin_width, a.bootstrap,
                       a.graine_bootstrap, a.maxit, mode)
    except Refus as e:
        print(f"refus : {e}", file=sys.stderr)
        return 2
    sortie = {
        'logR': res['logR'],
        'newton_ok': res['newton_ok'],
        'newton_iterations': res['newton_iterations'],
        'bin_width': float(a.bin_width),
        'nbins': res['nbins'],
        'beta': float(a.beta),
        'indice_beta': int(b),
        'groupe': int(infos['groupe']),
        'rang_dans_groupe': int(infos['rang_dans_groupe']),
        'K': int(infos['K']),
        'R': int(infos['R']),
        'nmeas': int(infos['nmeas']),
        'poids': infos['poids'],
        'n_bootstrap_demandes': int(a.bootstrap),
        'n_bootstrap_ok': int(res['n_bootstrap_ok']),
        'n_bootstrap_non_converge': int(res['n_bootstrap_non_converge']),
        'sigma_bootstrap': res['sigma_bootstrap'],
        'bootstrap_mode': res['bootstrap_mode'],
        'graine_bootstrap': int(a.graine_bootstrap),
        'graine_base': man.get('graine_base'),
        'appariement': man.get('appariement'),
        'torsion': man.get('torsion'),
        'torsion_fixe': man.get('torsion_fixe', None),
        'dossier': str(dossier),
        'sha256_manifeste': sha,
    }
    try:
        texte = json.dumps(sortie, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        print(f"refus : valeur non finie en sortie ({e})", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).write_text(texte + '\n', encoding='utf-8')
    print(texte)
    return 0 if res['newton_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
