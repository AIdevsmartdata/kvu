# SPDX-License-Identifier: Apache-2.0
"""Planifie et lit hors ligne un pas de transport à longueur physique commune."""

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

# Le dossier du script est absent de sys.path avec python3 -I.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transport_mesures import (
    ESTIMATEURS, MAX_GRAINE, colonnes_temporelles, entier, erreur_type, estimer, grille,
    json_objet, lire_paire, objet, poids_trapezes, reel,
)


TAILLES = (4, 5, 6, 8)
VERSION_PLAN = 'kvu-transport/1'
SOURCE_ECHELLE = 'LTW03 hep-lat/0307017, Table 1'
# Colonnes beta, a*sqrt(sigma), erreur publiée. Même convention que la note
# TRAJET_APPARIE ; les erreurs d'échelle ne sont pas propagées ici.
TABLE_ECHELLE = (
    (1.8800, 0.773, 0.016),
    (2.1768, 0.5149, 0.0077),
    (2.2986, 0.3667, 0.0018),
    (2.3726, 0.2879, 0.0010),
    (2.4265, 0.2388, 0.0009),
    (2.5115, 0.17680, 0.00076),
)


def beta_isotrope(taille, ell):
    cible = ell / taille
    for gauche, droite in zip(TABLE_ECHELLE, TABLE_ECHELLE[1:]):
        beta_0, a_0, _ = gauche
        beta_1, a_1, _ = droite
        if a_1 <= cible <= a_0:
            fraction = math.log(cible / a_0) / math.log(a_1 / a_0)
            return {
                'a_sqrt_sigma_cible': cible,
                'beta_isotrope': beta_0 + fraction * (beta_1 - beta_0),
                'encadrement_beta': [beta_0, beta_1],
                'lignes_ltw03': [gauche, droite],
            }
    raise ValueError(f'ℓ={ell} hors de la table LTW03 pour n={taille} (extrapolation refusée)')


def section_id(taille, duree):
    return f'n{taille}_t{duree}'


def geometrie(taille, duree):
    espace = (2 * taille,) * 3
    return (duree, *espace), (2 * duree, *espace)


def lire_grilles(arguments, ell):
    grilles = {}
    for argument in arguments:
        taille, separateur, noeuds = argument.partition(':')
        if not separateur or taille not in {str(n) for n in TAILLES}:
            raise ValueError('grille attendue sous la forme N:b0,b1,… avec N parmi 4,5,6,8')
        taille = int(taille)
        if taille in grilles:
            raise ValueError(f'grille répétée pour n={taille}')
        valeurs = grille([float(noeud) for noeud in noeuds.split(',')], argument)
        beta = beta_isotrope(taille, ell)['beta_isotrope']
        if valeurs[-1] > beta:
            raise ValueError(f'grille de n={taille} au-delà de son nœud isotrope {beta!r}')
        if valeurs[-1] < beta:
            valeurs += (beta,)
        if len(valeurs) < 2:
            raise ValueError(f'grille de n={taille} : au moins deux nœuds pour intégrer')
        grilles[taille] = valeurs
    if len(grilles) < 2:
        raise ValueError('au moins deux tailles avec leur grille sont requises')
    return grilles


def arguments_mesure(options, cotes_a, cotes_b, beta, noeuds, graine, destination):
    commande = [
        options.kvu, 'su2', 'thermo', 'mesure',
        '--cotes', ','.join(map(str, cotes_a)), '--cotes-b', ','.join(map(str, cotes_b)),
        '--beta-s', repr(beta), '--betas', ','.join(map(repr, noeuds)),
        '--repliques', str(options.repliques), '--therm', str(options.therm),
        '--nmeas', str(options.nmeas), '--skip', str(options.skip), '--nor', str(options.nor),
        '--graine', str(graine), '--depart', options.depart,
        '--backend', options.backend, '--precision', options.precision,
        '--torsion', 'aucune', '--poids', '1', '--sortie', str(destination),
    ]
    if any(cote % 2 for cote in cotes_a + cotes_b):
        commande += ['--impairs', 'oui']
    if options.estimateur != 'brut':
        commande.append('--lien-integre')
    return commande


def planifier(options):
    ell = reel(options.ell, 'longueur ℓ')
    grilles = lire_grilles(options.grille, ell)
    tailles = sorted(grilles)
    sections = {(n, n) for n in tailles}
    pas = []
    for n, m in zip(tailles, tailles[1:]):
        sections.add((n, m))
        pas.append({
            'de': n, 'vers': m,
            'E': {'depart': section_id(n, n), 'arrivee': section_id(m, m)},
            'D': {'depart': section_id(n, m), 'arrivee': section_id(m, m)},
        })
    entier(options.repliques, '--repliques', 2)
    if entier(options.nmeas, '--nmeas', 2) % 2:
        raise ValueError('--nmeas doit être pair')
    for cle in ('therm', 'skip', 'nor'):
        entier(getattr(options, cle), f'--{cle}')
    graine = entier(options.graine, '--graine')
    racine = options.sortie.resolve()
    acquisitions = {}
    for n, duree in sorted(sections):
        cle = section_id(n, duree)
        noeuds = grilles[n]
        beta = noeuds[-1]
        tranche = len(noeuds) * options.repliques
        if graine + tranche - 1 > MAX_GRAINE:
            raise ValueError('plage de graines hors u64')
        cotes_a, cotes_b = geometrie(n, duree)
        argv = arguments_mesure(options, cotes_a, cotes_b, beta, noeuds, graine, racine / cle)
        acquisitions[cle] = {
            'taille': n, 'duree_courte': duree, 'beta_s': beta, 'betas_t': noeuds,
            'graine_base': graine, 'argv': argv,
            'bras': {
                'A': {'cotes': cotes_a, 'dossier': f'{cle}/A'},
                'B': {'cotes': cotes_b, 'dossier': f'{cle}/B'},
            },
        }
        graine += tranche
    return {
        'schema': VERSION_PLAN, 'ell': ell, 'racine': str(racine), 'direction_du_temps': 0,
        'estimateur': options.estimateur,
        'echelle': {
            'source': SOURCE_ECHELLE, 'interpolation': 'log-linéaire en a*sqrt(sigma)',
            'colonnes': ['beta', 'a_sqrt_sigma', 'erreur_publiee'],
            'tailles': {str(n): beta_isotrope(n, ell) for n in tailles},
        },
        'acquisitions': acquisitions, 'pas': pas,
    }


def rapport_noeuds(a, b, noeuds, estimateur):
    facteur = 6 * math.prod(a.cotes)
    temporelles = colonnes_temporelles(a, estimateur)
    colonne = temporelles[0] if len(temporelles) == 1 else f"({' + '.join(temporelles)}) / 2"
    return {
        'A': str(a.chemin), 'B': str(b.chemin), 'cotes_A': a.cotes, 'cotes_B': b.cotes,
        'estimateur': estimateur, 'colonne_temporelle': colonne,
        'beta_s': a.beta_s, 'repliques': a.repliques, 'nmeas': a.nmeas, 'precision': a.precision,
        'graine_base': a.graine_base,
        'sha256_source_assemblee': {'A': a.source or None, 'B': b.source or None},
        'normalisation_densites': facteur,
        'noeuds': [
            {
                'beta_t': noeud.beta_t, 'force': statistics.fmean(noeud.forces),
                'erreur_type': erreur_type(noeud.forces),
                'difference_densites': statistics.fmean(noeud.forces) / facteur,
                'graines': [noeud.identites[0][1], noeud.identites[-1][1]],
            }
            for noeud in noeuds
        ],
    }


def lire_section(plan, racine, ell, taille, duree, estimateur, independance):
    cle = section_id(taille, duree)
    acquisition = objet(plan.get('acquisitions'), 'acquisitions').get(cle)
    acquisition = objet(acquisition, f'acquisitions.{cle}')
    attendue = grille(acquisition.get('betas_t'), f'{cle}.betas_t', 2)
    beta = beta_isotrope(taille, ell)['beta_isotrope']
    if reel(acquisition.get('beta_s'), f'{cle}.beta_s') != beta or attendue[-1] != beta:
        raise ValueError(f'{cle} : le plan doit finir au point isotrope LTW03 {beta!r}')
    a, b, noeuds = lire_paire(racine / cle / 'A', racine / cle / 'B', estimateur)
    if (a.cotes, b.cotes) != geometrie(taille, duree):
        raise ValueError(f'{cle} : géométrie mesurée incompatible avec le pas demandé')
    if a.beta_s != beta:
        raise ValueError(f'{cle} : beta_s mesuré incompatible avec l’endpoint isotrope {beta!r}')
    if a.betas_t != attendue:
        raise ValueError(f'{cle} : grille mesurée différente du plan complet jusqu’au point isotrope')
    poids = poids_trapezes(noeuds)
    termes = list(zip(poids, noeuds))
    rapport = rapport_noeuds(a, b, noeuds, estimateur)
    rapport.update({
        'section': cle, 'taille': taille, 'duree_courte': duree,
        'debut_beta_t': attendue[0], 'fin_beta_t': attendue[-1],
        'poids_trapezes': poids, 'trapezes': estimer(termes, independance),
    })
    return rapport, termes


def lire_ancre(ancres, section):
    cle = section['section']
    if cle not in ancres:
        return None
    ancre = objet(ancres[cle], cle)
    intervalle = ancre.get('intervalle')
    provenance = ancre.get('source')
    if not isinstance(intervalle, list) or len(intervalle) != 2:
        raise ValueError(f'{cle} : ancre attendue sous la forme [min, max]')
    lo, hi = (reel(x, f'{cle}.intervalle') for x in intervalle)
    if lo > hi or not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(f'{cle} : borne ou provenance de l’ancre invalide')
    beta_s = reel(ancre.get('beta_s'), f'{cle}.beta_s')
    beta_t = reel(ancre.get('beta_t'), f'{cle}.beta_t')
    if (beta_s, beta_t) != (section['beta_s'], section['debut_beta_t']):
        raise ValueError(f'{cle} : couplages de l’ancre différents du début du trajet')
    return {'intervalle': [lo, hi], 'source': provenance}


def comparer(options):
    plan = json_objet(options.plan)
    if plan.get('schema') != VERSION_PLAN or entier(plan.get('direction_du_temps'), 'axe du plan') != 0:
        raise ValueError('plan de transport axe temporel 0 attendu')
    ell = reel(plan.get('ell'), 'longueur ℓ')
    n, m = options.de, options.vers
    pas = plan.get('pas')
    if n >= m or not isinstance(pas, list) or not any(
        isinstance(p, dict) and (p.get('de'), p.get('vers')) == (n, m) for p in pas
    ):
        raise ValueError(f'pas {n}→{m} absent du plan')
    racine_plan = plan.get('racine')
    if not isinstance(racine_plan, str) or not racine_plan:
        raise ValueError('racine du plan absente')
    racine = (options.racine or Path(racine_plan)).resolve()
    justification = options.justification_independance
    if justification is not None and not justification.strip():
        raise ValueError('une justification non vide est requise pour supposer l’indépendance')
    depart, termes_depart = lire_section(
        plan, racine, ell, n, n if options.type == 'E' else m, options.estimateur, justification,
    )
    arrivee, termes_arrivee = lire_section(plan, racine, ell, m, m, options.estimateur, justification)
    if depart['precision'] != arrivee['precision']:
        raise ValueError('métadonnées incompatibles entre sections : precision')
    sources = {
        source for rapport in (depart, arrivee)
        for source in rapport['sha256_source_assemblee'].values() if source is not None
    }
    if len(sources) > 1:
        raise ValueError('métadonnées incompatibles entre sections : sha256_source_assemblee')
    termes = [(-poids, noeud) for poids, noeud in termes_depart] + termes_arrivee
    ancres = json_objet(options.ancres) if options.ancres else {}
    depart['ancre'] = lire_ancre(ancres, depart)
    arrivee['ancre'] = lire_ancre(ancres, arrivee)
    offset = None
    if depart['ancre'] is not None and arrivee['ancre'] is not None:
        lo_d, hi_d = depart['ancre']['intervalle']
        lo_a, hi_a = arrivee['ancre']['intervalle']
        offset = [lo_a - hi_d, hi_a - lo_d]
    return {
        'type': options.type, 'sens': 'arrivée moins départ', 'ell': ell,
        'estimateur': options.estimateur,
        'depart': depart, 'arrivee': arrivee, 'difference_trapezes': estimer(termes, justification),
        'difference_ancres': offset,
        'biais_a_controler': {'quadrature': None, 'echelle': None, 'precision_et_melange': None},
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commandes = cli.add_subparsers(dest='action', required=True)
    plan = commandes.add_parser('planifier', help='émettre un plan et les arguments kvu')
    plan.add_argument('--ell', type=float, required=True)
    plan.add_argument('--grille', action='append', required=True, metavar='N:B0,B1,…')
    plan.add_argument('--sortie', type=Path, required=True, help='racine des futures acquisitions')
    plan.add_argument('--kvu', default='kvu')
    plan.add_argument('--backend', choices=('hip', 'cuda'), required=True)
    plan.add_argument('--precision', choices=('f64', 'f32'), default='f64')
    plan.add_argument('--depart', choices=('alterne', 'froid', 'chaud'), default='alterne')
    plan.add_argument('--repliques', type=int, required=True)
    plan.add_argument('--nmeas', type=int, required=True)
    plan.add_argument('--therm', type=int, required=True)
    plan.add_argument('--skip', type=int, default=1)
    plan.add_argument('--nor', type=int, default=2)
    plan.add_argument('--graine', type=int, required=True)
    plan.add_argument('--commandes', action='store_true', help='imprimer les commandes POSIX au lieu du JSON')

    pilotes = commandes.add_parser('noeuds', help='lire les forces sur une grille même partielle')
    pilotes.add_argument('A', type=Path, help='dossier court')
    pilotes.add_argument('B', type=Path, help='dossier long')

    lecture = commandes.add_parser('lire', help='calculer le pas signé à partir du plan complet')
    lecture.add_argument('plan', type=Path)
    lecture.add_argument('--de', type=int, choices=TAILLES, required=True)
    lecture.add_argument('--vers', type=int, choices=TAILLES, required=True)
    lecture.add_argument('--type', choices=('E', 'D'), required=True)
    lecture.add_argument('--racine', type=Path, help='nouvel emplacement de la racine du plan')
    lecture.add_argument('--ancres', type=Path, help='bornes externes et leur provenance (JSON)')
    lecture.add_argument('--justification-independance', metavar='TEXTE')
    for sous_commande in (plan, pilotes, lecture):
        sous_commande.add_argument('--estimateur', choices=ESTIMATEURS, default='brut')
    options = cli.parse_args(argv)
    try:
        if options.action == 'planifier':
            sortie = planifier(options)
            if options.commandes:
                for acquisition in sortie['acquisitions'].values():
                    print(shlex.join(acquisition['argv']))
                return 0
        elif options.action == 'noeuds':
            sortie = rapport_noeuds(
                *lire_paire(options.A, options.B, options.estimateur), estimateur=options.estimateur,
            )
        else:
            sortie = comparer(options)
        print(json.dumps(sortie, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, OverflowError, csv.Error) as erreur:
        print(f'refus : {erreur}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
