# SPDX-License-Identifier: Apache-2.0
"""Lecture et propagation des erreurs des trajets en couplage temporel."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from sys import float_info


APPARIEMENT = 'la graine de la chaîne (nœud k, réplique i) est G + k·R + i pour tous les bras'
MAX_GRAINE = 2**64 - 1
ESTIMATEURS = ('brut', 'lien-integre', 'y0', 'ys')
ENTETE = (
    'poids', 'beta', 'rep', 'graine', 'depart', 'n', 'tot', 'fe',
    'h1_tot', 'h1_fe', 'h2_tot', 'h2_fe', 'beta_s', 'st', 'ss',
)
COLONNES_LI = ('li_tot', 'li_t', 'li_s')
COLONNES_LI_SEPAREES = ('y0_t', 'ys_t', 'zs_s', 'y0_tot', 'ys_tot')
COLONNES_REELLES = tuple(cle for cle in ENTETE if cle not in ('rep', 'graine', 'depart', 'n'))


def objet(valeur, nom):
    if not isinstance(valeur, dict):
        raise ValueError(f'{nom} : objet JSON attendu')
    return valeur


def entier(valeur, nom, minimum=0):
    if type(valeur) is not int or valeur < minimum:
        raise ValueError(f'{nom} : entier ≥ {minimum} attendu')
    return valeur


def reel(valeur, nom):
    if type(valeur) not in (int, float) or not math.isfinite(valeur):
        raise ValueError(f'{nom} : nombre fini attendu')
    return float(valeur)


def champs_json(paires):
    champs = dict(paires)
    if len(champs) != len(paires):
        raise ValueError('nom de champ JSON en double')
    return champs


def json_objet(chemin):
    texte = Path(chemin).read_text(encoding='utf-8')
    return objet(json.loads(texte, object_pairs_hook=champs_json), chemin)


def grille(valeurs, nom, minimum=1):
    if not isinstance(valeurs, list) or len(valeurs) < minimum:
        raise ValueError(f'{nom} : liste d’au moins {minimum} nœud(s) attendue')
    nombres = tuple(reel(val, nom) for val in valeurs)
    if nombres[0] < 0 or any(a >= b for a, b in zip(nombres, nombres[1:])):
        raise ValueError(f'{nom} : couplages ≥ 0 strictement croissants attendus')
    return nombres


@dataclass(frozen=True)
class Mesure:
    chemin: Path
    cotes: tuple
    beta_s: float
    betas_t: tuple
    repliques: int
    nmeas: int
    graine_base: int
    depart: str
    therm: int
    skip: int
    nor: int
    precision: str
    source: str
    lien_integre: tuple


@dataclass(frozen=True)
class Noeud:
    beta_t: float
    identites: tuple
    forces: tuple


def colonnes_lien_integre(man, estimateur):
    if estimateur == 'brut':
        return ()
    if 'lien_integre' not in man:
        raise ValueError(f'estimateur {estimateur} : marqueur lien_integre requis')
    marqueur = objet(man['lien_integre'], 'lien_integre')
    description = marqueur.get('colonnes_csv')
    if isinstance(description, str):
        # Le producteur donne les noms avant le premier deux-points, puis les définitions.
        noms = tuple(nom.strip() for nom in description.partition(':')[0].split(','))
        for colonnes in (COLONNES_LI, COLONNES_LI_SEPAREES):
            if sorted(noms) == sorted(colonnes):
                if colonnes == COLONNES_LI and estimateur != 'lien-integre':
                    raise ValueError(f'estimateur {estimateur} indisponible dans le format historique li_t')
                return colonnes
    raise ValueError('lien_integre.colonnes_csv : schéma inconnu, incomplet ou ambigu')


def colonnes_temporelles(mesure, estimateur):
    if estimateur == 'brut':
        return ('st',)
    if mesure.lien_integre == COLONNES_LI:
        return ('li_t',)
    if estimateur == 'lien-integre':
        return ('y0_t', 'ys_t')
    return (f'{estimateur}_t',)


def lire_manifeste(dossier, estimateur):
    chemin = dossier / 'manifest_thermo.json'
    man = json_objet(chemin)
    if man.get('schema') != 'kvu-thermo-mesure/1':
        raise ValueError('schéma kvu-thermo-mesure/1 attendu')
    sep = objet(man.get('couplages_separes'), 'couplages_separes')
    if type(sep.get('direction_du_temps')) is not int or sep['direction_du_temps'] != 0:
        raise ValueError('couplages_separes.direction_du_temps doit valoir 0')
    beta_s = reel(sep.get('beta_s'), 'couplages_separes.beta_s')
    betas_t = grille(sep.get('betas_t'), 'couplages_separes.betas_t')
    if beta_s <= 0 or grille(man.get('betas'), 'betas') != betas_t:
        raise ValueError('couplages spatiaux/temporels contradictoires')
    if (
        man.get('torsion') != 'aucune'
        or man.get('torsion_fixe', 'aucune') != 'aucune'
        or man.get('couture') is not None
    ):
        raise ValueError('mesure périodique sans torsion attendue')
    poids = man.get('poids')
    if not isinstance(poids, list) or len(poids) != 1 or reel(poids[0], 'poids[0]') != 1:
        raise ValueError('un seul bras de poids 1 attendu')
    if entier(man.get('plafonds_du_bain'), 'plafonds_du_bain') != 0:
        raise ValueError('acquisition invalide : plafonds_du_bain non nul')
    if man.get('appariement') != APPARIEMENT:
        raise ValueError('convention kvu de graine par nœud et réplique attendue')

    cotes = man.get('cotes')
    if not isinstance(cotes, list) or len(cotes) != 4:
        raise ValueError('quatre côtés dans l’ordre temps, espace attendus')
    cotes = tuple(entier(cote, 'cotes', 1) for cote in cotes)
    repliques = entier(man.get('repliques'), 'repliques', 2)
    nmeas = entier(man.get('nmeas'), 'nmeas', 2)
    if nmeas % 2:
        raise ValueError('nmeas pair attendu')
    graine = entier(man.get('graine_base'), 'graine_base')
    if graine + len(betas_t) * repliques - 1 > MAX_GRAINE:
        raise ValueError('plage de graines hors u64')
    if 'series' in man:
        series = objet(man['series'], 'series')
        for cle, attendu in (('betas_par_groupe', 1), ('repliques', repliques), ('nmeas', nmeas)):
            if entier(series.get(cle), f'series.{cle}', 1) != attendu:
                raise ValueError(f'series.{cle} incompatible avec le manifeste anisotrope')
    depart = man.get('depart')
    if depart not in ('alterne', 'froid', 'chaud'):
        raise ValueError('depart inconnu')
    precision = man.get('precision')
    if precision not in ('f64', 'f32'):
        raise ValueError('precision f64 ou f32 attendue')
    source = man.get('sha256_source_assemblee')
    if source is None:
        source = ''
    if not isinstance(source, str) or (source and not re.fullmatch(r'[0-9a-f]{64}', source)):
        raise ValueError('sha256_source_assemblee invalide')
    return Mesure(
        chemin, cotes, beta_s, betas_t, repliques, nmeas, graine, depart,
        entier(man.get('therm'), 'therm'), entier(man.get('skip'), 'skip'),
        entier(man.get('nor'), 'nor'), precision, source,
        colonnes_lien_integre(man, estimateur),
    )


def lire_points(mesure, estimateur):
    dossier = mesure.chemin.parent / 'points'
    fichiers = [dossier / f'bras0_g{indice:03d}.csv' for indice in range(len(mesure.betas_t))]
    presents = set(dossier.glob('*.csv'))
    attendus = set(fichiers)
    if presents != attendus:
        manquent = sorted(p.name for p in attendus - presents)
        excedent = sorted(p.name for p in presents - attendus)
        raise ValueError(f'{dossier} : CSV manquants {manquent}, inattendus {excedent}')
    return tuple(
        lire_groupe(fichier, mesure, indice, estimateur) for indice, fichier in enumerate(fichiers)
    )


def lire_groupe(fichier, mesure, indice, estimateur):
    repliques = {}
    beta_t = mesure.betas_t[indice]
    suffixe = mesure.lien_integre
    temporelles = colonnes_temporelles(mesure, estimateur)
    with fichier.open(encoding='utf-8', newline='') as entree:
        lignes = csv.reader(entree)
        entete = next(lignes, [])
        noms = set(entete)
        if len(noms) != len(entete):
            raise ValueError(f'{fichier} : nom de colonne CSV en double')
        if suffixe and (noms & set(COLONNES_LI)) and (noms & set(COLONNES_LI_SEPAREES)):
            raise ValueError(f'{fichier} : schéma lien_integre ambigu dans l’entête CSV')
        manquent = set(ENTETE + suffixe) - noms
        if manquent:
            raise ValueError(f'{fichier} : colonnes CSV requises manquantes {sorted(manquent)}')
        for numero, champs in enumerate(lignes, 2):
            try:
                if len(champs) != len(entete):
                    raise ValueError('nombre de colonnes inattendu')
                ligne = dict(zip(entete, champs))
                rep, graine, mesures = (int(ligne[cle]) for cle in ('rep', 'graine', 'n'))
                vals = {cle: reel(float(ligne[cle]), cle) for cle in COLONNES_REELLES + suffixe}
                if (vals['poids'], vals['beta'], vals['beta_s']) != (1, beta_t, mesure.beta_s):
                    raise ValueError('poids ou couplages contredisant le manifeste')
                if rep in repliques or not 0 <= rep < mesure.repliques:
                    raise ValueError(f'identité de réplique répétée ou hors limites : {rep}')
                if graine != mesure.graine_base + indice * mesure.repliques + rep:
                    raise ValueError(f'graine incompatible avec le nœud et la réplique {rep}')
                depart = mesure.depart
                if depart == 'alterne':
                    depart = 'chaud' if rep % 2 else 'froid'
                if ligne['depart'] != depart or mesures != mesure.nmeas:
                    raise ValueError('départ ou nombre de mesures incompatible avec le manifeste')
                if vals['st'] != vals['fe'] or not math.isclose(
                    vals['tot'], vals['st'] + vals['ss'], rel_tol=1e-12, abs_tol=1e-10,
                ):
                    raise ValueError('colonnes extensives incompatibles : st != fe ou tot != st + ss')
                if suffixe == COLONNES_LI_SEPAREES:
                    if beta_t == 0 and vals['y0_t'] != 0:
                        raise ValueError('y0_t doit être exactement nul à beta_t = 0')
                    for temporal, total in (('y0_t', 'y0_tot'), ('ys_t', 'ys_tot')):
                        t, s, somme = vals[temporal], vals['zs_s'], vals[total]
                        # Le producteur additionne les moyennes F64, même en mode champ F32.
                        # L'échelle des termes reste pertinente quand leur somme s'annule.
                        tolerance = 8 * float_info.epsilon * max(abs(t), abs(s), abs(somme))
                        if not math.isclose(somme, t + s, rel_tol=0, abs_tol=tolerance):
                            raise ValueError(f'{total} incohérent avec {temporal} + zs_s')
                action = vals[temporelles[0]]
                if len(temporelles) == 2:
                    action = action * 0.5 + vals[temporelles[1]] * 0.5
                repliques[rep] = (graine, action)
            except ValueError as erreur:
                raise ValueError(f'{fichier}:{numero} : {erreur}') from erreur
    if len(repliques) != mesure.repliques:
        raise ValueError(f'{fichier} : {mesure.repliques} répliques distinctes attendues')
    return tuple(repliques[i] for i in range(mesure.repliques))


def lire_paire(court, long, estimateur):
    mesures = []
    for dossier in (court, long):
        try:
            mesures.append(lire_manifeste(Path(dossier), estimateur))
        except ValueError as erreur:
            raise ValueError(f'{dossier}/manifest_thermo.json : {erreur}') from erreur
    a, b = mesures
    duree, *espace = a.cotes
    if b.cotes != (2 * duree, *espace):
        raise ValueError(f'géométrie A/B incompatible : {a.cotes} et {b.cotes}')
    for cle in (
        'beta_s', 'betas_t', 'repliques', 'nmeas', 'graine_base', 'depart',
        'therm', 'skip', 'nor', 'precision',
    ):
        if getattr(a, cle) != getattr(b, cle):
            raise ValueError(f'métadonnées A/B incompatibles : {cle}')
    if a.source and b.source and a.source != b.source:
        raise ValueError('métadonnées A/B incompatibles : sha256_source_assemblee')
    if a.lien_integre != b.lien_integre:
        raise ValueError('schémas lien_integre différents entre les bras A/B')
    noeuds = []
    for beta_t, aa, bb in zip(a.betas_t, lire_points(a, estimateur), lire_points(b, estimateur)):
        identites = tuple((rep, graine) for rep, (graine, _) in enumerate(aa))
        forces = tuple(math.fsum((2 * action_a, -action_b)) for (_, action_a), (_, action_b) in zip(aa, bb))
        noeuds.append(Noeud(beta_t, identites, forces))
    return a, b, noeuds


def poids_trapezes(noeuds):
    pas = [(b.beta_t - a.beta_t) / 2 for a, b in zip(noeuds, noeuds[1:])]
    return tuple(a + b for a, b in zip((0, *pas), (*pas, 0)))


def erreur_type(valeurs):
    return statistics.stdev(valeurs) / math.sqrt(len(valeurs))


def estimer(termes, independance=None):
    """Estime la somme pondérée. Les covariances entre blocs restent inconnues."""
    blocs = {}
    for poids, noeud in termes:
        blocs.setdefault(noeud.identites, []).append((poids, noeud.forces))
    erreurs = []
    graines = set()
    for identites, bloc in blocs.items():
        if independance is not None:
            nouvelles = {graine for _, graine in identites}
            if graines & nouvelles:
                raise ValueError('indépendance incompatible avec les graines partagées entre blocs')
            graines.update(nouvelles)
        scores = [
            math.fsum(poids * forces[rep] for poids, forces in bloc)
            for rep in range(len(identites))
        ]
        erreurs.append(erreur_type(scores))
    sigma = math.fsum(erreurs)
    mode = 'covariance empirique' if len(erreurs) == 1 else 'borne triangulaire'
    if independance is not None and len(erreurs) > 1:
        sigma = math.hypot(*erreurs)
        mode = 'indépendance déclarée entre blocs de graines disjoints'
    return {
        'integrale': math.fsum(poids * statistics.fmean(noeud.forces) for poids, noeud in termes),
        'erreur_type': sigma,
        'covariance': mode,
        'erreurs_type_blocs': erreurs,
        'justification_independance': independance,
    }
