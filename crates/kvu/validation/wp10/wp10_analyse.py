# SPDX-License-Identifier: Apache-2.0
"""Analyse de campagne WP10 : LOI-2, recertification et domaine RG."""

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import json
import math

from wp10_stat import combiner_chaines, lire_csv, stat_serie, z_score
from wp10_post import Parseur, ErreurUsage, denominateur_valide, ecrire_json, entier, lire_json

OBSERVABLES = ('P', 'Pb1', 'Pb2')
GROUPES = {'P': 'F32', 'C': 'F64', 'R': 'acemagic', 'M': 'mutant', 'N': 'mutant_faible'}


def nom_cellule(tache):
    return tache['cellule'] if 'cellule' in tache else f"L{tache['L']}_b{int(tache['beta']):02d}"


def conclusive(stat):
    return stat['converge'] and stat['n_eff'] >= 20


def machine_chaine(rep):
    try:
        machine = lire_json(rep / 'manifest.json').get('machine')
        if isinstance(machine, dict):
            machine = machine.get('nom')
        if isinstance(machine, str):
            machine = machine.strip().lower()
            if 'acemagic' in machine:
                return 'acemagic'
            if 'gamer' in machine:
                return 'gamer'
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return 'inconnue'


def comparaison(gauche, droite):
    z = z_score(gauche['moy'], gauche['err'], droite['moy'], droite['err'])
    conclusif = conclusive(gauche) and conclusive(droite)
    return dict(z=z, ok=abs(z) < 3 if conclusif else None,
                verdict=('PASS' if abs(z) < 3 else 'FAIL') if conclusif else 'SKIP',
                gauche=gauche, droite=droite)


def raisons_inconclusives(bilan, gauche, droite):
    return '; '.join(f"{nom}, converge={s['converge']}, n_eff={s['n_eff']:.6g}"
                     for nom, s in ((gauche, bilan['gauche']), (droite, bilan['droite']))
                     if not conclusive(s))


def verifier_completude(rep, tache, exiger_bloc3b):
    """Un défaut donne son propre diagnostic, indépendamment de l'analyse."""
    erreurs = []
    for nom in ('COMPLETE', 'regen_bloc3.json', 'H13_PASS') + (('regen_bloc3b.json',) if exiger_bloc3b else ()):
        if not (rep / nom).is_file():
            erreurs.append(f'{nom} absent')
    n = entier(tache['nmeas'])
    for nom, cle, attendus in (
            ('measurements.csv', 'measurement_index', list(range(n + 1))),
            ('bloc3.csv', 'pas', None),
            *((('bloc3b.csv', 'pas', None),) if exiger_bloc3b else ())):
        try:
            lignes = lire_csv(rep / nom)
            nombre = n + 1 if nom == 'measurements.csv' else n
            if len(lignes) != nombre:
                erreurs.append(f'{nom} : {len(lignes)}/{nombre} lignes')
            indices = [entier(r[cle]) for r in lignes]
            if any(a >= b for a, b in zip(indices, indices[1:])):
                erreurs.append(f'{nom} : indices non uniques ou non croissants')
            if attendus is None and 'therm' in tache and 'skip' in tache:
                attendus = [tache['therm'] + m*tache['skip'] for m in range(1, n + 1)]
            if attendus is not None and indices != attendus:
                erreurs.append(f'{nom} : indices incomplets ou incorrects')
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            erreurs.append(f'{nom} : {exc}')
    return erreurs


def texte_comparaison(bilan, gauche, droite):
    a, b = bilan['gauche'], bilan['droite']
    return (f"z={bilan['z']:.6g} ({gauche} {a['moy']:.6g}±{a['err']:.6g}, "
            f"{droite} {b['moy']:.6g}±{b['err']:.6g})")


def analyse(racine, taches, exiger_bloc3b=False):
    """Rendre toutes les valeurs à pleine précision et les verdicts en ordre stable."""
    resultat = dict(chaines={}, cellules={}, loi2=[], mutants_loi2=[], recert=[],
                    mutants_recert=[], mutants_faibles=[], mde=[], equivalence=[],
                    completude=[], domaine=[], tau=[], absents=[], verdicts=[])
    verdicts = resultat['verdicts']

    def ligne(niveau, gate, detail):
        verdicts.append(f'{niveau} {gate} {detail}')

    def verdict(ok, gate, detail):
        ligne('PASS' if ok else 'FAIL', gate, detail)

    def rapport_comparaison(bilan, gate, cellule, obs, gauche, droite):
        detail = f'{cellule} {obs} '
        if bilan['verdict'] == 'SKIP':
            raison = raisons_inconclusives(bilan, gauche, droite)
            if bilan.get('absence'):
                raison = '; '.join(filter(None, (raison, 'chaîne prévue absente')))
            detail += f'INCONCLUANT ({raison}) '
        ligne(bilan['verdict'], gate, detail + texte_comparaison(bilan, gauche, droite))
        a, b = bilan['gauche'], bilan['droite']
        sigma = math.hypot(a['err'], b['err'])
        delta = abs(a['moy'] - b['moy'])
        marge = 2.1e-6 * abs(b['moy'])
        equivalent = delta + 1.645 * sigma < marge
        nom = f"{'LOI2' if gate == 'WP10-LOI2' else 'RECERT'} {cellule}"
        resultat['equivalence'].append(dict(comparaison=nom, observable=obs, delta=delta,
                                             seuil=marge, sigma_diff=sigma, equivalent=equivalent))
        ligne('INFO', 'WP10-EQUIV', f"{nom} {obs} |Δ|={delta:.6g} δ={marge:.6g} "
              f"equivalent={'oui' if equivalent else 'non'}")
        if gate == 'WP10-LOI2':
            mde = dict(cellule=cellule, observable=obs, sigma_diff=sigma,
                       MDE50=3*sigma, MDE90=4.282*sigma)
            resultat['mde'].append(mde)
            ligne('INFO', 'WP10-LOI2-MDE', f"{cellule} {obs} sigma_diff={sigma:.6g} "
                  f"MDE50=3σ={mde['MDE50']:.6g} MDE90=4.282σ={mde['MDE90']:.6g}")

    for tache in sorted(taches, key=lambda t: t['id']):
        ident = tache['id']
        rep = racine / ident
        defauts = verifier_completude(rep, tache, exiger_bloc3b)
        resultat['completude'].append(dict(id=ident, ok=not defauts, erreurs=defauts))
        for defaut in defauts:
            ligne('FAIL', 'WP10-COMPLETUDE', f'{ident} {defaut}')
        if not (rep / 'COMPLETE').is_file():
            resultat['absents'].append(ident)
            ligne('INFO', 'WP10-ABSENT', ident)
            continue
        try:
            cellule = nom_cellule(tache)
            mesures = lire_csv(rep / 'measurements.csv')
            echantillons = [r for r in mesures if r['phase'] == 'sample' and r['measurement_index'] > 0]
            nmeas = tache.get('nmeas', len(echantillons))
            if [r['measurement_index'] for r in echantillons] != list(range(1, nmeas + 1)):
                raise ValueError('indices des mesures incomplets ou désordonnés')
            series = {'P': [r['plaquette'] for r in echantillons]}
            blocs = {}
            for obs, fichier in (('Pb1', 'bloc3.csv'), ('Pb2', 'bloc3b.csv')):
                if (rep / fichier).is_file():
                    lignes = lire_csv(rep / fichier)
                    if len(lignes) != nmeas:
                        raise ValueError(f'{fichier} : nombre de mesures incorrect')
                    if 'therm' in tache and 'skip' in tache:
                        if [entier(r['pas']) for r in lignes] != [tache['therm'] + m*tache['skip'] for m in range(1, nmeas+1)]:
                            raise ValueError(f'{fichier} : pas incorrects')
                    blocs[obs] = lignes
                    series[obs] = [r['plaq_grossiere'] for r in lignes]
                elif obs == 'Pb2' and exiger_bloc3b:
                    raise ValueError('bloc3b.csv absent malgré --bloc3b')
            if any(not valeurs or not all(math.isfinite(v) for v in valeurs) for valeurs in series.values()):
                raise ValueError('série vide ou non finie')
            genre = tache['genre']
            if genre not in GROUPES:
                raise ValueError(f'genre inconnu : {genre}')
            resultat['chaines'][ident] = dict(id=ident, cellule=cellule, genre=genre,
                                              machine=machine_chaine(rep),
                                              statistiques={o: stat_serie(x) for o, x in series.items()}, blocs=blocs)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            ligne('FAIL', 'WP10-DONNEES', f'{ident} {exc}')

    completes = sum(c['ok'] for c in resultat['completude'])
    verdict(completes == len(taches), 'WP10-COMPLETUDE', f'TOUT {completes}/{len(taches)} tâches')
    chaines = resultat['chaines']
    cellules = sorted({nom_cellule(t) for t in taches})
    comparaisons_manquantes = False
    for cellule in cellules:
        membres = [c for c in chaines.values() if c['cellule'] == cellule]
        prevues = [t for t in taches if nom_cellule(t) == cellule]
        genres = {genre: [c for c in membres if c['genre'] == genre] for genre in GROUPES}
        groupes = {}
        for genre, nom in GROUPES.items():
            groupes[nom] = {}
            for obs in OBSERVABLES:
                liste = genres[genre]
                # Une moyenne de cellule n'utilise jamais un sous-ensemble silencieux.
                if liste and all(obs in c['statistiques'] for c in liste):
                    try:
                        groupes[nom][obs] = combiner_chaines([c['statistiques'][obs] for c in liste])
                    except ValueError as exc:
                        ligne('FAIL', 'WP10-DONNEES', f'{cellule} {nom} {obs} {exc}')
        resultat['cellules'][cellule] = dict(groupes=groupes, chaines=[c['id'] for c in membres],
                                            machines_inconnues=sum(c['machine'] == 'inconnue' for c in membres))
        attendue = all(any(t['genre'] == g for t in prevues) for g in ('P', 'C'))
        obs_attendues = ('P', 'Pb1') + (('Pb2',) if exiger_bloc3b or any('Pb2' in c['statistiques'] for c in membres) else ())
        groupes_complets = all(t['id'] in chaines for t in prevues if t['genre'] in ('P', 'C'))
        for obs in obs_attendues if attendue else ():
            if obs in groupes['F32'] and obs in groupes['F64']:
                bilan = comparaison(groupes['F32'][obs], groupes['F64'][obs])
                if not groupes_complets:
                    bilan.update(ok=None, verdict='SKIP', absence=True)
                resultat['loi2'].append(dict(cellule=cellule, observable=obs, **bilan))
                rapport_comparaison(bilan, 'WP10-LOI2', cellule, obs, 'F32', 'F64')
                if not groupes_complets:
                    comparaisons_manquantes = True
            else:
                comparaisons_manquantes = True
                ligne('SKIP', 'WP10-LOI2', f'{cellule} {obs} INCONCLUANT (F32 ou F64 absent) z=nan')
        if genres['M']:
            for obs in OBSERVABLES:
                if obs in groupes['mutant'] and obs in groupes['F64']:
                    bilan = comparaison(groupes['mutant'][obs], groupes['F64'][obs])
                    rouge = abs(bilan['z']) > 3
                    if obs == 'P':
                        resultat['mutants_loi2'].append(dict(cellule=cellule, z=bilan['z'], ok=rouge))
                        verdict(rouge, 'WP10-LOI2-MUT', f"{cellule} P |z_mut|={abs(bilan['z']):.6g} "
                                + ('ROUGE' if rouge else 'ECHEC_OUTIL (gate sans puissance)'))
                    else:
                        ligne('INFO', 'WP10-LOI2-MUT', f"{cellule} {obs} |z_mut|={abs(bilan['z']):.6g}")
        if 'P' in groupes['mutant_faible'] and 'P' in groupes['F64']:
            bilan = comparaison(groupes['mutant_faible']['P'], groupes['F64']['P'])
            z = abs(bilan['z'])
            puissance = .5 * (1 + math.erf((z - 3) / math.sqrt(2)))
            resultat['mutants_faibles'].append(dict(cellule=cellule, z=bilan['z'], puissance=puissance))
            ligne('INFO', 'WP10-LOI2-MUT-FAIBLE', f'{cellule} P |z_mut|={z:.6g} '
                  f'(attendu ≈ 5,6 ; puissance calculée = Φ(|z| − 3)={puissance:.6g})')
        elif genres['N']:
            ligne('INFO', 'WP10-LOI2-MUT-FAIBLE', f'{cellule} P contrôle F64 absent')

        mde_p = next((m for m in resultat['mde'] if m['cellule'] == cellule and m['observable'] == 'P'), None)
        if genres['M'] and 'P' in groupes['mutant'] and 'P' in groupes['F64'] and mde_p:
            protocoles = {(t['beta'], t.get('beta_run', t['beta'])) for t in prevues if t['genre'] == 'M'}
            if len(protocoles) == 1:
                beta, beta_run = next(iter(protocoles))
                if beta_run != beta and beta != 0:
                    pente = (groupes['mutant']['P']['moy'] - groupes['F64']['P']['moy']) / (beta_run - beta)
                    relatif = mde_p['MDE90'] / (abs(pente)*beta) if pente else math.inf
                    mde_p.update(pente=pente, decalage_relatif_beta=relatif)
                    ligne('INFO', 'WP10-LOI2-MDE', f'{cellule} P pente={pente:.6g} '
                          f'décalage relatif équivalent de β={relatif:.6g}')
                else:
                    ligne('INFO', 'WP10-LOI2-MDE', f'{cellule} P pente indéfinie (beta_run = beta ou beta nul)')

        # La recertification concerne la production gamer et le témoin acemagic.
        recert_attendue = all(any(t['genre'] == g for t in prevues) for g in ('P', 'R'))
        if recert_attendue and (not genres['R'] or not genres['P']):
            for obs in obs_attendues:
                ligne('SKIP', 'WP10-RECERT', f'{cellule} {obs} INCONCLUANT (production ou témoin absent) z=nan')
        if genres['R'] and genres['P']:
            inconnues = [c['id'] for c in genres['P'] + genres['R'] if c['machine'] == 'inconnue']
            if inconnues:
                ligne('FAIL', 'WP10-RECERT', f"{cellule} machine inconnue : {','.join(inconnues)}")
            gamer = [c for c in genres['P'] if c['machine'] == 'gamer']
            temoins = [c for c in genres['R'] if c['machine'] == 'acemagic']
            if gamer and not temoins:
                ligne('FAIL', 'WP10-RECERT', f'{cellule} témoin acemagic absent')
            if gamer and temoins:
                for obs in obs_attendues:
                    if not all(obs in c['statistiques'] for c in gamer + temoins):
                        ligne('SKIP', 'WP10-RECERT', f'{cellule} {obs} INCONCLUANT (observable absente) z=nan')
                        continue
                    try:
                        a = combiner_chaines([c['statistiques'][obs] for c in gamer])
                        b = combiner_chaines([c['statistiques'][obs] for c in temoins])
                    except ValueError as exc:
                        ligne('FAIL', 'WP10-RECERT', f'{cellule} {obs} {exc}')
                        continue
                    bilan = comparaison(a, b)
                    if any(t['id'] not in chaines for t in prevues if t['genre'] in ('P', 'R')):
                        bilan.update(ok=None, verdict='SKIP', absence=True)
                    resultat['recert'].append(dict(cellule=cellule, observable=obs, **bilan))
                    rapport_comparaison(bilan, 'WP10-RECERT', cellule, obs, 'P (gamer)', 'R (acemagic)')
                    # Décalage vers l'extérieur : garantit cinq sigma supplémentaires
                    # même si la différence initiale est négative.
                    copies = copy.deepcopy([c['statistiques'][obs] for c in gamer])
                    sigma = math.hypot(a['err'], b['err'])
                    decalage = math.copysign(5 * sigma, a['moy'] - b['moy'])
                    for s in copies:
                        s['moy'] += decalage
                    mutant = comparaison(combiner_chaines(copies), b)
                    rouge = abs(mutant['z']) > 3
                    resultat['mutants_recert'].append(dict(cellule=cellule, observable=obs, z=mutant['z'], ok=rouge))
                    verdict(rouge, 'WP10-RECERT-MUT', f"{cellule} {obs} |z_mut|={abs(mutant['z']):.6g} "
                            + ('ROUGE' if rouge else 'ECHEC_OUTIL'))
            elif not inconnues and not gamer:
                ligne('INFO', 'WP10-RECERT', f'{cellule} production gamer absente')

        for pas, obs in ((1, 'Pb1'), (2, 'Pb2')):
            lignes = [r for c in genres['P'] for r in c['blocs'].get(obs, [])]
            if not lignes:
                ligne('INFO', 'WP10-DOMAINE', f'{cellule} pas{pas} absent')
                continue
            fractions = [r['fraction_hors'] for r in lignes]
            angles = [r['theta_max'] for r in lignes]
            ambigues = [r['n_ambigues'] for r in lignes]
            denominateurs_ok = all(denominateur_valide(r) for r in lignes)
            dans = sum(denominateur_valide(r) and r['fraction_hors'] <= 1e-3
                       and r['n_ambigues'] == 0 for r in lignes)
            complet = all(obs in c['blocs'] for c in genres['P'])
            etiquette = 'DANS_DOMAINE' if dans == len(lignes) and complet else 'HORS_DOMAINE'
            if not denominateurs_ok:
                etiquette = 'INCONNU'
            bilan = dict(cellule=cellule, pas=pas, etiquette=etiquette, dans=dans, total=len(lignes),
                         f_moyen=sum(fractions)/len(fractions), f_max=max(fractions),
                         theta_max=max(angles), ambigues=sum(ambigues), complet=complet,
                         denominateurs_ok=denominateurs_ok, couverture_99=dans/len(lignes) >= .99)
            resultat['domaine'].append(bilan)
            ligne('INFO', 'WP10-DOMAINE', f"{cellule} pas{pas} {etiquette} (configurations dans le domaine : "
                  f"{dans}/{len(lignes)} ; f moyen={bilan['f_moyen']:.6g}, f max={bilan['f_max']:.6g}, "
                  f"θ max={bilan['theta_max']:.6g}, ambiguës={bilan['ambigues']:.6g} ; "
                  f"couverture ≥ 99 % : {'oui' if bilan['couverture_99'] else 'non'})")
        for obs in OBSERVABLES:
            diagnostics = []
            for c in membres:
                if obs in c['statistiques']:
                    s = c['statistiques'][obs]
                    diagnostics.append(dict(id=c['id'], genre=c['genre'], tau=s['tau'], n=s['n'],
                                            fenetre=s['fenetre'], converge=s['converge'],
                                            n_eff=s['n_eff'], N_eff=s['n_eff']))
                    ligne('INFO', 'WP10-TAU', f"{cellule} {obs} {c['id']} τ={s['tau']:.6g} "
                          f"fenêtre={s['fenetre']} converge={s['converge']} n_eff={s['n_eff']:.6g} (unités de mesure)")
            effectif = sum(d['N_eff'] for d in diagnostics if d['genre'] == 'P')
            pauvre = effectif < 20
            resultat['tau'].append(dict(cellule=cellule, observable=obs, chaines=diagnostics,
                                        N_eff=effectif, STAT_PAUVRE=pauvre))
            details = '; '.join(f"{d['id']} τ={d['tau']:.6g} N_eff={d['N_eff']:.6g}" for d in diagnostics)
            ligne('INFO', 'WP10-TAU', f"{cellule} {obs} (unités de mesure) {details} ; "
                  f"N_eff cellule (P)={effectif:.6g}" + (' STAT_PAUVRE' if pauvre else ''))

    if not resultat['mutants_loi2']:
        ligne('SKIP', 'WP10-LOI2-MUT', 'TOUT mutant ou contrôle absent')
    mutant_manquant = (not resultat['mutants_loi2'] or
                       any(t['id'] not in chaines or 'P' not in resultat['cellules'][nom_cellule(t)]['groupes']['F64']
                           for t in taches if t['genre'] == 'M'))
    if any(v.startswith('FAIL ') for v in verdicts):
        global_loi2 = 'FAIL'
    elif (any(v.startswith('SKIP ') for v in verdicts) or comparaisons_manquantes
          or not resultat['loi2'] or mutant_manquant):
        global_loi2 = 'SKIP'
    else:
        global_loi2 = 'PASS'
    resultat['verdict_loi2'] = global_loi2
    ligne(global_loi2, 'WP10-LOI2', 'TOUT' + (' inconclusif ou absent' if global_loi2 == 'SKIP' else ''))
    resultat['ok'] = not any(v.startswith('FAIL ') for v in verdicts)
    return resultat


def json_fini(valeur):
    """Les statistiques indéfinies deviennent null dans un JSON strict."""
    if isinstance(valeur, float) and not math.isfinite(valeur):
        return None
    if isinstance(valeur, dict):
        return {k: json_fini(v) for k, v in valeur.items()}
    if isinstance(valeur, list):
        return [json_fini(v) for v in valeur]
    return valeur


def tables(resultat):
    lignes = ['| Chaîne | Observable | n | Moyenne | Erreur | τ | Fenêtre | Converge | n_eff |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for ident, chaine in resultat['chaines'].items():
        for obs, s in chaine['statistiques'].items():
            lignes.append(f"| {ident} | {obs} | {s['n']} | {s['moy']:.6g} | {s['err']:.6g} | "
                          f"{s['tau']:.6g} | {s['fenetre']} | {s['converge']} | {s['n_eff']:.6g} |")
    lignes += ['', '| Cellule | Groupe | Observable | n | Moyenne | Erreur |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for cellule, contenu in resultat['cellules'].items():
        for groupe, observations in contenu['groupes'].items():
            for obs, s in observations.items():
                lignes.append(f"| {cellule} | {groupe} | {obs} | {s['n']} | {s['moy']:.6g} | {s['err']:.6g} |")
    lignes += ['', '| Cellule | LOI-2 | z | Verdict |', '| --- | --- | ---: | --- |']
    for r in resultat['loi2']:
        lignes.append(f"| {r['cellule']} | {r['observable']} | {r['z']:.6g} | {r['verdict']} |")
    lignes += ['', '| Cellule | Pas | Domaine | Dans/total | f moyen | f max | θ max | Ambiguës |',
               '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in resultat['domaine']:
        lignes.append(f"| {r['cellule']} | {r['pas']} | {r['etiquette']} | {r['dans']}/{r['total']} | "
                      f"{r['f_moyen']:.6g} | {r['f_max']:.6g} | {r['theta_max']:.6g} | {r['ambigues']:.6g} |")
    lignes += ['', '| Cellule | Observable | Chaîne | τ (mesures) | Fenêtre | Converge | N_eff | Diagnostic cellule |',
               '| --- | --- | --- | ---: | ---: | --- | ---: | --- |']
    for r in resultat['tau']:
        for c in r['chaines']:
            lignes.append(f"| {r['cellule']} | {r['observable']} | {c['id']} | {c['tau']:.6g} | "
                          f"{c['fenetre']} | {c['converge']} | {c['N_eff']:.6g} | {'STAT_PAUVRE' if r['STAT_PAUVRE'] else ''} |")
    return '\n'.join(lignes) + '\n'


def main(argv=None):
    try:
        p = Parseur(description=__doc__)
        p.add_argument('--racine', type=Path, required=True)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--sortie', type=Path, required=True)
        p.add_argument('--json', action='store_true', help='analyse.json est toujours produit')
        p.add_argument('--bloc3b', action='store_true', help='exiger le deuxième pas pour les tâches COMPLETE')
        args = p.parse_args(argv)
        plan = lire_json(args.plan)
        taches = plan['taches'] if isinstance(plan, dict) else plan
        if len({t['id'] for t in taches}) != len(taches):
            raise ErreurUsage('identifiants dupliqués dans le plan')
        resultat = analyse(args.racine, taches, args.bloc3b)
        args.sortie.mkdir(parents=True, exist_ok=True)
        ecrire_json(args.sortie / 'analyse.json', json_fini(resultat))
        texte = '\n'.join(resultat['verdicts']) + '\n'
        (args.sortie / 'verdicts.txt').write_text(texte, encoding='utf-8')
        (args.sortie / 'tables.md').write_text(tables(resultat), encoding='utf-8')
        print(texte, end='')
        return 0 if resultat['ok'] else 1
    except ErreurUsage as exc:
        print(f'INFO WP10-USAGE erreur : {exc}')
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'FAIL WP10-ANALYSE erreur : {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
