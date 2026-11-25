# SPDX-License-Identifier: Apache-2.0
"""Portes sans GPU du générateur et du lecteur de transport."""

import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


OUTIL = Path(__file__).resolve().parents[1] / 'thermo' / 'transport.py'
ENTETE = 'poids,beta,rep,graine,depart,n,tot,fe,h1_tot,h1_fe,h2_tot,h2_fe,beta_s,st,ss'
APPARIEMENT = 'la graine de la chaîne (nœud k, réplique i) est G + k·R + i pour tous les bras'


def ecrire_json(chemin, objet):
    chemin.write_text(json.dumps(objet, ensure_ascii=False), encoding='utf-8')


def fabriquer(dossier, cotes, beta_s, noeuds, graines, actions, inverse=False,
              actions_li=None, actions_separees=None):
    """Même schéma, colonnes et groupes que les sorties anisotropes de kvu."""
    points = dossier / 'points'
    points.mkdir(parents=True, exist_ok=True)
    for fichier in points.glob('*.csv'):
        fichier.unlink()
    nombre = len(actions[0])
    manifeste = {
        'schema': 'kvu-thermo-mesure/1', 'cotes': cotes, 'betas': noeuds, 'poids': [1.0],
        'torsion': 'aucune', 'appariement': APPARIEMENT, 'graine_base': graines,
        'precision': 'f64', 'therm': 100, 'skip': 2, 'nor': 2, 'depart': 'alterne',
        'repliques': nombre, 'nmeas': 80, 'plafonds_du_bain': 0, 'paire': False,
        'sha256_source_assemblee': '1' * 64,
        'couplages_separes': {'direction_du_temps': 0, 'beta_s': beta_s, 'betas_t': noeuds},
        'series': {'betas_par_groupe': 1, 'nmeas': 80, 'repliques': nombre},
    }
    colonnes = ENTETE.split(',')
    if actions_li is not None:
        manifeste['lien_integre'] = {'colonnes_csv': 'li_tot, li_t, li_s'}
        colonnes += ['li_tot', 'li_t', 'li_s']
    if actions_separees is not None:
        colonnes_li = ['y0_t', 'ys_t', 'zs_s', 'y0_tot', 'ys_tot']
        manifeste['lien_integre'] = {
            'colonnes_csv': ', '.join(colonnes_li) + ' : moyennes par replique sur les nmeas mesures.'
        }
        colonnes += colonnes_li
    ecrire_json(dossier / 'manifest_thermo.json', manifeste)
    for k, (beta_t, scores) in enumerate(zip(noeuds, actions)):
        with (points / f'bras0_g{k:03d}.csv').open('w', encoding='utf-8', newline='') as flux:
            lignes = csv.writer(flux)
            lignes.writerow(colonnes)
            indices = range(nombre - 1, -1, -1) if inverse else range(nombre)
            for rep in indices:
                st = scores[rep]
                ss = 3 * math.prod(cotes) * 0.5
                tot = st + ss
                ligne = [
                    1, beta_t, rep, graines + k * nombre + rep,
                    'chaud' if rep % 2 else 'froid', 80, tot, st,
                    tot, st, tot, st, beta_s, st, ss,
                ]
                if actions_li is not None:
                    li_t = actions_li[k][rep]
                    ligne += [li_t + ss, li_t, ss]
                if actions_separees is not None:
                    y0, ys = (serie[k][rep] for serie in actions_separees)
                    ligne += [y0, ys, ss, y0 + ss, ys + ss]
                lignes.writerow(ligne)


class TestTransport(unittest.TestCase):
    def setUp(self):
        temporaire = tempfile.TemporaryDirectory()
        self.addCleanup(temporaire.cleanup)
        self.tmp = Path(temporaire.name)
        self.racine = self.tmp / 'essai avec espaces'
        self.plan = self.tmp / 'plan.json'

    def lancer(self, *args):
        return subprocess.run(
            [sys.executable, '-I', '-B', str(OUTIL), *map(str, args)],
            cwd=self.tmp, capture_output=True, text=True, timeout=30,
        )

    def ok(self, *args):
        sortie = self.lancer(*args)
        self.assertEqual(sortie.returncode, 0, sortie.stderr)
        return json.loads(sortie.stdout)

    def refus(self, message, *args):
        sortie = self.lancer(*args)
        self.assertEqual(sortie.returncode, 2, sortie.stderr)
        self.assertEqual(sortie.stdout, '')
        self.assertIn(message, sortie.stderr)
        self.assertNotIn('Traceback', sortie.stderr)

    def args_plan(self, ell, grilles):
        args = [
            'planifier', '--ell', ell, '--sortie', self.racine,
            '--repliques', 4, '--therm', 100, '--nmeas', 80, '--skip', 2,
            '--graine', 100, '--backend', 'hip',
        ]
        for grille in grilles:
            args += ['--grille', grille]
        return args

    def preparer_plan(self, grilles=('4:0.2,0.7', '5:0.2,0.7')):
        plan = self.ok(*self.args_plan(2.40, grilles))
        ecrire_json(self.plan, plan)
        return plan['acquisitions']

    def lire(self, type_, *args):
        return self.ok('lire', self.plan, '--de', 4, '--vers', 5, '--type', type_, *args)

    def preparer_section(self, section, force, base=None, grille=None, force_li=None,
                         forces_separees=None):
        noeuds = section['betas_t'] if grille is None else grille
        graines = section['graine_base'] if base is None else base
        court = [[80.0 + 2 * k + rep * 3 for rep in range(4)] for k in range(len(noeuds))]
        long = [
            [2 * st - force(noeud, rep) for rep, st in enumerate(sts)]
            for noeud, sts in zip(noeuds, court)
        ]
        court_li = long_li = None
        if force_li is not None:
            court_li = [[st + 10 for st in sts] for sts in court]
            long_li = [
                [2 * st - force_li(noeud, rep) for rep, st in enumerate(sts)]
                for noeud, sts in zip(noeuds, court_li)
            ]
        extra = {'A': court_li, 'B': long_li}
        separees = {'A': None, 'B': None}
        if forces_separees is not None:
            separees = {'A': [], 'B': []}
            for i, force_sep in enumerate(forces_separees):
                serie_a = [
                    [0 if i == 0 and beta == 0 else st + 10 for st in sts]
                    for beta, sts in zip(noeuds, court)
                ]
                serie_b = [
                    [2 * st - force_sep(beta, rep) for rep, st in enumerate(sts)]
                    for beta, sts in zip(noeuds, serie_a)
                ]
                separees['A'].append(serie_a)
                separees['B'].append(serie_b)
        for bras, actions in (('A', court), ('B', long)):
            info = section['bras'][bras]
            fabriquer(
                self.racine / info['dossier'], info['cotes'], section['beta_s'],
                noeuds, graines, actions, inverse=(bras == 'B'), actions_li=extra[bras],
                actions_separees=separees[bras],
            )

    def test_plan_ltw03_et_durees_explicites(self):
        grilles = ('4:0.19,0.6', '5:0.19,0.6', '6:0.213,0.6', '8:0.184,0.6')
        args = self.args_plan(2.40, grilles)
        plan = self.ok(*args)
        self.assertEqual(plan['direction_du_temps'], 0)
        self.assertIn('hep-lat/0307017', plan['echelle']['source'])
        cibles = plan['echelle']['tailles']
        for taille, beta in (('4', 2.06506750), ('5', 2.20198564), ('6', 2.2674), ('8', 2.3600)):
            valeur = cibles[taille]['beta_isotrope']
            with self.subTest(taille=taille):
                self.assertAlmostEqual(valeur, beta, delta=6e-9 if taille in ('4', '5') else 1e-4)
                lo, hi = cibles[taille]['encadrement_beta']
                self.assertLess(lo, valeur)
                self.assertLess(valeur, hi)
                self.assertEqual(plan['acquisitions'][f'n{taille}_t{taille}']['betas_t'][-1], valeur)
        self.assertEqual(plan['pas'][0]['E'], {'depart': 'n4_t4', 'arrivee': 'n5_t5'})
        self.assertEqual(plan['pas'][0]['D'], {'depart': 'n4_t5', 'arrivee': 'n5_t5'})
        acquisition = plan['acquisitions']['n4_t5']
        bras = acquisition['bras']
        self.assertEqual(bras['A']['cotes'], [5, 8, 8, 8])
        self.assertEqual(bras['B']['cotes'], [10, 8, 8, 8])
        cmd = acquisition['argv']
        self.assertEqual(cmd[:4], ['kvu', 'su2', 'thermo', 'mesure'])
        self.assertNotIn('--temps', cmd)
        self.assertNotIn('--lien-integre', cmd)
        self.assertEqual(cmd[cmd.index('--impairs') + 1], 'oui')
        self.assertEqual(cmd[cmd.index('--cotes') + 1], '5,8,8,8')
        self.assertEqual(cmd[cmd.index('--cotes-b') + 1], '10,8,8,8')
        self.assertEqual(cmd[cmd.index('--sortie') + 1], str(self.racine / 'n4_t5'))
        self.assertNotIn('--impairs', plan['acquisitions']['n4_t4']['argv'])
        commandes = self.lancer(*args, '--commandes')
        self.assertEqual(commandes.returncode, 0, commandes.stderr)
        lignes = [shlex.split(ligne) for ligne in commandes.stdout.splitlines()]
        self.assertEqual(lignes.count(cmd), 1)
        self.assertEqual(len(lignes), len(plan['acquisitions']))
        self.assertFalse(self.racine.exists())

    def test_plan_refuse_extrapolation_grilles_et_u64(self):
        self.refus('extrapolation refusée', *self.args_plan(40, ('4:0.2', '5:0.2')))
        self.refus('strictement croissants', *self.args_plan(2.4, ('4:0.2,0.1', '5:0.2')))
        self.refus('au-delà', *self.args_plan(2.4, ('4:2.2', '5:0.2')))
        args = self.args_plan(2.40, ('4:0.2', '5:0.2'))
        args[args.index('--graine') + 1] = 2**64 - 1
        self.refus('hors u64', *args)

    def test_constante_affine_signes_E_D_et_ancres(self):
        acquisitions = self.preparer_plan()
        self.preparer_section(acquisitions['n4_t4'], lambda _, rep: 3.5)
        self.preparer_section(acquisitions['n4_t5'], lambda _, rep: -2.25)
        self.preparer_section(acquisitions['n5_t5'], lambda beta, rep: 2 * beta - 6)
        n = acquisitions['n4_t4']['beta_s']
        m = acquisitions['n5_t5']['beta_s']
        chemin = self.tmp / 'ancres.json'
        ecrire_json(chemin, {
            'n4_t5': {'beta_s': n, 'beta_t': 0.2, 'source': 'fixture', 'intervalle': [0.01, 0.02]},
            'n5_t5': {'beta_s': m, 'beta_t': 0.2, 'source': 'fixture', 'intervalle': [0.03, 0.05]},
        })
        arrivee = (m**2 - 0.2**2) - 6 * (m - 0.2)
        for type_, terme, facteur in (('E', 3.5, 12288), ('D', -2.25, 15360)):
            with self.subTest(type=type_):
                sortie = self.lire(type_, '--ancres', chemin)
                self.assertEqual(sortie['sens'], 'arrivée moins départ')
                self.assertEqual(sortie['depart']['normalisation_densites'], facteur)
                self.assertEqual(sortie['arrivee']['normalisation_densites'], 30000)
                self.assertAlmostEqual(sortie['depart']['trapezes']['integrale'], terme * (n - 0.2))
                self.assertAlmostEqual(sortie['arrivee']['trapezes']['integrale'], arrivee)
                delta = sortie['difference_trapezes']
                self.assertAlmostEqual(delta['integrale'], arrivee - terme * (n - 0.2))
                self.assertAlmostEqual(delta['erreur_type'], 0)
                self.assertIsNone(sortie['biais_a_controler']['quadrature'])
                self.assertIsNone(sortie['biais_a_controler']['echelle'])
                if type_ == 'E':
                    self.assertIsNone(sortie['difference_ancres'])
                else:
                    self.assertAlmostEqual(sortie['difference_ancres'][0], 0.01)
                    self.assertAlmostEqual(sortie['difference_ancres'][1], 0.04)

    def test_covariance_annule_le_bruit_entre_sections(self):
        acquisitions = self.preparer_plan(('4:0.2', '5:0.2'))
        for cle in ('n4_t4', 'n5_t5'):
            info = acquisitions[cle]
            largeur = info['beta_s'] - 0.2
            self.preparer_section(info, lambda _, rep: (rep - 1.5) / largeur, base=500)
        sortie = self.lire('E')
        self.assertGreater(sortie['depart']['trapezes']['erreur_type'], 0.5)
        self.assertGreater(sortie['arrivee']['trapezes']['erreur_type'], 0.5)
        delta = sortie['difference_trapezes']
        self.assertEqual(len(delta['erreurs_type_blocs']), 2)
        self.assertAlmostEqual(delta['erreur_type'], 0, delta=1e-12)

    def test_graines_distinctes_ou_partagees_en_partie(self):
        acquisitions = self.preparer_plan(('4:0.2', '5:0.2'))

        def score(beta, rep):
            return beta + rep

        self.preparer_section(acquisitions['n4_t4'], score, base=500)
        self.preparer_section(acquisitions['n5_t5'], score, base=900)
        sortie = self.lire('E')
        delta = sortie['difference_trapezes']
        self.assertEqual(delta['covariance'], 'borne triangulaire')
        sigma_depart = sortie['depart']['trapezes']['erreur_type']
        sigma_arrivee = sortie['arrivee']['trapezes']['erreur_type']
        self.assertAlmostEqual(delta['erreur_type'], sigma_depart + sigma_arrivee)
        indep = self.lire('E', '--justification-independance', 'essai synthétique à flux disjoints')
        self.assertAlmostEqual(
            indep['difference_trapezes']['erreur_type'], math.hypot(*delta['erreurs_type_blocs']),
        )
        self.preparer_section(acquisitions['n5_t5'], score, base=501)
        self.assertEqual(self.lire('E')['difference_trapezes']['covariance'], 'borne triangulaire')
        self.refus(
            'graines partagées', 'lire', self.plan, '--de', 4, '--vers', 5, '--type', 'E',
            '--justification-independance', 'demande invalide',
        )

    def test_noeuds_pilotes_sans_integration_et_fichiers_manquants(self):
        sources = self.preparer_plan()
        self.preparer_section(sources['n4_t4'], lambda _, rep: 2, grille=[0.2, 0.7])
        court, long = (self.racine / 'n4_t4' / nom for nom in ('A', 'B'))
        resultat = self.ok('noeuds', court, long)
        self.assertEqual([x['force'] for x in resultat['noeuds']], [2, 2])
        self.assertNotIn('trapezes', resultat)
        self.refus('grille mesurée différente du plan complet', 'lire', self.plan,
                   '--de', 4, '--vers', 5, '--type', 'E')
        (long / 'points/bras0_g001.csv').unlink()
        self.refus('CSV manquants', 'noeuds', court, long)

    def test_rejets_manifeste_A_B(self):
        sources = self.preparer_plan()
        info = sources['n4_t5']
        self.preparer_section(info, lambda _, rep: 2)
        court, long = (self.racine / 'n4_t5' / nom for nom in ('A', 'B'))
        fichier = long / 'manifest_thermo.json'
        original = json.loads(fichier.read_text(encoding='utf-8'))
        for mutation, message in (
            ({'schema': 'inconnu'}, 'schéma'),
            ({'cotes': [10, 4, 8, 16]}, 'géométrie A/B'),
            ({'poids': [False]}, 'poids'),
            ({'torsion_fixe': '23'}, 'sans torsion'),
            ({'couture': {'axe': 0, 'decalage': [0, 1, 0, 0]}}, 'périodique'),
            ({'plafonds_du_bain': 1}, 'plafonds_du_bain'),
            ({'graine_base': 27}, 'graine_base'),
            ({'nmeas': False}, 'nmeas'),
            ({'precision': 'f32'}, 'precision'),
            ({'sha256_source_assemblee': '2' * 64}, 'sha256_source_assemblee'),
            ({'series': dict(original['series'], betas_par_groupe=2)}, 'betas_par_groupe'),
            ({'couplages_separes': dict(original['couplages_separes'], direction_du_temps=1)},
             'direction_du_temps'),
            ({'couplages_separes': dict(original['couplages_separes'], beta_s=2.0)}, 'beta_s'),
        ):
            with self.subTest(mutation=mutation):
                ecrire_json(fichier, original | mutation)
                self.refus(message, 'noeuds', court, long)

    def test_rejets_CSV_repliques_couplage_et_finitude(self):
        sources = self.preparer_plan()
        self.preparer_section(sources['n4_t4'], lambda _, rep: 2)
        court, long = (self.racine / 'n4_t4' / nom for nom in ('A', 'B'))
        fichier = court / 'points/bras0_g000.csv'
        with fichier.open(encoding='utf-8', newline='') as flux:
            original = list(csv.reader(flux))

        def ecrire(lignes):
            with fichier.open('w', encoding='utf-8', newline='') as flux:
                csv.writer(flux).writerows(lignes)

        for cle, valeur, message in (
            ('rep', '1', 'graine incompatible'),
            ('graine', '999', 'graine incompatible'),
            ('n', '81', 'nombre de mesures'),
            ('beta', '0.700', 'couplages'),
            ('poids', '0.5', 'poids'),
            ('st', 'NaN', 'nombre fini'),
            ('ss', '0', 'colonnes extensives'),
        ):
            with self.subTest(cle=cle):
                lignes = [ligne.copy() for ligne in original]
                lignes[1][lignes[0].index(cle)] = valeur
                ecrire(lignes)
                self.refus(message, 'noeuds', court, long)
        ecrire(original + [original[1]])
        self.refus('réplique répétée', 'noeuds', court, long)
        ecrire(original[:-1])
        self.refus('répliques distinctes attendues', 'noeuds', court, long)

    def test_csv_par_nom_avec_extension_et_source_absente(self):
        sources = self.preparer_plan()
        self.preparer_section(sources['n4_t4'], lambda _, rep: 2)
        court, long = (self.racine / 'n4_t4' / nom for nom in ('A', 'B'))
        fichier = long / 'points/bras0_g000.csv'
        with fichier.open(encoding='utf-8', newline='') as flux:
            entete, *lignes = list(csv.reader(flux))
        manifeste = long / 'manifest_thermo.json'
        man = json.loads(manifeste.read_text(encoding='utf-8'))
        del man['sha256_source_assemblee']
        man['lien_integre'] = {'colonnes_csv': 'li_tot, li_t, li_s'}
        ecrire_json(manifeste, man)
        with fichier.open('w', encoding='utf-8', newline='') as flux:
            csv_sortie = csv.writer(flux)
            csv_sortie.writerow([*reversed(entete), 'li_tot', 'li_t', 'li_s'])
            for ligne in lignes:
                csv_sortie.writerow([*reversed(ligne), 700, 700, 0])
        sortie = self.ok('noeuds', court, long)
        self.assertEqual([noeud['force'] for noeud in sortie['noeuds']], [2, 2, 2])
        self.assertEqual(sortie['sha256_source_assemblee'], {'A': '1' * 64, 'B': None})
        for nom, message in (('fe', 'nom de colonne CSV en double'),
                             ('autre', 'colonnes CSV requises manquantes')):
            invalide = entete.copy()
            invalide[invalide.index('st')] = nom
            with fichier.open('w', encoding='utf-8', newline='') as flux:
                csv.writer(flux).writerows([invalide, *lignes])
            self.refus(message, 'noeuds', court, long)
        ecrire_json(manifeste, man)
        contenu = manifeste.read_text(encoding='utf-8')
        manifeste.write_text(contenu[:-1] + ', "poids": [1]}', encoding='utf-8')
        self.refus('nom de champ JSON en double', 'noeuds', court, long)

    def test_estimateur_lien_integre_separe_de_brut(self):
        plan = self.ok(*self.args_plan(2.40, ('4:0.2', '5:0.2')), '--estimateur', 'lien-integre')
        self.assertEqual(plan['estimateur'], 'lien-integre')
        for acquisition in plan['acquisitions'].values():
            self.assertEqual(acquisition['argv'].count('--lien-integre'), 1)
        ecrire_json(self.plan, plan)
        acquisitions = plan['acquisitions']
        self.preparer_section(acquisitions['n4_t4'], lambda beta, rep: 2, force_li=lambda beta, rep: -4)
        self.preparer_section(acquisitions['n5_t5'], lambda beta, rep: 3, force_li=lambda beta, rep: -7)
        beta_4 = acquisitions['n4_t4']['beta_s'] - 0.2
        beta_5 = acquisitions['n5_t5']['beta_s'] - 0.2
        brut = self.lire('E')
        li = self.lire('E', '--estimateur', 'lien-integre')
        self.assertEqual(brut['estimateur'], 'brut')
        self.assertEqual(li['estimateur'], 'lien-integre')
        self.assertAlmostEqual(brut['difference_trapezes']['integrale'], 3 * beta_5 - 2 * beta_4)
        self.assertAlmostEqual(li['difference_trapezes']['integrale'], -7 * beta_5 + 4 * beta_4)

        court, long = (self.racine / 'n4_t4' / nom for nom in ('A', 'B'))
        integre = ('noeuds', court, long, '--estimateur', 'lien-integre')
        resultat = self.ok(*integre)
        self.assertEqual(resultat['colonne_temporelle'], 'li_t')
        self.assertEqual([x['force'] for x in resultat['noeuds']], [-4, -4])
        for estimateur in ('y0', 'ys'):
            self.refus('format historique li_t', 'noeuds', court, long, '--estimateur', estimateur)
        fichier = long / 'points/bras0_g000.csv'
        with fichier.open(encoding='utf-8', newline='') as flux:
            original = list(csv.reader(flux))
        for indice, valeur, message in ((0, 'autre', 'colonnes CSV requises manquantes'),
                                        (1, 'inf', 'li_t : nombre fini')):
            lignes = [ligne.copy() for ligne in original]
            lignes[indice][original[0].index('li_t')] = valeur
            with fichier.open('w', encoding='utf-8', newline='') as flux:
                csv.writer(flux).writerows(lignes)
            self.refus(message, *integre)
            self.assertEqual(self.ok('noeuds', court, long)['noeuds'][0]['force'], 2)
        chemin = long / 'manifest_thermo.json'
        man = json.loads(chemin.read_text(encoding='utf-8'))
        del man['lien_integre']
        ecrire_json(chemin, man)
        self.refus('marqueur lien_integre requis', *integre)

    def test_estimateurs_separes_deux_grilles_et_echange(self):
        grilles = ('4:0.2,0.7', '5:0.3,0.9')
        for estimateur in ('lien-integre', 'y0', 'ys'):
            with self.subTest(estimateur=estimateur):
                plan = self.ok(*self.args_plan(2.40, grilles), '--estimateur', estimateur)
                self.assertEqual(plan['estimateur'], estimateur)
                for acquisition in plan['acquisitions'].values():
                    self.assertEqual(acquisition['argv'].count('--lien-integre'), 1)
        acquisitions = self.preparer_plan(grilles)
        depart, arrivee = (acquisitions[cle] for cle in ('n4_t4', 'n5_t5'))
        largeur_4 = depart['beta_s'] - 0.2
        largeur_5 = arrivee['beta_s'] - 0.3
        y0, ys = (2, 7), (5, -3)
        integrales = {
            'brut': 3 * largeur_5 - 11 * largeur_4,
            'y0': y0[1] * largeur_5 - y0[0] * largeur_4,
            'ys': ys[1] * largeur_5 - ys[0] * largeur_4,
        }
        integrales['lien-integre'] = (integrales['y0'] + integrales['ys']) / 2
        for inverse in (False, True):
            for i, section in enumerate((depart, arrivee)):
                p, q = (ys[i], y0[i]) if inverse else (y0[i], ys[i])
                self.preparer_section(
                    section, lambda _, rep: (11, 3)[i],
                    forces_separees=(lambda _, rep: p, lambda _, rep: q),
                )
            colonnes = {
                'brut': 'st', 'y0': 'y0_t', 'ys': 'ys_t', 'lien-integre': '(y0_t + ys_t) / 2',
            }
            for estimateur, colonne in colonnes.items():
                with self.subTest(estimateur=estimateur, inverse=inverse):
                    args = () if estimateur == 'brut' else ('--estimateur', estimateur)
                    doc = self.lire('E', *args)
                    self.assertEqual(doc['estimateur'], estimateur)
                    for ext in ('depart', 'arrivee'):
                        self.assertEqual(doc[ext]['colonne_temporelle'], colonne)
                    attendu = {'y0': 'ys', 'ys': 'y0'}.get(estimateur, estimateur) if inverse else estimateur
                    self.assertAlmostEqual(doc['difference_trapezes']['integrale'], integrales[attendu])

    def test_estimateurs_separes_origine_totaux_et_schema(self):
        section = self.preparer_plan(('4:0,0.7', '5:0.3'))['n4_t4']
        self.preparer_section(
            section, lambda _, rep: 2,
            forces_separees=(lambda beta, rep: 4 * beta, lambda _, rep: 6),
        )
        court, long = (self.racine / 'n4_t4' / nom for nom in ('A', 'B'))
        args = ('noeuds', court, long, '--estimateur')
        for estimateur, attendu in (('y0', 0), ('ys', 6), ('lien-integre', 3)):
            resultat = self.ok(*args, estimateur)
            self.assertEqual(resultat['noeuds'][0]['force'], attendu)
        for dossier in (court, long):
            manifeste = dossier / 'manifest_thermo.json'
            man = json.loads(manifeste.read_text(encoding='utf-8'))
            man['precision'] = 'f32'
            ecrire_json(manifeste, man)

        fichier = long / 'points/bras0_g001.csv'
        with fichier.open(encoding='utf-8', newline='') as flux:
            original = list(csv.reader(flux))

        def ecrire(lignes):
            with fichier.open('w', encoding='utf-8', newline='') as flux:
                csv.writer(flux).writerows(lignes)

        lignes = [ligne.copy() for ligne in original]
        index = {cle: original[0].index(cle) for cle in ('y0_t', 'ys_t', 'zs_s', 'y0_tot', 'ys_tot')}
        for cle, valeur in (
            ('y0_t', 1024), ('ys_t', 1025), ('zs_s', -1024), ('y0_tot', 1e-13), ('ys_tot', 1),
        ):
            lignes[1][index[cle]] = valeur
        ecrire(lignes)
        self.ok(*args, 'ys')
        for cle in ('y0_tot', 'ys_tot'):
            with self.subTest(total=cle):
                invalide = [ligne.copy() for ligne in lignes]
                invalide[1][index[cle]] = '0.1'
                ecrire(invalide)
                self.refus(f'{cle} incohérent', *args, 'ys')
        for cle in index:
            with self.subTest(colonne=cle):
                invalide = [ligne.copy() for ligne in original]
                invalide[0][index[cle]] = 'inconnue'
                ecrire(invalide)
                self.refus('colonnes CSV requises manquantes', *args, 'y0')
                invalide = [ligne.copy() for ligne in original]
                invalide[1][index[cle]] = 'nan'
                ecrire(invalide)
                self.refus(f'{cle} : nombre fini', *args, 'y0')
        ecrire([original[0] + ['li_t'], *(ligne + ['1'] for ligne in original[1:])])
        self.refus('schéma lien_integre ambigu', *args, 'lien-integre')
        self.assertEqual(self.ok('noeuds', court, long)['noeuds'][0]['force'], 2)
        ecrire(original)

        marqueur = man.pop('lien_integre')
        ecrire_json(manifeste, man)
        self.refus('marqueur lien_integre requis', *args, 'ys')
        man['lien_integre'] = {'colonnes_csv': marqueur['colonnes_csv'].replace('y0_t', 'li_t')}
        ecrire_json(manifeste, man)
        self.refus('schéma inconnu, incomplet ou ambigu', *args, 'lien-integre')
        man['lien_integre'] = {'colonnes_csv': 'li_tot, li_t, li_s'}
        ecrire_json(manifeste, man)
        self.refus('schémas lien_integre différents entre les bras A/B', *args, 'lien-integre')
        man['lien_integre'] = marqueur
        ecrire_json(manifeste, man)

        fichier = long / 'points/bras0_g000.csv'
        with fichier.open(encoding='utf-8', newline='') as flux:
            lignes = list(csv.reader(flux))
        lignes[1][index['y0_t']] = '1'
        lignes[1][index['y0_tot']] = float(lignes[1][index['zs_s']]) + 1
        ecrire(lignes)
        self.refus('y0_t doit être exactement nul', *args, 'y0')


if __name__ == '__main__':
    unittest.main()
