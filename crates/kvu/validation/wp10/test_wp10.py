# SPDX-License-Identifier: Apache-2.0
"""Tests WP10, CPU seulement ; toutes les écritures de données sont temporaires.

Exécution isolée : python3 -I -m unittest discover -s crates/kvu/validation/wp10 -v
Définir KVU_BIN pour inclure le rejeu CPU du deuxième blocage.
"""

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wp10_analyse as analyse
import wp10_post as post
import wp10_plan as plan_campagne
from wp10_stat import (combiner_chaines, lire_csv, lire_q64, plaquette_q64, qdag, qmul,
                       NB_BOUCLES, sha256_fichier, stat_serie, tau_int, tau_int_diag, unitarite_q64, z_score)

ICI = Path(__file__).resolve().parent
DEPOT = ICI.parents[3]
FIXTURES = Path('/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp10/fixtures/E')
FIXTURE = FIXTURES / 'P18-b04-c1'
KVU_BIN = os.environ.get('KVU_BIN')


def froid(l=2):
    q = np.zeros((4, l, l, l, l, 4), dtype='<f8')
    q[..., 0] = 1
    return q


def uniforme(n):
    graine = 12345
    valeurs = []
    for _ in range(n):
        graine = (graine * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        valeurs.append((graine >> 11) / (1 << 53) - 0.5)
    return valeurs


def appel(fonction, arguments):
    sortie = io.StringIO()
    with redirect_stdout(sortie):
        code = fonction(arguments)
    return code, sortie.getvalue()


class Statistiques(unittest.TestCase):
    def test_constante(self):
        self.assertEqual(tau_int([3.] * 100), (0.5, 0, 0.))
        s = stat_serie([3.] * 100)
        self.assertEqual((s['moy'], s['sd'], s['err'], s['n']), (3., 0., 0., 100))

    def test_series_courtes_et_vides(self):
        for x in ([], [1], [1, 3], list(range(7))):
            self.assertEqual(tau_int(x), (0.5, 0, 0.))
        self.assertTrue(math.isnan(stat_serie([])['moy']))
        self.assertTrue(math.isnan(stat_serie([1])['err']))
        self.assertEqual(stat_serie([1, 3])['err'], 1.)
        self.assertAlmostEqual(stat_serie([1, 3])['sd'], math.sqrt(2))

    def test_bruit_blanc_et_ar1(self):
        bruit = uniforme(200000)
        t, w, _ = tau_int(bruit)
        self.assertAlmostEqual(t, 0.5, delta=0.03)
        self.assertLessEqual(w, 4)
        courant, ar = 0., []
        for b in bruit:
            courant = .9*courant + b
            ar.append(courant)
        self.assertAlmostEqual(tau_int(ar)[0], 9.5, delta=.25*9.5)

    def test_reference_rust(self):
        chemin = DEPOT / 'crates/kvu/tests/donnees/rg_flot/serie_plaquette_L12_b2.3_c1.txt'
        if not chemin.exists():
            self.skipTest('série de référence Rust absente')
        x = [float(v) for v in chemin.read_text().splitlines()]
        self.assertEqual(len(x), 2000)
        t, w, dt = tau_int(x)
        self.assertAlmostEqual(t, 1.6900620161191204, delta=1e-12)
        self.assertEqual(w, 11)
        self.assertAlmostEqual(dt, .25631059521907873, delta=1e-12)

    def test_fenetre_et_variance_reference_independante(self):
        x = uniforme(160)
        moy = sum(x)/len(x)
        centre = [v-moy for v in x]
        var = sum(v*v for v in centre)/len(x)
        t, w = .5, 0
        for w in range(1, len(x)//4):
            t += sum(centre[i]*centre[i+w] for i in range(len(x)-w))/(len(x)-w)/var
            if w >= 6*t:
                break
        self.assertAlmostEqual(tau_int(x)[0], max(.5, t), delta=1e-14)
        self.assertEqual(tau_int(x)[1], w)
        s = stat_serie(x)
        self.assertAlmostEqual(s['sd']**2, var*len(x)/(len(x)-1), delta=1e-15)
        self.assertAlmostEqual(s['err']**2, 2*s['tau']*s['sd']**2/len(x), delta=1e-15)

    def test_combinaison(self):
        stats = [stat_serie([i, i+1, i+2]) for i in range(3)]
        self.assertIs(combiner_chaines([stats[0]]), stats[0])
        s = combiner_chaines(stats)
        self.assertEqual(s['moy'], 2.)
        self.assertAlmostEqual(s['err'], math.sqrt(sum(c['err']**2 for c in stats))/3)
        self.assertEqual(s['n'], 9)
        with self.assertRaises(ValueError):
            combiner_chaines([])
        with self.assertRaises(ValueError):
            combiner_chaines([stat_serie([1, 2]), stat_serie([1, 2, 3])])

    def test_z(self):
        self.assertEqual(z_score(5, 3, 0, 4), 1.)
        self.assertEqual(z_score(0, 0, 2, 1), -2.)
        self.assertTrue(math.isnan(z_score(1, 0, 1, 0)))
        self.assertTrue(math.isnan(z_score(1, 0, 2, 0)))

    def test_tau_diag_convergence_et_compatibilite(self):
        bruit = uniforme(2000)
        self.assertTrue(tau_int_diag(bruit)[3])
        for x in (bruit, [], list(range(7)), [2.]*100):
            self.assertEqual(tau_int_diag(x)[:3], tau_int(x))
        for x in ([], list(range(7)), [2.]*100):
            self.assertFalse(tau_int_diag(x)[3])

    def test_tau_diag_ar1_lente(self):
        courant, serie = 10., []
        for bruit in uniforme(50):
            courant = .999*courant + .01*bruit
            serie.append(courant)
        self.assertFalse(tau_int_diag(serie)[3])
        self.assertEqual(tau_int_diag(serie)[1], len(serie)//4 - 1)

    def test_convergence_au_dernier_tour(self):
        # n=8 : un seul tour, qui atteint bien le critère avec c=0.
        self.assertTrue(tau_int_diag(range(8), c=0)[3])

    def test_effectifs_et_convergence_combines(self):
        stats = [stat_serie(uniforme(100)), stat_serie([1.]*100)]
        for s in stats:
            self.assertEqual(s['n_eff'], s['n']/(2*s['tau']))
        self.assertIs(combiner_chaines([stats[0]]), stats[0])
        c = combiner_chaines(stats)
        self.assertFalse(c['converge'])
        self.assertEqual(c['n_eff'], sum(s['n_eff'] for s in stats))
        self.assertTrue(combiner_chaines([stats[0], stats[0]])['converge'])

    def test_frontieres_decision(self):
        for z in (-3.01, -3., -2.99, 2.99, 3., 3.01):
            with self.subTest(z=z):
                a = dict(moy=z, err=1., converge=True, n_eff=20.)
                b = dict(moy=0., err=0., converge=True, n_eff=20.)
                r = analyse.comparaison(a, b)
                self.assertEqual(r['z'], z)
                self.assertEqual(r['verdict'], 'PASS' if abs(z) < 3 else 'FAIL')

    def test_conclusivite_deux_cotes(self):
        s = dict(moy=0., err=1., converge=True, n_eff=20.)
        self.assertTrue(analyse.conclusive(s))
        for cote in (0, 1):
            for changement in (dict(converge=False), dict(n_eff=19.999)):
                with self.subTest(cote=cote, changement=changement):
                    stats = [dict(s), dict(s)]
                    stats[cote].update(changement)
                    r = analyse.comparaison(*stats)
                    self.assertEqual(r['verdict'], 'SKIP')
                    self.assertIsNone(r['ok'])

    def test_plan_pilote_amende(self):
        taches = plan_campagne.plan()
        faible = next(t for t in taches if t['id'] == 'N36-b08')
        self.assertEqual(faible['genre'], 'N')
        self.assertAlmostEqual(faible['beta_run']/faible['beta'], 1.0002, delta=1e-15)
        controle = next(t for t in taches if t['id'] == 'C54-b16')
        self.assertEqual((controle['therm'], controle['nmeas'], controle['skip']), (2000, 200, 10))
        self.assertTrue(all(t['garder'] == {'un_sur': 100} for t in taches if t['genre'] in 'RCMN'))


class Quaternions(unittest.TestCase):
    def test_froid(self):
        self.assertEqual(plaquette_q64(froid()), 1.)
        self.assertEqual(unitarite_q64(froid()), 0.)

    def test_jauge_pure(self):
        g = np.random.default_rng(414).normal(size=(3, 3, 3, 3, 4))
        g /= np.linalg.norm(g, axis=-1, keepdims=True)
        q = np.stack([qmul(g, qdag(np.roll(g, -1, axis=mu))) for mu in range(4)])
        self.assertAlmostEqual(plaquette_q64(q), 1., delta=1e-13)
        self.assertLessEqual(unitarite_q64(q), 1e-13)

    def test_produit_convention(self):
        i, j, moins_k = np.eye(4)[1], np.eye(4)[2], -np.eye(4)[3]
        np.testing.assert_array_equal(qmul(i, j), moins_k)

    def test_lecture_q64_et_rejets(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'q.npy'
            for q in (froid().astype('f4'), froid().astype('>f8'), np.zeros((4, 2, 2, 3, 2, 4)),
                      np.asfortranarray(froid()), np.zeros((4, 0, 0, 0, 0, 4))):
                np.save(p, q)
                with self.assertRaises(ValueError):
                    lire_q64(p)
            np.save(p, froid())
            q, l = lire_q64(p)
            self.assertEqual(l, 2)
            np.testing.assert_array_equal(q, froid())

    @unittest.skipUnless(FIXTURE.exists(), 'fixtures CPU absentes')
    def test_plaquettes_fixtures_rust(self):
        for rep in sorted(FIXTURES.iterdir()):
            if not rep.is_dir() or not (rep/'bloc3.csv').is_file():
                continue
            for r in lire_csv(rep/'bloc3.csv'):
                with self.subTest(tache=rep.name, pas=r['pas']):
                    p = rep/post.nom_cfg(r['pas'])
                    if not p.is_file():
                        self.fail(f'fixture incomplète : {p}')
                    q, _ = lire_q64(p)
                    self.assertAlmostEqual(plaquette_q64(q), r['plaq_grossiere'], delta=1e-13)

    def test_csv_et_hachage_flux(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'f'
            donnees = b'wp10' * (3*1024*1024)
            p.write_bytes(donnees)
            self.assertEqual(sha256_fichier(p), hashlib.sha256(donnees).hexdigest())
            p.write_text('pas,phase,sha256_cfg,x\n2,sample,abc,1.25e-1\n')
            self.assertEqual(lire_csv(p), [dict(pas=2., phase='sample', sha256_cfg='abc', x=.125)])


@unittest.skipUnless(FIXTURE.exists(), 'fixtures CPU absentes')
class Integrite(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory(prefix='wp10-test-')
        self.addCleanup(self.temporaire.cleanup)
        self.rep = Path(self.temporaire.name)/'C18-b04'
        shutil.copytree(FIXTURE, self.rep)

    def test_h13_et_mutants(self):
        avant = {p.name: sha256_fichier(p) for p in self.rep.iterdir() if p.is_file()}
        destination = self.rep/'bilan.json'
        rc, texte = appel(post.main, ['h13', str(self.rep), '--mutants', '--json', str(destination)])
        self.assertEqual(rc, 0, texte)
        self.assertIn('PASS WP10-H13 ', texte)
        for m in ('m1', 'm2', 'm3', 'm4'):
            self.assertIn(f'PASS WP10-H13-MUT-{m}', texte)
        self.assertTrue(post.lire_json(destination)['ok'])
        self.assertTrue((self.rep/'H13_PASS').exists())
        self.assertEqual(avant, {nom: sha256_fichier(self.rep/nom) for nom in avant})

    def test_corruption_sha_et_marqueur_perime(self):
        self.assertEqual(appel(post.main, ['h13', str(self.rep)])[0], 0)
        p = post.configurations(self.rep)[0]
        with p.open('r+b') as f:
            f.seek(-1, 2)
            b = f.read(1)[0]
            f.seek(-1, 2)
            f.write(bytes([b ^ 1]))
        rc, texte = appel(post.main, ['h13', str(self.rep)])
        self.assertEqual(rc, 1)
        self.assertIn('sha256', texte)
        self.assertFalse((self.rep/'H13_PASS').exists())

    def test_unitarite_rouge(self):
        lignes = lire_csv(self.rep/'bloc3.csv')
        p = self.rep/post.nom_cfg(lignes[0]['pas'])
        q, _ = lire_q64(p)
        q[0, 0, 0, 0, 0] *= 1+1e-9
        np.save(p, q)
        self.assertIn('unitarité', post.verifier_configuration(p, lignes[0])['erreurs'])
        self.assertFalse(post.verifier_h13(self.rep)['ok'])

    def test_plaquette_rouge(self):
        lignes = lire_csv(self.rep/'bloc3.csv')
        lignes[0]['plaq_grossiere'] += 1e-9
        post.ecrire_csv(self.rep/'bloc3.csv', post.COLONNES, lignes)
        erreurs = post.verifier_h13(self.rep)['erreurs']
        self.assertTrue(any('plaquette' in e for e in erreurs), erreurs)

    def test_mutants_isoles(self):
        mutants = {m['mutant']: m for m in post.mutants_h13(self.rep)}
        self.assertIn('sha256', mutants['m1']['erreurs'])
        for mutant, erreur in (('m2', 'unitarité'), ('m3', 'plaquette'), ('m4', 'dénominateur')):
            self.assertTrue(mutants[mutant]['ok'], mutants[mutant])
            self.assertEqual(mutants[mutant]['erreurs'], [erreur])

    def test_nan_configuration_rouge(self):
        ligne = lire_csv(self.rep/'bloc3.csv')[0]
        p = self.rep/post.nom_cfg(ligne['pas'])
        q, _ = lire_q64(p)
        q[0, 0, 0, 0, 0, 1] = math.nan
        np.save(p, q)
        ligne['sha256_cfg'] = sha256_fichier(p)
        self.assertEqual(post.verifier_configuration(p, ligne)['erreurs'], ['non fini'])
        bilan = post.verifier_h13(self.rep)
        self.assertFalse(bilan['ok'])
        self.assertTrue(any('non fini' in e for e in bilan['erreurs']))
        rc, texte = appel(post.main, ['h13', str(self.rep), '--json', str(self.rep/'h13.json')])
        self.assertEqual(rc, 1, texte)
        self.assertIn('non fini', texte)
        self.assertFalse(post.lire_json(self.rep/'h13.json')['ok'])

    def test_denominateur_compteurs_et_fraction(self):
        ligne = lire_csv(self.rep/'bloc3.csv')[0]
        p = self.rep/post.nom_cfg(ligne['pas'])
        for cle, valeur in (('n_boucles', ligne['n_boucles']-1), ('n_boucles', ligne['n_boucles']+.5),
                            ('n_ambigues', -1), ('n_ambigues', .5), ('fraction_hors', -.001),
                            ('fraction_hors', 1.001), ('fraction_hors', math.nan)):
            with self.subTest(cle=cle, valeur=valeur):
                copie = dict(ligne, **{cle: valeur})
                self.assertEqual(post.verifier_configuration(p, copie)['erreurs'], ['dénominateur'])
        for valeur in (0., 1.):
            self.assertTrue(post.denominateur_valide(dict(ligne, fraction_hors=valeur)))

    def test_denominateur_controle_apres_elagage(self):
        self.assertEqual(appel(post.main, ['h13', str(self.rep)])[0], 0)
        self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4', '--execute'])[0], 0)
        journal = post.lire_json(self.rep/'ELAGAGE.json')
        self.assertEqual(journal['un_sur'], 4)
        self.assertEqual(journal['indices_gardes'], [4, 8, 12])
        lignes = lire_csv(self.rep/'bloc3.csv')
        self.assertFalse((self.rep/post.nom_cfg(lignes[0]['pas'])).exists())
        lignes[0]['n_boucles'] -= 1
        post.ecrire_csv(self.rep/'bloc3.csv', post.COLONNES, lignes)
        bilan = post.verifier_h13(self.rep)
        self.assertFalse(bilan['ok'])
        self.assertTrue(any('dénominateur' in e for e in bilan['erreurs']))

    def test_presence_pas_et_trajectoire(self):
        (self.rep/'COMPLETE').unlink()
        lignes = lire_csv(self.rep/'bloc3.csv')
        lignes[0]['pas'] += 1
        post.ecrire_csv(self.rep/'bloc3.csv', post.COLONNES, lignes)
        mesures = lire_csv(self.rep/'measurements.csv')
        mesures[1]['P0'] = math.inf
        post.ecrire_csv(self.rep/'measurements.csv', tuple(mesures[0]), mesures)
        erreurs = post.verifier_h13(self.rep)['erreurs']
        for motif in ('COMPLETE', 'pas incorrects', 'non fini'):
            self.assertTrue(any(motif in e for e in erreurs), erreurs)

    def test_bloc2_usage_sans_lancement(self):
        p = post.configurations(self.rep)[0]
        p.unlink()
        with patch.object(post.subprocess, 'run', side_effect=AssertionError('lancement interdit')):
            rc, _ = appel(post.main, ['bloc2', str(self.rep), '--kvu', '/absent'])
            self.assertEqual(rc, 2)
            (self.rep/'bloc3.csv').unlink()
            self.assertEqual(appel(post.main, ['bloc2', str(self.rep), '--kvu', '/absent'])[0], 2)

    def test_elagage_sans_bloc2_et_reprise(self):
        avant = sorted(p.name for p in self.rep.iterdir())
        rc, _ = appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4'])
        self.assertEqual(rc, 0)
        self.assertEqual(avant, sorted(p.name for p in self.rep.iterdir() if p.name != 'ELAGAGE_PROPOSE.json'))
        self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4', '--execute'])[0], 2)
        self.assertEqual(appel(post.main, ['h13', str(self.rep)])[0], 0)
        self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4', '--execute'])[0], 0)
        self.assertEqual(len(post.configurations(self.rep)), 3)
        bilan = post.verifier_h13(self.rep)
        self.assertTrue(bilan['ok'], bilan)
        self.assertEqual(bilan['elagues'], 9)
        self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '8', '--execute'])[0], 0)
        bilan = post.verifier_h13(self.rep)
        self.assertTrue(bilan['ok'], bilan)
        self.assertEqual(bilan['elagues'], 11)
        self.assertTrue((self.rep/'therm.csv').is_file())

    def test_elagage_refuse_marqueur_perime(self):
        self.assertEqual(appel(post.main, ['h13', str(self.rep)])[0], 0)
        (self.rep/'measurements.csv').write_text('cassé\n')
        self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4', '--execute'])[0], 2)
        self.assertEqual(len(post.configurations(self.rep)), 12)

    @unittest.skipUnless(KVU_BIN, 'KVU_BIN absent : intégration du blocage CPU ignorée')
    @patch.dict(os.environ, {'KEVINOTRON_SSD_MIN_GO': '1'})
    def test_deuxieme_bloc_manifeste_et_elagage(self):
        # Le tmpfs de test a moins de 100 Go : conserver une réserve de 1 Go
        # uniquement pour ces petits fichiers ; aucune politique du dépôt ne change.
        args = ['bloc2', str(self.rep), '--kvu', KVU_BIN, '--fils', '1']
        rc, texte = appel(post.main, args)
        self.assertEqual(rc, 0, texte)
        self.assertEqual(len(lire_csv(self.rep/'bloc3b.csv')), 12)
        self.assertTrue(all(r['n_boucles'] == NB_BOUCLES * 4 * r['l_grossier']**4
                            for r in lire_csv(self.rep/'bloc3b.csv')))
        self.assertEqual(len(lire_csv(self.rep/'bloc3b_hist.csv')), 12)
        avant = {p.name: p.stat().st_mtime_ns for p in post.configurations(self.rep, True)}
        with patch.object(post.subprocess, 'run', side_effect=AssertionError('reprise a relancé le binaire')):
            self.assertEqual(appel(post.main, args + ['--garder-existants'])[0], 0)
        self.assertEqual(avant, {p.name: p.stat().st_mtime_ns for p in post.configurations(self.rep, True)})
        self.assertEqual(len(lire_csv(self.rep/'bloc3b.csv')), 12)
        rc, texte = appel(post.main, ['h13', str(self.rep), '--mutants'])
        self.assertEqual(rc, 0, texte)
        rc, texte = appel(post.main, ['manifeste', str(self.rep), '--kvu', KVU_BIN])
        self.assertEqual(rc, 0, texte)
        m = post.lire_json(self.rep/'regen_bloc3b.json')
        self.assertEqual(m['classe_regen'], 'C')
        self.assertEqual(len(m['fichiers']), 26)
        self.assertEqual(len(m['entrees']), 12)
        validation = subprocess.run([sys.executable, '-I', str(DEPOT/'scripts/regen_valider.py'),
                                     str(self.rep/'regen_bloc3b.json')], capture_output=True, text=True)
        self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
        # Une sortie corrompue doit être régénérée pendant la reprise.
        p = post.configurations(self.rep, True)[0]
        p.write_bytes(b'invalide')
        self.assertEqual(appel(post.main, args + ['--garder-existants'])[0], 0)
        # La validation refuse un manifeste dérivé d'une source malformée.
        source = post.lire_json(self.rep/'regen_bloc3.json')
        source['parametres']['L'] = 0
        post.ecrire_json(self.rep/'regen_bloc3.json', source)
        (self.rep/'regen_bloc3b.json').unlink()
        rc, texte = appel(post.main, ['manifeste', str(self.rep), '--kvu', KVU_BIN])
        self.assertEqual(rc, 1, texte)
        self.assertFalse((self.rep/'regen_bloc3b.json').exists())
        self.assertFalse((self.rep/'regen_bloc3b.json.part').exists())
        source['parametres']['L'] = 18
        post.ecrire_json(self.rep/'regen_bloc3.json', source)
        self.assertEqual(appel(post.main, ['manifeste', str(self.rep), '--kvu', KVU_BIN])[0], 0)
        for execute in ([], ['--execute']):
            self.assertEqual(appel(post.main, ['elaguer', str(self.rep), '--un-sur', '4'] + execute)[0], 0)
        rc, texte = appel(post.main, ['h13', str(self.rep)])
        self.assertEqual(rc, 0, texte)
        self.assertIn('élagués : 18', texte)
        self.assertEqual(len(post.configurations(self.rep, True)), 3)
        # Les CSV, manifestes et historiques sont conservés.
        self.assertTrue((self.rep/'bloc3b.csv').exists())
        self.assertEqual(len(list(self.rep.glob('bloc3b_cfg_*.manifeste.json'))), 12)
        self.assertEqual(appel(post.main, ['manifeste', str(self.rep), '--kvu', KVU_BIN])[0], 0)
        m = post.lire_json(self.rep/'regen_bloc3b.json')
        self.assertEqual(len(m['fichiers']), 17)  # 3 configurations, 12 manifestes, 2 CSV


class AnalyseCampagne(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory(prefix='wp10-analyse-')
        self.addCleanup(self.temporaire.cleanup)
        self.racine = Path(self.temporaire.name)/'E'
        self.racine.mkdir()
        self.taches = []
        self.creer()

    def creer(self, n=256, decalage_p=0., decalage_r=0., mutant=.04, fraction=0., ambigues=0):
        self.taches = []
        for i, (ident, genre, machine, decalage) in enumerate((
                ('P1', 'P', 'gamer', decalage_p), ('P2', 'P', {'nom': 'gamer'}, decalage_p),
                ('P3', 'P', 'gamer', decalage_p), ('C', 'C', 'acemagic', 0.),
                ('R', 'R', {'nom': 'acemagic'}, decalage_r), ('M', 'M', 'gamer', mutant))):
            rep = self.racine/ident
            rep.mkdir(exist_ok=True)
            tache = dict(id=ident, genre=genre, cellule='L18_b04', L=18, beta=4., nmeas=n, therm=20, skip=2)
            tache['beta_run'] = 4.008 if genre == 'M' else 4.
            self.taches.append(tache)
            (rep/'COMPLETE').touch()
            (rep/'H13_PASS').touch()
            for nom in ('regen_bloc3.json', 'regen_bloc3b.json'):
                post.ecrire_json(rep/nom, {})
            post.ecrire_json(rep/'manifest.json', dict(machine=machine))
            bruit = np.random.default_rng(922+i).normal(size=n)
            bruit = .01*(bruit-bruit.mean())
            valeurs = .4+decalage+bruit
            mesures = [dict(measurement_index=0, total_steps=20, phase='initial', plaquette=-.99, P0=0., P1=0., P2=0., P3=0.)]
            mesures += [dict(measurement_index=m, total_steps=20+2*m, phase='sample', plaquette=float(v),
                              P0=0., P1=0., P2=0., P3=0.) for m, v in enumerate(valeurs, 1)]
            post.ecrire_csv(rep/'measurements.csv', tuple(mesures[0]), mesures)
            for fichier, facteur in (('bloc3.csv', .9), ('bloc3b.csv', .8)):
                cote = 6 if fichier == 'bloc3.csv' else 2
                lignes = [dict(pas=20+2*m, l_grossier=cote, n_boucles=NB_BOUCLES*4*cote**4, fraction_hors=fraction,
                               theta_max=.1, n_ambigues=ambigues, plaq_grossiere=float(facteur*v), sha256_cfg='a'*64)
                          for m, v in enumerate(valeurs, 1)]
                post.ecrire_csv(rep/fichier, post.COLONNES, lignes)

    def resultat(self, bloc3b=False):
        return analyse.analyse(self.racine, self.taches, bloc3b)

    def test_pass_complet_et_initial_exclu(self):
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(len(r['loi2']), 3)
        self.assertEqual(len(r['recert']), 3)
        self.assertTrue(all(m['ok'] for m in r['mutants_recert']))
        self.assertEqual(r['chaines']['P1']['statistiques']['P']['n'], 256)
        self.assertAlmostEqual(r['chaines']['P1']['statistiques']['P']['moy'], .4, delta=1e-15)
        self.assertAlmostEqual(r['cellules']['L18_b04']['groupes']['F32']['P']['moy'], .4)
        self.assertTrue(all(d['etiquette'] == 'DANS_DOMAINE' for d in r['domaine']))
        self.assertFalse(any(t['STAT_PAUVRE'] for t in r['tau']))

    def test_loi2_decalage_cinq_sigma(self):
        r = self.resultat()
        groupes = r['cellules']['L18_b04']['groupes']
        sigma = math.hypot(groupes['F32']['P']['err'], groupes['F64']['P']['err'])
        self.creer(decalage_p=5*sigma)
        r = self.resultat()
        self.assertFalse(r['ok'])
        p = next(v for v in r['loi2'] if v['observable'] == 'P')
        self.assertAlmostEqual(p['z'], 5., delta=1e-12)
        self.assertFalse(p['ok'])
        self.assertIn('FAIL WP10-LOI2 TOUT', r['verdicts'])

    def test_mutant_vert_echec_outil(self):
        self.creer(mutant=0.)
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertFalse(r['mutants_loi2'][0]['ok'])
        self.assertTrue(any('FAIL WP10-LOI2-MUT' in v and 'ECHEC_OUTIL' in v for v in r['verdicts']))

    def test_mutant_absent_echec_global(self):
        (self.racine/'M'/'COMPLETE').unlink()
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertIn('M', r['absents'])
        self.assertIn('INFO WP10-ABSENT M', r['verdicts'])
        self.assertIn('FAIL WP10-LOI2 TOUT', r['verdicts'])

    def test_recert_rouge(self):
        r = self.resultat()
        c = r['recert'][0]
        sigma = math.hypot(c['gauche']['err'], c['droite']['err'])
        self.creer(decalage_r=5*sigma)
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertTrue(all(not c['ok'] for c in r['recert']))
        self.assertTrue(all(c['ok'] for c in r['mutants_recert']))

    def test_machine_inconnue(self):
        for machine in (None, 'inconnue', {}, {'nom': 'ordinateur-sans-origine'}):
            with self.subTest(machine=machine):
                post.ecrire_json(self.racine/'P1'/'manifest.json', dict(machine=machine))
                r = self.resultat()
                self.assertFalse(r['ok'])
                self.assertTrue(any(v.startswith('FAIL WP10-RECERT') and 'machine inconnue' in v for v in r['verdicts']))

    def test_domaine_hors_jamais_fail(self):
        for fraction, ambigues, etiquette in ((1e-3, 0, 'DANS_DOMAINE'), (1.001e-3, 0, 'HORS_DOMAINE'), (0., 1, 'HORS_DOMAINE')):
            with self.subTest(fraction=fraction, ambigues=ambigues):
                self.creer(fraction=fraction, ambigues=ambigues)
                r = self.resultat()
                self.assertTrue(r['ok'], r['verdicts'])
                self.assertTrue(all(d['etiquette'] == etiquette for d in r['domaine']))

    def test_domaine_une_configuration_suffit(self):
        p = self.racine/'P2'/'bloc3b.csv'
        lignes = lire_csv(p)
        lignes[3]['fraction_hors'] = .1
        post.ecrire_csv(p, post.COLONNES, lignes)
        r = self.resultat()
        self.assertEqual([d['etiquette'] for d in r['domaine']], ['DANS_DOMAINE', 'HORS_DOMAINE'])
        self.assertEqual(r['domaine'][1]['dans'], 3*256-1)

    def test_stat_pauvre(self):
        self.creer(n=4, mutant=.1)
        r = self.resultat()
        self.assertTrue(all(t['STAT_PAUVRE'] for t in r['tau']))
        self.assertTrue(any('STAT_PAUVRE' in v for v in r['verdicts']))

    def test_bloc3b_optionnel_ou_exige(self):
        for ident in ('P1', 'P2', 'P3', 'C', 'R', 'M'):
            (self.racine/ident/'bloc3b.csv').unlink()
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(len(r['loi2']), 2)
        self.assertFalse(self.resultat(True)['ok'])

    def test_nan_et_indices_refuses(self):
        p = self.racine/'P1'/'measurements.csv'
        lignes = lire_csv(p)
        lignes[2]['plaquette'] = math.nan
        lignes[1]['measurement_index'] = 0
        post.ecrire_csv(p, tuple(lignes[0]), lignes)
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertTrue(any(v.startswith('FAIL WP10-DONNEES P1') for v in r['verdicts']))

    def test_erreurs_nulles_sans_division(self):
        for ident in ('P1', 'P2', 'P3', 'C', 'R', 'M'):
            p = self.racine/ident/'measurements.csv'
            lignes = lire_csv(p)
            for r in lignes:
                r['plaquette'] = .5
            post.ecrire_csv(p, tuple(lignes[0]), lignes)
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertTrue(math.isnan(r['loi2'][0]['z']))
        self.assertFalse(r['mutants_recert'][0]['ok'])
        json.dumps(analyse.json_fini(r), allow_nan=False)

    def test_cli_isole_sorties_stables(self):
        plan = self.racine.parent/'plan.json'
        post.ecrire_json(plan, dict(schema='wp10-plan/1', taches=self.taches))
        sortie = self.racine.parent/'rapport'
        commande = [sys.executable, '-I', str(ICI/'wp10_analyse.py'), '--racine', str(self.racine),
                    '--plan', str(plan), '--sortie', str(sortie), '--json', '--bloc3b']
        execution = subprocess.run(commande, text=True, capture_output=True)
        self.assertEqual(execution.returncode, 0, execution.stdout + execution.stderr)
        r = post.lire_json(sortie/'analyse.json')
        self.assertEqual(execution.stdout, (sortie/'verdicts.txt').read_text())
        self.assertEqual(execution.stdout.splitlines(), r['verdicts'])
        self.assertTrue(all(v.startswith(('PASS ', 'FAIL ', 'INFO ')) for v in r['verdicts']))
        self.assertIn('| Cellule |', (sortie/'tables.md').read_text())
        self.assertIn('τ (mesures)', (sortie/'tables.md').read_text())
        execution2 = subprocess.run(commande, text=True, capture_output=True)
        self.assertEqual(execution2.stdout, execution.stdout)

    def test_inconcluant_loi2_et_recert(self):
        self.creer(n=16, mutant=.1)
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(r['verdict_loi2'], 'SKIP')
        for gate in ('WP10-LOI2', 'WP10-RECERT'):
            lignes = [v for v in r['verdicts'] if f'{gate} L18_b04 ' in v]
            self.assertEqual(len(lignes), 3)
            self.assertTrue(all(v.startswith('SKIP ') and 'INCONCLUANT' in v
                                and 'converge=' in v and 'n_eff=' in v and 'z=' in v for v in lignes))
        self.assertIn('SKIP WP10-LOI2 TOUT inconclusif ou absent', r['verdicts'])
        self.assertIn('| SKIP |', analyse.tables(r))

    def test_controle_f64_judge_seul(self):
        # La production demeure riche ; le contrôle seul n'atteint pas Sokal.
        p = self.racine/'C'/'measurements.csv'
        lignes = lire_csv(p)
        for i, ligne in enumerate(lignes[1:]):
            ligne['plaquette'] = .4 + 1e-4*(i - 127.5)
        post.ecrire_csv(p, tuple(lignes[0]), lignes)
        r = self.resultat()
        groupes = r['cellules']['L18_b04']['groupes']
        self.assertTrue(analyse.conclusive(groupes['F32']['P']))
        self.assertFalse(groupes['F64']['P']['converge'])
        self.assertEqual(next(c for c in r['loi2'] if c['observable'] == 'P')['verdict'], 'SKIP')
        self.assertTrue(any('F64, converge=False' in v for v in r['verdicts']))

    def test_verdict_global_trois_etats(self):
        self.assertEqual(self.resultat()['verdict_loi2'], 'PASS')
        self.taches = [t for t in self.taches if t['genre'] != 'M']
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(r['verdict_loi2'], 'SKIP')
        self.creer(mutant=0.)
        self.assertEqual(self.resultat()['verdict_loi2'], 'FAIL')

    def test_comparaison_attendue_manquante(self):
        # Sans --bloc3b, une Pb2 absente d'un seul côté reste attendue.
        (self.racine/'C'/'bloc3b.csv').unlink()
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(r['verdict_loi2'], 'SKIP')
        self.assertTrue(any(v.startswith('SKIP WP10-LOI2 L18_b04 Pb2') for v in r['verdicts']))

    def test_recert_observable_manquante(self):
        (self.racine/'R'/'bloc3b.csv').unlink()
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertIn('SKIP WP10-RECERT L18_b04 Pb2 INCONCLUANT (observable absente) z=nan', r['verdicts'])
        self.assertEqual(r['verdict_loi2'], 'SKIP')

    def test_fail_prioritaire_sur_skip(self):
        self.creer(n=16, mutant=0.)
        r = self.resultat()
        self.assertTrue(any(v.startswith('SKIP WP10-LOI2 L18_b04') for v in r['verdicts']))
        self.assertTrue(any(v.startswith('FAIL WP10-LOI2-MUT') for v in r['verdicts']))
        self.assertEqual(r['verdict_loi2'], 'FAIL')

    def test_completude_tache_manquante(self):
        shutil.rmtree(self.racine/'P1')
        r = self.resultat()
        self.assertFalse(r['ok'])
        self.assertEqual(r['verdict_loi2'], 'FAIL')
        self.assertIn('FAIL WP10-COMPLETUDE TOUT 5/6 tâches', r['verdicts'])
        self.assertTrue(any(v.startswith('FAIL WP10-COMPLETUDE P1 ') for v in r['verdicts']))
        self.assertFalse(any(v.startswith('PASS WP10-LOI2 L18_b04 ') for v in r['verdicts']))

    def test_completude_marqueurs_et_manifestes(self):
        for nom in ('COMPLETE', 'H13_PASS', 'regen_bloc3.json', 'regen_bloc3b.json'):
            with self.subTest(nom=nom):
                p = self.racine/'P1'/nom
                contenu = p.read_bytes()
                p.unlink()
                r = self.resultat(True)
                self.assertFalse(r['ok'])
                self.assertIn(f'FAIL WP10-COMPLETUDE P1 {nom} absent', r['verdicts'])
                p.write_bytes(contenu)

    def test_completude_nombres_et_indices(self):
        for nom in ('measurements.csv', 'bloc3.csv', 'bloc3b.csv'):
            p = self.racine/'P1'/nom
            lignes = lire_csv(p)
            colonnes = tuple(lignes[0])
            for defaut in ('nombre', 'doublon', 'ordre'):
                with self.subTest(nom=nom, defaut=defaut):
                    copie = copy.deepcopy(lignes)
                    cle = 'measurement_index' if nom == 'measurements.csv' else 'pas'
                    if defaut == 'nombre':
                        copie.pop()
                    elif defaut == 'doublon':
                        copie[2][cle] = copie[1][cle]
                    else:
                        copie[1], copie[2] = copie[2], copie[1]
                    post.ecrire_csv(p, colonnes, copie)
                    r = self.resultat(True)
                    self.assertFalse(r['ok'])
                    self.assertTrue(any(v.startswith(f'FAIL WP10-COMPLETUDE P1 {nom}') for v in r['verdicts']))
            post.ecrire_csv(p, colonnes, lignes)

    def test_genre_n_et_puissance(self):
        rep = self.racine/'N'
        shutil.copytree(self.racine/'C', rep)
        tache = dict(next(t for t in self.taches if t['genre'] == 'M'), id='N', genre='N', beta_run=4.0008)
        self.taches.append(tache)
        base = self.resultat()['chaines']['C']['statistiques']['P']
        decalage = 5.6*math.hypot(base['err'], base['err'])
        p = rep/'measurements.csv'
        lignes = lire_csv(p)
        for ligne in lignes[1:]:
            ligne['plaquette'] += decalage
        post.ecrire_csv(p, tuple(lignes[0]), lignes)
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertIn('mutant_faible', r['cellules']['L18_b04']['groupes'])
        faible = r['mutants_faibles'][0]
        self.assertAlmostEqual(faible['z'], 5.6, delta=1e-12)
        self.assertAlmostEqual(faible['puissance'], .5*(1+math.erf(2.6/math.sqrt(2))), delta=1e-12)
        lignes = [v for v in r['verdicts'] if 'WP10-LOI2-MUT-FAIBLE' in v]
        self.assertEqual(len(lignes), 1)
        self.assertTrue(lignes[0].startswith('INFO '))
        self.assertIn('attendu ≈ 5,6', lignes[0])

    def test_mde_pente_et_beta_equivalent(self):
        r = self.resultat()
        self.assertEqual(len(r['mde']), 3)
        groupes = r['cellules']['L18_b04']['groupes']
        for m in r['mde']:
            obs = m['observable']
            sigma = math.hypot(groupes['F32'][obs]['err'], groupes['F64'][obs]['err'])
            self.assertEqual(m['sigma_diff'], sigma)
            self.assertEqual(m['MDE50'], 3*sigma)
            self.assertEqual(m['MDE90'], 4.282*sigma)
        m = next(v for v in r['mde'] if v['observable'] == 'P')
        self.assertAlmostEqual(m['pente'], .04/(4.008-4), delta=1e-10)
        self.assertAlmostEqual(m['decalage_relatif_beta'], m['MDE90']/(abs(m['pente'])*4), delta=1e-15)
        self.assertTrue(any('INFO WP10-LOI2-MDE' in v and 'MDE90=4.282σ=' in v for v in r['verdicts']))

    def test_equivalence_et_reference(self):
        r = self.resultat()
        self.assertEqual(len(r['equivalence']), 6)
        self.assertTrue(all(not e['equivalent'] for e in r['equivalence']))
        # Une très faible dispersion à moyenne commune permet l'équivalence.
        for ident in ('P1', 'P2', 'P3', 'C', 'R'):
            rep = self.racine/ident
            for nom, cle in (('measurements.csv', 'plaquette'), ('bloc3.csv', 'plaq_grossiere'),
                             ('bloc3b.csv', 'plaq_grossiere')):
                p = rep/nom
                lignes = lire_csv(p)
                moyenne = sum(l[cle] for l in lignes[1:])/len(lignes[1:]) if nom == 'measurements.csv' else sum(l[cle] for l in lignes)/len(lignes)
                for ligne in lignes:
                    ligne[cle] = moyenne + 1e-5*(ligne[cle]-moyenne)
                post.ecrire_csv(p, tuple(lignes[0]), lignes)
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertTrue(all(e['equivalent'] for e in r['equivalence']))
        for e in r['equivalence']:
            comparaison = r['loi2'] if e['comparaison'].startswith('LOI2 ') else r['recert']
            c = next(c for c in comparaison if c['observable'] == e['observable'])
            self.assertEqual(e['seuil'], 2.1e-6*abs(c['droite']['moy']))
            self.assertEqual(e['equivalent'], e['delta'] + 1.645*e['sigma_diff'] < e['seuil'])

    def test_equivalence_reference_decalee(self):
        self.creer(decalage_p=.001, decalage_r=.002)
        r = self.resultat()
        for e in r['equivalence']:
            obs = e['observable']
            if e['comparaison'].startswith('RECERT '):
                c = next(c for c in r['recert'] if c['observable'] == obs)
                self.assertNotEqual(c['gauche']['moy'], c['droite']['moy'])
                self.assertEqual(e['seuil'], 2.1e-6*abs(c['droite']['moy']))

    def test_domaine_denominateur_inconnu(self):
        p = self.racine/'P1'/'bloc3.csv'
        lignes = lire_csv(p)
        lignes[0]['n_boucles'] -= 1
        post.ecrire_csv(p, post.COLONNES, lignes)
        r = self.resultat()
        self.assertTrue(r['ok'], r['verdicts'])
        self.assertEqual(r['domaine'][0]['etiquette'], 'INCONNU')
        self.assertTrue(r['domaine'][0]['couverture_99'])
        self.assertEqual(r['domaine'][0]['dans'], 3*256-1)
        self.assertTrue(any('pas1 INCONNU' in v and 'couverture ≥ 99 % : oui' in v for v in r['verdicts']))

    def test_domaine_couverture_ne_change_pas_etiquette(self):
        p = self.racine/'P1'/'bloc3.csv'
        lignes = lire_csv(p)
        lignes[0]['fraction_hors'] = .1
        post.ecrire_csv(p, post.COLONNES, lignes)
        d = self.resultat()['domaine'][0]
        self.assertTrue(d['couverture_99'])
        self.assertEqual(d['etiquette'], 'HORS_DOMAINE')

    def test_tau_trois_observables_par_chaine(self):
        r = self.resultat()
        self.assertEqual({d['observable'] for d in r['tau']}, {'P', 'Pb1', 'Pb2'})
        for d in r['tau']:
            self.assertEqual(len(d['chaines']), 6)
            for c in d['chaines']:
                self.assertIn('fenetre', c)
                self.assertIn('converge', c)
                self.assertEqual(c['n_eff'], c['n']/(2*c['tau']))
                self.assertTrue(any(v.startswith(f"INFO WP10-TAU L18_b04 {d['observable']} {c['id']} ")
                                    and 'fenêtre=' in v and 'converge=' in v and 'n_eff=' in v for v in r['verdicts']))

    def test_recert_mutant_ecart_negatif(self):
        self.creer(decalage_r=.01)
        r = self.resultat()
        self.assertTrue(all(m['z'] < -5 and m['ok'] for m in r['mutants_recert']))

    def test_cli_skip_code_zero(self):
        self.creer(n=16, mutant=.1)
        plan = self.racine.parent/'plan.json'
        post.ecrire_json(plan, self.taches)
        rc, texte = appel(analyse.main, ['--racine', str(self.racine), '--plan', str(plan),
                                         '--sortie', str(self.racine.parent/'rapport')])
        self.assertEqual(rc, 0, texte)
        self.assertIn('SKIP WP10-LOI2 TOUT inconclusif ou absent', texte)


if __name__ == '__main__':
    unittest.main()
