# SPDX-License-Identifier: Apache-2.0
"""Données synthétiques uniquement ; aucun journalctl/ssh réel."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import wp10_bilan as bilan
import wp10_bilan_rejoue as rejoue
import wp10_reconstat as reconstat

ICI = Path(__file__).resolve().parent
FAUTE = ('2026-10-26T03:28:20+01:00 acemagic kernel: amdgpu 0000:c5:00.0: '
         'amdgpu: [gfxhub] page fault Process kvu pid 2387416\n')
SAIN = '2026-10-26T03:28:20+01:00 gamer kernel: démarrage normal\n'
LOG_FAUTE = 'Queue error - HSA_STATUS_ERROR_MEMORY_FAULT\n'


def instantane(dossier):
    return sorted((str(p.relative_to(dossier)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in dossier.rglob('*'))


def appel(f, args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return f(args)


class Reconstat(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.E = self.base / 'E'
        (self.E / 'etat/logs').mkdir(parents=True)
        (self.E / 'outils').mkdir()
        (self.E / 'analyse').mkdir()
        (self.E / 'analyse/verdicts.txt').write_bytes(b'PASS WP10-TEST analyse figee\n')
        # Analyse figée synthétique : vérifie le routage et le contrôle d'identité,
        # sans introduire de données numériques ou réécrire les outils réels.
        (self.E / 'outils/wp10_analyse.py').write_text('''import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--racine'); p.add_argument('--plan'); p.add_argument('--sortie'); p.add_argument('--bloc3b', action='store_true')
a = p.parse_args()
d = Path(a.sortie); d.mkdir(parents=True, exist_ok=True)
(d / 'verdicts.txt').write_bytes((Path(a.racine) / 'analyse/verdicts.txt').read_bytes())
(d / 'tables.md').write_text('table synthétique\\n')
print('PASS WP10-TEST analyse figee')
''')
        self.rapport = self.base / 'rapport.md'
        self.rapport.write_bytes(b'# Rapport\r\nTexte avant : \xc3\xa9\r\n\r\n' + bilan.DEBUT.encode() +
                                 b'\nancien bloc\n' + bilan.FIN.encode() + b'\r\nTexte apres\r\n')
        self.config = self.E / 'orch_config.json'
        self.config.write_text(json.dumps({'racine': str(self.E), 'rapport': str(self.rapport), 'ssh': 'machine-test'}))
        plan = [{'id': 'A'}, {'id': 'B'}, {'id': 'C'}]
        (self.E / 'plan.json').write_text(json.dumps({'taches': plan}))
        # Une fenêtre rc=3 fautive, une rc=0 saine, une rc=143 saine sur gam.
        from datetime import datetime
        centre = datetime.fromisoformat('2026-10-26T03:28:20+01:00').timestamp()
        self.centre = centre
        for ident, rc, machine in [('A', 3, 'ace'), ('B', 0, 'ace'), ('C', 143, 'gam')]:
            t0 = centre - 10 if ident != 'B' else centre + 100
            w = {'k': 1, 'rc': rc, 't0': t0, 't1': t0 + 20, 'duree_s': 20, 'calcul_s': 10,
                 'machine': machine, 'verrou': 'excl', 'fautes_gpu': [], 'journal_lisible': True}
            e = {'id': ident, 'statut': 'post_ok', 'machine': machine, 'fenetres': [w], 'alertes': 0}
            (self.E / 'etat' / f'{ident}.json').write_text(json.dumps(e))
            d = self.E / 'etat/logs' / ident
            d.mkdir()
            (d / 'fenetre_001.log').write_text(LOG_FAUTE if ident == 'A' else 'Terminé, 10.00 s\n')
        self.ace, self.gam = self.base / 'ace.txt', self.base / 'gam.txt'
        self.ace.write_text(FAUTE + '2026-10-26T13:23:00+01:00 ace kernel: amdgpu: GPU reset\n')
        self.gam.write_text(SAIN)
        self.sortie = self.base / 'constat'
        self.args = ['--config', str(self.config), '--sortie', str(self.sortie),
                     '--journal', f'ace={self.ace}', '--journal', f'gam={self.gam}']

    def produire(self):
        # Une régression qui tente une lecture réelle échoue immédiatement.
        with patch.object(reconstat.subprocess, 'run', side_effect=AssertionError('commande réelle interdite')):
            self.assertEqual(appel(reconstat.main, self.args), 0)
        return json.loads((self.sortie / 'constat.json').read_bytes())

    def bilan_args(self, *options):
        return ['--config', str(self.config), '--sortie', str(self.base / 'bilan.md'), *options]

    def test_constat_et_sources_intactes(self):
        avant = instantane(self.E)
        r = self.produire()
        self.assertEqual(r['totaux'], {'fenetres_controlees': 3, 'fenetres_fautives': 1, 'illisibles': 0})
        a = r['taches']['A'][0]
        self.assertEqual(a['rc'], 3)
        self.assertEqual(a['fautes_noyau'], [FAUTE.strip()])
        self.assertEqual(a['fautes_log'], LOG_FAUTE.splitlines())
        self.assertIn('FAIL WP10-ALERTE-IGPU-RECONSTAT', (self.sortie / 'constat.txt').read_text())
        self.assertEqual((self.sortie / 'journal_noyau_ace.txt').read_text(), FAUTE)
        for nom, sha in r['sha256_fichiers_lus'].items():
            self.assertEqual(hashlib.sha256(Path(nom).read_bytes()).hexdigest(), sha)
        self.assertEqual(instantane(self.E), avant)

    def test_reproduction_pass_puis_fail_et_analyse_deplacee(self):
        self.produire()
        avant = instantane(self.E)
        args = self.bilan_args('--analyse-sortie', str(self.base / 'analyse'))
        self.assertEqual(appel(bilan.main, args), 0)
        self.assertIn('PASS WP10-ALERTE-IGPU 0 alerte(s)', (self.base / 'bilan.md').read_text())
        self.assertEqual(appel(bilan.main, args + ['--constat', str(self.sortie / 'constat.json')]), 0)
        txt = (self.base / 'bilan.md').read_text()
        self.assertIn('FAIL WP10-ALERTE-IGPU 1 alerte(s)', txt)
        self.assertIn('INFO WP10-ALERTE-IGPU-CONSTAT', txt)
        self.assertIn('A fenêtre 1 rc=3', txt)
        self.assertRegex(txt, r'\| A \| ace \| post_ok .*\| 1 \|')
        self.assertEqual(instantane(self.E), avant)
        self.assertTrue((self.base / 'analyse/verdicts.txt').is_file())

    def test_bilan_defaut_identique_ancien(self):
        ancien = subprocess.run(['git', '-C', str(ICI), 'show', '6056036:crates/kvu/validation/wp10/wp10_bilan.py'],
                                capture_output=True, check=True).stdout
        p = self.base / 'ancien_bilan.py'
        p.write_bytes(ancien)
        # Sans aucune des deux nouvelles options, avec l'analyse par défaut.
        args = self.bilan_args()
        subprocess.run([sys.executable, '-B', str(p), *args], check=True, capture_output=True)
        original = (self.base / 'bilan.md').read_bytes()
        self.assertEqual(appel(bilan.main, args), 0)
        nouveau = (self.base / 'bilan.md').read_bytes()
        normaliser = lambda b: re.sub(rb'le \d{4}-\d\d-\d\d \d\d:\d\d:\d\d', b'le DATE', b)
        self.assertEqual(normaliser(original), normaliser(nouveau))

    def test_rejeu_extraits_sauves_identique(self):
        r = self.produire()
        autre = self.base / 'replay'
        args = ['--config', str(self.config), '--sortie', str(autre)]
        for m in ('ace', 'gam'):
            args += ['--journal', f'{m}={self.sortie}/journal_noyau_{m}.txt']
        with patch.object(reconstat.subprocess, 'run', side_effect=AssertionError('commande réelle interdite')):
            self.assertEqual(appel(reconstat.main, args), 0)
        r2 = json.loads((autre / 'constat.json').read_bytes())
        self.assertEqual(r['taches'], r2['taches'])
        self.assertEqual(r['totaux'], r2['totaux'])

    def test_rejeu_extraits_conserve_illisibilite(self):
        self.ace.write_text('sans date\n')
        r = self.produire()
        args = ['--config', str(self.config), '--sortie', str(self.base / 'replay-illisible')]
        for m in ('ace', 'gam'):
            args += ['--journal', f'{m}={self.sortie}/journal_noyau_{m}.txt']
        with patch.object(reconstat.subprocess, 'run', side_effect=AssertionError('commande réelle interdite')):
            self.assertEqual(appel(reconstat.main, args), 0)
        r2 = json.loads((self.base / 'replay-illisible/constat.json').read_bytes())
        self.assertEqual(r['totaux'], r2['totaux'])
        self.assertFalse(r2['taches']['A'][0]['lisible'])
        self.assertFalse(r2['taches']['B'][0]['lisible'])

    def test_extrait_altere_refuse(self):
        self.produire()
        (self.sortie / 'journal_noyau_ace.txt').write_text('')
        args = ['--config', str(self.config), '--sortie', str(self.base / 'altere'),
                '--journal', f'ace={self.sortie}/journal_noyau_ace.txt', '--journal', f'gam={self.gam}']
        self.assertEqual(appel(reconstat.main, args), 1)
        self.assertFalse((self.base / 'altere').exists())

    def test_illisible_skip_et_faute_prioritaire(self):
        self.ace.write_text('format sans date\n')
        r = self.produire()
        self.assertEqual(r['totaux']['illisibles'], 2)
        self.assertEqual(r['totaux']['fenetres_fautives'], 1)  # log reste déterminant
        (self.E / 'etat/logs/A/fenetre_001.log').write_text('Terminé\n')
        self.produire()
        self.assertIn('SKIP WP10-ALERTE-IGPU-RECONSTAT', (self.sortie / 'constat.txt').read_text())
        appel(bilan.main, self.bilan_args('--sans-analyse', '--constat', str(self.sortie / 'constat.json')))
        self.assertIn('SKIP WP10-ALERTE-IGPU', (self.base / 'bilan.md').read_text())

    def test_pass_sans_faute(self):
        self.ace.write_text('')
        (self.E / 'etat/logs/A/fenetre_001.log').write_text('Terminé\n')
        r = self.produire()
        self.assertEqual(r['totaux']['fenetres_fautives'], 0)
        self.assertIn('PASS WP10-ALERTE-IGPU-RECONSTAT', (self.sortie / 'constat.txt').read_text())

    def test_log_manquant_inconcluant(self):
        (self.E / 'etat/logs/B/fenetre_001.log').unlink()
        r = self.produire()
        self.assertEqual(r['totaux']['illisibles'], 1)
        self.assertFalse(r['taches']['B'][0]['log_lisible'])

    def test_max_alertes_et_constat_complet(self):
        p = self.E / 'etat/A.json'
        e = json.loads(p.read_bytes())
        e['alertes'] = 2
        p.write_text(json.dumps(e))
        self.produire()
        appel(bilan.main, self.bilan_args('--sans-analyse', '--constat', str(self.sortie / 'constat.json')))
        self.assertIn('FAIL WP10-ALERTE-IGPU 2 alerte(s)', (self.base / 'bilan.md').read_text())
        e['fenetres'][0]['t1'] += 1
        p.write_text(json.dumps(e))
        with self.assertRaisesRegex(ValueError, 'périmé'):
            appel(bilan.main, self.bilan_args('--sans-analyse', '--constat', str(self.sortie / 'constat.json')))

    def test_intervalle_et_machine_distante_simulee(self):
        args = ['--config', str(self.config), '--sortie', str(self.sortie)]
        def commande(argv, **kwargs):
            if argv[0] == 'ssh':
                self.assertEqual(argv[:4], ['ssh', '-o', 'BatchMode=yes', 'machine-test'])
                self.assertIn('--until', argv[-1])
                return subprocess.CompletedProcess(argv, 1, '', 'refus simulé')
            self.assertIn('--since', argv)
            self.assertIn('--until', argv)
            return subprocess.CompletedProcess(argv, 0, FAUTE if argv[argv.index('--since') + 1] == f'@{self.centre - 15:.6f}' else '', '')
        with patch.object(reconstat.subprocess, 'run', side_effect=commande) as mock:
            self.assertEqual(appel(reconstat.main, args), 0)
            self.assertEqual(mock.call_count, 3)
        r = json.loads((self.sortie / 'constat.json').read_bytes())
        self.assertEqual(r['totaux']['illisibles'], 1)
        self.assertFalse(r['taches']['C'][0]['lisible'])

    def test_bornes_fichier_et_fuseaux(self):
        from datetime import datetime, timezone
        t = self.centre
        lignes = []
        for offset in (-15.001, -15, 15, 15.001):
            lignes.append(datetime.fromtimestamp(t + offset, timezone.utc).isoformat() + ' kernel: normal\n')
        r = reconstat.filtrer_journal(''.join(lignes).encode(), t - 10, t + 10)
        self.assertEqual(r, ''.join(lignes[1:3]))

    def test_sortie_dans_E_refusee(self):
        avant = instantane(self.E)
        args = list(self.args)
        args[args.index('--sortie') + 1] = str(self.E / 'interdit')
        self.assertEqual(appel(reconstat.main, args), 1)
        self.assertEqual(avant, instantane(self.E))

    def test_journal_argument_invalide(self):
        for valeur in ('ace', 'ace=', 'autre=journal'):
            with self.subTest(valeur=valeur), self.assertRaises(ValueError):
                reconstat.journaux_arguments([valeur])

    def args_rejeu(self, dossier, ecrire=False):
        args = ['--config', str(self.config), '--travail', str(self.base / dossier),
                '--journal', f'ace={self.ace}', '--journal', f'gam={self.gam}']
        if ecrire:
            args += ['--ecrire', '--commit', 'correctif-test']
        return args

    def test_rejeu_sans_ecrire(self):
        avant = instantane(self.E)
        rapport = self.rapport.read_bytes()
        self.assertEqual(appel(rejoue.main, self.args_rejeu('travail')), 0)
        self.assertEqual(self.rapport.read_bytes(), rapport)
        self.assertEqual(instantane(self.E), avant)
        d = self.base / 'travail'
        archive = next(d.glob('BILAN-AUTO_avant_*.md'))
        self.assertEqual(archive.read_bytes(), rejoue.decouper(rapport)[1])
        self.assertEqual(Path(str(archive) + '.sha256').read_text().strip(), hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertIn('-ancien bloc', (d / 'diff_bloc.txt').read_text())
        self.assertIn('+FAIL WP10-ALERTE-IGPU 1 alerte(s)', (d / 'diff_bloc.txt').read_text())
        p = json.loads((d / 'provenance.json').read_bytes())
        self.assertEqual(p['sha256_hors_bloc_avant'], p['sha256_hors_bloc_apres'])
        self.assertEqual(p['sha256_verdicts_reference'], p['sha256_verdicts_rejoues'])

    def test_rejeu_avec_ecrire_bloc_et_avenant(self):
        avant = instantane(self.E)
        original = self.rapport.read_bytes()
        self.assertEqual(appel(rejoue.main, self.args_rejeu('publication', True)), 0)
        nouveau = self.rapport.read_bytes()
        a, ancien, z = rejoue.decouper(original)
        b, bloc, fin = rejoue.decouper(nouveau)
        self.assertEqual(a, b)
        self.assertNotEqual(ancien, bloc)
        self.assertTrue(fin.startswith(z))
        self.assertIn(b'FAIL WP10-ALERTE-IGPU 1 alerte(s)', bloc)
        avenant = fin[len(z):].decode()
        self.assertIn('## Avenant WP10-SUITE du ', avenant)
        self.assertIn('correctif-test', avenant)
        self.assertIn('ligne 369', avenant)
        self.assertIn('verdicts.txt identique', avenant)
        self.assertIn('Toutes les autres lignes modifiées', avenant)
        self.assertIn('sha256_extraits', avenant)
        self.assertEqual(instantane(self.E), avant)

    def test_analyse_divergente_interdit_publication(self):
        p = self.E / 'outils/wp10_analyse.py'
        p.write_text(p.read_text().replace("read_bytes())", "read_bytes() + b'DIFFERENT')"))
        original = self.rapport.read_bytes()
        self.assertEqual(appel(rejoue.main, self.args_rejeu('divergent', True)), 1)
        self.assertEqual(original, self.rapport.read_bytes())

    def test_rejeu_analyse_reelle_inchangee(self):
        # Exécute aussi les vrais modules, copiés sans modification dans E/outils.
        plan = json.loads((self.E / 'plan.json').read_bytes())
        for t in plan['taches']:
            t.update(nmeas=2, cellule='L18_b04', genre='P')
        (self.E / 'plan.json').write_text(json.dumps(plan))
        for nom in ('wp10_analyse.py', 'wp10_stat.py', 'wp10_post.py'):
            shutil.copyfile(ICI / nom, self.E / 'outils' / nom)
        r = subprocess.run([sys.executable, '-B', str(self.E / 'outils/wp10_analyse.py'),
                            '--racine', str(self.E), '--plan', str(self.E / 'plan.json'),
                            '--sortie', str(self.E / 'analyse'), '--bloc3b'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)  # FAIL scientifique attendu : sorties absentes.
        self.assertTrue((self.E / 'analyse/verdicts.txt').read_bytes().startswith(b'FAIL WP10-COMPLETUDE'))
        avant = instantane(self.E)
        self.assertEqual(appel(rejoue.main, self.args_rejeu('analyse-reelle')), 0)
        self.assertEqual(instantane(self.E), avant)
        self.assertEqual((self.E / 'analyse/verdicts.txt').read_bytes(),
                         (self.base / 'analyse-reelle/analyse/verdicts.txt').read_bytes())

    def test_ecrire_exige_commit(self):
        args = self.args_rejeu('sans_commit') + ['--ecrire']
        self.assertEqual(appel(rejoue.main, args), 1)
        self.assertFalse((self.base / 'sans_commit').exists())

    def test_verdicts_perimes_refuses(self):
        d = self.base / 'perime/analyse'
        d.mkdir(parents=True)
        shutil.copyfile(self.E / 'analyse/verdicts.txt', d / 'verdicts.txt')
        original = self.rapport.read_bytes()
        self.assertEqual(appel(rejoue.main, self.args_rejeu('perime', True)), 1)
        self.assertEqual(original, self.rapport.read_bytes())


if __name__ == '__main__':
    unittest.main()
