# SPDX-License-Identifier: Apache-2.0
"""Contrôle GPU sans GPU, shell, ssh ni journalctl réel."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from wp10_orch import Orch

NOYAU = ('2026-10-26T03:28:20+01:00 acemagic kernel: amdgpu 0000:c5:00.0: '
         'amdgpu: [gfxhub] page fault (src_id:0 ring:40 vmid:2 pasid:32774) Process kvu pid 2387416')
LOG = 'Queue error - HSA_STATUS_ERROR_MEMORY_FAULT\nkevinotron su2 : GPU : hipMemcpyDtoHAsync : an illegal memory access was encountered (700)\n'
# Lignes de R54-b16-c2/fenetre_004.log, relevées en lecture seule.
NORMAL = ('# progression : pas 3030/4000 (75.8 %), 1.09 pas/s, reste ~15 min\n'
          '# progression : pas 3070/4000 (76.8 %), 1.19 pas/s, reste ~13 min\n')


class ControleFautes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.E = Path(self.temp.name)
        (self.E / 'etat/logs/T').mkdir(parents=True)
        (self.E / 'plan.json').write_text(json.dumps({'taches': [{'id': 'T'}]}))
        cfg = self.E / 'orch_config.json'
        cfg.write_text(json.dumps({'racine': str(self.E), 'journal': str(self.E / 'journal.txt')}))
        self.o = Orch(cfg)
        self.o.attendre_ressources = Mock(return_value=True)
        self.o.pool = Mock()
        self.o.log = Mock()
        self.o.sh = Mock(return_value=(0, NOYAU + '\n', ''))
        self.o.aide = Mock(side_effect=self.aide)
        sommeil = patch('wp10_orch.time.sleep', return_value=None)
        sommeil.start()
        self.addCleanup(sommeil.stop)

    def aide(self, machine, commande, *args):
        if commande == 'info':
            return {'existe': True, 'reprise': {'pas_faits': 2990}, 'complete': False}
        return {}

    def executer(self, rc=3, log=LOG, noyau=NOYAU, lisible=True):
        (self.E / 'etat/logs/T/fenetre_001.log').write_text(log)
        self.o.sh.return_value = (0 if lisible else 1, noyau, '')
        def fenetre(*args):
            self.o.arret.set()  # une seule fenêtre, puis état réellement sauvegardé
            return rc, 100.25, 200.75, 'excl'
        self.o.fenetre = Mock(side_effect=fenetre)
        self.o.executer_tache('ace', 'ace', {'id': 'T'})
        return self.o.etat('T')

    def test_decisif_rc3_faute_restaure(self):
        # N'importe aucun nouveau symbole : doit échouer PAR ASSERTION sur 6056036.
        e = self.executer()
        self.assertTrue(e['fenetres'][0]['fautes_gpu'])
        self.assertTrue(e['fenetres'][0]['journal_lisible'])
        self.assertEqual(e['alertes'], 1)
        self.assertTrue(any('ALERTE-IGPU' in c.args[1] for c in self.o.log.call_args_list))
        self.assertIn('ALERTE-IGPU', (self.E / 'journal.txt').read_text())
        self.assertTrue(any(c.args[1] == 'restaurer' for c in self.o.aide.call_args_list))
        self.assertEqual(e['fenetres'][0]['rc'], 3)

    def test_journal_illisible_rc_non_nul(self):
        e = self.executer(log=NORMAL, noyau='', lisible=False)
        self.assertFalse(e['fenetres'][0]['journal_lisible'])
        self.assertEqual(e['journal_noyau_illisible'], 1)
        self.assertEqual(e.get('alertes', 0), 0)

    def test_faute_seulement_log(self):
        e = self.executer(noyau='')
        self.assertEqual(e['alertes'], 1)
        self.assertTrue(e['fenetres'][0]['fautes_log'])
        self.assertFalse(e['fenetres'][0]['fautes_noyau'])

    def test_rc0_sain(self):
        e = self.executer(rc=0, noyau='', log=NORMAL)
        self.assertEqual(e.get('alertes', 0), 0)
        self.assertEqual(e['fenetres'][0]['fautes_gpu'], [])
        self.assertTrue(any(c.args[1] == 'noyer' for c in self.o.aide.call_args_list))
        self.assertFalse(any(c.args[1] == 'restaurer' for c in self.o.aide.call_args_list))

    def test_intervalle_borne(self):
        self.executer()
        argv = self.o.sh.call_args.args[1]
        self.assertEqual(argv[argv.index('--since') + 1], '@95.250000')
        self.assertEqual(argv[argv.index('--until') + 1], '@205.750000')

    def test_tous_rc_controles(self):
        for rc in (0, 3, 75, 99, 143):
            with self.subTest(rc=rc):
                self.o.arret.clear()
                self.o.chemin_etat('T').unlink(missing_ok=True)
                e = self.executer(rc=rc)
                self.assertEqual(e['alertes'], 1)
                self.assertTrue(e['fenetres'][0]['fautes_gpu'])

    def test_recidive_echec(self):
        self.o.sauver({'id': 'T', 'statut': 'en_cours', 'fenetres': [], 'alertes': 1, 'tentatives': 0})
        e = self.executer()
        self.assertEqual(e['alertes'], 2)
        self.assertEqual(e['statut'], 'echec')

    def test_detection_pure(self):
        from wp10_orch import detecter_fautes
        n = [NOYAU, 'kernel: NVRM: Xid (PCI:0000:01:00): 31', 'amdgpu: ring gfx timeout',
             'amdgpu: GPU reset', 'amdgpu: VM_CONTEXT fault', 'amdgpu: hang']
        negatifs = ['amdgpu: VRAM: 32768M', 'kernel: démarrage terminé']
        r = detecter_fautes(n + negatifs, NORMAL + LOG)
        self.assertEqual(r['fautes_noyau'], n)
        self.assertEqual(r['fautes_log'], LOG.splitlines())
        self.assertEqual(detecter_fautes(negatifs, NORMAL), {'fautes_noyau': [], 'fautes_log': []})
        self.assertTrue(detecter_fautes([], 'HIPERRORILLEGALADDRESS\nXid 31')['fautes_log'])


if __name__ == '__main__':
    unittest.main()
