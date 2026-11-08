"""Bilan complet : un état explicitement illisible ne peut être vert."""
import json
import unittest
import test_wp10_reconstat as fixture
import wp10_bilan as bilan

class BilanLogAbsent(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture.Reconstat()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def modifier(self,kernel=True,log=False,legacy=0):
        p=self.fixture.E/'etat/A.json'
        e=json.loads(p.read_text())
        e['fenetres'][0]['journal_lisible']=kernel
        e['fenetres'][0]['log_lisible']=log
        e['journal_noyau_illisible']=legacy
        p.write_text(json.dumps(e))

    def bilan(self):
        self.assertEqual(fixture.appel(bilan.main,self.fixture.bilan_args('--sans-analyse')),0)
        return (self.fixture.base/'bilan.md').read_text()

    def test_log_illisible_refuse_pass(self):
        self.modifier()
        text=self.bilan()
        self.assertIn('SKIP WP10-ALERTE-IGPU 0 alerte(s)',text)
        self.assertNotIn('PASS WP10-ALERTE-IGPU',text)

    def test_deux_journaux_illisibles_une_seule_fenetre(self):
        self.modifier(kernel=False,log=False,legacy=1)
        text=self.bilan()
        self.assertIn('illisible pour 1 fenêtre(s)',text)
        self.assertNotIn('illisible pour 2 fenêtre(s)',text)

    def test_compteur_historique_conserve(self):
        self.modifier(kernel=True,log=True,legacy=2)
        self.assertIn('illisible pour 2 fenêtre(s)',self.bilan())

if __name__=='__main__':unittest.main()
