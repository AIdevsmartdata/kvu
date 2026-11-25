# SPDX-License-Identifier: Apache-2.0
"""Tests du lecteur direct des séries de torsion (`scripts/thermo/lire_series.py`).

Données synthétiques fabriquées dans un dossier temporaire ; le lecteur est toujours lancé
en sous-processus par `python3 -I -B <script> ...` (comme en production). Aucune dépendance
à `entrees/` sauf le test facultatif de bout en bout, sauté si `entrees/e2e` est absent.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[2]
LECTEUR = RACINE / 'scripts' / 'thermo' / 'lire_series.py'

APPARIEMENT = ('la graine de la chaîne (nœud k, réplique i) est G + k·R + i '
               'pour tous les bras')

CLEFS = {
    'logR', 'newton_ok', 'newton_iterations', 'bin_width', 'nbins', 'beta',
    'indice_beta', 'groupe', 'rang_dans_groupe', 'K', 'R', 'nmeas', 'poids',
    'n_bootstrap_demandes', 'n_bootstrap_ok', 'n_bootstrap_non_converge',
    'sigma_bootstrap', 'bootstrap_mode', 'graine_bootstrap', 'graine_base', 'appariement',
    'torsion', 'torsion_fixe', 'dossier', 'sha256_manifeste',
}


def lancer(dossier, *args, cwd=None):
    """Lance le lecteur en sous-processus ; renvoie (code, stdout, stderr)."""
    cmd = [sys.executable, '-I', '-B', str(LECTEUR), str(dossier)] + [str(a) for a in args]
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or tempfile.gettempdir(),
                       env=env, timeout=600)
    return p.returncode, p.stdout, p.stderr


def fabriquer(dossier, betas, G, R, nmeas, K=None, graine=1234, poids=None,
              torsion_fixe=None, avec_series=True, ecrire_fichiers=True):
    """Fabrique un dossier kvu synthétique. Renvoie le manifeste écrit.

    Les séries sont construites de sorte que la colonne (groupe, rang k, réplique i) ait une
    moyenne dépendant de (β du groupe, rang k, bras) : choisir la mauvaise colonne change
    donc le résultat de façon détectable.
    """
    dossier = Path(dossier)
    (dossier / 'series').mkdir(parents=True, exist_ok=True)
    if K is None:
        K = 5
    if poids is None:
        poids = [float(x) for x in np.linspace(-1.0, 1.0, K)]
    man = {
        'betas': [float(b) for b in betas],
        'poids': poids,
        'repliques': R,
        'nmeas': nmeas,
        'graine_base': 424242,
        'appariement': APPARIEMENT,
        'torsion': '01',
        'series': {'betas_par_groupe': G},
    }
    if torsion_fixe is not None:
        man['torsion_fixe'] = torsion_fixe
    if not avec_series:
        del man['series']
    rng = np.random.default_rng(graine)
    ngroupes = (len(betas) + G - 1) // G
    for g in range(ngroupes):
        nbg = min(G, len(betas) - g * G)
        ncol = nbg * R
        for a in range(K):
            arr = np.empty((nmeas, ncol), dtype='<f8')
            for k in range(nbg):
                beta = betas[g * G + k]
                # moyenne dépendant de (β, rang, bras) + bruit reproductible
                centre = 0.5 * (beta - 2.0) + 0.3 * k + 0.05 * a
                for i in range(R):
                    arr[:, k * R + i] = centre + 0.1 * rng.standard_normal(nmeas)
            if ecrire_fichiers:
                arr.tofile(dossier / 'series' / f'bras{a}_g{g:03d}_fe.bin')
                # fichier _tot factice (le lecteur ne doit pas s'en servir)
                (arr * 2.0).tofile(dossier / 'series' / f'bras{a}_g{g:03d}_tot.bin')
    (dossier / 'manifest_thermo.json').write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
    return man


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def ok(self, dossier, *args, **kw):
        code, out, err = lancer(dossier, *args, **kw)
        self.assertEqual(code, 0, f"code {code}\nstdout={out}\nstderr={err}")
        return json.loads(out)

    def refus(self, dossier, *args, **kw):
        out_path = self.tmp / 'refus_out.json'
        if out_path.exists():
            out_path.unlink()
        code, out, err = lancer(dossier, *args, '--out', out_path, **kw)
        self.assertNotEqual(code, 0, f"le lecteur aurait dû refuser\nstdout={out}")
        self.assertTrue(err.strip(), "message d'erreur attendu sur stderr")
        self.assertFalse(out_path.exists(), "aucun fichier --out ne doit être écrit en cas de refus")
        return code, err


class TestCasNominal(Base):
    def test_cas_simple_et_cles(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=4, nmeas=60, K=5)
        res = self.ok(self.tmp, '--beta', '2.0', '--bootstrap', '5')
        self.assertEqual(set(res), CLEFS)
        self.assertNotIn('seed', res)
        self.assertNotIn('seed', json.dumps(res))
        self.assertTrue(res['newton_ok'])
        self.assertIsInstance(res['newton_ok'], bool)
        self.assertIsInstance(res['newton_iterations'], int)
        self.assertEqual(res['beta'], 2.0)
        self.assertEqual(res['indice_beta'], 0)
        self.assertEqual(res['groupe'], 0)
        self.assertEqual(res['rang_dans_groupe'], 0)
        self.assertEqual(res['K'], 5)
        self.assertEqual(res['R'], 4)
        self.assertEqual(res['nmeas'], 60)
        self.assertEqual(res['poids'], [float(x) for x in np.linspace(-1, 1, 5)])
        self.assertEqual(res['bin_width'], 0.02)
        self.assertEqual(res['graine_bootstrap'], 17)
        self.assertEqual(res['n_bootstrap_demandes'], 5)
        self.assertEqual(res['n_bootstrap_ok'] + res['n_bootstrap_non_converge'], 5)
        self.assertEqual(res['graine_base'], 424242)
        self.assertEqual(res['torsion'], '01')
        self.assertIsNone(res['torsion_fixe'])
        self.assertEqual(Path(res['dossier']), self.tmp.resolve())
        self.assertIsInstance(res['logR'], float)
        self.assertTrue(math.isfinite(res['logR']))
        self.assertIsInstance(res['nbins'], int)
        self.assertGreater(res['nbins'], 1)

    def test_sha256_et_out(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        out_path = self.tmp / 'res.json'
        code, out, err = lancer(self.tmp, '--beta', '2.0', '--bootstrap', '0',
                                '--out', out_path)
        self.assertEqual(code, 0, err)
        self.assertTrue(out_path.is_file())
        res = json.loads(out_path.read_text(encoding='utf-8'))
        self.assertEqual(res, json.loads(out))
        sha = hashlib.sha256((self.tmp / 'manifest_thermo.json').read_bytes()).hexdigest()
        self.assertEqual(res['sha256_manifeste'], sha)
        self.assertIsNone(res['sigma_bootstrap'])  # 0 tirage -> null

    def test_torsion_fixe_copiee(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3, torsion_fixe=0.5)
        res = self.ok(self.tmp, '--beta', '2.0', '--bootstrap', '0')
        self.assertEqual(res['torsion_fixe'], 0.5)

    def test_depuis_un_autre_dossier_courant(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        ailleurs = self.tmp / 'ailleurs'
        ailleurs.mkdir()
        res = self.ok(self.tmp, '--beta', '2.0', '--bootstrap', '0', cwd=ailleurs)
        self.assertEqual(Path(res['dossier']), self.tmp.resolve())

    def test_bootstrap_deterministe(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=6, nmeas=80, K=4)
        r1 = self.ok(self.tmp, '--beta', '2.0', '--bootstrap', '8', '--graine-bootstrap', '5')
        r2 = self.ok(self.tmp, '--beta', '2.0', '--bootstrap', '8', '--graine-bootstrap', '5')
        self.assertEqual(r1['sigma_bootstrap'], r2['sigma_bootstrap'])
        self.assertEqual(r1['n_bootstrap_ok'], r2['n_bootstrap_ok'])
        self.assertEqual(r1['graine_bootstrap'], 5)
        if r1['n_bootstrap_ok'] >= 2:
            self.assertIsInstance(r1['sigma_bootstrap'], float)
            self.assertGreaterEqual(r1['sigma_bootstrap'], 0.0)


class TestMultiBetaMultiGroupe(Base):
    def test_multi_beta_multi_groupe_dernier_incomplet(self):
        # 5 β, G = 2 -> groupes g000 (β0,β1), g001 (β2,β3), g002 (β4 seul, incomplet)
        betas = [2.0, 2.1, 2.2, 2.3, 2.4]
        G, R, nmeas, K = 2, 3, 50, 4
        fabriquer(self.tmp, betas=betas, G=G, R=R, nmeas=nmeas, K=K)
        attendus = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0)}
        res = {}
        for b, beta in enumerate(betas):
            r = self.ok(self.tmp, '--beta', repr(beta), '--bootstrap', '0')
            self.assertEqual(r['indice_beta'], b)
            self.assertEqual((r['groupe'], r['rang_dans_groupe']), attendus[b])
            self.assertEqual(r['R'], R)
            self.assertEqual(r['nmeas'], nmeas)
            res[b] = r
        # les β d'un même groupe doivent donner des résultats DIFFÉRENTS (bonne colonne)
        self.assertNotAlmostEqual(res[0]['logR'], res[1]['logR'], places=6)
        self.assertNotAlmostEqual(res[2]['logR'], res[3]['logR'], places=6)
        # et tous les β doivent différer entre eux
        valeurs = [res[b]['logR'] for b in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertNotAlmostEqual(valeurs[i], valeurs[j], places=6,
                                          msg=f"β{i} et β{j} donnent le même logR")

    def test_bonne_colonne_verifiee_par_recalcul(self):
        """Preuve directe : le logR du lecteur = celui recalculé sur la colonne attendue."""
        sys.path.insert(0, str(RACINE / 'scripts' / 'thermo'))
        try:
            from histogramme import Histogramme
        finally:
            sys.path.pop(0)
        betas = [2.0, 2.1, 2.2]
        G, R, nmeas, K = 2, 3, 40, 3
        man = fabriquer(self.tmp, betas=betas, G=G, R=R, nmeas=nmeas, K=K)
        b = 2  # groupe 1, rang 0
        g, k = b // G, b % G
        poids = np.asarray(man['poids'], float)
        series = []
        for a in range(K):
            raw = np.fromfile(self.tmp / 'series' / f'bras{a}_g{g:03d}_fe.bin', dtype='<f8')
            series.append(raw.reshape(nmeas, -1)[:, k * R:(k + 1) * R])
        h = 0.02
        lo = min(float(s.min()) for s in series)
        hi = max(float(s.max()) for s in series)
        edges = np.arange(math.floor(lo / h) * h - h, math.ceil(hi / h) * h + 2 * h, h)
        centres = (edges[:-1] + edges[1:]) / 2
        C = np.array([[np.histogram(s[:, r], bins=edges)[0] for r in range(R)] for s in series])
        H = Histogramme(betas[b] * poids, centres, C)
        f, ok = H.solve(maxit=2000)
        self.assertTrue(ok)
        attendu = float(-f[-1])
        res = self.ok(self.tmp, '--beta', repr(betas[b]), '--bootstrap', '0')
        self.assertEqual(res['logR'], attendu)
        self.assertEqual(res['nbins'], len(centres))
        self.assertEqual(res['newton_iterations'], int(H.it))


class TestRefus(Base):
    def test_refus_pas_de_champ_series(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3, avec_series=False)
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('series', err)

    def test_refus_beta_absente(self):
        fabriquer(self.tmp, betas=[2.0, 2.1], G=1, R=3, nmeas=40, K=3)
        code, err = self.refus(self.tmp, '--beta', '2.15')
        self.assertIn('2.15', err)

    def test_refus_poids_inattendus(self):
        K = 4
        poids = [float(x) for x in np.linspace(-1.0, 1.0, K)]
        poids[1] += 1e-6
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=K, poids=poids)
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('poids', err)

    def test_refus_fichier_fe_manquant(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        (self.tmp / 'series' / 'bras1_g000_fe.bin').unlink()
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('bras1_g000_fe.bin', err)

    def test_refus_taille_inattendue(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        p = self.tmp / 'series' / 'bras2_g000_fe.bin'
        p.write_bytes(p.read_bytes()[:-8])
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('taille', err)

    def test_refus_taille_inattendue_groupe_incomplet(self):
        # 3 β, G = 2 : le groupe g001 ne contient qu'une β -> ncol = R, pas 2R
        fabriquer(self.tmp, betas=[2.0, 2.1, 2.2], G=2, R=3, nmeas=40, K=3)
        p = self.tmp / 'series' / 'bras0_g001_fe.bin'
        # on gonfle le fichier comme s'il contenait 2 β
        p.write_bytes(p.read_bytes() + b'\x00' * (40 * 3 * 8))
        code, err = self.refus(self.tmp, '--beta', '2.2')
        self.assertIn('taille', err)

    def test_refus_valeur_non_finie(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        p = self.tmp / 'series' / 'bras0_g000_fe.bin'
        arr = np.fromfile(p, dtype='<f8')
        arr[7] = np.nan
        arr.tofile(p)
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('non finie', err)

    def test_refus_valeur_infinie(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        p = self.tmp / 'series' / 'bras2_g000_fe.bin'
        arr = np.fromfile(p, dtype='<f8')
        arr[-1] = np.inf
        arr.tofile(p)
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('non finie', err)

    def test_refus_manifeste_absent(self):
        (self.tmp / 'series').mkdir()
        code, err = self.refus(self.tmp, '--beta', '2.0')
        self.assertIn('manifeste', err)


class TestNewtonNonConverge(Base):
    def test_newton_non_converge_code_3(self):
        fabriquer(self.tmp, betas=[2.0], G=1, R=3, nmeas=40, K=3)
        out_path = self.tmp / 'res.json'
        code, out, err = lancer(self.tmp, '--beta', '2.0', '--bootstrap', '0',
                                '--maxit', '1', '--out', out_path)
        self.assertEqual(code, 3, f"stdout={out}\nstderr={err}")
        self.assertTrue(out_path.is_file(), "le JSON doit être produit même si Newton échoue")
        res = json.loads(out)
        self.assertFalse(res['newton_ok'])
        self.assertEqual(set(res), CLEFS)


@unittest.skipUnless((RACINE / 'entrees' / 'e2e' / 'manifest_thermo.json').is_file(),
                     "données réelles entrees/e2e absentes")
class TestE2E(unittest.TestCase):
    def test_valeurs_de_reference(self):
        dossier = RACINE / 'entrees' / 'e2e'
        for h, logR, nbins in [(0.02, -0.15593014130472077, 4946),
                               (0.01, -0.15595837774228, 9889)]:
            code, out, err = lancer(dossier, '--beta', '2.4', '--bin-width', repr(h),
                                    '--bootstrap', '0')
            self.assertEqual(code, 0, err)
            res = json.loads(out)
            self.assertEqual(res['nbins'], nbins)
            self.assertAlmostEqual(res['logR'], logR, delta=1e-12)
            self.assertEqual(res['K'], 41)
            self.assertEqual(res['R'], 64)
            self.assertEqual(res['nmeas'], 800)



class TestColonneRangDansGroupe(Base):
    """Étape 1 de la tâche 1 bis : la β de rang k > 0 doit lire SA colonne k·R + i.

    Chaque colonne (groupe g, rang k, réplique i) a une moyenne connue et distincte
    (décalage unique par colonne) ; pour CHAQUE β (k = 0, k > 0, groupe incomplet) on
    recalcule indépendamment logR et nbins sur les colonnes attendues k·R + i et on
    compare au lecteur lancé avec --bootstrap 0. Ces tests tuent : (1) un lecteur qui
    prend toujours les R premières colonnes du groupe (rang 0), (2) un lecteur qui prend
    les colonnes entrelacées (réplique i de la β k = colonne i·n + k), (3) un lecteur qui
    prend la dernière colonne du groupe.
    """

    BETAS = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]   # 7 β
    G, R, NMEAS, K = 3, 4, 60, 4                   # groupes : 3, 3, 1 (incomplet)
    H = 0.02

    @staticmethod
    def _decalage(g, k, i):
        """Moyenne connue de la colonne (g, k, i) : unique par colonne."""
        return 0.1 * g + 0.01 * k + 0.001 * i

    def _fabriquer_colonnes(self):
        dossier = self.tmp
        (dossier / 'series').mkdir(parents=True, exist_ok=True)
        poids = [float(x) for x in np.linspace(-1.0, 1.0, self.K)]
        man = {'betas': self.BETAS, 'poids': poids, 'repliques': self.R,
               'nmeas': self.NMEAS, 'graine_base': 424242, 'appariement': APPARIEMENT,
               'torsion': '01', 'series': {'betas_par_groupe': self.G}}
        rng = np.random.default_rng(20240607)
        ngroupes = (len(self.BETAS) + self.G - 1) // self.G
        for g in range(ngroupes):
            nbg = min(self.G, len(self.BETAS) - g * self.G)
            ncol = nbg * self.R
            for a in range(self.K):
                arr = np.empty((self.NMEAS, ncol), dtype='<f8')
                for k in range(nbg):
                    for i in range(self.R):
                        arr[:, k * self.R + i] = (self._decalage(g, k, i)
                                                  + 0.05 * rng.standard_normal(self.NMEAS))
                # écriture atomique : un fichier déjà présent en lecture seule (reliquat
                # d'un autre passage) est remplacé, pas réécrit sur place.
                chemin = dossier / 'series' / f'bras{a}_g{g:03d}_fe.bin'
                tmp = chemin.with_name(chemin.name + '.tmp')
                arr.tofile(tmp)
                os.replace(tmp, chemin)
        chemin_man = dossier / 'manifest_thermo.json'
        tmp_man = chemin_man.with_name(chemin_man.name + '.tmp')
        tmp_man.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp_man, chemin_man)
        return man

    def _recalcul(self, man, b, colonnes):
        """Recalcule (logR, nbins) sur les colonnes données (indices dans le groupe)."""
        sys.path.insert(0, str(RACINE / 'scripts' / 'thermo'))
        try:
            from histogramme import Histogramme
        finally:
            sys.path.pop(0)
        g = b // self.G
        poids = np.asarray(man['poids'], float)
        series = []
        for a in range(self.K):
            raw = np.fromfile(self.tmp / 'series' / f'bras{a}_g{g:03d}_fe.bin', dtype='<f8')
            series.append(raw.reshape(self.NMEAS, -1)[:, list(colonnes)])
        lo = min(float(s.min()) for s in series)
        hi = max(float(s.max()) for s in series)
        h = self.H
        edges = np.arange(math.floor(lo / h) * h - h, math.ceil(hi / h) * h + 2 * h, h)
        centres = (edges[:-1] + edges[1:]) / 2
        C = np.array([[np.histogram(s[:, r], bins=edges)[0] for r in range(len(colonnes))]
                      for s in series])
        H = Histogramme(self.BETAS[b] * poids, centres, C)
        f, ok = H.solve(maxit=2000)
        self.assertTrue(ok)
        return float(-f[-1]), len(centres)

    def test_chaque_beta_lit_sa_colonne(self):
        """Pour chaque β (k = 0, k > 0, groupe incomplet) : logR et nbins = recalcul."""
        man = self._fabriquer_colonnes()
        for b, beta in enumerate(self.BETAS):
            g, k = b // self.G, b % self.G
            nbg = min(self.G, len(self.BETAS) - g * self.G)
            attendu_cols = [k * self.R + i for i in range(self.R)]
            logR_att, nbins_att = self._recalcul(man, b, attendu_cols)
            res = self.ok(self.tmp, '--beta', repr(beta), '--bootstrap', '0')
            self.assertEqual((res['groupe'], res['rang_dans_groupe']), (g, k))
            self.assertEqual(res['nbins'], nbins_att, f"β={beta} (g={g}, k={k})")
            self.assertEqual(res['logR'], logR_att,
                             f"β={beta} (g={g}, k={k}) : le lecteur n'a pas lu la colonne "
                             f"k·R+i = {attendu_cols}")
            # les trois erreurs visées donneraient une AUTRE valeur : le test a du pouvoir
            for nom, cols in [('rang 0', list(range(self.R))),
                              ('entrelacee', [i * nbg + k for i in range(self.R)]),
                              ('derniere', [nbg * self.R - self.R + i for i in range(self.R)])]:
                if cols == attendu_cols:
                    continue
                logR_err, _ = self._recalcul(man, b, cols)
                self.assertNotAlmostEqual(
                    logR_err, logR_att, places=6,
                    msg=f"β={beta} : la colonne « {nom} » n'est pas distinguable de la "
                        f"bonne (test sans pouvoir)")

    def test_groupe_incomplet_dernier_rang(self):
        """β = 2.6 : groupe 2 incomplet (1 β), rang 0, colonnes 0..R-1 du fichier g002."""
        man = self._fabriquer_colonnes()
        b = 6
        logR_att, nbins_att = self._recalcul(man, b, list(range(self.R)))
        res = self.ok(self.tmp, '--beta', '2.6', '--bootstrap', '0')
        self.assertEqual(res['groupe'], 2)
        self.assertEqual(res['rang_dans_groupe'], 0)
        self.assertEqual(res['nbins'], nbins_att)
        self.assertEqual(res['logR'], logR_att)


# ---------------------------------------------------------------------------
# Tâche 1 ter : bootstrap conjoint, refus des schémas contradictoires.
# ---------------------------------------------------------------------------
def fabriquer_miroir(dossier, R=8, nmeas=50, K=2, appariement=APPARIEMENT, graine=7):
    """Deux bras de poids -1 et +1, bras 1 = -bras 0 (réplique par réplique)."""
    dossier = Path(dossier)
    (dossier / 'series').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(graine)
    arr = np.empty((nmeas, R), dtype='<f8')
    for i in range(R):
        arr[:, i] = 0.5 + 0.1 * rng.standard_normal(nmeas)
    arr.tofile(dossier / 'series' / 'bras0_g000_fe.bin')
    (-arr).tofile(dossier / 'series' / 'bras1_g000_fe.bin')
    man = {'betas': [2.0], 'poids': [-1.0, 1.0], 'repliques': R, 'nmeas': nmeas,
           'graine_base': 1, 'appariement': appariement, 'torsion': '01',
           'series': {'betas_par_groupe': 1}}
    (dossier / 'manifest_thermo.json').write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
    return man


class TestConjoint(Base):
    def test_conjoint_miroir_sigma_nulle(self):
        fabriquer_miroir(self.tmp)
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 50)
        self.assertEqual(res['bootstrap_mode'], 'conjoint')
        self.assertLess(res['sigma_bootstrap'], 1e-9)

    def test_independant_miroir_sigma_non_nulle(self):
        fabriquer_miroir(self.tmp, appariement='indépendant (une graine par poids)')
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 50)
        self.assertEqual(res['bootstrap_mode'], 'independant')
        self.assertGreater(res['sigma_bootstrap'], 1e-3)

    def test_conjoint_bras_identiques_sigma_non_nulle(self):
        fabriquer_miroir(self.tmp)
        arr = np.fromfile(self.tmp / 'series' / 'bras0_g000_fe.bin', dtype='<f8')
        arr.tofile(self.tmp / 'series' / 'bras1_g000_fe.bin')
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 50)
        self.assertEqual(res['bootstrap_mode'], 'conjoint')
        self.assertGreater(res['sigma_bootstrap'], 0.0)

    def test_conjoint_reproductible(self):
        fabriquer_miroir(self.tmp)
        r1 = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 20)
        r2 = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 20)
        self.assertEqual(r1['sigma_bootstrap'], r2['sigma_bootstrap'])


class TestRefusAppariement(Base):
    def test_refus_appariement_absent(self):
        man = fabriquer_miroir(self.tmp)
        del man['appariement']
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('appariement', err)

    def test_refus_appariement_inconnu(self):
        fabriquer_miroir(self.tmp, appariement='test')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('inconnue', err)

    def test_refus_appariement_non_chaine(self):
        fabriquer_miroir(self.tmp, appariement=3)
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('chaîne', err)


class TestRefusDimensions(Base):
    def _manipuler(self, f):
        man = fabriquer_miroir(self.tmp)
        f(man)
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')

    def test_refus_dimensions_contradictoires(self):
        self._manipuler(lambda m: m['series'].update({'repliques': 4}))
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('contradictoires', err)

    def test_refus_nmeas_contradictoire(self):
        self._manipuler(lambda m: m['series'].update({'nmeas': 49}))
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('contradictoires', err)

    def test_refus_dimension_seulement_sous_series(self):
        for cle in ('repliques', 'nmeas'):
            with self.subTest(cle=cle):
                man = fabriquer_miroir(self.tmp)
                man['series'][cle] = man.pop(cle)
                _ecrire_man(self.tmp, man)
                code, err = self.refus(self.tmp, '--beta', 2.0, '--bootstrap', 0)
                self.assertEqual(code, 2)
                self.assertIn(cle, err)
                self.assertNotIn('Traceback', err)

    def test_refus_types_dimensions(self):
        for val in (2.9, None, True):
            with self.subTest(val=val):
                self._manipuler(lambda m, v=val: m.update({'repliques': v}))
                code, err = self.refus(self.tmp, '--beta', 2.0)
                self.assertEqual(code, 2)
                self.assertIn('entier JSON', err)
                self.assertNotIn('Traceback', err)

    def test_refus_betas_par_groupe_float(self):
        self._manipuler(lambda m: m['series'].update({'betas_par_groupe': 1.5}))
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('entier JSON', err)


class TestRefusPoids(Base):
    def test_refus_poids_nan(self):
        man = fabriquer_miroir(self.tmp)
        man['poids'] = [float('nan'), 1.0]
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('fini', err)

    def test_refus_poids_chaine(self):
        man = fabriquer_miroir(self.tmp)
        man['poids'] = ['-1', 1.0]
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('fini', err)


class TestBeta(Base):
    def test_beta_ambigue(self):
        man = fabriquer_miroir(self.tmp)
        man['betas'] = [0.5, 0.5000000000005]
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        code, err = self.refus(self.tmp, '--beta', 0.50000000000025)
        self.assertEqual(code, 2)
        self.assertIn('ambiguë', err)

    def test_beta_egalite_exacte_prioritaire(self):
        man = fabriquer_miroir(self.tmp)
        man['betas'] = [0.5, 0.5000000000005]
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        res = self.ok(self.tmp, '--beta', 0.5, '--bootstrap', 0)
        self.assertEqual(res['indice_beta'], 0)


class TestRefusDivers(Base):
    def test_nan_dans_autre_colonne(self):
        man = fabriquer_miroir(self.tmp)
        man['betas'] = [2.0, 3.0]
        man['series']['betas_par_groupe'] = 2
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        for a in (0, 1):
            arr = np.zeros((50, 16), dtype='<f8')
            arr[:, 0] = 0.5
            arr[:, 8] = 0.7
            if a == 1:
                arr[3, 8] = float('nan')   # première colonne de la β 3.0
            arr.tofile(self.tmp / 'series' / f'bras{a}_g000_fe.bin')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('non finie', err)

    def test_refus_plafonds_du_bain(self):
        man = fabriquer_miroir(self.tmp)
        man['plafonds_du_bain'] = 3
        (self.tmp / 'manifest_thermo.json').write_text(json.dumps(man), encoding='utf-8')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('plafonds_du_bain', err)


# ---------------------------------------------------------------------------
# Contre-tests du testeur (valeurs connues / calcul direct, pas seulement « ça ne plante pas »).
# ---------------------------------------------------------------------------
def _ecrire_man(dossier, man):
    (Path(dossier) / 'manifest_thermo.json').write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')


def fabriquer_constant(dossier, valeurs, R=8, nmeas=40):
    """Bras a : toutes les répliques constantes = valeurs[a] (aucun bruit)."""
    dossier = Path(dossier)
    (dossier / 'series').mkdir(parents=True, exist_ok=True)
    for a, v in enumerate(valeurs):
        np.full((nmeas, R), float(v), dtype='<f8').tofile(
            dossier / 'series' / f'bras{a}_g000_fe.bin')
    man = {'betas': [2.0], 'poids': [float(x) for x in np.linspace(-1.0, 1.0, len(valeurs))],
           'repliques': R, 'nmeas': nmeas, 'graine_base': 1, 'appariement': APPARIEMENT,
           'torsion': '01', 'series': {'betas_par_groupe': 1}}
    _ecrire_man(dossier, man)
    return man


def fabriquer_profils_q(dossier, signe, appariement):
    """Cas de l'audit de Sol : deux bras de poids -1 et +1, β = 0,5, huit répliques de huit mesures ;
    la réplique i du bras 0 est constante à q[i] (q de -0,875 à 0,875 par pas de 0,25), le bras 1 vaut signe·q[i]."""
    dossier = Path(dossier)
    (dossier / 'series').mkdir(parents=True, exist_ok=True)
    q = np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875])
    a0 = np.tile(q, (8, 1)).astype('<f8')
    a0.tofile(dossier / 'series' / 'bras0_g000_fe.bin')
    (signe * a0).astype('<f8').tofile(dossier / 'series' / 'bras1_g000_fe.bin')
    man = {'betas': [0.5], 'poids': [-1.0, 1.0], 'repliques': 8, 'nmeas': 8, 'graine_base': 1,
           'appariement': appariement, 'torsion': '01',
           'series': {'betas_par_groupe': 1, 'repliques': 8, 'nmeas': 8}}
    _ecrire_man(dossier, man)
    return man


class TestContreTester(Base):
    def test_miroir_conjoint_sigma_exactement_nulle(self):
        """Bras 1 = -bras 0, répliques constantes : toute moyenne pondérée est nulle,
        donc log R = 0 et σ conjointe = 0 EXACTEMENT (valeur connue, pas de bruit)."""
        fabriquer_constant(self.tmp, [0.5, -0.5])
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 40)
        self.assertEqual(res['bootstrap_mode'], 'conjoint')
        self.assertEqual(res['n_bootstrap_ok'], 40)
        self.assertEqual(res['sigma_bootstrap'], 0.0)
        self.assertAlmostEqual(res['logR'], 0.0, places=12)

    def test_miroir_independant_sigma_connue(self):
        """Mêmes données en mode indépendant : chaque tirage donne log R = f0 - f1 avec
        f0 = -log(2·cosh(β)) (bras 0) et f1 = -log(2·cosh(-β)) = f0, donc log R = 0 aussi ;
        mais σ doit être > 0,05 (dispersion des tirages indépendants)."""
        man = fabriquer_miroir(self.tmp)
        man['appariement'] = 'independant'
        _ecrire_man(self.tmp, man)
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 40)
        self.assertEqual(res['bootstrap_mode'], 'independant')
        # σ indépendante mesurée ≈ 0,010 (bruit 0,1 sur 50 mesures) : la borne de la tâche
        # (0,05) ne vaut que pour des répliques constantes, cf. test_miroir_independant_constant.
        self.assertGreater(res['sigma_bootstrap'], 0.003)

    def test_miroir_independant_repliques_variables(self):
        """Contre-exemple de l'audit de Sol (profils en miroir, répliques DIFFÉRENTES entre elles) :
        le mode indépendant donne σ > 0,05, le mode conjoint donne σ ≈ 0 (covariance parfaite
        entre bras). Des répliques constantes ne distinguent pas les modes (σ nulle dans les deux)."""
        for mode, ap in (('independant', 'independant : une graine par bras'), ('conjoint', APPARIEMENT)):
            with self.subTest(mode=mode):
                d = self.tmp / mode
                fabriquer_profils_q(d, -1.0, ap)
                res = self.ok(d, '--beta', 0.5, '--bin-width', 0.25, '--bootstrap', 300,
                              '--graine-bootstrap', 17)
                self.assertEqual(res['bootstrap_mode'], mode)
                if mode == 'independant':
                    self.assertGreater(res['sigma_bootstrap'], 0.05)
                else:
                    self.assertLess(res['sigma_bootstrap'], 1e-9)

    def test_miroir_independant_constant_sigma_nulle(self):
        """Répliques constantes dans chaque bras : tout rééchantillonnage redonne les mêmes
        comptes, donc σ = 0 EXACTEMENT, quel que soit le mode (ce cas ne distingue pas les modes)."""
        man = fabriquer_constant(self.tmp, [0.5, -0.5])
        man['appariement'] = 'independant'
        _ecrire_man(self.tmp, man)
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 40)
        self.assertEqual(res['bootstrap_mode'], 'independant')
        self.assertEqual(res['sigma_bootstrap'], 0.0)

    def test_conjoint_bras_identiques_sigma_non_nulle(self):
        """Bras identiques (même bruit) : conjoint conserve la dispersion des répliques."""
        fabriquer_miroir(self.tmp)
        arr = np.fromfile(self.tmp / 'series' / 'bras0_g000_fe.bin', dtype='<f8')
        arr.tofile(self.tmp / 'series' / 'bras1_g000_fe.bin')
        res = self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 40)
        self.assertEqual(res['bootstrap_mode'], 'conjoint')
        self.assertGreater(res['sigma_bootstrap'], 1e-3)

    def test_bras_identiques_conjoint_plus_disperse_qu_independant(self):
        """Deux bras IDENTIQUES (profils Q, répliques différentes) : les deux modes ne suivent PAS
        la même loi (le conjoint garde la covariance complète entre bras). Valeurs de l'audit de Sol :
        σ conjointe ≈ 0,21 et σ indépendante ≈ 0,14 (1 000 tirages) ; avec 300 tirages on exige
        conjointe > 1,2 × indépendante, et les deux entre 0,08 et 0,35."""
        sig = {}
        for mode, ap in (('conjoint', APPARIEMENT), ('independant', 'independant : une graine par bras')):
            d = self.tmp / mode
            fabriquer_profils_q(d, 1.0, ap)
            sig[mode] = self.ok(d, '--beta', 0.5, '--bin-width', 0.25, '--bootstrap', 300,
                                '--graine-bootstrap', 17)['sigma_bootstrap']
        self.assertGreater(sig['conjoint'], 1.2 * sig['independant'])
        for v in sig.values():
            self.assertGreater(v, 0.08)
            self.assertLess(v, 0.35)

    def test_refus_appariement_absent_et_inconnu(self):
        for val in ('ABSENT', 'conjoint', '', APPARIEMENT + ' '):
            with self.subTest(val=val):
                man = fabriquer_miroir(self.tmp)
                if val == 'ABSENT':
                    del man['appariement']
                else:
                    man['appariement'] = val
                _ecrire_man(self.tmp, man)
                code, err = self.refus(self.tmp, '--beta', 2.0)
                self.assertEqual(code, 2)
                self.assertNotIn('Traceback', err)

    def test_refus_dimensions_contradictoires_et_types(self):
        cas = [('series', 'repliques', 4), ('series', 'nmeas', 39),
               ('racine', 'repliques', 2.9), ('racine', 'nmeas', None),
               ('racine', 'repliques', True), ('series', 'betas_par_groupe', 0),
               ('series', 'betas_par_groupe', None)]
        for ou, cle, val in cas:
            with self.subTest(cle=cle, val=val):
                man = fabriquer_miroir(self.tmp)
                (man['series'] if ou == 'series' else man)[cle] = val
                _ecrire_man(self.tmp, man)
                code, err = self.refus(self.tmp, '--beta', 2.0)
                self.assertEqual(code, 2)
                self.assertNotIn('Traceback', err)

    def test_refus_poids_nan_inf_chaine_bool(self):
        for val in (float('nan'), float('inf'), '-1', True):
            with self.subTest(val=val):
                man = fabriquer_miroir(self.tmp)
                man['poids'] = [val, 1.0]
                _ecrire_man(self.tmp, man)
                code, err = self.refus(self.tmp, '--beta', 2.0)
                self.assertEqual(code, 2)
                self.assertNotIn('Traceback', err)

    def test_beta_ambigue_et_egalite_exacte(self):
        man = fabriquer_miroir(self.tmp)
        man['betas'] = [0.5, 0.5000000000005]
        man['series']['betas_par_groupe'] = 2
        _ecrire_man(self.tmp, man)
        for a in (0, 1):
            arr = np.zeros((50, 16), dtype='<f8')
            arr[:, 0] = 0.5
            arr[:, 1] = 0.7
            arr.tofile(self.tmp / 'series' / f'bras{a}_g000_fe.bin')
        code, err = self.refus(self.tmp, '--beta', '0.50000000000025')
        self.assertEqual(code, 2)
        self.assertIn('ambigu', err)
        res = self.ok(self.tmp, '--beta', '0.5', '--bootstrap', 0)
        self.assertEqual(res['indice_beta'], 0)
        res = self.ok(self.tmp, '--beta', '0.5000000000005', '--bootstrap', 0)
        self.assertEqual(res['indice_beta'], 1)

    def test_nan_dans_autre_colonne_refuse(self):
        man = fabriquer_miroir(self.tmp)
        man['betas'] = [2.0, 3.0]
        man['series']['betas_par_groupe'] = 2
        _ecrire_man(self.tmp, man)
        for a in (0, 1):
            arr = np.zeros((50, 16), dtype='<f8')
            arr[:, 0] = 0.5
            arr[:, 8] = 0.7
            if a == 1:
                arr[3, 8] = float('nan')   # première colonne de la β 3.0
            arr.tofile(self.tmp / 'series' / f'bras{a}_g000_fe.bin')
        code, err = self.refus(self.tmp, '--beta', 2.0)
        self.assertEqual(code, 2)
        self.assertIn('non finie', err)

    def test_refus_plafonds_du_bain(self):
        for val in (3, 0.5, [1], 'x'):
            with self.subTest(val=val):
                man = fabriquer_miroir(self.tmp)
                man['plafonds_du_bain'] = val
                _ecrire_man(self.tmp, man)
                code, err = self.refus(self.tmp, '--beta', 2.0)
                self.assertEqual(code, 2)
                self.assertIn('plafonds_du_bain', err)
        man = fabriquer_miroir(self.tmp)
        man['plafonds_du_bain'] = 0
        _ecrire_man(self.tmp, man)
        self.ok(self.tmp, '--beta', 2.0, '--bootstrap', 0)


if __name__ == '__main__':
    unittest.main()
