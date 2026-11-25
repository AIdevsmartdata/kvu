# SPDX-License-Identifier: Apache-2.0
"""Tests de scripts/spectro/graine_coutures.py (unittest).

Lancement : python3 -I -B -m unittest discover -s scripts/tests -p test_graine_coutures.py -v
"""
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECTRO = os.path.join(RACINE, "scripts", "spectro")
SCRIPT = os.path.join(SPECTRO, "graine_coutures.py")
EXEMPLE = os.path.join(RACINE, "entrees", "exemple", "A")
sys.path.insert(0, SPECTRO)

import graine_coutures as gc  # noqa: E402


def ecrire_jeu(dossier, cotes, betas, mesures_par_chaine, phase_initial=True):
    """Fabrique un dossier au format `kvu su2 gen`."""
    os.makedirs(dossier, exist_ok=True)
    argv = ["kvu", "su2", "gen", "--cotes", ",".join(str(c) for c in cotes),
            "--decalage", "0,-3,0,0", "--backend", "cpu",
            "--beta", ",".join(repr(float(b)) for b in betas),
            "--graine", ",".join(str(i + 1) for i in range(len(betas))),
            "--outdir", dossier]
    with open(os.path.join(dossier, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "kevinotron-unifie/1", "argv": argv,
                   "tube": {"cotes": list(cotes), "betas": [float(b) for b in betas]}}, f)
    for i, mesures in enumerate(mesures_par_chaine):
        ch = os.path.join(dossier, "c%03d" % i)
        os.makedirs(ch, exist_ok=True)
        with open(os.path.join(ch, "geometrie.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "kvu-geometrie/1", "cotes": list(cotes)}, f)
        with open(os.path.join(ch, "measurements.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["measurement_index", "total_steps", "phase", "plaquette",
                        "P0", "P1", "P2", "P3"])
            if phase_initial:
                w.writerow([0, 5, "initial", 0.5, 0, 0, 0, 0])
            for k, v in enumerate(mesures):
                w.writerow([k + 1, 5, "sample", repr(float(v)), 0, 0, 0, 0])
    return dossier


def ar1(p, phi, sigma, n, rng):
    """Chaine AR(1) de moyenne p."""
    x = p
    out = []
    for _ in range(n):
        x = p + phi * (x - p) + rng.gauss(0.0, sigma)
        out.append(x)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gc_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lancer(self, *args):
        return subprocess.run([sys.executable, "-I", "-B", SCRIPT] + list(args),
                              capture_output=True, text=True)


class TestCalculs(Base):
    def test_estimation_ponctuelle_trapeze(self):
        betas = [1.0, 1.5, 2.0]
        a = ecrire_jeu(os.path.join(self.tmp, "A"), (3, 12, 8, 8), betas,
                       [[0.50] * 20, [0.55] * 20, [0.60] * 20])
        b = ecrire_jeu(os.path.join(self.tmp, "B"), (6, 12, 8, 8), betas,
                       [[0.45] * 20, [0.48] * 20, [0.50] * 20])
        sortie = os.path.join(self.tmp, "res.json")
        r = self.lancer(a, b, "--blocs", "4", "--sortie", sortie)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 4)
        with open(sortie, encoding="utf-8") as f:
            res = json.load(f)
        d = [0.05, 0.07, 0.10]
        integ = 0.5 * (d[0] + d[1]) / 2.0 + 0.5 * (d[1] + d[2]) / 2.0
        self.assertAlmostEqual(res["integrale"], integ, places=12)
        self.assertAlmostEqual(res["a3"], 6 * 4608 * integ, places=9)
        self.assertEqual(res["n_blocs"], 4)
        self.assertEqual(res["n_mesures"], 20)
        self.assertEqual(res["betas"], betas)
        self.assertEqual(res["sigma_jackknife"], 0.0)

    def test_ligne_initial_ignoree(self):
        betas = [1.0, 2.0]
        a = ecrire_jeu(os.path.join(self.tmp, "A"), (3, 12, 8, 8), betas,
                       [[0.50] * 10, [0.60] * 10])
        b = ecrire_jeu(os.path.join(self.tmp, "B"), (6, 12, 8, 8), betas,
                       [[0.40] * 10, [0.50] * 10])
        res = gc.calculer(a, b, 5)
        self.assertAlmostEqual(res["d"][0], 0.10, places=12)
        self.assertAlmostEqual(res["d"][1], 0.10, places=12)
        self.assertEqual(res["n_mesures"], 10)

    def test_jackknife_blocs_formule(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m, v = gc.jackknife_blocs(x, 3)
        self.assertAlmostEqual(m, 3.5, places=12)
        self.assertGreater(v, 0.0)
        # blocs de 2 : moyennes 1.5, 3.5, 5.5 -> pseudo (3*3.5 - mb)/2
        pseudo = [(3 * 3.5 - mb) / 2.0 for mb in (1.5, 3.5, 5.5)]
        mp = sum(pseudo) / 3.0
        attendu = (3 - 1) / 3.0 * sum((p - mp) ** 2 for p in pseudo)
        self.assertAlmostEqual(v, attendu, places=12)

    def test_determinisme(self):
        rng = random.Random(7)
        betas = [1.0, 1.5, 2.0]
        a = ecrire_jeu(os.path.join(self.tmp, "A"), (3, 12, 8, 8), betas,
                       [ar1(p, 0.5, 0.02, 60, rng) for p in (0.50, 0.55, 0.60)])
        b = ecrire_jeu(os.path.join(self.tmp, "B"), (6, 12, 8, 8), betas,
                       [ar1(p, 0.5, 0.02, 60, rng) for p in (0.45, 0.48, 0.50)])
        s1 = os.path.join(self.tmp, "r1.json")
        s2 = os.path.join(self.tmp, "r2.json")
        r1 = self.lancer(a, b, "--blocs", "6", "--sortie", s1)
        r2 = self.lancer(a, b, "--blocs", "6", "--sortie", s2)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(r1.stdout, r2.stdout)
        with open(s1, "rb") as f1, open(s2, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())


class TestRefus(Base):
    def _jeu(self, cotes_a=(3, 12, 8, 8), cotes_b=(6, 12, 8, 8), betas=(1.0, 2.0),
             n_a=2, n_b=2, n_mes=10):
        a = ecrire_jeu(os.path.join(self.tmp, "A"), cotes_a, betas,
                       [[0.5] * n_mes for _ in range(n_a)])
        b = ecrire_jeu(os.path.join(self.tmp, "B"), cotes_b, betas,
                       [[0.4] * n_mes for _ in range(n_b)])
        return a, b

    def _refus(self, a, b, *args, motif=None):
        sortie = os.path.join(self.tmp, "sortie.json")
        r = self.lancer(a, b, "--sortie", sortie, *args)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertFalse(os.path.exists(sortie))
        if motif:
            self.assertIn(motif, r.stderr.lower())
        return r

    def test_refus_blocs_invalides(self):
        a, b = self._jeu()
        self._refus(a, b, "--blocs", "1", motif="blocs")

    def test_refus_chaine_trop_courte(self):
        a, b = self._jeu(n_mes=5)
        self._refus(a, b, "--blocs", "6", motif="blocs")

    def test_refus_manifeste_illisible(self):
        a, b = self._jeu()
        with open(os.path.join(a, "manifest.json"), "w", encoding="utf-8") as f:
            f.write("{pas du json")
        self._refus(a, b, motif="manifeste")

    def test_refus_volume(self):
        a, b = self._jeu(cotes_b=(5, 12, 8, 8))
        self._refus(a, b, motif="v_b")

    def test_refus_nombre_de_chaines(self):
        a, b = self._jeu(n_b=3)
        self._refus(a, b, motif="chaine")

    def test_refus_valeur_non_finie(self):
        a, b = self._jeu()
        chemin = os.path.join(a, "c000", "measurements.csv")
        with open(chemin, encoding="utf-8") as f:
            lignes = f.read().splitlines()
        lignes[2] = lignes[2].replace(",0.5,", ",nan,")
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes) + "\n")
        self._refus(a, b)

    def test_refus_betas_differents(self):
        a = ecrire_jeu(os.path.join(self.tmp, "A"), (3, 12, 8, 8), [1.0, 2.0],
                       [[0.5] * 10, [0.5] * 10])
        b = ecrire_jeu(os.path.join(self.tmp, "B"), (6, 12, 8, 8), [1.0, 2.5],
                       [[0.4] * 10, [0.4] * 10])
        self._refus(a, b, motif="beta")


class TestExempleReel(Base):
    def test_exemple_reel(self):
        if not os.path.isdir(EXEMPLE):
            self.skipTest("entrees/exemple absent (jeu réel minuscule fourni par la tâche)")
        a = os.path.join(self.tmp, "A")
        shutil.copytree(EXEMPLE, a)
        b = ecrire_jeu(os.path.join(self.tmp, "B"), (6, 12, 8, 8), [2.0, 2.07],
                       [[0.45] * 6, [0.47] * 6])
        sortie = os.path.join(self.tmp, "res.json")
        r = self.lancer(a, b, "--blocs", "3", "--sortie", sortie)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 4)
        with open(sortie, encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual(res["n_mesures"], 6)
        self.assertEqual(res["n_blocs"], 3)
        self.assertAlmostEqual(res["a3"], 6 * 4608 * res["integrale"], places=9)


class TestEtalonnage(Base):
    def test_etalonnage_ar1(self):
        rng = random.Random(20240501)
        betas = [1.0, 1.5, 2.0]
        p_a = [0.50, 0.55, 0.60]
        p_b = [0.45, 0.48, 0.50]
        a3_vrai = 6 * 4608 * 0.0725
        couverts = 0
        positifs = 0
        n_jeux = 60
        for jeu in range(n_jeux):
            da = os.path.join(self.tmp, "jeu%02d" % jeu, "A")
            db = os.path.join(self.tmp, "jeu%02d" % jeu, "B")
            ecrire_jeu(da, (3, 12, 8, 8), betas,
                       [ar1(p, 0.5, 0.02, 200, rng) for p in p_a])
            ecrire_jeu(db, (6, 12, 8, 8), betas,
                       [ar1(p, 0.5, 0.02, 200, rng) for p in p_b])
            res = gc.calculer(da, db, 20)
            s = res["sigma_jackknife"]
            if s > 0:
                positifs += 1
            if abs(res["a3"] - a3_vrai) <= s:
                couverts += 1
        fraction = couverts / n_jeux
        self.assertGreaterEqual(fraction, 0.50)
        self.assertLessEqual(fraction, 0.85)
        self.assertGreaterEqual(positifs, 55)


if __name__ == "__main__":
    unittest.main()
