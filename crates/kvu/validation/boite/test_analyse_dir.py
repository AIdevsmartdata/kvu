#!/usr/bin/env python3
"""Tests CPU autonomes des analyses et des observables des boîtes."""

import sys

sys.dont_write_bytecode = True

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

DOSSIER = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER))
import analyse_dir as analyse
import gamma

os.environ.pop("QW1_MUTANT", None)
import box_su2


class TestAnalyseDir(unittest.TestCase):
    """Données fabriquées sans moteur de simulation ni accélérateur."""

    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory(prefix="tests_boite_")
        self.addCleanup(self.temporaire.cleanup)
        self.racine = Path(self.temporaire.name)
        self.rng = np.random.default_rng(261025)
        self.compteur = 0

    def executer(self, *arguments, rc=0, script="analyse_dir.py", environnement=None):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        if environnement:
            env.update(environnement)
        resultat = subprocess.run([sys.executable, "-I", str(DOSSIER / script), *map(str, arguments)],
                                  capture_output=True, text=True, env=env, timeout=20)
        self.assertEqual(resultat.returncode, rc, resultat.stdout + resultat.stderr)
        if script == "analyse_dir.py" and rc != 2 and "--help" not in arguments:
            for ligne in resultat.stdout.splitlines():
                self.assertRegex(ligne, r"^(PASS|FAIL|INFO|SKIP) G-DIR-\S+( .*)?$")
        return resultat

    def serie(self, moyenne, erreur, n=512):
        """Bruit recentré, calibré par Γ pour une erreur fixée sans hasard du verdict."""
        bruit = self.rng.normal(size=n)
        bruit -= bruit.mean()
        bruit *= erreur / analyse.uwerr(bruit)[1]
        return moyenne + bruit

    def ecrire_run(self, ell, cellules, beta=2.5):
        """Écrit le schéma minimal et garde les tableaux originaux pour comparaison."""
        self.compteur += 1
        dossier = self.racine / f"run_{self.compteur}"
        dossier.mkdir()
        manifeste = {"parametres": {"L": ell, "beta": beta,
                                    "boite": {"ell": ell, "chaines": []}}}
        attendu = {"ell": ell, "beta": beta, "chaines": [], "profils": {}}
        c = 0
        for bord, observables in cellules.items():
            for _ in range(2):
                description = {"chaine": c, "bord": bord, "graine": 10000 + c}
                manifeste["parametres"]["boite"]["chaines"].append(description)
                pc = self.serie(*observables["pc"][:1], observables["pc"][1] * math.sqrt(2))
                w2 = (self.serie(*observables["w2"][:1], observables["w2"][1] * math.sqrt(2))
                      if ell >= 4 else np.full(pc.size, np.nan))
                pas = np.arange(1, pc.size + 1, dtype=np.int64) * 3
                # Le profil peut être enregistré à une cadence différente des mesures.
                profil = self.rng.uniform(size=(pas[::7].size, ell // 2))
                chemin = dossier / f"c{c:03d}"
                chemin.mkdir()
                np.savetxt(chemin / "mesures.csv", np.column_stack((pas, pc, w2)),
                           delimiter=",", header="pas,pc,w2", comments="", fmt=["%d", "%.17g", "%.17g"])
                np.savetxt(chemin / "profil.csv", np.column_stack((pas[::7], profil)),
                           delimiter=",", header="pas," + ",".join(f"d{d}" for d in range(ell // 2)),
                           comments="", fmt=["%d"] + ["%.17g"] * (ell // 2))
                attendu["chaines"].append(dict(description, pas=pas, pc=pc, w2=w2))
                attendu["profils"][c] = profil
                c += 1
        (dossier / "manifest_regen.json").write_text(json.dumps(manifeste), encoding="utf-8")
        return dossier, attendu

    def test_profil_jamais_mesure_est_admis(self):
        """--profil-tous 0 : profil.csv ne contient que son en-tête ; les séries pc et w2 restent lisibles (ajout WP-DIR)."""
        dossier, attendu = self.ecrire_run(4, {"cold": {"pc": (0.67, 1e-3), "w2": (0.32, 1e-3)}})
        for chemin in dossier.glob("c*/profil.csv"):
            chemin.write_text("pas,d0,d1\n", encoding="utf-8")
        run = analyse.lire_run(dossier)
        self.assertEqual(len(run["chaines"]), 2)
        for c in run["chaines"]:
            self.assertEqual(run["profils"][c["chaine"]].shape, (0, 2))
        self.assertEqual(run["chaines"][0]["pc"].tolist(), attendu["chaines"][0]["pc"].tolist())
        # une série de mesures vide, elle, reste refusée
        (dossier / "c000" / "mesures.csv").write_text("pas,pc,w2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            analyse.lire_run(dossier)

    def run_pc(self, ell, moyennes, erreur=1e-4, beta=2.5):
        return self.ecrire_run(ell, {bord: {"pc": (m, erreur), "w2": (0.2, erreur)}
                                   for bord, m in moyennes.items()}, beta)[0]

    def runs_qw1(self, mutant=False, decalage8=0.0, decalage4=0.0):
        arguments = []
        for ell in (2, 4, 6, 8):
            ref = analyse.reference_qw1()["cellules"][str(ell)]
            cellules = {}
            for bord in ("cold", "z2full"):
                cellules[bord] = {}
                for obs in (["pc", "w2"] if ell >= 4 else ["pc"]):
                    m, s = ref[obs][bord]
                    # Moyennes à +/- une erreur de référence, donc verdict certain.
                    m += s if bord == "cold" else -s
                    if mutant and ell in {4, 8} and obs == "pc":
                        m = ref[obs]["cold"][0]
                    if bord == "cold" and obs == "pc":
                        if ell == 8:
                            m += decalage8
                        if ell == 4:
                            m += decalage4
                    cellules[bord][obs] = (m, s)
            dossier, _ = self.ecrire_run(ell, cellules)
            arguments.extend(["--run", f"{ell}={dossier}"])
        return arguments

    def json_obs(self, nom, pc=0.4, w2=0.2, profil=None):
        chemin = self.racine / nom
        chemin.write_text(json.dumps({"pc": pc, "w2": w2,
                                      "profil": [0.1, 0.3] if profil is None else profil}), encoding="utf-8")
        return chemin

    def configuration(self, ell=4, froide=False, dtype=np.float64):
        forme = (4,) + (ell + 2,) * 4 + (4,)
        a = np.zeros(forme, dtype=dtype) if froide else self.rng.normal(size=forme).astype(dtype)
        if froide:
            a[..., 0] = 1
        else:
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
        chemin = self.racine / f"cfg_{ell}_{froide}_{np.dtype(dtype).name}.npy"
        np.save(chemin, a)
        return chemin, a

    def test_uwerr_reference_intacte(self):
        self.assertIs(analyse.uwerr, gamma.uwerr)
        self.assertEqual(Path(gamma.__file__).resolve(), analyse.REFERENCE / "gamma.py")
        self.assertEqual(hashlib.sha256(Path(gamma.__file__).read_bytes()).hexdigest(),
                         "439ed2f3d4dd93e542eee21a3cef2243036dfd3625ebf2759d2834e92bb78cae")

    def test_cellule_ar1(self):
        n, m, phi, vraie = 20000, 8, 0.5, 0.63
        chaines = []
        for c in range(m):
            bruit = self.rng.normal(size=n)
            serie = np.empty(n)
            serie[0] = bruit[0] / math.sqrt(1 - phi ** 2)
            for t in range(1, n):
                serie[t] = phi * serie[t - 1] + bruit[t]
            chaines.append({"bord": "cold", "pc": vraie + serie})
        moyenne, erreur, details = analyse.cellule({"chaines": chaines}, "cold", "pc")
        theorie = math.sqrt(1 / ((1 - phi) ** 2 * n * m))
        self.assertLess(abs(moyenne - vraie), 4 * erreur)
        self.assertGreaterEqual(erreur / theorie, 0.7)
        self.assertLessEqual(erreur / theorie, 1.4)
        self.assertEqual(len(details), m)
        self.assertEqual(moyenne, math.fsum(d[0] for d in details) / m)
        self.assertEqual(erreur, math.sqrt(math.fsum(d[1] ** 2 for d in details)) / m)

    def test_z_et_delta(self):
        self.assertEqual(analyse.z(5, 3, 0, 4), 1)
        self.assertEqual(analyse.z(0, 0, 1, 0), -math.inf)
        self.assertEqual(analyse.z(1, 0, 0, 0), math.inf)
        self.assertEqual(analyse.z(1, 0, 1, 0), 0)
        # Chaînes constantes : erreurs nulles et moyennes calculables à la main.
        run = {"chaines": [{"bord": "a", "pc": np.full(16, m)} for m in (2., 4.)]
               + [{"bord": "b", "pc": np.full(16, 1.)}]}
        self.assertEqual(analyse.cellule(run, "a", "pc"),
                         (3., 0., [(2., 0., 0.5), (4., 0., 0.5)]))
        self.assertEqual(analyse.delta(run, "a", "b", "pc"), (2., 0.))
        # Erreurs non nulles, et chaînes de longueurs différentes, sans pondération par N.
        run = {"chaines": [{"bord": "a", "pc": self.serie(2, .03, 512)},
                             {"bord": "a", "pc": self.serie(4, .04, 1024)},
                             {"bord": "b", "pc": self.serie(1, .12, 512)}]}
        d, s = analyse.delta(run, "a", "b", "pc")
        self.assertAlmostEqual(d, 2)
        self.assertAlmostEqual(s, math.hypot(.025, .12))
        with self.assertRaisesRegex(ValueError, "aucune chaîne"):
            analyse.cellule(run, "absent", "pc")

    def test_lire_run_exactement(self):
        for ell in (2, 4, 6, 8):
            with self.subTest(ell=ell):
                chemin, attendu = self.ecrire_run(ell, {"cold": {"pc": (.6, .001), "w2": (.2, .002)},
                                                        "z2full": {"pc": (.5, .001), "w2": (.1, .002)}})
                run = analyse.lire_run(chemin)
                self.assertEqual(run["ell"], ell)
                self.assertEqual(run["beta"], 2.5)
                for a, b in zip(run["chaines"], attendu["chaines"]):
                    for cle in ("chaine", "bord", "graine"):
                        self.assertEqual(a[cle], b[cle])
                    for cle in ("pas", "pc", "w2"):
                        np.testing.assert_array_equal(a[cle], b[cle], strict=True)
                    np.testing.assert_array_equal(run["profils"][a["chaine"]],
                                                  attendu["profils"][a["chaine"]], strict=True)

    def test_qw1_vert_et_info(self):
        resultat = self.executer("qw1", *self.runs_qw1())
        self.assertIn("PASS G-DIR-QW1", resultat.stdout)
        self.assertEqual(resultat.stdout.count("reference=independante"), 3)
        self.assertEqual(resultat.stdout.count(" bord="), 14)
        self.assertEqual(resultat.stdout.count(" delta="), 7)

    def test_qw1_rouge_decalage_dpc8(self):
        sr = analyse.reference_qw1()["cellules"]["8"]["dpc"][1]
        resultat = self.executer("qw1", *self.runs_qw1(decalage8=10 * math.sqrt(2) * sr), rc=1)
        self.assertIn("FAIL G-DIR-QW1 ell=8,dpc", resultat.stdout)

    def test_qw1_autres_deltas_obligatoires(self):
        resultat = self.executer("qw1", *self.runs_qw1(decalage4=.02), rc=1)
        self.assertIn("FAIL G-DIR-QW1 ell=4,dpc", resultat.stdout)

    def test_qw1_ell8_obligatoire(self):
        arguments = self.runs_qw1()[:2]
        resultat = self.executer("qw1", *arguments, rc=1)
        self.assertIn("ell=8,dpc et ell=8,dw2 manquants", resultat.stdout)

    def test_qw1_mutant_detecte(self):
        resultat = self.executer("qw1", *self.runs_qw1(mutant=True), "--mutant")
        self.assertIn("PASS G-DIR-QW1-MUTANT", resultat.stdout)

    def test_qw1_mutant_non_detecte(self):
        resultat = self.executer("qw1", *self.runs_qw1(), "--mutant", rc=1)
        self.assertIn("FAIL G-DIR-QW1-MUTANT ECHEC_OUTIL", resultat.stdout)

    def test_qw1_infos_non_bloquantes(self):
        ref = analyse.reference_qw1()["cellules"]["8"]
        cellules = {bord: {obs: (ref[obs][bord][0] + .2, ref[obs][bord][1])
                           for obs in ("pc", "w2")} for bord in ("cold", "z2full")}
        dossier, _ = self.ecrire_run(8, cellules)
        self.assertIn("PASS G-DIR-QW1", self.executer("qw1", "--run", f"8={dossier}").stdout)

    def test_jauge_nominale_et_mutante(self):
        erreur = 1e-4
        for ecart, rc, rcm in ((0, 0, 1), (6 * math.sqrt(2) * erreur, 1, 0)):
            with self.subTest(ecart=ecart):
                dossier = self.run_pc(2, {"cold": .6, "gauge": .6 + ecart, "haar": .6 + ecart}, erreur)
                arg = f"2={dossier}"
                self.executer("jauge", "--run", arg, rc=rc)
                resultat = self.executer("jauge", "--run", arg, "--mutant", rc=rcm)
                self.assertIn("G-DIR-JAUGE-MUTANT", resultat.stdout)
                if rcm:
                    self.assertIn("ECHEC_OUTIL", resultat.stdout)

    def test_jauge_mutant_exige_ell2_ou4(self):
        dossier = self.run_pc(6, {"cold": .6, "haar": .61})
        self.executer("jauge", "--run", f"6={dossier}", "--mutant", rc=1)

    def test_agit_et_mutant_inverse(self):
        for valeur_z, rc, rcm in ((6, 0, 1), (1, 1, 0)):
            with self.subTest(z=valeur_z):
                dossier = self.run_pc(2, {"cold": .6, "z2full": .6 - valeur_z * math.sqrt(2) * 1e-4})
                self.executer("agit", "--run", f"2={dossier}", rc=rc)
                self.executer("agit", "--run", f"2={dossier}", "--mutant", rc=rcm)

    def test_seuils_aux_frontieres(self):
        # Valeurs exactes imposées à l'estimateur : vérifie <, <=, > et >= sans arrondi des CSV.
        reference = analyse.reference_qw1()
        ref8 = reference["cellules"]["8"]
        args = type("Arguments", (), {"run": ["8=inutilise"], "mutant": False})()
        run = {"chaines": []}
        with contextlib.redirect_stdout(io.StringIO()), patch.object(analyse, "_runs", return_value={8: run}):
            with patch.object(analyse, "z", return_value=3.), patch.object(analyse, "delta", return_value=(0., 0.)):
                self.assertEqual(analyse._qw1(args), 1)
                args.mutant = True
                self.assertEqual(analyse._qw1(args), 0)
            args.mutant = False
            with patch.object(analyse, "cellule", return_value=(0., 1., [])), patch.object(analyse, "z", return_value=3.):
                self.assertEqual(analyse._jauge(args), 0)
            args.run = "8=inutilise"
            with patch.object(analyse, "cellule", return_value=(0., 1., [])), patch.object(analyse, "z", return_value=5.):
                self.assertEqual(analyse._agit(args), 1)
                args.mutant = True
                self.assertEqual(analyse._agit(args), 0)
        # Les deux deltas de ell=8 doivent réellement être testés.
        self.assertIn("dw2", ref8)

    def test_bessel_par_fraction_continue(self):
        beta = .2
        # I_(nu+1)/I_nu = x / (2(nu+1) + x I_(nu+2)/I_(nu+1)).
        rapport = 0.
        for nu in range(120, 0, -1):
            rapport = beta / (2 * (nu + 1) + beta * rapport)
        u = analyse.bessel_i(2, beta) / analyse.bessel_i(1, beta)
        self.assertAlmostEqual(u, rapport, places=16)
        self.assertGreater(u, .0498)
        self.assertLess(u, .05)
        self.assertAlmostEqual(analyse.attendu_sc(beta), rapport + 4 * rapport ** 5, places=16)
        self.assertEqual(analyse.attendu_sc(0), 0.)

    def test_sc_nominal_et_mutant(self):
        attendu = analyse.attendu_sc(.2)
        for facteur, rc in ((1., 0), (1.02, 1)):
            dossier = self.run_pc(6, {b: facteur * attendu for b in ("cold", "z2full", "flux", "haar")}, beta=.2)
            resultat = self.executer("sc", "--run", dossier, rc=rc)
            self.assertEqual(resultat.stdout.count("INFO G-DIR-SC bord="), 4)
        dossier = self.run_pc(6, {"cold": 1.05 * attendu}, beta=.21)
        self.executer("sc", "--run", dossier, "--mutant", "--beta-nominal", .2)
        self.executer("sc", "--run", dossier, "--mutant", rc=2)
        dossier = self.run_pc(6, {"cold": attendu}, beta=.21)
        resultat = self.executer("sc", "--run", dossier, "--mutant", "--beta-nominal", .2, rc=1)
        self.assertIn("ECHEC_OUTIL", resultat.stdout)

    def test_obs_cmp_exact_et_tolerances(self):
        a = self.json_obs("a.json")
        b = self.json_obs("b.json")
        self.executer("obs-cmp", a, b)
        self.executer("obs-cmp", a, b, "--identique")
        for cle in ("pc", "w2", "profil"):
            with self.subTest(cle=cle):
                objet = json.loads(a.read_text(encoding="utf-8"))
                if cle == "profil":
                    objet[cle][1] += 1e-9
                else:
                    objet[cle] += 1e-9
                b.write_text(json.dumps(objet), encoding="utf-8")
                self.executer("obs-cmp", a, b, "--tol", "1e-12", rc=1)
                self.executer("obs-cmp", a, b, "--tol", "1e-8")
                self.executer("obs-cmp", a, b, "--tol", "1e-8", "--identique", rc=1)

    def test_obs_cmp_null_et_longueurs(self):
        a = self.json_obs("a.json", w2=None)
        b = self.json_obs("b.json", w2=None)
        self.executer("obs-cmp", a, b, "--identique")
        b = self.json_obs("b.json", w2=.2)
        self.executer("obs-cmp", a, b, rc=1)
        b = self.json_obs("b.json", w2=None, profil=[.1])
        self.executer("obs-cmp", a, b, rc=1)

    def test_obs_cmp_bits_zero_signe(self):
        a = self.json_obs("a.json", pc=0.)
        b = self.json_obs("b.json", pc=-0.)
        self.executer("obs-cmp", a, b)
        self.executer("obs-cmp", a, b, "--identique", rc=1)
        b = self.json_obs("b.json", pc=np.nextafter(0.4, 1.))
        a = self.json_obs("a.json", pc=.4)
        self.executer("obs-cmp", a, b, "--identique", rc=1)

    def test_stats_json_et_w2_indefini(self):
        dossier = self.run_pc(4, {"cold": .6, "z2full": .5})
        run = analyse.lire_run(dossier)
        for obs in ("pc", "w2"):
            resultat = self.executer("stats", dossier, "--obs", obs)
            for ligne in resultat.stdout.splitlines():
                objet = json.loads(ligne.split(" ", 2)[2])
                m, s, details = analyse.cellule(run, objet["bord"], obs)
                self.assertEqual(objet["M"], 2)
                self.assertEqual(objet["moyenne"], m)
                self.assertEqual(objet["erreur"], s)
                self.assertEqual(objet["tau"], sum(d[2] for d in details) / 2)
        dossier = self.run_pc(2, {"cold": .8})
        self.assertIn("SKIP G-DIR-STATS", self.executer("stats", dossier, "--obs", "w2").stdout)

    def test_erreurs_usage_et_entrees(self):
        self.executer("qw1", rc=2)
        self.executer("agit", "--run", "invalide", rc=2)
        dossier = self.run_pc(2, {"cold": .8})
        self.executer("jauge", "--run", f"4={dossier}", rc=2)
        self.executer("jauge", "--run", f"2={dossier}", rc=2)
        mesures = dossier / "c000" / "mesures.csv"
        mesures.write_text("pas,pc,w2\n1,nan,nan\n", encoding="utf-8")
        self.executer("stats", dossier, rc=2)

    def test_ref_obs_froide_et_ell2(self):
        for ell in (2, 4):
            with self.subTest(ell=ell):
                chemin, _ = self.configuration(ell=ell, froide=True)
                resultat = self.executer("--cfg", chemin, "--ell", ell, script="ref_obs.py")
                self.assertEqual(len(resultat.stdout.splitlines()), 1)
                self.assertEqual(json.loads(resultat.stdout), {"ell": ell, "backend": "python-qw1", "pc": 1.,
                                 "w2": 1. if ell >= 4 else None, "profil": [1.] * (ell // 2)})

    def test_ref_obs_aleatoire_bits_et_mutant(self):
        chemin, a = self.configuration()
        n = 5
        u = np.transpose(a, (0, 5, 1, 2, 3, 4))[:, :, :n, :n, :n, :n]
        boite = box_su2.Box(4, 2.5)
        boite.U = u
        p = boite.plaq_all()
        pc, w2 = box_su2.measure(boite, p)
        d = box_su2.depth_classes(4)
        profil = [float(p[d == couche].mean()) for couche in range(2)]
        resultat = self.executer("--cfg", chemin, "--ell", 4, script="ref_obs.py",
                                 environnement={"QW1_MUTANT": "M2"})
        objet = json.loads(resultat.stdout)
        for obtenu, attendu in zip([objet["pc"], objet["w2"], *objet["profil"]], [pc, w2, *profil]):
            self.assertEqual(struct.pack("!d", obtenu), struct.pack("!d", attendu))
        mutant = json.loads(self.executer("--cfg", chemin, "--ell", 4, "--centre-decale",
                                          script="ref_obs.py").stdout)
        self.assertGreater(max(abs(mutant[obs] - objet[obs]) for obs in ("pc", "w2")), 1e-3)
        self.assertEqual(mutant["profil"], objet["profil"])
        boite.U = np.roll(u, -1, axis=(2, 3, 4, 5))
        mpc, mw2 = box_su2.measure(boite)
        self.assertEqual(mutant["pc"], mpc)
        self.assertEqual(mutant["w2"], mw2)

    def test_ref_obs_valide_entrees(self):
        chemin, _ = self.configuration(froide=True)
        for ell in (0, 3, -2, 6):
            self.executer("--cfg", chemin, "--ell", ell, script="ref_obs.py", rc=2)
        chemin, _ = self.configuration(dtype=np.float32)
        self.executer("--cfg", chemin, "--ell", 4, script="ref_obs.py", rc=2)

    def test_ref_obs_retire_mutant_avant_import(self):
        # M2 change les masques à l'import : on vérifie aussi l'état du module et de l'environnement.
        code = ("import sys, runpy, os; sys.dont_write_bytecode=True; "
                f"runpy.run_path({str(DOSSIER / 'ref_obs.py')!r}, run_name='reference_test'); "
                "assert 'QW1_MUTANT' not in os.environ; "
                "assert sys.modules['box_su2'].MUT == ''")
        resultat = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                                  env=dict(os.environ, QW1_MUTANT="M2", OPENBLAS_NUM_THREADS="1"), timeout=20)
        self.assertEqual(resultat.returncode, 0, resultat.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
