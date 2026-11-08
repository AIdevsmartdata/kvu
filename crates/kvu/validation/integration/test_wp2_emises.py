"""Tests GATES v1 / WP2-EMISES, exécutables directement sous python3 -I."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHEMIN_MODULE = Path(__file__).resolve().with_name("wp2_emises.py")
SPEC = importlib.util.spec_from_file_location("wp2_emises", CHEMIN_MODULE)
wp2_emises = importlib.util.module_from_spec(SPEC)
# Ne créer aucun __pycache__ lors du chargement explicite du module voisin.
_ancien_dont_write_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(wp2_emises)
finally:
    sys.dont_write_bytecode = _ancien_dont_write_bytecode


def gate(nom, machine="gamer", statut="PASS"):
    return {"nom": nom, "machine": machine, "statut": statut, "detail": ""}


class ComparaisonTests(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporaire.cleanup)
        self.repertoire = Path(self.temporaire.name)
        self.reference = self.repertoire / "reference.json"
        self.emises = self.repertoire / "emises.json"
        self.extras = [gate(nom) for nom in wp2_emises.EXTRAS_PAR_DEFAUT]

    def ecrire(self, chemin, gates):
        document = {
            "gates": gates,
            "fail": sum(g["statut"] == "FAIL" for g in gates),
            "skip": sum(g["statut"] == "SKIP" for g in gates),
        }
        chemin.write_text(json.dumps(document), encoding="utf-8")

    def appeler(self, options=(), sortie_json=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = [str(self.reference), str(self.emises), *options]
        if sortie_json:
            arguments.append("--json")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = wp2_emises.main(arguments)
        return rc, stdout.getvalue(), stderr.getvalue()

    def comparer_fichiers(self, reference, emises, options=()):
        self.ecrire(self.reference, reference)
        self.ecrire(self.emises, emises)
        rc, stdout, stderr = self.appeler(options)
        self.assertEqual(stderr, "")
        self.assertEqual(len(stdout.splitlines()), 1)
        return rc, json.loads(stdout)

    def test_meme_ensemble_et_extras_par_defaut(self):
        rc, resultat = self.comparer_fichiers(
            [gate("OBS-1")], [gate("OBS-1"), *self.extras])
        self.assertEqual(rc, 0)
        self.assertEqual(resultat, {
            "manquants": [], "en_trop": [], "skips_non_couverts": [], "ok": True})

    def test_identifiant_manquant(self):
        rc, resultat = self.comparer_fichiers([gate("OBS-1")], self.extras)
        self.assertEqual(rc, 8)
        self.assertEqual(resultat["manquants"], ["OBS-1"])
        self.assertFalse(resultat["ok"])

    def test_identifiant_en_trop_informatif(self):
        rc, resultat = self.comparer_fichiers([], [*self.extras, gate("SURPLUS")])
        self.assertEqual(rc, 0)
        self.assertEqual(resultat["en_trop"], ["SURPLUS"])

    def test_equivalence_renommage(self):
        for options, rc_attendu, manquants, en_trop in (
            (["--equiv", "OBS-4=OBS-4-L8"], 0, [], []),
            ([], 8, ["OBS-4"], ["OBS-4-L8"]),
        ):
            with self.subTest(options=options):
                rc, resultat = self.comparer_fichiers(
                    [gate("OBS-4")], [gate("OBS-4-L8"), *self.extras], options)
                self.assertEqual(rc, rc_attendu)
                self.assertEqual(resultat["manquants"], manquants)
                self.assertEqual(resultat["en_trop"], en_trop)

    def test_skip_couvert_uniquement_par_une_autre_machine(self):
        cas = (
            ([], 8),
            ([gate("OBS-1", "gamer")], 8),
            ([gate("AUTRE", "acemagic")], 8),
            ([gate("OBS-1", "acemagic", "INFO")], 8),
            ([gate("OBS-1", "acemagic")], 0),
        )
        for lignes, rc_attendu in cas:
            with self.subTest(lignes=lignes):
                rc, resultat = self.comparer_fichiers(
                    [gate("OBS-1")],
                    [gate("OBS-1", "gamer", "SKIP"), *lignes, *self.extras])
                self.assertEqual(rc, rc_attendu)
                self.assertEqual(resultat["skips_non_couverts"],
                                 ["OBS-1"] if rc_attendu == 8 else [])

    def test_chaque_machine_skip_doit_etre_couverte(self):
        rc, resultat = self.comparer_fichiers([], [
            *self.extras, gate("OBS-1", "gamer", "SKIP"),
            gate("OBS-1", "acemagic", "SKIP"), gate("OBS-1", "acemagic"),
            gate("OBS-1", "acemagic", "SKIP"),
        ])
        self.assertEqual(rc, 8)
        self.assertEqual(resultat["skips_non_couverts"], ["OBS-1"])

    def test_fichier_absent_ou_json_invalide(self):
        for chemin in (self.reference, self.emises):
            for contenu in (None, "{invalide", "{}", "[]", '{"gates": null}'):
                with self.subTest(chemin=chemin.name, contenu=contenu):
                    self.ecrire(self.reference, [])
                    self.ecrire(self.emises, self.extras)
                    if contenu is None:
                        chemin.unlink()
                    else:
                        chemin.write_text(contenu, encoding="utf-8")
                    rc, stdout, stderr = self.appeler()
                    self.assertEqual(rc, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn(str(chemin), stderr)

    def test_equivalences_mal_formees(self):
        for texte in ("OBS-4", "=NOUVEAU", "ANCIEN=", "=", ""):
            with self.subTest(texte=texte):
                rc, stdout, stderr = self.appeler(["--equiv", texte])
                self.assertEqual(rc, 2)
                self.assertEqual(stdout, "")
                self.assertIn("ANCIEN=NOUVEAU", stderr)

    def test_usage_invalide(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            rc = wp2_emises.main([])
        self.assertEqual(rc, 2)
        self.assertTrue(stderr.getvalue())

    def test_extra_remplace_les_defauts_et_ignore_le_vide(self):
        for options, emises in (
            (["--extra", ""], []),
            (["--extra", "AJOUT", "--extra", ""], [gate("AJOUT")]),
            (["--extra", "B", "--extra", "A"], [gate("A"), gate("B")]),
        ):
            with self.subTest(options=options):
                rc, resultat = self.comparer_fichiers([], emises, options)
                self.assertEqual(rc, 0)
                self.assertEqual(resultat["manquants"], [])
                self.assertEqual(resultat["en_trop"], [])

    def test_extras_par_defaut_obligatoires(self):
        rc, resultat = self.comparer_fichiers([], [])
        self.assertEqual(rc, 8)
        self.assertEqual(resultat["manquants"], ["FILE-1", "MACH-1", "VERROU-1"])

    def test_tri_deduplication_et_fonction_pure(self):
        reference = [gate("Z"), gate("A"), gate("Z")]
        emises = [gate("Y", statut="SKIP"), gate("B", statut="SKIP"),
                  gate("Y", statut="SKIP")]
        extra, equiv = ["C", "", "C"], {"Z": "RENOMME"}
        entrees = (reference, emises, extra, equiv)
        copie = copy.deepcopy(entrees)
        resultat = wp2_emises.comparer(*entrees)
        self.assertEqual(resultat, {
            "manquants": ["A", "C", "Z"], "en_trop": ["B", "Y"],
            "skips_non_couverts": ["B", "Y"], "ok": False})
        self.assertEqual(entrees, copie)

    def test_equivalences_repetables(self):
        rc, resultat = self.comparer_fichiers(
            [gate("A"), gate("B")], [gate("A2"), gate("B2")],
            ["--extra", "", "--equiv", "A=A1", "--equiv", "A=A2",
             "--equiv", "B=B2"])
        self.assertEqual(rc, 0)
        self.assertEqual(resultat["en_trop"], [])

    def test_equivalence_ne_couvre_pas_un_skip_d_un_autre_nom(self):
        rc, resultat = self.comparer_fichiers(
            [gate("A")], [gate("A", "gamer", "SKIP"), gate("A2", "acemagic")],
            ["--extra", "", "--equiv", "A=A2"])
        self.assertEqual(rc, 8)
        self.assertEqual(resultat["skips_non_couverts"], ["A"])

    def test_sortie_lisible(self):
        self.ecrire(self.reference, [gate("ABSENT")])
        self.ecrire(self.emises, [gate("SUPPLEMENT", statut="SKIP")])
        rc, stdout, stderr = self.appeler(["--extra", ""], sortie_json=False)
        self.assertEqual(rc, 8)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "manquants (1) :\n  ABSENT\n"
                         "en trop (1) :\n  SUPPLEMENT\n"
                         "skips non couverts (1) :\n  SUPPLEMENT\n")

    def test_commande_isolee_depuis_un_autre_repertoire(self):
        self.ecrire(self.reference, [])
        self.ecrire(self.emises, self.extras)
        execution = subprocess.run(
            [sys.executable, "-I", str(CHEMIN_MODULE), str(self.reference),
             str(self.emises), "--json"], cwd=self.repertoire,
            capture_output=True, text=True, check=False)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        self.assertEqual(execution.stderr, "")
        self.assertTrue(json.loads(execution.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
