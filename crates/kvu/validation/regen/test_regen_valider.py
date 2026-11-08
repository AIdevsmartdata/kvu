# SPDX-License-Identifier: Apache-2.0
"""Vérifier le validateur et son interface avec des manifestes synthétiques."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RACINE = Path(__file__).resolve().parents[4]
SCRIPT = RACINE / "scripts/regen_valider.py"
SPECIFICATION = importlib.util.spec_from_file_location("regen_valider", SCRIPT)
VALIDATEUR = importlib.util.module_from_spec(SPECIFICATION)
# Empêcher l'import par chemin de créer un fichier de cache dans le dépôt.
_ANCIEN_REGLAGE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPECIFICATION.loader.exec_module(VALIDATEUR)
finally:
    sys.dont_write_bytecode = _ANCIEN_REGLAGE


def manifeste_valide():
    """Construire en mémoire une chaîne SU(2) sur un accélérateur HIP."""
    hachage = "a" * 64
    return {
        "schema": "kvu-regen/1",
        "origine": {
            "schema": "su2_hip/1", "manifeste": "origine.json",
            "sha256_manifeste": hachage,
        },
        "binaire_sha256": hachage,
        "git_head": "a" * 40,
        "source_hash": hachage,
        "noyaux_embarques": [{"nom": "su2", "sha256": hachage}],
        "modules_compiles": [{
            "source_sha256": hachage, "options": "-O3",
            "arguments": ["--offload-arch=gfx1151"], "objet_sha256": hachage,
        }],
        "backend": "hip",
        "arch": "gfx1151",
        "versions": {"hip": "6.4"},
        "machine": {
            "nom": "calcul-su2", "cpu": "AMD Ryzen", "gpu": "gfx1151", "ram": 64,
        },
        "graines": {"cle": 42, "etiquettes": ["chaine-su2"]},
        "parametres": {
            "groupe": "SU(2)", "L": 12, "beta": 2.5, "start": "hot",
            "therm": 100, "nmeas": 50, "skip": 10, "nor": 4,
            "precision": "f64", "options": "ref-k1",
        },
        "commande": ["kvu", "su2", "--L", "12", "--beta", "2.5"],
        "entrees": [{"nom": "initial.bin", "sha256": hachage}],
        "fichiers": [{"nom": "mesures.csv", "sha256": hachage, "octets": 4096}],
        "duree_s": 12.5,
        "statut": "COMPLETE",
        "classe_regen": "A",
        "regen_verifiee": {"date": None, "machine": None, "resultat": "non_verifiee"},
    }


class TestsValidateur(unittest.TestCase):
    """Exercer les règles du schéma réel et les limites du sous-ensemble."""

    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((RACINE / "schemas/kvu-regen-1.schema.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.manifeste = manifeste_valide()

    def test_manifeste_valide(self):
        self.assertEqual(VALIDATEUR.valider(self.schema, self.manifeste), [])

    def test_mutants_champs_retires(self):
        original = copy.deepcopy(self.manifeste)
        schema_original = copy.deepcopy(self.schema)
        mutants = VALIDATEUR.mutants_champ_retire(self.schema, self.manifeste)
        self.assertGreaterEqual(len(mutants), 20)
        noms = [nom for nom, _ in mutants]
        self.assertEqual(len(noms), len(set(noms)))
        self.assertIn("/machine/gpu", noms)
        self.assertIn("/fichiers/0/octets", noms)
        self.assertIn("/modules_compiles/0/objet_sha256", noms)
        for nom, mutant in mutants:
            with self.subTest(mutant=nom):
                self.assertTrue(VALIDATEUR.valider(self.schema, mutant))
        self.assertEqual(self.manifeste, original)
        self.assertEqual(self.schema, schema_original)
        mutants[0][1]["machine"]["nom"] = "copie modifiée"
        self.assertEqual(self.manifeste, original)
        self.assertEqual(mutants[1][1]["machine"]["nom"], original["machine"]["nom"])

    def test_valeurs_invalides(self):
        cas = [
            (("binaire_sha256",), "a" * 63),
            (("binaire_sha256",), "A" + "a" * 63),
            (("classe_regen",), "D"),
            (("parametres", "L"), True),
            (("parametres", "L"), 1),
        ]
        for chemin, valeur in cas:
            with self.subTest(chemin=chemin, valeur=valeur):
                instance = copy.deepcopy(self.manifeste)
                parent = instance
                for nom in chemin[:-1]:
                    parent = parent[nom]
                parent[chemin[-1]] = valeur
                self.assertTrue(VALIDATEUR.valider(self.schema, instance))
        del self.manifeste["statut"]
        self.assertIn("/statut : champ obligatoire absent", VALIDATEUR.valider(self.schema, self.manifeste))

    def test_condition_classes(self):
        self.manifeste["binaire_sha256"] = None
        for classe in ("A", "B", "C"):
            with self.subTest(classe=classe):
                self.manifeste["classe_regen"] = classe
                self.assertEqual(bool(VALIDATEUR.valider(self.schema, self.manifeste)), classe != "C")

    def test_mot_cle_inconnu(self):
        for schema in (
            {"type": "object", "inconnu": True},
            {"properties": {"absent": {"inconnu": True}}},
            {"if": {"const": False}, "then": {"inconnu": True}},
        ):
            with self.subTest(schema=schema):
                with self.assertRaises(VALIDATEUR.ErreurUsage):
                    VALIDATEUR.valider(schema, {})

    def test_sous_ensemble(self):
        cas = [
            ({"type": "integer"}, True, False),
            ({"type": "integer"}, 12, True),
            ({"type": "integer"}, 12.0, False),
            ({"type": "number"}, False, False),
            ({"type": "number"}, 2, True),
            ({"type": "number"}, 2.5, True),
            ({"type": ["string", "null"], "format": "sha256"}, None, True),
            ({"required": ["x"], "additionalProperties": False}, [], True),
            ({"items": {"type": "string"}, "minItems": 2}, {}, True),
            ({"minimum": 2, "minLength": 3, "format": "sha256"}, None, True),
            ({"minimum": 2}, True, True),
            ({"minItems": 2}, [], False),
            ({"minLength": 2}, "é", False),
            ({"format": "sha256"}, "a" * 64 + "\n", False),
            ({"additionalProperties": False}, {"x": 1}, False),
            ({"properties": {"x": {}}, "additionalProperties": False}, {"x": 1}, True),
            ({"enum": [1]}, True, False),
            ({"const": {"x": [1]}}, {"x": [True]}, False),
            ({"const": 1}, 1.0, True),
            ({"then": {"type": "string"}}, 1, True),
            ({"if": {"const": 1}, "then": {"type": "string"}}, 2, True),
        ]
        for schema, instance, accepte in cas:
            with self.subTest(schema=schema, instance=instance):
                self.assertEqual(VALIDATEUR.valider(schema, instance) == [], accepte)
        with self.assertRaises(VALIDATEUR.ErreurUsage):
            VALIDATEUR.valider({"additionalProperties": {}}, {})

    def test_parcours_mutants(self):
        schema = {
            "properties": {"a/b~": {"items": {"required": ["x"]}}},
            "if": {"required": ["ignore"]},
            "then": {"required": ["ignore"]},
        }
        instance = {"a/b~": [{"x": 1}, {"x": 2}], "ignore": 0}
        mutants = VALIDATEUR.mutants_champ_retire(schema, instance)
        self.assertEqual(mutants, [("/a~1b~0/0/x", {"a/b~": [{}, {"x": 2}], "ignore": 0})])
        self.assertEqual(VALIDATEUR.mutants_champ_retire(schema, {"a/b~": []}), [])

    def test_interface_commande(self):
        with tempfile.TemporaryDirectory() as repertoire:
            dossier = Path(repertoire)

            def ecrire(nom, valeur):
                chemin = dossier / nom
                chemin.write_text(json.dumps(valeur), encoding="utf-8")
                return str(chemin)

            def lancer(*arguments):
                return subprocess.run(
                    ["python3", "-I", str(SCRIPT), *map(str, arguments)],
                    cwd=dossier, capture_output=True, text=True, check=False,
                )

            valide = ecrire("valide.json", self.manifeste)
            invalide = ecrire("invalide.json", {})
            schema = ecrire("schema.json", self.schema)
            resultat = lancer(valide)
            self.assertEqual(resultat.returncode, 0, resultat.stderr)
            self.assertEqual(resultat.stdout, f"PASS {valide}\n")
            resultat = lancer("--schema", schema, valide, invalide)
            self.assertEqual(resultat.returncode, 8, resultat.stderr)
            self.assertEqual(resultat.stdout.splitlines()[0], f"PASS {valide}")
            erreurs = resultat.stdout.splitlines()[1:]
            self.assertEqual(len(erreurs), 20)
            self.assertTrue(all(ligne.startswith(f"FAIL {invalide} : /") for ligne in erreurs))
            resultat = lancer("--mutants", valide)
            self.assertEqual(resultat.returncode, 0, resultat.stderr)
            nombre = len(VALIDATEUR.mutants_champ_retire(self.schema, self.manifeste))
            lignes = resultat.stdout.splitlines()
            self.assertEqual(len(lignes), nombre + 1)
            self.assertTrue(all(ligne.startswith("PASS mutant /") for ligne in lignes[:-1]))
            self.assertEqual(lignes[-1], f"PASS REGEN-SCHEMA-MUTANTS {nombre}/{nombre}")
            resultat = lancer("--mutants", invalide)
            self.assertEqual(resultat.returncode, 8)
            self.assertTrue(resultat.stdout.endswith("FAIL REGEN-SCHEMA-MUTANTS 0/0\n"))
            malforme = dossier / "malforme.json"
            malforme.write_text("{", encoding="utf-8")
            mauvais_schema = ecrire("schema_inconnu.json", {"inconnu": True})
            for arguments in (
                (), ("--inconnu",), ("--schema",),
                (dossier / "absent.json",), (malforme,),
                ("--schema", mauvais_schema, valide),
                ("--schema", malforme, valide),
                ("--mutants", valide, invalide),
            ):
                with self.subTest(arguments=arguments):
                    resultat = lancer(*arguments)
                    self.assertEqual(resultat.returncode, 2, resultat.stderr)
                    self.assertIn("Erreur d'usage", resultat.stderr)


if __name__ == "__main__":
    unittest.main()
