# SPDX-License-Identifier: Apache-2.0
"""Tests de scripts/recette/noms_tests.py (et, à travers lui, d'inventaire_cargo.py) sur des sorties de cargo synthétiques.

Lancement : python3 -I -B -m unittest discover -s scripts/tests -p 'test_noms_tests.py' -v
Aucun de ces tests n'a besoin de cargo.
"""

import io
import importlib.util
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

ICI = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(ICI), "recette", "noms_tests.py")
spec = importlib.util.spec_from_file_location("noms_tests", SCRIPT)
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)

LISTE = """\
   Compiling kvu v0.1.0
    Finished `release` profile [optimized] target(s) in 1.00s
     Running unittests src/lib.rs (target/release/deps/kvu-0123)
a::tests::un: test
a::tests::deux: test
b::tests::gpu: test

3 tests, 0 benchmarks
     Running tests/cli.rs (target/release/deps/cli-4567)
cli_aide: test

1 test, 0 benchmarks
   Doc-tests kvu
0 tests, 0 benchmarks
"""
IGNORES = """\
     Running unittests src/lib.rs (target/release/deps/kvu-0123)
b::tests::gpu: test

1 test, 0 benchmarks
     Running tests/cli.rs (target/release/deps/cli-4567)
0 tests, 0 benchmarks
   Doc-tests kvu
0 tests, 0 benchmarks
"""
EXECUTION = """\
     Running unittests src/lib.rs (target/release/deps/kvu-0123)
running 3 tests
test a::tests::un ... ok
test a::tests::deux ... ok
test b::tests::gpu ... ignored, GPU
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
     Running tests/cli.rs (target/release/deps/cli-4567)
running 1 test
test cli_aide ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"""


class NomsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def ecrire(self, nom, texte):
        p = os.path.join(self.d.name, nom)
        with open(p, "w", encoding="utf-8") as f:
            f.write(texte)
        return p

    def lancer(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = nt.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def reference(self, liste=LISTE, ignores=IGNORES):
        rc, out, _ = self.lancer("reference", self.ecrire("l.out", liste), self.ecrire("i.out", ignores),
                                 "--commit", "abc", "--etape", "cpu", "--source", "essai")
        self.assertEqual(rc, 0)
        return self.ecrire("ref.txt", out)

    def test_reference_canonique(self):
        with open(self.reference(), encoding="utf-8") as f:
            texte = f.read()
        self.assertIn("# tests 4 actifs 3 ignores 1 cibles 2", texte)
        self.assertIn("unittests src/lib.rs\tb::tests::gpu\tignore", texte)
        self.assertNotIn("doc-tests", texte)   # exécutable sans test : omis

    def test_identique_et_execution_ok(self):
        ref = self.reference()
        rc, out, _ = self.lancer("verifier", ref, self.ecrire("l2", LISTE), self.ecrire("i2", IGNORES), self.ecrire("t", EXECUTION))
        self.assertEqual(rc, 0, out)
        self.assertIn("OK", out)

    def test_ajout_accepte(self):
        ref = self.reference()
        plus = LISTE.replace("cli_aide: test\n\n1 test,", "cli_aide: test\ncli_nouveau: test\n\n2 tests,")
        rc, out, _ = self.lancer("verifier", ref, self.ecrire("l2", plus), self.ecrire("i2", IGNORES))
        self.assertEqual(rc, 0, out)

    def test_retrait_refuse(self):
        ref = self.reference()
        moins = LISTE.replace("a::tests::deux: test\n", "").replace("3 tests, 0 benchmarks", "2 tests, 0 benchmarks")
        rc, out, _ = self.lancer("verifier", ref, self.ecrire("l2", moins), self.ecrire("i2", IGNORES))
        self.assertEqual(rc, 1)
        self.assertIn("a::tests::deux", out)

    def test_devenu_ignore_refuse(self):
        ref = self.reference()
        ign = IGNORES.replace("b::tests::gpu: test\n\n1 test,", "b::tests::gpu: test\na::tests::un: test\n\n2 tests,")
        rc, out, _ = self.lancer("verifier", ref, self.ecrire("l2", LISTE), self.ecrire("i2", ign))
        self.assertEqual(rc, 1)
        self.assertIn("devenu ignoré", out)

    def test_actif_non_execute_refuse(self):
        ref = self.reference()
        partielle = EXECUTION.replace("test a::tests::deux ... ok\n", "")
        rc, out, _ = self.lancer("verifier", ref, self.ecrire("l2", LISTE), self.ecrire("i2", IGNORES), self.ecrire("t", partielle))
        self.assertEqual(rc, 1)
        self.assertIn("absent de la sortie d'exécution", out)

    def test_sortie_tronquee_erreur(self):
        ref = self.reference()
        tronquee = LISTE.split("3 tests, 0 benchmarks")[0]
        rc, _, err = self.lancer("verifier", ref, self.ecrire("l2", tronquee), self.ecrire("i2", IGNORES))
        self.assertEqual(rc, 2)
        self.assertIn("résumé", err)


if __name__ == "__main__":
    unittest.main()
