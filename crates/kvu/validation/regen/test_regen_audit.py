# SPDX-License-Identifier: Apache-2.0
"""Vérifier l'audit WP4 et l'intégrité sur des ensembles temporaires."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[4] / "scripts/regen_audit.py"
SPECIFICATION = importlib.util.spec_from_file_location("regen_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPECIFICATION)
# Le chargement par chemin ne doit pas créer de cache dans le dépôt.
_ANCIEN_REGLAGE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPECIFICATION.loader.exec_module(AUDIT)
finally:
    sys.dont_write_bytecode = _ANCIEN_REGLAGE


def ecrire_json(chemin, valeur):
    chemin.write_text(json.dumps(valeur, ensure_ascii=False), encoding="utf-8")


def instantane(racine):
    """Relever noms, dates et tailles, y compris celles des répertoires."""
    chemins = [racine, *sorted(racine.rglob("*"))]
    return [(str(p.relative_to(racine)), p.lstat().st_mtime_ns, p.lstat().st_size)
            for p in chemins]


def du_sb(chemin):
    resultat = subprocess.run(["du", "-sb", "--", str(chemin)],
                              capture_output=True, text=True, check=True)
    return int(resultat.stdout.split()[0])


def rapport_fixe(chemin, valeurs, hors=0):
    """Construire un rapport kvu avec des octets fixés par classe."""
    chaines = []
    classes = {}
    for classe, source in zip("ABC", ("su2_hip/1", "su2_gen/1", None)):
        octets = valeurs.get(classe, 0)
        n = 1 if octets else 0
        classes[classe] = {"chaines": n, "octets": octets + (hors if classe == "C" else 0)}
        if n:
            chaines.append({"nom": "chaîne " + classe, "source": source, "classe": classe,
                            "raisons": ["raison " + classe], "fichiers": 1, "octets": octets})
    return {
        "simulation": True, "gbit_prouvees": ["su2_hip/1"],
        "ensembles": [{"racine": str(chemin), "total_fichiers": len(chaines) + bool(hors),
                       "total_octets": sum(valeurs.values()) + hors,
                       "classe_dominante": "A", "classes": classes,
                       "hors_chaines": {"fichiers": int(bool(hors)), "octets": hors},
                       "chaines": chaines}],
    }


class BaseTemporaire(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporaire.cleanup)
        self.dossier = Path(self.temporaire.name)

    def lancer(self, *arguments):
        return subprocess.run(["python3", "-I", str(SCRIPT), *map(str, arguments)],
                              cwd=self.dossier, capture_output=True, text=True, check=False)


class TestsAudit(BaseTemporaire):
    """Exercer la CLI avec un faux kvu shell et le véritable du -sb."""

    def setUp(self):
        super().setUp()
        self.racines = []
        self.rapports = {}
        for i in range(8):
            # Espaces et métacaractères vérifient l'absence de shell côté audit.
            chemin = self.dossier / f"ensemble {i} $sans_shell `inactif`"
            chemin.mkdir()
            (chemin / "donnees.bin").write_bytes(b"x" * (12000 + i))
            self.racines.append(chemin)
            self.rapports[chemin] = rapport_fixe(chemin, {"A": du_sb(chemin)})
        self.binaire = self.dossier / "faux kvu"
        self.liste = self.dossier / "ensembles.txt"
        self.sortie_json = self.dossier / "audit.json"
        self.sortie_md = self.dossier / "audit.md"
        self.ecrire_liste()
        self.ecrire_binaire()

    def ecrire_liste(self, nombre=7, chemins=None):
        chemins = chemins or [[p] for p in self.racines[:nombre]]
        self.liste.write_text("# liste d'essai\n\n" + "\n".join(
            f"essai {i}|{';'.join(map(str, groupe))}|note libre {i}"
            for i, groupe in enumerate(chemins)) + "\n", encoding="utf-8")

    def ecrire_binaire(self, echec=None, malforme=None):
        # Le programme ne lit aucun fichier de l'ensemble et ne crée aucun journal.
        lignes = ["#!/bin/sh", '[ "$1" = store ] && [ "$2" = classer ] || exit 93',
                  '[ "$3" = --chemin ] && [ "$5" = --json ] || exit 94', 'case "$4" in']
        for chemin, rapport in self.rapports.items():
            lignes.append(shlex.quote(str(chemin)) + ")")
            if chemin == echec:
                lignes.extend(["printf '%s\\n' 'échec simulé du classement' >&2", "exit 17"])
            else:
                contenu = "{invalide" if chemin == malforme else json.dumps(rapport, ensure_ascii=False)
                lignes.append("printf '%s\\n' " + shlex.quote(contenu))
            lignes.append(";;")
        lignes.extend(["*) exit 95 ;;", "esac"])
        self.binaire.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        self.binaire.chmod(0o755)

    def audit(self, *options):
        avant = [instantane(p) for p in self.racines]
        resultat = self.lancer("--kvu", self.binaire, "--ensembles", self.liste,
                              "--sortie-json", self.sortie_json, "--sortie-md", self.sortie_md,
                              *options)
        self.assertEqual([instantane(p) for p in self.racines], avant,
                         "l'audit a modifié un ensemble")
        return resultat

    def test_sept_ensembles_pass_et_sorties(self):
        preuve = self.dossier / "preuves.json"
        ecrire_json(preuve, {})
        resultat = self.audit("--preuve-gbit", preuve)
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        self.assertIn("PASS AUDIT-7", resultat.stdout)
        self.assertEqual(sum(l.startswith("INFO ") for l in resultat.stdout.splitlines()), 7)
        brut = self.sortie_json.read_text(encoding="utf-8")
        rapport = json.loads(brut)
        self.assertEqual(rapport["schema"], "kvu-audit-ensembles/1")
        self.assertEqual(rapport["verdict"]["AUDIT-7"], "PASS")
        self.assertEqual(rapport["binaire_sha256"], hashlib.sha256(self.binaire.read_bytes()).hexdigest())
        self.assertEqual(rapport["preuves_gbit"], ["su2_hip/1"])
        self.assertEqual(rapport["total_octets"], rapport["total_du_sb"])
        self.assertEqual(rapport["ecart_total_relatif"], 0)
        self.assertEqual(brut, json.dumps(rapport, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        for ensemble in rapport["ensembles"]:
            self.assertEqual(ensemble["ecart_relatif"], 0)
            self.assertEqual(ensemble["classe_dominante"], "A")
            self.assertEqual(ensemble["regenerable"],
                             "oui, bit-identique après rejeu de contrôle (REGEN-2)")
        markdown = self.sortie_md.read_text(encoding="utf-8")
        self.assertIn("| ensemble | classe | Go mesurés | chaînes A/B/C | régénérable | raison |", markdown)
        self.assertIn("## essai 0", markdown)
        self.assertIn("raison A ; note libre 0", markdown)
        self.assertIn("PASS AUDIT-7", markdown)

    def test_fusion_classes_sources_raisons_et_hors_chaines(self):
        p, q = self.racines[:2]
        mesure_p, mesure_q = du_sb(p), du_sb(q)
        self.rapports[p] = rapport_fixe(p, {"A": mesure_p - 20}, hors=20)
        self.rapports[q] = rapport_fixe(q, {"A": 100, "B": 200, "C": mesure_q - 300})
        # Compter les chaînes concernées, même si leur liste contient une répétition.
        self.rapports[p]["ensembles"][0]["chaines"][0]["raisons"] = ["commun", "commun", "rare"]
        self.rapports[q]["ensembles"][0]["chaines"][0]["raisons"] = ["commun"]
        self.ecrire_binaire()
        self.ecrire_liste(chemins=[[p, q], *[[r] for r in self.racines[2:]]])
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        ensemble = json.loads(self.sortie_json.read_text())["ensembles"][0]
        self.assertEqual(ensemble["chaines"], {
            "A": {"n": 2, "octets": mesure_p + 80},
            "B": {"n": 1, "octets": 200},
            "C": {"n": 1, "octets": mesure_q - 280},
        })
        self.assertEqual(ensemble["hors_chaines"], {"fichiers": 1, "octets": 20})
        self.assertEqual(ensemble["octets"], mesure_p + mesure_q)
        self.assertEqual(ensemble["du_sb"], mesure_p + mesure_q)
        self.assertEqual(ensemble["fichiers"], 5)
        self.assertEqual(ensemble["sources"], {"su2_hip/1": 2, "su2_gen/1": 1, "(non reconnu)": 1})
        self.assertEqual(ensemble["raisons"]["A"], [["commun", 2], ["rare", 1]])
        self.assertEqual(ensemble["classe_dominante"], "A")
        self.assertEqual(ensemble["go"], round((mesure_p + mesure_q) / 2**30, 2))
        self.assertEqual(ensemble["regenerable"], "A : 0.00 Go, B : 0.00 Go, C : 0.00 Go")

    def test_six_ensembles_fail(self):
        self.ecrire_liste(6)
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("FAIL AUDIT-7 6/7", resultat.stdout)

    def test_reunion_sans_double_compte_des_chemins_emboites_ou_repetes(self):
        parent = self.racines[0]
        enfant = parent / "enfant"
        enfant.mkdir()
        (enfant / "donnees.bin").write_bytes(b"y" * 9000)
        self.rapports[parent] = rapport_fixe(parent, {"A": du_sb(parent)})
        self.rapports[enfant] = rapport_fixe(enfant, {"B": du_sb(enfant)})
        self.ecrire_liste(chemins=[[enfant, parent, parent],
                                 *[[p] for p in self.racines[1:7]]])
        self.ecrire_binaire()
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        ensemble = json.loads(self.sortie_json.read_text())["ensembles"][0]
        self.assertEqual(ensemble["octets"], du_sb(parent))
        self.assertEqual(ensemble["du_sb"], du_sb(parent))
        self.assertEqual(ensemble["chaines"]["A"]["n"], 1)
        self.assertEqual(ensemble["chaines"]["B"]["n"], 0)

    def test_ecart_individuel_superieur_un_pourcent_fail(self):
        p = self.racines[0]
        self.rapports[p] = rapport_fixe(p, {"A": du_sb(p) * 102 // 100})
        self.ecrire_binaire()
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 8)
        rapport = json.loads(self.sortie_json.read_text())
        self.assertGreater(rapport["ensembles"][0]["ecart_relatif"], 0.01)
        self.assertLess(rapport["ecart_total_relatif"], 0.01)
        self.assertEqual(rapport["verdict"]["AUDIT-7"], "FAIL")

    def test_ecart_inferieur_un_pourcent_pass(self):
        p = self.racines[0]
        self.rapports[p] = rapport_fixe(p, {"A": du_sb(p) * 1005 // 1000})
        self.ecrire_binaire()
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        ensemble = json.loads(self.sortie_json.read_text())["ensembles"][0]
        self.assertAlmostEqual(ensemble["ecart_relatif"],
                               abs(ensemble["octets"] - du_sb(p)) / du_sb(p))

    def test_rc_non_nul_et_stderr_kvu_fail(self):
        self.ecrire_binaire(echec=self.racines[0])
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("FAIL AUDIT-7", resultat.stdout)
        self.assertIn("code 17", resultat.stdout)
        self.assertIn("échec simulé du classement", resultat.stdout)

    def test_json_kvu_malforme_fail(self):
        self.ecrire_binaire(malforme=self.racines[0])
        resultat = self.audit()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("classement impossible", resultat.stdout)

    def test_sorties_sous_ensemble_refusees_avant_toute_ecriture(self):
        for option in ("--sortie-json", "--sortie-md"):
            with self.subTest(option=option):
                interdit = self.racines[0] / "sous-dossier" / "rapport"
                resultat = self.audit(option, interdit)
                self.assertEqual(resultat.returncode, 2, resultat.stderr)
                self.assertIn("sortie interdite sous un ensemble", resultat.stderr)
                self.assertFalse(self.sortie_json.exists())
                self.assertFalse(self.sortie_md.exists())

    def test_sortie_existante_sous_ensemble_preservee(self):
        cible = self.racines[0] / "donnees.bin"
        resultat = self.audit("--sortie-json", cible)
        self.assertEqual(resultat.returncode, 2)
        self.assertEqual(cible.read_bytes(), b"x" * 12000)

    def test_liens_symboliques_et_physiques_refuses(self):
        alias = self.dossier / "alias"
        alias.symlink_to(self.racines[0], target_is_directory=True)
        lien_fichier = self.dossier / "lien.json"
        lien_fichier.symlink_to(self.racines[0] / "donnees.bin")
        lien_physique = self.dossier / "physique.json"
        os.link(self.racines[0] / "donnees.bin", lien_physique)
        for cible in (alias / "nouveau.json", lien_fichier, lien_physique):
            with self.subTest(cible=cible):
                resultat = self.audit("--sortie-json", cible)
                self.assertEqual(resultat.returncode, 2, resultat.stderr)
                self.assertFalse(self.sortie_md.exists())

    def test_egalite_classes_et_textes_regenerable(self):
        p = self.racines[0]
        mesure = du_sb(p)
        # Une différence d'un octet avec du reste largement sous le seuil.
        for valeurs, dominante, texte in (
            ({"A": mesure // 2, "B": mesure // 2}, "B", "A :"),
            ({"B": mesure // 2, "C": mesure // 2}, "C", "A :"),
            ({"B": mesure}, "B", "statistiquement (nouvelle réalisation)"),
            ({"C": mesure}, "C", "non : à garder"),
        ):
            with self.subTest(valeurs=valeurs):
                self.rapports[p] = rapport_fixe(p, valeurs)
                self.ecrire_binaire()
                resultat = self.audit()
                self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
                ensemble = json.loads(self.sortie_json.read_text())["ensembles"][0]
                self.assertEqual(ensemble["classe_dominante"], dominante)
                self.assertTrue(ensemble["regenerable"].startswith(texte))


class TestsIntegrite(BaseTemporaire):
    """Hacher trois chaînes jouets et vérifier leur immutabilité."""

    def setUp(self):
        super().setUp()
        self.ensemble = self.dossier / "ensemble"
        self.ensemble.mkdir()
        self.chaines = []
        self.octets_attendus = 0
        for i in range(3):
            chaine = self.ensemble / f"chaîne {i}"
            chaine.mkdir()
            (chaine / "sorties").mkdir()
            donnees = bytes([i]) * (1000 + i)
            csv = f"indice,valeur\n{i},0.5\n".encode()
            (chaine / "sorties" / "configuration.bin").write_bytes(donnees)
            (chaine / "mesures.csv").write_bytes(csv)
            (chaine / "COMPLETE").write_text("terminé\n", encoding="utf-8")
            (chaine / "journal.txt").write_text("information hors référence\n", encoding="utf-8")
            ecrire_json(chaine / "manifest.json", {
                "schema": "su2_gen/1", "engine": "su2_gen", "statut": "COMPLETE",
                "sorties_sha256": {"sorties/configuration.bin": hashlib.sha256(donnees).hexdigest()},
                "csv_sha256": {"mesures.csv": hashlib.sha256(csv).hexdigest()},
            })
            self.chaines.append(chaine)
            self.octets_attendus += len(donnees) + len(csv)
        self.sortie_json = self.dossier / "integrite.json"

    def integrite(self, *options):
        avant = instantane(self.ensemble)
        resultat = self.lancer("--integrite", self.ensemble, "--fils", "2",
                              "--sortie-json", self.sortie_json, *options)
        self.assertEqual(instantane(self.ensemble), avant,
                         "l'intégrité a modifié l'arbre audité")
        return resultat

    def test_tout_ok_et_non_listes_informatifs(self):
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        self.assertIn("PASS REGEN-INTEG 6/6 fichiers de 3 chaînes", resultat.stdout)
        self.assertIn(f"{self.octets_attendus} octets lus", resultat.stdout)
        self.assertIn("chaîne 3/3", resultat.stderr)
        self.assertIn("débit", resultat.stderr)
        self.assertIn("journal.txt : fichier non listé", resultat.stderr)
        rapport = json.loads(self.sortie_json.read_text())
        self.assertEqual(rapport["totaux"]["octets_lus"], self.octets_attendus)
        self.assertEqual(rapport["totaux"]["n_ok"], 6)
        for chaine in rapport["chaines"]:
            self.assertEqual(chaine["non_listes"], ["journal.txt"])
            self.assertEqual([f["statut"] for f in chaine["fichiers"]], ["OK", "OK"])
            self.assertEqual(chaine["erreurs"], [])

    def test_octet_modifie_different(self):
        cible = self.chaines[1] / "sorties" / "configuration.bin"
        donnees = cible.read_bytes()
        cible.write_bytes(b"z" + donnees[1:])
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("DIFFERENT", resultat.stdout)
        self.assertIn("FAIL REGEN-INTEG 5/6", resultat.stdout)
        rapport = json.loads(self.sortie_json.read_text())
        self.assertEqual(rapport["totaux"]["octets_lus"], self.octets_attendus)
        self.assertEqual(rapport["chaines"][1]["fichiers"][1]["statut"], "DIFFERENT")

    def test_fichier_absent(self):
        cible = self.chaines[0] / "mesures.csv"
        taille = cible.stat().st_size
        cible.unlink()
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("ABSENT", resultat.stdout)
        rapport = json.loads(self.sortie_json.read_text())
        self.assertEqual(rapport["totaux"]["octets_lus"], self.octets_attendus - taille)
        self.assertEqual(rapport["totaux"]["n_total"], 6)

    def test_manifeste_sans_table(self):
        ecrire_json(self.chaines[0] / "manifest.json", {"schema": "su2_gen/1"})
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("aucune référence de hachage", resultat.stdout)
        self.assertIn("FAIL REGEN-INTEG", resultat.stdout)

    def test_tables_vides_ou_invalides_et_json_invalide(self):
        manifeste = self.chaines[0] / "manifest.json"
        for contenu in ("{", '[]', '{"sorties_sha256": {}, "csv_sha256": {}}',
                        '{"sorties_sha256": [], "csv_sha256": {}}'):
            with self.subTest(contenu=contenu):
                manifeste.write_text(contenu, encoding="utf-8")
                resultat = self.integrite()
                self.assertEqual(resultat.returncode, 8, resultat.stdout + resultat.stderr)

    def test_ne_descend_pas_sous_une_chaine(self):
        imbriquee = self.chaines[0] / "imbriquée"
        imbriquee.mkdir()
        ecrire_json(imbriquee / "manifest.json", {})
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        rapport = json.loads(self.sortie_json.read_text())
        self.assertEqual(rapport["totaux"]["n_chaines"], 3)
        self.assertIn("imbriquée/manifest.json", rapport["chaines"][0]["non_listes"])

    def test_racine_peut_etre_une_chaine_et_sortie_optionnelle(self):
        avant = instantane(self.ensemble)
        resultat = self.lancer("--integrite", self.chaines[0])
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        self.assertIn("PASS REGEN-INTEG 2/2 fichiers de 1 chaînes", resultat.stdout)
        self.assertFalse(self.sortie_json.exists())
        self.assertEqual(instantane(self.ensemble), avant)

    def test_sortie_sous_ensemble_refusee(self):
        resultat = self.integrite("--sortie-json", self.ensemble / "rapport.json")
        self.assertEqual(resultat.returncode, 2)
        self.assertFalse(self.sortie_json.exists())

    def test_reference_hors_chaine_refusee(self):
        ecrire_json(self.chaines[0] / "manifest.json", {
            "sorties_sha256": {"../inconnu.bin": "a" * 64}})
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("chemin de référence hors de la chaîne", resultat.stdout)

    def test_ensemble_vide_fail(self):
        vide = self.dossier / "vide"
        vide.mkdir()
        avant = instantane(vide)
        resultat = self.lancer("--integrite", vide)
        self.assertEqual(resultat.returncode, 8)
        self.assertIn("aucune chaîne", resultat.stdout)
        self.assertEqual(instantane(vide), avant)

    def test_progression_toutes_les_dix_chaines(self):
        for i in range(3, 11):
            chaine = self.ensemble / f"chaîne {i:02}"
            chaine.mkdir()
            (chaine / "vide.bin").write_bytes(b"")
            ecrire_json(chaine / "manifest.json", {
                "sorties_sha256": {"vide.bin": hashlib.sha256(b"").hexdigest()}})
        resultat = self.integrite()
        self.assertEqual(resultat.returncode, 0, resultat.stdout + resultat.stderr)
        self.assertIn("chaîne 10/11", resultat.stderr)
        self.assertIn("chaîne 11/11", resultat.stderr)

    def test_priorite_avant_lecture_et_absence_ionice_toleree(self):
        evenements = []
        options = AUDIT._arguments(["--integrite", str(self.ensemble)])

        def nice(valeur):
            evenements.append(("nice", valeur))

        def lancer(arguments):
            evenements.append(("ionice", arguments))
            raise FileNotFoundError("ionice absent")

        def chaines(racine):
            evenements.append(("parcours", racine))
            return []

        with mock.patch.object(AUDIT.os, "nice", side_effect=nice), \
                mock.patch.object(AUDIT, "_executer", side_effect=lancer), \
                mock.patch.object(AUDIT, "_trouver_chaines", side_effect=chaines), \
                mock.patch("builtins.print") as afficher:
            self.assertEqual(AUDIT.integrite(options), 8)
        self.assertEqual([e[0] for e in evenements], ["nice", "ionice", "parcours"])
        self.assertEqual(evenements[0][1], 19)
        self.assertEqual(evenements[1][1], ["ionice", "-c3", "-p", str(os.getpid())])
        self.assertTrue(any("ionice" in str(appel) and "stderr" in str(appel)
                            for appel in afficher.call_args_list))


class TestsUsage(BaseTemporaire):
    def test_arguments_invalides(self):
        for arguments in ((), ("--inconnu",), ("--kvu",),
                          ("--integrite", self.dossier, "--fils", "0"),
                          ("--integrite", self.dossier, "--fils", "x"),
                          ("--integrite", self.dossier, "--sortie-md", "rapport.md")):
            with self.subTest(arguments=arguments):
                resultat = self.lancer(*arguments)
                self.assertEqual(resultat.returncode, 2, resultat.stderr)
                self.assertIn("Erreur d'usage", resultat.stderr)


if __name__ == "__main__":
    unittest.main()
