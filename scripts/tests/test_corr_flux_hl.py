# SPDX-License-Identifier: Apache-2.0
"""Tests de `scripts/spectro/corr_flux_hl.py` (unittest, sans pytest).

Chaque test fabrique ses propres jeux synthétiques (dossier temporaire,
ops aléatoires reproductibles, petit tt, tt impair, nlev = 1 et 2) et compare
la sortie du script à un calcul DIRECT en Python pur écrit ici d'après la
formule du Rust (`flux.rs::base_corr`, `flux.rs::corr_flux`,
`correlateur.rs::paires`, `correlateur.rs::correlateur` avec `Mutant::Aucun`).

Lancement (depuis la racine de l'espace) :
    python3 -I -B -m unittest discover -s scripts/tests -p test_corr_flux_hl.py -v
"""

import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(RACINE, "scripts", "spectro", "corr_flux_hl.py")
EXEMPLE = os.path.join(RACINE, "entrees", "exemple")


# --------------------------------------------------------------------------
# Calcul DIRECT de référence (Python pur, transcription littérale du Rust)
# --------------------------------------------------------------------------
def base_corr_direct(ops, nlev, tt):
    """flux.rs::base_corr -> v[lev*5+k][t] = [valeur, 0.0]."""
    v = [[[0.0, 0.0] for _ in range(tt)] for _ in range(nlev * 5)]
    for t in range(tt):
        for lev in range(nlev):
            r = ops[(t * nlev + lev) * 7:(t * nlev + lev + 1) * 7]
            b = [r[0] + r[1], r[2] + r[3], r[4], r[5], r[6]]
            for k in range(5):
                v[lev * 5 + k][t] = [b[k], 0.0]
    return v


def paires_direct(n):
    """correlateur.rs::paires."""
    return [(a, b) for a in range(n) for b in range(a, n)]


def correlateur_direct(phis, tau_max):
    """correlateur.rs::correlateur avec Mutant::Aucun."""
    tt = len(phis[0])
    out = []
    for (a, b) in paires_direct(len(phis)):
        ligne = []
        for tau in range(tau_max + 1):
            acc = 0.0
            for t in range(tt):
                t2 = (t + tau) % tt
                x, y = phis[a][t], phis[b][t2]
                x2, y2 = phis[b][t], phis[a][t2]
                acc += x[0] * y[0] + x[1] * y[1] + x2[0] * y2[0] + x2[1] * y2[1]
            ligne.append(acc / (2.0 * tt))
        out.append(ligne)
    return out


def corr_direct(ops, nlev, tt, tau_max):
    """corr_flux.rs::corr_flux : liste [paire][tau]."""
    return correlateur_direct(base_corr_direct(ops, nlev, tt), tau_max)


def entete_direct(nlev, tau_max):
    cols = ["cfg", "px", "py"]
    for (a, b) in paires_direct(nlev * 5):
        for tau in range(tau_max + 1):
            cols.append("c%d_%d_%d" % (a, b, tau))
    return cols


# --------------------------------------------------------------------------
# Outils de fabrication de jeux synthétiques
# --------------------------------------------------------------------------
def ecrire_jeu(dossier, cfg, tt, nlev, tau_max=3, graine=1234, cotes=None):
    """Écrit manifeste.json + ops_flux.bin ; renvoie les ops (liste plate par cfg)."""
    os.makedirs(dossier, exist_ok=True)
    rng = random.Random(graine)
    nops = nlev * 7
    toutes = []
    with open(os.path.join(dossier, "ops_flux.bin"), "wb") as f:
        for _ in range(cfg):
            vals = [rng.uniform(-3.0, 3.0) for _ in range(tt * nops)]
            toutes.append(vals)
            f.write(struct.pack("<%dd" % len(vals), *vals))
    if cotes is None:
        cotes = [4, 4, 4, tt]
    man = {
        "schema": "kvu-spectro-run/1",
        "cotes": cotes,
        "temps": 3,
        "tau_max": tau_max,
        "secteur_flux": {"niveaux": list(range(nlev)), "tau_max": tau_max},
    }
    with open(os.path.join(dossier, "manifeste.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    return toutes


def lancer(dossier, *args, cwd=None):
    """Lance le script ; renvoie (code, stdout, stderr)."""
    cmd = [sys.executable, "-I", "-B", SCRIPT, dossier] + list(args)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or tempfile.gettempdir())
    return p.returncode, p.stdout, p.stderr


def lire_csv(chemin):
    with open(chemin, "r", encoding="ascii") as f:
        lignes = f.read().split("\n")
    if lignes and lignes[-1] == "":
        lignes.pop()
    entete = lignes[0].split(",")
    lignes_valeurs = [l.split(",") for l in lignes[1:]]
    return entete, lignes_valeurs


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="corrflux_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dossier(self, nom="d"):
        return os.path.join(self.tmp, nom)

    def verifier_egal_direct(self, dossier, toutes, tt, nlev, tau_max, tol=1e-12):
        """Compare le CSV produit au calcul direct, valeur par valeur."""
        code, out, err = lancer(dossier)
        self.assertEqual(code, 0, "code %d, stderr=%s" % (code, err))
        resume = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(resume["cfg"], len(toutes))
        self.assertEqual(resume["tt"], tt)
        self.assertEqual(resume["nlev"], nlev)
        self.assertEqual(resume["tau_max"], tau_max)
        npaires = (nlev * 5) * (nlev * 5 + 1) // 2
        self.assertEqual(resume["colonnes"], npaires * (tau_max + 1))
        chemin = os.path.join(dossier, "corr_flux_hl.csv")
        self.assertEqual(os.path.abspath(resume["sortie"]), os.path.abspath(chemin))
        entete, lignes = lire_csv(chemin)
        self.assertEqual(entete, entete_direct(nlev, tau_max))
        self.assertEqual(len(lignes), len(toutes))
        for icfg, ops in enumerate(toutes):
            ref = corr_direct(ops, nlev, tt, tau_max)
            ligne = lignes[icfg]
            self.assertEqual(ligne[0], str(icfg))
            self.assertEqual(ligne[1], "0")
            self.assertEqual(ligne[2], "0")
            vals = [float(x) for x in ligne[3:]]
            plat = [v for paire in ref for v in paire]
            self.assertEqual(len(vals), len(plat))
            for i, (a, b) in enumerate(zip(vals, plat)):
                self.assertAlmostEqual(a, b, delta=tol * max(1.0, abs(b)),
                                       msg="cfg %d colonne %d (%s)" % (icfg, i, entete[3 + i]))
        return entete, lignes


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
class TestValeurs(Base):
    def test_nlev1_tt_impair(self):
        """nlev=1, tt=5 (impair) : comparaison au calcul direct."""
        d = self.dossier()
        toutes = ecrire_jeu(d, cfg=3, tt=5, nlev=1, tau_max=3)
        self.verifier_egal_direct(d, toutes, 5, 1, 3)

    def test_nlev2_tt_impair(self):
        """nlev=2, tt=7 (impair) : 15 opérateurs de base, 120 paires."""
        d = self.dossier()
        toutes = ecrire_jeu(d, cfg=2, tt=7, nlev=2, tau_max=4)
        self.verifier_egal_direct(d, toutes, 7, 2, 4)

    def test_tau_max_superieur_a_tt(self):
        """tau_max = 2*tt+1 : décalages modulo tt (comme le Rust)."""
        d = self.dossier()
        toutes = ecrire_jeu(d, cfg=2, tt=3, nlev=1, tau_max=7)
        entete, lignes = self.verifier_egal_direct(d, toutes, 3, 1, 7)
        # propriété du repli modulo tt : C(tau) == C(tau mod tt)
        vals = [float(x) for x in lignes[0][3:]]
        ntau = 8
        for ip in range(len(vals) // ntau):
            bloc = vals[ip * ntau:(ip + 1) * ntau]
            for tau in range(ntau):
                self.assertAlmostEqual(bloc[tau], bloc[tau % 3], delta=1e-12,
                                       msg="paire %d tau %d" % (ip, tau))

    def test_valeur_connue_analytique(self):
        """Cas à valeur connue : ops constants en t -> C_ab(tau) = x_a*x_b pour tout tau."""
        d = self.dossier()
        os.makedirs(d, exist_ok=True)
        tt, nlev = 4, 1
        ops = [0.5, 1.5, -2.0, 3.0, 0.25, 1.0, -1.0] * tt  # 7 ops répétés sur tt
        with open(os.path.join(d, "ops_flux.bin"), "wb") as f:
            f.write(struct.pack("<%dd" % len(ops), *ops))
        with open(os.path.join(d, "manifeste.json"), "w", encoding="utf-8") as f:
            json.dump({"cotes": [2, 2, 2, tt], "temps": 3, "tau_max": 2,
                       "secteur_flux": {"niveaux": [0]}}, f)
        code, out, err = lancer(d)
        self.assertEqual(code, 0, err)
        entete, lignes = lire_csv(os.path.join(d, "corr_flux_hl.csv"))
        base = [0.5 + 1.5, -2.0 + 3.0, 0.25, 1.0, -1.0]
        attendu = {}
        for (a, b) in paires_direct(5):
            for tau in range(3):
                attendu["c%d_%d_%d" % (a, b, tau)] = base[a] * base[b]
        ligne = dict(zip(entete, lignes[0]))
        for nom, v in attendu.items():
            self.assertAlmostEqual(float(ligne[nom]), v, delta=1e-12, msg=nom)

    def test_entete_et_ordre_des_paires(self):
        """En-tête exact : paires ia<=ib, ia croissant puis ib, puis tau = 0..tau_max."""
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=2, tau_max=2)
        code, out, err = lancer(d)
        self.assertEqual(code, 0, err)
        entete, _ = lire_csv(os.path.join(d, "corr_flux_hl.csv"))
        self.assertEqual(entete, entete_direct(2, 2))
        self.assertEqual(entete[:6], ["cfg", "px", "py", "c0_0_0", "c0_0_1", "c0_0_2"])
        self.assertEqual(entete[6], "c0_1_0")
        self.assertEqual(entete[-1], "c9_9_2")

    def test_format_et_determinisme(self):
        """Format %.17e, et deux lancements identiques -> fichiers identiques octet pour octet."""
        d = self.dossier()
        ecrire_jeu(d, cfg=2, tt=4, nlev=1, tau_max=2)
        code, _, err = lancer(d, "--out", os.path.join(self.tmp, "a.csv"))
        self.assertEqual(code, 0, err)
        code, _, err = lancer(d, "--out", os.path.join(self.tmp, "b.csv"))
        self.assertEqual(code, 0, err)
        with open(os.path.join(self.tmp, "a.csv"), "rb") as f:
            a = f.read()
        with open(os.path.join(self.tmp, "b.csv"), "rb") as f:
            b = f.read()
        self.assertEqual(a, b)
        texte = a.decode("ascii")
        self.assertTrue(texte.endswith("\n"))
        premiere = texte.split("\n")[1].split(",")
        for champ in premiere[3:]:
            self.assertRegex(champ, r"^-?[0-9]\.[0-9]{17}e[+-][0-9]{2,3}$")

    def test_tau_max_option_et_cwd_quelconque(self):
        """--tau-max 0 et 1 : nombre de colonnes et valeurs conformes ; lancé depuis un autre cwd."""
        d = self.dossier()
        toutes = ecrire_jeu(d, cfg=2, tt=4, nlev=1, tau_max=3)
        for tm in (0, 1):
            code, out, err = lancer(d, "--tau-max", str(tm), cwd=self.tmp)
            self.assertEqual(code, 0, err)
            resume = json.loads(out.strip().splitlines()[-1])
            self.assertEqual(resume["tau_max"], tm)
            self.assertEqual(resume["colonnes"], 15 * (tm + 1))
            entete, lignes = lire_csv(os.path.join(d, "corr_flux_hl.csv"))
            self.assertEqual(entete, entete_direct(1, tm))
            ref = corr_direct(toutes[0], 1, 4, tm)
            vals = [float(x) for x in lignes[0][3:]]
            self.assertEqual(len(vals), len(ref) * (tm + 1))
            for i, (a, b) in enumerate(zip(vals, [v for p in ref for v in p])):
                self.assertAlmostEqual(a, b, delta=1e-12, msg="colonne %d" % i)


class TestRefus(Base):
    def _verifier_refus(self, d, *args):
        sortie = os.path.join(d, "corr_flux_hl.csv")
        code, out, err = lancer(d, *args)
        self.assertEqual(code, 2, "code %d (attendu 2), stdout=%r stderr=%r" % (code, out, err))
        self.assertTrue(err.strip(), "message d'erreur attendu sur stderr")
        self.assertFalse(os.path.exists(sortie), "aucun fichier de sortie ne doit être écrit")
        self.assertFalse(os.path.exists(sortie + ".tmp"), "aucun fichier temporaire ne doit rester")
        return err

    def test_refus_manifeste_absent(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        os.remove(os.path.join(d, "manifeste.json"))
        self._verifier_refus(d)

    def test_refus_binaire_absent(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        os.remove(os.path.join(d, "ops_flux.bin"))
        self._verifier_refus(d)

    def test_refus_sans_secteur_flux(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        with open(os.path.join(d, "manifeste.json"), "w", encoding="utf-8") as f:
            json.dump({"cotes": [2, 2, 2, 3], "temps": 3, "tau_max": 2}, f)
        self._verifier_refus(d)

    def test_refus_taille_non_multiple(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        with open(os.path.join(d, "ops_flux.bin"), "ab") as f:
            f.write(b"\x00" * 8)  # 8 octets de trop
        self._verifier_refus(d)

    def test_refus_binaire_vide(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        open(os.path.join(d, "ops_flux.bin"), "wb").close()
        self._verifier_refus(d)

    def test_refus_valeur_non_finie(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        chemin = os.path.join(d, "ops_flux.bin")
        with open(chemin, "r+b") as f:
            f.seek(8 * 5)  # 6e valeur
            f.write(struct.pack("<d", float("nan")))
        self._verifier_refus(d)

    def test_refus_tau_max_negatif(self):
        d = self.dossier()
        ecrire_jeu(d, cfg=1, tt=3, nlev=1)
        self._verifier_refus(d, "--tau-max", "-1")


class TestExemple(unittest.TestCase):
    def test_exemple_egal_au_rust(self):
        """Facultatif : la sortie doit être égale à entrees/exemple/corr_flux.csv (écrit par le Rust)."""
        if not os.path.isdir(EXEMPLE):
            self.skipTest("entrees/exemple absent")
        tmp = tempfile.mkdtemp(prefix="corrflux_ex_")
        try:
            out = os.path.join(tmp, "corr.csv")
            code, so, err = lancer(EXEMPLE, "--out", out)
            self.assertEqual(code, 0, err)
            with open(out, "r", encoding="ascii") as f:
                a = f.read().split("\n")
            with open(os.path.join(EXEMPLE, "corr_flux.csv"), "r", encoding="ascii") as f:
                b = f.read().split("\n")
            self.assertEqual(len(a), len(b))
            self.assertEqual(a[0], b[0])
            for i, (la, lb) in enumerate(zip(a[1:], b[1:])):
                if not la and not lb:
                    continue
                va = la.split(",")
                vb = lb.split(",")
                self.assertEqual(va[:3], vb[:3], "ligne %d" % i)
                self.assertEqual(len(va), len(vb), "ligne %d" % i)
                for j, (x, y) in enumerate(zip(va[3:], vb[3:])):
                    self.assertAlmostEqual(float(x), float(y), delta=1e-10 * max(1.0, abs(float(y))),
                                           msg="ligne %d colonne %d" % (i, j))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
