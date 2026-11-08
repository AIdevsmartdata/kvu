#!/usr/bin/env python3
"""Références RG-QL-REF (WP22) : sorties de la référence Python BLOC2 (`operateurs.py`, LECTURE SEULE) pour les opérateurs
DEC, APE6, APE13, APE40 (mutant hors famille) et AGR1, sur des configurations fixées et hachées.

Usage (venv épinglé, UNE fois ; aucun fichier n'est écrit hors du dossier de sortie, bytecode désactivé) :
    /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python3 -I -B gen_references.py <dossier_sortie> [<racine grandL>]

Entrées :
  - les 3 fixtures de WP6 `validation/rg/fixtures/cfg_s{7,11,13}_q64.npy` (9⁴, amplitude 0,6 ; sha256 contrôlés contre leur manifeste) ;
  - si <racine grandL> est donnée : `L12_b2.3_c1/cfg_002000.npy` (c16, lecture seule), converti en quaternions par la formule de kvu
    `su2_hb::mat_to_quat` (moyennes : a0 = (Re m00 + Re m11)/2, a1 = (Im m01 + Im m10)/2, a2 = (Re m01 − Re m10)/2,
    a3 = (Im m00 − Im m11)/2), comme `gen_fixtures.py` de WP6.
Sorties : v_<OP>_<nom>_q64.npy (4, L/3, L/3, L/3, L/3, 4) float64 = `operateurs.grossier(operateurs.epais(q, OP))` ; manifeste.json
(sha256 des entrées, des sorties, de operateurs.py ; versions ; diagnostic min |s| des projections APE/AGR recalculé à côté).
"""
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

BLOC2 = "/chemin/vers/projet/chantiers/vision_20261025/calculs/BLOC2_operateurs/code/bloc2/operateurs.py"
ICI = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(ICI, "..", "rg", "fixtures")
OPS = ["DEC", "APE6", "APE13", "APE40", "AGR1"]
GRANDL = ("L12_b2.3_c1", "cfg_002000.npy")


def sha(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def charger_operateurs():
    spec = importlib.util.spec_from_file_location("bloc2_operateurs_lecture", BLOC2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quat_kvu(M):
    """matrices (...,2,2) -> quaternions (...,4) : formule de su2_hb::mat_to_quat (moyennes)."""
    a0 = (M[..., 0, 0].real + M[..., 1, 1].real) / 2
    a1 = (M[..., 0, 1].imag + M[..., 1, 0].imag) / 2
    a2 = (M[..., 0, 1].real - M[..., 1, 0].real) / 2
    a3 = (M[..., 0, 0].imag - M[..., 1, 1].imag) / 2
    return np.stack([a0, a1, a2, a3], axis=-1)


def norme_min(op, q, nom):
    """min |s| des projections faites par l'opérateur (diagnostic ; même arithmétique que ape_pas / epais_agr)."""
    if nom.startswith("APE"):
        n = int(nom[3:])
        m = np.inf
        for _ in range(n):
            out = np.empty_like(q)
            for mu in range(4):
                out[mu], nr = op.projeter((1 - 0.5) * q[mu] + (0.5 / 6.0) * op.agrafes(q, mu))
                m = min(m, float(nr.min()))
            q = out
        return m
    if nom.startswith("AGR"):
        c = float(nom[3:])
        m = np.inf
        for mu in range(4):
            S = op.droit(q, mu)
            tot = S.copy()
            for nu in range(4):
                if nu == mu:
                    continue
                a = op.qmul(op.qmul(q[nu], op.dec_sites(S, nu, 1)), op.qdag(op.dec_sites(q[nu], mu, 3)))
                unm = op.dec_sites(q[nu], nu, -1)
                b = op.qmul(op.qmul(op.qdag(unm), op.dec_sites(S, nu, -1)), op.dec_sites(unm, mu, 3))
                tot += c * (a + b)
            nr = np.linalg.norm(tot, axis=-1)[:: 3, :: 3, :: 3, :: 3]  # seules les projections des sites 3X sont gardées
            m = min(m, float(nr.min()))
        return m
    return None


def main():
    sortie = sys.argv[1]
    grandl = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(sortie, exist_ok=True)
    op = charger_operateurs()
    man_fix = json.load(open(os.path.join(FIXTURES, "manifeste.json")))
    entrees = []
    for g in (7, 11, 13):
        nom = f"cfg_s{g}_q64.npy"
        chemin = os.path.join(FIXTURES, nom)
        assert sha(chemin) == man_fix["fichiers"][nom], f"fixture {nom} : sha256 différent du manifeste de WP6"
        entrees.append((f"s{g}", chemin, np.load(chemin)))
    if grandl:
        chemin = os.path.join(grandl, *GRANDL)
        U = np.load(chemin, mmap_mode="r")
        assert U.dtype == np.complex128 and U.shape[0] == 4 and U.shape[5:] == (2, 2), U.shape
        entrees.append((f"{GRANDL[0]}_{GRANDL[1][:-4]}", chemin, quat_kvu(np.asarray(U))))
    man = {"format": "kvu-rg-ql-references/1", "operateurs_py": BLOC2, "operateurs_py_sha256": sha(BLOC2),
           "numpy": np.__version__, "python": sys.version.split()[0], "alpha_ape": 0.5,
           "conversion_c16": "su2_hb::mat_to_quat (moyennes)", "entrees": {}, "fichiers": {}, "norme_min_projection": {}}
    for nom_e, chemin, q in entrees:
        q = np.ascontiguousarray(q, dtype=np.float64)
        assert q.ndim == 6 and q.shape[0] == 4 and q.shape[-1] == 4 and q.shape[1] % 3 == 0, q.shape
        man["entrees"][nom_e] = {"chemin": os.path.relpath(chemin, ICI) if chemin.startswith(os.path.dirname(ICI)) else chemin,
                                 "sha256": sha(chemin), "L": int(q.shape[1]),
                                 "defaut_unitarite": float(np.abs(np.linalg.norm(q, axis=-1) - 1).max())}
        for o in OPS:
            V = op.grossier(op.epais(q, o))
            nom = f"v_{o}_{nom_e}_q64.npy"
            np.save(os.path.join(sortie, nom), np.ascontiguousarray(V, dtype="<f8"))
            man["fichiers"][nom] = sha(os.path.join(sortie, nom))
            nm = norme_min(op, q, o)
            man["norme_min_projection"][nom] = nm
            print(nom, "L", q.shape[1], "->", V.shape[1], "unitarité", float(np.abs(np.linalg.norm(V, axis=-1) - 1).max()),
                  "min|s|", nm, flush=True)
    with open(os.path.join(sortie, "manifeste.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print("manifeste écrit :", os.path.join(sortie, "manifeste.json"))


if __name__ == "__main__":
    main()
