#!/usr/bin/env python3
"""Recalcule les observables d'une configuration q64 avec la référence QW1."""

import sys

sys.dont_write_bytecode = True

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

# Le mode isolé retire le répertoire du script du chemin de recherche.
_spec = importlib.util.spec_from_file_location("analyse_dir", Path(__file__).with_name("analyse_dir.py"))
_analyse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyse)

os.environ.pop("QW1_MUTANT", None)
import box_su2


def main(argv=None):
    parseur = _analyse.Parseur(description=__doc__)
    parseur.add_argument("--cfg", required=True, metavar="X.npy", help="configuration q64 à lire")
    parseur.add_argument("--ell", required=True, type=_analyse.entier, metavar="L",
                         help="côté pair de la boîte, supérieur ou égal à 2")
    parseur.add_argument("--centre-decale", action="store_true",
                         help="contrôle négatif : mesurer au centre c+1")
    args = parseur.parse_args(argv)
    try:
        ell = _analyse._ell(args.ell)
        a = np.load(args.cfg, allow_pickle=False)
        if not isinstance(a, np.ndarray) or a.dtype != np.dtype("float64"):
            raise ValueError("la configuration doit être un tableau NPY float64")
        if a.shape != (4,) + (ell + 2,) * 4 + (4,):
            raise ValueError("forme attendue : (4, ell+2, ell+2, ell+2, ell+2, 4)")
        if not np.isfinite(a).all():
            raise ValueError("la configuration contient des valeurs non finies")
        n = ell + 1
        u = np.transpose(a, (0, 5, 1, 2, 3, 4))[:, :, :n, :n, :n, :n]
        box = box_su2.Box(ell, 2.5)
        box.U = u
        p = box.plaq_all()
        d = box_su2.depth_classes(ell)
        profil = [float(p[d == couche].mean()) for couche in range(ell // 2)]
        if args.centre_decale:
            # Un décalage de -1 lit en x la valeur de l'ancien site x+1.
            box.U = np.roll(u, -1, axis=(2, 3, 4, 5))
            p = box.plaq_all()
        pc, w2 = box_su2.measure(box, p)
        resultat = {"ell": ell, "backend": "python-qw1", "pc": float(pc),
                    "w2": float(w2) if ell >= 4 else None, "profil": profil}
        # L'encodeur JSON de Python emploie repr(float), exact à la relecture.
        print(json.dumps(resultat, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as erreur:
        parseur.error(f"configuration invalide ({type(erreur).__name__}) : {erreur}")


if __name__ == "__main__":
    sys.exit(main())
