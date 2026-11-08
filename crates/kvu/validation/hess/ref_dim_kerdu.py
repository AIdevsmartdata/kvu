"""Script INDÉPENDANT de HESS-KERDU (iii) : dimension de X = ker DŪ ∩ (jauge résiduelle)^⊥ par SVD dense numpy de A = [J ; générateurs de la jauge résiduelle]
(J et générateurs : kerdu_ref.py, aucun code de kvu). dim X = 12 L^4 − rang(A), rang = nombre de σ > 1e-10 σ_max (marge consignée).
Usage : ref_dim_kerdu.py SORTIE.json  (fonds : froid et Haar L = 6 graine 26102613071)."""
import os, sys, json, numpy as np
d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, d)
import outils as O, kerdu_ref as R
n = R.n

def traiter(C):
    J, G = R.construire(C)
    A = np.vstack([J, G])
    s = np.linalg.svd(A, compute_uv=False)
    smax = s[0]
    rang = int(np.sum(s > 1e-10 * smax))
    marge = [float(s[rang - 1] / smax), float(s[rang] / smax) if rang < len(s) else 0.0]
    return {'dim': int(n - rang), 'rang': rang, 'lignes': int(A.shape[0]), 'rang_J': int(np.sum(np.linalg.svd(J, compute_uv=False) > 1e-10)),
            'sigma_rel_derniere_gardee_et_premiere_ecartee': marge, 'norme_JGt': float(np.abs(J @ G.T).max())}

L = R.L
froid = np.zeros((4, L, L, L, L, 2, 2), complex); froid[..., 0, 0] = 1; froid[..., 1, 1] = 1
res = {'froid': traiter(froid), 'haar': traiter(O.haar_c16(L, 26102613071))}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(res))
