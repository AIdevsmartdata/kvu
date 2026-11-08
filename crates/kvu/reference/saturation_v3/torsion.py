#!/usr/bin/env python3
"""Saturation v3 — famille TORDUE (lecture secondaire (ii)), sans toucher à noyau2 (cb.py importé, sha vérifié par l'appelant).

Objet [dérivation, écrite avant toute donnée] :
  Prop. O / THEORIE_CIBLE_A §3.5 (1) : Cov(F, G) = E⟨∇F, 𝒟 ∇G⟩ (gradient de FIBRE).  Le twist agit sur l'INDICE
  (la coordonnée de fibre, repérée par son site de base) : ‖e^{ψ} 𝒟 e^{−ψ}‖.  Pour toutes F, G :
      E⟨e^{−ψ}∇F, (e^{ψ}𝒟e^{−ψ}) e^{ψ}∇G⟩ = Cov(F, G)   ⟹
      Λ_ψ(𝒜) := sup_{F,G ∈ 𝒜} |Cov(F,G)| / ( ‖e^{−ψ}∇F‖ ‖e^{ψ}∇G‖ )  ⩽  ‖e^{ψ}𝒟e^{−ψ}‖     (MINORANT, jamais majorant).
  Sur une famille linéaire 𝒜 = span(F_1..F_n) : Λ_ψ = σ_max( Dm^{-1/2} C Dp^{-1/2} ), C = Cov(F), Dm = E[G_ψ⁻ᵀ G_ψ⁻],
  Dp = E[G_ψ⁺ᵀ G_ψ⁺], où G_ψ^± = lignes de gradient de fibre multipliées par e^{±ψ(site de la coordonnée)}.
  ψ = 0 ⟹ Dm = Dp = D0 et Λ_0 = λ_max(D0⁻¹C) (= l'estimateur non tordu `est` de v2).
Poids (cible C*, THEORIE_CIBLE_A §3.4 (C*-c) et §3.5 test S7) : ψ(x) = κ · dist_tore(x, x₀), κ = c/L, x₀ = 0,
  dist_tore = distance euclidienne sur le tore (Σ_i min(x_i, L − x_i)²)^{1/2}.  c = C_TW = 1 (gelé).
Site de la coordonnée de fibre : lien libre (μ, x) → x ; paire (e1 = (μ, x), e2 = (μ, x + μ̂)) → x (site de e1).
Famille 𝒜 (18 fonctionnelles) : S1 (6 lignes n = 0, = lignes 9p + 8 de fourier_weights_sparse) ; T+ (6 lignes,
  e^{+ψ(x)}/L² sur le plan p) ; T− (6 lignes, e^{−ψ(x)}/L²).  Ordre des colonnes : [S1 p=0..5, T+ p=0..5, T− p=0..5].
"""
import numpy as np
import scipy.sparse as sp

C_TW = 1.0


def psi_sites(L, c=C_TW):
    X = np.indices((L,) * 4).reshape(4, -1).T
    d = np.minimum(X, L - X).astype(float)
    return (c / L) * np.sqrt((d ** 2).sum(1))


def poids_tw(L, c=C_TW):
    """CSR (18, 6V) de la famille 𝒜 dans l'ordre documenté."""
    V = L ** 4
    psi = psi_sites(L, c)
    blocs = [np.ones(V), np.exp(psi), np.exp(-psi)]
    rows, cols, vals = [], [], []
    r = 0
    for b in blocs:
        for p in range(6):
            w = b / L ** 2
            rows.append(np.full(V, r)); cols.append(p * V + np.arange(V)); vals.append(w); r += 1
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(18, 6 * V))


def prep_tw(geo, W):
    """(ptr, idx, val, nF) par plaquette + supports par μ — cache PROPRE (ne touche pas geo._wf / geo._sup de noyau2)."""
    import cb
    WT = sp.csr_matrix(W.T); WT.sum_duplicates(); WT.sort_indices()
    prep = (WT.indptr.astype(np.int64), WT.indices.astype(np.int64), WT.data.astype(np.float64), WT.shape[1])
    return prep, cb._supports(geo, prep)


def F_tw(pv, L, psi):
    """F de la famille 𝒜 (18,) à partir de pv (6V,) : sommes pondérées par plan (numpy, ordre documenté)."""
    P = pv.reshape(6, -1)
    e = np.exp(psi); em = np.exp(-psi)
    return np.concatenate([P.sum(1), P @ e, P @ em]) / L ** 2


CHUNK = 1 << 16


def D_tw(Uf, geo, gp, prep, sup, psi):
    """(D0, Dm, Dp) (18×18 chacun) : Gram du gradient de fibre de 𝒜, non pondéré / pondéré e^{−2ψ} / e^{+2ψ}
    par coordonnée de fibre. Même remplissage que cb._D_blas (cb._k_rows), ligne 3t..3t+2 = élément t."""
    import cb
    ptr, idx, val, nF = prep
    D0 = np.zeros((nF, nF)); Dm = np.zeros((nF, nF)); Dp = np.zeros((nF, nF))
    vide = np.zeros(0, np.int64)
    for mu in range(4):
        cols = sup[mu]
        if len(cols) == 0:
            continue
        cmap = np.full(nF, -1, np.int64); cmap[cols] = np.arange(len(cols))
        sf = geo.free_all[mu]; lf = mu * geo.V + sf
        s1 = geo.p1[mu]; e1 = mu * geo.V + s1; e2 = mu * geo.V + geo.p1pm[mu]
        items = [('f', i) for i in range(0, len(lf), CHUNK)] + [('p', i) for i in range(0, len(e1), CHUNK)]
        a0 = np.zeros((len(cols), len(cols))); am = a0.copy(); ap = a0.copy()
        for kind, i in items:
            if kind == 'f':
                a = np.ascontiguousarray(lf[i:i + CHUNK]); b1 = b2 = vide; site = sf[i:i + CHUNK]
            else:
                a = vide; b1 = np.ascontiguousarray(e1[i:i + CHUNK]); b2 = np.ascontiguousarray(e2[i:i + CHUNK])
                site = s1[i:i + CHUNK]
            G = np.zeros((3 * len(site), len(cols)))
            cb._k_rows(Uf, a, b1, b2, geo.lp_plaq, geo.lp_pos, gp, ptr, idx, val, cmap, G)
            w = np.repeat(np.exp(-psi[site]), 3)[:, None]
            a0 += G.T @ G
            Gm = G * w; am += Gm.T @ Gm
            Gp = G / w; ap += Gp.T @ Gp
        ix = np.ix_(cols, cols)
        D0[ix] += a0; Dm[ix] += am; Dp[ix] += ap
    return D0, Dm, Dp


def lam_tw(C, Dm, Dp):
    """σ_max(Dm^{-1/2} C Dp^{-1/2}) via Cholesky (Dm = Lm Lmᵀ : ‖Lm⁻¹ C Lp⁻ᵀ‖₂, même valeur singulière)."""
    Lm = np.linalg.cholesky(Dm); Lp = np.linalg.cholesky(Dp)
    M = np.linalg.solve(Lm, np.linalg.solve(Lp, C.T).T)
    return float(np.linalg.svd(M, compute_uv=False)[0])
