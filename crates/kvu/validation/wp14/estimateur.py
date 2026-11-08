"""WP14 : estimateur de l'exposant α (ε′ ∝ g^α), intervalle bootstrap, règle de lecture, combinaison des espaces.

Module SANS entrée-sortie (importé par verdict.py et par l'auto-test) : aucune valeur de physique n'y est écrite.
Toutes les constantes ci-dessous sont celles du PREREG (WP14_PREREG.md §4) ; elles sont affichées par `verdict.py autotest`.

Définitions (gelées) :
  g(β) = √(4/β) ;
  m(β) = médiane, sur les fonds de β, de ε′ ;
  α = pente des moindres carrés ordinaires (poids égaux) de ln m(β) contre ln g(β) sur β ∈ BETAS_AJUSTEMENT ;
  IC 95 % = percentiles 2,5 et 97,5 de α sur B_BOOT rééchantillonnages bootstrap des fonds, STRATIFIÉS par β (n fonds tirés avec remise
  dans chaque β), générateur numpy.random.default_rng([GRAINE_BOOT, indice_espace]).
Règle de lecture d'un espace (IC = [lo, hi], largeur = hi − lo) :
  INDETERMINE   si largeur > LARGEUR_MAX, ou si une médiane n'est pas strictement positive, ou si le nombre de fonds n'est pas N_FONDS ;
  DIFFERENT-    si hi < BORNE_DIFF_BAS            (α significativement plus petit que S2) ;
  DIFFERENT+    si lo > BORNE_DIFF_HAUT           (α significativement plus grand que S2) ;
  COMME-S2      si lo ≥ BORNE_COMME_BAS et hi ≤ BORNE_COMME_HAUT ;
  INDETERMINE   sinon.
"""
import math

import numpy as np

BETAS_AJUSTEMENT = (4.0, 8.0, 16.0)
N_FONDS = 8
B_BOOT = 2000
GRAINE_BOOT = 2611011401
LARGEUR_MAX = 0.4
BORNE_COMME_BAS, BORNE_COMME_HAUT = 0.9, 1.1
BORNE_DIFF_BAS, BORNE_DIFF_HAUT = 0.8, 1.2
INDICE_ESPACE = {"tcov": 0, "kerdu": 1, "kerB": 2}   # indices des flux aléatoires (ne dépendent pas de l'ordre d'exécution)


def g_de_beta(beta):
    return math.sqrt(4.0 / beta)


def pente_ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def alpha_depuis_medianes(medianes, betas=BETAS_AJUSTEMENT):
    """medianes : dict β -> m(β) > 0 ; rend α (OLS de ln m contre ln g)."""
    x = [math.log(g_de_beta(b)) for b in betas]
    y = [math.log(medianes[b]) for b in betas]
    return pente_ols(x, y)


def alpha_bootstrap(valeurs, indice_espace, betas=BETAS_AJUSTEMENT, b_boot=B_BOOT, graine=GRAINE_BOOT):
    """valeurs : dict β -> tableau de ε′ (un par fond). Rend (α, lo, hi, medianes, nb_boot_valides)."""
    arr = [np.asarray(valeurs[b], float) for b in betas]
    med = {b: float(np.median(a)) for b, a in zip(betas, arr)}
    alpha = alpha_depuis_medianes(med, betas) if all(m > 0 for m in med.values()) else float("nan")
    x = np.array([math.log(g_de_beta(b)) for b in betas])
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    rng = np.random.default_rng([graine, indice_espace])
    pentes = np.full(b_boot, np.nan)
    for i in range(b_boot):
        y = np.empty(len(betas))
        for j, a in enumerate(arr):
            m = float(np.median(a[rng.integers(0, len(a), size=len(a))]))
            y[j] = math.log(m) if m > 0 else float("nan")
        if np.all(np.isfinite(y)):
            pentes[i] = float(np.dot(xc, y)) / den   # Σ xc = 0 : Σ xc (y − ȳ) = Σ xc y
    bons = pentes[np.isfinite(pentes)]
    ok = int(bons.size)
    if ok < 0.99 * b_boot:   # trop de rééchantillons à médiane ≤ 0 : l'IC n'est pas défini
        return alpha, float("nan"), float("nan"), med, ok
    lo, hi = np.percentile(bons, [2.5, 97.5])
    return alpha, float(lo), float(hi), med, ok


def lecture_espace(alpha, lo, hi, nb_fonds_par_beta, medianes):
    """Rend (étiquette, motif) selon la règle gelée."""
    if any(n != N_FONDS for n in nb_fonds_par_beta):
        return "INDETERMINE", f"nombre de fonds ≠ {N_FONDS} : {list(nb_fonds_par_beta)}"
    if not all(math.isfinite(m) and m > 0 for m in medianes.values()):
        return "INDETERMINE", "médiane non strictement positive"
    if not (math.isfinite(alpha) and math.isfinite(lo) and math.isfinite(hi)):
        return "INDETERMINE", "α ou IC non défini"
    if hi - lo > LARGEUR_MAX:
        return "INDETERMINE", f"IC trop large ({hi - lo:.3f} > {LARGEUR_MAX})"
    if hi < BORNE_DIFF_BAS:
        return "DIFFERENT-", f"IC [{lo:.3f} ; {hi:.3f}] sous {BORNE_DIFF_BAS}"
    if lo > BORNE_DIFF_HAUT:
        return "DIFFERENT+", f"IC [{lo:.3f} ; {hi:.3f}] au-dessus de {BORNE_DIFF_HAUT}"
    if lo >= BORNE_COMME_BAS and hi <= BORNE_COMME_HAUT:
        return "COMME-S2", f"IC [{lo:.3f} ; {hi:.3f}] dans [{BORNE_COMME_BAS} ; {BORNE_COMME_HAUT}]"
    return "INDETERMINE", f"IC [{lo:.3f} ; {hi:.3f}] ni dans [{BORNE_COMME_BAS} ; {BORNE_COMME_HAUT}] ni hors de [{BORNE_DIFF_BAS} ; {BORNE_DIFF_HAUT}]"


def combiner(etiquettes, controle_ok, instrument_ok):
    """etiquettes : dict espace -> étiquette, pour les espaces DÉCISIFS. Rend (verdict, motif).
    Contrôle positif ou gates d'instrument non verts => INDETERMINE. Un espace DIFFERENT± suffit pour DIFFERENT ; COMME-S2 exige tous les espaces COMME-S2."""
    if not instrument_ok:
        return "INDETERMINE", "gate d'instrument non verte"
    if not controle_ok:
        return "INDETERMINE", "contrôle positif HESS-S2 non vert"
    vals = list(etiquettes.values())
    if not vals:
        return "INDETERMINE", "aucun espace décisif"
    diff = {e: v for e, v in etiquettes.items() if v.startswith("DIFFERENT")}
    if diff:
        sens = {v[-1] for v in diff.values()}
        return "DIFFERENT", "espaces " + ", ".join(f"{e}:{v}" for e, v in sorted(diff.items())) + ("" if len(sens) == 1 else " (sens mixte)")
    if all(v == "COMME-S2" for v in vals):
        return "COMME-S2", "tous les espaces décisifs : COMME-S2"
    return "INDETERMINE", "espaces : " + ", ".join(f"{e}:{v}" for e, v in sorted(etiquettes.items()))
