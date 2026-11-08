#!/usr/bin/env python3
"""WP12a, re-verdict « échecs de Newton : coût ou biais ? » (PREREG_NEWTON_2026-11.md). Le statut est CALCULÉ ici à partir du JSON
produit par le test `gates_newton::controle_newton_courbe` (+ facultativement celui du rejeu SU(2)). Vérité et prédictions recalculées
ici, indépendamment du code Rust : quadrature de CT-7 (poids exp(-cos x/2 - a² sin² x/2), a = 1,2), biais prédit des estimateurs
« sans échecs » et « réessai » : -Cov_pi(O, f)/(1 - E_pi f) à partir du profil d'échec mesuré.
Usage : python3 -I -B reverdict_newton.py --courbe FICHIER.json [--su2 FICHIER.json] --sortie DIR
"""
import argparse, json, math, os, sys
import numpy as np

S_T = 3.88          # seuil empirique m = 16 de WP12a (stat.rs SEUILS.moy_m16, avenant WP12a-2), repris tel quel
A = 1.2

def exact():
    x = 2 * np.pi * np.arange(200000) / 200000
    w = np.exp(-0.5 * np.cos(x) - 0.5 * A * A * np.sin(x) ** 2)
    return float((w * np.cos(x)).sum() / w.sum()), float((w * np.cos(2 * x)).sum() / w.sum())

def prediction(prof):
    x = np.array(prof["x"]); f = np.array(prof["echecs"]) / prof["essais"]
    var_f = f * (1 - f) / prof["essais"]
    w = np.exp(-0.5 * np.cos(x) - 0.5 * A * A * np.sin(x) ** 2); w /= w.sum()
    ef = float((w * f).sum()); out = []; err = []
    for O in (np.cos(x), np.cos(2 * x)):
        eo = float((w * O).sum())
        cov = float((w * (O - eo) * (f - ef)).sum())
        b = -cov / (1 - ef)
        out.append(b)
        # erreur MC du profil, dérivée COMPLÈTE : db/df_k = -w_k [(O_k - eo) - b] / (1 - ef)   (numérateur ET dénominateur)
        err.append(math.sqrt(float(((w * ((O - eo) - b) / (1 - ef)) ** 2 * var_f).sum())))
    return out, err, ef

CELLULES_GELEES = [(1.0, 0.15), (1.0, 0.5), (1.0, 0.8), (1.0, 1.0), (4.0, 0.15), (4.0, 0.3), (4.0, 0.4), (4.0, 0.5)]

def _sans_doublon(paires):
    d = {}
    for k, v in paires:
        if k in d:
            raise ValueError(f"clé JSON dupliquée : {k!r}")
        d[k] = v
    return d

def _constante_refusee(c):
    raise ValueError(f"constante JSON non finie refusée : {c}")

def _flottant_fini(texte):
    x = float(texte)
    if not math.isfinite(x):
        raise ValueError(f"flottant JSON non fini refusé : {texte}")
    return x

def charger_json(f):
    """json.load STRICT : refuse les clés dupliquées (Python garde sinon la dernière et masque la première valeur), NaN/Infinity et les flottants qui débordent (1e999).
    Rust écrit NaN en `null` : les entrées honnêtes passent."""
    return json.load(f, object_pairs_hook=_sans_doublon, parse_constant=_constante_refusee, parse_float=_flottant_fini)

def entier(x):
    return type(x) is int

def reel(x):
    return type(x) in (int, float) and math.isfinite(x)

CLES_TOP = {"schema", "hors_verdict", "cellules"}
CLES_TOP_RUN2 = CLES_TOP | {"graine_base"}
# Protocoles gelés : run 1 (PREREG_NEWTON_2026-11 : « zéro exception ») et run 2 (PREREG_NEWTON_RUN2_2026-11 : taux borné, graines neuves)
PROTOCOLES = {26_110_160_000: {"ninv": 200_000, "gate": "zero"}, 26_110_170_000: {"ninv": 10_000_000, "gate": "taux"}}
FACTEUR_SECURITE = 32   # ε_max(observable) = σ_stat / (32 τ) / R_π : déplacement de la moyenne ≤ σ_stat/8 avec osc(g) ≲ 2τ·osc(O) = 4τ (PREREG_NEWTON_RUN2 §3)
ALPHA_FAMILLE = 0.05    # niveau de la famille des 8 cellules (Bonferroni : ALPHA_FAMILLE/8 par cellule, unilatéral)
ALPHA_TAU = 0.05        # niveau de la famille des 16 estimations de τ_int de production (Bonferroni : ALPHA_TAU/16 chacune, unilatéral)

def rapport_pi():
    """R = max_x π(x)/U(x) (U uniforme sur [−π, π)) : les départs de l'invariance sont tirés UNIFORMÉMENT alors que le biais relève du taux sous π ; ε_π ≤ R·ε_U.
    π(x) ∝ exp(−cos x/2 − 0,72 sin² x) (CT-7 ; indépendant de la masse M2). Quadrature à 200 000 points."""
    x = 2 * math.pi * np.arange(200000) / 200000
    w = np.exp(-0.5 * np.cos(x) - 0.5 * A * A * np.sin(x) ** 2)
    return float(w.max() / w.mean())

def tau_majore(tau, n):
    """Borne supérieure de confiance sur un τ_int de Sokal : τ̂·(1 + z·√(2(2W+1)/n)) avec z = Φ⁻¹(1 − ALPHA_TAU/16) et W ≤ 6τ̂ + 6 (règle d'arrêt w ≥ 6τ(w) de
    stat.rs, au plus 6 de dépassement) : δτ/τ = √(2(2W+1)/n) est la formule de Sokal enregistrée par stat_serie (champ dtau). n = nombre de valeurs retenues."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - ALPHA_TAU / 16)
    return tau * (1 + z * math.sqrt(2 * (2 * (6 * tau + 6) + 1) / n))
CLES_CELLULE = {"cellule", "dt", "graines", "invariance", "m2", "max_newton", "n", "nb_pas", "production", "profil", "reessai", "sans_echecs",
                "taux_acceptation", "taux_echec", "tol_newton", "tol_rev"}
CLES_RESUME = {"obs", "n", "moyenne", "erreur", "tau_int", "bloc", "statut", "domaine_campagne"}
CLES_INVARIANCE = {"n", "aller_echec", "aller_ok", "retour_echec", "retour_sans_retour_au_depart", "pire_ecart"}
CLES_PROFIL = {"x", "essais", "echecs"}
CLES_SU2 = {"acceptees", "attendus_echecs", "echecs", "fenetres", "identique_n2bis", "refus_metropolis", "resolus_cap500", "resolus_cap500_acceptes", "resume"}
STATUTS = {"OK", "PUISSANCE", "NON-STATIONNAIRE", "TAU-NON-ESTIMABLE"}
N_TRAJ = 2_000_000
K_BLOCS = 40

def n_apres_thermalisation(longueur):
    """`stat::analyser` écarte floor(0,2·longueur) valeurs en tête."""
    return longueur - int(longueur * 0.2)

def resume_ok(r, k, n_attendu, ic, est, ec):
    """Un résumé de série (champs enregistrés par gates_newton.rs::resume) : clés exactes, types, cohérence interne et avec le protocole."""
    if not (isinstance(r, dict) and set(r) == CLES_RESUME):
        ec.append(f"clés du résumé {est} de la cellule {ic}"); return
    err = lambda m: ec.append(f"{m} du résumé {est} (obs {k}) de la cellule {ic}")
    if not (entier(r["obs"]) and r["obs"] == k): err("obs")
    if not (entier(r["n"]) and r["n"] == n_attendu): err("n")
    if r["statut"] not in STATUTS: err("statut"); return
    if type(r["domaine_campagne"]) is not bool: err("domaine_campagne"); return
    ok = r["statut"] == "OK"
    for nom in ("moyenne", "erreur", "tau_int"):
        v = r[nom]
        if ok:
            if not reel(v): err(nom)
        elif not (v is None or reel(v)): err(nom)
    if ok:
        if not (abs(r["moyenne"]) <= 1 and r["erreur"] > 0 and r["tau_int"] > 0): err("bornes")
        if not (entier(r["bloc"]) and r["bloc"] == r["n"] // K_BLOCS): err("bloc")
        elif r["domaine_campagne"] != (r["bloc"] >= 40 * r["tau_int"]): err("domaine_campagne recalculé")
    else:
        if r["domaine_campagne"]: err("domaine_campagne d'un statut non OK")
        # formes écrites par stat.rs::analyser_avec : NON-STATIONNAIRE et TAU-NON-ESTIMABLE « tardif » (demi-séries) conservent les statistiques et le bloc ;
        # PUISSANCE et TAU-NON-ESTIMABLE « initial » (τ de la série entière non estimable) rendent une analyse vide (bloc 0, erreur et τ null)
        if r["statut"] == "NON-STATIONNAIRE" or (r["statut"] == "TAU-NON-ESTIMABLE" and r["tau_int"] is not None):
            if not (reel(r["moyenne"]) and reel(r["erreur"]) and reel(r["tau_int"]) and r["erreur"] > 0 and r["tau_int"] > 0): err("statistiques d'une analyse conservée")
            attendu_bloc = r["n"] // K_BLOCS
        else:
            attendu_bloc = 0
        if not (entier(r["bloc"]) and r["bloc"] == attendu_bloc): err("bloc")

def protocole_ok(d):
    """Le fichier est-il une exécution de VERDICT du protocole gelé (PREREG §2) ? Renvoie la liste des écarts. Clés exactes à chaque niveau, types stricts
    (booléens refusés), cohérence interne des champs enregistrés par gates_newton.rs (n, bloc, taux, invariance)."""
    ec = []
    if not isinstance(d, dict) or set(d) not in (CLES_TOP, CLES_TOP_RUN2): return ["clés du fichier"]
    base = d.get("graine_base", 26_110_160_000)            # absent = run 1 (fichiers gelés)
    if not (entier(base) and base in PROTOCOLES): return ["graine_base inconnue"]
    if d["schema"] != "wp12a-newton-courbe/1": ec.append("schéma")
    if d["hors_verdict"] is not False: ec.append("exécution marquée hors verdict ou champ absent")
    cs = d["cellules"]
    if not isinstance(cs, list) or len(cs) != len(CELLULES_GELEES): return sorted(set(ec + ["liste des cellules"]))
    grille = [-math.pi + (k + 0.5) * 2 * math.pi / 400 for k in range(400)]
    for ic, (c, (m2, dt)) in enumerate(zip(cs, CELLULES_GELEES)):
        try:
            if not (isinstance(c, dict) and set(c) == CLES_CELLULE): ec.append(f"clés de la cellule {ic}"); continue
            if not (entier(c["cellule"]) and c["cellule"] == ic): ec.append(f"identifiant de cellule {ic}")
            if not (reel(c["m2"]) and reel(c["dt"]) and c["m2"] == m2 and c["dt"] == dt): ec.append(f"paramètres (M2, dt) de la cellule {ic}")
            if not (entier(c["n"]) and c["n"] == N_TRAJ): ec.append(f"n de la cellule {ic}")
            if not (entier(c["nb_pas"]) and c["nb_pas"] == 6 and entier(c["max_newton"]) and c["max_newton"] == 50): ec.append(f"nb_pas / max_newton de la cellule {ic}")
            if not (reel(c["tol_newton"]) and c["tol_newton"] == 1e-13 and reel(c["tol_rev"]) and c["tol_rev"] == 1e-9): ec.append(f"tolérances de la cellule {ic}")
            g = base + 10 * ic
            if not (isinstance(c["graines"], list) and c["graines"] == [g + 1, g + 2, g + 3, g + 4] and all(entier(x) for x in c["graines"])): ec.append(f"graines de la cellule {ic}")
            # taux : entiers de comptage, somme ≤ 1
            nf = na = None
            if reel(c["taux_echec"]) and reel(c["taux_acceptation"]) and 0 <= c["taux_echec"] <= 1 and 0 <= c["taux_acceptation"] <= 1:
                nf, na = c["taux_echec"] * N_TRAJ, c["taux_acceptation"] * N_TRAJ
                if abs(nf - round(nf)) > 1e-6 or abs(na - round(na)) > 1e-6 or round(nf) + round(na) > N_TRAJ: ec.append(f"taux de la cellule {ic}"); nf = None
                else: nf = round(nf)
            else: ec.append(f"taux de la cellule {ic}")
            # résumés
            for est in ("production", "sans_echecs", "reessai"):
                r = c[est]
                if not (isinstance(r, list) and len(r) == 2): ec.append(f"résumé {est} de la cellule {ic}"); continue
                if est == "sans_echecs":
                    if nf is None: continue
                    n_att = n_apres_thermalisation(N_TRAJ - nf)
                else: n_att = n_apres_thermalisation(N_TRAJ)
                for k in (0, 1): resume_ok(r[k], k, n_att, ic, est, ec)
            # invariance
            iv = c["invariance"]
            if not (isinstance(iv, dict) and set(iv) == CLES_INVARIANCE): ec.append(f"clés de l'invariance de la cellule {ic}")
            else:
                cn = ("n", "aller_echec", "aller_ok", "retour_echec", "retour_sans_retour_au_depart")
                if not (all(entier(iv[k]) and iv[k] >= 0 for k in cn) and iv["n"] == PROTOCOLES[base]["ninv"] and iv["aller_echec"] + iv["aller_ok"] == iv["n"]
                        and iv["retour_echec"] + iv["retour_sans_retour_au_depart"] <= iv["aller_ok"]): ec.append(f"invariance de la cellule {ic}")
                elif not (reel(iv["pire_ecart"]) and iv["pire_ecart"] >= 0 and (iv["retour_sans_retour_au_depart"] > 0 or iv["pire_ecart"] < 1e-7)): ec.append(f"pire_ecart de la cellule {ic}")
            # profil
            pr = c["profil"]
            if not (isinstance(pr, dict) and set(pr) == CLES_PROFIL): ec.append(f"clés du profil de la cellule {ic}")
            else:
                xs, nfp = pr["x"], pr["echecs"]
                if not (entier(pr["essais"]) and pr["essais"] == 20_000 and isinstance(xs, list) and isinstance(nfp, list) and len(xs) == 400 == len(nfp)
                        and all(reel(x) and abs(x - g0) < 1e-12 for x, g0 in zip(xs, grille)) and all(entier(k) and 0 <= k <= 20_000 for k in nfp)): ec.append(f"profil d'échec de la cellule {ic}")
        except Exception as e:
            ec.append(f"cellule {ic} mal formée : {type(e).__name__}")
    return sorted(set(ec))

def su2_ok(s):
    """Schéma et cohérence du rejeu SU(2) (gates_newton.rs::rejeu_su2_n2bis) ; renvoie la liste des écarts."""
    if not (isinstance(s, dict) and set(s) == CLES_SU2): return ["clés du rejeu SU(2)"]
    ec = []
    ints = ("acceptees", "attendus_echecs", "echecs", "refus_metropolis", "resolus_cap500", "resolus_cap500_acceptes")
    if not all(entier(s[k]) and s[k] >= 0 for k in ints): return ["types du rejeu SU(2)"]
    if s["attendus_echecs"] != 29: ec.append("attendus_echecs")
    if s["acceptees"] + s["refus_metropolis"] + s["echecs"] != 1000: ec.append("somme des trajectoires")
    f = s["fenetres"]
    if not (isinstance(f, list) and len(f) == 10 and all(entier(x) and x >= 0 for x in f) and sum(f) == s["echecs"]): ec.append("fenêtres")
    if not (s["resolus_cap500"] <= s["echecs"] and s["resolus_cap500_acceptes"] <= s["resolus_cap500"]): ec.append("résolus")
    if not (type(s["identique_n2bis"]) is bool and s["identique_n2bis"] == (s["echecs"] == 29 and s["acceptees"] == 939)): ec.append("identique_n2bis")
    # le texte de résumé écrit par Rust doit redire les mêmes comptes (aucun champ libre non recoupé)
    import re
    r = s["resume"]
    attendu = [rf"acceptées {s['acceptees']},", rf"refus Metropolis {s['refus_metropolis']},", rf"échecs de calcul {s['echecs']} ",
               rf"par fenêtre de 100 {re.escape(str(s['fenetres']))} ;", rf"{s['resolus_cap500']} sur {s['echecs']} aboutissent \({s['resolus_cap500_acceptes']} acceptées\)"]
    if not (isinstance(r, str) and r.startswith("1000 trajectoires : ") and all(re.search(m, r) for m in attendu)): ec.append("résumé incohérent avec les comptes")
    return ec

def fnum(x, spec):
    """Formate un nombre ; les null légitimes (NaN → null dans le JSON de Rust) sont affichés tels quels, sans arithmétique."""
    return "null" if x is None else format(x, spec)

def calculer(d, s_doc):
    """Rend (lignes, res, statut). Un fichier hors protocole ou mal formé donne HORS_PROTOCOLE sans traiter les nombres."""
    ex = exact()
    L = []; res = {"cellules": []}
    try:
        ecarts_protocole = protocole_ok(d)
    except Exception as e:
        ecarts_protocole = [f"fichier mal formé : {type(e).__name__}"]
    if ecarts_protocole:
        L.append(f"FAIL NEWTON-PROTOCOLE exécution non conforme au protocole gelé : {ecarts_protocole}")
        L.append("STATUT NEWTON-REVERDICT HORS_PROTOCOLE")
        res["statut"] = "HORS_PROTOCOLE"
        return L, res, "HORS_PROTOCOLE"
    L.append(f"PASS NEWTON-PROTOCOLE exécution de verdict conforme (8 cellules, graines, tolérances, n = 2 000 000, invariance {format(PROTOCOLES[d.get('graine_base', 26_110_160_000)]['ninv'], ',').replace(',', ' ')}, profil 400 × 20 000)")
    valide_prod, tprod, tmut, tpred, nonvalide = [], [], [], [], 0
    inv_tot = dict(aller_ok=0, retour_echec=0, retour_faux=0, min_ok=10**9)
    for c in d["cellules"]:
        pred, perr, ef = prediction(c["profil"])
        cell = {"m2": c["m2"], "dt": c["dt"], "taux_echec_chaine": c["taux_echec"], "taux_echec_profil_pi": ef, "acceptation": c["taux_acceptation"], "pred_biais": pred}
        for k in (0, 1):
            for est in ("production", "sans_echecs", "reessai"):
                r = c[est][k]
                ok = r["statut"] == "OK" and r["domaine_campagne"]
                t = (r["moyenne"] - ex[k]) / r["erreur"] if ok else float("nan")
                tp = (r["moyenne"] - (ex[k] + pred[k])) / math.hypot(r["erreur"], perr[k]) if ok else float("nan")
                cell[f"{est}_obs{k}"] = dict(moyenne=r["moyenne"], erreur=r["erreur"], tau=r["tau_int"], valide=ok, t_exact=t, t_pred=tp,
                                             ecart=(r["moyenne"] - ex[k]) if r["moyenne"] is not None else None)
                if est == "production":
                    if ok: tprod.append(abs(t))
                    else: nonvalide += 1
                    L.append(f"INFO NEWTON-PROD M2={c['m2']} dt={c['dt']} obs{k} : {fnum(r['moyenne'], '.6f')} ± {fnum(r['erreur'], '.1e')} (exact {ex[k]:.6f}) t = {t:+.2f} ; échec {c['taux_echec']:.3f} ; τ = {fnum(r['tau_int'], '.2f')} ; statut {r['statut']}")
                else:
                    if ok:
                        tmut.append((est, c["taux_echec"], abs(t)))
                        if c["taux_echec"] >= 0.04: tpred.append(abs(tp))
                    L.append(f"INFO NEWTON-{est.upper()} M2={c['m2']} dt={c['dt']} obs{k} : biais mesuré {fnum(None if r['moyenne'] is None else r['moyenne'] - ex[k], '+.2e')} ± {fnum(r['erreur'], '.1e')} ; biais prédit {pred[k]:+.2e} ; t_exact {t:+.1f} ; t_pred {tp:+.2f}")
        iv = c["invariance"]
        inv_tot["aller_ok"] += iv["aller_ok"]; inv_tot["retour_echec"] += iv["retour_echec"]; inv_tot["retour_faux"] += iv["retour_sans_retour_au_depart"]; inv_tot["min_ok"] = min(inv_tot["min_ok"], iv["aller_ok"])
        L.append(f"INFO NEWTON-INVARIANCE M2={c['m2']} dt={c['dt']} : {iv['aller_ok']} allers réussis sur {iv['n']}, retours en échec {iv['retour_echec']}, retours ne revenant pas au départ {iv['retour_sans_retour_au_depart']}, pire écart {iv['pire_ecart']:.1e}")
        res["cellules"].append(cell)
    # gates
    n_cmp = 16
    if nonvalide == 0 and len(tprod) == n_cmp:
        mx = max(tprod); L.append(f"{'PASS' if mx < S_T else 'FAIL'} NEWTON-ABSENCE-DE-BIAIS production (échec = rejet compté) : max|t| = {mx:.2f} < {S_T} sur {n_cmp} comparaisons à vérité connue (taux d'échec de {min(c['taux_echec'] for c in d['cellules']):.3f} à {max(c['taux_echec'] for c in d['cellules']):.3f})")
        g_prod = "PASS" if mx < S_T else "FAIL"
    else:
        L.append(f"SKIP NEWTON-ABSENCE-DE-BIAIS PUISSANCE : {nonvalide} comparaisons hors domaine, {len(tprod)} valides sur {n_cmp}"); g_prod = "SKIP"
    attendues = sum(2 * 2 for c in d["cellules"] if c["taux_echec"] >= 0.04)   # 2 mutants × 2 observables par cellule à échec ≥ 4 %
    mut_fortes = [t for e, f, t in tmut if f >= 0.04]
    complet = len(mut_fortes) == attendues and len(tpred) == attendues and attendues > 0
    if not mut_fortes:
        g_mut = "SKIP"; L.append(f"SKIP NEWTON-PUISSANCE PUISSANCE : aucune comparaison de mutant valide ({attendues} attendues)")
    else:
        g_mut = "PASS" if max(mut_fortes) >= S_T else "FAIL"
        L.append(f"{g_mut} NEWTON-PUISSANCE mutants « sans échecs » et « réessai » : max|t| = {max(mut_fortes):.1f} ≥ {S_T} ; {len(mut_fortes)} comparaisons valides sur {attendues} attendues (taux d'échec ≥ 4 %, ROUGE exigé)")
    if not complet:
        g_pred = "SKIP"; L.append(f"SKIP NEWTON-PREDICTION PUISSANCE : {len(tpred)} comparaisons valides sur {attendues} attendues")
    else:
        g_pred = "PASS" if max(tpred) < S_T else "FAIL"
        L.append(f"{g_pred} NEWTON-PREDICTION biais mesuré des mutants = −Cov(O,f)/(1−f̄) du profil mesuré : max|t_pred| = {max(tpred):.2f} < {S_T} sur {len(tpred)} comparaisons (toutes les {attendues} attendues)")
    proto = PROTOCOLES[d.get("graine_base", 26_110_160_000)]
    if proto["gate"] == "zero":
        g_inv = "PASS" if inv_tot["retour_echec"] == 0 and inv_tot["retour_faux"] == 0 and inv_tot["min_ok"] >= 10**4 else "FAIL"
        L.append(f"{g_inv} NEWTON-INVARIANCE aller réussi ⇒ retour réussi et revenant au départ : {inv_tot['aller_ok']} allers réussis, {inv_tot['retour_echec']} retours en échec, {inv_tot['retour_faux']} sans retour (≥ 10⁴ allers par cellule exigés)")
    else:
        from scipy.stats import beta as loi_beta
        etats_cell, lignes_cell = [], []
        alpha = ALPHA_FAMILLE / len(d["cellules"])
        for ic, c in enumerate(d["cellules"]):
            iv = c["invariance"]; s_c = iv["aller_ok"]; k_c = iv["retour_echec"] + iv["retour_sans_retour_au_depart"]
            if not all(c["production"][k]["statut"] == "OK" and c["production"][k]["domaine_campagne"] for k in (0, 1)):
                etats_cell.append("SKIP"); lignes_cell.append(f"INFO NEWTON-INVARIANCE-TAUX cellule {ic} : série de production hors domaine : non évaluable"); continue
            taus = [tau_majore(c["production"][k]["tau_int"], c["production"][k]["n"]) for k in (0, 1)]
            # seuil par observable : ε_max = σ_stat/(32 τ_ub)/R_π, σ_stat = erreur jackknife de la production (mesurée), τ_ub = borne de confiance de son τ_int ; minimum des deux observables
            eps_max = min(c["production"][k]["erreur"] / (FACTEUR_SECURITE * taus[k]) for k in (0, 1)) / rapport_pi()
            tau_c = max(taus)
            ub = 1.0 if k_c >= s_c else float(loi_beta.ppf(1 - alpha, k_c + 1, s_c - k_c))
            ub0 = float(loi_beta.ppf(1 - alpha, 1, s_c))        # borne qu'on aurait avec zéro exception : la cellule peut-elle certifier ε_max ?
            etat = "SKIP" if ub0 > eps_max else ("PASS" if ub <= eps_max else "FAIL")
            etats_cell.append(etat)
            lignes_cell.append(f"INFO NEWTON-INVARIANCE-TAUX cellule {ic} (M2={c['m2']}, dt={c['dt']}) : {k_c} exceptions sur {s_c} allers réussis (ε̂ = {k_c / s_c:.2e}), borne de confiance sur τ_int = {tau_c:.2f}, ε_max = {eps_max:.2e} (R_π = {rapport_pi():.3f}), borne de Clopper–Pearson (unilatérale, {100 * (1 - alpha):.3f} %) = {ub:.2e} : {etat}" + (" (puissance insuffisante même à zéro exception)" if etat == "SKIP" else ""))
        L += lignes_cell
        g_inv = "FAIL" if "FAIL" in etats_cell else ("SKIP" if "SKIP" in etats_cell else "PASS")
        L.append(f"{g_inv} NEWTON-INVARIANCE-TAUX par cellule : {etats_cell.count('PASS')} PASS, {etats_cell.count('FAIL')} FAIL, {etats_cell.count('SKIP')} SKIP sur {len(etats_cell)} ; ε_max(cellule) = min sur les 2 observables de σ_stat/({FACTEUR_SECURITE}·τ_ub)/R_π (σ_stat = erreur jackknife de la production, τ_ub = borne supérieure de confiance Bonferroni sur 16 du τ_int, R_π = max π/uniforme)")
    if s_doc is not None and (ec_su2 := su2_ok(s_doc)):
        L.append(f"FAIL NEWTON-PROTOCOLE rejeu SU(2) non conforme : {ec_su2}")
        L.append("STATUT NEWTON-REVERDICT HORS_PROTOCOLE")
        res["statut"] = "HORS_PROTOCOLE"
        return L, res, "HORS_PROTOCOLE"
    if s_doc is not None:
        s = s_doc; res["su2"] = s
        L.append(f"INFO NEWTON-SU2-REJEU {s.get('resume')}")
        # recalculé ici : 29 échecs et 939 acceptées (reprise n2bis, journal verdicts_n2bis/su2.log)
        g_su2 = "PASS" if (entier(s.get("echecs")) and s["echecs"] == 29 and entier(s.get("acceptees")) and s["acceptees"] == 939) else "FAIL"
        L.append(f"{g_su2} NEWTON-SU2-REJEU reproduction de la reprise n2bis : {s.get('echecs')} échecs (29 attendus), {s.get('acceptees')} acceptées (939 attendues)")
    else:
        g_su2 = "SKIP"
        L.append("SKIP NEWTON-SU2-REJEU fichier du rejeu SU(2) absent (--su2 obligatoire pour un verdict)")
    if g_prod == "FAIL": statut = "BIAIS_DETECTE"
    elif g_prod == "PASS" and g_mut == "PASS" and g_pred == "PASS" and g_inv == "PASS" and g_su2 == "PASS": statut = "COUT_SANS_BIAIS"
    else: statut = "INDETERMINE"
    L.append(f"STATUT NEWTON-REVERDICT {statut}")
    L.append("PORTEE : « coût sans biais détecté sur CT-7 périodique, pour les échecs de Newton et de réversibilité couverts » ; ne modifie pas le SKIP formel FIB-CONTR PUISSANCE (seuil 1 % inchangé) ; échecs d'évaluation (branche du logarithme) non couverts ; le jouet 1D ne démontre rien sur SU(2).")
    res["statut"] = statut; res["exact"] = ex
    return L, res, statut

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--courbe", required=True); ap.add_argument("--su2"); ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    if os.path.exists(a.sortie): sys.exit("sortie existante refusée")
    os.makedirs(a.sortie)
    try:
        with open(a.courbe) as f:
            d = charger_json(f)
        s_doc = None
        if a.su2:
            with open(a.su2) as f:
                s_doc = charger_json(f)
    except Exception as e:
        d, s_doc = {}, None
    L, res, statut = calculer(d, s_doc)
    open(f"{a.sortie}/verdicts.txt", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{a.sortie}/analyse.json", "w"), indent=1)
    print("\n".join(L))
    sys.exit(0 if statut == "COUT_SANS_BIAIS" else 1)

if __name__ == "__main__":
    main()
