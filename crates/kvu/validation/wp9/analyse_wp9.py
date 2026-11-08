#!/usr/bin/env python3
"""WP9 — analyse de la campagne « couplage effectif, un pas de bloc » (règles GELÉES : kvu/prereg/WP9_PREREG.md §6, GATES.md v3.5).

Entrées : un fichier de description (--cellules) qui liste les cellules (flot.json d'une cellule chacun), les recoupements entre
machines, les paires de localité et les groupes de réplicats ; les courbes d'étalonnage wp9 L = 4 et L = 8.
Sorties (dossier --sortie, refus d'écraser) : analyse.json, table_a1.md, verdicts.txt (une ligne PASS|FAIL|SKIP|INFO par contrôle).
Déterministe : graines gelées (rééchantillonnage des courbes 26102699001, Monte-Carlo sous le nul 26102699002).
Aucune règle ne porte sur la valeur de a1 = β_eff − β − 0,4081.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

DELTA_BETA_1 = 0.4081
GRAINE_BOOT = 26102699001
GRAINE_MC = 26102699002
NBOOT = 2000
NMC = 200000
PART_HORS_MAX = 0.01          # > 1 % des répliques d'étalonnage hors plage ⇒ « non calculable »
SEUIL_EFFET = 0.00135         # R4 : budget 0,0027 partagé entre les deux tests
SEUIL_COMPAT = 0.0455         # R4 : non-rejet au niveau « 2σ » pour chacun des deux tests
SEUIL_REPLIQUE = 0.001        # R5
TOL_INV = 1e-8                # WP9-ANALYSE-INV
TOL_2MACH = 1e-9              # WP9-2MACH
TOL_THETA = 1e-12
DOMAINE_F = 1e-3
ESTIMATEURS = ("sd", "plaq", "mcrg2")


# ----------------------------------------------------------------------------------------------------------------------
# courbes
# ----------------------------------------------------------------------------------------------------------------------
class Courbe:
    def __init__(self, chemin):
        d = json.load(open(chemin))
        if d.get("protocole") != "wp9" or d.get("independante") or d.get("milieux") or d.get("facteur_beta") != 1.0:
            raise SystemExit(f"{chemin} : courbe wp9 de production attendue")
        self.chemin = chemin
        self.L = d["l"]
        pts = d["courbe"]
        self.beta = np.array([p["beta"] for p in pts], dtype=float)
        self.y = {"P": np.array([p["p"] for p in pts], dtype=float), "R": np.array([p["r"] for p in pts], dtype=float)}
        self.s = {"P": np.array([p["sigma_p"] for p in pts], dtype=float), "R": np.array([p["sigma_r"] for p in pts], dtype=float)}

    def inverser(self, yn, cible):
        """β tel que PCHIP(yn)(β) = cible ; None si hors plage ou courbe non monotone."""
        if not np.all(np.diff(yn) > 0):
            return None
        if not (yn[0] <= cible <= yn[-1]):
            return None
        k = int(np.searchsorted(yn, cible, side="right")) - 1
        k = min(max(k, 0), len(yn) - 2)
        if cible == yn[k]:
            return float(self.beta[k])
        if cible == yn[k + 1]:
            return float(self.beta[k + 1])
        f = PchipInterpolator(self.beta, yn)
        return float(brentq(lambda b: float(f(b)) - cible, self.beta[k], self.beta[k + 1], xtol=1e-15, rtol=1e-15, maxiter=500))


# ----------------------------------------------------------------------------------------------------------------------
# lecture des cellules
# ----------------------------------------------------------------------------------------------------------------------
def jk_tailles(b):
    t = b.get("jackknife_bins_tailles") or []
    out = []
    for x in t:
        out.append({"taille": x["taille"], "n_groupes": x["n_groupes"], "erreur": x.get("erreur")})
    return out


def lire_cellule(desc, base):
    chemin = os.path.join(base, desc["flot"])
    d = json.load(open(chemin))
    if len(d["cellules"]) != 1:
        raise SystemExit(f"{chemin} : une cellule attendue, {len(d['cellules'])} trouvées")
    c = d["cellules"][0]
    if abs(c["beta"] - desc["beta"]) > 1e-9 or c["l_entree"] != desc["L"]:
        raise SystemExit(f"{chemin} : β/L incohérents avec la description ({c['beta']}, {c['l_entree']})")
    dom = c["domaine"][0]
    cfg = c.get("configs") or []
    f_cfg = np.array([x["fraction_hors"] for x in cfg]) if cfg else np.array([])
    amb_cfg = np.array([x["n_ambigues"] for x in cfg]) if cfg else np.array([])
    taus = [ch.get("plaquette_bloquee_indicatif", {}).get("tau") for ch in c["tau_int"]["chaines"]]
    taus_prod = [(ch.get("plaquette_production") or {}).get("tau") for ch in c["tau_int"]["chaines"]]
    cel = {
        "id": desc["id"], "famille": desc["famille"], "L": desc["L"], "beta": c["beta"], "l_fin": c["l_fin"],
        "n_chaines": c["n_chaines"], "n_configs": c["n_configs"], "flot": desc["flot"], "machine": desc.get("machine"),
        "backend": (d.get("calcul") or {}).get("backend", "cpu"),
        "p_moyenne": c["p_moyenne"], "r_moyenne": c["r_moyenne"],
        "domaine": {"fraction_hors": dom["fraction_hors"], "n_ambigues": dom["n_ambigues"], "theta_max": dom["theta_max"],
                    "n_boucles": dom["n_boucles"], "etiquette": dom["etiquette"],
                    "etiquette_recalculee": "DANS_DOMAINE" if (dom["fraction_hors"] <= DOMAINE_F and dom["n_ambigues"] == 0) else "HORS_DOMAINE",
                    "f_config_min": float(f_cfg.min()) if f_cfg.size else None,
                    "f_config_mediane": float(np.median(f_cfg)) if f_cfg.size else None,
                    "f_config_max": float(f_cfg.max()) if f_cfg.size else None,
                    "part_configs_hors": float(np.mean((f_cfg > DOMAINE_F) | (amb_cfg > 0))) if f_cfg.size else None},
        "bins": c.get("bins"),
        "tau_bloquee": taus, "tau_production": taus_prod,
        "est": {},
    }
    for e in ESTIMATEURS:
        b = c["estimateurs"].get(e)
        r = {"etat": None}
        if b is None:
            r["etat"] = "absent"
        elif b.get("etat") == "ok":
            t = jk_tailles(b)
            r.update({"etat": "ok", "valeur": b["valeur"], "sigma_chaine": b.get("erreur_jackknife"),
                      "sigma_cal_rust": b.get("sigma_etalonnage"), "tailles": t})
            if e == "mcrg2":
                r["beta_p"], r["beta_r"] = b["beta_p"], b["beta_r"]
                r["ecart_schema"] = b["ecart_schema"]["valeur"]
            errs = [x["erreur"] for x in t]
            if len(t) < 2 or any(x is None for x in errs):
                r["etat"] = "non_calculable"
                r["raison"] = "réplicat jackknife hors plage ou indéfini (σ_K ou σ_2K absent)"
            else:
                k = int(np.argmax(errs))
                r["sigma_K"], r["sigma_2K"] = errs[0], errs[1]
                r["G_K"], r["G_2K"] = t[0]["n_groupes"], t[1]["n_groupes"]
                r["sigma_stat"] = max(errs)
                r["nu_retenu"] = t[k]["n_groupes"] - 1
        elif b.get("etat") == "indefini":
            r.update({"etat": "borne", "borne": "> 50 (bissection sd)"})
        elif b.get("etat") == "hors_etalonnage":
            cote = None
            if "cote" in b:
                cote = b["cote"]
            else:
                cotes = [b.get(k, {}).get("cote") for k in ("plaquette", "rectangle") if isinstance(b.get(k), dict)]
                cotes = [x for x in cotes if x]
                cote = "/".join(sorted(set(cotes))) if cotes else None
            r.update({"etat": "borne", "borne": {"bas": "< 0,25", "haut": "> 64"}.get(cote, f"hors étalonnage ({cote})")})
        else:
            r.update({"etat": b.get("etat", "inconnu")})
        cel["est"][e] = r
    return cel


# ----------------------------------------------------------------------------------------------------------------------
# contrôles
# ----------------------------------------------------------------------------------------------------------------------
def controle_inversion(cellules, courbes):
    ecarts = []
    for c in cellules:
        cb = courbes[c["l_fin"]]
        for e in ("plaq", "mcrg2"):
            r = c["est"][e]
            if r["etat"] not in ("ok", "non_calculable"):
                continue
            bp = cb.inverser(cb.y["P"], c["p_moyenne"])
            if e == "plaq":
                ecarts.append((abs(bp - r["valeur"]) / abs(r["valeur"]), c["id"], e))
            else:
                br = cb.inverser(cb.y["R"], c["r_moyenne"])
                ecarts.append((abs(0.5 * (bp + br) - r["valeur"]) / abs(r["valeur"]), c["id"], e))
                ecarts.append((abs(bp - r["beta_p"]) / abs(r["beta_p"]), c["id"], e + ":P"))
                ecarts.append((abs(br - r["beta_r"]) / abs(r["beta_r"]), c["id"], e + ":R"))
    if not ecarts:
        return "SKIP", "aucune cellule dans la plage", 0.0
    m = max(ecarts)
    return ("PASS" if m[0] <= TOL_INV else "FAIL"), f"max écart relatif {m[0]:.3e} ({m[1]} {m[2]}) sur {len(ecarts)} valeurs (seuil {TOL_INV:g})", m[0]


def reechantillonner(cellules, courbes):
    """C_cal par rééchantillonnage paramétrique des nœuds (P, R d'un nœud : même déviée), PCHIP + inversion refaits."""
    rng = np.random.default_rng(GRAINE_BOOT)
    ids = [c["id"] for c in cellules]
    idx = {i: k for k, i in enumerate(ids)}
    rep = {e: np.full((NBOOT, len(ids)), np.nan) for e in ("plaq", "mcrg2")}
    for L in sorted(courbes):
        cb = courbes[L]
        eps = rng.standard_normal((NBOOT, len(cb.beta)))
        concernees = [c for c in cellules if c["l_fin"] == L and any(c["est"][e]["etat"] in ("ok",) for e in ("plaq", "mcrg2"))]
        for b in range(NBOOT):
            yp = cb.y["P"] + cb.s["P"] * eps[b]
            yr = cb.y["R"] + cb.s["R"] * eps[b]
            for c in concernees:
                k = idx[c["id"]]
                bp = cb.inverser(yp, c["p_moyenne"])
                if c["est"]["plaq"]["etat"] == "ok" and bp is not None:
                    rep["plaq"][b, k] = bp
                if c["est"]["mcrg2"]["etat"] == "ok":
                    br = cb.inverser(yr, c["r_moyenne"])
                    if bp is not None and br is not None:
                        rep["mcrg2"][b, k] = 0.5 * (bp + br)
    ccal = {}
    for e in ("plaq", "mcrg2"):
        n = len(ids)
        C = np.zeros((n, n))
        part_hors = np.zeros(n)
        for k, c in enumerate(cellules):
            if c["est"][e]["etat"] == "ok":
                part_hors[k] = np.mean(np.isnan(rep[e][:, k]))
        for a in range(n):
            for b in range(n):
                ca, cb_ = cellules[a], cellules[b]
                if ca["l_fin"] != cb_["l_fin"] or ca["est"][e]["etat"] != "ok" or cb_["est"][e]["etat"] != "ok":
                    continue
                m = ~np.isnan(rep[e][:, a]) & ~np.isnan(rep[e][:, b])
                if m.sum() > 10:
                    C[a, b] = np.cov(rep[e][m, a], rep[e][m, b])[0, 1]
        ccal[e] = (C, part_hors)
    ccal["sd"] = (np.zeros((len(ids), len(ids))), np.zeros(len(ids)))
    return ids, ccal


def satterthwaite(parts):
    """parts = liste de (variance, ν) ; ν = None ⇒ ∞."""
    num = sum(v for v, _ in parts) ** 2
    den = sum(v * v / n for v, n in parts if n is not None and n > 0)
    return num / den if den > 0 else math.inf


def tirer_sigma2(rng, s2, nuK, nu2K, taille):
    """σ̂² retenu = max(σ̂_K², σ̂_2K²), chacun σ² χ²_ν/ν (règle du PREREG, reproduite sous le nul)."""
    a = s2 * rng.chisquare(nuK, taille) / nuK
    b = s2 * rng.chisquare(nu2K, taille) / nu2K
    return np.maximum(a, b)


def gls(D, V):
    Vi = np.linalg.inv(V)
    un = np.ones(len(D))
    W = un @ Vi @ un
    d = (un @ Vi @ D) / W
    chi0 = D @ Vi @ D
    return chi0, d, 1.0 / math.sqrt(W), chi0 - d * d * W


def localite(cellules, ids, ccal, paires, rng):
    idx = {i: k for k, i in enumerate(ids)}
    cel = {c["id"]: c for c in cellules}
    res = {}
    for e in ESTIMATEURS:
        C = ccal[e][0]
        part_hors = ccal[e][1]
        valides = []
        for p in paires:
            a, b = cel.get(p["c12"]), cel.get(p["c24"])
            if a is None or b is None:
                continue
            ra, rb = a["est"][e], b["est"][e]
            ok = ra["etat"] == "ok" and rb["etat"] == "ok" and part_hors[idx[a["id"]]] <= PART_HORS_MAX and part_hors[idx[b["id"]]] <= PART_HORS_MAX
            dans = a["domaine"]["etiquette_recalculee"] == "DANS_DOMAINE" and b["domaine"]["etiquette_recalculee"] == "DANS_DOMAINE"
            entree = {"famille": p["famille"], "beta": p["beta"], "c12": a["id"], "c24": b["id"], "dans_dans": dans,
                      "etat12": ra["etat"], "etat24": rb["etat"]}
            if ok:
                entree.update({"delta": ra["valeur"] - rb["valeur"], "sigma_stat12": ra["sigma_stat"], "sigma_stat24": rb["sigma_stat"],
                               "sigma_cal12": math.sqrt(max(C[idx[a["id"]], idx[a["id"]]], 0)),
                               "sigma_cal24": math.sqrt(max(C[idx[b["id"]], idx[b["id"]]], 0))})
                entree["sigma_delta"] = math.sqrt(entree["sigma_stat12"] ** 2 + entree["sigma_stat24"] ** 2 + entree["sigma_cal12"] ** 2 + entree["sigma_cal24"] ** 2)
                entree["t"] = entree["delta"] / entree["sigma_delta"]
                nu = satterthwaite([(ra["sigma_stat"] ** 2, ra["nu_retenu"]), (rb["sigma_stat"] ** 2, rb["nu_retenu"]),
                                    (entree["sigma_cal12"] ** 2, None), (entree["sigma_cal24"] ** 2, None)])
                entree["nu_welch_indicatif"] = nu
                entree["z_indicatif"] = float(stats.norm.ppf(stats.t.cdf(entree["t"], nu))) if math.isfinite(nu) else entree["t"]
                valides.append((entree, ra, rb, idx[a["id"]], idx[b["id"]]))
            res.setdefault(e, {"paires": []})["paires"].append(entree)
        res.setdefault(e, {"paires": []})
        for nom, filtre in (("DANS_DANS", lambda x: x[0]["dans_dans"]), ("AUTRES", lambda x: not x[0]["dans_dans"])):
            vs = [x for x in valides if filtre(x)]
            n = len(vs)
            bloc = {"n": n}
            if n == 0:
                bloc["verdict"] = "NON_TESTABLE"
            else:
                D = np.array([x[0]["delta"] for x in vs])
                s12 = np.array([x[1]["sigma_stat"] for x in vs])
                s24 = np.array([x[2]["sigma_stat"] for x in vs])
                i12 = [x[3] for x in vs]
                i24 = [x[4] for x in vs]
                Ccal = C[np.ix_(i12, i12)] + C[np.ix_(i24, i24)]
                V = np.diag(s12 ** 2 + s24 ** 2) + Ccal
                chi0, d, sd_, chi1 = gls(D, V)
                bloc.update({"chi2_0": chi0, "delta_hat": d, "sigma_delta": sd_, "z_delta": d / sd_, "chi2_1": chi1,
                             "p0_chi2_indicatif": float(stats.chi2.sf(chi0, n)),
                             "betas": [x[0]["beta"] for x in vs], "familles": [x[0]["famille"] for x in vs]})
                # MC sous le nul : Δ* ~ N(0, V), variances re-tirées (règle du max), statistique recalculée
                Lc = np.linalg.cholesky(V)
                n12K = np.array([x[1]["G_K"] - 1 for x in vs]); n122K = np.array([x[1]["G_2K"] - 1 for x in vs])
                n24K = np.array([x[2]["G_K"] - 1 for x in vs]); n242K = np.array([x[2]["G_2K"] - 1 for x in vs])
                ge0 = 0
                gez = 0
                reste = NMC
                while reste > 0:
                    m = min(reste, 50000)
                    Dz = rng.standard_normal((m, n)) @ Lc.T
                    v12 = np.stack([tirer_sigma2(rng, s12[i] ** 2, n12K[i], n122K[i], m) for i in range(n)], axis=1)
                    v24 = np.stack([tirer_sigma2(rng, s24[i] ** 2, n24K[i], n242K[i], m) for i in range(n)], axis=1)
                    Vs = np.zeros((m, n, n)) + Ccal
                    Vs[:, np.arange(n), np.arange(n)] += v12 + v24
                    Vi = np.linalg.inv(Vs)
                    un = np.ones(n)
                    W = np.einsum("i,mij,j->m", un, Vi, un)
                    dd = np.einsum("i,mij,mj->m", un, Vi, Dz) / W
                    c0 = np.einsum("mi,mij,mj->m", Dz, Vi, Dz)
                    zz = dd * np.sqrt(W)
                    ge0 += int(np.sum(c0 >= chi0))
                    gez += int(np.sum(np.abs(zz) >= abs(d / sd_)))
                    reste -= m
                p0 = (1 + ge0) / (1 + NMC)
                pd = (1 + gez) / (1 + NMC)
                w = 1.0 / np.diag(V)
                nus = []
                for x in vs:
                    nus.append(satterthwaite([(x[1]["sigma_stat"] ** 2, x[1]["nu_retenu"]), (x[2]["sigma_stat"] ** 2, x[2]["nu_retenu"])]))
                nu_d = (w.sum() ** 2) / sum(wi * wi / ni for wi, ni in zip(w, nus) if math.isfinite(ni))
                tq = float(stats.t.ppf(0.975, nu_d)) if math.isfinite(nu_d) else 1.959964
                bloc.update({"p0_mc": p0, "p_delta_mc": pd, "nu_delta_indicatif": nu_d, "borne_delta_95": abs(d) + tq * sd_})
                if n < 2:
                    bloc["verdict"] = "NON_TESTABLE"
                elif p0 < SEUIL_EFFET or pd < SEUIL_EFFET:
                    bloc["verdict"] = "EFFET_DE_VOLUME"
                elif p0 >= SEUIL_COMPAT and pd >= SEUIL_COMPAT:
                    bloc["verdict"] = "COMPATIBLE_LOCALITE"
                else:
                    bloc["verdict"] = "INDETERMINE"
            res[e][nom] = bloc
    return res


def repliques(cellules, ids, ccal, groupes, rng):
    idx = {i: k for k, i in enumerate(ids)}
    cel = {c["id"]: c for c in cellules}
    res = {}
    for e in ESTIMATEURS:
        C, part_hors = ccal[e]
        membres = []
        structure = []
        for g in groupes:
            ok = [cel[i] for i in g["cellules"] if i in cel and cel[i]["est"][e]["etat"] == "ok" and part_hors[idx[i]] <= PART_HORS_MAX]
            structure.append({"beta": g["beta"], "cellules_valides": [c["id"] for c in ok], "cellules": g["cellules"]})
            if len(ok) >= 2:
                membres.append(ok[:3])
        ncont = sum(len(m) - 1 for m in membres)
        bloc = {"groupes": structure, "n_contrastes": ncont}
        if ncont < 2:
            bloc["verdict"] = "SKIP"
            res[e] = bloc
            continue
        cells = [c for m in membres for c in m]
        ci = [idx[c["id"]] for c in cells]
        v = np.array([c["est"][e]["valeur"] for c in cells])
        s = np.array([c["est"][e]["sigma_stat"] for c in cells])
        nK = np.array([c["est"][e]["G_K"] - 1 for c in cells])
        n2K = np.array([c["est"][e]["G_2K"] - 1 for c in cells])
        Cc = C[np.ix_(ci, ci)]

        def matrice(sig2):
            """Contrastes (une ligne par contraste), poids 1/σ² pour le troisième membre d'un triplet ; sig2 de forme (..., n)."""
            lignes = []
            pos = 0
            for m in membres:
                if len(m) >= 2:
                    a = np.zeros(sig2.shape[:-1] + (len(cells),))
                    a[..., pos] = 1.0
                    a[..., pos + 1] = -1.0
                    lignes.append(a)
                if len(m) == 3:
                    w1 = 1.0 / sig2[..., pos]
                    w2 = 1.0 / sig2[..., pos + 1]
                    a = np.zeros(sig2.shape[:-1] + (len(cells),))
                    a[..., pos + 2] = 1.0
                    a[..., pos] = -w1 / (w1 + w2)
                    a[..., pos + 1] = -w2 / (w1 + w2)
                    lignes.append(a)
                pos += len(m)
            return np.stack(lignes, axis=-2)

        A = matrice(s ** 2)
        Vb = np.diag(s ** 2) + Cc
        cvec = A @ v
        Vc = A @ Vb @ A.T
        chi = float(cvec @ np.linalg.solve(Vc, cvec))
        Lc = np.linalg.cholesky(Vb)
        ge = 0
        reste = NMC
        while reste > 0:
            m = min(reste, 50000)
            bz = rng.standard_normal((m, len(cells))) @ Lc.T
            s2 = np.stack([tirer_sigma2(rng, s[i] ** 2, nK[i], n2K[i], m) for i in range(len(cells))], axis=1)
            As = matrice(s2)
            Vbs = np.zeros((m, len(cells), len(cells))) + Cc
            Vbs[:, np.arange(len(cells)), np.arange(len(cells))] += s2
            cs = np.einsum("mkj,mj->mk", As, bz)
            Vcs = np.einsum("mkj,mjl,mnl->mkn", As, Vbs, As)
            x = np.linalg.solve(Vcs, cs[..., None])[..., 0]
            chis = np.einsum("mk,mk->m", cs, x)
            ge += int(np.sum(chis >= chi))
            reste -= m
        p = (1 + ge) / (1 + NMC)
        sfac = math.sqrt(chi / ncont)
        bloc.update({"chi2_R": chi, "p_R_mc": p, "p_R_chi2_indicatif": float(stats.chi2.sf(chi, ncont)),
                     "surdispersion": sfac,
                     "surdispersion_ic95": [math.sqrt(chi / stats.chi2.ppf(0.975, ncont)), math.sqrt(chi / stats.chi2.ppf(0.025, ncont)) if chi > 0 else 0.0],
                     "puissance_facteur2_chi2": float(stats.chi2.sf(stats.chi2.isf(SEUIL_REPLIQUE, ncont) / 4.0, ncont)),
                     "contrastes": [float(x) for x in cvec], "sigma_contrastes": [float(math.sqrt(Vc[i, i])) for i in range(len(cvec))]})
        bloc["verdict"] = "PASS" if p >= SEUIL_REPLIQUE else "FAIL"
        res[e] = bloc
    return res


def deux_machines(recoupements, base):
    lignes = []
    for r in recoupements:
        a = json.load(open(os.path.join(base, r["a"])))["cellules"][0]
        b = json.load(open(os.path.join(base, r["b"])))["cellules"][0]
        pires = []
        ok = True
        for e in ESTIMATEURS:
            ea, eb = a["estimateurs"].get(e, {}), b["estimateurs"].get(e, {})
            if ea.get("etat") != eb.get("etat"):
                ok = False
                pires.append((math.inf, f"{e}:etat"))
                continue
            if ea.get("etat") != "ok":
                continue
            champs = [("valeur", ea.get("valeur"), eb.get("valeur")), ("erreur_jackknife", ea.get("erreur_jackknife"), eb.get("erreur_jackknife"))]
            for ta, tb in zip(ea.get("jackknife_bins_tailles") or [], eb.get("jackknife_bins_tailles") or []):
                champs.append((f"bins{ta['taille']}", ta.get("erreur"), tb.get("erreur")))
            for nom, x, y in champs:
                if x is None and y is None:
                    continue
                if x is None or y is None:
                    ok = False
                    pires.append((math.inf, f"{e}:{nom}"))
                    continue
                rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
                pires.append((rel, f"{e}:{nom}"))
                ok &= rel <= TOL_2MACH
        da, db = a["domaine"][0], b["domaine"][0]
        dom_ok = da["fraction_hors"] == db["fraction_hors"] and da["n_ambigues"] == db["n_ambigues"] and da["n_boucles"] == db["n_boucles"] and abs(da["theta_max"] - db["theta_max"]) <= TOL_THETA
        ca, cb = a.get("configs") or [], b.get("configs") or []
        cfg_ok = len(ca) == len(cb) and all(x["fraction_hors"] == y["fraction_hors"] and x["n_ambigues"] == y["n_ambigues"] and abs(x["theta_max"] - y["theta_max"]) <= TOL_THETA for x, y in zip(ca, cb))
        ok = ok and dom_ok and cfg_ok
        pire = max(pires) if pires else (0.0, "-")
        lignes.append((("PASS" if ok else "FAIL"), f"WP9-2MACH-{r['id']}", f"pire écart relatif {pire[0]:.3e} ({pire[1]}), domaine identique {dom_ok}, configurations identiques {cfg_ok} ({len(ca)}) ; seuil {TOL_2MACH:g}"))
    return lignes


def ajustement_descriptif(cellules, ids, ccal, famille, betas):
    idx = {i: k for k, i in enumerate(ids)}
    out = {}
    for e in ESTIMATEURS:
        C = ccal[e][0]
        for L in (12, 24):
            cs = [c for c in cellules if c["famille"] == famille and c["L"] == L and any(abs(c["beta"] - b) < 1e-9 for b in betas)]
            cs = [c for c in cs if c["est"][e]["etat"] == "ok"]
            if len(cs) < 3:
                out[f"{e}_L{L}"] = {"n": len(cs), "note": "moins de 3 points calculables"}
                continue
            x = np.array([c["beta"] for c in cs])
            y = np.array([c["est"][e]["valeur"] for c in cs])
            ii = [idx[c["id"]] for c in cs]
            V = np.diag([c["est"][e]["sigma_stat"] ** 2 for c in cs]) + C[np.ix_(ii, ii)]
            X = np.stack([x, np.ones_like(x)], axis=1)
            Vi = np.linalg.inv(V)
            cov = np.linalg.inv(X.T @ Vi @ X)
            par = cov @ X.T @ Vi @ y
            resid = y - X @ par
            out[f"{e}_L{L}"] = {"n": len(cs), "betas": x.tolist(), "Z": par[0], "sigma_Z": math.sqrt(cov[0, 0]), "c": par[1],
                                "sigma_c": math.sqrt(cov[1, 1]), "chi2": float(resid @ Vi @ resid), "ddl": len(cs) - 2,
                                "note": "descriptif : cellules HORS domaine incluses, aucune conclusion"}
    return out


# ----------------------------------------------------------------------------------------------------------------------
def fmt(v, s, nd=3):
    if v is None:
        return "—"
    if s is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ± {s:.{nd}f}"


def main():
    global NMC, NBOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellules", required=True)
    ap.add_argument("--courbe4", required=True)
    ap.add_argument("--courbe8", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--nmc", type=int, default=NMC)
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()
    NMC, NBOOT = a.nmc, a.nboot
    if os.path.exists(a.sortie):
        raise SystemExit(f"{a.sortie} existe déjà : refus d'écraser")
    desc = json.load(open(a.cellules))
    base = os.path.dirname(os.path.abspath(a.cellules))
    courbes = {4: Courbe(a.courbe4), 8: Courbe(a.courbe8)}
    if courbes[4].L != 4 or courbes[8].L != 8:
        raise SystemExit("courbes : L = 4 et L = 8 attendus")
    cellules = [lire_cellule(d, base) for d in desc["cellules"]]
    verdicts = []
    st, det, _ = controle_inversion(cellules, courbes)
    verdicts.append((st, "WP9-ANALYSE-INV", det))
    ids, ccal = reechantillonner(cellules, courbes)
    idx = {i: k for k, i in enumerate(ids)}
    # table a1
    for c in cellules:
        for e in ESTIMATEURS:
            r = c["est"][e]
            C, part = ccal[e]
            k = idx[c["id"]]
            if r["etat"] == "ok":
                if part[k] > PART_HORS_MAX:
                    r["etat"] = "non_calculable"
                    r["raison"] = f"{100 * part[k]:.1f} % des répliques d'étalonnage hors plage"
                r["sigma_cal_boot"] = math.sqrt(max(C[k, k], 0.0))
                r["part_repliques_hors"] = float(part[k])
            if r["etat"] in ("ok", "non_calculable") and "valeur" in r:
                r["a1"] = r["valeur"] - c["beta"] - DELTA_BETA_1
                if "sigma_stat" in r:
                    r["sigma_tot"] = math.sqrt(r["sigma_stat"] ** 2 + r.get("sigma_cal_boot", 0.0) ** 2)
    rng = np.random.default_rng(GRAINE_MC)
    loc = localite(cellules, ids, ccal, desc.get("paires_localite", []), rng)
    rep = repliques(cellules, ids, ccal, desc.get("groupes_repliques", []), rng)
    for e in ESTIMATEURS:
        b = rep[e]
        if b["verdict"] == "SKIP":
            verdicts.append(("SKIP", f"WP9-REPLIQUE-{e}", f"{b['n_contrastes']} contraste(s) calculable(s) (< 2) : erreurs non contrôlées par réplicats"))
        else:
            verdicts.append((b["verdict"], f"WP9-REPLIQUE-{e}", f"χ²_R = {b['chi2_R']:.2f} pour {b['n_contrastes']} contrastes, p_R(MC) = {b['p_R_mc']:.4f} (seuil ≥ {SEUIL_REPLIQUE}) ; surdispersion {b['surdispersion']:.2f} [IC95 {b['surdispersion_ic95'][0]:.2f} ; {b['surdispersion_ic95'][1]:.2f}] ; puissance contre un facteur 2 ≈ {b['puissance_facteur2_chi2']:.2f} ; " + ("aucune anomalie détectée" if b["verdict"] == "PASS" else "ERREURS NON CALIBRÉES")))
        for nom in ("DANS_DANS", "AUTRES"):
            l = loc[e][nom]
            v = l["verdict"]
            if b["verdict"] == "FAIL" and v in ("COMPATIBLE_LOCALITE", "EFFET_DE_VOLUME"):
                v = "INDETERMINE (erreurs non calibrées)"
            if b["verdict"] == "SKIP":
                v += " [erreurs non contrôlées par réplicats]"
            l["verdict_final"] = v
            det = f"n = {l['n']}"
            if l["n"] >= 1 and "delta_hat" in l:
                det += f", δ̂ = {l['delta_hat']:+.4f} ± {l['sigma_delta']:.4f}, χ²₀ = {l['chi2_0']:.2f}, p₀(MC) = {l['p0_mc']:.4g}, p_δ(MC) = {l['p_delta_mc']:.4g}, borne |δ| < {l['borne_delta_95']:.4f}, β = {l['betas']}"
            verdicts.append(("INFO", f"WP9-LOCALITE-{e}-{nom}", f"{v} ; {det}" + (" (publié à part, sans conclusion)" if nom == "AUTRES" else "")))
    verdicts += deux_machines(desc.get("recoupements", []), base)
    fit = ajustement_descriptif(cellules, ids, ccal, "grandL", [4.0, 8.0, 16.0])
    # INFO divers
    for c in cellules:
        for e in ESTIMATEURS:
            r = c["est"][e]
            if "sigma_K" in r and r["sigma_K"]:
                r["rapport_2K_K"] = r["sigma_2K"] / r["sigma_K"]
                r["rapport_chaine_K"] = (r["sigma_chaine"] / r["sigma_K"]) if r.get("sigma_chaine") else None
        K = (c.get("bins") or {}).get("taille")
        taus = [t for t in c["tau_bloquee"] if t is not None]
        c["controle_tau"] = {"tau_max_bloquee": max(taus) if taus else None, "K": K, "ok_2tau_le_K": (2 * max(taus) <= K) if (taus and K) else None}
    os.makedirs(a.sortie)
    doc = {"format": "kvu-wp9-analyse/1", "prereg": "kvu/prereg/WP9_PREREG.md (gelé, sha256 f85b8173…)", "delta_beta_1": DELTA_BETA_1,
           "graines": {"boot": GRAINE_BOOT, "mc": GRAINE_MC}, "nboot": NBOOT, "nmc": NMC,
           "courbes": {str(L): cb.chemin for L, cb in courbes.items()},
           "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]},
           "cellules": cellules, "localite": loc, "repliques": rep, "ajustement_descriptif_grandL": fit,
           "verdicts": [{"verdict": v, "id": i, "detail": d} for v, i, d in verdicts]}
    with open(os.path.join(a.sortie, "analyse.json"), "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=float)
    with open(os.path.join(a.sortie, "verdicts.txt"), "w") as f:
        for v, i, d in verdicts:
            f.write(f"{v} {i} {d}\n")
    # table Markdown
    L_ = []
    L_.append("| famille | L | β | chaînes × cfg | domaine (f ; étiquette) | sd : a₁ | plaq : a₁ | mcrg2 : a₁ (|β′_R − β′_P|) |")
    L_.append("|---|---|---|---|---|---|---|---|")
    for c in sorted(cellules, key=lambda c: (c["L"], c["famille"], c["beta"], c["id"])):
        cols = []
        for e in ESTIMATEURS:
            r = c["est"][e]
            if r["etat"] == "ok":
                s = fmt(r["a1"], r["sigma_tot"])
            elif r["etat"] == "non_calculable":
                s = (fmt(r.get("a1"), None) + " (non calculable)") if "a1" in r else "non calculable"
            elif r["etat"] == "borne":
                s = f"β_eff {r['borne']}"
            else:
                s = r["etat"]
            if e == "mcrg2" and r.get("ecart_schema") is not None and r["etat"] in ("ok", "non_calculable"):
                s += f" ({r['ecart_schema']:.3f})"
            cols.append(s)
        d = c["domaine"]
        L_.append(f"| {c['famille']} | {c['L']} | {c['beta']:g} | {c['n_chaines']} × {c['n_configs'] // max(c['n_chaines'], 1)} | {d['fraction_hors']:.3g} ; {'DANS' if d['etiquette_recalculee'] == 'DANS_DOMAINE' else 'HORS'} | " + " | ".join(cols) + " |")
    with open(os.path.join(a.sortie, "table_a1.md"), "w") as f:
        f.write("\n".join(L_) + "\n")
    for v, i, d in verdicts:
        print(f"{v} {i} {d}")
    nf = sum(1 for v, _, _ in verdicts if v == "FAIL")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
