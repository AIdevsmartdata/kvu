"""WP14 : contrôle positif HESS-S2. Régénère EXACTEMENT les fonds gaussiens « grossiers » 4⁴ de S2 (même code `lat.py` haché, mêmes graines, même suite de tirages),
calcule ε′ (T-fix-fluct = espace kerB, coordonnées additives, référence Q0, fond fixé en jauge de Landau) par `lat.py` (dense) puis, via le lanceur, par `kvu hess lmin`.

Usage (python3 -I, venv épinglé) :
    s2_controle.py generer DOSSIER           écrit DOSSIER/fonds_s2.json et DOSSIER/fonds/*.npy ; calcule ε′ de lat.py ; contrôle contre les valeurs enregistrées de S2
    s2_controle.py commandes DOSSIER KVU     imprime, une par ligne, « id<TAB>commande kvu » (le lanceur exécute ; rien n'est lancé ici)
    s2_controle.py comparer DOSSIER RAPPORTS écrit les lignes PASS|FAIL|SKIP des gates HESS-S2-* sur la sortie standard

Gates (seuils du PREREG §3, gelés) :
  HESS-S2-REGEN    : INFO seulement (mesuré avant gel : les fonds régénérés ne sont pas ceux de S2, même code et mêmes graines : base propre de eigh dépendante de la bibliothèque) ;
  HESS-S2-MED      : médiane (16 fonds) de ε′ de kvu = médiane de lat.py, pour chaque β ∈ {8 … 256}, |Δ| ≤ 1e-8 ;
  HESS-S2-ALPHA    : α(kvu) (moindres carrés de ln médiane contre ln g, β ∈ {8 … 256}) ∈ [0,99 ; 1,05] (= 1,02 ± 0,03) ;
  HESS-S2-FENETRE  : α sur la fenêtre {4, 8, 16} de la campagne : |α(kvu) − α(lat.py)| ≤ 1e-7 (INFO : α(S2 enregistré) sur la même fenêtre).
"""
import json
import math
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "hess"))
import outils as O   # noqa: E402  (outils de WP13 : charger_lat, liens_lat_vers_champ, ecrire_npy, sha256 ; haché au PREREG)

S2_RES = "/chemin/vers/projet/chantiers/vision_20261025/calculs/S2_dependance_champ/res"
BETAS_S2 = [2.5, 4, 8, 16, 32, 64, 128, 256]       # suite complète de S2 (la suite de tirages en dépend)
BETAS_CONTROLE = [4, 8, 16, 32, 64, 128, 256]      # évalués ; l'ajustement de S2 porte sur {8 … 256} ; 4 sert à la fenêtre de la campagne
BETAS_AJUSTEMENT_S2 = [8, 16, 32, 64, 128, 256]
GRAINES = [1, 2, 3, 4]
NF = 4
N = 4
ALPHA_CIBLE, ALPHA_TOL = 1.02, 0.03
TOL_REGEN, TOL_MED, TOL_FENETRE = 1e-9, 1e-8, 1e-7


def ident(graine, beta, fond):
    return f"s{graine}_b{beta:g}_f{fond}"


def alpha_ols(medianes, betas):
    x = np.array([math.log(math.sqrt(4.0 / b)) for b in betas])
    y = np.array([math.log(medianes[b]) for b in betas])
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def enregistre_s2(graine):
    """{(β, fond): ε′} de S2 pour les fonds « grossier », T-fix-fluct."""
    out = {}
    for ligne in open(os.path.join(S2_RES, f"su2_N4_s{graine}.jsonl")):
        r = json.loads(ligne)
        if r["kind"] == "grossier":
            out[(float(r["beta"]), int(r["fond"]))] = r["Tfix_fluct"][0]
    return out


def generer(dossier):
    if os.path.exists(os.path.join(dossier, "fonds_s2.json")):
        print("s2_controle : fonds_s2.json existe (jamais écrasé)", file=sys.stderr)
        return 3
    os.makedirs(os.path.join(dossier, "fonds"), exist_ok=True)
    lat = O.charger_lat()
    geo = lat.Geo(N)
    K = (geo.R.T @ geo.R).toarray()
    Q0 = 0.25 * np.kron(K, np.eye(3))
    Vfl, _ = lat.basis_fix(geo, fluct=True, ncol=3)
    gs = lat.Gauss(geo, 0.25)
    fonds = []
    for graine in GRAINES:
        rng = np.random.default_rng(1000 * graine + 1 + 17 * N)   # S2/prod.py : modèle su2
        stock = enregistre_s2(graine)
        for beta in BETAS_S2:
            g = math.sqrt(4 / beta)
            invb = (g * g / 4)   # S2/prod.py : invb = g*g/4 (modèle su2)
            for r in range(NF):
                for kind in ("grossier", "plein"):
                    A = gs.draw(rng, invb, 3, coarse=(kind == "grossier"))   # la suite de tirages DOIT suivre celle de S2
                    if kind != "grossier" or beta not in BETAS_CONTROLE:
                        continue
                    U = lat.su2_exp(A)
                    Uf, theta, it = lat.landau_fix(geo, U, tol=1e-24, maxit=5000)
                    Af = lat.su2_log(Uf)
                    H, _ = lat.hess_su2_additive(geo, Af)
                    lmin, _, _ = lat.gen_eig_extremes(H, Q0, Vfl)
                    npy = os.path.join(dossier, "fonds", ident(graine, beta, r) + ".npy")
                    O.ecrire_npy(npy, O.liens_lat_vers_champ(Uf, N))
                    fonds.append({"id": ident(graine, beta, r), "graine": graine, "beta": float(beta), "fond": r, "chemin": os.path.abspath(npy),
                                  "sha256": O.sha256(npy), "eps_lat": 1.0 - float(lmin), "eps_s2_enregistre": stock[(float(beta), r)],
                                  "landau_theta": float(theta), "landau_it": int(it)})
        print(f"graine {graine} : {sum(1 for f in fonds if f['graine'] == graine)} fonds", flush=True)
    with open(os.path.join(dossier, "fonds_s2.json"), "x", encoding="utf-8") as f:
        json.dump({"schema": "wp14-s2-fonds/1", "lat_sha256": O.sha256(os.path.join(O.LATDIR, "lat.py")), "fonds": fonds}, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


def commandes(dossier, kvu):
    doc = json.load(open(os.path.join(dossier, "fonds_s2.json")))
    for x in doc["fonds"]:
        rap = os.path.join(dossier, "rapports", "lmin_kerB_" + x["id"] + ".json")
        cmd = [kvu, "hess", "lmin", "--config", x["chemin"], "--espace", "kerB", "--coord", "additive", "--reference", "q0", "--beta", f"{x['beta']:g}",
               "--graine", "1", "--tol", "1e-10", "--fils", "1", "--sortie", rap]
        print(x["id"] + "\t" + " ".join(cmd))
    return 0


def mediane_par_beta(valeurs):
    return {b: float(np.median(v)) for b, v in valeurs.items()}


def comparer(dossier, rapports):
    doc = json.load(open(os.path.join(dossier, "fonds_s2.json")))
    fonds = doc["fonds"]
    lignes = []
    # --- HESS-S2-REGEN
    pire = max(abs(x["eps_lat"] - x["eps_s2_enregistre"]) for x in fonds)
    lignes.append(("INFO", "HESS-S2-REGEN", f"{len(fonds)} fonds régénérés ; pire |ε′(lat.py) − ε′(S2 enregistré)| = {pire:.2e} : les fonds ne sont PAS ceux de S2 (la base propre des modes dégénérés de eigh dépend de la bibliothèque) ; contrôle statistique, pas bit à bit"))
    # --- kvu
    eps_kvu, manque, non_cv = {}, [], []
    for x in fonds:
        chemin = os.path.join(rapports, "lmin_kerB_" + x["id"] + ".json")
        try:
            r = json.load(open(chemin))
        except (OSError, ValueError):
            manque.append(x["id"])
            continue
        q = r.get("q_restreinte", {})
        if not (r.get("solveur", {}).get("methode") == "lobpcg" and r["solveur"].get("converge") is True and r.get("residu_projete_rel", 1) <= 1e-9
                and q.get("converge_min") is True and q.get("rapport_borne", 0) >= 1e-8):
            non_cv.append(x["id"])
        eps_kvu[x["id"]] = float(r["eps_prime"])
    if manque or non_cv:
        lignes.append(("FAIL", "HESS-S2-MED", f"rapports absents : {manque[:5]}… ({len(manque)}) ; LOBPCG non convergé : {non_cv[:5]}… ({len(non_cv)})"))
        lignes.append(("SKIP", "HESS-S2-ALPHA", "non évalué : rapports incomplets"))
        lignes.append(("SKIP", "HESS-S2-FENETRE", "non évalué : rapports incomplets"))
        return lignes
    par_beta_kvu, par_beta_lat, par_beta_s2 = {}, {}, {}
    for x in fonds:
        par_beta_kvu.setdefault(int(x["beta"]), []).append(eps_kvu[x["id"]])
        par_beta_lat.setdefault(int(x["beta"]), []).append(x["eps_lat"])
        par_beta_s2.setdefault(int(x["beta"]), []).append(x["eps_s2_enregistre"])
    mk, ml, ms = mediane_par_beta(par_beta_kvu), mediane_par_beta(par_beta_lat), mediane_par_beta(par_beta_s2)
    ecarts = {b: abs(mk[b] - ml[b]) for b in mk}
    pire_med = max(ecarts[b] for b in BETAS_AJUSTEMENT_S2)
    nfonds = {b: len(v) for b, v in par_beta_kvu.items()}
    ok_n = all(nfonds[b] == NF * len(GRAINES) for b in BETAS_CONTROLE)
    lignes.append(("PASS" if (pire_med <= TOL_MED and ok_n) else "FAIL", "HESS-S2-MED",
                   f"16 fonds par β ({'ok' if ok_n else 'NON'}) ; pire |médiane kvu − médiane lat.py| sur β ∈ {{8…256}} = {pire_med:.2e} (seuil {TOL_MED:g}) ; médianes kvu "
                   + " ".join(f"{b}:{mk[b]:.6f}" for b in BETAS_AJUSTEMENT_S2)))
    a_kvu = alpha_ols(mk, BETAS_AJUSTEMENT_S2)
    a_lat = alpha_ols(ml, BETAS_AJUSTEMENT_S2)
    ok_a = abs(a_kvu - ALPHA_CIBLE) <= ALPHA_TOL
    lignes.append(("PASS" if ok_a else "FAIL", "HESS-S2-ALPHA", f"α(kvu) = {a_kvu:.4f} ; α(lat.py) = {a_lat:.4f} ; cible {ALPHA_CIBLE} ± {ALPHA_TOL} ; α(S2 enregistré) = {alpha_ols(ms, BETAS_AJUSTEMENT_S2):.4f}"))
    fen = [4, 8, 16]
    f_kvu, f_lat, f_s2 = alpha_ols(mk, fen), alpha_ols(ml, fen), alpha_ols(ms, fen)
    lignes.append(("PASS" if abs(f_kvu - f_lat) <= TOL_FENETRE else "FAIL", "HESS-S2-FENETRE",
                   f"fenêtre {{4, 8, 16}} : α(kvu) = {f_kvu:.4f}, α(lat.py) = {f_lat:.4f}, |Δ| = {abs(f_kvu - f_lat):.2e} (seuil {TOL_FENETRE:g}) ; INFO α(S2 enregistré) = {f_s2:.4f}"))
    return lignes


def main():
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "generer":
        return generer(a[1])
    if len(a) == 3 and a[0] == "commandes":
        return commandes(a[1], a[2])
    if len(a) == 3 and a[0] == "comparer":
        for s, i, d in comparer(a[1], a[2]):
            print(f"{s} {i} {d}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
