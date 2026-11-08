"""WP14 : gates et verdict CALCULÉS depuis les fichiers de la campagne (jamais écrits en dur). python3 -I -B.
  verdict.py autotest                                   auto-test de l'estimateur et de la règle (réponses connues)
  verdict.py verdict CAMP FONDS.json SHA_CANDIDAT SHA_KVU [WP13_VERDICT.txt] [ignoré]   lignes PASS|FAIL|SKIP puis « VERDICT WP14 ... »
Espaces décisifs : tcov (a) et kerB (c) ; (b) kerdu non exécuté (coût), (d) fibre exclu (WP12b non vert)."""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimateur as E
import plan as P

ESPACES = ("tcov", "kerB")
DECISIFS = ("tcov", "kerB")
BETAS_BAS = (2.3, 2.5, 2.8)
RESIDU_MAX = 1e-9
STAB_TOL = 1e-6
G3_TOL = {"kerB": 1e-6, "tcov": 1e-9}
MUT_ECART = 0.1
SHA_FONDS = "19b584e6006b8a67a16f39053079bb3202e3894e755c77f6e4da7bc0468f6d04"
SHA_LAT = "a29c19be8eb6d07685068de0ae2eade0594cbc893f0ae86150f506446f4871af"
BETAS_S2 = (4, 8, 16, 32, 64, 128, 256)
N_FONDS_TOTAL = 36


def charge(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def autotest():
    out = []
    d = charge("/chemin/vers/projet/chantiers/vision_20261025/calculs/S2_dependance_champ/resultats.json")["su2|Tfix_fluct|grossier|eps"]
    med = {float(b): m for b, m in d["mediane"].items()}
    a = E.alpha_depuis_medianes(med, (8.0, 16.0, 32.0, 64.0, 128.0, 256.0))
    out.append(("PASS" if abs(a - d["alpha"]) < 1e-9 else "FAIL", "ANA-S2", f"α recalculé {a:.6f} = α de S2 {d['alpha']:.6f}"))
    rng = np.random.default_rng(7)
    def cas(alpha, sig, nrep=200):
        res = {}
        for r in range(nrep):
            vals = {b: 0.25 * E.g_de_beta(b) ** alpha * np.exp(sig * rng.standard_normal(8)) for b in E.BETAS_AJUSTEMENT}
            al, lo, hi, m, _ = E.alpha_bootstrap(vals, 0, b_boot=300)
            et, _ = E.lecture_espace(al, lo, hi, [8] * 3, m)
            res[et] = res.get(et, 0) + 1
        return {k: v / nrep for k, v in res.items()}
    r = cas(1.0, 0.02); out.append(("PASS" if r.get("COMME-S2", 0) >= 0.8 and sum(v for k, v in r.items() if k.startswith("DIFF")) <= 0.02 else "FAIL", "ANA-ALPHA1", str(r)))
    r = cas(2.0, 0.05); out.append(("PASS" if r.get("DIFFERENT+", 0) >= 0.95 else "FAIL", "ANA-ALPHA2", str(r)))
    r = cas(0.5, 0.05); out.append(("PASS" if r.get("DIFFERENT-", 0) >= 0.95 else "FAIL", "ANA-ALPHA05", str(r)))
    r = cas(1.0, 0.3); out.append(("PASS" if r.get("INDETERMINE", 0) >= 0.9 else "FAIL", "ANA-BRUIT", str(r)))
    c = E.combiner({"tcov": "COMME-S2", "kerB": "DIFFERENT+"}, True, True)[0], E.combiner({"tcov": "COMME-S2", "kerB": "COMME-S2"}, True, True)[0], E.combiner({"tcov": "COMME-S2"}, False, True)[0]
    out.append(("PASS" if c == ("DIFFERENT", "COMME-S2", "INDETERMINE") else "FAIL", "ANA-COMBINER", str(c)))
    return out


def sha_fichier(p):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def resoudre(camp, nom, dossier="manifestes"):
    """Tentative à juger. nom.json d'abord ; nom.a2 n'est admis QUE si nom est REFUSE (rien n'avait démarré), nom.a3 que si nom et nom.a2 sont REFUSE.
    Toute autre situation après une tentative non terminée par un refus : la tentative fautive est rendue (donc rejetée)."""
    m = None
    for suf in ("", ".a2", ".a3"):
        m = charge(os.path.join(camp, dossier, nom + suf + ".json"))
        if m is None:
            return None if suf == "" else prec
        if m.get("statut") == "TERMINE" and m.get("rc") == 0:
            return m
        if m.get("statut") != "REFUSE":
            return m
        prec = m
    return m


def manifeste_ok(camp, nom, sha_cand, sha_kvu, rapport=None, entrees=None, argv=None, dossier="manifestes", script=None):
    """Certifie UNE tentative : rc 0, dépôt propre au SHA candidat, binaire kvu déclaré (sha256 attendu), chaque entrée attendue déclarée avec le sha256 ACTUEL du fichier,
    entrées inchangées, commande EXACTEMENT égale à argv, rapport = sortie hachée du manifeste."""
    m = resoudre(camp, nom, dossier)
    if m is None:
        return False, "manifeste absent"
    if m.get("statut") != "TERMINE" or m.get("rc") != 0:
        return False, f"statut {m.get('statut')} rc {m.get('rc')}"
    dp = m.get("depot", {})
    if dp.get("sha") != sha_cand or dp.get("sale"):
        return False, "dépôt ≠ candidat ou sale"
    ent = {e.get("chemin"): e.get("sha256") for e in m.get("entrees_avant", [])}
    if sha_kvu and not any(c.endswith("/kvu") and s == sha_kvu for c, s in ent.items()):
        return False, "binaire kvu non déclaré en entrée"
    for e in entrees or []:
        if e not in ent or ent[e] != sha_fichier(e):
            return False, f"entrée attendue absente ou de sha256 ≠ fichier actuel : {os.path.basename(e)}"
    if rapport:
        sh = sha_fichier(rapport)
        if not sh or sh not in [f.get("sha256") for f in m.get("sorties", {}).get("fichiers", [])]:
            return False, "rapport ≠ sortie hachée du manifeste"
    if m.get("entrees_modifiees"):
        return False, "entrée modifiée"
    if argv is not None and [str(x) for x in m.get("commande", [])] != [str(x) for x in argv]:
        return False, "commande ≠ commande prévue"
    cmd = [str(x) for x in m.get("commande", [])]
    if sha_kvu and cmd and os.path.basename(cmd[0]) == "kvu" and ent.get(cmd[0]) != sha_kvu:
        return False, "l'exécutable de la commande n'est pas le binaire déclaré"
    return True, ""


def certifie(r):
    """Critères numériques communs à TOUT rapport lmin."""
    if r is None:
        return False
    s, q = r.get("solveur", {}), r.get("q_restreinte", {})
    return bool(s.get("methode") == "lobpcg" and s.get("converge") is True and r.get("residu_projete_rel", 1) <= RESIDU_MAX and q.get("converge_min") is True
                and q.get("rapport_borne", 0) >= 1e-8 and all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ("eps_prime", "lambda_min")))


def kvu_de(camp, nom, dossier="manifestes"):
    m = resoudre(camp, nom, dossier)
    return str(m["commande"][0]) if m and m.get("commande") else "?"


def verdict(camp, fonds_json, sha_cand, sha_kvu, wp13=None, ignore=None):
    L = []
    camp = os.path.abspath(camp)
    # --- liste gelée des fonds : sha256 du PREREG, 36 identifiants uniques
    fsha_liste = sha_fichier(fonds_json)
    doc = charge(fonds_json) or {"fonds": []}
    fl = doc["fonds"]
    ids = [x["id"] for x in fl]
    ok_liste = fsha_liste == SHA_FONDS and len(ids) == len(set(ids)) == N_FONDS_TOTAL
    L.append(("PASS" if ok_liste else "FAIL", "WP14-FONDS", f"liste de fonds : sha256 {'conforme' if fsha_liste == SHA_FONDS else 'NON conforme'} au PREREG, {len(set(ids))} identifiants uniques sur {len(ids)}"))
    L += autotest()
    vals = {e: {} for e in ESPACES}; echec = []
    fond_par_id = {x["id"]: x for x in fl}
    def chemins(e, x):
        fixe = os.path.join(camp, "travail", x["id"] + "_fixe.npy")
        return (x["chemin"] if e == "tcov" else fixe), fixe
    for x in fl:
        for e in ESPACES:
            nom = f"lmin_{e}_{x['id']}"
            rp = os.path.join(camp, "rapports", nom + ".json")
            conf, fixe = chemins(e, x)
            kvu = kvu_de(camp, nom)
            ok, mot = manifeste_ok(camp, nom, sha_cand, sha_kvu, rp, [conf], P.lmin_argv(kvu, e, conf, f"{x['beta']:g}", rp, P.GRAINE_LOBPCG, P.BLOC))
            if ok and e == "kerB":
                ok, mot = manifeste_ok(camp, "landau_" + x["id"], sha_cand, sha_kvu, fixe, [x["chemin"]], [kvu, "hess", "landau", "--config", x["chemin"], "--sortie", fixe, "--fils", "1"])
                mot = ("landau : " + mot) if mot else ""
            r = charge(rp)
            if not ok or not certifie(r):
                echec.append(f"{nom}: {mot or 'non certifié'}"); continue
            vals[e].setdefault(x["beta"], []).append(r["eps_prime"])
    L.append(("PASS" if not echec and ok_liste else "FAIL", "WP14-EXEC-CONV", f"{len(echec)} calcul(s) manquant(s) ou non certifié(s) {echec[:6]}"))
    # --- stabilité
    st = []
    for e in ESPACES:
        for b in ("4", "8", "16"):
            xid = f"b{b}_c1_000600"; x = fond_par_id.get(xid)
            if x is None:
                st.append(False); continue
            nom = f"stab_{e}_{xid}"; rp = os.path.join(camp, "rapports", nom + ".json")
            conf, fixe = chemins(e, x)
            ok, _ = manifeste_ok(camp, nom, sha_cand, sha_kvu, rp, [conf], P.lmin_argv(kvu_de(camp, nom), e, conf, f"{x['beta']:g}", rp, P.STAB_GRAINE, P.STAB_BLOC))
            r, r0 = charge(rp), charge(os.path.join(camp, "rapports", f"lmin_{e}_{xid}.json"))
            st.append(bool(ok and certifie(r) and certifie(r0) and abs(r["lambda_min"] - r0["lambda_min"]) <= STAB_TOL))
    L.append(("PASS" if all(st) else "FAIL", "WP14-STAB", f"graine 2, bloc 6 : {sum(st)}/{len(st)} accords |Δλ_min| ≤ {STAB_TOL:g}"))
    # --- G3 L = 12 (le fond de jauge pure est certifié : script, graine, sortie)
    pur, fxg = os.path.join(camp, "travail", "g3_pur.npy"), os.path.join(camp, "travail", "g3_pur_fixe.npy")
    mf = resoudre(camp, "g3_fond")
    script_g3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g3_l12.py")
    ok_fond, _ = manifeste_ok(camp, "g3_fond", sha_cand, None, pur, [script_g3], [mf["commande"][0], "-I", "-B", script_g3, pur] if mf and mf.get("commande") else [])
    kvu_g3 = kvu_de(camp, "g3_landau")
    ok_land, _ = manifeste_ok(camp, "g3_landau", sha_cand, sha_kvu, fxg, [pur], [kvu_g3, "hess", "landau", "--config", pur, "--sortie", fxg, "--fils", "1"])
    g = [ok_fond and ok_land]
    for e, tol in G3_TOL.items():
        rp = os.path.join(camp, "rapports", f"g3_lmin_{e}.json"); r = charge(rp)
        conf = fxg if e == "kerB" else pur
        ok, _ = manifeste_ok(camp, f"g3_lmin_{e}", sha_cand, sha_kvu, rp, [conf], P.lmin_argv(kvu_de(camp, f"g3_lmin_{e}"), e, conf, "8", rp, P.GRAINE_LOBPCG, P.BLOC))
        g.append(bool(ok and certifie(r) and abs(r["eps_prime"]) <= tol))
    L.append(("PASS" if all(g) else "FAIL", "WP14-G3-L12", f"jauge pure L=12 certifiée (fond, Landau, kerB ≤ 1e-6, tcov ≤ 1e-9) : {g}"))
    # --- mutant M-b
    mu = []
    for i in P.MUT_FONDS:
        x = fond_par_id.get(i)
        if x is None:
            mu.append(False); continue
        rp = os.path.join(camp, "rapports", f"mut_kerB_{i}.json"); fixe = os.path.join(camp, "travail", i + "_fixe.npy")
        ok, _ = manifeste_ok(camp, f"mut_kerB_{i}", sha_cand, sha_kvu, rp, [fixe], P.lmin_argv(kvu_de(camp, f"mut_kerB_{i}"), "kerB", fixe, f"{x['beta']:g}", rp, P.GRAINE_LOBPCG, P.BLOC) + ["--mutant", "M-b"])
        r, r0 = charge(rp), charge(os.path.join(camp, "rapports", f"lmin_kerB_{i}.json"))
        mu.append(bool(ok and certifie(r) and certifie(r0) and abs(r["eps_prime"] - r0["eps_prime"]) >= MUT_ECART))
    L.append(("PASS" if all(mu) else "FAIL", "WP14-MUT", f"M-b s'écarte d'au moins {MUT_ECART} du calcul sain : {mu}"))
    # --- instrument (rejeu WP13)
    iw = open(wp13).read().strip() if wp13 and os.path.exists(wp13) else ""
    lancer_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hess", "lancer.sh")
    mi = resoudre(camp, "instr")
    kvu_i = str(mi["commande"][1])[len("KVU_BIN="):] if mi and len(mi.get("commande", [])) == 5 and str(mi["commande"][1]).startswith("KVU_BIN=") else "?"
    argv_i = ["env", "KVU_BIN=" + kvu_i, "bash", lancer_sh, os.path.join(camp, "instr_out")] if mi and mi.get("commande") else []
    # le chemin exact de lancer.sh est celui déclaré au manifeste (même racine que le dépôt)
    if mi and mi.get("commande") and len(mi["commande"]) == 5 and os.path.realpath(mi["commande"][3]) == os.path.realpath(lancer_sh):
        argv_i[3] = mi["commande"][3]
    oki, moti = manifeste_ok(camp, "instr", sha_cand, sha_kvu, wp13 if wp13 else None, [], argv_i)
    L.append(("PASS" if (oki and iw.startswith("VERDICT WP13 VERT")) else "FAIL", "WP14-INSTR", (iw[:150] or "VERDICT WP13 absent") + ("" if oki else " ; manifeste : " + moti)))
    # --- contrôle positif HESS-S2 : recalculé ICI depuis les rapports actuels
    import s2_controle as S
    f2p = os.path.join(camp, "s2", "fonds_s2.json")
    f2 = charge(f2p)
    attendus = {S.ident(g_, b, r) for g_ in S.GRAINES for b in BETAS_S2 for r in range(S.NF)}
    ids2 = [x["id"] for x in f2["fonds"]] if f2 else []
    ok_ids = f2 is not None and len(ids2) == len(set(ids2)) == len(attendus) and set(ids2) == attendus and f2.get("lat_sha256") == SHA_LAT
    sg = resoudre(camp, "s2_generer")
    ok_gen, _ = manifeste_ok(camp, "s2_generer", sha_cand, None, f2p, [os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_controle.py")]) if sg else (False, "")
    mans = []
    for x in (f2["fonds"] if f2 else []):
        rp = os.path.join(camp, "s2", "rapports", "lmin_kerB_" + x["id"] + ".json")
        argv = [kvu_de(camp, "s2_" + x["id"], os.path.join("s2", "manifestes")), "hess", "lmin", "--config", x["chemin"], "--espace", "kerB", "--coord", "additive", "--reference", "q0",
                "--beta", f"{x['beta']:g}", "--graine", "1", "--tol", "1e-10", "--fils", "1", "--sortie", rp]
        ok, _ = manifeste_ok(camp, "s2_" + x["id"], sha_cand, sha_kvu, rp, [x["chemin"]], argv, dossier=os.path.join("s2", "manifestes"))
        mans.append(ok)
    sl = []
    if f2 and ok_ids:
        for s_, i_, d_ in S.comparer(os.path.join(camp, "s2"), os.path.join(camp, "s2", "rapports")):
            sl.append([s_, i_, d_])
    sl.append(["PASS" if (ok_ids and ok_gen and mans and all(mans)) else "FAIL", "HESS-S2-MANIFESTES", f"{sum(mans)}/{len(mans)} calculs avec manifeste conforme ; 112 identifiants exacts : {ok_ids} ; régénération certifiée : {ok_gen}"])
    ids_requis = {"HESS-S2-MED", "HESS-S2-ALPHA", "HESS-S2-FENETRE", "HESS-S2-MANIFESTES"}
    ctl = ids_requis <= {l[1] for l in sl if l[0] == "PASS"} and not any(l[0] in ("FAIL", "SKIP") for l in sl)
    L.append(("PASS" if ctl else "FAIL", "HESS-S2", "; ".join(f"{l[0]} {l[1]}" for l in sl) or "absent"))
    instr_ok = all(s == "PASS" for s, i, _ in L if i in ("WP14-FONDS", "WP14-EXEC-CONV", "WP14-STAB", "WP14-G3-L12", "WP14-MUT", "WP14-INSTR") or i.startswith("ANA-"))
    # --- estimation
    etiq, info = {}, []
    for e in ESPACES:
        v = vals[e]
        nb = [len(v.get(b, [])) for b in E.BETAS_AJUSTEMENT]
        if any(b not in v for b in E.BETAS_AJUSTEMENT):
            etiq[e] = "INDETERMINE"; info.append(f"{e}: β d'ajustement incomplets"); continue
        a, lo, hi, med, _ = E.alpha_bootstrap({b: v[b] for b in E.BETAS_AJUSTEMENT}, E.INDICE_ESPACE[e])
        et, mot = E.lecture_espace(a, lo, hi, nb, med); etiq[e] = et
        info.append(f"{e}: α = {a:.4f} IC95 [{lo:.4f} ; {hi:.4f}] -> {et} ({mot}) ; médianes ε′ " + " ".join(f"β{b:g}:{m:.5f}" for b, m in sorted(med.items())) + " ; ε′/g " + " ".join(f"β{b:g}:{m / E.g_de_beta(b):.4f}" for b, m in sorted(med.items())))
        bas = " ".join(f"β{b:g}:{np.median(v[b]):.5f}(n={len(v[b])}) ε′/g={np.median(v[b]) / E.g_de_beta(b):.4f}" for b in BETAS_BAS if b in v)
        info.append(f"{e} β rapportés à part (hors ajustement) : {bas}")
    vd, mot = E.combiner({e: etiq[e] for e in DECISIFS}, ctl, instr_ok)
    for s, i, d in L: print(f"{s} {i} {d}")
    for l in sl: print(f"CONTROLE {l[0]} {l[1]} {l[2]}")
    for l in info: print("INFO " + l)
    print(f"VERDICT WP14 {vd} : {mot}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["autotest"]:
        r = autotest()
        for s, i, d in r: print(s, i, d)
        sys.exit(0 if all(s == "PASS" for s, _, _ in r) else 1)
    if a[:1] == ["verdict"] and len(a) >= 5:
        sys.exit(verdict(*a[1:]))
    print(__doc__); sys.exit(2)
