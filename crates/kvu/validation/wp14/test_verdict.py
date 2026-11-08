"""WP14 : test de bout en bout de verdict.py sur une campagne FABRIQUÉE (fichiers factices, aucune physique). python3 -I -B test_verdict.py
Positifs : α = 1 → COMME-S2 ; α = 2 → DIFFERENT ; refus puis .a2 réussi → accepté.
Falsifications (toutes → INDETERMINE) : rapport périmé, Landau absent, mutant sans --mutant, fond fixé remplacé, reprise .a2 après un calcul DÉMARRÉ (rc 1), liste de fonds modifiée,
liste à doublons, contrôle S2 réduit à un fond, mauvais β dans la commande, mutant non convergé, g3_fond sans manifeste, commande avec mauvais --fils."""
import contextlib, hashlib, io, json, math, os, shutil, sys, tempfile
import numpy as np
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import verdict as V
import estimateur as E
import plan as P
import s2_controle as S

SHA = "c" * 40
PY = sys.executable
SCRIPT_G3 = os.path.join(ICI, "g3_l12.py")
SCRIPT_S2 = os.path.join(ICI, "s2_controle.py")
LANCER = os.path.join(ICI, "..", "hess", "lancer.sh")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ecrire(p, contenu=b"x"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(contenu)


def ent(p):
    return {"chemin": os.path.abspath(p), "sha256": sha(p), "octets": os.path.getsize(p)}


def manif(camp, nom, entrees, sortie, cmd, dossier="manifestes", statut="TERMINE", rc=0):
    m = {"statut": statut, "rc": rc, "depot": {"sha": SHA, "sale": False}, "commande": [str(c) for c in cmd], "entrees_avant": [ent(e) for e in entrees],
         "entrees_modifiees": [], "sorties": {"fichiers": [ent(sortie)] if sortie else []}}
    p = os.path.join(camp, dossier, nom + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(m, open(p, "w"))


def rapport(p, eps, conv=True, lam=None):
    d = {"eps_prime": eps, "lambda_min": (1 - eps) if lam is None else lam, "residu_projete_rel": 1e-11, "q_restreinte": {"converge_min": True, "rapport_borne": 0.02},
         "solveur": {"methode": "lobpcg", "converge": conv}}
    ecrire(p, json.dumps(d).encode())


def fabriquer(camp, alpha, rng):
    kvu = os.path.join(camp, "bin", "kvu"); ecrire(kvu, b"binaire")
    fonds = []
    for beta, n in (("2.3", 4), ("2.5", 4), ("2.8", 4), ("4", 8), ("8", 8), ("16", 8)):
        for i in range(n):
            fid = f"b{beta}_c{1 + i % 2}_{600 + 600 * (i // 2):06d}"
            ch = os.path.join(camp, "fonds", fid + ".npy"); ecrire(ch, fid.encode())
            fonds.append({"id": fid, "beta": float(beta), "beta_dir": beta, "chemin": ch, "sha256": sha(ch)})
    fj = os.path.join(camp, "fonds.json"); json.dump({"fonds": fonds}, open(fj, "w"), sort_keys=True)
    V.SHA_FONDS = sha(fj)
    for x in fonds:
        fx = os.path.join(camp, "travail", x["id"] + "_fixe.npy"); ecrire(fx, b"fixe" + x["id"].encode())
        manif(camp, "landau_" + x["id"], [x["chemin"], kvu], fx, [kvu, "hess", "landau", "--config", x["chemin"], "--sortie", fx, "--fils", "1"])
        for e in V.ESPACES:
            eps = 0.25 * E.g_de_beta(x["beta"]) ** alpha * math.exp(0.02 * rng.standard_normal())
            rp = os.path.join(camp, "rapports", f"lmin_{e}_{x['id']}.json"); rapport(rp, eps)
            conf = x["chemin"] if e == "tcov" else fx
            manif(camp, f"lmin_{e}_{x['id']}", [conf, kvu], rp, P.lmin_argv(kvu, e, conf, f"{x['beta']:g}", rp, P.GRAINE_LOBPCG, P.BLOC))
    ids = {x["id"]: x for x in fonds}
    for e in V.ESPACES:
        for b in ("4", "8", "16"):
            xid = f"b{b}_c1_000600"; x = ids[xid]
            r0 = json.load(open(os.path.join(camp, "rapports", f"lmin_{e}_{xid}.json")))
            rp = os.path.join(camp, "rapports", f"stab_{e}_{xid}.json"); rapport(rp, r0["eps_prime"] + 1e-9, lam=r0["lambda_min"] - 1e-9)
            conf = x["chemin"] if e == "tcov" else os.path.join(camp, "travail", xid + "_fixe.npy")
            manif(camp, f"stab_{e}_{xid}", [conf, kvu], rp, P.lmin_argv(kvu, e, conf, f"{x['beta']:g}", rp, P.STAB_GRAINE, P.STAB_BLOC))
    pur, fxg = os.path.join(camp, "travail", "g3_pur.npy"), os.path.join(camp, "travail", "g3_pur_fixe.npy")
    ecrire(pur, b"pur"); ecrire(fxg, b"purfixe")
    manif(camp, "g3_fond", [SCRIPT_G3], pur, [PY, "-I", "-B", SCRIPT_G3, pur])
    manif(camp, "g3_landau", [pur, kvu], fxg, [kvu, "hess", "landau", "--config", pur, "--sortie", fxg, "--fils", "1"])
    for e in V.ESPACES:
        rp = os.path.join(camp, "rapports", f"g3_lmin_{e}.json"); rapport(rp, 1e-13)
        conf = fxg if e == "kerB" else pur
        manif(camp, f"g3_lmin_{e}", [conf, kvu], rp, P.lmin_argv(kvu, e, conf, "8", rp, P.GRAINE_LOBPCG, P.BLOC))
    for i in P.MUT_FONDS:
        x = ids[i]; fixe = os.path.join(camp, "travail", i + "_fixe.npy")
        r0 = json.load(open(os.path.join(camp, "rapports", f"lmin_kerB_{i}.json")))
        rp = os.path.join(camp, "rapports", f"mut_kerB_{i}.json"); rapport(rp, r0["eps_prime"] + 0.5)
        manif(camp, f"mut_kerB_{i}", [fixe, kvu], rp, P.lmin_argv(kvu, "kerB", fixe, f"{x['beta']:g}", rp, P.GRAINE_LOBPCG, P.BLOC) + ["--mutant", "M-b"])
    wp13 = os.path.join(camp, "VERDICT_WP13.txt"); ecrire(wp13, b"VERDICT WP13 VERT : 59 PASS / 0 FAIL-SKIP")
    manif(camp, "instr", [kvu, LANCER], wp13, ["env", "KVU_BIN=" + kvu, "bash", LANCER, os.path.join(camp, "instr_out")])
    f2 = []
    for g_ in S.GRAINES:
        for b in V.BETAS_S2:
            for r in range(S.NF):
                sid = S.ident(g_, b, r)
                ch = os.path.join(camp, "s2", "fonds", sid + ".npy"); ecrire(ch, sid.encode())
                el = 0.25 * E.g_de_beta(b) ** 1.02 * math.exp(0.01 * rng.standard_normal())
                f2.append({"id": sid, "graine": g_, "beta": float(b), "fond": r, "chemin": ch, "sha256": sha(ch), "eps_lat": el, "eps_s2_enregistre": el})
                rp = os.path.join(camp, "s2", "rapports", f"lmin_kerB_{sid}.json"); rapport(rp, el)
                manif(camp, "s2_" + sid, [ch, kvu], rp, [kvu, "hess", "lmin", "--config", ch, "--espace", "kerB", "--coord", "additive", "--reference", "q0", "--beta", f"{float(b):g}",
                                                        "--graine", "1", "--tol", "1e-10", "--fils", "1", "--sortie", rp], dossier=os.path.join("s2", "manifestes"))
    f2p = os.path.join(camp, "s2", "fonds_s2.json")
    json.dump({"lat_sha256": V.SHA_LAT, "fonds": f2}, open(f2p, "w"))
    manif(camp, "s2_generer", [SCRIPT_S2], f2p, [PY, "-I", "-B", SCRIPT_S2, "generer", os.path.join(camp, "s2")])
    return kvu, wp13, fj


def lancer(camp, kvu, wp13, fj):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        V.verdict(camp, fj, SHA, sha(kvu), wp13)
    sortie = buf.getvalue()
    return [l for l in sortie.splitlines() if l.startswith("VERDICT WP14")][0], sortie


def main():
    ok = True
    def rapporter(bon, texte):
        nonlocal ok
        ok &= bon; print("PASS" if bon else "FAIL", texte)
    for alpha, attendu in ((1.0, "COMME-S2"), (2.0, "DIFFERENT")):
        camp = tempfile.mkdtemp(prefix="wp14test_")
        kvu, wp13, fj = fabriquer(camp, alpha, np.random.default_rng(3))
        v, sortie = lancer(camp, kvu, wp13, fj)
        rapporter(f"VERDICT WP14 {attendu}" in v, f"synthétique α={alpha} : {v[:90]}")
        if f"VERDICT WP14 {attendu}" not in v:
            print(sortie)
        shutil.rmtree(camp)
    def essai(nom, mod, attendu="INDETERMINE"):
        c = tempfile.mkdtemp(prefix="wp14test_"); k, w, fj = fabriquer(c, 1.0, np.random.default_rng(3))
        x8 = [x for x in json.load(open(fj))["fonds"] if x["beta_dir"] == "8"][0]
        res = mod(c, k, x8, fj)
        fj = res if isinstance(res, str) else fj
        v2, _ = lancer(c, k, w, fj)
        rapporter(attendu in v2, f"{nom} → {v2[14:75]}")
        shutil.rmtree(c)
    def m_rapport(c, k, x, fj):
        rapport(os.path.join(c, "rapports", f"lmin_tcov_{x['id']}.json"), 0.01)
    def m_landau(c, k, x, fj):
        os.remove(os.path.join(c, "manifestes", f"landau_{x['id']}.json"))
    def m_mut(c, k, x, fj):
        p = os.path.join(c, "manifestes", "mut_kerB_b8_c1_000600.json"); m = json.load(open(p)); m["commande"] = m["commande"][:-2]; json.dump(m, open(p, "w"))
    def m_fixe(c, k, x, fj):
        ecrire(os.path.join(c, "travail", x["id"] + "_fixe.npy"), b"autre")
    def m_reprise(c, k, x, fj):
        p = os.path.join(c, "manifestes", f"lmin_tcov_{x['id']}.json"); os.replace(p, os.path.join(c, "manifestes", f"lmin_tcov_{x['id']}.a2.json"))
        json.dump({"statut": "TERMINE", "rc": 1}, open(p, "w"))
    def m_liste(c, k, x, fj):
        open(fj, "a").write(" ")
    def m_doublons(c, k, x, fj):
        d = json.load(open(fj)); d["fonds"] = [d["fonds"][i % 3] for i in range(36)]
        n = fj + ".dup"; json.dump(d, open(n, "w"), sort_keys=True); V.SHA_FONDS = sha(n); return n
    def m_s2(c, k, x, fj):
        p = os.path.join(c, "s2", "fonds_s2.json"); d = json.load(open(p)); d["fonds"] = d["fonds"][:1]; json.dump(d, open(p, "w"))
    def m_beta(c, k, x, fj):
        p = os.path.join(c, "manifestes", f"lmin_kerB_{x['id']}.json"); m = json.load(open(p)); cmd = m["commande"]; cmd[cmd.index("--beta") + 1] = "2.3"; json.dump(m, open(p, "w"))
    def m_mutconv(c, k, x, fj):
        p = os.path.join(c, "rapports", "mut_kerB_b8_c1_000600.json"); rapport(p, json.load(open(p))["eps_prime"], conv=False)
        q = os.path.join(c, "manifestes", "mut_kerB_b8_c1_000600.json"); m = json.load(open(q)); m["sorties"]["fichiers"] = [ent(p)]; json.dump(m, open(q, "w"))
    def m_g3(c, k, x, fj):
        os.remove(os.path.join(c, "manifestes", "g3_fond.json"))
    def m_fils(c, k, x, fj):
        p = os.path.join(c, "manifestes", f"lmin_tcov_{x['id']}.json"); m = json.load(open(p)); cmd = m["commande"]; cmd[cmd.index("--fils") + 1] = "12"; json.dump(m, open(p, "w"))
    for nom, mod in (("falsification rapport périmé", m_rapport), ("falsification Landau absent", m_landau), ("falsification mutant sans --mutant", m_mut), ("falsification fond fixé remplacé", m_fixe),
                     ("falsification reprise .a2 après calcul démarré", m_reprise), ("falsification liste de fonds modifiée", m_liste), ("falsification liste à doublons", m_doublons),
                     ("falsification S2 réduit à un fond", m_s2), ("falsification mauvais β dans la commande", m_beta), ("falsification mutant non convergé", m_mutconv),
                     ("falsification g3_fond sans manifeste", m_g3), ("falsification mauvais --fils", m_fils)):
        essai(nom, mod)
    def m_a2(c, k, x, fj):
        p = os.path.join(c, "manifestes", f"lmin_tcov_{x['id']}.json"); os.replace(p, os.path.join(c, "manifestes", f"lmin_tcov_{x['id']}.a2.json"))
        json.dump({"statut": "REFUSE", "rc": 75}, open(p, "w"))
    essai("refus puis .a2 réussi", m_a2, "COMME-S2")
    print("TEST WP14-VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
