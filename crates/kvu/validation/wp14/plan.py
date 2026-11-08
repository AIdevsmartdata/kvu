"""WP14 : plan de calcul. Écrit un fichier de travaux (TSV, format de driver.py) à partir de la liste de fonds, SANS rien lancer.
Toutes les constantes ci-dessous sont celles du PREREG (WP14_PREREG.md §3) ; le sha256 de ce fichier est consigné au PREREG.

Usage (python3 -I -B) :
    plan.py SECTION CAMPAGNE_DIR FONDS.json KVU TRAVAUX.tsv        SECTION ∈ {campagne, stabilite, mut, g3, s2}
      campagne  : un travail `lmin_<espace>_<fond>` par (espace, fond) du plan (précédé de `landau_<fond>` pour kerB) ;
      stabilite : mêmes calculs avec graine 2 et bloc 6 sur les fonds de STAB_FONDS (travaux `stab_<espace>_<fond>`) ;
      g3        : fond de jauge pure L = 12 (g3_l12.py) puis landau + lmin kerB et lmin tcov (travaux `g3_*`) ;
      s2        : (lecture de CAMPAGNE_DIR/s2/fonds_s2.json, FONDS.json ignoré) les 112 calculs du contrôle positif.
"""
import json
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))

# ---- constantes du PREREG ---------------------------------------------------------------------------------------------------------------
ESPACES = {   # espace -> (coordonnées, référence, entrée : fond brut ou fond fixé en jauge de Landau)
    "tcov": ("gauche", "qcov", "brut"),
    "kerB": ("additive", "q0", "fixe"),
}
PLAN_BETAS = {   # espace -> β calculés ; les β de l'ajustement (4, 8, 16) d'abord
    "tcov": ["4", "8", "16", "2.8", "2.5", "2.3"],
    "kerB": ["4", "8", "16", "2.8", "2.5", "2.3"],
}
TOL = "1e-10"
ITER_MAX = "3000"
BLOC = "4"
GRAINE_LOBPCG = "1"
DUREE_MAX_S = 21600          # 6 h par travail ; au-delà : rc 124, consigné, non relancé
MEM_MAX = "6G"
STAB_FONDS = ["c1_000600"]   # un fond par β (suffixe de l'identifiant)
STAB_GRAINE, STAB_BLOC = "2", "6"
STAB_BETAS = ["4", "8", "16"]
# ----------------------------------------------------------------------------------------------------------------------------------------


def ligne(nom, apres, duree, mem, entrees, sorties, argv):
    return "\t".join([nom, ",".join(apres) if apres else "-", str(duree), mem, ";".join(entrees) or "-", ";".join(sorties) or "-", " ".join(str(x) for x in argv)])


def lmin_argv(kvu, espace, config, beta, rapport, graine, bloc):
    coord, ref, _ = ESPACES[espace]
    return [kvu, "hess", "lmin", "--config", config, "--espace", espace, "--coord", coord, "--reference", ref, "--beta", beta, "--graine", graine,
            "--tol", TOL, "--iter-max", ITER_MAX, "--bloc", bloc, "--fils", "1", "--sortie", rapport]


def travaux_campagne(camp, fonds, kvu, stabilite):
    sortie = []
    ordre = {b: i for i, b in enumerate(["4", "8", "16", "2.8", "2.5", "2.3"])}
    for espace, betas in PLAN_BETAS.items():
        coord, ref, sur = ESPACES[espace]
        for beta in sorted(betas, key=lambda b: ordre[b]):
            for x in fonds:
                if x["beta_dir"] != beta:
                    continue
                if stabilite and not any(x["id"].endswith(s) for s in STAB_FONDS):
                    continue
                if stabilite and beta not in STAB_BETAS:
                    continue
                fid = x["id"]
                pre = "stab" if stabilite else "lmin"
                apres, config, entrees = [], x["chemin"], [x["chemin"], kvu]
                if sur == "fixe":
                    fixe = os.path.join(camp, "travail", fid + "_fixe.npy")
                    land = "landau_" + fid
                    if not stabilite:
                        sortie.append(ligne(land, [], 1800, "4G", [x["chemin"], kvu], [fixe], [kvu, "hess", "landau", "--config", x["chemin"], "--sortie", fixe, "--fils", "1"]))
                    apres, config, entrees = [land], fixe, [fixe, kvu]
                rap = os.path.join(camp, "rapports", f"{pre}_{espace}_{fid}.json")
                graine, bloc = (STAB_GRAINE, STAB_BLOC) if stabilite else (GRAINE_LOBPCG, BLOC)
                sortie.append(ligne(f"{pre}_{espace}_{fid}", apres if not stabilite else ([] if sur != "fixe" else ["landau_" + fid]), DUREE_MAX_S, MEM_MAX, entrees, [rap],
                                    lmin_argv(kvu, espace, config, f"{float(beta):g}", rap, graine, bloc)))
    return sortie


MUT_FONDS = ["b8_c1_000600", "b16_c1_000600"]


def travaux_mut(camp, fonds, kvu):
    """WP14-MUT : mutant compilé M-b (terme même-lien retiré) sur deux fonds MC, espace kerB ; ε′ doit s'écarter d'au moins 0,1 du calcul sain."""
    t = []
    for x in fonds:
        if x["id"] in MUT_FONDS:
            fixe = os.path.join(camp, "travail", x["id"] + "_fixe.npy")
            rap = os.path.join(camp, "rapports", "mut_kerB_" + x["id"] + ".json")
            argv = lmin_argv(kvu, "kerB", fixe, f"{x['beta']:g}", rap, GRAINE_LOBPCG, BLOC) + ["--mutant", "M-b"]
            t.append(ligne("mut_kerB_" + x["id"], ["landau_" + x["id"]], DUREE_MAX_S, MEM_MAX, [fixe, kvu], [rap], argv))
    return t


def travaux_g3(camp, kvu):
    py = sys.executable
    pur = os.path.join(camp, "travail", "g3_pur.npy")
    fixe = os.path.join(camp, "travail", "g3_pur_fixe.npy")
    t = [ligne("g3_fond", [], 600, "4G", [os.path.join(ICI, "g3_l12.py")], [pur], [py, "-I", "-B", os.path.join(ICI, "g3_l12.py"), pur]),
         ligne("g3_landau", ["g3_fond"], 1800, "4G", [pur, kvu], [fixe], [kvu, "hess", "landau", "--config", pur, "--sortie", fixe, "--fils", "1"])]
    for espace in ("kerB", "tcov"):
        coord, ref, sur = ESPACES[espace]
        rap = os.path.join(camp, "rapports", f"g3_lmin_{espace}.json")
        config, apres, entrees = (fixe, ["g3_landau"], [fixe, kvu]) if sur == "fixe" else (pur, ["g3_fond"], [pur, kvu])
        t.append(ligne(f"g3_lmin_{espace}", apres, DUREE_MAX_S, MEM_MAX, entrees, [rap], lmin_argv(kvu, espace, config, "8", rap, GRAINE_LOBPCG, BLOC)))
    return t


def travaux_s2(camp, kvu):
    doc = json.load(open(os.path.join(camp, "s2", "fonds_s2.json")))
    t = []
    for x in doc["fonds"]:
        rap = os.path.join(camp, "s2", "rapports", "lmin_kerB_" + x["id"] + ".json")
        argv = [kvu, "hess", "lmin", "--config", x["chemin"], "--espace", "kerB", "--coord", "additive", "--reference", "q0", "--beta", f"{x['beta']:g}",
                "--graine", "1", "--tol", "1e-10", "--fils", "1", "--sortie", rap]
        t.append(ligne("s2_" + x["id"], [], 3600, "4G", [x["chemin"], kvu], [rap], argv))
    return t


def main():
    if len(sys.argv) != 6 or sys.argv[1] not in ("campagne", "stabilite", "mut", "g3", "s2"):
        print(__doc__)
        return 2
    section, camp, fonds_json, kvu, sortie = sys.argv[1:]
    camp, kvu = os.path.abspath(camp), os.path.abspath(kvu)
    if os.path.exists(sortie):
        print(f"plan : {sortie} existe (jamais écrasé)", file=sys.stderr)
        return 3
    if section in ("campagne", "stabilite"):
        fonds = json.load(open(fonds_json))["fonds"]
        lignes = travaux_campagne(camp, fonds, kvu, section == "stabilite")
    elif section == "mut":
        lignes = travaux_mut(camp, json.load(open(fonds_json))["fonds"], kvu)
    elif section == "g3":
        lignes = travaux_g3(camp, kvu)
    else:
        lignes = travaux_s2(camp, kvu)
    with open(sortie, "x", encoding="utf-8") as f:
        f.write("# plan WP14 section " + section + " : nom, apres, duree_max_s, mem_max, entrees, sorties, commande\n")
        f.write("\n".join(lignes) + "\n")
    print(f"plan : {len(lignes)} travaux ({section}) écrits dans {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
