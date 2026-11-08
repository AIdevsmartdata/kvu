"""WP14 : étapes de la campagne, UNE exécution du lanceur chacune (le lanceur tient le verrou machine ; les travaux internes passent par driver.py → executer.py, un manifeste chacun).
python3 -I -B etape.py ETAPE CAMP --executer E --depot D --sha SHA --kvu KVU [--creneaux 12]
  controle : liste des fonds (sha256 vérifié contre le PREREG), régénération du contrôle S2, 112 calculs S2, rejeu des gates de WP13 (INSTR), comparaison S2 ;
  campagne : plan campagne + stabilité + mutant + G3 puis driver.
Ne décide d'aucun verdict : verdict.py lit les fichiers."""
import argparse, os, subprocess, sys

ICI = os.path.dirname(os.path.abspath(__file__))
SHA_FONDS = "19b584e6006b8a67a16f39053079bb3202e3894e755c77f6e4da7bc0468f6d04"
PY = sys.executable


def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print("ETAPE ECHEC", cmd[:6], "rc", r.returncode, flush=True)
    return r.returncode


def sha256(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ligne(nom, apres, duree, mem, entrees, sorties, argv):
    sys.path.insert(0, ICI)
    import plan
    return plan.ligne(nom, apres, duree, mem, entrees, sorties, argv)


def driver(a, tsv, manifestes, creneaux):
    return subprocess.Popen([PY, "-I", "-B", os.path.join(ICI, "driver.py"), tsv, "--executer", a.executer, "--depot", a.depot, "--sha", a.sha,
                             "--manifestes", manifestes, "--creneaux", str(creneaux)])


def ecrire(chemin, lignes):
    with open(chemin, "x") as f:
        f.write("\n".join(lignes) + "\n")


def plan(section, camp, fonds, kvu, tsv):
    return run([PY, "-I", "-B", os.path.join(ICI, "plan.py"), section, camp, fonds, kvu, tsv])


def controle(a):
    camp = a.camp
    os.makedirs(os.path.join(camp, "manifestes"), exist_ok=True)
    os.makedirs(os.path.join(camp, "s2", "manifestes"), exist_ok=True)
    os.makedirs(os.path.join(camp, "s2", "rapports"), exist_ok=True)
    fl = os.path.join(camp, "fonds_liste.json")
    # phase A : liste des fonds, régénération S2
    A = os.path.join(camp, "phaseA.tsv")
    ecrire(A, [ligne("fonds_liste", [], 3600, "2G", [os.path.join(ICI, "fonds.py")], [fl], [PY, "-I", "-B", os.path.join(ICI, "fonds.py"), "ecrire", fl]),
               ligne("s2_generer", [], 14400, "8G", [os.path.join(ICI, "s2_controle.py")], [os.path.join(camp, "s2", "fonds_s2.json")],
                     [PY, "-I", "-B", os.path.join(ICI, "s2_controle.py"), "generer", os.path.join(camp, "s2")])])
    if driver(a, A, os.path.join(camp, "manifestes"), 2).wait() != 0:
        return 1
    if sha256(fl) != SHA_FONDS:
        print("ETAPE ECHEC : sha256 de la liste de fonds ≠ PREREG", sha256(fl))
        return 1
    # phase B : calculs S2 (manifestes dans s2/) en parallèle avec le rejeu de WP13
    B = os.path.join(camp, "phaseB_s2.tsv")
    if plan("s2", camp, fl, a.kvu, B) != 0:
        return 1
    C = os.path.join(camp, "phaseC_instr.tsv")
    out = os.path.join(camp, "instr_out")
    ecrire(C, [ligne("instr", [], 14400, "16G", [a.kvu, os.path.join(ICI, "..", "hess", "lancer.sh")], [os.path.join(out, "VERDICT.txt")],
                     ["env", "KVU_BIN=" + a.kvu, "bash", os.path.join(ICI, "..", "hess", "lancer.sh"), out])])
    pb = driver(a, B, os.path.join(camp, "s2", "manifestes"), max(1, a.creneaux - 4))
    pc = driver(a, C, os.path.join(camp, "manifestes"), 1)
    rb, rc_ = pb.wait(), pc.wait()
    # phase D : comparaison S2 (stdout du travail = fichier lu par verdict.py)
    D = os.path.join(camp, "phaseD.tsv")
    ecrire(D, [ligne("s2_comparer", [], 1800, "2G", [os.path.join(camp, "s2", "fonds_s2.json")], [],
                     [PY, "-I", "-B", os.path.join(ICI, "s2_controle.py"), "comparer", os.path.join(camp, "s2"), os.path.join(camp, "s2", "rapports")])])
    rd = driver(a, D, os.path.join(camp, "manifestes"), 1).wait()
    print("ETAPE controle : S2", rb, "INSTR", rc_, "comparer", rd, flush=True)
    return 0 if (rb == 0 and rc_ == 0 and rd == 0) else 1


def campagne(a):
    camp = a.camp
    os.makedirs(os.path.join(camp, "manifestes"), exist_ok=True)
    os.makedirs(os.path.join(camp, "travail"), exist_ok=True)
    os.makedirs(os.path.join(camp, "rapports"), exist_ok=True)
    fl = os.path.join(camp, "fonds_liste.json")
    if sha256(fl) != SHA_FONDS:
        print("ETAPE ECHEC : liste de fonds ≠ PREREG")
        return 1
    lignes = []
    for section in ("campagne", "stabilite", "mut", "g3"):
        t = os.path.join(camp, f"plan_{section}.tsv")
        if plan(section, camp, fl, a.kvu, t) != 0:
            return 1
        lignes += [l for l in open(t).read().splitlines() if l and not l.startswith("#")]
    T = os.path.join(camp, "travaux_campagne.tsv")
    ecrire(T, lignes)
    return driver(a, T, os.path.join(camp, "manifestes"), a.creneaux).wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etape", choices=("controle", "campagne"))
    ap.add_argument("camp")
    ap.add_argument("--executer", required=True)
    ap.add_argument("--depot", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--kvu", required=True)
    ap.add_argument("--creneaux", type=int, default=12)
    a = ap.parse_args()
    a.camp = os.path.abspath(a.camp)
    return controle(a) if a.etape == "controle" else campagne(a)


if __name__ == "__main__":
    sys.exit(main())
