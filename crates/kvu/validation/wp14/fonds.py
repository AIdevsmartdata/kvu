"""WP14 : liste des fonds Monte-Carlo (grandL L = 12), construite par une règle fixe, avec sha256 de chaque fichier.

Usage (python3 -I, venv épinglé /chemin/vers/projet/outils/ym4_noyaux/venv) :
    fonds.py ecrire SORTIE.json      écrit la liste (refuse d'écraser un fichier existant)
    fonds.py verifier LISTE.json     recalcule les sha256 et compare (rc 1 au moindre écart)

Règle (gelée au PREREG, écrite avant toute donnée) : pour chaque β de BETAS, les chaînes c1 et c2 (départ chaud) fournissent les dumps
600, 1200 et 1800, la chaîne c3 (départ froid) les dumps 900 et 1800, soit 8 fonds par β. Espacement minimal dans une chaîne : 600 pas
(≥ 2 τ_Pol,max = 289 pas à L = 12, RAPPORT_GPU_GRANDL.md T3) ; τ_int(plaquette) ≤ 2,3 pas (critère C4 ⩾ 5 τ_int, vérifié par la source).
Les dumps 100 (c1) servent aux essais de coût : ils ne sont dans aucune liste de campagne.
Aucune lecture de ε′, de plaquette ni d'aucun résultat n'intervient dans le choix."""
import hashlib
import json
import os
import sys

GRANDL = "/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production"
L = 12
BETAS = ["2.3", "2.5", "2.8", "4", "8", "16"]
DUMPS = {1: [600, 1200, 1800], 2: [600, 1200, 1800], 3: [900, 1800]}      # β d'ajustement (4, 8, 16) : 8 fonds
DUMPS_BAS = {1: [600, 1800], 2: [600, 1800]}                              # β rapportés à part (2,3 ; 2,5 ; 2,8) : 4 fonds (coût mesuré : ~40 min-cœur par calcul)
BETAS_BAS = ("2.3", "2.5", "2.8")
BANC = {"chaine": 1, "dump": 100}   # fonds d'essai de coût, hors campagne


def sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def construire():
    liste = []
    for beta in BETAS:
        for chaine, dumps in (DUMPS_BAS if beta in BETAS_BAS else DUMPS).items():
            dossier = os.path.join(GRANDL, f"L{L}_b{beta}_c{chaine}")
            if not os.path.exists(os.path.join(dossier, "COMPLETE")):
                raise SystemExit(f"fonds : {dossier} sans marqueur COMPLETE")
            man = json.load(open(os.path.join(dossier, "manifest.json")))
            assert man["l"] == L and abs(float(man["beta"]) - float(beta)) < 1e-12, (dossier, man["l"], man["beta"])
            for d in dumps:
                assert not (chaine == BANC["chaine"] and d == BANC["dump"])
                chemin = os.path.join(dossier, f"cfg_{d:06d}.npy")
                liste.append({
                    "id": f"b{beta}_c{chaine}_{d:06d}", "beta": float(beta), "beta_dir": beta, "chaine": chaine, "dump": d,
                    "chemin": chemin, "octets": os.path.getsize(chemin), "sha256": sha256(chemin),
                    "graine_chaine": man["seed"], "depart": man["start"],
                })
    return liste


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("ecrire", "verifier"):
        print(__doc__)
        return 2
    cmd, chemin = sys.argv[1], sys.argv[2]
    if cmd == "ecrire":
        if os.path.exists(chemin):
            print(f"fonds : {chemin} existe (jamais écrasé)", file=sys.stderr)
            return 3
        liste = construire()
        assert len(liste) == 8 * 3 + 4 * 3 and len({x["id"] for x in liste}) == len(liste)
        with open(chemin, "x", encoding="utf-8") as f:
            json.dump({"schema": "wp14-fonds/1", "L": L, "regle": "voir docstring de fonds.py", "fonds": liste}, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"fonds : {len(liste)} fonds écrits dans {chemin}")
        return 0
    doc = json.load(open(chemin))
    ecarts = 0
    for x in doc["fonds"]:
        if not os.path.exists(x["chemin"]) or sha256(x["chemin"]) != x["sha256"] or os.path.getsize(x["chemin"]) != x["octets"]:
            print(f"ECART {x['id']}")
            ecarts += 1
    print(f"fonds : {len(doc['fonds'])} fonds contrôlés, {ecarts} écart(s)")
    return 1 if ecarts else 0


if __name__ == "__main__":
    sys.exit(main())
