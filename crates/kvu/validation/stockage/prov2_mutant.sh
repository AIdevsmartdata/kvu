#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# PROV-2 (mutant, GATES v1 §1/WP1) : la provenance voit une modification de noyau.
# Un run CPU (L = 4) du binaire KVU_BIN écrit son manifeste ; on copie le dépôt (sans .git ni target) dans le scratch,
# on modifie UN octet de crates/kvu/noyaux/su2_commun.h DANS LA COPIE (jamais dans le dépôt), puis on exige :
#   - témoin : sur la copie intacte, le recalcul Python de kvu-source/1 = source_hash du manifeste (comparaison rc 0) ;
#   - mutant : sur la copie modifiée, le recalcul diffère du source_hash du manifeste (comparaison rc ≠ 0) ;
#   - mutant : le sha256 du su2_commun.h modifié diffère de l'entrée `su2_commun.h` de `noyaux_embarques` du manifeste
#     (qui, elle, égale le sha256 du fichier du dépôt).
# Variables : KVU_BIN (défaut kvu), KVU_RACINE (défaut : racine du dépôt), KVU_SCRATCH (obligatoire, sur le SSD), KVU_PY.
# Sortie : une ligne « PASS PROV-2 … » ou « FAIL PROV-2 … ». Code retour 0 dans les deux cas (le verdict est la ligne).
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACINE="${KVU_RACINE:-$(cd "$ICI/../../../.." && pwd)}"
BIN="${KVU_BIN:-kvu}"
PY="${KVU_PY:-python3}"
[ -n "${KVU_SCRATCH:-}" ] || { echo "FAIL PROV-2 KVU_SCRATCH non défini"; exit 0; }
D="$KVU_SCRATCH/prov2"; rm -rf "$D"; mkdir -p "$D"
if ! "$BIN" su2 gen --backend cpu --L 4 --beta 2.3 --graine 1 --start hot --therm 2 --nmeas 2 --skip 1 --nor 4 \
      --outdir "$D/run" >"$D/run.log" 2>&1; then
  echo "FAIL PROV-2 run CPU de L = 4 en échec : $(tail -2 "$D/run.log" | tr '\n' ' ')"; exit 0
fi
M="$D/run/manifest.json"
[ -f "$M" ] || { echo "FAIL PROV-2 manifest.json absent"; exit 0; }
mkdir -p "$D/copie"
rsync -a --exclude .git --exclude target --exclude __pycache__ "$RACINE/" "$D/copie/" || { echo "FAIL PROV-2 copie du dépôt impossible"; exit 0; }
NOYAU=crates/kvu/noyaux/su2_commun.h
"$PY" -I - "$M" "$D/copie" "$ICI/../historique/source_hash_kvu.py" "$NOYAU" "$RACINE/$NOYAU" <<'PYEOF'
import hashlib, json, subprocess, sys
man = json.load(open(sys.argv[1])); copie, script, noyau, original = sys.argv[2:6]
def cle(m, nom):
    if nom in m: return m[nom]
    for v in m.values():
        if isinstance(v, dict) and nom in v: return v[nom]
    return None
def recalcul():
    o = subprocess.run([sys.executable, '-I', script, copie], capture_output=True, text=True)
    return o.stdout.splitlines()[0] if o.returncode == 0 and o.stdout else None
def comparaison(h):          # « la comparaison rend rc ≠ 0 » : 0 si identiques, 1 sinon
    return 0 if h is not None and h == cle(man, 'source_hash') else 1
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
pb = []
# témoin (copie intacte)
if comparaison(recalcul()) != 0: pb.append("témoin : recalcul de la copie intacte ≠ source_hash du manifeste")
entree = next((e['sha256'] for e in (cle(man, 'noyaux_embarques') or []) if e.get('nom') == 'su2_commun.h'), None)
if entree is None: pb.append("entrée su2_commun.h absente de noyaux_embarques")
elif entree != sha(original): pb.append("témoin : entrée su2_commun.h de noyaux_embarques ≠ sha256 du fichier du dépôt")
# mutant : un octet modifié dans la copie
chemin = copie + '/' + noyau
b = bytearray(open(chemin, 'rb').read()); b[-1] ^= 1; open(chemin, 'wb').write(bytes(b))
if comparaison(recalcul()) == 0: pb.append("mutant : source_hash inchangé après modification du noyau (comparaison rc 0)")
if entree is not None and sha(chemin) == entree: pb.append("mutant : sha256 du noyau modifié = entrée de noyaux_embarques")
if pb: print("FAIL PROV-2 " + " ; ".join(pb))
else: print("PASS PROV-2 noyau modifié d'un octet dans une copie jetable : source_hash change (comparaison rc 1) et l'entrée su2_commun.h de noyaux_embarques change ; témoin intact rc 0")
PYEOF
exit 0
