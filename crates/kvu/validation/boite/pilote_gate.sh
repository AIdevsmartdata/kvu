#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pilote_gate.sh — gate G-DIR-PILOTE (GATES-DIR.md §3 D), parties (c) exécution réelle sur les deux GPU et (d) verrou par GPU.
# Les parties (a) plan déterministe et (b) refus d'un lot trop long sont des tests de `tests/boite_pilote.rs` (cargo test), comme
# les tests du verrou déclaré (`--verrou`). Passe 2 (25/10, après le FAIL de (c) et (d)-gamer de la passe 1 : rapatriement rsync et
# verrou du gamer, deux défauts du pilote corrigés) : les deux cibles déclarent leur verrou ; (d) teste acemagic sans déclaration
# (le `kvu` local prend lui-même son verrou), acemagic avec déclaration (enveloppe `flock`, kvu voit « ancetre ») et gamer avec déclaration.
#
# USAGE (depuis acemagic, SANS verrou GPU tenu par l'appelant : le pilote et `kvu` prennent chacun le verrou de leur machine) :
#   KVU_ACE=<kvu acemagic> KVU_GAMER=<chemin du kvu sur le gamer> HOTE_GAMER=<alias ssh> SORTIE=<dossier neuf> bash pilote_gate.sh
# Sortie : lignes `PASS|FAIL|INFO G-DIR-PILOTE… : détail` ; code de sortie = nombre de FAIL.
set -u
ICI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACINE=$(cd "$ICI/../../../.." && pwd)
: "${KVU_ACE:?KVU_ACE requis}" "${KVU_GAMER:?KVU_GAMER requis}" "${HOTE_GAMER:?HOTE_GAMER requis}" "${SORTIE:?SORTIE requis}"
VERROU_ACE=${VERROU_ACE:-/tmp/kevinotron_gpu.lock}
VERROU_GAMER=${VERROU_GAMER:-/tmp/gamer_gpu.lock}
mkdir -p "$SORTIE" || exit 2
V="$SORTIE/verdicts_pilote.txt"; : > "$V"
NFAIL=0
verdict() { echo "$1 $2 : $3" | tee -a "$V"; [ "$1" = FAIL ] && NFAIL=$((NFAIL + 1)); }
PILOTE="python3 -I $RACINE/scripts/pilote_boites.py"

# ---- (c) quatre tâches, deux cellules (ℓ = 4, β = 2,5 et 3,0) × deux bords, sur les deux GPU en parallèle -------------------------
cat > "$SORTIE/taches_c.csv" <<'EOF'
ell,beta,bord,graine,therm,nmeas
4,2.5,cold,26102610501,20,200
4,2.5,z2full,26102610502,20,200
4,3.0,cold,26102610503,20,200
4,3.0,z2full,26102610504,20,200
EOF
t0=$(date +%s)
$PILOTE "$SORTIE/taches_c.csv" --sortie "$SORTIE/c" --cible "acemagic=local:$KVU_ACE" --cible "gamer=ssh:$HOTE_GAMER:$KVU_GAMER" \
  --verrou "acemagic=$VERROU_ACE" --verrou "gamer=$VERROU_GAMER" \
  --rep-distant "~/kvu_pilote_gate" > "$SORTIE/c.out" 2> "$SORTIE/c.err"
rc=$?
dt=$(( $(date +%s) - t0 ))
python3 -I - "$SORTIE/c/index.json" "$rc" "$dt" "$VERROU_ACE" "$VERROU_GAMER" <<'PYEOF' | tee -a "$V"
import json, sys
idx = json.load(open(sys.argv[1]))
rc, dt = int(sys.argv[2]), sys.argv[3]
declares = {"acemagic": sys.argv[4], "gamer": sys.argv[5]}
lots = idx["lots"]
par_cible = {}
for l in lots:
    par_cible.setdefault(l["cible"], []).append(l)
ok = rc == 0 and all(l["statut"] == "ok" and l["rc"] == 0 for l in lots)
deux = set(par_cible) == {"acemagic", "gamer"}
verrous = [l["verrou"] for l in lots]
verrou_ok = all(v in ("pris_ici", "ancetre") for v in verrous)
taches = idx["taches"]
coherent = len(taches) == 4 and all(any(l["id"] == t["lot"] for l in lots) for t in taches)
verrous_index = {k: v.get("verrou") for k, v in idx["cibles"].items()} == declares
res = "PASS" if (ok and deux and verrou_ok and coherent and verrous_index) else "FAIL"
print(f"{res} G-DIR-PILOTE-C : rc={rc}, {len(lots)} lots, cibles={ {k: len(v) for k, v in par_cible.items()} }, statuts={[l['statut'] for l in lots]}, verrous={verrous}, verrous déclarés dans l'index={verrous_index}, index cohérent={coherent}, durée {dt} s")
for l in lots:
    print(f"INFO G-DIR-PILOTE-C lot {l['id']} cible={l['cible']} duree_s={l['duree_s']:.1f} perf={l.get('perf')}")
PYEOF
NFAIL=$((NFAIL + $(grep -c '^FAIL' "$V")))
NFAIL=$(grep -c '^FAIL' "$V")

# ---- (d) verrou par GPU : un verrou tenu 8 s sur chaque GPU, le lot du pilote ne démarre pas avant sa libération ---------------------
tester_verrou() { # $1 = acemagic|gamer ; $2 = seul (kvu prend lui-même le verrou) | flock (verrou déclaré, enveloppe du pilote)
  local cible=$1 variante=$2 rep="$SORTIE/d_$1_$2" declaration=()
  mkdir -p "$rep"
  [ "$variante" = flock ] && [ "$cible" = acemagic ] && declaration=(--verrou "acemagic=$VERROU_ACE")  # gamer : toujours déclaré (ssh l'exige)
  cat > "$rep/taches.csv" <<EOF
ell,beta,bord,graine,therm,nmeas
2,2.5,cold,26102610510,5,20
EOF
  if [ "$cible" = acemagic ]; then
    ( flock -w 7200 "$VERROU_ACE" bash -c "date +%s.%N > '$rep/acquis.txt'; sleep 8; date +%s.%N > '$rep/libere.txt'" ) &
    local pid=$!
    local n=0; until [ -f "$rep/acquis.txt" ] || [ $n -ge 3600 ]; do sleep 0.5; n=$((n + 1)); done
    [ -f "$rep/acquis.txt" ] || { verdict FAIL G-DIR-PILOTE-D "$cible : verrou non obtenu pour le test"; return; }
    $PILOTE "$rep/taches.csv" --sortie "$rep/s" --cible "acemagic=local:$KVU_ACE" "${declaration[@]}" > "$rep/pilote.out" 2> "$rep/pilote.err"
    wait $pid
    local man="$rep/s/lots/L02_b2.5_000/manifest.json"
  else
    ssh -o BatchMode=yes "$HOTE_GAMER" "mkdir -p ~/kvu_pilote_gate/d && rm -f ~/kvu_pilote_gate/d/acquis.txt ~/kvu_pilote_gate/d/libere.txt; (flock -w 7200 $VERROU_GAMER bash -c 'date +%s.%N > ~/kvu_pilote_gate/d/acquis.txt; sleep 8; date +%s.%N > ~/kvu_pilote_gate/d/libere.txt') > /dev/null 2>&1 &"
    local n=0; until ssh -o BatchMode=yes "$HOTE_GAMER" 'test -f ~/kvu_pilote_gate/d/acquis.txt' 2>/dev/null || [ $n -ge 3600 ]; do sleep 1; n=$((n + 1)); done
    $PILOTE "$rep/taches.csv" --sortie "$rep/s" --cible "gamer=ssh:$HOTE_GAMER:$KVU_GAMER" --verrou "gamer=$VERROU_GAMER" \
      --rep-distant "~/kvu_pilote_gate/d_run" > "$rep/pilote.out" 2> "$rep/pilote.err"
    local m=0; until ssh -o BatchMode=yes "$HOTE_GAMER" 'test -f ~/kvu_pilote_gate/d/libere.txt' 2>/dev/null || [ $m -ge 60 ]; do sleep 1; m=$((m + 1)); done
    scp -q "$HOTE_GAMER:kvu_pilote_gate/d/acquis.txt" "$HOTE_GAMER:kvu_pilote_gate/d/libere.txt" "$rep/" 2>/dev/null
    local man="$rep/s/lots/L02_b2.5_000/manifest.json"
  fi
  python3 -I - "$man" "$rep/acquis.txt" "$rep/libere.txt" "$cible/$variante" <<'PYEOF' | tee -a "$V"
import json, sys
man, acq, lib, cible = sys.argv[1:5]
try:
    m = json.load(open(man))
    debut = float(m["debut_unix"])
    acquis, libere = float(open(acq).read()), float(open(lib).read())
except Exception as e:
    print(f"FAIL G-DIR-PILOTE-D : {cible} : lecture impossible ({e})")
    sys.exit(0)
attente = debut - acquis
ok = debut >= libere - 0.05
print(f"{'PASS' if ok else 'FAIL'} G-DIR-PILOTE-D : {cible} : verrou tenu de {acquis:.2f} à {libere:.2f}, le lot a démarré à {debut:.2f} (+{debut - libere:.2f} s après la libération, verrou = {m.get('verrou')})")
PYEOF
  NFAIL=$(grep -c '^FAIL' "$V")
}
tester_verrou acemagic seul
tester_verrou acemagic flock
tester_verrou gamer flock
echo "# fin : $NFAIL FAIL (verdicts : $V)"
exit "$NFAIL"
