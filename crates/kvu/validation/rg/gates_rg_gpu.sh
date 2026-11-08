#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gates_rg_gpu.sh — gates GPU du bloc de Bałaban (WP7) : RG-GPU, RG-DOMAINE-GPU, RG-GPU-MUT, G-DET (module et CLI), ENLIGNE-1.
# RG-HIP-CUDA se juge ensuite entre les deux sorties : `kvu rg comparer --a <DIR_hip>/hip --b <DIR_cuda>/cuda`.
#
# USAGE (TOUJOURS sous le verrou GPU de la machine, fenêtre ≤ 10 min, ≥ 30 Go libres, 24 Go sur le gamer) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock env KVU_BIN=<kvu> BACKEND=hip  GRANDL=<…/ensembles/production> OUT=<dossier neuf> bash gates_rg_gpu.sh
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock      env KVU_BIN=<kvu> BACKEND=cuda GRANDL=<…>                     OUT=<dossier neuf> bash gates_rg_gpu.sh
# Variables : KVU_BIN, BACKEND (hip|cuda), GRANDL (racine des ensembles ; sur le gamer, une copie de L24_b8_c1/cfg_002000.npy suffit),
#   OUT (dossier NEUF), FIXT (défaut : validation/rg/fixtures à côté du script), FILS (CPU de référence, défaut 8).
# Sortie : une ligne PASS|FAIL|SKIP|INFO par gate sur la sortie standard ; code de sortie = nombre de FAIL + de SKIP.
set -u
: "${KVU_BIN:?KVU_BIN requis}" "${BACKEND:?BACKEND requis (hip|cuda)}" "${OUT:?OUT requis}"
ICI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FIXT=${FIXT:-$ICI/fixtures}
FILS=${FILS:-8}
GRANDL=${GRANDL:-}
[ -x "$KVU_BIN" ] || { echo "FAIL BUILD binaire $KVU_BIN introuvable"; exit 1; }
[ -e "$OUT" ] && { echo "REFUS : $OUT existe déjà (dossier neuf requis)"; exit 2; }
mkdir -p "$OUT" || exit 2
NB=0
verdict() { echo "$1 $2 $3"; case "$1" in FAIL|SKIP) NB=$((NB + 1)) ;; esac; }

# --- 1. batterie du binaire : RG-GPU, RG-DOMAINE-GPU, RG-GPU-MUT, G-DET (module), temps ---
ARGS=(rg gates-gpu --backend "$BACKEND" --fixtures "$FIXT" --fils "$FILS" --sortie "$OUT/sorties")
[ -n "$GRANDL" ] && ARGS+=(--grandl "$GRANDL")
"$KVU_BIN" "${ARGS[@]}" > "$OUT/gates_gpu.out" 2>&1; RC=$?
grep -E "^(PASS|FAIL|SKIP|INFO) " "$OUT/gates_gpu.out"
[ $RC -ne 0 ] && NB=$((NB + 1))
echo "INFO gates-gpu rc=$RC"

# --- 2. G-DET au niveau du CLI : deux `rg bloc --backend` ⇒ NPY et manifeste identiques à l'octet ---
L24=""; [ -n "$GRANDL" ] && L24="$GRANDL/L24_b8_c1/cfg_002000.npy"
for spec in "fixture_s7:$FIXT/cfg_s7_q64.npy" "L24:$L24"; do
  nom=${spec%%:*}; f=${spec#*:}
  if [ ! -f "$f" ]; then verdict SKIP "G-DET-CLI-$nom" "entrée absente ($f)"; continue; fi
  "$KVU_BIN" rg bloc --backend "$BACKEND" --entree "$f" --sortie "$OUT/det_${nom}_1.npy" > "$OUT/det_${nom}_1.log" 2>&1; r1=$?
  "$KVU_BIN" rg bloc --backend "$BACKEND" --entree "$f" --sortie "$OUT/det_${nom}_2.npy" > "$OUT/det_${nom}_2.log" 2>&1; r2=$?
  if [ $r1 = 0 ] && [ $r2 = 0 ] && cmp -s "$OUT/det_${nom}_1.npy" "$OUT/det_${nom}_2.npy" && cmp -s "$OUT/det_${nom}_1.npy.manifeste.json" "$OUT/det_${nom}_2.npy.manifeste.json"; then
    verdict PASS "G-DET-CLI-$nom" "$BACKEND : deux runs, cmp identique (NPY et manifeste), rc $r1/$r2"
  else verdict FAIL "G-DET-CLI-$nom" "$BACKEND : rc $r1/$r2 ou cmp différent"; fi
done

# --- 3. ENLIGNE-1 : su2 gen --mesures bloc:3 (L = 12, β = 4, 200 pas, graine 26102607001) ---
# Protocole gelé au journal avant run : therm 100 + 10 mesures × skip 10 = 200 pas, nor 4, départ chaud, f64 ref-k1, dumps q64 à chaque mesure ;
# comparaison à `rg bloc` CPU (8 fils) sur les dumps du même run ≤ 1e-12 ; le run sans --mesures est comparé à l'octet (dumps et CSV).
E1="$OUT/enligne"; mkdir -p "$E1"
COMMUN=(su2 gen --backend "$BACKEND" --precision f64 --options ref-k1 --L 12 --beta 4 --graine 26102607001 --start hot --therm 100 --nmeas 10 --skip 10 --nor 4 --dump-every 1 --format q64)
"$KVU_BIN" "${COMMUN[@]}" --mesures bloc:3:1 --outdir "$E1/avec" > "$E1/avec.log" 2>&1; ra=$?
"$KVU_BIN" "${COMMUN[@]}" --outdir "$E1/sans" > "$E1/sans.log" 2>&1; rs=$?
if [ $ra != 0 ] || [ $rs != 0 ]; then
  verdict FAIL ENLIGNE-1 "rc avec=$ra sans=$rs (journaux $E1/avec.log, $E1/sans.log)"
else
  mkdir -p "$E1/A" "$E1/B"; ok=1; det=""
  for m in 1 2 3 4 5 6 7 8 9 10; do
    pas=$((100 + 10 * m)); on=$(printf "%s/avec/bloc3_cfg_%08d.npy" "$E1" "$pas"); dump=$(printf "%s/avec/cfg_%06d.npy" "$E1" "$m")
    if [ ! -f "$on" ] || [ ! -f "$dump" ]; then ok=0; det="$det m$m absent;"; continue; fi
    cp "$on" "$E1/A/m$m.npy"
    "$KVU_BIN" rg bloc --backend cpu --fils "$FILS" --entree "$dump" --sortie "$E1/B/m$m.npy" > "$E1/B/m$m.log" 2>&1 || { ok=0; det="$det rg bloc m$m rc≠0;"; }
  done
  "$KVU_BIN" rg comparer --a "$E1/A" --b "$E1/B" --min 10 --id ENLIGNE-1 --sans-theta > "$E1/comparer.out" 2>&1; rc=$?
  cat "$E1/comparer.out" | grep -E "^(PASS|FAIL) " | sed "s/^\(PASS\|FAIL\) ENLIGNE-1 /\1 ENLIGNE-1 $BACKEND : configurations grossières en ligne contre rg bloc hors ligne ; /"
  [ $rc != 0 ] && NB=$((NB + 1))
  [ $ok = 1 ] || verdict FAIL ENLIGNE-1-FICHIERS "$det"
  # run sans --mesures inchangé à l'octet (dumps et CSV) : le run « avec » ne doit rien changer à ce que le run « sans » produit
  n=0; d=""
  for f in $(cd "$E1/sans" && ls cfg_*.npy measurements.csv therm.csv); do
    if cmp -s "$E1/sans/$f" "$E1/avec/$f"; then n=$((n + 1)); else d="$d $f"; fi
  done
  tot=$(cd "$E1/sans" && ls cfg_*.npy measurements.csv therm.csv | wc -l)
  [ "$n" = "$tot" ] && verdict PASS ENLIGNE-1-INCHANGE "$BACKEND : $n/$tot fichiers identiques à l'octet avec et sans --mesures bloc:3" || verdict FAIL ENLIGNE-1-INCHANGE "$n/$tot, différents :$d"
  # statistiques de domaine en ligne (CSV) contre les manifestes hors ligne (INFO) et manifeste kvu-regen/1
  python3 -I - "$E1" > "$E1/stats.out" 2>&1 <<'PY'
import csv, json, sys
d = sys.argv[1]
rows = list(csv.DictReader(open(d + "/avec/bloc3.csv")))
mx = 0.0; amb = 0
for r in rows:
    m = (int(r["pas"]) - 100) // 10
    t = json.load(open(f"{d}/B/m{m}.npy.manifeste.json"))["theta"]
    mx = max(mx, abs(float(r["fraction_hors"]) - t["fraction_hors"]), abs(float(r["theta_max"]) - t["theta_max"]))
    amb += abs(int(r["n_ambigues"]) - t["n_ambigues"])
j = json.load(open(d + "/avec/regen_bloc3.json"))
print(f"INFO ENLIGNE-1-STATS {len(rows)} mesures : max |Δf|, |Δθmax| en ligne contre hors ligne = {mx:.2e} ; Δ ambiguës {amb} ; manifeste kvu-regen/1 parametres.mesures = {j['parametres']['mesures']!r}, statut {j['statut']}, {len(j['fichiers'])} fichiers")
PY
  cat "$E1/stats.out"
fi
echo "INFO gates_rg_gpu.sh $BACKEND : $NB (FAIL + SKIP)"
exit $NB
