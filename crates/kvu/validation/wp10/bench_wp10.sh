#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# bench_wp10.sh — mesure de ≈ 5 minutes par machine pour la RÉPARTITION des tâches de WP10 (PREREG v1 §6). Rapporté, jamais une gate.
# Chaque débit est une DIFFÉRENCE entre deux runs de longueurs différentes dans la même configuration (mise en route du GPU, compilation en cache,
# initialisation du champ s'annulent) ; les mises en route et les coûts de fenêtre sont mesurés à part. Graines 26102619xxx, sorties jetées.
#   s<L>   : mise en route d'un run (1 pas) ; g<L> : pas/s F32 `rapide` (plaquette mesurée à chaque pas comprise) ; f<L> : pas/s F64 `compat-prod`
#   b<L>   : secondes par bloc en ligne F32 (promotion, bloc F64, écriture de la configuration grossière et des CSV), runs avec et sans bloc, SANS dump
#   w54    : coût d'une fenêtre à L = 54 : run coupé après une itération (point de reprise écrit), puis reprise coupée de même
#   dump54 : un dump q64 de L = 54 (entrée de off54) ; off54 : `rg bloc --backend` hors ligne sur ce dump (temps mur, lecture du fichier comprise)
# USAGE (sous le verrou GPU, fenêtre ≤ 10 min, MemAvailable ≥ 30 Go / 24 Go sur le gamer) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock env KVU_BIN=<kvu> BACKEND=hip  OUT=<dossier neuf> bash bench_wp10.sh
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock      env KVU_BIN=<kvu> BACKEND=cuda OUT=<dossier neuf> bash bench_wp10.sh
# Variables : KVU_BIN, BACKEND, OUT (NEUF), CONF (kvu.toml de la campagne : profil de machine), DUMP (dump q64 L = 54 à donner à off54 : comparer deux machines
#   sur la même entrée), SEULEMENT_OFF=1 (ne faire que off54 sur DUMP), RAYON_NUM_THREADS (défaut 8). Sortie : lignes `BENCH …`, `bench.tsv`, `bench.json`.
set -u
export LC_ALL=C   # séparateur décimal « . » dans les temps (awk, printf)
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
[ -n "${CONF:-}" ] && export KEVINOTRON_CONFIG="$CONF"
: "${KVU_BIN:?KVU_BIN requis}" "${BACKEND:?BACKEND requis (hip|cuda)}" "${OUT:?OUT requis}"
[ -e "$OUT" ] && { echo "REFUS : $OUT existe déjà (dossier neuf requis)"; exit 2; }
mkdir -p "$OUT" || exit 2
RES="$OUT/bench.tsv"; : > "$RES"
t() { # t <nom> <commande…> : temps mur (s) du processus, rc dans $RC
  local n=$1; shift; local a=$EPOCHREALTIME
  "$@" > "$OUT/$n.log" 2>&1; RC=$?
  local b=$EPOCHREALTIME
  awk -v a="$a" -v b="$b" -v n="$n" -v rc="$RC" 'BEGIN{printf "%s %.3f %d\n", n, b-a, rc}' >> "$RES"
}
gen() { # gen <nom> <L> <précision> <therm> <nmeas> [options supplémentaires…]
  local n=$1 l=$2 p=$3 th=$4 nm=$5; shift 5
  local opt=rapide; [ "$p" = f64 ] && opt=compat-prod
  t "$n" "$KVU_BIN" su2 gen --backend "$BACKEND" --precision "$p" --options "$opt" --L "$l" --beta 8 --graine 26102619301 --start hot \
    --therm "$th" --nmeas "$nm" --skip 10 --nor 4 "$@" --outdir "$OUT/$n"
}
val() { awk -v n="$1" '$1==n{print $2}' "$RES"; }
rc_ok() { awk -v n="$1" '$1==n{print $3}' "$RES"; }
rate() { awk -v t1="$(val "$1")" -v t2="$(val "$2")" -v d="$3" 'BEGIN{ if (t2>t1) printf "%.3f", d/(t2-t1); else print "nan" }'; }
dt() { awk -v t1="$(val "$1")" -v t2="$(val "$2")" -v d="$3" 'BEGIN{ printf "%.3f", (t2-t1)/d }'; }
echo "INFO bench_wp10 $BACKEND début $(date '+%H:%M:%S') ; MemAvailable $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) Go ; charge $(cut -d' ' -f1-3 /proc/loadavg) ; profil $(KEVINOTRON_CONFIG=${KEVINOTRON_CONFIG:-} "$KVU_BIN" doctor --json 2>/dev/null | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["machine"])' 2>/dev/null)"
if [ "${SEULEMENT_OFF:-0}" != 1 ]; then
  for L in 54 36; do
    if [ "$L" = 54 ]; then n1=1; n2=101; f1=1; f2=13; nb=4; else n1=1; n2=301; f1=1; f2=61; nb=10; fi
    gen "g${L}_a" "$L" f32 "$n1" 0;  gen "g${L}_b" "$L" f32 "$n2" 0
    echo "BENCH g$L mise_en_route_s=$(val g${L}_a) pas_par_s_F32=$(rate g${L}_a g${L}_b $((n2 - n1))) rc=$(rc_ok g${L}_a)/$(rc_ok g${L}_b)"
    gen "f${L}_a" "$L" f64 "$f1" 0;  gen "f${L}_b" "$L" f64 "$f2" 0
    echo "BENCH f$L mise_en_route_s=$(val f${L}_a) pas_par_s_F64=$(rate f${L}_a f${L}_b $((f2 - f1))) rc=$(rc_ok f${L}_a)/$(rc_ok f${L}_b)"
    gen "b${L}_sans" "$L" f32 10 "$nb"
    gen "b${L}_avec" "$L" f32 10 "$nb" --mesures bloc:3:1
    echo "BENCH b$L s_par_bloc_en_ligne=$(dt b${L}_sans b${L}_avec $nb) (t_sans=$(val b${L}_sans) s, t_avec=$(val b${L}_avec) s, $nb mesures) rc=$(rc_ok b${L}_sans)/$(rc_ok b${L}_avec)"
    rm -rf "$OUT/g${L}_a" "$OUT/g${L}_b" "$OUT/f${L}_a" "$OUT/f${L}_b" "$OUT/b${L}_sans"
  done
  # coût d'une fenêtre à L = 54 : départ coupé après une itération, puis reprise coupée de même (deux points de reprise de 1,09 Go)
  gen w54_depart 54 f32 5 5 --reprise-every 0 --mesures bloc:3:1 --fenetre-s 0.01
  t w54_reprise "$KVU_BIN" su2 gen --backend "$BACKEND" --reprendre "$OUT/w54_depart" --mesures bloc:3:1 --fenetre-s 0.01
  echo "BENCH w54 depart_coupe_s=$(val w54_depart) reprise_coupee_s=$(val w54_reprise) (une itération chacun ; = mise en route + point de reprise de 1,09 Go (+ relecture pour la reprise)) rc=$(rc_ok w54_depart)/$(rc_ok w54_reprise)"
  rm -rf "$OUT/w54_depart" "$OUT/b54_avec"
  gen dump54 54 f32 2 1 --skip 1 --dump-every 1 --format q64
fi
D=${DUMP:-$OUT/dump54/cfg_000001.npy}
if [ -f "$D" ]; then
  t off54 "$KVU_BIN" rg bloc --backend "$BACKEND" --entree "$D" --sortie "$OUT/off54_sortie.npy"
  echo "BENCH off54 s_mur_rg_bloc=$(val off54) rc=$(rc_ok off54) entree=$D ($(stat -c %s "$D") octets)"
else echo "BENCH off54 absent (dump $D introuvable)"; fi
python3 -I - "$RES" "$BACKEND" > "$OUT/bench.json" <<'PY'
import json, sys
lignes = [l.split() for l in open(sys.argv[1])]
print(json.dumps({"backend": sys.argv[2], "temps_mur_s": {n: float(t) for n, t, rc in lignes}, "rc": {n: int(rc) for n, t, rc in lignes}}, indent=1))
PY
echo "INFO bench_wp10 $BACKEND fin $(date '+%H:%M:%S') ; charge $(cut -d' ' -f1-3 /proc/loadavg)"
