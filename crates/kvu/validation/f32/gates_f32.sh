#!/usr/bin/env bash
# Gates du chemin FP32 rapide (tâche T4) : LOI-0 à LOI-4, LOI-τ, LOT-1-F32, REPRISE-1-F32, banc.
#
# Usage : gates_f32.sh [tout|gen|juge|f64|bench|nettoyer] [--gamer-dir DIR]
#   gen       génère les sorties F32 de CETTE machine (fenêtres GPU de 10 minutes au plus, verrou rendu entre elles) ;
#   f64       génère le banc FP64 (su2_hip / su2_cuda de reference/, rebâtis ici) et, sur acemagic, les comparateurs FP64 de LOI-2 à L = 16 ;
#   juge      (acemagic) juge les sorties (Python du venv ym4_noyaux) : PASS/FAIL de chaque gate, rc 0 si tout PASS ;
#   bench     banc F32 contre FP64 (liens/s, non gaté) ;
#   tout      gen + f64 (acemagic) + juge. Sur le gamer : `gen` seulement, puis rsync vers acemagic et `juge --gamer-dir`.
#   nettoyer  supprime le répertoire de travail (scratch) de cette machine.
#
# VERROU GPU : le script prend lui-même le verrou de la machine (acemagic /tmp/kevinotron_gpu.lock, gamer
# /tmp/gamer_gpu.lock) PAR FENÊTRE (`flock -w 7200 … timeout 600`). Lancé sous un flock externe
# (`flock -w 7200 /tmp/kevinotron_gpu.lock bash gates_f32.sh`), il détecte le verrou tenu par un ancêtre
# (/proc/locks) et ne le reprend pas (pas d'interblocage) ; il tiendra alors le GPU pendant tout le script, ce qui
# dépasse la fenêtre de 10 minutes : préférer le lancement direct.
#
# Garde SSD : avant CHAQUE fenêtre et chaque build, `df` ; sous le seuil (acemagic 110 Go, gamer 50 Go) : arrêt, rc 7.
set -u -o pipefail

MODE="${1:-tout}"
shift || true
GAMER_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gamer-dir) GAMER_DIR="$2"; shift 2 ;;
    *) echo "option inconnue : $1" >&2; exit 2 ;;
  esac
done

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACINE="$(cd "$ICI/../../../.." && pwd)"           # racine du dépôt (worktree)
export PYTHONDONTWRITEBYTECODE=1

# ---- machine
H="$(hostname)"
case "$H" in
  *acemagic*) MACHINE=acemagic ;;
  *pc-nvidia-exemple*) MACHINE=gamer ;;
  *) MACHINE=inconnue ;;
esac
if [ "$MACHINE" = gamer ]; then
  LOCK=/tmp/gamer_gpu.lock; SSD_MIN=50; FEATURES="--features cuda"; JOBS=6
  PREFIXE="nice -n 10 taskset -c 12-19"
  export LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
  CARGO="$HOME/.cargo/bin/cargo"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/kvu_unif/cibles/T4}"
  SCRATCH="${KVU_F32_SCRATCH:-$HOME/kvu_unif/scratch/f32}"
else
  LOCK=/tmp/kevinotron_gpu.lock; SSD_MIN=110; FEATURES=""; JOBS=8
  PREFIXE="nice -n 10"
  CARGO=cargo
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/chemin/vers/projet/chantiers/kevinotron_unification/cibles/T4}"
  SCRATCH="${KVU_F32_SCRATCH:-/chemin/vers/projet/chantiers/kevinotron_unification/scratch/f32}"
fi
PY="${PY:-/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python}"
[ -x "$PY" ] || PY=python3
export KVU_LOI_OUT="$SCRATCH/$MACHINE"
export KVU_F32_SSD_MIN_GO="$SSD_MIN"
F64DIR="$SCRATCH/f64"                              # comparateurs FP64 (acemagic)
FENETRE_S=600                                      # une fenêtre GPU : 10 minutes au plus
NB_FAIL=0

journal() { echo "[gates_f32 $(date +%H:%M:%S)] $*"; }

garde_ssd() {
  local libre
  libre=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${libre:-0}" -lt "$SSD_MIN" ]; then
    journal "GARDE SSD : ${libre} Go libres < ${SSD_MIN} Go : ARRÊT (rc 7)"; exit 7
  fi
}

verrou_tenu_par_ancetre() {
  local ino pids p q
  ino=$(stat -c %i "$LOCK" 2>/dev/null) || return 1
  pids=$(awk -v ino="$ino" '$2=="FLOCK" { split($6, a, ":"); if (a[3] == ino) print $5 }' /proc/locks)
  p=$$
  while [ "$p" -gt 1 ]; do
    for q in $pids; do [ "$q" = "$p" ] && return 0; done
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break
  done
  return 1
}

# fenetre NOM CMD... : exécute CMD sous le verrou GPU, 10 minutes au plus, SSD gardé avant.
fenetre() {
  local nom="$1"; shift
  garde_ssd
  journal "fenêtre GPU « $nom » (verrou $LOCK)"
  local t0=$SECONDS rc
  if verrou_tenu_par_ancetre; then
    timeout "$FENETRE_S" env KVU_F32_SOUS_VERROU=1 "$@"; rc=$?
  else
    flock -w 7200 "$LOCK" timeout "$FENETRE_S" env KVU_F32_SOUS_VERROU=1 "$@"; rc=$?
  fi
  journal "fenêtre « $nom » : rc=$rc en $((SECONDS - t0)) s (verrou rendu)"
  if [ "$rc" -eq 124 ]; then journal "FAIL FENETRE-$nom : plus de ${FENETRE_S} s, découper le lot"; fi
  return "$rc"
}

# ---- build du binaire de test (hors verrou)
construire() {
  garde_ssd
  journal "build des tests (cible $CARGO_TARGET_DIR)"
  ( cd "$RACINE" && $PREFIXE "$CARGO" test -p kvu --release -j "$JOBS" $FEATURES --test su2_f32_gpu --no-run --message-format=json 2>"$SCRATCH/build.err" ) \
    | "$PY" -c "
import sys, json
exe = None
for l in sys.stdin:
    try: m = json.loads(l)
    except Exception: continue
    if m.get('reason') == 'compiler-artifact' and m.get('target', {}).get('name') == 'su2_f32_gpu' and m.get('executable'):
        exe = m['executable']
print(exe or '')
" > "$SCRATCH/exe.txt"
  EXE="$(cat "$SCRATCH/exe.txt")"
  if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then journal "build en échec :"; tail -30 "$SCRATCH/build.err"; exit 1; fi
  if grep -q "^warning" "$SCRATCH/build.err"; then journal "AVERTISSEMENTS à la compilation :"; grep -A5 "^warning" "$SCRATCH/build.err" | head -20; fi
}

# Une fenêtre peut enchaîner plusieurs sous-commandes (le verrou GPU est disputé : on regroupe, 10 min au plus).
# lot_cmd L BETAS SOUS FACTEUR DUMP SEED0 : commande d'un lot de chaînes, à enchaîner dans `fenetre … bash -c`.
lot_cmd() {
  local l="$1" betas="$2" sous="$3" facteur="${4:-1}" dump="${5:-1000}" seed0="${6:-20261004000000}"
  echo "env KVU_F32_L=$l KVU_F32_BETAS=$betas KVU_F32_SOUS=$sous KVU_F32_FACTEUR_BETA=$facteur KVU_F32_DUMP_EVERY=$dump KVU_F32_SEED0=$seed0 KVU_F32_THERM=2000 KVU_F32_NMEAS=20000 KVU_F32_NOR=4 $PREFIXE $EXE --ignored --exact pilote_chaines --nocapture --test-threads=1 ;"
}

# veut N : la fenêtre N est-elle demandée ? (KVU_F32_FENETRES="3 4" relance seulement ces fenêtres ; vide = toutes)
veut() { [ -z "${KVU_F32_FENETRES:-}" ] || [[ " $KVU_F32_FENETRES " == *" $1 "* ]]; }

gen() {
  mkdir -p "$SCRATCH" "$KVU_LOI_OUT"
  construire
  if [ -z "${KVU_F32_SANS_UNITAIRES:-}" ] && veut 1; then
    journal "== fenêtre 1 : tests GPU unitaires (démarrage, mesures contre l'hôte, gel, miroir x0, compteurs, aller-retour, déterminisme, provenance, mutant)"
    fenetre unitaires $PREFIXE "$EXE" --ignored --nocapture --test-threads=1 f32_ || NB_FAIL=$((NB_FAIL + 1))
  fi
  if veut 2; then
    journal "== fenêtre 2 : LOT-1-F32, REPRISE-1-F32, LOI-0, LOI-1, LOI-4b, banc F32"
    fenetre verifs $PREFIXE "$EXE" --ignored --nocapture --test-threads=1 --exact lot_1_f32 reprise_1_f32 loi_0_pilote loi_1_pilote \
      loi_4b_pilote_mutant_metro bench_f32 || NB_FAIL=$((NB_FAIL + 1))
  fi
  if veut 3; then
    journal "== fenêtre 3 : chaînes F32 des 6 cellules (L = 6, 8)"
    fenetre chaines-6-8 bash -c "$(lot_cmd 6 2.3,2.8,3.5 chaines/f32) $(lot_cmd 8 2.3,2.8,3.5 chaines/f32)" || NB_FAIL=$((NB_FAIL + 1))
  fi
  if veut 4; then
    journal "== fenêtre 4 : chaînes F32 à L = 16 (β = 2,5 et 16)"
    fenetre chaines-16 bash -c "$(lot_cmd 16 2.5,16 chaines/f32 1 0)" || NB_FAIL=$((NB_FAIL + 1))
  fi
  if veut 5; then
    journal "== fenêtre 5 : contrôle négatif β x 1,002 (L = 6, 8)"
    fenetre chaines-mutant bash -c "$(lot_cmd 6 2.3,2.8,3.5 chaines/f32_beta1002 1.002 0 20261004500000) $(lot_cmd 8 2.3,2.8,3.5 chaines/f32_beta1002 1.002 0 20261004500000)" \
      || NB_FAIL=$((NB_FAIL + 1))
  fi
  if [ "$MACHINE" = gamer ] && [ -z "${KVU_F32_FENETRES:-}" ]; then f64; bench_rapport; fi
}

# FP64 de même protocole : su2_hip (acemagic) et su2_cuda (gamer) de reference/, REBÂTIS ICI (jamais le binaire de
# production), mêmes options de compilation que leur build.sh. Acemagic : comparateurs de LOI-2 à L = 16 + banc.
# Gamer : banc seulement (les comparateurs L = 16 sont ceux d'acemagic : la loi ne dépend pas de la machine).
f64() {
  mkdir -p "$F64DIR"
  garde_ssd
  local bin sha="ed1e0fce255cb15d640ab603b3af0b6c50075bbb65252ce5d7e351abd3061e30"
  if [ "$MACHINE" = gamer ]; then
    bin="$SCRATCH/su2_cuda_ref"
    if [ ! -x "$bin" ]; then
      journal "build de su2_cuda depuis reference/ (nvcc 13.3 -arch=sm_120 -O3 -std=c++17, comme build_cuda.sh)"
      ( cd "$RACINE/crates/kvu/reference/su2_cuda" && /usr/local/cuda-13.3/bin/nvcc -arch=sm_120 -O3 -std=c++17 -o "$bin" su2_cuda.cu ) \
        || { journal "build su2_cuda en échec"; exit 1; }
    fi
  else
    bin="$SCRATCH/su2_hip_ref"
    if [ ! -x "$bin" ]; then
      journal "build de su2_hip depuis reference/ (hipcc --offload-arch=gfx1151 -O3 -std=c++17, comme build.sh)"
      ( cd "$SCRATCH" && hipcc --offload-arch=gfx1151 -O3 -std=c++17 "$RACINE/crates/kvu/reference/su2_hip/su2_hip.cpp" -o "$bin" ) \
        || { journal "build su2_hip en échec"; exit 1; }
    fi
    # L = 16, beta 2,5 et 16 : c1, c2 chauds, c3 froid ; une fenêtre par bêta
    for b in 2.5 16; do
      local cmds=""
      for r in 1 2 3; do
        local dep=hot; [ "$r" = 3 ] && dep=cold
        local seed=$((20261004700000 + 100 * ${b%.*} + r))
        cmds+="$bin --L 16 --beta $b --seed $seed --start $dep --therm 2000 --nmeas 20000 --skip 1 --nor 4 --dump-every 0 --source-hash $sha --outdir $F64DIR/L16_b${b}_c$r > $F64DIR/L16_b${b}_c$r.log 2>&1 ; "
      done
      fenetre "f64-L16-b$b" bash -c "$cmds" || NB_FAIL=$((NB_FAIL + 1))
    done
  fi
  # banc FP64 (--bench), même métrique que bench_f32
  fenetre "f64-bench" bash -c "for L in 16 32; do $bin --bench --L \$L --beta 2.3 --nsweeps \$((L==16 ? 200 : 60)); done > $F64DIR/bench_f64.log 2>&1" \
    || NB_FAIL=$((NB_FAIL + 1))
}

bench_rapport() {
  local py="$PY"; [ -x "$py" ] || py=python3
  "$py" - "$KVU_LOI_OUT" "$F64DIR/bench_f64.log" <<'PYEOF'
import json, re, sys, os
out, f64log = sys.argv[1], sys.argv[2]
f32 = [json.loads(l) for l in open(os.path.join(out, 'bench_f32.jsonl')) if l.strip()] if os.path.exists(os.path.join(out, 'bench_f32.jsonl')) else []
f64 = {}
if os.path.exists(f64log):
    for l in open(f64log):
        if l.startswith('BENCH'):
            kv = dict(re.findall(r'(\w+)=([-\d.e+]+)', l))
            f64[int(kv['L'])] = dict(hb=float(kv['liens_par_s_hb']), orr=float(kv['liens_par_s_or']))
lignes = []
for b in f32:
    r = dict(b)
    if b['chaines'] == 1 and b['l'] in f64:
        r['f64_hb_liens_par_s'] = f64[b['l']]['hb']; r['f64_or_liens_par_s'] = f64[b['l']]['orr']
        r['gain_hb'] = b['hb_liens_par_s'] / f64[b['l']]['hb']; r['gain_or'] = b['or_liens_par_s'] / f64[b['l']]['orr']
    lignes.append(r)
    print('BANC', json.dumps(r))
json.dump(dict(lignes=lignes, note='non gaté ; FP64 = su2_hip de reference/ rebâti (même métrique)'), open(os.path.join(out, 'bench.json'), 'w'), indent=1)
PYEOF
}

juge() {
  [ -x "$PY" ] || { journal "Python du venv absent : $PY"; exit 1; }
  local rc=0
  [ "$MACHINE" = acemagic ] || journal "NB : le jugement se fait sur acemagic (règle deux machines)"
  [ -f "$KVU_LOI_OUT/bench_f32.jsonl" ] && bench_rapport
  local l16=""; [ -d "$F64DIR" ] && l16="--f64-l16 $F64DIR"
  journal "== jugement de $KVU_LOI_OUT"
  "$PY" "$ICI/juge_f32.py" --dir "$KVU_LOI_OUT" $l16 --json "$KVU_LOI_OUT/juge.json" || rc=$?
  if [ -n "$GAMER_DIR" ]; then
    journal "== jugement des sorties du gamer (rapatriées) : $GAMER_DIR"
    "$PY" "$ICI/juge_f32.py" --dir "$GAMER_DIR" $l16 --json "$GAMER_DIR/juge.json" || rc=$?
  fi
  return $rc
}

case "$MODE" in
  gen) gen ;;
  f64) f64; bench_rapport ;;
  bench) mkdir -p "$SCRATCH" "$KVU_LOI_OUT"; construire; fenetre bench $PREFIXE "$EXE" --ignored --nocapture --test-threads=1 --exact bench_f32; f64; bench_rapport ;;
  juge) juge; exit $? ;;
  tout) gen; [ "$MACHINE" = acemagic ] && f64; juge; RC=$?; [ "$NB_FAIL" -gt 0 ] && RC=8; exit $RC ;;
  nettoyer) rm -rf "$SCRATCH"; journal "scratch supprimé : $SCRATCH"; exit 0 ;;
  *) echo "mode inconnu : $MODE" >&2; exit 2 ;;
esac
[ "$NB_FAIL" -gt 0 ] && { journal "$NB_FAIL fenêtre(s) en échec"; exit 8; }
exit 0
