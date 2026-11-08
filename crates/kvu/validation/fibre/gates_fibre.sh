#!/usr/bin/env bash
# Gates de la mesure de fibre (T6) : FIB-1 (CPU et GPU), FIB-2 (déterminisme GPU), FIB-3 (CPU = GPU à l'octet).
#
# Sur acemagic (HIP) :
#   flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/fibre/gates_fibre.sh
# Sur le gamer (CUDA), depuis une copie du worktree (rsync par le câble) :
#   flock -w 7200 /tmp/gamer_gpu.lock bash crates/kvu/validation/fibre/gates_fibre.sh
# (le verrou est pris par l'appelant ; le script ne le reprend pas : pas d'interblocage.)
#
# Variables : CARGO_TARGET_DIR (défaut <dev>/targets/T6, gamer : ~/kvu_unif/cibles/T6),
#   KVU_FIBRE_FIXTURES (jeu complet existant ; sinon généré par gen_fixtures.py dans un répertoire du SSD),
#   KVU_PYTHON (venv ym4_noyaux), KVU_NOYAU2 (noyau2 de référence), KVU_FIBRE_GARDER=1 (garde les fixtures),
#   KVU_FIBRE_BANC=0 (saute le banc), KVU_FIBRE_BANC_L=16,24,32 (tailles du banc).
# Sortie : une ligne PASS/FAIL par gate, puis « GATES FIBRE : VERT|ROUGE ». RC = 0 si tout est vert.
set -uo pipefail

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACINE="$(cd "$ICI/../../../.." && pwd)"
cd "$RACINE" || exit 2

HOTE="$(hostname)"
if [ "$HOTE" = "pc-nvidia-exemple" ]; then
  MACHINE=gamer; SEUIL_GO=50
  export PATH="$HOME/.cargo/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
  FEATURES="--features cuda"
  PREFIXE="nice -n 5 taskset -c 12-19"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/kvu_unif/cibles/T6}"
  PY="${KVU_PYTHON:-$HOME/ym4_calcul/ym4_noyaux_venv/bin/python}"
  TMPBASE="${KVU_TMP:-$HOME/kvu_unif/cibles/fibre_tmp}"
else
  MACHINE=acemagic; SEUIL_GO=110
  FEATURES=""
  PREFIXE="nice -n 10"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/chemin/vers/projet/chantiers/kevinotron_unification/cibles/T6}"
  PY="${KVU_PYTHON:-/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python}"
  TMPBASE="${KVU_TMP:-/chemin/vers/projet/chantiers/kevinotron_unification/cibles/fibre_tmp}"
fi
JOURNAL_DIR="${KVU_FIBRE_JOURNAUX:-$TMPBASE/journaux}"
mkdir -p "$TMPBASE" "$JOURNAL_DIR"

# --- garde SSD : on s'arrête sous le seuil (110 Go acemagic, 50 Go gamer)
libre_go=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "${libre_go:-0}" -lt "$SEUIL_GO" ]; then
  echo "ARRET : $libre_go Go libres sur / (seuil $SEUIL_GO Go) : rien n'est construit ni écrit."
  exit 7
fi
echo "# machine $MACHINE, $libre_go Go libres, cible cargo $CARGO_TARGET_DIR"

ROUGE=0
rouge() { echo "FAIL $1"; ROUGE=1; }

# --- fixtures (jeu complet : 14 cas du plan de test_concordance.py + 2 bonus de production L = 8)
FIX="${KVU_FIBRE_FIXTURES:-}"
GENERE=0
if [ -z "$FIX" ] || [ ! -f "$FIX/index.json" ]; then
  FIX="${FIX:-$TMPBASE/fixtures_$$}"
  echo "# génération des fixtures dans $FIX"
  if ! systemd-run --scope -p MemoryMax=14G --quiet $PREFIXE "$PY" "$ICI/gen_fixtures.py" --sortie "$FIX" > "$JOURNAL_DIR/gen_fixtures.log" 2>&1; then
    tail -5 "$JOURNAL_DIR/gen_fixtures.log"
    echo "FAIL gen_fixtures (voir $JOURNAL_DIR/gen_fixtures.log)"; exit 1
  fi
  tail -1 "$JOURNAL_DIR/gen_fixtures.log"
  GENERE=1
fi
export KVU_FIBRE_FIXTURES="$FIX"

# --- construction (hors du temps GPU utile, mais sous le verrou de l'appelant)
if ! $PREFIXE cargo test -p kvu --release $FEATURES --no-run --test fibre_cpu --test fibre_gpu -j 6 > "$JOURNAL_DIR/build.log" 2>&1; then
  tail -20 "$JOURNAL_DIR/build.log"; echo "FAIL construction"; exit 1
fi

# --- FIB-1 CPU
$PREFIXE cargo test -p kvu --release $FEATURES --test fibre_cpu -- --nocapture > "$JOURNAL_DIR/fib1_cpu.log" 2>&1
RC=$?
grep -E "^(PASS|FAIL) FIB-1 CPU" "$JOURNAL_DIR/fib1_cpu.log"
[ $RC -eq 0 ] && grep -q "^PASS FIB-1 CPU" "$JOURNAL_DIR/fib1_cpu.log" || rouge "FIB-1 CPU (rc=$RC, voir $JOURNAL_DIR/fib1_cpu.log)"

# --- GPU : FIB-1, FIB-2, FIB-3, champ résident
$PREFIXE cargo test -p kvu --release $FEATURES --test fibre_gpu -- --ignored --nocapture --test-threads=1 --skip banc > "$JOURNAL_DIR/gpu.log" 2>&1
RC=$?
grep -E "^(PASS|FAIL|SKIP) " "$JOURNAL_DIR/gpu.log"
if [ $RC -ne 0 ]; then rouge "tests GPU (rc=$RC, voir $JOURNAL_DIR/gpu.log)"; fi
grep -q "^PASS FIB-1 GPU" "$JOURNAL_DIR/gpu.log" || rouge "FIB-1 GPU absent ou ROUGE (un SKIP ne vaut pas vert)"
grep -q "^PASS FIB-2 GPU" "$JOURNAL_DIR/gpu.log" || rouge "FIB-2 GPU absent ou ROUGE"
grep -q "^PASS FIB-3 GPU" "$JOURNAL_DIR/gpu.log" || rouge "FIB-3 GPU absent ou ROUGE"

# --- banc (rapporté, non gaté) : mesure à L = 16, 24, 32 contre cb.mesure (README de noyau2), et GPU = CPU à l'octet
if [ "${KVU_FIBRE_BANC:-1}" = "1" ]; then
  $PREFIXE cargo test -p kvu --release $FEATURES --test fibre_gpu banc -- --ignored --nocapture --test-threads=1 > "$JOURNAL_DIR/banc.log" 2>&1
  RC=$?
  grep -E "^BANC" "$JOURNAL_DIR/banc.log"
  [ $RC -eq 0 ] || rouge "banc (rc=$RC, voir $JOURNAL_DIR/banc.log) : GPU et CPU diffèrent à l'octet, ou erreur"
fi

if [ "${KVU_FIBRE_GARDER:-0}" != "1" ] && [ $GENERE -eq 1 ]; then rm -rf "$FIX"; fi
if [ $ROUGE -eq 0 ]; then echo "GATES FIBRE : VERT"; else echo "GATES FIBRE : ROUGE"; fi
exit $ROUGE
