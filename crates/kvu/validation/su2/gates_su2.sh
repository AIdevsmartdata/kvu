#!/usr/bin/env bash
# gates_su2.sh — gates de bit-identité et de déterminisme du générateur SU(2) unifié (T3, ARCHITECTURE_UNIFIEE.md §6.1).
#
# USAGE (depuis la racine du worktree ; TOUJOURS sous le verrou GPU de la machine, fenêtre ≤ 10 min) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/su2/gates_su2.sh
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock      bash crates/kvu/validation/su2/gates_su2.sh   (SUR le gamer)
# Variables : KVU_GATE_MACHINE=acemagic|gamer (défaut : détectée par le nom d'hôte), CARGO_TARGET_DIR,
#   KVU_SCRATCH (références et sorties temporaires), KVU_SU2_GEN (su2_gen historique déjà bâti).
# Sortie : une ligne `PASS|FAIL|SKIP <gate> : détail` par gate ; code de sortie = nombre de FAIL (SKIP ne vaut pas vert).
#
# Gates : G-BIT-K1 (acemagic, `cmp` contre su2_gen --hip bâti depuis main), G-BIT-HIP-PROD (repli gelé §4.4 si besoin),
# G-BIT-CUDA et G-BIT-BORD (gamer), G-CPU-GPU, G-DET, LOT-1-F64, REPRISE-1-F64, REPRISE-1-BORD, BORD-C1, BORD-C2a,
# G-H04 (tirage de x0, 7 α, 2·10^7, avec mutant). Les références (gpu_port, fixtures, su2_gen) sont lues EN PLACE.
set -u
ICI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACINE=$(cd "$ICI/../../../.." && pwd)
cd "$RACINE" || exit 99

HOTE=$(hostname)
MACHINE=${KVU_GATE_MACHINE:-}
if [ -z "$MACHINE" ]; then
  case "$HOTE" in *pc-nvidia-exemple*) MACHINE=gamer ;; *) MACHINE=acemagic ;; esac
fi
if [ "$MACHINE" = gamer ]; then
  FEAT="--features cuda"; SEUIL=50; PREF="nice -n 5 taskset -c 12-19"
  export PATH="$HOME/.cargo/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  : "${CARGO_TARGET_DIR:=$HOME/kvu_unif/cibles/T3}"
  BACKEND=cuda
else
  FEAT=""; SEUIL=110; PREF="nice -n 10"
  : "${CARGO_TARGET_DIR:=/chemin/vers/projet/chantiers/kevinotron_unification/cibles/T3}"
  BACKEND=hip
fi
export CARGO_TARGET_DIR
SCRATCH=${KVU_SCRATCH:-$(dirname "$(dirname "$CARGO_TARGET_DIR")")/scratch/T3}
mkdir -p "$SCRATCH"
NFAIL=0
verdict() { # $1 = PASS|FAIL|SKIP, $2 = gate, $3 = détail
  echo "$1 $2 : $3"
  [ "$1" = FAIL ] && NFAIL=$((NFAIL + 1))
}

# --- garde SSD : on s'arrête sous le seuil de la machine -------------------------------------------------------
LIBRE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
echo "# machine=$MACHINE backend=$BACKEND SSD libre=${LIBRE} Go (arrêt sous $SEUIL Go)"
if [ "${LIBRE:-0}" -lt "$SEUIL" ]; then
  echo "ARRET : SSD sous le seuil ($LIBRE Go < $SEUIL Go) : aucune écriture lourde, aucun build"
  exit 7
fi

# --- binaires de test (les builds se font AVANT le verrou ; ici cargo ne fait que vérifier) -------------------------
trouver_bin() { # $1 = nom du test
  cargo test -j "${KVU_JOBS:-6}" -p kvu --release $FEAT --test "$1" --no-run --message-format=json 2>/dev/null |
    python3 -c '
import sys, json
nom = sys.argv[1]
exe = None
for l in sys.stdin:
    try:
        m = json.loads(l)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == nom and m.get("executable"):
        exe = m["executable"]
print(exe or "")
' "$1"
}
BIN_GPU=$(trouver_bin su2_gpu)
BIN_CLI=$(trouver_bin su2_cli)
if [ -z "$BIN_GPU" ] || [ ! -x "$BIN_GPU" ] || [ -z "$BIN_CLI" ] || [ ! -x "$BIN_CLI" ]; then
  verdict FAIL BUILD "binaires de test introuvables (cargo test --release $FEAT --test su2_gpu --test su2_cli --no-run)"
  exit 1
fi
cli() { # exécute `su2 <args…>` par la passerelle (KVU_SU2_ARGV séparé par U+001F)
  local a
  a=$(IFS=$'\x1f'; printf '%s' "$*")   # IFS modifié seulement dans la sous-coquille (sinon $PREF ne se découpe plus)
  KVU_SU2_ARGV="$a" $PREF "$BIN_CLI" --ignored --exact shim_cli --nocapture --test-threads=1
}

# --- G-BIT-K1 (acemagic) : `kvu su2 gen --backend hip --precision f64 --options ref-k1` contre su2_gen --hip -------
unset KVU_REF_K1
if [ "$MACHINE" = acemagic ]; then
  # Le moteur historique n'est PAS dans ce dépôt (kvu autonome, WP1) : aucun build de su2_gen ici. La référence est un binaire
  # existant, fourni par KVU_SU2_GEN (acemagic : .../brancheC_saturation_v3_20261004/kev/target/release/su2_gen, sha256 cb29307d…,
  # lecture seule). Sans lui la gate est FAIL (jamais SKIP vert).
  GEN=${KVU_SU2_GEN:-}
  if [ -z "$GEN" ] || [ ! -x "$GEN" ]; then
    verdict FAIL G-BIT-K1 "su2_gen historique introuvable : KVU_SU2_GEN=${GEN:-(non défini)} (voir GATES.md G-BIT-K1)"
  else
    rm -rf "$SCRATCH/k1"; mkdir -p "$SCRATCH/k1"
    OK=1; DETAIL=""
    for spec in "12 926193012" "16 926193016"; do
      set -- $spec; L=$1; G=$2
      $PREF "$GEN" --hip --L "$L" --beta 2.3 --seed "$G" --start hot --therm 300 --nmeas 1000 --skip 1 --nor 4 \
        --dump-every 250 --outdir "$SCRATCH/k1/ref_L$L" >"$SCRATCH/k1/ref_L$L.log" 2>&1 || { OK=0; DETAIL="$DETAIL su2_gen L$L rc≠0 ;"; continue; }
      cli gen --backend hip --precision f64 --options ref-k1 --L "$L" --beta 2.3 --seed "$G" --start hot --therm 300 \
        --nmeas 1000 --skip 1 --nor 4 --dump-every 250 --outdir "$SCRATCH/k1/kvu_L$L" >"$SCRATCH/k1/kvu_L$L.log" 2>&1 || { OK=0; DETAIL="$DETAIL kvu su2 gen L$L rc≠0 ;"; continue; }
      N=0
      for f in cfg_000250.npy cfg_000500.npy cfg_000750.npy cfg_001000.npy therm.csv measurements.csv; do
        if cmp -s "$SCRATCH/k1/ref_L$L/$f" "$SCRATCH/k1/kvu_L$L/$f"; then N=$((N + 1)); else OK=0; DETAIL="$DETAIL L$L/$f DIFFÈRE ;"; fi
      done
      DETAIL="$DETAIL L$L : $N/6 identiques ;"
    done
    if [ $OK = 1 ]; then verdict PASS G-BIT-K1 "cmp identique : 4 NPY + 2 CSV, L = 12 et 16 contre su2_gen --hip ($DETAIL)"; else verdict FAIL G-BIT-K1 "$DETAIL"; fi
    export KVU_REF_K1="$SCRATCH/k1/ref"  # (les gates Rust relisent aussi les références produites ci-dessus)
    mkdir -p "$KVU_REF_K1"; for L in 12 16; do rm -rf "$KVU_REF_K1/L$L"; ln -s "$SCRATCH/k1/ref_L$L" "$KVU_REF_K1/L$L"; done
  fi
else
  verdict SKIP G-BIT-K1 "gate propre à acemagic (su2_gen --hip)"
fi

# --- gates Rust (verrou déjà tenu par l'appelant) : un seul lancement, fenêtre bornée à 10 min ------------------
LOG="$SCRATCH/gates_su2_rust_$MACHINE.log"
timeout 600 $PREF "$BIN_GPU" --ignored --nocapture --test-threads=1 >"$LOG" 2>&1
RC=$?
grep -oE '(PASS|FAIL|SKIP) [A-Za-z0-9_.-]+ :.*' "$LOG" | sort -u | while IFS= read -r l; do echo "$l"; done
NF=$(grep -oE '(^|[^A-Za-z])FAIL [A-Za-z0-9_.-]+ :.*' "$LOG" | sort -u | wc -l)
NFAIL=$((NFAIL + NF))
if [ "$RC" -ne 0 ] && [ "$NF" -eq 0 ]; then
  verdict FAIL RUST "su2_gpu a rendu rc=$RC sans ligne FAIL (voir $LOG)"
fi
NP=$(grep -cE '(^|[^A-Za-z])PASS [A-Za-z0-9_.-]+ :' "$LOG")
[ "$NP" -eq 0 ] && verdict FAIL RUST "aucune ligne PASS dans $LOG (un SKIP ne vaut pas vert)"

# --- nettoyage des sorties (SSD) ---------------------------------------------------------------------------------
rm -rf "$CARGO_TARGET_DIR/tmp/su2_gpu" "$SCRATCH/k1"
echo "# fin : $NFAIL FAIL ; journaux : $LOG"
exit "$NFAIL"
