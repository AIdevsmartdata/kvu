#!/usr/bin/env bash
# Gates MACH-1, VERROU-1 et FILE-1 (ARCHITECTURE_UNIFIEE.md §6.5) + plan de build_auto.sh (golden).
#
# À lancer SOUS le verrou GPU de la machine, dans une fenêtre courte (quelques secondes une fois
# compilé) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/machine/gates_machine.sh
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock bash crates/kvu/validation/machine/gates_machine.sh
# Sortie : lignes « PASS <gate> », « FAIL <gate> » ou « SKIP <gate> » (SKIP ne vaut pas vert).
# Code de retour : 0 seulement si les trois gates sont PASS (et le golden de build_auto identique).
set -u
VAL=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
KVU=$(cd "$VAL/../.." && pwd)
DEPOT=$(cd "$KVU/../.." && pwd)
HOTE=$(cat /proc/sys/kernel/hostname)
HOME_DIR=${HOME:-/home/utilisateur}

case "$HOTE" in
  acemagic)
    MACHINE=acemagic; CARGO=cargo; JOBS=8; AFFINITE=""; SEUIL=110; FEATURES=""
    CIBLE_DEFAUT=/chemin/vers/projet/chantiers/kevinotron_unification/cibles/$(basename "$DEPOT") ;;
  pc-nvidia-exemple)
    MACHINE=gamer; CARGO=$HOME_DIR/.cargo/bin/cargo; JOBS=6; AFFINITE="12-19"; SEUIL=60; FEATURES="--features cuda"
    CIBLE_DEFAUT=$HOME_DIR/kvu_unif/cibles/$(basename "$DEPOT")
    export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
  *)
    echo "SKIP MACH-1 : hôte $HOTE inconnu, les gates ne valent rien ici"; exit 1 ;;
esac
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$CIBLE_DEFAUT}

# Garde SSD : on ne compile pas sous le seuil (110 Go sur acemagic, 60 sur le gamer).
LIBRE=$(df --output=avail -BG "$(dirname "$CARGO_TARGET_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$LIBRE" ] || [ "$LIBRE" -lt "$SEUIL" ]; then
  echo "FAIL SSD : ${LIBRE:-?} Go libres, il en faut au moins $SEUIL : ARRÊT (rien n'est compilé)"; exit 7
fi

PREFIXE="nice -n 10"
[ -n "$AFFINITE" ] && PREFIXE="$PREFIXE taskset -c $AFFINITE"
JOURNAL=$(mktemp "${TMPDIR:-/tmp}/gates_machine.XXXXXX")
trap 'rm -f "$JOURNAL"' EXIT
cd "$DEPOT" || exit 2

echo "== machine $MACHINE ($HOTE), cible $CARGO_TARGET_DIR, features « ${FEATURES:-aucune} » =="
# shellcheck disable=SC2086
$PREFIXE "$CARGO" test -p kvu -j "$JOBS" --offline $FEATURES --test machine_gates --test machine_detection --no-run > "$JOURNAL" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "FAIL BUILD : compilation des tests en échec (rc $RC)"; tail -n 20 "$JOURNAL"; exit 1; fi
if grep -q '^warning' "$JOURNAL"; then echo "FAIL BUILD : avertissements de compilation"; grep -A6 '^warning' "$JOURNAL" | head -30; exit 1; fi

# shellcheck disable=SC2086
$PREFIXE "$CARGO" test -p kvu -j "$JOBS" --offline $FEATURES --test machine_gates --test machine_detection \
  -- --ignored --nocapture --test-threads=1 > "$JOURNAL" 2>&1
RC=$?
grep -oE '(PASS|FAIL|SKIP) [A-Z0-9-]+ .*' "$JOURNAL"

STATUT=0
for g in MACH-1 VERROU-1 FILE-1; do
  if ! grep -qE "PASS $g " "$JOURNAL"; then echo "FAIL $g : pas de ligne PASS"; STATUT=1; fi
done
if [ $RC -ne 0 ]; then echo "FAIL cargo test : rc $RC"; grep -E 'panicked|FAILED' "$JOURNAL" | head; STATUT=1; fi

# (d) plan de build_auto.sh conforme au golden (features, taskset, cible, seuils). WP1 : un seul crate (kvu) ; la racine
# des sorties (dossier parent du dépôt) est remplacée par <U> pour que le golden ne dépende pas de l'emplacement.
GOLDEN_OK=1
comparer() { # $1 golden  $2.. arguments de build_auto.sh
  local g=$1; shift
  env -u CARGO_TARGET_DIR -u KEVINOTRON_BIN_DIR -u KEVINOTRON_MACHINE -u KEVINOTRON_SSD_MIN_GO -u KVU_LOT -u KVU_SORTIES \
      bash "$DEPOT/scripts/build_auto.sh" --dry-run "$@" \
    | sed '/^--- fin du plan ---$/q' | sed -e "s|$(dirname "$DEPOT")|<U>|g" -e "s|$HOME_DIR|~|g" | diff -u "$VAL/$g" - > "$JOURNAL.diff" \
    || { echo "FAIL BUILD-AUTO-GOLDEN : $g diffère :"; cat "$JOURNAL.diff"; GOLDEN_OK=0; }
}
comparer golden_build_auto_acemagic.txt --machine acemagic
comparer golden_build_auto_gamer.txt --machine gamer
comparer golden_build_auto_gamer_kvu_lto.txt --machine gamer --lto --nettoyer
rm -f "$JOURNAL.diff"
if [ $GOLDEN_OK = 1 ]; then echo "PASS BUILD-AUTO-GOLDEN : 3 plans (acemagic, gamer, gamer+lto) identiques aux golden"; else STATUT=1; fi

# Refus de build_auto.sh (SSD, dépôt sale, production, cible trop large…), sans rien compiler.
SORTIE_REFUS=$(bash "$VAL/test_build_auto.sh" 2>&1); RC_REFUS=$?
printf '%s\n' "$SORTIE_REFUS" | grep -E '^(PASS|FAIL)'
[ $RC_REFUS = 0 ] || STATUT=1

[ $STATUT = 0 ] && echo "== gates machine : TOUTES VERTES ($MACHINE) ==" || echo "== gates machine : ROUGE ($MACHINE) =="
exit $STATUT
