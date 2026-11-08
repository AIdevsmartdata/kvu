#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# exemples_readme.sh : exécute, telles quelles et dans l'ordre, les commandes du bloc d'exemples du README.md
# (entre les marqueurs <!-- exemples:debut --> et <!-- exemples:fin -->), dans un dossier jetable.
#
# Usage : scripts/exemples_readme.sh [BINAIRE_KVU] [DOSSIER]   (défauts : target/release/kvu, dossier temporaire neuf)
# Rend 0 si chaque commande rend 0 et si les sorties attendues existent, 1 sinon. Aucune commande n'utilise le GPU
# (`--backend cpu`), sauf `kvu doctor`, qui ne fait que sonder les bibliothèques GPU présentes.
set -u
RACINE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KVU=${1:-$RACINE/target/release/kvu}
[ -x "$KVU" ] || { echo "binaire kvu introuvable : $KVU (cargo build --release -p kvu)" >&2; exit 2; }
KVU=$(cd "$(dirname "$KVU")" && pwd)/$(basename "$KVU")
D=${2:-$(mktemp -d "${TMPDIR:-/tmp}/kvu_exemples.XXXXXX")}
mkdir -p "$D" && cd "$D" || exit 2
[ -z "$(ls -A .)" ] || { echo "le dossier $D n'est pas vide" >&2; exit 2; }
BIN=$D/.bin; mkdir -p "$BIN"; ln -s "$KVU" "$BIN/kvu"; export PATH="$BIN:$PATH"
# le bloc de commandes : lignes entre les marqueurs, hors clôtures ``` et commentaires
mapfile -t CMDS < <(awk '/<!-- exemples:debut -->/{f=1;next} /<!-- exemples:fin -->/{f=0} f' "$RACINE/README.md" \
  | grep -v -E '^\s*(```|#|$)' | sed -E 's/\s+#\s.*$//')
[ "${#CMDS[@]}" -ge 5 ] || { echo "bloc d'exemples introuvable ou trop court dans README.md" >&2; exit 2; }
ECHEC=0; i=0
for c in "${CMDS[@]}"; do
  i=$((i+1)); t0=$(date +%s.%N)
  bash -c "$c" > "ex_$i.log" 2>&1; rc=$?
  dt=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
  if [ $rc = 0 ]; then echo "PASS [$dt s] $c"; else echo "FAIL rc=$rc [$dt s] $c"; sed 's/^/    /' "ex_$i.log" | tail -15; ECHEC=1; fi
done
for f in ens_L6/COMPLETE ens_L6/manifest.json ens_L6/cfg_000020.npy arch_L4/archive.json arch_L4_cfg10.npy action_L6/action.json flux_L6.json tordu_L4/COMPLETE; do
  [ -e "$f" ] && echo "PASS sortie $f" || { echo "FAIL sortie absente : $f"; ECHEC=1; }
done
echo "dossier : $D"
[ $ECHEC = 0 ] && echo "EXEMPLES VERTS (${#CMDS[@]} commandes)" || echo "EXEMPLES ROUGES"
exit $ECHEC
