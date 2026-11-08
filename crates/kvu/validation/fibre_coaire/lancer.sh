#!/usr/bin/env bash
# WP11 — validation de docs/FIBRE_COAIRE.md (fibre de Bałaban, jacobien de la coaire, cas exacts).
# Usage : bash lancer.sh [DOSSIER_SORTIES]        (défaut : ./sorties ; ~15 min sur acemagic, aucun GPU, < 4 Go de RAM)
# Python : python3 système (sympy 1.14, mpmath 1.3, numpy 2.5) en mode isolé (-I) sans bytecode (-B).
# Lectures seules : ../rg/fixtures (WP6) et le dossier S1 du chantier vision (geom.py, picov.py, bundle_q3).
# Sortie : une ligne par script dans rc.txt, toutes les lignes PASS/FAIL/SKIP dans verdicts.txt, empreintes dans SHA256SUMS ;
# code de sortie 0 ssi aucun FAIL ni SKIP et tous les scripts rc 0.
set -u
ICI=$(cd "$(dirname "$0")" && pwd)
SORT=$(mkdir -p "${1:-$ICI/sorties}" && cd "${1:-$ICI/sorties}" && pwd)
export FIBRE_SORTIES="$SORT"
# machine partagée : fils BLAS bornés (8 par défaut) ; les verdicts n'en dépendent pas (aucune gate au bit)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
S1=${KVU_S1:-/chemin/vers/projet/chantiers/vision_20261025/calculs/S1_bloc_symetrique_L3}
FIX="$ICI/../rg/fixtures"
PY=(python3 -I -B)
BORNE=()
if command -v calc_borne > /dev/null 2>&1; then BORNE=(calc_borne --mem 10G --nice 10 --); fi
: > "$SORT/rc.txt"
{ date '+%F %T'; python3 -I -B -c "import sys, sympy, mpmath, numpy; print('python', sys.version.split()[0], 'sympy', sympy.__version__, 'mpmath', mpmath.__version__, 'numpy', numpy.__version__)"; } > "$SORT/versions.txt" 2>&1
run() {
  local nom=$1; shift
  local t0=$SECONDS
  "$@" > "$SORT/$nom.out" 2>&1
  local rc=$?
  echo "$nom rc=$rc durée=$((SECONDS - t0))s" >> "$SORT/rc.txt"
}
run t1_dlog            "${PY[@]}" "$ICI/t1_dlog.py"
run t2_jouet           "${PY[@]}" "$ICI/t2_jouet.py"
run t3_chemins         "${PY[@]}" "$ICI/t3_chemins.py"
run t4_balaban         "${PY[@]}" "$ICI/t4_balaban.py" "$FIX"
run t5_u1              "${BORNE[@]}" "${PY[@]}" "$ICI/t5_u1.py"
run t6_beta_infini     "${BORNE[@]}" "${PY[@]}" "$ICI/t6_beta_infini.py" "$S1" "$SORT/valeurs_u1.json"
run t7_rattle_arbitrage "${PY[@]}" "$ICI/t7_rattle_arbitrage.py"
run t8_puissance_mutant "${BORNE[@]}" "${PY[@]}" "$ICI/t8_puissance_mutant.py"
grep -h -E '^(PASS|FAIL|SKIP) ' "$SORT"/t*.out > "$SORT/verdicts.txt"
grep -h -E '^INFO ' "$SORT"/t*.out > "$SORT/info.txt"
npass=$(grep -c '^PASS ' "$SORT/verdicts.txt"); nnv=$(grep -c -E '^(FAIL|SKIP) ' "$SORT/verdicts.txt")
nrc=$(grep -v -c ' rc=0 ' "$SORT/rc.txt")
( cd "$SORT" && sha256sum t*.out valeurs_*.json verdicts.txt info.txt rc.txt versions.txt > SHA256SUMS )
echo "BILAN WP11 : $npass PASS, $nnv non vert(s), $nrc script(s) en rc != 0 — sorties dans $SORT"
[ "$nnv" -eq 0 ] && [ "$nrc" -eq 0 ]
