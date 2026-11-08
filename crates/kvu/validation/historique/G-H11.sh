#!/usr/bin/env bash
# G-H11 (ancienne P4, AMD) : critères C1-C4 de PREREG_CRITERES sur les 18 chaînes HIP unifiées (gh07, produites
# sur acemagic par G-H07). Mêmes seuils gelés que G-H10.
#
# ARBITRAGE D3 (GATES.md v1, lot WP2 ; docs/ARBITRAGES_D3.md) : la cellule (L = 8, β = 3,5) rend C2 rouge avec z = −3,28 ;
# c'est un fait statistique du protocole (bit-identique à su2_hip lancé seul sur les mêmes graines), pas une régression, et
# elle n'est JAMAIS relancée avec d'autres graines. Le statut devient FAIL-DOCUMENTE (non bloquant, toujours listé)
# SEULEMENT si les trois conditions suivantes sont réunies ; sinon la ligne reste un FAIL réel (régression) :
#   1. le seul critère rouge est L8b3.5:C2_ok et son z vaut −3,28 (mêmes chiffres que valide_complet_1ae422f.json) ;
#   2. measurements.csv des 3 chaînes de graines 27092708021/22/23 (gh07/L8_hot/c004, c005, gh07/L8_cold/c002) a le sha256
#      de preuves/gh11_bit_identite.txt (section b : su2_hip lancé seul, rejeu du 04/10) ;
#   3. therm.csv de ces 3 chaînes (et leurs measurements.csv) a le sha256 de preuves/gh11_therm_wp2.txt (su2_hip de
#      production lancé seul sur les mêmes graines, fenêtre GPU du lot WP2).
GATE=G-H11
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
[ "$(nb_complete "$STORE_V/gh07")" -ge 18 ] || { fail "$GATE" "ensemble gh07 absent ou incomplet (G-H07 doit passer avant)"; exit 0; }

PREUVES="${KVU_PREUVES:-$HIST/../integration/preuves}"   # surchargeable pour les tests (tests/wp2_arbitrages.rs)
nice -n 10 "$PY" "$HIST/analyse_historique.py" p4hip --hip "$STORE_V/gh07" --json "$SCRATCH/gh11.json" \
  >"$SCRATCH/gh11.out" 2>"$SCRATCH/gh11.err"
cat "$SCRATCH/gh11.err" >&2
verdict="$(tail -1 "$SCRATCH/gh11.out")"
case "$verdict" in
  "FAIL $GATE "*) ;;
  *) echo "$verdict"; exit 0;;   # PASS (ou sortie inattendue) : rien à arbitrer
esac

refus=""
case "$verdict" in
  *"C1-C4 ROUGES : L8b3.5:C2_ok") ;;
  *) refus="le rouge n'est pas le seul critère L8b3.5:C2_ok";;
esac
if [ -z "$refus" ]; then
  grep -qE '^L8 b3\.5: .* C2 -3\.28 ' "$SCRATCH/gh11.err" || refus="z de L8 b3.5 / C2 différent de −3,28"
fi
verifier_sha() { # fichier_de_preuve motif_chemin : compare sha256 des fichiers listés (chemin relatif à gh07) ; vide si tout est identique
  local preuve="$1" motif="$2" h rel n=0
  while read -r h rel; do
    [ -f "$STORE_V/gh07/$rel" ] || { echo "fichier absent : gh07/$rel"; return; }
    [ "$(sha256sum "$STORE_V/gh07/$rel" | cut -d' ' -f1)" = "$h" ] || { echo "sha256 différent : gh07/$rel"; return; }
    n=$((n + 1))
  done < <(awk -v m="$motif" '$2 ~ m { k = index($2, "/gh07/"); if (k) print $1, substr($2, k + 6); else print $1, $2 }' "$preuve")
  [ "$n" -ge 3 ] || echo "moins de 3 empreintes lues dans $(basename "$preuve") ($n)"
}
if [ -z "$refus" ]; then
  # section (b) de gh11_bit_identite.txt : measurements.csv des chaînes de graines 27092708021/22/23 (lignes du store gh07)
  refus="$(verifier_sha "$PREUVES/gh11_bit_identite.txt" 'validation_unifie/gh07/L8_(hot|cold)/c00[0-9]/measurements.csv$')"
fi
if [ -z "$refus" ]; then
  refus="$(verifier_sha "$PREUVES/gh11_therm_wp2.txt" 'L8_(hot|cold)/c00[0-9]/(therm|measurements).csv$')"
fi
if [ -z "$refus" ]; then
  echo "FAIL-DOCUMENTE $GATE ${verdict#FAIL $GATE } ; z = −3,28 (L = 8, β = 3,5, C2 chaud/froid) ; chaînes de graines 27092708021/22/23 identiques à l'octet (measurements.csv et therm.csv) à su2_hip lancé seul : fait statistique du protocole, documenté par l'arbitrage D3 (GATES v1), cellule non relancée"
else
  echo "$verdict ; ARBITRAGE D3 REFUSÉ ($refus) : FAIL réel, régression à examiner"
fi
exit 0
