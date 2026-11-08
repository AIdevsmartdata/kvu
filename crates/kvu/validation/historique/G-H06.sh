#!/usr/bin/env bash
# G-H06 (ancien « runs CUDA 26/26 ») : 18 chaînes (L ∈ {6, 8} × β ∈ {2,3 ; 2,8 ; 3,5} × 3), therm 2000,
# nmeas 20000, EN LOTS (4 lancements : L × départ ; 6 chaînes chaudes ou 3 froides chacun) sur le gamer : rc 0 et COMPLETE pour les 18.
# Les ensembles restent sous $STORE_V/gh06 (RECHERCHE : rapatriés et analysés sur acemagic, G-H08..G-H13).
GATE=G-H06
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a gamer
exiger_gpu; binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rm -rf "$STORE_V/gh06"; mkdir -p "$STORE_V/gh06"
lancer_les_lots gh06
pb="$PB"
n="$(nb_complete "$STORE_V/gh06")"
if [ -n "$pb" ]; then fail "$GATE" "$pb"
elif [ "$n" -ne 18 ]; then fail "$GATE" "$n chaînes COMPLETE au lieu de 18"
else pass "$GATE" "18 chaînes en 4 lots (L = 6, 8 × chaud, froid), rc 0, 18 COMPLETE, sous $STORE_V/gh06"; fi
exit 0
