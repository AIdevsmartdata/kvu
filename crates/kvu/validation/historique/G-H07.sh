#!/usr/bin/env bash
# G-H07 (ancien « runs AMD 20/20 ») : les mêmes 18 chaînes, EN LOTS (4 lancements : L × départ), sur acemagic (HIP) : rc 0, 18 COMPLETE.
GATE=G-H07
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
exiger_gpu; binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rm -rf "$STORE_V/gh07"; mkdir -p "$STORE_V/gh07"
lancer_les_lots gh07
pb="$PB"
n="$(nb_complete "$STORE_V/gh07")"
if [ -n "$pb" ]; then fail "$GATE" "$pb"
elif [ "$n" -ne 18 ]; then fail "$GATE" "$n chaînes COMPLETE au lieu de 18"
else pass "$GATE" "18 chaînes en 4 lots (L = 6, 8 × chaud, froid), rc 0, 18 COMPLETE, sous $STORE_V/gh07"; fi
exit 0
