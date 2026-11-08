#!/usr/bin/env bash
# R2-3 / S2 : une CELLULE de prod_s2 en lot de répliques (su2 gen --bord).
#
# Cellule : tag b3.0 (β = 3,0), L = 16, conditions au bord cold / hot / eq1 / eq1g, répliques r = 1..4, comme
# R2_3_influence_bord/calc/prod.sh (su2_bord, binaire 5af3ea49) :
#   graine de chaîne s = 2809280100000 + b·1000 + L·10 + r           (b = 2 pour β = 3,0)
#   graine du bord   : cold 0 ; hot 2809280200000 + b·1000 + L·10 + 1 ; eq1 …+2 ; jauge 2809280300000 + b·1000 + L·10
#   --pre 2000 --therm 4000 --nmeas 50000 --bloc 100 --nor 4
# Les répliques d'un même (L, β) partagent leurs graines d'une condition au bord à l'autre : nombres aléatoires
# communs. En F64 compat-prod, `su2 gen --bord` est bit-identique à su2_bord (gate G-BIT-BORD) : on peut CONTINUER la
# production S2 sans changer la loi. Un passage en F32 exige un AVENANT au PREREG (la loi est prouvée, pas le protocole).
#
# Le LOT : les 4 répliques sont UN lancement (`--chaines 4 --graine S1 --graine-pas 1`), un sous-répertoire c000…c003
# par réplique ; la chaîne c d'un lot donne les mêmes octets que la même chaîne lancée seule (gate LOT-1).
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
if [ "$EXEMPLE_MODE" = petit ]; then
  L=4; B=3.0; PRE=20; THERM=20; NMEAS=40; BLOC=10; BACKEND="--backend cpu"
else
  L=16; B=3.0; PRE=2000; THERM=4000; NMEAS=50000; BLOC=100; BACKEND=""
fi
b=2
base=$((2809280100000 + b * 1000 + L * 10))
sh=$((2809280200000 + b * 1000 + L * 10 + 1)); se1=$((2809280200000 + b * 1000 + L * 10 + 2)); sj=$((2809280300000 + b * 1000 + L * 10))
VERROU="${VERROU:-/tmp/gamer_gpu.lock}"
for bc in "cold --bc-graine 0" "hot --bc-graine $sh" "eq --bc-graine $se1" "eq --bc-graine $se1 --jauge $sj"; do
  nom="$(echo "$bc" | awk '{print $1}')$(echo "$bc" | grep -q -- --jauge && echo g)"
  lancer "$KVU_BIN" su2 gen --bord --L "$L" --beta "$B" --chaines 4 --graine "$((base + 1))" --graine-pas 1 \
    --bc $bc --pre "$PRE" --therm "$THERM" --nmeas "$NMEAS" --bloc "$BLOC" --nor 4 \
    --precision f64 --options compat-prod $BACKEND --outdir "$SORTIE/s2_L${L}_b${B}_$nom" || exit $?
done
echo "Sorties : $SORTIE/s2_L${L}_b${B}_{cold,hot,eq,eqg}/c00{0..3}/{therm_blocs.csv,blocs.csv,manifest.json}"
