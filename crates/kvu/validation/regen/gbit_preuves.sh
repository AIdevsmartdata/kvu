#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gbit_preuves.sh — produit le fichier de preuves G-BIT que `kvu regen --preuve-gbit` exige pour un binaire qui n'est pas
# le binaire attendu du manifeste (GATES.md §1/WP4, ARCHITECTURE §7.1). Les preuves sont liées au sha256 du binaire :
# une preuve d'un autre binaire est ignorée.
#
# USAGE (TOUJOURS sous le verrou GPU, fenêtre ≤ 10 min, ≥ 30 Go libres) :
#   flock -w 7200 /tmp/kevinotron_gpu.lock env KVU_BIN=<kvu> ANCRES=<anchors.sha256> SU2_GEN=<su2_gen historique> \
#        SCRATCH=<dossier neuf> PREUVES=<fichier de sortie> bash gbit_preuves.sh
# Sortie : une ligne `PASS|FAIL <ID> binaire_sha256=<sha> <détail>` par gate sur la sortie standard ET dans $PREUVES.
#   G-BIT-HIP-PROD : `kvu su2 gen --backend hip --precision f64 --options compat-prod` L=12 β=2.5 graine 26095612011 …
#                    sha256 de cfg_002000.npy, measurements.csv, therm.csv = lignes `hip` de anchors.sha256 (3/3).
#   G-BIT-K1       : `kvu su2 gen … --options ref-k1` contre `su2_gen --hip` (binaire existant, lecture seule), L=12 et 16 :
#                    cmp identique de 4 NPY + 2 CSV (si SU2_GEN absent : SKIP, qui n'est PAS une preuve).
# Code de sortie : nombre de FAIL.
set -u
: "${KVU_BIN:?KVU_BIN requis}" "${ANCRES:?ANCRES requis}" "${SCRATCH:?SCRATCH requis}" "${PREUVES:?PREUVES requis}"
SU2_GEN=${SU2_GEN:-}
[ -x "$KVU_BIN" ] || { echo "FAIL BUILD binaire $KVU_BIN introuvable"; exit 1; }
[ -e "$SCRATCH" ] && { echo "REFUS : $SCRATCH existe déjà (scratch neuf requis)"; exit 2; }
[ -e "$PREUVES" ] && { echo "REFUS : $PREUVES existe déjà"; exit 2; }
mkdir -p "$SCRATCH" || exit 2
SHA=$(sha256sum "$KVU_BIN" | cut -d' ' -f1)
NFAIL=0
: > "$PREUVES"
emit() { # $1 = PASS|FAIL|SKIP, $2 = ID, $3 = détail
  echo "$1 $2 binaire_sha256=$SHA $3" | tee -a "$PREUVES"
  [ "$1" = FAIL ] && NFAIL=$((NFAIL + 1))
}
ancre() { awk -v b="$1" -v f="$2" '$1==b && $2==f {print $3}' "$ANCRES"; }

# --- G-BIT-HIP-PROD ---
D="$SCRATCH/hip_prod"
timeout 600 "$KVU_BIN" su2 gen --backend hip --precision f64 --options compat-prod --L 12 --beta 2.5 --graine 26095612011 \
  --start hot --therm 5000 --nmeas 2000 --skip 1 --nor 4 --dump-every 100 --outdir "$D" > "$SCRATCH/hip_prod.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  emit FAIL G-BIT-HIP-PROD "rc=$RC (journal $SCRATCH/hip_prod.log)"
else
  N=0; DET=""
  for f in cfg_002000.npy measurements.csv therm.csv; do
    A=$(ancre hip "$f"); S=$(sha256sum "$D/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$A" ] && [ "$A" = "$S" ]; then N=$((N + 1)); else DET="$DET $f"; fi
  done
  if [ $N -eq 3 ]; then emit PASS G-BIT-HIP-PROD "3/3 identiques aux ancres (rc=0)"; else emit FAIL G-BIT-HIP-PROD "$N/3, différents :$DET"; fi
fi
rm -rf "$D"

# --- G-BIT-K1 ---
if [ -z "$SU2_GEN" ] || [ ! -x "$SU2_GEN" ]; then
  emit SKIP G-BIT-K1 "SU2_GEN absent ($SU2_GEN) : pas de preuve"
else
  OK=1; DET=""
  for spec in "12 926193012" "16 926193016"; do
    set -- $spec; L=$1; G=$2
    timeout 600 "$SU2_GEN" --hip --L "$L" --beta 2.3 --seed "$G" --start hot --therm 300 --nmeas 1000 --skip 1 --nor 4 \
      --dump-every 250 --outdir "$SCRATCH/k1_ref_L$L" > "$SCRATCH/k1_ref_L$L.log" 2>&1 || { OK=0; DET="$DET su2_gen L$L rc≠0;"; continue; }
    timeout 600 "$KVU_BIN" su2 gen --backend hip --precision f64 --options ref-k1 --L "$L" --beta 2.3 --graine "$G" --start hot \
      --therm 300 --nmeas 1000 --skip 1 --nor 4 --dump-every 250 --outdir "$SCRATCH/k1_kvu_L$L" > "$SCRATCH/k1_kvu_L$L.log" 2>&1 || { OK=0; DET="$DET kvu L$L rc≠0;"; continue; }
    N=0
    for f in cfg_000250.npy cfg_000500.npy cfg_000750.npy cfg_001000.npy therm.csv measurements.csv; do
      if cmp -s "$SCRATCH/k1_ref_L$L/$f" "$SCRATCH/k1_kvu_L$L/$f"; then N=$((N + 1)); else OK=0; DET="$DET L$L/$f DIFFÈRE;"; fi
    done
    DET="$DET L$L $N/6;"
    rm -rf "$SCRATCH/k1_ref_L$L" "$SCRATCH/k1_kvu_L$L"
  done
  if [ $OK = 1 ]; then emit PASS G-BIT-K1 "cmp identique 4 NPY + 2 CSV, L=12 et 16 contre su2_gen --hip ($DET )"; else emit FAIL G-BIT-K1 "$DET"; fi
fi
echo "# fin : $NFAIL FAIL ; preuves : $PREUVES"
exit "$NFAIL"
