#!/usr/bin/env bash
# Saturation v3 (verrou (b)) : génération de V puis chaînes conditionnelles `cond chaine --mode ref_rng`.
#
# 1. V : une configuration thermalisée, produite par `su2 gen` en F64 (remplace su2_gen K1, en continuité de bits :
#    gate G-BIT-K1) ; le dump c16 `cfg_NNNNNN.npy` est lu tel quel par numpy et par `cond chaine --cfg`.
# 2. Chaînes C-b : `cond chaine --mode ref_rng` reste la loi de RÉFÉRENCE imposée par G-LOI(a) (tirages numpy rejoués,
#    concordance ≤ 1e-12 par balayage avec cb.py), mais le calcul des agrafes est sur GPU.
# 3. Sorties pour torsion.py : F.npy (nmes, 54), D.npy (nD, 54, 54), blk.npy (nmes, 6, (L/2)^4) float32, controles.json.
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
if [ "$EXEMPLE_MODE" = petit ]; then
  L=4; THERM=10; NMEAS=4; NTHERM=4; NSWEEP=8; EVERY=2; DEACH=2; CKPT=4; BACKEND="--backend cpu"
else
  L=32; THERM=4000; NMEAS=1; NTHERM=100; NSWEEP=6000; EVERY=2; DEACH=5; CKPT=500; BACKEND=""
fi
VERROU="${VERROU:-/tmp/kevinotron_gpu.lock}"
lancer "$KVU_BIN" su2 gen --L "$L" --beta 2.3 --graine 926193012 --start hot --therm "$THERM" --nmeas "$NMEAS" --skip 1 --nor 4 \
  --dump-every 1 --precision f64 --options ref-k1 $BACKEND --outdir "$SORTIE/V_L$L" || exit $?
CFG="$SORTIE/V_L$L/cfg_$(printf '%06d' "$NMEAS").npy"
if [ "$EXEMPLE_MODE" = petit ] && ! [ -f "$CFG" ]; then echo "dump attendu absent : $CFG" >&2; exit 1; fi
lancer "$KVU_BIN" cond chaine --cfg "$CFG" --beta 2.3 --graine 2709 --mode ref_rng \
  --ntherm "$NTHERM" --nsweep "$NSWEEP" --every "$EVERY" --deach "$DEACH" --ckpt "$CKPT" $BACKEND \
  --out "$SORTIE/chaine_L${L}_b2.3" || exit $?

echo "# Lecture pour torsion.py (numpy suffit) :"
echo "#   F = np.load('$SORTIE/chaine_L${L}_b2.3/F.npy'); D = np.load('$SORTIE/chaine_L${L}_b2.3/D.npy')"
if [ "$EXEMPLE_MODE" = petit ]; then
  python3 - "$SORTIE/chaine_L${L}_b2.3" <<'PYEOF' || exit 1
import sys, os
try:
    import numpy as np
except ImportError:
    print("numpy absent : lecture NPY non vérifiée"); sys.exit(0)
d = sys.argv[1]
F, D = np.load(os.path.join(d, "F.npy")), np.load(os.path.join(d, "D.npy"))
assert F.ndim == 2 and F.shape[1] == 54, F.shape
assert D.ndim == 3 and D.shape[1:] == (54, 54), D.shape
print("lecture NPY ok : F", F.shape, "D", D.shape)
PYEOF
fi
