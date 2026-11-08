#!/usr/bin/env bash
# lancer.sh : campagne de verdicts de WP13 (voir PREVOL.md). Usage : [KVU_BIN=...] lancer.sh SORTIE_DIR   (à lancer sous calc_borne)
set -u
D=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
OUT=${1:?usage : lancer.sh SORTIE_DIR}
# dossier de sortie NEUF et chemin absolu (aucune reprise de résultats antérieurs)
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "REFUS : $OUT existe et n'est pas vide (dossier neuf exigé)" >&2; exit 3; fi
mkdir -p "$OUT" || exit 3
OUT=$(readlink -f "$OUT")
unset KVU_LATDIR KVU_JVP_H KVU_JVP_FONDS KVU_JVP_NDIR
# GEL : le sha256 de PREVOL.md doit figurer au journal (ligne « WP13 PREVOL sha256 ») AVANT toute campagne qui porte un verdict
JOURNAL=${KVU_JOURNAL:-/chemin/vers/projet/chantiers/kevinotron_unification/journal_tranche2.log}
SHA_PREVOL=$(sha256sum "$D/PREVOL.md" | cut -c1-64)
SHA_GRAINES=$(sha256sum "$D/graines.md" | cut -c1-64)
if ! grep -q "WP13 PREVOL sha256 $SHA_PREVOL" "$JOURNAL" || ! grep -q "graines.md sha256 $SHA_GRAINES" "$JOURNAL"; then
  echo "REFUS : PREVOL.md ($SHA_PREVOL) ou graines.md ($SHA_GRAINES) non haches au journal $JOURNAL : pas de verdict sans gel" >&2; exit 3
fi
PY=/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python3
export KVU_BIN=${KVU_BIN:-/chemin/vers/projet/chantiers/kevinotron_unification/bin/wp13/acemagic/kvu}
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 KVU_FILS=4
export KVU_SCRATCH="$OUT/travail"
mkdir -p "$KVU_SCRATCH"
V=$OUT/verdicts.txt; : > "$V"
M=$OUT/verdicts_manifeste.txt; : > "$M"
{
  echo "# campagne WP13 $(date '+%F %T') hote $(hostname)"
} >> "$M"
sha256sum "$KVU_BIN" "$KVU_BIN".commit "$D"/*.py "$D"/*.sh "$D"/PREVOL.md "$D"/graines.md "$D"/../../reference/hess/lat.py "$D"/../../reference/hess/balaban_np.py \
    /chemin/vers/projet/outils/ym4_noyaux/noyau3/ym4_hess/hess.py /chemin/vers/projet/outils/ym4_noyaux/noyau3/ym4_hess/_noyaux.py /chemin/vers/projet/outils/ym4_noyaux/noyau3/ym4_hess/__init__.py \
    /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production/L12_b2.3_c1/cfg_002000.npy /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production/L12_b16_c1/cfg_002000.npy >> "$M" 2>> "$M.err" \
  || { echo "REFUS : erreur de hachage des entrées (voir $M.err)" >&2; exit 3; }
echo "commit du binaire : $(cat "$KVU_BIN".commit 2>/dev/null)" >> "$M"
rc_tot=0
run() { local nom=$1; shift; echo "== $nom $(date '+%T')"; "$@" > "$OUT/$nom.out" 2> "$OUT/$nom.err"; local rc=$?; echo "rc=$rc" >> "$OUT/$nom.out"; [ $rc -eq 0 ] || rc_tot=1; echo "   rc=$rc"; }
cd "$D"
run ref_dim_kerdu "$PY" -I "$D/ref_dim_kerdu.py" "$OUT/ref_dim.json"
sha256sum "$OUT/ref_dim.json" >> "$M" || rc_tot=1
run hess_reg "$PY" -I "$D/hess_reg.py" "$V"
run hess_exact "$PY" -I "$D/hess_exact.py" "$V"
run hess_g "$PY" -I "$D/hess_g.py" "$V"
run hess_lmin "$PY" -I "$D/hess_lmin.py" "$V"
run hess_espaces "$PY" -I "$D/hess_espaces.py" "$V"
run hess_jvp "$PY" -I "$D/hess_jvp.py" "$V"
run hess_kerdu "$PY" -I "$D/hess_kerdu.py" "$V" "$OUT/ref_dim.json"
run hess_landau "$PY" -I "$D/hess_landau.py" "$V"
run hess_fils "$PY" -I "$D/hess_fils.py" "$V"
echo "# fin $(date '+%F %T') rc_tot=$rc_tot" | tee -a "$V"
"$PY" -I "$D/verifier.py" "$OUT"; rcv=$?
[ $rcv -eq 0 ] || rc_tot=1
grep -c '^PASS' "$V" | sed 's/^/PASS: /'; grep -c '^FAIL' "$V" | sed 's/^/FAIL: /'
exit $rc_tot
