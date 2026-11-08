#!/bin/bash
# Banc de la chaîne conditionnelle (T5) : chaîne standard de noyau2 (100 + 500 balayages, mesure tous les 2 : 250 mesures,
# D à chaque mesure) en modes ref_rng et rapide, à L = 16, 24, 32, contre le tableau du README de noyau2 (cb.py, 1 fil).
# Usage (sous le verrou GPU de la machine, fenêtre ≤ 10 min — L=32 est le gros morceau) :
#   flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/cond/banc_cond.sh [L…]    (défaut : 16 24 32)
#   KVU_BACKEND=cpu : référence rayon (sans GPU, hors verrou) ; KVU_NSWEEP=500 KVU_NTHERM=100 : longueur de la chaîne
# Les configurations de départ : production L16/L24 de noyau2/data (lues en place) ; L=32 : champ de Haar tiré ici
# (134 Mo, supprimé en sortie : seul V = dec(cfg) importe pour le temps). Sortie : une ligne par cas + tableau.
set -u
cd "$(dirname "$0")/../../../.." || exit 2
RACINE=$(pwd)
export CARGO_TARGET_DIR="${KVU_CIBLE:-${CARGO_TARGET_DIR:-$RACINE/../../targets/T5}}"
FEAT=""; [ -n "${KVU_FEATURES:-}" ] && FEAT="--features ${KVU_FEATURES}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-13.3/lib64"
export PATH="$HOME/.cargo/bin:$PATH"
TMP="${KVU_TMP:-/tmp}/kvu_cond_banc_$$"; mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
LS="${*:-16 24 32}"
NTHERM="${KVU_NTHERM:-100}"; NSWEEP="${KVU_NSWEEP:-500}"; BACKEND="${KVU_BACKEND:-auto}"
PY=python3
for c in /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python "$HOME/ym4_calcul/ym4_noyaux_venv/bin/python"; do [ -x "$c" ] && { PY=$c; break; }; done
DATA="${KVU_NOYAU2_DATA:-/chemin/vers/projet/outils/ym4_noyaux/noyau2/data}"; [ -d "$DATA" ] || DATA="$HOME/ym4_calcul/ym4_noyau2/data"
avail=$(df --output=avail -BG / | tail -1 | tr -d ' G'); echo "SSD libre : ${avail} Go"
[ "$avail" -lt 110 ] 2>/dev/null && [ "$(hostname)" = "acemagic" ] && { echo "ARRET : moins de 110 Go libres sur acemagic"; exit 7; }
nice -n 10 cargo test -j 8 --release $FEAT -p kvu --test cond_cli --no-run --message-format=json > "$TMP/b.json" 2>/dev/null || { echo "FAIL construction"; exit 2; }
CLI=$($PY - "$TMP/b.json" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    try: m = json.loads(l)
    except Exception: continue
    if m.get('executable') and m.get('target', {}).get('name') == 'cond_cli': print(m['executable'])
PY
)
[ -x "$CLI" ] || { echo "FAIL : binaire cond_cli introuvable"; exit 2; }
cond() { KVU_COND_ARGV="$(printf '%s\x1f' "$@" | sed 's/\x1f$//')" "$CLI" --ignored --exact pont_cli --nocapture --test-threads=1; }
# README de noyau2 (acemagic, 1 fil, machine chargée) : secondes de la chaîne standard
declare -A REF_REFRNG=([16]=95 [24]=572 [32]=1692) REF_RAPIDE=([16]=74 [24]=419 [32]=1260)
echo "L | mode | backend | secondes | dont therm+balayages | dont mesures | README cb (s) | gain"
for L in $LS; do
  case $L in
    16|24) CFG="$DATA/L${L}_b2.3.npy" ;;
    *)     CFG="$TMP/haar_L$L.q64.npy"
           $PY - "$L" "$CFG" <<'PY'
import numpy as np, sys
L = int(sys.argv[1]); rng = np.random.default_rng(1000 + L)
q = rng.standard_normal((4, L, L, L, L, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True); np.save(sys.argv[2], q)
PY
           ;;
  esac
  for mode in ref_rng rapide; do
    OUT="$TMP/run_${L}_$mode"
    export KEVINOTRON_SSD_MIN_GO="${KEVINOTRON_SSD_MIN_GO:-0}"
    rep=$(cond chaine --cfg "$CFG" --beta 2.3 --graine 2709 --mode $mode --ntherm "$NTHERM" --nsweep "$NSWEEP" --every 2 --deach 1 --ckpt 0 \
          --out "$OUT" --backend "$BACKEND" --sans-journal 2>"$TMP/err.txt" | grep -o '{.*}' | tail -1)
    if [ -z "$rep" ]; then echo "$L | $mode | ÉCHEC : $(tail -2 "$TMP/err.txt" | tr '\n' ' ')"; rm -rf "$OUT"; continue; fi
    sec=$(echo "$rep" | $PY -c 'import json,sys; print(json.loads(sys.stdin.read())["secondes"])')
    bk=$(grep -o '"backend": "[a-z]*"' "$OUT/manifest.json" | head -1 | cut -d'"' -f4)
    if [ "$mode" = ref_rng ]; then ref=${REF_REFRNG[$L]:-0}; else ref=${REF_RAPIDE[$L]:-0}; fi
    $PY - "$L" "$mode" "$bk" "$sec" "$ref" "$OUT/manifest.json" <<'PY'
import json, sys
L, mode, bk, sec, ref, man = sys.argv[1:7]
t = json.load(open(man)).get("temps_s") or {}
s, ref = float(sec), float(ref)
bal = t.get("thermalisation", 0) + t.get("balayages", 0)
print("%s | %s | %s | %.1f | %.1f | %.1f | %d | x%.1f" % (L, mode, bk, s, bal, t.get("mesures", 0), ref, ref / s if ref else 0))
PY
    rm -rf "$OUT"
  done
  rm -f "$TMP/haar_L$L.q64.npy"
done
