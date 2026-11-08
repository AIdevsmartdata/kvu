#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gates_wp10.sh — gates GPU propres à WP10 (PREREG v1 §4) sur le binaire de la campagne :
#   RG-GPU-PROMO         (= ENLIGNE-F32) `su2 gen --precision f32 --mesures bloc:3:1` (champ F32 promu en f64 puis renormalisé en F64) = `rg bloc` CPU sur les
#                        dumps du même run (qui sont aussi promus et renormalisés), ≤ 1e-12
#   RG-GPU-PROMO-INCHANGE le run avec et sans --mesures écrit les mêmes dumps et CSV (octets)
#   G-DET-PROMO         deux runs identiques (même graine) ⇒ mêmes octets (dumps, CSV, configurations grossières)
#   WP10-REPRISE-<cas>   des fenêtres de temps (--fenetre-s) enchaînées par --reprendre donnent les mêmes octets qu'un run sans coupure : F32 en production (L = 18), F64 de contrôle
#                        (L = 18, bloc sur champ F64 résident), F32 avec coupures pendant la thermalisation (L = 12)
# USAGE (le script entier tient en une fenêtre ≤ 10 min sous le verrou GPU de la machine, pris une fois par l'appelant) :
#   acemagic : flock -E 75 -s -w 7200 /tmp/kevinotron_gpu.lock gpu_partage -w 7200 -- env LOCK_TENU=1 KVU_BIN=<kvu> BACKEND=hip  LOCK=/tmp/kevinotron_gpu.lock OUT=<dossier neuf> CONF=<kvu.toml> ANCRES=<anchors.sha256> bash gates_wp10.sh
#   gamer    : flock -E 75 -s -w 7200 /tmp/gamer_gpu.lock      gpu_partage -w 7200 -- env LOCK_TENU=1 KVU_BIN=<kvu> BACKEND=cuda LOCK=/tmp/gamer_gpu.lock      OUT=<dossier neuf> CONF=<kvu.toml> ANCRES=<anchors.sha256> bash gates_wp10.sh
#   (sans LOCK_TENU, chaque commande prend le verrou : exclusif par défaut, partagé avec MODE_VERROU=partage)
# Variables : KVU_BIN, BACKEND, LOCK, OUT (dossier NEUF), CONF (kvu.toml de la campagne), ANCRES (anchors.sha256 : G-BIT), MODE_VERROU (excl|partage), FILS (CPU de référence, défaut 8). Sortie : lignes PASS|FAIL|SKIP|INFO ; code = nombre de FAIL + SKIP.
# Graines : 26102619xxx (série de qualification, exclue de l'analyse de la campagne).
set -u
export LC_ALL=C
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
[ -n "${CONF:-}" ] && export KEVINOTRON_CONFIG="$CONF"   # kvu.toml de la campagne : profil de machine (verrou, garde SSD) comme en production
: "${KVU_BIN:?KVU_BIN requis}" "${BACKEND:?BACKEND requis (hip|cuda)}" "${LOCK:?LOCK requis}" "${OUT:?OUT requis}"
FILS=${FILS:-8}
[ -x "$KVU_BIN" ] || { echo "FAIL BUILD binaire $KVU_BIN introuvable"; exit 1; }
[ -e "$OUT" ] && { echo "REFUS : $OUT existe déjà (dossier neuf requis)"; exit 2; }
mkdir -p "$OUT" || exit 2
NB=0
verdict() { echo "$1 $2 $3"; case "$1" in FAIL|SKIP) NB=$((NB + 1)) ;; esac; }
# MODE_VERROU=partage : verrou partagé (`flock -s` extérieur + gpu_partage), car des jobs partagés continus des autres chantiers affament le verrou exclusif ;
# ces gates (identité binaire, tolérance 1e-12) ne dépendent pas du mode de verrou. Défaut : exclusif.
# LOCK_TENU=1 : l'appelant tient déjà le verrou GPU pour tout le script (≤ 10 min) ; aucun verrou par commande.
gpu() {
  if [ "${LOCK_TENU:-0}" = 1 ]; then timeout 600 "$@"
  elif [ "${MODE_VERROU:-excl}" = partage ]; then flock -E 75 -s -w 7200 "$LOCK" gpu_partage -w 7200 -- timeout 600 "$@"
  else flock -w 7200 "$LOCK" timeout 600 "$@"; fi
}

# --- 0. G-BIT : ancre du moteur F64 sur le binaire de la campagne (le binaire change, pas les noyaux) ; mêmes commandes que GATES v1 §1/WP1
ANCRES=${ANCRES:-}
if [ -n "$ANCRES" ] && [ -f "$ANCRES" ]; then
  GB="$OUT/gbit"; mkdir -p "$GB"
  if [ "$BACKEND" = hip ]; then
    gpu "$KVU_BIN" su2 gen --backend hip --precision f64 --options compat-prod --L 12 --beta 2.5 --graine 26095612011 --start hot --therm 5000 --nmeas 2000 --skip 1 --nor 4 --dump-every 100 --outdir "$GB/run" > "$GB/run.log" 2>&1; rcb=$?
    nom=G-BIT-HIP-PROD; cible=cfg_002000.npy; ref=hip
  else
    gpu "$KVU_BIN" su2 gen --backend cuda --precision f64 --options compat-prod --L 8 --beta 2.8 --graine 27092708011 --start hot --therm 2000 --nmeas 20000 --skip 1 --nor 4 --dump-every 1000 --outdir "$GB/run" > "$GB/run.log" 2>&1; rcb=$?
    nom=G-BIT-CUDA; cible=cfg_020000.npy; ref=cuda
  fi
  nb=0; for f in $cible measurements.csv therm.csv; do e=$(awk -v b=$ref -v f=$f '$1==b && $2==f {print $3}' "$ANCRES"); g=$(sha256sum "$GB/run/$f" 2>/dev/null | cut -d' ' -f1); [ -n "$e" ] && [ "$e" = "$g" ] && nb=$((nb + 1)); done
  [ $rcb = 0 ] && [ $nb = 3 ] && verdict PASS "$nom" "$nb/3 octets identiques aux lignes $ref de anchors.sha256 (rc=$rcb, binaire $(sha256sum "$KVU_BIN" | cut -c1-16))" || verdict FAIL "$nom" "$nb/3 (rc=$rcb)"
else verdict SKIP G-BIT "ANCRES absent : fichier anchors.sha256 requis"; fi

# --- 1. RG-GPU-PROMO et RG-GPU-PROMO-INCHANGE (L = 12, β = 4, 100 pas de thermalisation + 10 mesures × skip 10, dumps q64 à chaque mesure)
E1="$OUT/enligne"; mkdir -p "$E1"
COMMUN=(su2 gen --backend "$BACKEND" --precision f32 --options rapide --L 12 --beta 4 --graine 26102619101 --start hot --therm 100 --nmeas 10 --skip 10 --nor 4 --dump-every 1 --format q64)
gpu "$KVU_BIN" "${COMMUN[@]}" --mesures bloc:3:1 --outdir "$E1/avec" > "$E1/avec.log" 2>&1; ra=$?
gpu "$KVU_BIN" "${COMMUN[@]}" --outdir "$E1/sans" > "$E1/sans.log" 2>&1; rs=$?
if [ $ra != 0 ] || [ $rs != 0 ]; then
  verdict FAIL RG-GPU-PROMO "rc avec=$ra sans=$rs (journaux $E1/avec.log, $E1/sans.log)"
else
  mkdir -p "$E1/A" "$E1/B"; ok=1; det=""
  for m in 1 2 3 4 5 6 7 8 9 10; do
    pas=$((100 + 10 * m)); on=$(printf "%s/avec/bloc3_cfg_%08d.npy" "$E1" "$pas"); dump=$(printf "%s/avec/cfg_%06d.npy" "$E1" "$m")
    if [ ! -f "$on" ] || [ ! -f "$dump" ]; then ok=0; det="$det m$m absent;"; continue; fi
    cp "$on" "$E1/A/m$m.npy"
    "$KVU_BIN" rg bloc --backend cpu --fils "$FILS" --entree "$dump" --sortie "$E1/B/m$m.npy" > "$E1/B/m$m.log" 2>&1 || { ok=0; det="$det rg bloc m$m rc≠0;"; }
  done
  "$KVU_BIN" rg comparer --a "$E1/A" --b "$E1/B" --min 10 --id RG-GPU-PROMO --sans-theta > "$E1/comparer.out" 2>&1; rc=$?
  grep -E "^(PASS|FAIL) " "$E1/comparer.out" | sed "s/^\(PASS\|FAIL\) RG-GPU-PROMO /\1 RG-GPU-PROMO $BACKEND : configurations grossières en ligne (champ F32 promu et renormalisé) contre rg bloc CPU sur les dumps ; /"
  [ $rc != 0 ] && NB=$((NB + 1))
  [ $ok = 1 ] || verdict FAIL RG-GPU-PROMO-FICHIERS "$det"
  n=0; d=""
  for f in $(cd "$E1/sans" && ls cfg_*.npy measurements.csv therm.csv); do
    if cmp -s "$E1/sans/$f" "$E1/avec/$f"; then n=$((n + 1)); else d="$d $f"; fi
  done
  tot=$(cd "$E1/sans" && ls cfg_*.npy measurements.csv therm.csv | wc -l)
  [ "$n" = "$tot" ] && verdict PASS RG-GPU-PROMO-INCHANGE "$BACKEND : $n/$tot fichiers identiques à l'octet avec et sans --mesures bloc:3 (F32)" || verdict FAIL RG-GPU-PROMO-INCHANGE "$n/$tot, différents :$d"
fi

# --- 2. G-DET-PROMO : le même run avec mesures, une seconde fois
gpu "$KVU_BIN" "${COMMUN[@]}" --mesures bloc:3:1 --outdir "$E1/avec2" > "$E1/avec2.log" 2>&1; r2=$?
if [ $r2 != 0 ]; then verdict FAIL G-DET-PROMO "rc=$r2"; else
  n=0; d=""; tot=0
  for f in $(cd "$E1/avec" && ls cfg_*.npy measurements.csv therm.csv bloc3.csv bloc3_hist.csv bloc3_cfg_*.npy); do
    tot=$((tot + 1)); if cmp -s "$E1/avec/$f" "$E1/avec2/$f"; then n=$((n + 1)); else d="$d $f"; fi
  done
  [ "$n" = "$tot" ] && [ "$tot" -gt 20 ] && verdict PASS G-DET-PROMO "$BACKEND : deux runs F32 + bloc:3, même graine : $n/$tot fichiers identiques à l'octet" || verdict FAIL G-DET-PROMO "$n/$tot, différents :$d"
fi

# --- 3. WP10-REPRISE : fenêtres de temps enchaînées contre run sans coupure, en trois cas (relecture de Sol, point 12) :
#   F32 production (L = 18, coupures en production), F64 contrôle (L = 18, bloc sur champ F64 résident), F32 avec coupures PENDANT la thermalisation (L = 12)
reprise_cas() { # reprise_cas <nom> <précision> <options> <L> <beta> <graine> <therm> <nmeas> <skip> <fenêtre s>
  local nom=$1 prec=$2 opt=$3 L=$4 beta=$5 graine=$6 th=$7 nm=$8 sk=$9 fe=${10}
  local R="$OUT/reprise_$nom"; mkdir -p "$R"
  local CR=(su2 gen --backend "$BACKEND" --precision "$prec" --options "$opt" --L "$L" --beta "$beta" --graine "$graine" --start hot --therm "$th" --nmeas "$nm" --skip "$sk" --nor 4 --reprise-every 0)
  gpu "$KVU_BIN" "${CR[@]}" --mesures bloc:3:1 --outdir "$R/ref" > "$R/ref.log" 2>&1; local rr=$?
  local fen=1 rcw=0 ct=0 mf
  coupe_therm() { mf=$(sed -n 's/.*"mesures_faites": *\([0-9]*\).*/\1/p' "$R/fen/reprise.json" 2>/dev/null | head -1); [ ! -f "$R/fen/COMPLETE" ] && [ "${mf:-1}" = 0 ] && ct=$((ct + 1)); }
  gpu "$KVU_BIN" "${CR[@]}" --mesures bloc:3:1 --fenetre-s "$fe" --outdir "$R/fen" > "$R/fen_1.log" 2>&1; rcw=$?; coupe_therm
  while [ $rcw = 0 ] && [ ! -f "$R/fen/COMPLETE" ] && [ $fen -lt 400 ]; do
    fen=$((fen + 1))
    gpu "$KVU_BIN" su2 gen --backend "$BACKEND" --reprendre "$R/fen" --mesures bloc:3:1 --fenetre-s "$fe" > "$R/fen_$fen.log" 2>&1; rcw=$?; coupe_therm
  done
  if [ $rr != 0 ] || [ $rcw != 0 ] || [ ! -f "$R/fen/COMPLETE" ]; then
    verdict FAIL "WP10-REPRISE-$nom" "rc référence=$rr, fenêtres=$rcw après $fen lancements, COMPLETE=$([ -f "$R/fen/COMPLETE" ] && echo oui || echo non)"
  elif [ "$fen" -lt 3 ]; then
    verdict FAIL "WP10-REPRISE-$nom" "seulement $fen fenêtre(s) : le run n'a pas été assez coupé, la gate serait sans puissance (réduire --fenetre-s)"
  else
    local n=0 d="" tot=0 nf
    for f in $(cd "$R/ref" && ls measurements.csv therm.csv bloc3.csv bloc3_hist.csv bloc3_cfg_*.npy); do
      tot=$((tot + 1)); if cmp -s "$R/ref/$f" "$R/fen/$f"; then n=$((n + 1)); else d="$d $f"; fi
    done
    nf=$(cd "$R/fen" && ls measurements.csv therm.csv bloc3.csv bloc3_hist.csv bloc3_cfg_*.npy | wc -l)
    # coupures pendant la thermalisation : fenêtres qui ont laissé un point de reprise avec mesures_faites = 0 (compté dans la boucle)
    if [ "$nom" = f32_therm ] && [ "$ct" -lt 2 ]; then verdict FAIL "WP10-REPRISE-$nom" "seulement $ct fenêtre(s) coupée(s) pendant la thermalisation : gate sans puissance"; return; fi
    [ "$n" = "$tot" ] && [ "$nf" = "$tot" ] && verdict PASS "WP10-REPRISE-$nom" "$BACKEND ($prec, L = $L) : $fen fenêtres de $fe s enchaînées par --reprendre, dont $ct coupées pendant la thermalisation ; $n/$tot fichiers identiques au run sans coupure ; manifeste kvu-regen/1 : $([ -f "$R/fen/regen_bloc3.json" ] && echo présent || echo ABSENT)" || verdict FAIL "WP10-REPRISE-$nom" "$n/$tot identiques, $nf fichiers côté fenêtres, différents :$d"
  fi
}
reprise_cas f32_prod  f32 rapide     18 8 26102619201 60   24 5 0.6
reprise_cas f64_ctrl  f64 compat-prod 18 8 26102619211 60   24 5 0.8
reprise_cas f32_therm f32 rapide     12 8 26102619221 3000 8  4 0.8
echo "INFO gates_wp10.sh $BACKEND : $NB (FAIL + SKIP)"
exit $NB
