#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gates_dir.sh — gates du lot WP-DIR (boîtes de Dirichlet SU(2) à bord fixé), seuils de GATES-DIR.md (v1 + v2, hachée au journal AVANT les runs).
#
# USAGE (TOUJOURS sous le verrou GPU de la machine pour les phases `produire` qui lancent le GPU, fenêtre ≤ 10 min, ≥ 24/30 Go libres) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock timeout 600 bash gates_dir.sh produire <groupe…>
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock      timeout 600 bash gates_dir.sh produire <groupe…>   (SUR le gamer : pas de numpy)
#   analyse (CPU, acemagic, numpy) : bash gates_dir.sh analyser <groupe…>      (relit SORTIE ; aucun GPU)
# Variables : KVU_BIN (binaire kvu, exigé), SORTIE (dossier de sortie ; défaut $KVU_SCRATCH/gates_dir), BACKEND (hip|cuda ; défaut par la machine),
#   ANCRES (anchors.sha256), SU2_GEN (su2_gen historique, G-DIR-BIT-K1), PYREF (python3 avec numpy), FENETRE (s, défaut 600).
# Sortie : une ligne `PASS|FAIL|SKIP|INFO <ID> : détail` par gate, aussi dans $SORTIE/verdicts.txt ; code de sortie = nombre de FAIL.
#
# Groupes `produire` : bit bord obs qw1 qw1m jauge jaugem agit agitm sc scm regen gain
# Groupes `analyser` : obs qw1 qw1m jauge jaugem agit agitm sc scm regen
# Seeds : GATES-DIR.md §4 (graines CUDA = graines HIP + 500, variable DEC).
set -u
ICI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RACINE=$(cd "$ICI/../../../.." && pwd)
: "${KVU_BIN:?KVU_BIN requis}"
HOTE=$(hostname)
if [ -z "${BACKEND:-}" ]; then case "$HOTE" in *pc-nvidia-exemple*) BACKEND=cuda ;; *) BACKEND=hip ;; esac; fi
if [ "$BACKEND" = cuda ]; then DEC=500; else DEC=0; fi
SORTIE=${SORTIE:-${KVU_SCRATCH:-/tmp}/gates_dir}
PYREF=${PYREF:-python3}
FENETRE=${FENETRE:-600}
mkdir -p "$SORTIE"
V="$SORTIE/verdicts.txt"
NFAIL=0
verdict() { # $1 = PASS|FAIL|SKIP|INFO, $2 = gate, $3 = détail
  echo "$1 $2 : $3" | tee -a "$V"
  [ "$1" = FAIL ] && NFAIL=$((NFAIL + 1))
}
SEUIL_RAM=30; [ "$BACKEND" = cuda ] && SEUIL_RAM=24
MEM=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
verdict INFO G-DIR-ENV "machine=$HOTE backend=$BACKEND MemAvailable=${MEM} Go (seuil ${SEUIL_RAM}) binaire=$(sha256sum "$KVU_BIN" | cut -c1-16) décalage_graines=$DEC"

boite() { # $1 = répertoire de run (NEUF), puis les arguments de `su2 boite` ; écrit $1.log ; rend le rc ; verdict INFO de la ligne PERF
  local rep=$1; shift
  [ -e "$rep" ] && { verdict FAIL G-DIR-RUN "$rep existe déjà (REFUS d'écraser)"; return 99; }
  mkdir -p "$(dirname "$rep")"
  timeout "$FENETRE" "$KVU_BIN" su2 boite --backend "$BACKEND" "$@" --outdir "$rep" > "$rep.log" 2>&1
  local rc=$?
  local perf; perf=$(grep '^PERF boite' "$rep.log" | tail -1)
  verdict INFO G-DIR-PERF "$(basename "$(dirname "$rep")")/$(basename "$rep") rc=$rc ${perf#PERF boite }"
  return $rc
}

# ELLS (facultatif) restreint les cellules d'un groupe `produire` (fenêtres ≤ 10 min : un protocole trop long est COUPÉ en cellules, jamais allégé)
ell_choisi() { [ -z "${ELLS:-}" ] && return 0; case " $ELLS " in *" $1 "*) return 0 ;; esac; return 1; }

# ------------------------------------------------------------------------------------------------------------------
# produire
# ------------------------------------------------------------------------------------------------------------------
produire_bit() {
  if [ "$BACKEND" = hip ]; then
    : "${ANCRES:?ANCRES (anchors.sha256) requis}" "${SU2_GEN:=}"
    local sc="$SORTIE/bit_scratch" pr="$SORTIE/bit_preuves.txt"
    rm -rf "$sc" "$pr" 2>/dev/null
    KVU_BIN="$KVU_BIN" ANCRES="$ANCRES" SU2_GEN="$SU2_GEN" SCRATCH="$sc" PREUVES="$pr" bash "$RACINE/crates/kvu/validation/regen/gbit_preuves.sh" > "$SORTIE/bit.out" 2>&1
    sed -e 's/G-BIT-HIP-PROD/G-DIR-BIT-HIP-PROD/' -e 's/G-BIT-K1/G-DIR-BIT-K1/' "$SORTIE/bit.out" | grep -E '^(PASS|FAIL|SKIP) ' | while read -r l; do
      echo "$l" | tee -a "$V"
    done
    NFAIL=$((NFAIL + $(grep -cE '^FAIL ' "$SORTIE/bit.out")))
  else
    : "${ANCRES:?ANCRES (anchors.sha256) requis}"
    local d="$SORTIE/bit_cuda"; rm -rf "$d"; mkdir -p "$d"
    timeout 600 "$KVU_BIN" su2 gen --backend cuda --precision f64 --options compat-prod --L 8 --beta 2.8 --graine 27092708011 --start hot --therm 2000 --nmeas 20000 --skip 1 --nor 4 --dump-every 1000 --outdir "$d/run" > "$d/run.log" 2>&1
    local rc=$? n=0 f e g
    for f in cfg_020000.npy measurements.csv therm.csv; do
      e=$(awk -v f=$f '$1=="cuda" && $2==f {print $3}' "$ANCRES"); g=$(sha256sum "$d/run/$f" 2>/dev/null | cut -d' ' -f1)
      [ -n "$e" ] && [ "$e" = "$g" ] && n=$((n + 1))
    done
    if [ $rc = 0 ] && [ $n = 3 ]; then verdict PASS G-DIR-BIT-CUDA "$n/3 identiques aux ancres (rc=0)"; else verdict FAIL G-DIR-BIT-CUDA "$n/3 (rc=$rc)"; fi
  fi
}

# G-DIR-BORD-INTACT (10 bords de QW1 à ℓ = 4) et ses mutants
produire_bord() {
  local g=$((26102610020 + DEC))
  boite "$SORTIE/bord/ok" --ell 4 --beta 2.5 --chaines "cold:$g,gauge:$((g+1)),z2full:$((g+2)),z2_01:$((g+3)),flux:$((g+4)),haar:$((g+5)),haar:$((g+6)),haar:$((g+7)),therm:$((g+8)),therm:$((g+9))" \
    --therm 20 --nmeas 100 --skip 1 --nor 2 --profil-tous 10 --therm-source 200
  local m
  for m in masque-ignore masque-ignore-bain masque-ignore-surrelax; do
    boite "$SORTIE/bord/mut_$m" --ell 4 --beta 2.5 --chaines "z2full:$g,cold:$((g+1))" --therm 20 --nmeas 20 --skip 1 --nor 2 --mutant "$m"
  done
}

# G-DIR-OBS : configurations évoluées (dump final) pour la comparaison Rust CPU / GPU / Python de QW1
produire_obs() {
  local g=$((26102610030 + DEC)) l c o
  for l in 4 6; do
    boite "$SORTIE/obs/l$l" --ell "$l" --beta 2.5 --chaines "cold:$g,haar:$((g+1)),z2full:$((g+2)),flux:$((g+3))" --therm 20 --nmeas 1 --skip 1 --nor 2 --profil-tous 1 --dump-final
    # observables de chaque configuration finale : hôte (référence Rust) et GPU (noyaux), relues par `su2 boite-obs`
    for c in 0 1 2 3; do
      o="$SORTIE/obs/l$l/obs_c$c"; mkdir -p "$o"
      "$KVU_BIN" su2 boite-obs --cfg "$SORTIE/obs/l$l/c00$c/config_finale.npy" --ell "$l" --backend cpu > "$o/cpu.json" 2> "$o/cpu.err"
      "$KVU_BIN" su2 boite-obs --cfg "$SORTIE/obs/l$l/c00$c/config_finale.npy" --ell "$l" --backend "$BACKEND" > "$o/gpu.json" 2> "$o/gpu.err"
    done
  done
}

# G-DIR-QW1 (ℓ = 8, 6, 4, 2 ; seeds 26102611000 + 1000·j + c, c = 0…31 : cold 0…15, z2full 16…31) — `qw1m` = mutant masque ignoré (ℓ = 8 et 4)
produire_qw1() { # $1 = '' | masque-ignore
  local mut=${1:-} spec l j g tag=qw1
  [ -n "$mut" ] && tag=qw1m
  for spec in "8 0" "6 1" "4 2" "2 3"; do
    set -- $spec; l=$1; j=$2
    ell_choisi "$l" || continue
    if [ -n "$mut" ] && [ "$l" != 8 ] && [ "$l" != 4 ]; then continue; fi
    g=$((26102611000 + 1000 * j + DEC))
    boite "$SORTIE/$tag/l$l" --ell "$l" --beta 2.5 --bords cold,z2full --repliques 16 --graine "$g" --graine-pas 1 --therm 1000 --nmeas 30000 --skip 1 --nor 2 --profil-tous 10 ${mut:+--mutant "$mut"}
  done
}

# G-DIR-JAUGE (ℓ = 2, 4, 6, 8, 12, 16 ; seeds 26102612000 + 100·j + c ; cold 0…7, gauge (ou haar pour le mutant) 8…15)
produire_jauge() { # $1 = '' | haar
  local alt=${1:-gauge} tag=jauge spec l j g nm
  [ "$alt" = haar ] && tag=jaugem
  for spec in "2 0" "4 1" "6 2" "8 3" "12 4" "16 5"; do
    set -- $spec; l=$1; j=$2
    ell_choisi "$l" || continue
    if [ "$alt" = haar ] && [ "$l" -gt 4 ]; then continue; fi
    nm=20000; [ "$l" -ge 12 ] && nm=5000
    g=$((26102612000 + 100 * j + DEC))
    boite "$SORTIE/$tag/l$l" --ell "$l" --beta 2.5 --bords "cold,$alt" --repliques 8 --graine "$g" --graine-pas 1 --therm 1000 --nmeas "$nm" --skip 1 --nor 2 --profil-tous 0
  done
}

# G-DIR-AGIT (ℓ = 2, β = 2,5, cold 0…7 / z2full 8…15, 1000 + 5000 pas ; mutant : masque ignoré)
produire_agit() { # $1 = '' | masque-ignore
  local mut=${1:-} tag=agit g=$((26102612000 + DEC))
  [ -n "$mut" ] && tag=agitm
  boite "$SORTIE/$tag/l2" --ell 2 --beta 2.5 --bords cold,z2full --repliques 8 --graine "$g" --graine-pas 1 --therm 1000 --nmeas 5000 --skip 1 --nor 2 --profil-tous 0 ${mut:+--mutant "$mut"}
}

# G-DIR-SC (β = 0,2, ℓ = 6, bords cold, z2full, flux, haar × 8, seeds 26102613000 + c ; mutant : β = 0,21 simulé, analysé à 0,2)
produire_sc() { # $1 = '' | 0.21
  local b=${1:-0.2} tag=sc g=$((26102613000 + DEC))
  [ "$b" != 0.2 ] && tag=scm
  boite "$SORTIE/$tag/l6" --ell 6 --beta "$b" --bords cold,z2full,flux,haar --repliques 8 --graine "$g" --graine-pas 1 --therm 1000 --nmeas 10000 --skip 1 --nor 2 --profil-tous 0
}

# G-DIR-MANIFESTE / G-DIR-REGEN : ℓ = 4, 3 bords, 200 pas
produire_regen() {
  local g=$((26102610040 + DEC))
  boite "$SORTIE/regen/origine" --ell 4 --beta 2.5 --chaines "cold:$g,z2full:$((g+1)),haar:$((g+2))" --therm 20 --nmeas 180 --skip 1 --nor 2 --profil-tous 10 --dump-final
  local man="$SORTIE/regen/origine/manifest_regen.json"
  timeout "$FENETRE" "$KVU_BIN" regen "$man" --sortie "$SORTIE/regen/rejeu" > "$SORTIE/regen/rejeu.log" 2>&1
  local rc=$?
  if [ $rc = 0 ] && grep -q "^PASS REGEN" "$SORTIE/regen/rejeu.log"; then verdict PASS G-DIR-REGEN "$(grep '^PASS REGEN' "$SORTIE/regen/rejeu.log" | cut -c1-120) ($BACKEND)"; else verdict FAIL G-DIR-REGEN "rc=$rc ($(tail -2 "$SORTIE/regen/rejeu.log" | tr '\n' ' '))"; fi
  # mutant : une graine modifiée dans la commande du manifeste (la sortie du rejeu doit différer : rc 8)
  mkdir -p "$SORTIE/regen/faux"
  $PYREF - "$man" "$SORTIE/regen/faux/manifest_regen.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
c = m["commande"]
i = c.index("--chaines")
parts = c[i + 1].split(",")
b, g = parts[-1].split(":")
parts[-1] = f"{b}:{int(g) + 1}"
c[i + 1] = ",".join(parts)
json.dump(m, open(sys.argv[2], "w"), indent=1)
PYEOF
  timeout "$FENETRE" "$KVU_BIN" regen "$SORTIE/regen/faux/manifest_regen.json" --sortie "$SORTIE/regen/rejeu_faux" > "$SORTIE/regen/rejeu_faux.log" 2>&1
  rc=$?
  if [ $rc = 8 ]; then verdict PASS G-DIR-REGEN-MUTANT "graine modifiée ⇒ rc 8 (ROUGE voulu)"; else verdict FAIL G-DIR-REGEN-MUTANT "rc=$rc au lieu de 8 : ECHEC_OUTIL"; fi
}

# G-DIR-GAIN (INFO) : temps par pas en régime établi, ℓ ∈ {8, 12, 16}, K ∈ {1, 16, 64}
produire_gain() {
  local l k g=$((26102614000 + DEC))
  for l in ${GAIN_L:-8 12 16}; do
    for k in ${GAIN_K:-1 16 64}; do
      boite "$SORTIE/gain/l${l}_k${k}" --ell "$l" --beta 2.5 --bords cold --repliques "$k" --graine "$g" --graine-pas 1 --therm "${GAIN_THERM:-20}" --nmeas "${GAIN_NMEAS:-100}" --skip 1 --nor 2 --profil-tous 10
    done
  done
}

# ------------------------------------------------------------------------------------------------------------------
# analyser (CPU, numpy)
# ------------------------------------------------------------------------------------------------------------------
lignes() { # relaie les lignes PASS|FAIL|INFO|SKIP d'un script d'analyse et compte les FAIL
  local f=$1 rc=$2
  grep -E '^(PASS|FAIL|INFO|SKIP) ' "$f" | tee -a "$V"
  NFAIL=$((NFAIL + $(grep -cE '^FAIL ' "$f")))
  [ "$rc" = 0 ] || [ "$rc" = 1 ] || verdict FAIL G-DIR-ANALYSE "rc=$rc ($f)"
}

analyser_obs() {
  local l c
  for l in 4 6; do
    for c in 0 1 2 3; do
      local cfg="$SORTIE/obs/l$l/c00$c/config_finale.npy" o="$SORTIE/obs/l$l/obs_c$c"
      $PYREF -I "$ICI/ref_obs.py" --cfg "$cfg" --ell "$l" > "$o/py.json" 2> "$o/py.err"
      $PYREF -I "$ICI/ref_obs.py" --cfg "$cfg" --ell "$l" --centre-decale > "$o/py_decale.json" 2> "$o/py_decale.err"
      $PYREF -I "$ICI/analyse_dir.py" obs-cmp "$o/cpu.json" "$o/py.json" --tol 1e-12 > "$o/cmp_cpu_py.out"; lignes "$o/cmp_cpu_py.out" $?
      $PYREF -I "$ICI/analyse_dir.py" obs-cmp "$o/gpu.json" "$o/cpu.json" --tol 1e-14 > "$o/cmp_gpu_cpu.out"; lignes "$o/cmp_gpu_cpu.out" $?
      # mutant M-DIR-8 : référence au centre décalé ⇒ doit être ROUGE à 1e-3
      $PYREF -I "$ICI/analyse_dir.py" obs-cmp "$o/cpu.json" "$o/py_decale.json" --tol 1e-3 > "$o/cmp_mutant.out"
      if [ $? = 1 ]; then verdict PASS G-DIR-OBS-MUTANT "ℓ=$l chaîne $c : centre décalé ⇒ écart > 1e-3 (ROUGE voulu)"; else verdict FAIL G-DIR-OBS-MUTANT "ℓ=$l chaîne $c : le centre décalé n'est pas détecté (ECHEC_OUTIL)"; fi
    done
  done
}

analyser_runs() { # $1 = sous-commande, $2 = tag, puis options propres
  local sc=$1 tag=$2; shift 2
  local args=() d
  for d in "$@"; do args+=("$d"); done
  $PYREF -I "$ICI/analyse_dir.py" "$sc" "${args[@]}" > "$SORTIE/analyse_${tag}.out" 2> "$SORTIE/analyse_${tag}.err"
  lignes "$SORTIE/analyse_${tag}.out" $?
}

# Vérifie les manifestes de tous les runs produits (schéma kvu-regen/1, et rejet de chaque champ retiré)
analyser_regen() {
  local m
  for m in "$SORTIE"/regen/origine/manifest_regen.json "$SORTIE"/qw1/l8/manifest_regen.json "$SORTIE"/bord/ok/manifest_regen.json; do
    [ -f "$m" ] || continue
    if python3 -I "$RACINE/scripts/regen_valider.py" "$m" > "$m.valid.out" 2>&1; then verdict PASS G-DIR-MANIFESTE "$m valide au schéma kvu-regen/1"; else verdict FAIL G-DIR-MANIFESTE "$m : $(head -3 "$m.valid.out" | tr '\n' ' ')"; fi
  done
  m="$SORTIE/regen/origine/manifest_regen.json"
  if [ -f "$m" ]; then
    if python3 -I "$RACINE/scripts/regen_valider.py" --mutants "$m" > "$SORTIE/regen/mutants_schema.out" 2>&1; then verdict PASS G-DIR-MANIFESTE-MUTANTS "$(tail -1 "$SORTIE/regen/mutants_schema.out")"; else verdict FAIL G-DIR-MANIFESTE-MUTANTS "$(tail -2 "$SORTIE/regen/mutants_schema.out" | tr '\n' ' ')"; fi
  fi
  # contrôle du bord et des paramètres consignés (G-DIR-BORD-INTACT : lecture des manifestes des runs de `bord`)
  local d
  for d in "$SORTIE"/bord/ok "$SORTIE"/bord/mut_*; do
    [ -f "$d/manifest.json" ] || continue
    python3 -I - "$d/manifest.json" "$(basename "$d")" <<'PYEOF' | tee -a "$V"
import json, sys
m = json.load(open(sys.argv[1]))
nom = sys.argv[2]
c = m["controle_bord"]
if nom == "ok":
    print(("PASS" if c["intact"] and c["max_dq"] == 0.0 else "FAIL"), f"G-DIR-BORD-INTACT : 10 bords, max|ΔU_bord| = {c['max_dq']}, intact = {c['intact']}")
else:
    print(("PASS" if (not c["intact"] and c["max_dq"] > 0.1) else "FAIL"), f"G-DIR-BORD-INTACT-MUTANT {nom} : max|ΔU_bord| = {c['max_dq']:.3g}, intact = {c['intact']} (ROUGE voulu)")
PYEOF
  done
}

# ------------------------------------------------------------------------------------------------------------------
cmd=${1:-}; shift || true
[ -n "$cmd" ] || { sed -n 2,14p "$0"; exit 2; }
for g in "$@"; do
  case "$cmd:$g" in
    produire:bit) produire_bit ;;
    produire:bord) produire_bord ;;
    produire:obs) produire_obs ;;
    produire:qw1) produire_qw1 ;;
    produire:qw1m) produire_qw1 masque-ignore ;;
    produire:jauge) produire_jauge gauge ;;
    produire:jaugem) produire_jauge haar ;;
    produire:agit) produire_agit ;;
    produire:agitm) produire_agit masque-ignore ;;
    produire:sc) produire_sc 0.2 ;;
    produire:scm) produire_sc 0.21 ;;
    produire:regen) produire_regen ;;
    produire:gain) produire_gain ;;
    analyser:obs) analyser_obs ;;
    analyser:qw1) analyser_runs qw1 qw1 --run "8=$SORTIE/qw1/l8" --run "6=$SORTIE/qw1/l6" --run "4=$SORTIE/qw1/l4" --run "2=$SORTIE/qw1/l2" ;;
    analyser:qw1m) analyser_runs qw1 qw1m --mutant --run "8=$SORTIE/qw1m/l8" --run "4=$SORTIE/qw1m/l4" ;;
    analyser:jauge) analyser_runs jauge jauge --run "2=$SORTIE/jauge/l2" --run "4=$SORTIE/jauge/l4" --run "6=$SORTIE/jauge/l6" --run "8=$SORTIE/jauge/l8" --run "12=$SORTIE/jauge/l12" --run "16=$SORTIE/jauge/l16" ;;
    analyser:jaugem) analyser_runs jauge jaugem --mutant --run "2=$SORTIE/jaugem/l2" --run "4=$SORTIE/jaugem/l4" ;;
    analyser:agit) analyser_runs agit agit --run "2=$SORTIE/agit/l2" ;;
    analyser:agitm) analyser_runs agit agitm --mutant --run "2=$SORTIE/agitm/l2" ;;
    analyser:sc) analyser_runs sc sc --run "$SORTIE/sc/l6" ;;
    analyser:scm) analyser_runs sc scm --mutant --beta-nominal 0.2 --run "$SORTIE/scm/l6" ;;
    analyser:regen) analyser_regen ;;
    *) echo "gates_dir : groupe inconnu « $cmd $g »" >&2; exit 2 ;;
  esac
done
echo "# fin : $(grep -c '^FAIL' "$V") FAIL (verdicts : $V)"
exit "$(grep -c '^FAIL' "$V")"
