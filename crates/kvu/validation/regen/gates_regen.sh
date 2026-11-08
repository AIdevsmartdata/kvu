#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gates_regen.sh — gates WP4 (GATES.md v1 §1/WP4) : REGEN-SCHEMA, REGEN-1, REGEN-2, REGEN-3, REGEN-INTEG, AUDIT-7, AUDIT-INTACT, T0-PROD.
# (REGEN-4 et la partie Rust de REGEN-SCHEMA sont des tests : `cargo test --release --offline regen` ; WP1-DEPS : `cargo tree`.)
#
# USAGE (depuis n'importe où) :  KVU_BIN=<kvu> bash gates_regen.sh <gate…>      gates : schema gbit regen1 regen2 regen3 integ audit avant intact prod
# Les gates GPU (gbit regen1 regen2 regen3) prennent EUX-MÊMES `flock -w 7200 /tmp/kevinotron_gpu.lock` (fenêtre ≤ 10 min, timeout 600 s
# par rejeu) et refusent de partir sous 30 Go de MemAvailable. Rien n'est écrit dans un ensemble : tout va dans $SCRATCH (neuf).
# Variables : KVU_BIN (obligatoire), U (racine du chantier), SCRATCH (défaut $U/scratch/wp4/gates/final), ENSEMBLES (fichier des 7 ensembles).
# Sortie : une ligne `PASS|FAIL|SKIP <ID> <détail>` par gate ; code de sortie = nombre de FAIL.
set -u
: "${KVU_BIN:?KVU_BIN requis}"
U=${U:-/chemin/vers/projet/chantiers/kevinotron_unification}
DEPOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
SCRATCH=${SCRATCH:-$U/scratch/wp4/gates/final}
ENSEMBLES=${ENSEMBLES:-$DEPOT/crates/kvu/validation/regen/ensembles_audit.txt}
ANCRES=${ANCRES:-/chemin/vers/projet/chantiers/vision_20261025/decisions/K5_su2gpu_depot/depot/su2-heatbath-gpu/tests/anchors.sha256}
SU2_GEN=${SU2_GEN:-/chemin/vers/projet/chantiers/brancheC_saturation_v3_20261004/kev/target/release/su2_gen}
GRANDL=/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production
YM4=/chemin/vers/projet/chantiers/ym4_mass_gap_0510/ens
LOCK=/tmp/kevinotron_gpu.lock
PREUVES=$SCRATCH/preuves_gbit.txt
NFAIL=0
mkdir -p "$SCRATCH" || exit 2
verdict() { echo "$1 $2 $3"; [ "$1" = FAIL ] && NFAIL=$((NFAIL + 1)); }
sain() { case "$1" in "$U"/scratch/wp4/*) return 0 ;; *) echo "REFUS : $1 hors de $U/scratch/wp4"; exit 2 ;; esac; }
sain "$SCRATCH"
ram_ok() { local m; m=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo); [ "${m:-0}" -ge 30 ] || { verdict FAIL "$1" "MemAvailable ${m} Go < 30 Go : fenêtre refusée"; return 1; }; }
gpu() { flock -w 7200 "$LOCK" timeout 700 "$@"; }

choix_regen2() { # règle gelée : ens/L16_* de plus petit wall_s (égalité : ordre lexicographique), wall_s ≤ 400 sinon la suivante
  python3 -I - "$YM4" <<'PY'
import glob, json, os, sys
c = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], "L16_*"))):
    m = json.load(open(os.path.join(d, "manifest.json")))
    c.append((m["wall_s"], os.path.basename(d)))
c.sort()
for w, n in c:
    if w <= 400:
        print(n, w)
        break
PY
}

gate_schema() {
  local d="$SCRATCH/schema"; rm -rf "$d"; mkdir -p "$d"
  [ -f "$PREUVES" ] && PR="--preuve-gbit $PREUVES" || PR=""
  "$KVU_BIN" regen "$GRANDL/L12_b2.5_c1/manifest.json" --adapter-seulement --hacher --vers "$d/adapte_su2_hip.json" $PR > "$d/a1.log" 2>&1; local r1=$?
  local ch; ch=$(choix_regen2 | cut -d' ' -f1)
  "$KVU_BIN" regen "$YM4/$ch/manifest.json" --adapter-seulement --vers "$d/adapte_su2_gen.json" $PR > "$d/a2.log" 2>&1; local r2=$?
  python3 -I "$DEPOT/scripts/regen_valider.py" "$d/adapte_su2_hip.json" "$d/adapte_su2_gen.json" > "$d/valide.log" 2>&1; local r3=$?
  python3 -I "$DEPOT/scripts/regen_valider.py" --mutants "$d/adapte_su2_hip.json" > "$d/mut_hip.log" 2>&1; local r4=$?
  python3 -I "$DEPOT/scripts/regen_valider.py" --mutants "$d/adapte_su2_gen.json" > "$d/mut_gen.log" 2>&1; local r5=$?
  if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ] && [ $r4 -eq 0 ] && [ $r5 -eq 0 ]; then
    verdict PASS REGEN-SCHEMA "2 manifestes adaptés (su2_hip/1 $(basename "$GRANDL")/L12_b2.5_c1, su2_gen/1 $ch) acceptés ; $(tail -1 "$d/mut_hip.log" | sed 's/^PASS //') ; $(tail -1 "$d/mut_gen.log" | sed 's/^PASS //')"
  else
    verdict FAIL REGEN-SCHEMA "rc adaptation $r1/$r2, validation $r3, mutants $r4/$r5 (voir $d)"
  fi
}

gate_gbit() {
  ram_ok G-BIT-PREUVES || return
  rm -rf "$SCRATCH/preuves_tmp" "$PREUVES"
  gpu env KVU_BIN="$KVU_BIN" ANCRES="$ANCRES" SU2_GEN="$SU2_GEN" SCRATCH="$SCRATCH/preuves_tmp" PREUVES="$PREUVES" \
    bash "$DEPOT/crates/kvu/validation/regen/gbit_preuves.sh" > "$SCRATCH/gbit.log" 2>&1; local r=$?
  cat "$SCRATCH/gbit.log" | grep -E '^(PASS|FAIL|SKIP) ' | while IFS= read -r l; do echo "$l"; done
  [ $r -eq 0 ] || NFAIL=$((NFAIL + r))
}

gate_regen1() {
  ram_ok REGEN-1 || return
  local d="$SCRATCH/regen1"; rm -rf "$d"
  gpu "$KVU_BIN" regen "$GRANDL/L12_b2.5_c1/manifest.json" --ancres "$ANCRES" --preuve-gbit "$PREUVES" --sortie "$d" > "$SCRATCH/regen1.log" 2>&1; local r=$?
  # contrôle INDÉPENDANT de `kvu regen` : sha256sum des trois fichiers rejoués contre les originaux ET anchors.sha256
  local n=0 f a o s
  for f in cfg_002000.npy measurements.csv therm.csv; do
    a=$(awk -v f="$f" '$1=="hip" && $2==f {print $3}' "$ANCRES"); o=$(sha256sum "$GRANDL/L12_b2.5_c1/$f" | cut -d' ' -f1); s=$(sha256sum "$d/rejeu/$f" 2>/dev/null | cut -d' ' -f1)
    [ -n "$s" ] && [ "$s" = "$o" ] && [ "$s" = "$a" ] && n=$((n + 1))
  done
  if [ $r -eq 0 ] && [ $n -eq 3 ]; then verdict PASS REGEN-1 "3/3 identiques aux originaux ET à anchors.sha256 (sha256sum indépendant) ; $(grep '^PASS REGEN' "$SCRATCH/regen1.log" | sed 's/^PASS REGEN //' | cut -c1-110)"
  else verdict FAIL REGEN-1 "rc=$r, $n/3 identiques (voir $SCRATCH/regen1.log)"; fi
  rm -rf "$d/rejeu"
}

gate_regen2() {
  local ch w; read -r ch w <<< "$(choix_regen2)"
  echo "# REGEN-2 : chaîne choisie par la règle gelée = ens/$ch (wall_s $w)"
  ram_ok REGEN-2 || return
  local d="$SCRATCH/regen2"; rm -rf "$d"
  gpu "$KVU_BIN" regen "$YM4/$ch/manifest.json" --preuve-gbit "$PREUVES" --sortie "$d" > "$SCRATCH/regen2.log" 2>&1; local r=$?
  local ind
  ind=$(python3 -I - "$YM4/$ch/manifest.json" "$d/rejeu" <<'PY'
import hashlib, json, os, sys
m = json.load(open(sys.argv[1])); rej = sys.argv[2]
att = dict(m["sorties_sha256"]); att.update(m["csv_sha256"])
ok = 0
for nom, sha in att.items():
    p = os.path.join(rej, nom)
    if os.path.isfile(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        ok += (h.hexdigest() == sha)
print(ok, len(att))
PY
)
  if [ $r -eq 0 ] && [ "${ind% *}" = "${ind#* }" ] && [ "${ind% *}" -gt 0 ]; then verdict PASS REGEN-2 "ens/$ch : ${ind% *}/${ind#* } fichiers (sorties_sha256 + csv_sha256) identiques à l'octet (recalcul Python indépendant) ; $(grep '^PASS REGEN' "$SCRATCH/regen2.log" | cut -c1-90)"
  else verdict FAIL REGEN-2 "ens/$ch rc=$r, indépendant ${ind} (voir $SCRATCH/regen2.log) : $YM4 à classer C, rien ne se supprime"; fi
  rm -rf "$d/rejeu"
}

gate_regen3() {
  local ch w; read -r ch w <<< "$(choix_regen2)"
  ram_ok REGEN-3 || return
  local d="$SCRATCH/regen3"; rm -rf "$d"
  gpu "$KVU_BIN" regen "$YM4/$ch/manifest.json" --preuve-gbit "$PREUVES" --sortie "$d" --mutant-graine > "$SCRATCH/regen3.log" 2>&1; local r=$?
  local nd; nd=$(grep -c '^DIFF ' "$SCRATCH/regen3.log")
  if [ $r -eq 8 ] && [ "$nd" -ge 1 ]; then verdict PASS REGEN-3 "graine + 1 : rc 8 (code VALIDATION), sha256 différents détectés ($(grep -m1 '^FAIL REGEN' "$SCRATCH/regen3.log" | cut -c1-80))"
  else verdict FAIL REGEN-3 "rc=$r (attendu 8), $nd ligne(s) DIFF (voir $SCRATCH/regen3.log)"; fi
  rm -rf "$d/rejeu"
}

gate_integ() {
  local cible=${1:-$YM4}
  local t0=$SECONDS
  python3 -I "$DEPOT/scripts/regen_audit.py" --integrite "$cible" --sortie-json "$SCRATCH/integ_$(basename "$(dirname "$cible")").json" > "$SCRATCH/integ_$(basename "$(dirname "$cible")").log" 2>&1; local r=$?
  grep -E '^(PASS|FAIL) REGEN-INTEG' "$SCRATCH/integ_$(basename "$(dirname "$cible")").log" | while IFS= read -r l; do echo "$l"; done
  [ $r -eq 0 ] || NFAIL=$((NFAIL + 1))
  echo "# REGEN-INTEG $cible : durée $((SECONDS - t0)) s (nice 19, ionice -c3)"
}

gate_audit() {
  [ -f "$PREUVES" ] && PR="--preuve-gbit $PREUVES" || PR=""
  python3 -I "$DEPOT/scripts/regen_audit.py" --kvu "$KVU_BIN" --ensembles "$ENSEMBLES" $PR --sortie-json "$SCRATCH/audit_ensembles.json" --sortie-md "$SCRATCH/audit_ensembles.md" > "$SCRATCH/audit.log" 2>&1; local r=$?
  cat "$SCRATCH/audit.log"
  [ $r -eq 0 ] || NFAIL=$((NFAIL + 1))
}

instantane() { # nom|chemin1;chemin2|note -> nom|fichiers|octets (lecture seule : find, jamais d'écriture)
  local n p c f o tf=0 to=0
  while IFS='|' read -r n p _; do
    [ -z "$n" ] && continue; case "$n" in \#*) continue ;; esac
    tf=0; to=0
    IFS=';' read -ra cs <<< "$p"
    for c in "${cs[@]}"; do f=$(find "$c" -type f | wc -l); o=$(du -sb "$c" | cut -f1); tf=$((tf + f)); to=$((to + o)); done
    echo "$n|fichiers=$tf|du_sb=$to"
  done < "$ENSEMBLES"
}
gate_avant() { instantane > "$SCRATCH/audit_intact_avant.txt"; echo "# AUDIT-INTACT : relevé AVANT ($(wc -l < "$SCRATCH/audit_intact_avant.txt") ensembles) dans $SCRATCH/audit_intact_avant.txt"; }
gate_intact() {
  [ -f "$SCRATCH/audit_intact_avant.txt" ] || { verdict FAIL AUDIT-INTACT "relevé avant absent (lancer « avant » d'abord)"; return; }
  instantane > "$SCRATCH/audit_intact_apres.txt"
  if diff -q "$SCRATCH/audit_intact_avant.txt" "$SCRATCH/audit_intact_apres.txt" > /dev/null; then verdict PASS AUDIT-INTACT "nombre de fichiers et du -sb identiques avant/après pour les $(wc -l < "$SCRATCH/audit_intact_avant.txt") ensembles"
  else verdict FAIL AUDIT-INTACT "différence : $(diff "$SCRATCH/audit_intact_avant.txt" "$SCRATCH/audit_intact_apres.txt" | head -5 | tr '\n' ' ')"; fi
}

gate_prod() {
  local a="7098f98fa07fd5ae15dce6b6058f9c8e33132d6daf1b614b0c35cd25b9c5dff5  /chemin/vers/projet/kevinotron_dev/bin/acemagic/kevinotron"
  local b="d62ff7d2a9cfc861e8c02faf4b1e284aaed7da8a2b9cd42f7aec14d335af93cf  /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/bin/su2_hip"
  local c="cb29307d9ef522e52f4ea227574587e9a50c3328b115982992caf7d13905febe  /chemin/vers/projet/chantiers/brancheC_saturation_v3_20261004/kev/target/release/su2_gen"
  if printf '%s\n%s\n%s\n' "$a" "$b" "$c" | sha256sum -c --quiet - > "$SCRATCH/t0prod.log" 2>&1; then verdict PASS T0-PROD "les 3 binaires de production ont le sha256 de référence (7098f98f…, d62ff7d2…, cb29307d…)"
  else verdict FAIL T0-PROD "$(tr '\n' ' ' < "$SCRATCH/t0prod.log")"; fi
}

[ $# -ge 1 ] || { echo "usage : KVU_BIN=<kvu> $0 <schema|gbit|regen1|regen2|regen3|integ [DIR]|audit|avant|intact|prod>…"; exit 2; }
while [ $# -gt 0 ]; do
  g=$1; shift
  case "$g" in
    schema) gate_schema ;; gbit) gate_gbit ;; regen1) gate_regen1 ;; regen2) gate_regen2 ;; regen3) gate_regen3 ;;
    integ) if [ $# -ge 1 ] && [ -d "$1" ]; then gate_integ "$1"; shift; else gate_integ; fi ;;
    audit) gate_audit ;; avant) gate_avant ;; intact) gate_intact ;; prod) gate_prod ;;
    *) echo "gate inconnue : $g"; exit 2 ;;
  esac
done
echo "# fin : $NFAIL FAIL"
exit "$NFAIL"
