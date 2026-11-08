#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Gate VERROU-PARTAGE (lot WP21, docs/MACHINES.md §8) : deux `kvu su2 boite` sur le GPU, lancés EN MÊME TEMPS sous le VRAI gpu_partage
# (/usr/local/bin/gpu_partage), sur un verrou de TEST (jamais celui de la machine : GPU_PARTAGE_VERROU, et un kvu.toml de test dont tous
# les profils pointent vers ce verrou, contrôlé par `kvu doctor --json` pour les deux binaires avant tout lancement).
#
# À lancer DANS une fenêtre GPU EXCLUSIVE de la machine (le GPU est réellement utilisé) ; le script le VÉRIFIE : un ancêtre vivant doit
# tenir <verrou_machine> en EXCLUSIF (/proc/locks), sinon rien n'est lancé :
#   flock -w 7200 <verrou_machine> bash gate_verrou_partage.sh <kvu> <kvu_avant_WP21> <hip|cuda> <dossier_neuf> <verrou_machine>
# Paramètres figés (aucune surcharge par l'environnement) : ℓ = 8, β = 2,5, 1 chaîne cold, --therm 0 --nmeas 8000 --profil-tous 0 ;
# attente du verrou de kvu = 20 s (kvu.toml de test) ; gpu_partage --places 2 -w 120.
# Lignes émises (seuils du bloc v5.WP21 de GATES.md) :
#   VERROU-PARTAGE              : rc 0 ×2, termine ×2, `verrou` = partage ×2, recouvrement min(fin) − max(début) ≥ 1,0 s (horodatages
#                                 debut_unix / fin_unix des manifest.json : pris APRÈS le verrou, posés AVANT sa libération) ;
#   VERROU-PARTAGE-EXCLUSIF     : un `flock -w 120 <verrou de test>` demandé quand les deux manifestes existent, avant min(fin), ne l'obtient
#                                 qu'à un instant ≥ max(fin) ;
#   VERROU-PARTAGE-MUT-SANS-ENV : même lancement, GPU_PARTAGE retiré devant kvu : le MÉCANISME attendu doit être observé pour les deux
#                                 lancements (rc 6, message « <verrou de test> non obtenu en 20 s », aucun manifeste termine, durée du
#                                 lancement ≥ 19 s) ; la gate de recouvrement est alors rouge (pas d'exécution concurrente réussie) ;
#   VERROU-PARTAGE-MUT-ANCIEN   : même chose avec le binaire d'avant WP21, PLUS un contrôle positif : ce binaire, seul et hors gpu_partage,
#                                 exécute le même cas (rc 0, termine, `pris_ici`) ;
#   VERROU-HORS-PARTAGE         : sans gpu_partage, deux kvu lancés ensemble sont SÉRIALISÉS : rc 0 ×2, `pris_ici` ×2, recouvrement ≤ 0.
# Code de retour : nombre de lignes FAIL (0 = tout vert) ; 2 si la gate ne peut pas s'exécuter (ECHEC_OUTIL, rien de lancé).
set -u
[ $# -eq 5 ] || { sed -n 2,26p "$0"; exit 2; }
KVU=$(readlink -f "$1"); ANCIEN=$(readlink -f "$2"); B=$3; D=$4; VM=$5
GP=/usr/local/bin/gpu_partage
NMEAS=8000
NFAIL=0
v() { echo "$*"; case "$1" in FAIL) NFAIL=$((NFAIL + 1));; esac; }
outil() { echo "FAIL VERROU-PARTAGE ECHEC_OUTIL : $* : rien n'est lancé"; exit 2; }
case "$B" in hip|cuda) ;; *) outil "backend « $B » (hip|cuda attendu)";; esac
[ -e "$D" ] && outil "$D existe déjà (dossier neuf requis)"
for x in "$KVU" "$ANCIEN" "$GP"; do [ -x "$x" ] || outil "$x absent ou non exécutable"; done
[ -f "$VM" ] || outil "verrou de la machine $VM absent"
# Fenêtre exclusive : un ANCÊTRE vivant de ce script tient $VM en écriture (FLOCK WRITE, ligne non « -> ») dans /proc/locks.
python3 -I - "$VM" "$$" <<'PY' || outil "aucun ancêtre ne tient $VM en EXCLUSIF (lancer dans une fenêtre exclusive : flock -w 7200 $VM …)"
import os, sys
st = os.stat(sys.argv[1]); maj, mi, ino = os.major(st.st_dev), os.minor(st.st_dev), st.st_ino
anc, p = [], int(sys.argv[2])
while p > 1 and len(anc) < 4096:
    try:
        s = open(f"/proc/{p}/stat").read()
    except OSError:
        break
    p = int(s[s.rindex(")") + 1:].split()[1]); anc.append(p)
ok = False
for l in open("/proc/locks"):
    c = l.split()
    if "->" in c or "FLOCK" not in c:
        continue
    i = c.index("FLOCK")
    try:
        pid = int(c[i + 3]); a, b, n = c[i + 4].split(":")
    except (IndexError, ValueError):
        continue
    if c[i + 2] == "WRITE" and (int(a, 16), int(b, 16), int(n)) == (maj, mi, ino) and pid in anc:
        ok = True
sys.exit(0 if ok else 1)
PY
mkdir -p "$D" || exit 2
D=$(readlink -f "$D")
TL=$D/verrou_test.lock; : > "$TL"
[ "$(stat -c '%d %i' "$TL")" != "$(stat -c '%d %i' "$VM")" ] || outil "le verrou de test est le verrou de la machine"
SRC=$(dirname "$(readlink -f "$0")")/../../kvu.toml
CFG=$D/kvu_test.toml
sed -E -e "s|^(verrou[[:space:]]*=[[:space:]]*).*|\1\"$TL\"|" -e "s|^(attente_verrou_s[[:space:]]*=[[:space:]]*).*|\120|" "$SRC" > "$CFG"
for x in "$KVU" "$ANCIEN"; do
  r=$(env -u GPU_PARTAGE -u GPU_PARTAGE_VERROU_TENU KEVINOTRON_CONFIG="$CFG" timeout 120 "$x" doctor --json 2>/dev/null \
      | python3 -I -c 'import json,sys; p=json.load(sys.stdin)["politique"]; print(p["verrou_gpu"], p["attente_verrou_s"])' 2>/dev/null)
  [ "$r" = "$TL 20" ] || outil "$x résout « $r » au lieu de « $TL 20 » (verrou de test, attente 20 s)"
done
echo "INFO gate VERROU-PARTAGE : backend $B, kvu $(sha256sum "$KVU" | cut -c1-16), avant WP21 $(sha256sum "$ANCIEN" | cut -c1-16), fenêtre exclusive tenue par un ancêtre sur $VM, verrou de test $TL, nmeas $NMEAS"

# lancer <mode> <binaire> <sortie> <graine> : partage | sans_env (gpu_partage, GPU_PARTAGE retiré devant kvu) | hors (sans gpu_partage).
# Écrit <sortie>.log, .rc et .duree (secondes, flottant).
lancer() {
  local mode=$1 bin=$2 out=$3 g=$4 t0
  local args=(su2 boite --backend "$B" --ell 8 --beta 2.5 --chaines "cold:$g" --therm 0 --nmeas "$NMEAS" --profil-tous 0 --outdir "$out")
  t0=$(date +%s.%N)
  case $mode in
    partage)  env -u KEVINOTRON_VERROU GPU_PARTAGE_VERROU="$TL" KEVINOTRON_CONFIG="$CFG" "$GP" --places 2 -w 120 -- "$bin" "${args[@]}" ;;
    sans_env) env -u KEVINOTRON_VERROU GPU_PARTAGE_VERROU="$TL" KEVINOTRON_CONFIG="$CFG" "$GP" --places 2 -w 120 -- env -u GPU_PARTAGE "$bin" "${args[@]}" ;;
    hors)     env -u KEVINOTRON_VERROU -u GPU_PARTAGE -u GPU_PARTAGE_VERROU_TENU KEVINOTRON_CONFIG="$CFG" "$bin" "${args[@]}" ;;
  esac > "$out.log" 2>&1
  echo $? > "$out.rc"
  python3 -I -c 'import sys,time; print(f"{time.time()-float(sys.argv[1]):.3f}")' "$t0" > "$out.duree"
}
# R2 : statut natif termine avec rc=0 et marqueur COMPLETE authentifié.
# lire <sortie> : « rc statut verrou debut fin » (champs du manifeste ; « - » si absent)
lire() {
  python3 -I "$(dirname "$(readlink -f "$0")")/lire_verrou_v2.py" "$1"
}
recouvre() { python3 -I -c 'import sys; a=[float(x) for x in sys.argv[1:]]; print(f"{min(a[1], a[3]) - max(a[0], a[2]):.3f}")' "$@"; }
ge() { python3 -I -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$1" "$2"; }
# mecanisme <sortie> : 0 ssi le lancement a échoué PAR LE VERROU : rc 6, message « <verrou de test> non obtenu en 20 s », pas de termine, durée ≥ 19 s
mecanisme() {
  local o=$1 r s
  read -r r s _ < <(lire "$o")
  [ "$r" = 6 ] && [ "$s" != termine ] && [ ! -e "$o/COMPLETE" ] && grep -qF "$TL non obtenu en 20 s" "$o.log" && ge "$(cat "$o.duree")" 19
}
# paire <mode> <binaire> <préfixe> : deux lancements simultanés, attente des deux
paire() {
  lancer "$1" "$2" "$D/$3_a" 26102800001 & local pa=$!
  lancer "$1" "$2" "$D/$3_b" 26102800002 & local pb=$!
  wait $pa $pb
}

# --- 1. binaire WP21 sous gpu_partage : recouvrement + demande exclusive pendant les deux
lancer partage "$KVU" "$D/p_a" 26102800001 & PA=$!
lancer partage "$KVU" "$D/p_b" 26102800002 & PB=$!
t=0; while [ $t -lt 600 ] && { [ ! -f "$D/p_a/manifest.json" ] || [ ! -f "$D/p_b/manifest.json" ]; }; do sleep 0.1; t=$((t + 1)); done
T_DEM=$(date +%s.%N)
flock -w 120 "$TL" sh -c 'date +%s.%N > "$1"' _ "$D/excl_t" & PE=$!
wait $PA $PB; wait $PE; RCE=$?
read -r ra sa va da fa < <(lire "$D/p_a"); read -r rb sb vb db fb < <(lire "$D/p_b")
if [ "$ra$rb" = 00 ] && [ "$sa$sb" = terminetermine ] && [ "$va" = partage ] && [ "$vb" = partage ]; then
  R=$(recouvre "$da" "$fa" "$db" "$fb")
  if ge "$R" 1.0; then v PASS VERROU-PARTAGE "2 kvu su2 boite ($B) sous gpu_partage : rc 0/0, termine ×2, verrou partage/partage, recouvrement $R s (seuil 1,0 s)"
  else v FAIL VERROU-PARTAGE "recouvrement $R s < 1,0 s (verrou $va/$vb)"; fi
  TX=$(cat "$D/excl_t" 2>/dev/null)
  if [ "$RCE" = 0 ] && [ -n "$TX" ] && python3 -I -c '
import sys
dem, x, fa, fb = map(float, sys.argv[1:])
sys.exit(0 if dem < min(fa, fb) and x >= max(fa, fb) else 1)' "$T_DEM" "$TX" "$fa" "$fb"; then
    v PASS VERROU-PARTAGE-EXCLUSIF "flock exclusif demandé $(python3 -I -c 'import sys; print(f"{min(float(sys.argv[2]),float(sys.argv[3]))-float(sys.argv[1]):.3f}")' "$T_DEM" "$fa" "$fb") s avant la première fin, obtenu $(python3 -I -c 'import sys; print(f"{float(sys.argv[1])-max(float(sys.argv[2]),float(sys.argv[3])):.3f}")' "$TX" "$fa" "$fb") s après la dernière"
  else v FAIL VERROU-PARTAGE-EXCLUSIF "demande $T_DEM, obtention « $TX » (rc $RCE), fins $fa $fb"; fi
else
  v FAIL VERROU-PARTAGE "rc $ra/$rb, statut $sa/$sb, verrou $va/$vb (journaux $D/p_a.log, $D/p_b.log)"
  v FAIL VERROU-PARTAGE-EXCLUSIF "non mesurable (lancements en échec)"
fi

# --- 2. mutants : le MÉCANISME de l'incident doit être observé (attente du verrou jusqu'au délai), et la gate de recouvrement est rouge
paire sans_env "$KVU" m1
if mecanisme "$D/m1_a" && mecanisme "$D/m1_b"; then
  v PASS VERROU-PARTAGE-MUT-SANS-ENV "GPU_PARTAGE retiré : les deux kvu attendent le verrou partagé de leur propre enveloppe et expirent (rc 6, « non obtenu en 20 s », durées $(cat "$D/m1_a.duree")/$(cat "$D/m1_b.duree") s, aucun termine) : gate de recouvrement ROUGE"
else v FAIL VERROU-PARTAGE-MUT-SANS-ENV "ECHEC_OUTIL : mécanisme non observé (rc $(cat "$D/m1_a.rc")/$(cat "$D/m1_b.rc"), durées $(cat "$D/m1_a.duree")/$(cat "$D/m1_b.duree") s ; journaux $D/m1_a.log, $D/m1_b.log)"; fi
lancer hors "$ANCIEN" "$D/m2_temoin" 26102800003
read -r rt st vt _ _ < <(lire "$D/m2_temoin")
paire partage "$ANCIEN" m2
if [ "$rt" = 0 ] && [ "$st" = termine ] && [ "$vt" = pris_ici ] && mecanisme "$D/m2_a" && mecanisme "$D/m2_b"; then
  v PASS VERROU-PARTAGE-MUT-ANCIEN "binaire d'avant WP21 : témoin seul hors gpu_partage rc 0, termine, pris_ici ; sous gpu_partage les deux expirent (rc 6, « non obtenu en 20 s », durées $(cat "$D/m2_a.duree")/$(cat "$D/m2_b.duree") s) : gate de recouvrement ROUGE"
else v FAIL VERROU-PARTAGE-MUT-ANCIEN "ECHEC_OUTIL : témoin rc $rt/$st/$vt ; mutant rc $(cat "$D/m2_a.rc")/$(cat "$D/m2_b.rc"), durées $(cat "$D/m2_a.duree")/$(cat "$D/m2_b.duree") s"; fi

# --- 3. hors gpu_partage : comportement d'avant (exclusif, sérialisé)
paire hors "$KVU" h
read -r ra sa va da fa < <(lire "$D/h_a"); read -r rb sb vb db fb < <(lire "$D/h_b")
if [ "$ra$rb" = 00 ] && [ "$sa$sb" = terminetermine ] && [ "$va" = pris_ici ] && [ "$vb" = pris_ici ]; then
  R=$(recouvre "$da" "$fa" "$db" "$fb")
  if ge 0 "$R"; then v PASS VERROU-HORS-PARTAGE "sans gpu_partage : pris_ici/pris_ici, rc 0/0, sérialisés (recouvrement $R s ≤ 0)"
  else v FAIL VERROU-HORS-PARTAGE "recouvrement $R s > 0 hors gpu_partage"; fi
else v FAIL VERROU-HORS-PARTAGE "rc $ra/$rb, statut $sa/$sb, verrou $va/$vb"; fi
echo "# fin gate VERROU-PARTAGE : $NFAIL FAIL"
exit $NFAIL
