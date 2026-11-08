#!/usr/bin/env bash
# Refus de scripts/build_auto.sh (sans rien compiler) : SSD sous le seuil, dépôt sale, dossiers de
# production, cible trop large, options invalides, copie rsync sans KEV_GIT_HEAD.
# Travaille dans un dépôt git JETABLE : aucun effet sur le vrai dépôt.
# Sortie : « PASS BUILD-AUTO-REFUS : n refus attendus » ou « FAIL … », code 0 / 1.
set -u
# Isoler le dépôt jetable des variables du pilote ; les cas fixent leurs propres surcharges.
unset KVU_SORTIES KVU_LOT CARGO_TARGET_DIR KEVINOTRON_BIN_DIR KEVINOTRON_MACHINE KEV_GIT_HEAD
VAL=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SCRIPT=$(cd "$VAL/../../../.." && pwd)/scripts/build_auto.sh
[ -f "$SCRIPT" ] || { echo "FAIL BUILD-AUTO-REFUS : $SCRIPT introuvable"; exit 1; }
T=$(mktemp -d /tmp/build_auto_wp21_r2.XXXXXX)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/depot/scripts" "$T/sans_git/scripts"
cp "$SCRIPT" "$T/depot/scripts/build_auto.sh"; cp "$SCRIPT" "$T/sans_git/scripts/build_auto.sh"
printf '[package]\nname = "faux"\nversion = "0.0.0"\n' > "$T/depot/Cargo.toml"
cp "$T/depot/Cargo.toml" "$T/sans_git/Cargo.toml"
( cd "$T/depot" && git init -q . && git add -A && git -c user.email=t@t -c user.name=t commit -q -m init ) || { echo "FAIL BUILD-AUTO-REFUS : git init"; exit 1; }
BA=$T/depot/scripts/build_auto.sh
CIBLE=$T/cible
N=0; ECHECS=0
attendre() { # $1 libellé, $2 rc attendu, $3 motif attendu dans la sortie, puis la commande
  local lib=$1 rc_att=$2 motif=$3; shift 3
  local sortie rc
  sortie=$("$@" 2>&1); rc=$?
  N=$((N + 1))
  if [ "$rc" -ne "$rc_att" ] || ! printf '%s' "$sortie" | grep -q -- "$motif"; then
    echo "FAIL BUILD-AUTO-REFUS : $lib : rc $rc (attendu $rc_att), motif « $motif » : $sortie"; ECHECS=$((ECHECS + 1))
  fi
}
# SSD=0 : le dossier temporaire est sur un autre disque (tmpfs) ; seul le garde-fou testé doit répondre.
LIB="env -u KEVINOTRON_MACHINE -u KEV_GIT_HEAD KEVINOTRON_SSD_MIN_GO=0"
attendre "SSD sous le seuil" 7 "REFUS" env KEVINOTRON_SSD_MIN_GO=1000000 CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR="$T/bin" bash "$BA" --machine acemagic --crate kvu
attendre "seuil non entier" 2 "entier de Go" env KEVINOTRON_SSD_MIN_GO=beaucoup CARGO_TARGET_DIR="$CIBLE" bash "$BA" --machine acemagic
echo "modif" >> "$T/depot/Cargo.toml"
attendre "dépôt sale" 2 "sale" $LIB CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR="$T/bin" bash "$BA" --machine acemagic --crate kvu
# --sale-ok passe la garde du dépôt (et s'arrête ensuite sur le seuil SSD simulé, preuve qu'on l'a franchie).
attendre "--sale-ok franchit la garde" 7 "Go libres" env KEVINOTRON_SSD_MIN_GO=1000000 CARGO_TARGET_DIR="$CIBLE" bash "$BA" --machine acemagic --crate kvu --sale-ok
( cd "$T/depot" && git checkout -q -- Cargo.toml )
attendre "sortie dans chantiers/" 2 "production" $LIB CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR=/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/bin bash "$BA" --machine acemagic --crate kvu
attendre "sortie su2_cuda" 2 "production" $LIB CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR="$HOME/ym4_calcul/su2_cuda/bin" bash "$BA" --machine gamer --crate kvu
attendre "cible de production" 2 "production" $LIB CARGO_TARGET_DIR=/chemin/vers/projet/kevinotron/target bash "$BA" --machine acemagic --crate kvu
attendre "cible = HOME" 2 "trop large" $LIB CARGO_TARGET_DIR="$HOME" bash "$BA" --machine acemagic --crate kvu
attendre "cible = dépôt" 2 "trop large" $LIB CARGO_TARGET_DIR="$T/depot" bash "$BA" --machine acemagic --crate kvu
attendre "option inconnue" 2 "option inconnue" bash "$BA" --nimporte
attendre "crate invalide" 2 "--crate" bash "$BA" --crate gros
attendre "machine invalide" 2 "inconnue" bash "$BA" --machine pc-de-mamie --dry-run
attendre "copie rsync sans KEV_GIT_HEAD" 2 "KEV_GIT_HEAD" $LIB CARGO_TARGET_DIR="$CIBLE" bash "$T/sans_git/scripts/build_auto.sh" --machine acemagic --crate kvu
attendre "dry-run ne crée rien" 0 "fin du plan" $LIB CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR="$T/bin" bash "$BA" --machine gamer --dry-run
[ ! -e "$CIBLE" ] && [ ! -e "$T/bin" ] || { echo "FAIL BUILD-AUTO-REFUS : --dry-run a créé des dossiers"; ECHECS=$((ECHECS + 1)); }
if [ $ECHECS -eq 0 ]; then echo "PASS BUILD-AUTO-REFUS : $N refus ou plans attendus (SSD, dépôt sale, production, cible trop large, options, rsync)"; exit 0; fi
exit 1
