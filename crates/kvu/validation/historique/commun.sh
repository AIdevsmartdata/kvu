#!/usr/bin/env bash
# Fonctions communes aux gates historiques G-H01..G-H13 (T9). À SOURCER : . commun.sh
#
# Variables (fournies par gates_unifie.sh ; valeurs par défaut pour un lancement à la main) :
#   KVU_MACHINE  acemagic|gamer|inconnue    KVU_BIN  binaire `kvu` à valider
#   KVU_RACINE   racine du dépôt            KVU_SCRATCH  répertoire de travail (petits fichiers)
#   KVU_GIT_HEAD commit attendu             KVU_SANS_GPU=1 : aucune gate GPU (mode rapide)
#   KVU_STORE_VALID, KVU_STORE_GAMER_VALID  racines des ensembles de validation (locale, côté gamer)
#   KVU_PY       python avec numpy+scipy (acemagic : /home/utilisateur/.venvs/certif/bin/python)
#
# Chaque script écrit sur stdout UNE ligne `PASS|FAIL|SKIP <gate> <détail>` (rien du tout si la gate ne
# concerne pas la machine) ; le bruit va sur stderr. Un script rend toujours 0 : son verdict est la ligne. (exception : G-H01.sh rend 1 quand il émet FAIL)
set -u
HIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACINE="${KVU_RACINE:-$(cd "$HIST/../../../.." && pwd)}"
REF="$RACINE/crates/kvu/reference"
MACHINE="${KVU_MACHINE:-inconnue}"
BIN="${KVU_BIN:-kvu}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ -z "${KVU_SCRATCH:-}" ]; then   # lancement à la main : travail sur le SSD (la garde SSD refuse /tmp, en RAM)
  _base="$([ "${KVU_MACHINE:-}" = gamer ] && echo "$HOME/kvu_unif/scratch" || echo /chemin/vers/projet/chantiers/kevinotron_unification/scratch)"
  [ -d "$_base" ] || _base="${TMPDIR:-/tmp}"   # jamais de dossier créé hors du dépôt : base absente => TMPDIR
  KVU_SCRATCH="$(mktemp -d "$_base/kvu_hist.XXXXXX")"
fi
SCRATCH="$KVU_SCRATCH"
PY="${KVU_PY:-/home/utilisateur/.venvs/certif/bin/python}"
[ -x "$PY" ] || PY=python3

case "$MACHINE" in
  gamer)
    VERROU=/tmp/gamer_gpu.lock; STORE_DEF="$HOME/kevinotron_store"; SSD_MIN_GO=50; PRE_CPU="nice -n 5 taskset -c 12-19";;
  *)
    VERROU=/tmp/kevinotron_gpu.lock; STORE_DEF=/chemin/vers/projet/kevinotron_store; SSD_MIN_GO=110; PRE_CPU="nice -n 10";;
esac
[ -n "${KVU_VERROU:-}" ] && VERROU="$KVU_VERROU"
[ -n "${KVU_SSD_MIN_GO:-}" ] && SSD_MIN_GO="$KVU_SSD_MIN_GO"   # surcharge du seuil de la garde SSD (tests : petit disque)
STORE_V="${KVU_STORE_VALID:-$STORE_DEF/validation_unifie}"
STORE_GAMER_V="${KVU_STORE_GAMER_VALID:-/home/utilisateur/kevinotron_store/validation_unifie}"
GPU_PORT="${KVU_GPU_PORT:-/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port}"

pass() { echo "PASS $1 ${2:-}"; }
fail() { echo "FAIL $1 ${2:-}"; }
log() { echo "[$GATE] $*" >&2; }

# Sortie immédiate si la gate ne concerne pas cette machine (rien n'est émis : le catalogue ne l'y attend pas).
reserve_a() { [ "$MACHINE" = "$1" ] || exit 0; }
# Gate GPU : en mode rapide, rien n'est lancé (rien n'est émis).
exiger_gpu() { [ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0; return 0; }

libre_go() { # Go libres du système de fichiers qui contient $1 (remonte au premier parent existant)
  local d="${1:-/}"; while [ ! -e "$d" ] && [ "$d" != / ]; do d="$(dirname "$d")"; done
  df --output=avail -BG "$d" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Refuse (FAIL puis exit) si le disque passerait sous le seuil de la machine (110 Go acemagic, 50 Go gamer).
exiger_ssd() { # gate [Go à écrire]
  local libre; libre="$(libre_go "$STORE_V")"; [ -z "$libre" ] && libre=0
  mkdir -p "$STORE_V" 2>/dev/null
  if [ "$((libre - ${2:-1}))" -lt "$SSD_MIN_GO" ]; then
    fail "$1" "garde SSD : ${libre} Go libres, seuil ${SSD_MIN_GO} Go (écriture de ${2:-1} Go refusée)"; exit 0
  fi
}

# Exécute une commande sous le verrou GPU de la machine (une fenêtre ; flock faux = transparent si ancêtre).
sous_verrou() { flock -w "${KVU_VERROU_ATTENTE_S:-7200}" "$VERROU" "$@"; }

binaire_ok() { # gate
  if ! command -v "$BIN" >/dev/null 2>&1 && [ ! -x "$BIN" ]; then fail "$1" "binaire introuvable : $BIN"; exit 0; fi
}

# Lots des « 18 chaînes » (gates G-H06, G-H07) : L ∈ {6, 8} × β ∈ {2,3 ; 2,8 ; 3,5} × (2 chaudes + 1 froide), graines de
# gpu_port (commandes_cuda.txt : 27092706001… pour L = 6, 27092708001… pour L = 8), therm 2000, nmeas 20000, un dump
# tous les 1000. Un lot est UN lancement de `su2 gen` (blockIdx.y = chaîne) : L et départ sont communs à ses
# chaînes (`--start` est unique par lancement), donc 4 lots : (L = 6, 8) × (chaud : 6 chaînes, froid : 3 chaînes).
# $1 = L, $2 = hot|cold, $3 = répertoire de sortie (NEUF)
lancer_lot() {
  local L="$1" depart="$2" out="$3" betas="" graines="" b k base ks
  case "$depart" in hot) ks="0 1";; cold) ks="2";; *) echo "départ inconnu : $depart" >&2; return 2;; esac
  for b in 2.3 2.8 3.5; do
    case "$b" in 2.3) base=1;; 2.8) base=11;; 3.5) base=21;; esac
    for k in $ks; do
      betas="$betas${betas:+,}$b"
      graines="$graines${graines:+,}$((27092700000 + L * 1000 + base + k))"
    done
  done
  $PRE_CPU "$BIN" su2 gen --L "$L" --beta "$betas" --graine "$graines" --start "$depart" \
    --therm 2000 --nmeas 20000 --skip 1 --nor 4 --dump-every 1000 --precision f64 --options compat-prod \
    --outdir "$out"
}

# Les 4 lots d'une machine sous $1 (gh06 ou gh07), chacun dans sa fenêtre de verrou. Rend 0 si les 4 lancements
# ont réussi ; écrit les anomalies dans $PB (variable globale).
lancer_les_lots() { # sous-répertoire du store
  local L dep
  PB=""
  for L in 6 8; do
    for dep in hot cold; do
      log "lot L=$L $dep"
      sous_verrou bash -c "$(declare -f lancer_lot); PRE_CPU='$PRE_CPU'; BIN='$BIN'; lancer_lot $L $dep '$STORE_V/$1/L${L}_$dep'" \
        >"$SCRATCH/$1_L${L}_$dep.log" 2>&1 || PB="$PB L=$L/$dep rc≠0 ($(tail -1 "$SCRATCH/$1_L${L}_$dep.log"));"
    done
  done
  [ -z "$PB" ]
}

# Nombre de chaînes COMPLETE sous un répertoire : une chaîne = un répertoire avec measurements.csv ; elle est
# COMPLETE si COMPLETE est dans son répertoire (run d'une chaîne) ou dans celui du lot (parent, cNNN/).
nb_complete() {
  local n=0 m d
  while IFS= read -r m; do
    d="$(dirname "$m")"
    if [ -f "$d/COMPLETE" ] || [ -f "$d/../COMPLETE" ]; then n=$((n + 1)); fi
  done < <(find "$1" -name measurements.csv 2>/dev/null)
  echo "$n"
}

# Rapatrie, si besoin, l'ensemble gh06 produit par le gamer (par le câble, sha256 vérifiés par fetch).
# Rend 0 si 18 chaînes COMPLETE sont présentes sous $STORE_V/gh06_gamer.
rapatrier_gh06() {
  local dest="$STORE_V/gh06_gamer"
  if [ "$(nb_complete "$dest")" -ge 18 ]; then return 0; fi
  mkdir -p "$STORE_V"
  "$BIN" fetch "gamer:$STORE_GAMER_V/gh06" "$dest" >&2 || return 1
  [ "$(nb_complete "$dest")" -ge 18 ]
}
