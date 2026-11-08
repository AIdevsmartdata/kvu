#!/usr/bin/env bash
# Fonctions communes des exemples (à sourcer). Trois modes, par EXEMPLE_MODE :
#   dry   (défaut) affiche les commandes et les estimations, ne lance RIEN ;
#   petit lance les commandes avec L réduit et peu de pas, backend cpu, dans un répertoire jetable (tests, 1 min) ;
#   prod  lance les vraies commandes (à ne faire que sous verrou GPU, sur la bonne machine, après `store du --projeter`).
# Variables : KVU_BIN (binaire, défaut kvu), EXEMPLE_SORTIE (répertoire de sortie, défaut : jetable en mode petit).
set -u
EXEMPLE_MODE="${EXEMPLE_MODE:-dry}"
KVU_BIN="${KVU_BIN:-kvu}"
case "$EXEMPLE_MODE" in dry|petit|prod) ;; *) echo "EXEMPLE_MODE : dry|petit|prod" >&2; exit 2;; esac

if [ "$EXEMPLE_MODE" = petit ]; then
  # Jamais dans /tmp (RAM sur acemagic : la garde SSD y refuse toute écriture) : répertoire jetable sur le SSD.
  EXEMPLE_BASE="${EXEMPLE_BASE:-$HOME/kvu_unif/scratch}"
  mkdir -p "$EXEMPLE_BASE"
  SORTIE="${EXEMPLE_SORTIE:-$(mktemp -d "$EXEMPLE_BASE/kvu_exemple.XXXXXX")}"
  [ -z "${EXEMPLE_SORTIE:-}" ] && trap 'rm -rf "$SORTIE"' EXIT
else
  SORTIE="${EXEMPLE_SORTIE:-$HOME/kevinotron_store/exemples}"
fi

# Lance (ou affiche) une commande. En mode prod, sous le verrou GPU de la machine si $VERROU est posé.
lancer() {
  echo "+ $*"
  [ "$EXEMPLE_MODE" = dry ] && return 0
  if [ "$EXEMPLE_MODE" = prod ] && [ -n "${VERROU:-}" ]; then
    flock -w 7200 "$VERROU" "$@"
  else
    "$@"
  fi
}

# Go d'une configuration : c16 = 256·L⁴ octets, q64 = 128·L⁴ ; durée d'un pas (ARCHITECTURE_UNIFIEE.md §2).
taille_go() { awk -v L="$1" -v f="$2" 'BEGIN { o = (f == "c16") ? 256 : 128; printf "%.2f", o * L^4 / 1e9 }'; }
