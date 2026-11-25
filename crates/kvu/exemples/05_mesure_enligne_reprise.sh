#!/usr/bin/env bash
# Mesure EN LIGNE, interruption, reprise : le parcours d'un premier calcul court (L petit) de bout en bout.
#
# MODE AUTOMATIQUE (modes dry et petit, celui qu'exécute le test du dépôt) : DÉTERMINISTE, sans signal, sans délai, sans attente.
#   1. `su2 gen … --mesures corr:…,wilson:… --reprise-every 10 --arret-apres 70` mesure en ligne et s'arrête PROPREMENT au pas 70
#      (après la thermalisation, en phase de mesure) en écrivant un point de reprise ; pas de COMPLETE ;
#   2. `su2 gen --reprendre DIR` continue jusqu'à COMPLETE (la spécification --mesures est rechargée de `mesures_en_ligne.json` :
#      ne pas la redonner ; le backend, lui, se redonne) ;
#   3. un run de référence neuf, d'une traite, est comparé à l'octet (`cmp`) sur les quatre CSV.
#   (`--arret-apres` est une option cachée de test.) Le générateur reçoit ici un arrêt DEMANDÉ ; les arrêts brutaux (SIGKILL) sont
#   éprouvés par les tests du dépôt , pas par cet exemple.
#
# Un run sans `COMPLETE` dans son dossier n'est pas terminé.
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
if [ "$EXEMPLE_MODE" = petit ]; then
  L=6; THERM=20; NMEAS=150; BACKEND="--backend cpu"
else
  L=16; THERM=500; NMEAS=1000; BACKEND=""
fi
ARRET0=$((THERM + 50))
MESURES="corr:brut+flot/0.02/10:1,wilson:R2xT2:1"
base() { # $1 = nombre de mesures
  ARGS=(su2 gen $BACKEND --L "$L" --beta 2.3 --graine 1 --start cold --therm "$THERM" --nmeas "$1" --nor 1 --reprise-every 10 --mesures "$MESURES"); }
comparer() { # $1 = dossier repris, $2 = référence
  for f in measurements.csv wilson.csv correlateur_enligne.csv therm.csv; do
    cmp "$2/$f" "$1/$f" || { echo "ÉCART : $f diffère entre le run d'une traite et le run repris" >&2; return 1; }
  done; }
if [ "$EXEMPLE_MODE" = dry ]; then
  base "$NMEAS"
  lancer "$KVU_BIN" "${ARGS[@]}" --arret-apres "$ARRET0" --outdir "$SORTIE/coupe"
  lancer "$KVU_BIN" su2 gen --reprendre "$SORTIE/coupe" $BACKEND
  lancer "$KVU_BIN" "${ARGS[@]}" --outdir "$SORTIE/ref"
  echo "# Puis : cmp ref/measurements.csv coupe/measurements.csv (idem wilson.csv, correlateur_enligne.csv, therm.csv)"
  exit 0
fi
mkdir -p "$SORTIE" && [ -d "$SORTIE" ] && [ -w "$SORTIE" ] || { echo "ERREUR : dossier de sortie inutilisable : $SORTIE" >&2; exit 1; }

base "$NMEAS"
lancer "$KVU_BIN" "${ARGS[@]}" --arret-apres "$ARRET0" --outdir "$SORTIE/coupe" || exit $?
[ -e "$SORTIE/coupe/COMPLETE" ] && { echo "ERREUR : COMPLETE existe après l'arrêt demandé" >&2; exit 1; }
echo "# run interrompu proprement au pas $ARRET0 : pas de COMPLETE dans $SORTIE/coupe ; reprise"
lancer "$KVU_BIN" su2 gen --reprendre "$SORTIE/coupe" $BACKEND || exit $?
[ -e "$SORTIE/coupe/COMPLETE" ] || { echo "ERREUR : pas de COMPLETE après la reprise" >&2; exit 1; }
lancer "$KVU_BIN" "${ARGS[@]}" --outdir "$SORTIE/ref" || exit $?
comparer "$SORTIE/coupe" "$SORTIE/ref" || exit 1
echo "OK : le run interrompu au pas $ARRET0 puis repris jusqu'à COMPLETE est identique à l'octet au run d'une traite"
