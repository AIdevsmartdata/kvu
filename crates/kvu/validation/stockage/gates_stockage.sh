#!/usr/bin/env bash
# gates_stockage.sh — gates de la tâche T8 (ARCHITECTURE_UNIFIEE.md §6.5) :
#   PROV-1  binaire_sha256, source_hash (recalculé par source_hash.py), noyaux_embarques, sha256_source, options
#   PROV-2  noyaux/ modifié => Unifiee change ; BuildSh et Complete inchangées
#   PROV-3  kill -9 ou rc != 0 => pas de COMPLETE
#   SSD-1   seuil forcé => refus avant tout octet (rc 7) ; projection c16 L=64 x 20 dumps refusée
#   SSD-2   gc simulé ne supprime rien ; --appliquer sous la racine, .garder, liens non suivis ; c16 -> q64 vérifié
#   FETCH-1 petit ensemble du gamer par gamer-eth : sha vérifiés, octet altéré => erreur, garde sur la taille distante
#
# Usage : bash gates_stockage.sh            (acemagic ; FETCH-1 parle au gamer par le câble)
#         NO_GAMER=1 bash gates_stockage.sh (FETCH-1 devient SKIP, qui ne vaut PAS vert)
# Sortie : une ligne « PASS <gate> » ou « FAIL <gate> » par gate, puis RC global (0 si tout est PASS).
# Garde disque : s'arrête sous SEUIL_GO Go libres (110 par défaut) AVANT de compiler.
set -u
ICI=$(cd "$(dirname "$0")" && pwd)
DEPOT=$(cd "$ICI/../../../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/chemin/vers/projet/chantiers/kevinotron_unification/cibles/T8}
SEUIL_GO=${SEUIL_GO:-110}
SCRATCH_JOURNAL=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/stockage
mkdir -p "$SCRATCH_JOURNAL"
JOURNAL=${JOURNAL:-$SCRATCH_JOURNAL/gates_stockage.$$.log}

libre=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "${libre:-0}" -lt "$SEUIL_GO" ]; then
  echo "ARRÊT : ${libre} Go libres sous le seuil de ${SEUIL_GO} Go : on ne compile pas, on le signale."
  exit 7
fi
echo "SSD libre : ${libre} Go (seuil ${SEUIL_GO} Go) ; cible de compilation : ${CARGO_TARGET_DIR}"

cd "$DEPOT" || exit 1
CARGO=${CARGO:-cargo}
rc_global=0

echo "== tests de stockage (npy, garde, store, provenance) =="
nice -n 10 $CARGO test -p kvu --test stockage_npy --test stockage_garde --test stockage_store --test stockage_provenance \
  -- --nocapture --test-threads=4 > "$JOURNAL" 2>&1
rc_tests=$?
echo "cargo test : RC=${rc_tests}  (journal : ${JOURNAL})"

if [ "${NO_GAMER:-0}" = "1" ]; then
  echo "SKIP FETCH-1 (NO_GAMER=1 : ne vaut pas vert)" | tee -a "$JOURNAL"
  rc_fetch=0
else
  echo "== FETCH-1 (gamer-eth) =="
  nice -n 10 $CARGO test -p kvu --test stockage_fetch -- --ignored --nocapture >> "$JOURNAL" 2>&1
  rc_fetch=$?
  echo "cargo test fetch : RC=${rc_fetch}"
fi

# gate : nombre minimal de lignes « PASS <gate> » ; aucun test en échec
verdict() {
  local gate=$1 min=$2 n
  n=$(grep -c "^PASS ${gate}\b" "$JOURNAL")
  if [ "$rc_tests" -eq 0 ] && [ "$n" -ge "$min" ]; then
    echo "PASS ${gate}  (${n} contrôle(s))"
  else
    echo "FAIL ${gate}  (${n}/${min} contrôle(s), cargo test RC=${rc_tests})"
    rc_global=1
  fi
}
echo "== verdicts =="
verdict PROV-1 1
verdict PROV-2 2
verdict PROV-3 3
verdict SSD-1 2
verdict SSD-2 7
if [ "${NO_GAMER:-0}" = "1" ]; then
  echo "SKIP FETCH-1"
  rc_global=1
elif [ "$rc_fetch" -eq 0 ] && grep -q "^PASS FETCH-1" "$JOURNAL"; then
  echo "PASS FETCH-1"
else
  echo "FAIL FETCH-1  (RC=${rc_fetch})"
  rc_global=1
fi

n_skip=$(grep -c "^SKIP" "$JOURNAL")
if [ "$n_skip" -gt 0 ]; then
  echo "ATTENTION : ${n_skip} contrôle(s) SKIP (ne valent pas vert) :"
  grep "^SKIP" "$JOURNAL" | sed 's/^/  /'
fi
echo "RC global = ${rc_global}"
exit "$rc_global"
