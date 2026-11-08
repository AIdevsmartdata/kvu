#!/bin/bash
# lancer_verdicts.sh : exécute TOUS les verdicts de WP12a sur acemagic, dans l'ordre du PREVOL (FIB-STAT-CALIB d'abord : son échec arrête le lot),
# avec codes retour de chaque commande, puis le contrôle d'inventaire. Environnement nettoyé (aucune variable FIBC_* sauf FIBC_FILS).
set -u
U=/chemin/vers/projet/chantiers/kevinotron_unification
A=$U/arbres/wp12a
OUT=$U/scratch/wp12a/verdicts
mkdir -p $OUT $U/scratch/wp12a/tmp
rm -f $OUT/*.log $OUT/rcs.txt $OUT/../verdicts.txt
E="env -i PATH=$PATH HOME=$HOME TMPDIR=$U/scratch/wp12a/tmp CARGO_TARGET_DIR=$U/cibles/wp12a FIBC_FILS=16"
cd $A
$E nice -n 10 cargo test --release --lib --no-run > $OUT/build.log 2>&1 || { echo "build rc != 0"; exit 3; }
BIN=$(grep -o '/[^ )]*deps/kvu-[0-9a-f]*' $OUT/build.log | tail -1)
echo "binaire de test : $BIN sha256 $(sha256sum $BIN | cut -c1-64)" | tee $OUT/binaire.txt
run() { nom=$1; shift; $E calc_borne --mem 20G -- "$@" > $OUT/$nom.log 2>&1; echo "$nom $?" >> $OUT/rcs.txt; }
OPT="--include-ignored --nocapture --test-threads=1"
run stat_calib $BIN fibre_contrainte::gates_penalite::fib_stat_calib $OPT
[ "$(grep '^stat_calib ' $OUT/rcs.txt | cut -d' ' -f2)" = 0 ] || { echo "FIB-STAT-CALIB non vert : lot arrete"; exit 4; }
run commun    $BIN fibre_contrainte::gates_commun $OPT
run jacobien  $BIN fibre_contrainte::gates_jacobien $OPT
run penalite  $BIN fibre_contrainte::gates_penalite $OPT --skip fib_stat_calib
run rattle    $BIN fibre_contrainte::gates_rattle $OPT
run su2       $BIN fibre_contrainte::gates_su2 $OPT --skip explo
run cli       cargo test --release --test fibre_contrainte_cli
cat $OUT/stat_calib.log $OUT/commun.log $OUT/jacobien.log $OUT/penalite.log $OUT/rattle.log $OUT/su2.log | sed -E 's/^test [^ ]+ \.\.\. //' | grep -E '^(PASS|FAIL|SKIP) ' > $U/scratch/wp12a/verdicts.txt
python3 -I -B $A/crates/kvu/validation/fibre_contrainte/inventaire.py $U/scratch/wp12a/verdicts.txt $OUT/rcs.txt $OUT/cli.log
