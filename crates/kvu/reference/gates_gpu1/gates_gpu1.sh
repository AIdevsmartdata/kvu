#!/usr/bin/env bash
# Gates GPU-1, ré-exécutables SUR ACEMAGIC (aucune dépendance au gamer).
# Usage : ./gates_gpu1.sh > logs/gates_run.log 2>&1; echo RC=$? >> logs/gates_run.log
set -u
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/utilisateur/.venvs/certif/bin/python
NF=0; pass() { echo "PASS $1"; }; fail() { echo "FAIL $1"; NF=$((NF + 1)); }
# s0 : copie = original ; port = diff mécanique figé ; binaire rapatrié = binaire compilé
(cd src_origine && sha256sum -c SHA256_COPIES.txt >/dev/null) && (cd src && sha256sum -c SHA256_PORT.txt >/dev/null) \
  && (cd cuda && grep bin/su2_cuda bin/SHA256SUMS | sha256sum -c - >/dev/null) && pass "s0 sha source/port/binaire" || fail s0
# s1 : fichiers rapatriés = fichiers du gamer (liste sha distante figée au transfert)
(cd cuda && sha256sum -c SHA256_DISTANT.txt >/dev/null) && pass "s1 482 fichiers CUDA = sha distants" || fail s1
# P0/P1/P7 : journaux du gamer (RECHERCHE ; relus ici)
grep -q "build OK" cuda/logs_build_rc.txt && grep -q "^RC=0" cuda/logs_build_rc.txt && pass "P0 build nvcc sm_120" || fail P0
grep -q "TEST_HEATBATH PASS" cuda/logs/test_heatbath.log && pass "P1 bain thermique exact 7 alpha" || fail P1
grep -q CMP_CSV_OK cuda/logs/p7.log && grep -q CMP_THERM_OK cuda/logs/p7.log && grep -q CMP_NPY_OK cuda/logs/p7.log && pass "P7 reproductibilite a l'octet" || fail P7
n=$(grep -c "^rc=0 " cuda/logs/rc_cuda.log); [ "$n" = 26 ] && pass "runs CUDA 26/26 rc=0" || fail "runs CUDA $n"
n=$(grep -c "^rc=0 " logs/rc_amd.log); [ "$n" = 20 ] && pass "runs AMD 20/20 rc=0" || fail "runs AMD $n"
# P2-P6 : analyse recalculée ici
nice -n 5 $PY src/analyse_gpu1.py > logs/gates_analyse.log 2>&1 || fail "analyse rc"
$PY - <<'P'
import json, sys
R = json.load(open('analyse/resultats_gpu1.json')); P = R['P3_P4']
g = [('P2 tirages initiaux identiques <1e-13', R['P2']['P2_ok']), ('P3 plaquette |z|<3 et chi2 p>0.01', P['P3_ok']),
     ('P4 C1-C4 CUDA', P['P4_cuda_ok']), ('P4 C1-C4 AMD neuves', all(v['amd'].get('neuve', True) for v in P['P4'].values())),
     ('P5 KS permutation p>0.01 (L6, L8)', R['P5']['P5_ok']), ('P6 integrite dumps CUDA', R['P6']['P6_ok'])]
for n, ok in g: print(('PASS ' if ok else 'FAIL ') + n)
sys.exit(sum(not ok for _, ok in g))
P
NF=$((NF + $?))
echo "FIN gates : $NF echec(s)"; exit $NF
