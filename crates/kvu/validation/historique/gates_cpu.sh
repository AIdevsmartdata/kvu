#!/usr/bin/env bash
# TESTS-CPU : tous les tests NON ignorés du crate kvu (tests unitaires et d'intégration de T1..T9, CPU seulement ;
# les gates GPU sont des tests #[ignore = "GPU"] et ne tournent pas ici). Aucun GPU, aucun verrou : c'est la gate du
# mode rapide. Émet UNE ligne PASS|FAIL TESTS-CPU avec le nombre de tests.
# Cible de build : CARGO_TARGET_DIR (fournie par gates_unifie.sh, hors dépôt) ; garde SSD avant la compilation.
GATE=TESTS-CPU
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_ssd "$GATE" 8
[ "$MACHINE" = gamer ] && export PATH="$HOME/.cargo/bin:$PATH"
LOG="$SCRATCH/tests_cpu.log"
# Les variables KVU_* de la suite (KVU_SANS_GPU, KVU_MACHINES, KVU_DISTANT, KVU_GROUPE…) ne doivent pas fuir dans les tests,
# qui lancent eux-mêmes l'orchestrateur et les gates avec leur propre environnement.
for v in $(env | grep -o '^KVU_[A-Z0-9_]*'); do unset "$v"; done
unset KEVINOTRON_BACKEND KEVINOTRON_MACHINE
( cd "$RACINE" && $PRE_CPU cargo test --release --no-fail-fast -j "$([ "$MACHINE" = gamer ] && echo 6 || echo 8)" -p kvu >"$LOG" 2>&1 )
rc=$?
"$PY" - "$LOG" "$rc" <<'PYEOF'
import re, sys
log, rc = open(sys.argv[1], errors='replace').read(), int(sys.argv[2])
ok = sum(int(x) for x in re.findall(r'test result: ok\. (\d+) passed', log))
ign = sum(int(x) for x in re.findall(r'(\d+) ignored', log))
ko = sum(int(x) for x in re.findall(r'; (\d+) failed', log))
echecs = re.findall(r'^test (\S+) \.\.\. FAILED', log, re.M)
if rc != 0 or ko:
    print(f"FAIL TESTS-CPU rc={rc} : {ko} test(s) en échec " + ' '.join(echecs[:6]) + ('' if echecs else ' ' + ' '.join(log.strip().split('\n')[-2:])))
else:
    print(f"PASS TESTS-CPU {ok} tests passés, {ign} ignorés (gates GPU)")
# RNG-NP (§6.3) : la gate est l'ensemble des tests de tests/cond_numpy.rs (GenerateurNumpy contre numpy 2.5.3 à l'octet, hb_numpy
# contre cb._hb_numpy), qui ne s'impriment pas eux-mêmes PASS/FAIL : on lit leur section dans la sortie de cargo.
sec = re.search(r'Running tests/cond_numpy\.rs.*?(test result: [^\n]*)', log, re.S)
if not sec:
    print("FAIL RNG-NP tests/cond_numpy.rs absent de la sortie de cargo test")
else:
    m = re.search(r'test result: (\w+)\. (\d+) passed; (\d+) failed', sec.group(1))
    if m and m.group(1) == 'ok' and m.group(3) == '0':
        print(f"PASS RNG-NP {m.group(2)} tests de tests/cond_numpy.rs (random et standard_normal à l'octet contre numpy, hb_numpy contre cb)")
    else:
        print(f"FAIL RNG-NP {sec.group(1)}")
PYEOF
exit 0
