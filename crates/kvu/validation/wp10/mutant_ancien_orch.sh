#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eu
export PYTHONDONTWRITEBYTECODE=1
python3 -B - "$(dirname -- "${BASH_SOURCE[0]}")" <<'PY'
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ici = Path(sys.argv[1]).resolve()
test = 'test_wp10_orch_fautes.ControleFautes.test_decisif_rc3_faute_restaure'
runner = '''import io, sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
r = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
print('RESULTAT', r.testsRun, len(r.failures), len(r.errors), len(r.skipped))
'''
try:
    arbre = subprocess.run(['git', '-C', str(ici), 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True).stdout.strip()
    ancien = subprocess.run(['git', '-C', arbre, 'show', '6056036:crates/kvu/validation/wp10/wp10_orch.py'], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(prefix='wp10-mutant-') as temporaire:
        dossier = Path(temporaire)
        for source in ici.glob('*.py'):
            shutil.copyfile(source, dossier / source.name)
        (dossier / 'wp10_orch.py').write_bytes(ancien)
        rouge = subprocess.run([sys.executable, '-B', '-c', runner, test], cwd=dossier, capture_output=True, text=True)
        print(rouge.stderr, end='')
        compte = rouge.stdout.strip().splitlines()[-1].split()
        if rouge.returncode or compte != ['RESULTAT', '1', '1', '0', '0']:
            raise ValueError(f'ancien code : pas exactement un échec par assertion ; {rouge.stdout} {rouge.stderr}')
        vert = subprocess.run([sys.executable, '-B', '-m', 'unittest', test], cwd=ici, capture_output=True, text=True)
        print(vert.stderr, end='')
        if vert.returncode:
            raise ValueError('nouveau code : test décisif non vert')
        print('PASS WP10S-MUTANT-ORCH ancien code ROUGE (1 échecs par assertion)')
except Exception as exc:
    print(f'FAIL ECHEC_OUTIL WP10S-MUTANT-ORCH {exc}')
    sys.exit(1)
PY
