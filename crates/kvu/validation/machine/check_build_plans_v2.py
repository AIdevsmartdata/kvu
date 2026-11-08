#!/usr/bin/env python3
"""Golden de plans de build, sans compilation ; refus des défauts de remappage."""
from pathlib import Path
import difflib
import os
import subprocess
import sys

VAL = Path(__file__).resolve().parent
DEPOT = VAL.parents[3]

def normaliser(text):
    # Le dépôt physique doit être remplacé avant son parent.
    return text.replace(str(DEPOT), '<DEPOT>').replace(str(DEPOT.parent), '<U>').replace(str(Path.home()), '~')

def conforme(got, expected, rc):
    return rc == 0 and got == expected

def main():
    env = dict(os.environ)
    for k in ('CARGO_TARGET_DIR','CARGO_HOME','KEVINOTRON_BIN_DIR','KEVINOTRON_MACHINE','KEVINOTRON_SSD_MIN_GO','KVU_LOT','KVU_SORTIES'):
        env.pop(k, None)
    failures, mutations = [], 0
    for machine, extra in (('acemagic', []), ('gamer', []), ('gamer', ['--lto', '--nettoyer'])):
        suffix = machine+('_kvu_lto' if extra else '')
        golden = VAL/f'golden_build_auto_{suffix}_v2.txt'
        p = subprocess.run(['bash', str(DEPOT/'scripts/build_auto.sh'), '--dry-run', '--machine', machine, *extra], env=env, capture_output=True, text=True)
        lines = p.stdout.splitlines(keepends=True)
        try:
            stop = lines.index('--- fin du plan ---\n')+1
        except ValueError:
            failures.append(suffix+': fin du plan absente')
            continue
        actual = normaliser(''.join(lines[:stop]))
        expected = golden.read_text()
        if not conforme(actual, expected, p.returncode):
            failures.append(suffix+f': rc={p.returncode}\n'+''.join(difflib.unified_diff(expected.splitlines(True),actual.splitlines(True))))
        if '--self-test' in sys.argv:
            changes = [actual.replace('=/kvu,', '=/faux,'), actual.replace('=/cargo,', '=/faux,'),
                       actual.replace('=/cible\n', '=/faux\n'), actual.replace('jobs            : -j', 'jobs            : -x'),
                       actual.replace('seuil SSD       :', 'seuil SSD IGNORE :')]
            for changed in changes:
                if changed == actual or conforme(changed, expected, 0):
                    failures.append(suffix+': mutation acceptée ou inerte')
                else:
                    mutations += 1
            if conforme(expected, expected, 7):
                failures.append(suffix+': rc en échec accepté')
            else:
                mutations += 1
    if failures:
        print('FAIL BUILD-AUTO-GOLDEN : '+'\n'.join(failures))
        return 1
    print('PASS BUILD-AUTO-GOLDEN : 3 plans identiques, remap chemins compris')
    if '--self-test' in sys.argv:
        print(f'PASS BUILD-AUTO-GOLDEN-MUTANTS : {mutations} défauts rejetés')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
