#!/usr/bin/env python3
"""Lit le contrat natif RepertoireRun, y compris COMPLETE et son empreinte."""
from pathlib import Path
import hashlib
import json
import math
import sys

def lire(out):
    out = Path(out)
    try:
        rc = str(int(Path(str(out)+'.rc').read_text().strip()))
    except (OSError, ValueError):
        rc = '-'
    try:
        raw = (out/'manifest.json').read_bytes()
        m = json.loads(raw)
        status = m.get('statut', '-')
        lock = m.get('verrou', '-')
        start, end = m.get('debut_unix', '-'), m.get('fin_unix', '-')
        complete = (out/'COMPLETE').read_text()
        expected = 'rc=0\nmanifest_sha256='+hashlib.sha256(raw).hexdigest()+'\n'
        valid = (rc == '0' and status == 'termine' and type(m.get('rc')) is int and m['rc'] == 0
                 and complete == expected and type(start) in (int, float) and type(end) in (int, float)
                 and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
        if not valid:
            status = 'invalide'
        return rc, status, lock, start, end
    except (OSError, ValueError, TypeError, AttributeError):
        return rc, '-', '-', '-', '-'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: lire_verrou_v2.py <sortie>')
    print(*lire(sys.argv[1]))
