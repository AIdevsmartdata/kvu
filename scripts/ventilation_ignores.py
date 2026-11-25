#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ventilation des tests ignorés de kvu par motif #[ignore = "..."] (usage : python3 scripts/ventilation_ignores.py . , depuis la racine du dépôt)."""
import re, sys, subprocess, collections
racine = sys.argv[1]
sortie = subprocess.run(['grep', '-rnE', r'^\s*#\[ignore', '--include=*.rs', 'crates'], cwd=racine, capture_output=True, text=True).stdout.splitlines()
cat = collections.Counter(); ex = collections.defaultdict(list)
for l in sortie:
    f, n, t = l.split(':', 2); n = int(n); t = t.strip()
    lignes = open(racine + '/' + f, encoding='utf-8').read().split('\n')
    nom = next((m.group(1) for k in range(n, min(n + 8, len(lignes))) for m in [re.search(r'fn\s+(\w+)', lignes[k])] if m), '?')
    if t.startswith('#[ignore = "GPU'): c = 'carte : motif « GPU »'
    elif t == '#[ignore]' and re.search(r'carte|hip|cuda', nom): c = 'carte : sans motif écrit (comparaison carte/processeur)'
    elif 'EQC-4' in t: c = 'carte : enfant d\'une gate de carte'
    elif '"gamer"' in t: c = 'seconde machine de l\'auteur (ssh)'
    elif 'passerelle CLI' in t or 'pont CLI' in t: c = 'passerelle lancée par des scripts'
    elif 'KVU_EXP_RETRO_DIR' in t: c = 'exige une copie de données (variable d\'environnement)'
    else: c = 'processeur : lourd, manuel ou exploration'
    cat[c] += 1; ex[c].append(nom)
for c, k in sorted(cat.items(), key=lambda x: -x[1]): print(f'{k:4d}  {c}')
print(f'{sum(cat.values()):4d}  total')
