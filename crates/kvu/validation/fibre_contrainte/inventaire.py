#!/usr/bin/env python3
"""Contrôle d'inventaire des verdicts de WP12a : chaque ID attendu doit apparaître avec le bon nombre de lignes PASS, aucune ligne FAIL, et les
SKIP sont listés (un SKIP de gate = non vert). Usage : inventaire.py verdicts.txt rcs.txt cli.log
(verdicts : lignes `PASS|FAIL|SKIP <ID> <détail>` ; rcs.txt : lignes `<nom> <rc>` de TOUTES les commandes ; cli.log : sortie de `cargo test --test fibre_contrainte_cli`).
Rend rc 0 seulement si tout est présent et vert, tous les codes retour sont nuls et les 4 tests CLI (FIB-DET, FIB-FILS, FIB-MANIFESTE, usage) passent."""
import re, sys, collections
ATTENDU = {
    'ENTREE-WP11': 1, 'FIB-JAC': 1, 'FIB-JAC-MUTANT': 1, 'FIB-LOGDET': 1, 'FIB-LOGDET-MUTANT': 1, 'CT-9': 1,
    'FIB-CHOL-FACTEUR': 1, 'FIB-CHOL-LOGDET': 1, 'FIB-UBAR-LOCAL': 1,
    'FIB-RATTLE-POT': 1, 'FIB-RATTLE-POT-MUTANT': 2, 'FIB-JOUET-B': 1, 'FIB-JOUET-A': 1, 'FIB-JOUET-MUTANT': 1,
    'FIB-A-LIBRE': 1, 'FIB-A-LIBRE-MUTANT': 1, 'FIB-CONTR': 3, 'FIB-REV': 1, 'FIB-ORDRE': 1, 'FIB-ORDRE-MUTANT': 1,
    'FIB-CREUTZ': 1, 'FIB-CREUTZ-MUTANT': 1, 'FIB-VOLUME': 1, 'FIB-VOLUME-MUTANT': 1,
    'FIB-STAT-CALIB': 48, 'FIB-STAT-CALIB-VALIDITE': 6, 'FIB-STAT-CALIB-EFFECTIF': 6, 'FIB-STAT-CALIB-MUTANT': 1,
}
cnt = collections.Counter(); autres = []
for l in open(sys.argv[1], encoding='utf-8'):
    m = re.match(r'^(PASS|FAIL|SKIP) (\S+)', l)
    if m:
        cnt[(m.group(1), m.group(2))] += 1
        if m.group(1) != 'PASS':
            autres.append(l.rstrip())
rc = 0
for i, n in ATTENDU.items():
    p = cnt[('PASS', i)]
    if p != n:
        print(f'MANQUE {i} : {p} PASS au lieu de {n}'); rc = 1
for (e, i), n in cnt.items():
    if e == 'FAIL':
        print(f'FAIL {i} x{n}'); rc = 1
for l in autres:
    if l.startswith('SKIP'):
        print('SKIP (non vert) :', l[:200])
        if not re.match(r'^SKIP \S+-CMP ', l):
            rc = 1
NOMS_RC = {'stat_calib', 'commun', 'jacobien', 'penalite', 'rattle', 'su2', 'cli'}
vus = {}
for l in open(sys.argv[2], encoding='utf-8'):
    t = l.split()
    if len(t) == 2:
        vus[t[0]] = t[1]
for n in sorted(NOMS_RC):
    if vus.get(n) != '0':
        print(f'RC non nul ou absent : {n} = {vus.get(n)}'); rc = 1
cli = open(sys.argv[3], encoding='utf-8').read()
for t in ('fibre_contrainte_fib_det_et_fils', 'fibre_contrainte_jouets_et_rejeu', 'fibre_contrainte_erreurs_usage_et_sortie_neuve', 'fibre_contrainte_depart_froid_projection_et_l_impair'):
    if f'test {t} ... ok' not in cli:
        print(f'TEST CLI absent ou en échec : {t}'); rc = 1
if 'test result: ok. 4 passed; 0 failed' not in cli:
    print('CLI : « test result: ok. 4 passed; 0 failed » absent'); rc = 1
print('INVENTAIRE', 'VERT' if rc == 0 else 'NON VERT')
sys.exit(rc)
