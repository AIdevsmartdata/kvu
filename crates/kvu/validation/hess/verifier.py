"""Vérificateur de campagne WP13 : lit verdicts.txt, verdicts_manifeste.txt et les sorties de chaque script ; exige chaque ID attendu en PASS, aucune ligne FAIL/SKIP
(y compris les suppléments INFO-*, dont un FAIL est BLOQUANT), les rc de tous les scripts nuls, chaque fichier d'entrée haché (64 hexadécimaux, aucune erreur de hachage).
Usage : verifier.py SORTIE_DIR  -> écrit `VERDICT WP13 VERT|ROUGE ...` (rc 0 si VERT)."""
import sys, os, re
ATTENDUS = ['HESS-REG', 'HESS-REG-MUTANT-FACTEUR2', 'HESS-EXACT', 'G1', 'M-b', 'G2', 'G3-tcov', 'G3-tfix', 'M-c', 'HESS-LMIN', 'HESS-ESPACES', 'HESS-JVP', 'HESS-KERDU', 'HESS-FILS']
SCRIPTS = ['ref_dim_kerdu', 'hess_reg', 'hess_exact', 'hess_g', 'hess_lmin', 'hess_espaces', 'hess_jvp', 'hess_kerdu', 'hess_landau', 'hess_fils']
FICHIERS_HACHES = ['kvu', 'kvu.commit', 'outils.py', 'hess_reg.py', 'hess_exact.py', 'hess_g.py', 'hess_lmin.py', 'hess_espaces.py', 'hess_jvp.py', 'hess_kerdu.py', 'hess_landau.py',
                   'hess_fils.py', 'kerdu_ref.py', 'ref_dim_kerdu.py', 'verifier.py', 'lancer.sh', 'PREVOL.md', 'graines.md', 'lat.py', 'balaban_np.py', 'hess.py', '_noyaux.py', '__init__.py',
                   'cfg_002000.npy', 'ref_dim.json']
d = sys.argv[1]
probs = []
lignes = [l.rstrip('\n') for l in open(os.path.join(d, 'verdicts.txt')) if l.strip() and not l.startswith('#')]
pass_ids = set(); fail = []
for l in lignes:
    m = re.match(r'^(PASS|FAIL|SKIP)\s+(\S+)', l)
    if not m:
        probs.append('ligne illisible : ' + l[:80]); continue
    if m.group(1) == 'PASS':
        pass_ids.add(m.group(2))
    else:
        fail.append(l[:140])
manq = [i for i in ATTENDUS if i not in pass_ids]
man = open(os.path.join(d, 'verdicts_manifeste.txt')).read()
if 'No such file' in man or 'sha256sum:' in man:
    probs.append('erreur de hachage dans le manifeste')
hach = {}
for l in man.splitlines():
    m = re.match(r'^([0-9a-f]{64})  (.+)$', l)
    if m:
        hach.setdefault(os.path.basename(m.group(2)), []).append(m.group(1))
for f in FICHIERS_HACHES:
    if f not in hach:
        probs.append('empreinte absente : ' + f)
for s in SCRIPTS:
    try:
        sortie = open(os.path.join(d, s + '.out')).read().strip().splitlines()
        if not sortie or sortie[-1] != 'rc=0':
            probs.append('rc non nul ou absent : ' + s)
    except OSError:
        probs.append('sortie absente : ' + s)
ok = not manq and not fail and not probs
msg = 'VERDICT WP13 %s : %d PASS / %d FAIL-SKIP ; IDs attendus manquants : %s ; problemes : %s' % ('VERT' if ok else 'ROUGE', len(pass_ids), len(fail), manq, probs)
print(msg)
for l in fail:
    print('  ' + l)
open(os.path.join(d, 'VERDICT.txt'), 'w').write(msg + '\n')
sys.exit(0 if ok else 1)
