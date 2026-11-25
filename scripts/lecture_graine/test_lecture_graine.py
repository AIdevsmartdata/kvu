#!/usr/bin/env python3
"""Tests de lecture_graine.py (20/11/2026). Lancer : python3 test_lecture_graine.py   (donnees reelles de kvu_prod lues, jamais modifiees)."""
import os, sys, subprocess, importlib.util, random
ICI = os.path.dirname(os.path.abspath(__file__))
K = os.environ.get('KVU_DONNEES_REELLES', '/chemin/vers/donnees_reelles')
L = K + '/loue'
spec = importlib.util.spec_from_file_location('lg', ICI + '/lecture_graine.py'); lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)
ech = []
def ok(nom, cond, detail=''):
    ech.append(cond); print(('OK    ' if cond else 'ECHEC ') + nom + (' : ' + detail if detail else ''))
def lance(*a):
    return subprocess.run([sys.executable, '-I', ICI + '/lecture_graine.py', *a], capture_output=True, text=True).stdout

# 1. tri numerique des tranches (defaut du 19/11 : tri dans l'ordre du systeme de fichiers)
noms = [f'/x/T{i:03d}/A/manifest_thermo.json' for i in range(1, 1201)] + ['/x/T1201/A/manifest_thermo.json']
for graine in range(5):
    m = noms[:]; random.Random(graine).shuffle(m)
    if lg.tri_tranches(m) != noms: ok('tri numerique T001..T1201 (melange %d)' % graine, False); break
else:
    ok('tri numerique T001..T1201 sur 5 melanges', True)

# 2. rejouer la lecture combinee de 07h a l'octet, sur les ORIGINAUX (gamer 83 + Radeon 75)
sortie = lance('gamer=' + K + '/gamer_n8', 'radeon=' + K + '/radeon/n8', '--max-tranches', 'gamer=83', 'radeon=75', '--sans-controle')
ref = open(K + '/radeon/LECTURE_07H_N8_COMBINEE.txt').read()
ok('LECTURE_07H_N8_COMBINEE.txt a l\'octet (originaux)', sortie == ref, f'{len(sortie)} octets')
ok('beta* combine 07h = 2.3971563492134815', '2.3971563492134815' in sortie)

# 3. beta* combine de la nuit 2 : 7 sources, Radeon limitee a 82 tranches (ce qui existait alors)
src = ['gamer=' + K + '/gamer_n8', 'radeon=' + K + '/radeon/n8'] + [f'{c}={L}/n8_5090{c}/n8' for c in 'abcde']
s7 = lance(*src, '--max-tranches', 'radeon=82', '--sans-controle')
ok('7 sources : beta* combine 2,3899 (RESULTATS_N8_NUIT2)', 'beta* = 2.3898926294419858' in s7.split('COMBINEE')[1])
ok('7 sources : 228/82/45/10/8/328/327 tranches', all(t in s7 for t in ['gamer : 228 tranches', 'radeon : 82 tranches', 'd : 328 tranches', 'e : 327 tranches']))

# 4. controle de positivite present et verdict
sc = lance('gamer=' + K + '/gamer_n8', 'radeon=' + K + '/radeon/n8', '--max-tranches', 'gamer=83', 'radeon=75')
ok('bloc de controle (positivite) imprime', 'CONTROLE' in sc.upper() and 'POSITIVIT' in sc.upper())

# 5. forme globale : bloc secondaire present, sortie historique inchangee devant
sf = lance(*src, '--max-tranches', 'radeon=82', '--sans-controle', '--forme-globale')
ok('forme globale : sortie historique inchangee en tete', sf.startswith(s7))
ok('forme globale : bloc etiquette SECONDAIRE', '-- FORME GLOBALE (SECONDAIRE' in sf and 'critere fixe d\'avance' in sf)

# 6. forme globale : retrouve un beta* connu sur des donnees synthetiques exactes (sans bruit)
nodes = [2.275 + 0.0125 * i for i in range(10)]
c0, a0, b0 = 3e-7, 5e-9, 45.0
import math
d = [c0 + a0 * math.exp(b0 * (n - nodes[0])) for n in nodes]; e = [1e-8] * 10
fit = lg.ajuste_forme_globale(nodes, d, e)
ok('forme globale : (c, a, b) exacts sur donnees sans bruit', abs(fit['b'] - b0) < 1e-3 and abs(fit['c'] / c0 - 1) < 1e-4 and abs(fit['a'] / a0 - 1) < 1e-3, f"b = {fit['b']:.4f}")
ok('forme globale : integrale analytique = trapeze fin',
   abs(lg.a3_forme(nodes, fit, lg.PREF, nodes[-1]) - sum(lg.PREF * (c0 + a0 * math.exp(b0 * (nodes[0] + (k + .5) * 1e-5 - nodes[0]))) * 1e-5 for k in range(int((nodes[-1] - nodes[0]) / 1e-5)))) < 1e-3)
print(f'\n{sum(ech)}/{len(ech)} OK'); sys.exit(0 if all(ech) else 1)
