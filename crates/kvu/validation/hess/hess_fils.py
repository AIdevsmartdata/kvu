"""HESS-FILS : `hess hv`, `hess jvp`, `hess lmin` et `hess espace` à 1, 4 et 8 fils : sorties NPY et JSON `cmp` identiques (octets).
Fonds : Haar L = 8 (graine 26102613081) pour hv (gauche et additive, base s2 et ym4) ; Haar L = 4 (graine 26102613082) pour lmin (libre, kerB, tcov, projecteurs svd et cg) ;
Haar L = 6 (graine 26102613071) pour jvp et kerdu (espace + lmin à tol 1e-8)."""
import os, sys, json, filecmp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
sc = O.SCRATCH + '/hess_fils'; os.makedirs(sc, exist_ok=True)
tout = True

def cmp_fils(ident, builder, sorties):
    global tout
    res = {}
    for fils in (1, 4, 8):
        os.environ['KVU_FILS'] = str(fils); O.FILS = str(fils)
        args = builder(fils)
        O.kvu(args)
        res[fils] = [open(s % fils, 'rb').read() for s in sorties]
        for s_ in sorties:
            if s_.endswith('.npy'):
                O.fin(np.load(s_ % fils))
            elif s_.endswith('.json') and ('lmin' in s_ or '/lk_' in s_):
                O.rapport_lmin(s_ % fils)
    ok = all(res[1][i] == res[k][i] for k in (4, 8) for i in range(len(sorties)))
    tout &= ok
    O.verdict(f, ident, ok, 'octets identiques a 1, 4 et 8 fils : %s (%d fichier(s), %s octets)' % (ok, len(sorties), [len(b) for b in res[1]]))

# hv
C8 = O.haar_c16(8, 26102613081); cfg8 = sc + '/c8.npy'; O.ecrire_npy(cfg8, C8)
rng = np.random.default_rng(26102613083); V8 = rng.standard_normal((6, 12 * 8 ** 4)); O.ecrire_npy(sc + '/v8.npy', V8)
for base, coord in (('ym4', 'gauche'), ('s2', 'gauche'), ('s2', 'additive')):
    cmp_fils('HESS-FILS-hv-%s-%s' % (base, coord),
             lambda fils, base=base, coord=coord: ['hess', 'hv', '--config', cfg8, '--vecteurs', sc + '/v8.npy', '--sortie', sc + '/hv_%s_%s_%%d.npy' % (base, coord) % fils, '--base', base, '--coord', coord, '--beta', 2.3, '--fils', fils],
             [sc + '/hv_%s_%s_%%d.npy' % (base, coord)])
# jvp
C6 = O.haar_c16(6, 26102613071); cfg6 = sc + '/c6.npy'; O.ecrire_npy(cfg6, C6)
rng = np.random.default_rng(26102613084); V6 = rng.standard_normal((4, 12 * 6 ** 4)); O.ecrire_npy(sc + '/v6.npy', V6)
cmp_fils('HESS-FILS-jvp', lambda fils: ['hess', 'jvp', '--config', cfg6, '--vecteurs', sc + '/v6.npy', '--sortie', sc + '/jvp_%d.npy' % fils, '--ubar-sortie', sc + '/ub_%d.npy' % fils, '--fils', fils],
         [sc + '/jvp_%d.npy', sc + '/ub_%d.npy'])
# lmin
C4 = O.haar_c16(4, 26102613082); cfg4 = sc + '/c4.npy'; O.ecrire_npy(cfg4, C4)
for espace, coord, ref, proj in (('libre', 'additive', 'q0', 'cg'), ('kerB', 'additive', 'q0', 'cg'), ('tcov', 'gauche', 'qcov', 'cg'), ('tcov', 'gauche', 'qcov', 'svd')):
    cmp_fils('HESS-FILS-lmin-%s-%s' % (espace, proj),
             lambda fils, espace=espace, coord=coord, ref=ref, proj=proj: ['hess', 'lmin', '--config', cfg4, '--espace', espace, '--coord', coord, '--reference', ref, '--beta', 2.3, '--graine', 7, '--projecteur', proj,
                                                  '--sortie', sc + '/lmin_%s_%s_%%d.json' % (espace, proj) % fils, '--vecteur-sortie', sc + '/lmin_%s_%s_v%%d.npy' % (espace, proj) % fils, '--fils', fils],
             [sc + '/lmin_%s_%s_%%d.json' % (espace, proj), sc + '/lmin_%s_%s_v%%d.npy' % (espace, proj)])
# espace kerdu
cmp_fils('HESS-FILS-espace-kerdu', lambda fils: ['hess', 'espace', '--config', cfg6, '--espace', 'kerdu', '--projeter', sc + '/v6.npy', '--sortie', sc + '/pk_%d.npy' % fils, '--rapport', sc + '/ek_%d.json' % fils, '--fils', fils],
         [sc + '/pk_%d.npy', sc + '/ek_%d.json'])
cmp_fils('HESS-FILS-lmin-kerdu', lambda fils: ['hess', 'lmin', '--config', cfg6, '--espace', 'kerdu', '--coord', 'gauche', '--reference', 'q0', '--beta', 2.3, '--graine', 7, '--tol', '1e-8',
                                                '--sortie', sc + '/lk_%d.json' % fils, '--vecteur-sortie', sc + '/lk_v%d.npy' % fils, '--fils', fils], [sc + '/lk_%d.json', sc + '/lk_v%d.npy'])
O.verdict(f, 'HESS-FILS', tout, 'hv (3 variantes), jvp, lmin (libre, kerB, tcov cg, tcov svd, kerdu), espace kerdu')
sys.exit(0 if tout else 1)
