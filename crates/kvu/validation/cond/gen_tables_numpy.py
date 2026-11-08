#!/usr/bin/env python3
"""Régénère src/cond/numpy_tables.rs depuis ziggurat_constants.h de numpy 2.5.3 (copie littérale des constantes).
Usage : gen_tables_numpy.py <ziggurat_constants.h> <sortie.rs>
(la source se récupère sur https://raw.githubusercontent.com/numpy/numpy/v2.5.3/numpy/random/src/distributions/ziggurat_constants.h,
sha256 9f8b114014b8475777e88c6af80e126654fe0753693d6eabba365d3c4051236c). L'en-tête de licence est celui du fichier commité :
on ne remplace que les trois tableaux, par un contrôle de cohérence (les valeurs doivent être identiques)."""
import re, sys
src = open(sys.argv[1]).read()
rs = open(sys.argv[2]).read()
def table(name):
    return re.search(r'static const \w+ %s\[\] = \{(.*?)\};' % name, src, re.S).group(1)
ki = [int(v, 16) for v in re.findall(r'0x([0-9A-F]+)ULL', table('ki_double'))]
wi = [float(v) for v in re.findall(r'[0-9.]+e[+-]\d+', table('wi_double'))]
fi = [float(v) for v in re.findall(r'[0-9.]+e[+-]\d+', table('fi_double'))]
def rtable(name):
    return re.search(r'pub static %s: \[\w+; 256\] = \[(.*?)\];' % name, rs, re.S).group(1)
rki = [int(v, 16) for v in re.findall(r'0x([0-9a-f]+)', rtable('KI_DOUBLE'))]
rwi = [float(v) for v in re.findall(r'[0-9.]+e[+-]\d+', rtable('WI_DOUBLE'))]
rfi = [float(v) for v in re.findall(r'[0-9.]+e[+-]\d+', rtable('FI_DOUBLE'))]
assert ki == rki and wi == rwi and fi == rfi, 'tables différentes de la source numpy'
print('tables identiques à la source numpy (256 x 3)')
