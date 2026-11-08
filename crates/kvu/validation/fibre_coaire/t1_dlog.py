#!/usr/bin/env python3
"""WP11 / FIB-DLOG : dérivées de log et exp sur SU(2) en forme fermée (docs/FIBRE_COAIRE.md §3.1).

(a) SymPy (exact) : Lg Eg = 1, Ld Ed = 1 (identités trigonométriques, sur le plan orthogonal à n) ; développement de Lg à l'ordre
    w^2 = BCH (1 + [w x] + 1/3 [w x]^2) ; comportement en th -> pi.
(b) mpmath (50 chiffres) : Lg, Ld, Eg, Ed contre différences finies centrées des fonctions GÉNÉRIQUES mpmath.logm / expm, sur 24
    points (th de 1e-6 à pi - 1e-3) ; seuil 1e-30 (largement sous la gate 1e-8).
(c) R(A) : homomorphisme, orthogonal, R(exp(i th n.s)) = rotation d'angle -2 th autour de n.
Sortie : lignes PASS/FAIL ; code retour 0 ssi tout PASS.
"""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import mpmath as mp
import su2_mp as S

ok = True
def ligne(etat, ident, detail):
    global ok
    ok &= (etat == 'PASS')
    print(f'{etat} {ident} {detail}', flush=True)

# ---------------------------------------------------------------- (a) SymPy exact
th = sp.symbols('theta', positive=True)
I = sp.I
# sur le plan orthogonal à n, [n x] agit comme la multiplication par i (K^2 = -1) : représentation complexe
Lg_c = th * sp.cos(th) / sp.sin(th) + I * th            # Lg|perp
Ld_c = th * sp.cos(th) / sp.sin(th) - I * th
Eg_c = sp.sin(th) * sp.cos(th) / th - I * sp.sin(th)**2 / th
Ed_c = sp.sin(th) * sp.cos(th) / th + I * sp.sin(th)**2 / th
r1 = sp.simplify(sp.expand_complex(Lg_c * Eg_c) - 1)
r2 = sp.simplify(sp.expand_complex(Ld_c * Ed_c) - 1)
ligne('PASS' if (r1 == 0 and r2 == 0) else 'FAIL', 'DLOG-INVERSE-SYMPY', f'Lg*Eg - 1 = {r1} ; Ld*Ed - 1 = {r2} (plan orthogonal, exact)')
# Lg = g(ad_X), g(z) = z/(e^z - 1), ad_X <-> -2 th K, K <-> i sur le plan orthogonal : g(-2 i th)
z = sp.symbols('z')
g = z / (sp.exp(z) - 1)
r3 = sp.simplify(sp.expand_complex(sp.simplify(g.subs(z, -2 * I * th))) - Lg_c)
ligne('PASS' if r3 == 0 else 'FAIL', 'DLOG-BERNOULLI-SYMPY', f'g(-2i th) - Lg|perp = {r3}  (g(z) = z/(e^z-1), dérivée de log trivialisée à gauche)')
# BCH : Lg = 1 + [w x] + 1/3 [w x]^2 + O(w^4)
w1, w2, w3 = sp.symbols('w1 w2 w3', real=True)
W = sp.Matrix([w1, w2, w3]); t = sp.sqrt(w1**2 + w2**2 + w3**2)
K = sp.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
f_cot = sp.series(sp.Symbol('u') * sp.cos(sp.Symbol('u')) / sp.sin(sp.Symbol('u')), sp.Symbol('u'), 0, 4).removeO()
Lg_s = (W * W.T) / t**2 + f_cot.subs(sp.Symbol('u'), t) * (sp.eye(3) - (W * W.T) / t**2) + K
bch = sp.eye(3) + K + K * K / 3
r4 = sp.simplify(sp.expand(Lg_s - bch))
ligne('PASS' if r4 == sp.zeros(3, 3) else 'FAIL', 'DLOG-BCH-SYMPY', f'Lg - (1 + [w x] + [w x]^2/3) = O(w^4) : reste à l\'ordre 2 = {r4.tolist() if r4 != sp.zeros(3,3) else 0}')
# comportement en th -> pi : th cot th ~ -pi/(pi - th)
lim = sp.limit((th * sp.cos(th) / sp.sin(th)) * (sp.pi - th), th, sp.pi)
ligne('PASS' if lim == -sp.pi else 'FAIL', 'DLOG-PI-SYMPY', f'lim_(th->pi) (pi - th) th cot th = {lim} : d log diverge en W -> -1 (coupure)')

# arbitrage D2 (Kimi : E_g -> n n^T - (1/pi)[n x] quand th -> pi) : limites des deux coefficients de E_g
l1 = sp.limit(sp.sin(th) * sp.cos(th) / th, th, sp.pi); l2 = sp.limit(sp.sin(th)**2 / th**2, th, sp.pi)
ligne('PASS' if (l1 == 0 and l2 == 0) else 'FAIL', 'DLOG-EG-PI-SYMPY', f'lim_(th->pi) sin th cos th/th = {l1}, lim sin^2 th/th^2 = {l2} : E_g(th -> pi) = n n^T (pas n n^T - (1/pi)[n x] : désaccord D2 tranché)')

# ---------------------------------------------------------------- (b) mpmath contre différences finies génériques
mp.mp.dps = 60
rng = random.Random(26102611)
h = mp.mpf('1e-25')
pts = []
for k in range(24):
    thk = [mp.mpf('1e-6'), mp.mpf('1e-3'), mp.mpf('0.3'), mp.mpf('1.2'), mp.mpf('2.5'), mp.pi - mp.mpf('1e-3')][k % 6] if k < 6 else mp.mpf(rng.uniform(0.01, 3.1))
    n = [mp.mpf(rng.gauss(0, 1)) for _ in range(3)]; nn = S.norme(n)
    pts.append([thk * x / nn for x in n])
err = {'Lg': 0, 'Ld': 0, 'Eg': 0, 'Ed': 0}
for w in pts:
    Wm = S.expm_gen(w)
    w0 = S.logm_gen(Wm)
    for a in range(3):
        e = [mp.mpf(0)] * 3; e[a] = h; em = [mp.mpf(0)] * 3; em[a] = -h
        # Lg : log(exp(i e.s) W)
        dl = [(p - m) / (2 * h) for p, m in zip(S.logm_gen(S.expm_gen(e) * Wm), S.logm_gen(S.expm_gen(em) * Wm))]
        dr = [(p - m) / (2 * h) for p, m in zip(S.logm_gen(Wm * S.expm_gen(e)), S.logm_gen(Wm * S.expm_gen(em)))]
        wp = [x + y for x, y in zip(w, e)]; wm = [x + y for x, y in zip(w, em)]
        Wi = S.dag(Wm)
        deg = [(p - m) / (2 * h) for p, m in zip(S.logm_gen(S.expm_gen(wp) * Wi), S.logm_gen(S.expm_gen(wm) * Wi))]
        ded = [(p - m) / (2 * h) for p, m in zip(S.logm_gen(Wi * S.expm_gen(wp)), S.logm_gen(Wi * S.expm_gen(wm)))]
        A, B, C, D = S.Lg(w), S.Ld(w), S.Eg(w), S.Ed(w)
        for b in range(3):
            err['Lg'] = max(err['Lg'], abs(A[b, a] - dl[b])); err['Ld'] = max(err['Ld'], abs(B[b, a] - dr[b]))
            err['Eg'] = max(err['Eg'], abs(C[b, a] - deg[b])); err['Ed'] = max(err['Ed'], abs(D[b, a] - ded[b]))
for k, v in err.items():
    ligne('PASS' if v < mp.mpf('1e-25') else 'FAIL', f'DLOG-DF-{k}', f'max |forme fermée - DF centrée (log par diagonalisation mp.eig / expm générique, 60 chiffres, h = 1e-25)| = {mp.nstr(v, 3)} sur {len(pts)} points, th in [1e-6, pi - 1e-3] (seuil 1e-25)')

# ---------------------------------------------------------------- (c) R(A)
e3 = 0
for k in range(10):
    A = S.su2_aleatoire(rng); B = S.su2_aleatoire(rng)
    RA, RB, RAB = S.Rad(A), S.Rad(B), S.Rad(A * B)
    e3 = max(e3, S.max_abs(RA * RB - RAB), S.max_abs(RA.T * RA - mp.eye(3)), abs(mp.det(RA) - 1))
    # action : A (i e.s) A^dag = i (R e).s
    e = [mp.mpf(rng.gauss(0, 1)) for _ in range(3)]
    Re = RA * mp.matrix(e)
    e3 = max(e3, S.max_abs(A * S.alg(e) * S.dag(A) - S.alg([Re[0], Re[1], Re[2]])))
# rotation d'angle -2 th autour de n
thr = mp.mpf('0.7'); n = [mp.mpf(0), mp.mpf(0), mp.mpf(1)]
Rz = S.Rad(S.expm_gen([thr * x for x in n]))
rot = mp.matrix([[mp.cos(-2 * thr), -mp.sin(-2 * thr), 0], [mp.sin(-2 * thr), mp.cos(-2 * thr), 0], [0, 0, 1]])
e3 = max(e3, S.max_abs(Rz - rot))
ligne('PASS' if e3 < mp.mpf('1e-40') else 'FAIL', 'DLOG-ADJOINT', f'R(AB) = R(A)R(B), R orthogonale det 1, A(i e.s)A^dag = i(Re).s, R(exp(i th e3.s)) = rotation -2th : max écart {mp.nstr(e3, 3)}')

# ---------------------------------------------------------------- (d) dérivées de Lg et Eg (force de la route B : grad log det JJ^T)
# Lg = f I + h w w^T + [w x], f = th cot th, h = (1 - f)/th^2 ; Eg = s I + k w w^T - m [w x], s = sin th cos th/th, k = (1 - s)/th^2, m = sin^2 th/th^2
# d/dw_k : (F'/th) w_k I + (H'/th) w_k w w^T + H (e_k w^T + w e_k^T) +- (M'/th) w_k [w x] +- M [e_k x]
def coefs(t):
    f = t * mp.cos(t) / mp.sin(t); h = (1 - f) / t**2
    fp = mp.cos(t) / mp.sin(t) - t / mp.sin(t)**2; hp = -fp / t**2 - 2 * (1 - f) / t**3
    s_ = mp.sin(2 * t) / (2 * t); k = (1 - s_) / t**2
    sp_ = mp.cos(2 * t) / t - mp.sin(2 * t) / (2 * t**2); kp = -sp_ / t**2 - 2 * (1 - s_) / t**3
    m = mp.sin(t)**2 / t**2; mp_ = mp.sin(2 * t) / t**2 - 2 * mp.sin(t)**2 / t**3
    return f, h, fp, hp, s_, k, sp_, kp, m, mp_
def dL(w, kk, quoi):
    t = S.norme(w); f, h, fp, hp, s_, k, sp_, kp, m, mp_ = coefs(t)
    W = mp.matrix(w); e = mp.matrix([1 if j == kk else 0 for j in range(3)])
    if quoi == 'Lg':
        return (fp / t) * w[kk] * mp.eye(3) + (hp / t) * w[kk] * (W * W.T) + h * (e * W.T + W * e.T) + S.croix([e[0], e[1], e[2]])
    return (sp_ / t) * w[kk] * mp.eye(3) + (kp / t) * w[kk] * (W * W.T) + k * (e * W.T + W * e.T) - (mp_ / t) * w[kk] * S.croix(w) - m * S.croix([e[0], e[1], e[2]])
eD = 0
for w in pts[6:16]:
    for kk in range(3):
        e = [mp.mpf(0)] * 3; e[kk] = h
        for quoi, F in (('Lg', S.Lg), ('Eg', S.Eg)):
            dfd = (F([a + b for a, b in zip(w, e)]) - F([a - b for a, b in zip(w, e)])) / (2 * h)
            eD = max(eD, S.max_abs(dfd - dL(w, kk, quoi)))
ligne('PASS' if eD < mp.mpf('1e-25') else 'FAIL', 'DLOG-DERIVEES', f'dLg/dw_k, dEg/dw_k (formes fermées, docs §3.2) = DF centrées : max écart {mp.nstr(eD, 3)} sur 10 points x 3 directions')
tt = sp.symbols('t', positive=True)
f_ = tt * sp.cos(tt) / sp.sin(tt); h_ = (1 - f_) / tt**2; s_ = sp.sin(2 * tt) / (2 * tt); k_ = (1 - s_) / tt**2; m_ = sp.sin(tt)**2 / tt**2
lims = [sp.limit(sp.diff(g_, tt) / tt, tt, 0) for g_ in (f_, h_, s_, k_, m_)]
attendu = [sp.Rational(-2, 3), sp.Rational(2, 45), sp.Rational(-4, 3), sp.Rational(-4, 15), sp.Rational(-2, 3)]
ligne('PASS' if lims == attendu else 'FAIL', 'DLOG-DERIVEES-ZERO', f'limites en th -> 0 de F\'/th pour F = f, h, s, k, m : {lims} (attendu {attendu}) : dérivées régulières en w = 0')
print('BILAN FIB-DLOG', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
