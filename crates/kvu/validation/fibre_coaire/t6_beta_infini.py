#!/usr/bin/env python3
"""WP11 / FIB-BETA-INF : limite beta -> infini, SU(2), réseau 6^4 (docs/FIBRE_COAIRE.md §6).

(a) J(1) = Q (x) 1_3 : jacobien en forme fermée à la configuration froide, 64 liens grossiers ;
(b) Q (balaban_np, énumération indépendante) = op_Q de S1 (geom.py, LECTURE SEULE) assemblé en espace direct ;
(c) « fibre linéarisée = KKT de S1 » : la table des variances de plaquettes et M_c calculées en espace direct (t5, base de la tranche,
    extension harmonique H = (1 - Gam M) Q^+) = celles de picov.Classe (S1, Bloch, KKT, Gam) moyennées sur les 16 moments de L_c = 2 ;
(d) det(J J^T) invariant de jauge (jauge de Haar quelconque en chaque site) sur une configuration d'amplitude 0,3 (J complet 192 x 15552) ;
(e) log det(J J^T) n'a PAS de terme linéaire en U = 1 (partie impaire cubique) ; coefficient quadratique rapporté ;
(f) valeurs gelées SU(2) beta -> infini (= 3/2 x U(1) gaussienne, coordonnées w) -> valeurs_beta_infini.json.
Usage : t6_beta_infini.py <dossier S1 (lecture seule)> <valeurs_u1.json>
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SORTIES = os.environ.get('FIBRE_SORTIES', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sorties'))
os.makedirs(SORTIES, exist_ok=True)
import numpy as np
import balaban_np as B
import operateurs as O

ok = True
def ligne(etat, ident, detail):
    global ok
    if etat == 'FAIL':
        ok = False
    print(f'{etat} {ident} {detail}', flush=True)

t0 = time.time()
S1 = sys.argv[1]; u1 = json.load(open(sys.argv[2]))
L = 6
ops = O.Ops(L); Q = ops.Q; bloc = ops.bloc
coarse = [(z, mu) for mu in range(4) for z in np.ndindex(2, 2, 2, 2)]
def cidx(z, mu):
    return mu * ops.Vc + ((z[0] * 2 + z[1]) * 2 + z[2]) * 2 + z[3]
TAB = {(z, mu): bloc.tables(z, mu) for z, mu in coarse}

def J_complet(q):
    J = np.zeros((3 * ops.n, 3 * ops.N))
    for z, mu in coarse:
        c = cidx(z, mu)
        liens, Jb, _ = bloc.jac(q, z, mu, TAB[(z, mu)])
        for i, l in enumerate(liens):
            J[3 * c:3 * c + 3, 3 * l:3 * l + 3] = Jb[i]
    return J

# ---------------------------------------------------------------- (a)
froid = np.zeros((ops.N, 4)); froid[:, 0] = 1.0
J1 = J_complet(froid)
ea = float(np.abs(J1 - np.kron(Q, np.eye(3))).max())
ligne('PASS' if ea < 1e-12 else 'FAIL', 'FIB-BINF-J1', f'J(U = 1) = Q (x) 1_3 : max écart {ea:.1e} (64 liens grossiers, 5184 liens fins ; seuil 1e-12 : sommes de 10 473 termes en F64 ; le premier passage à 1e-14 a échoué à 5,5e-14, arrondi d\'accumulation)')

# ---------------------------------------------------------------- (b) op_Q de S1 en espace direct
sys.path.insert(0, os.path.join(S1, 'src')); sys.path.insert(0, os.path.join(S1, 'bundle_q3'))
os.environ['S1_GEOM'] = '3centre'
sys.dont_write_bytecode = True
import geom as GE
E, shp = GE.OPS['Q']
Qs = np.zeros_like(Q)
for z in np.ndindex(2, 2, 2, 2):
    for mu, j, o, coef in E:
        ir, nu = divmod(j, 4)
        r = [(ir // 3**m) % 3 for m in range(4)]
        x = [(3 * (z[m] + o[m]) + r[m]) % L for m in range(4)]
        Qs[cidx(z, mu), nu * ops.V + ((x[0] * L + x[1]) * L + x[2]) * L + x[3]] += coef
eb = float(np.abs(Qs - Q).max())
ligne('PASS' if eb < 1e-14 else 'FAIL', 'FIB-BINF-Q-S1', f'Q (énumération indépendante) = op_Q de geom.py (S1) en espace direct : max écart {eb:.1e}')

# ---------------------------------------------------------------- (c) picov (Bloch, KKT) contre l'espace direct de t5
import picov as PC
qs = GE.grid(2)
acc_pl = np.zeros(6 * 81); acc_mc = np.zeros((4, 4), complex)
Mc_q = []
for q in qs:
    cl = PC.Classe(q, full=False)
    curl = GE.bloch(GE.OPS['curl'], q)
    acc_pl += 0.5 * np.real(np.einsum('ij,jk,ik->i', curl, cl.Gam, np.conj(curl)))
    acc_mc += cl.Mc; Mc_q.append(cl.Mc)
acc_pl /= len(qs); acc_mc /= len(qs)
# plaquettes locales de geom (ip = 6 ir + k, r petit-boutiste base 3) -> classes de t5 (plan, r)
ec = 0.0
for ip in range(6 * 81):
    ir, k = divmod(ip, 6)
    r = [(ir // 3**m) % 3 for m in range(4)]
    mu, nu = O.PAIRES[k]
    cle = f'{mu}{nu}_r{"".join(map(str, r))}'
    ec = max(ec, abs(acc_pl[ip] - u1['beta_th2_par_classe'][cle]))
emc = abs(np.real(acc_mc[0, 0]) - u1['Mc_diag'])
ligne('PASS' if ec < 1e-10 and emc < 1e-10 else 'FAIL', 'FIB-BINF-KKT-S1',
      f'variances de plaquettes par classe : espace direct (t5) = picov.Classe (Bloch, 16 moments) à {ec:.1e} près ; M_c[c,c] = {u1["Mc_diag"]:.12f} contre {np.real(acc_mc[0,0]):.12f} (écart {emc:.1e})')
ligne('INFO', 'FIB-BINF-MC-BLOCH', f'spectre de M_c(q) aux 16 moments : q = 0 : {np.round(np.linalg.eigvalsh(Mc_q[0]), 6).tolist()} ; q = (pi,pi,pi,pi) : {np.round(np.linalg.eigvalsh(Mc_q[-1]), 6).tolist()}')

# ---------------------------------------------------------------- (d) invariance de jauge de det(J J^T)
rng = np.random.default_rng(26102617)
q = B.qexp(0.3 * rng.normal(size=(ops.N, 3)))
J = J_complet(q)
s1, ld1 = np.linalg.slogdet(J @ J.T)
g = rng.normal(size=(ops.V, 4)); g /= np.linalg.norm(g, axis=1)[:, None]
e4 = np.eye(4, dtype=int)
qg = np.empty_like(q)
for nu in range(4):
    s0 = ops.site(ops.X); s1_ = ops.site(ops.X + e4[nu])
    qg[nu * ops.V + s0] = B.qmul(B.qmul(g[s0], q[nu * ops.V + s0]), B.qdag(g[s1_]))
Jg = J_complet(qg)
s2, ld2 = np.linalg.slogdet(Jg @ Jg.T)
# contrôle : Ubar transformé de façon covariante
eu = 0.0
for z, mu in coarse[:8]:
    u, _ = bloc.ubar(q, z, mu, TAB[(z, mu)]); ug, _ = bloc.ubar(qg, z, mu, TAB[(z, mu)])
    c = np.array([3 * v + 1 for v in z]); c2 = c.copy(); c2[mu] += 3
    pred = B.qmul(B.qmul(g[ops.site(c)], u), B.qdag(g[ops.site(c2)]))
    eu = max(eu, float(np.abs(ug - pred).max()))
ld0 = 3 * np.linalg.slogdet(Q @ Q.T)[1]
ligne('PASS' if abs(ld1 - ld2) < 1e-9 and s1 > 0 and s2 > 0 and eu < 1e-13 else 'FAIL', 'FIB-BINF-JAUGE',
      f'log det(J J^T) = {ld1:.12f} ; après jauge de Haar en chaque site : {ld2:.12f} (écart {abs(ld1-ld2):.1e}) ; covariance de Ubar {eu:.1e} ; à U = 1 : 3 log det(Q Q^T) = {ld0:.12f}')

# ---------------------------------------------------------------- (e) pas de terme linéaire
a = rng.normal(size=(ops.N, 3))
f = {}
for s_ in (0.001, 0.002, 0.004, -0.001, -0.002, -0.004, 0.01, -0.01):
    Js = J_complet(B.qexp(s_ * a))
    f[s_] = np.linalg.slogdet(Js @ Js.T)[1]
imp = {s_: (f[s_] - f[-s_]) / 2 for s_ in (0.001, 0.002, 0.004)}
cs = np.linalg.solve(np.array([[s_, s_**3, s_**5] for s_ in (0.001, 0.002, 0.004)]), np.array([imp[s_] for s_ in (0.001, 0.002, 0.004)]))
c1, c3, c5 = cs
pair1 = (f[0.01] + f[-0.01]) / 2 - ld0; pair2 = (f[0.002] + f[-0.002]) / 2 - ld0
ligne('PASS' if abs(c1) < 1e-6 else 'FAIL', 'FIB-BINF-SANS-LINEAIRE',
      f'log det JJ^T(exp(s a)), partie impaire c1 s + c3 s^3 + c5 s^5 ajustée sur s = 0,001/0,002/0,004 : c1 = {c1:.2e} (seuil 1e-6 ; bruit ~1e-9) ; c3 = {c3:.4e} ; c5 = {c5:.4e} ; partie paire / s^2 : {pair2/0.002**2:.4f} (s = 0,002), {pair1/0.01**2:.4f} (s = 0,01)')
ligne('PASS' if abs(c3) > 1.0 else 'FAIL', 'FIB-BINF-PAS-PAIR', f'log det JJ^T n\'est PAS pair en A (c3 = {c3:.3e}, partie impaire {imp[0.004]:.3e} à s = 0,004, ~1e5 x le bruit) : l\'argument « inversion des liens » de Kimi (Ubar(U^-1) = Ubar(U)^-1) est faux en non abélien ; seul le terme LINÉAIRE s\'annule (trace de couleur, docs §6.3)')
# contrôle direct : Ubar(U^-1) contre Ubar(U)^-1 (U^-1 lien par lien)
qi = B.qdag(q)
einv = 0.0
for z, mu in coarse[:4]:
    u, _ = bloc.ubar(q, z, mu, TAB[(z, mu)]); ui, _ = bloc.ubar(qi, z, mu, TAB[(z, mu)])
    einv = max(einv, float(np.abs(ui - B.qdag(u)).max()))
ligne('INFO', 'FIB-BINF-INVERSION', f'max |Ubar(U^-1) - Ubar(U)^-1| = {einv:.3e} (amplitude 0,3) : l\'équivariance par inversion n\'a pas lieu (les holonomies U(G) ne sont pas inversées, leur ordre est conservé)')
c2 = pair1 / 0.01**2
ligne('INFO', 'FIB-BINF-QUADRATIQUE', f'log det JJ^T - 3 log det QQ^T ~ c2 s^2 avec a ~ N(0,1) par composante : c2 = {c2:.4f} pour {3*ops.N} composantes ({c2/(3*ops.N):.3e} par composante fine, {c2/(3*ops.n):.3e} par composante grossière)')

# ---------------------------------------------------------------- (f) valeurs gelées SU(2) beta -> infini
gel = dict(
    convention='SU(2), action de Wilson, beta -> infini, V = 1, coordonnées w (demi-angle) ; chaque couleur = U(1) gaussienne S = beta w^T M w',
    beta_un_moins_P_moy=1.5 * u1['beta_th2_moy'], attendu_20_sur_27=20 / 27,
    beta_un_moins_P_par_classe={k: round(1.5 * v, 14) for k, v in u1['beta_th2_par_classe'].items()},
    beta_un_moins_W_boucle_centres=1.5 * u1['beta_var_boucle_centres_conditionnelle'],
    beta_un_moins_W_boucle_centres_libre=1.5 * u1['beta_var_boucle_centres_libre'],
    covariance_w='cov(w) = Gam/(2 beta) (x) 1_3 dans la tranche ; Gam = picov.Classe.Gam (S1) ; x = sqrt2 w (S1, ym4_hess) : cov(x) = Gam/beta ; a = 2w (kvu WP5) : cov(a) = 2 Gam/beta',
    logdet_JJT_froid=ld0, logdet_c2=c2)
json.dump(gel, open(os.path.join(SORTIES, 'valeurs_beta_infini.json'), 'w'), indent=1, ensure_ascii=False)
ligne('PASS' if abs(gel['beta_un_moins_P_moy'] - 20 / 27) < 1e-11 else 'FAIL', 'FIB-BINF-EQUIPARTITION', f'lim beta <1 - P> (conditionnel, V = 1) = {gel["beta_un_moins_P_moy"]:.12f} = 20/27 = (3/4)(1 - 3^-4) ; boucle des centres : beta <1 - 1/2 Tr W> = {gel["beta_un_moins_W_boucle_centres"]:.12f} (libre {gel["beta_un_moins_W_boucle_centres_libre"]:.12f})')
ligne('INFO', 'FIB-BINF-DUREE', f'{time.time() - t0:.0f} s')
print('BILAN FIB-BETA-INF', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
