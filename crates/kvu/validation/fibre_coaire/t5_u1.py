#!/usr/bin/env python3
"""WP11 / FIB-U1 : cas U(1) exact (docs/FIBRE_COAIRE.md §5), réseau 6^4 (L_c = 2).

(a) moyenne linéaire Q : jauge résiduelle (lambda = 0 aux centres) dans ker Q ; Q(champ constant a e_mu) = 3a (RG-ABEL) ;
(b) la carte SU(2) de balaban_np restreinte au sous-groupe diagonal exp(i th s3) EST la carte U(1) :
    petits angles : phibar = Q th exactement ; grands angles : phibar = Q th - 2 pi sum_G w_G n_G (n_G entiers de branche) ;
(c) jacobien CONSTANT : bloc (3,3) de J = Q pour toute configuration diagonale (même à grands angles), et découplage des
    directions chargées ;
(d) loi conditionnelle gaussienne exacte (S = beta th^T M th, M = 1/2 rot^T rot) : dimensions, équipartition, table des
    variances de plaquettes par classe, boucle 3x3 des centres, extension harmonique H (KKT) et M_c ;
(e) route A (pénalité) gaussienne exacte en fonction de kappa = beta eps^2 (aux kappa de FIB-AB et FIB-GAUSS) ;
(f) VALEURS GELÉES -> valeurs_u1.json.
"""
import os, sys, json, time
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
L = 6
ops = O.Ops(L)
Q = ops.Q
rng = np.random.default_rng(26102616)

# ---------------------------------------------------------------- (a)
D = ops.d(); nc = np.setdiff1d(np.arange(ops.V), ops.centres)
e1 = float(np.abs(Q @ D[:, nc]).max())
cst = np.zeros(ops.N); cst[0:ops.V] = 0.37                     # a = 0,37 sur les liens mu = 0
e2 = float(np.abs(Q @ cst - np.concatenate([np.full(ops.Vc, 3 * 0.37), np.zeros(3 * ops.Vc)])).max())
e3 = float(np.abs(Q.sum(1) - 3.0).max())                        # somme signée des liens d'un chemin de c à c' = 3 pas dans mu
ligne('PASS' if max(e1, e2, e3) < 1e-13 else 'FAIL', 'FIB-U1-Q', f'Q d lambda = 0 si lambda(centres) = 0 : {e1:.1e} ; Q(a e_0) = 3a e_0 : {e2:.1e} ; somme signée des lignes = 3 : {e3:.1e} ; (rang de Q = {np.linalg.matrix_rank(Q)} = n = {ops.n})')

# ---------------------------------------------------------------- (b), (c)
def diag_q(th):
    q = np.zeros((ops.N, 4)); q[:, 0] = np.cos(th); q[:, 3] = np.sin(th); return q
eb = 0.0; ebr = 0.0; ec = 0.0; ed = 0.0; nsaut = 0
for amp in (0.1, 1.0):
    th = amp * rng.normal(size=ops.N)
    q = diag_q(th)
    for (z, mu) in [((0, 0, 0, 0), 0), ((1, 0, 1, 1), 1), ((0, 1, 1, 0), 3)]:
        tab = ops.bloc.tables(z, mu)
        c = mu * ops.Vc + ((z[0] * 2 + z[1]) * 2 + z[2]) * 2 + z[3]
        u, _ = ops.bloc.ubar(q, z, mu, tab)
        phib = np.arctan2(u[3], u[0])
        I, Sg, W, Iref = tab
        thG = np.sum(np.where(Sg != 0, Sg * th[I], 0.0), axis=1); th0 = th[Iref].sum()
        red = (thG - th0 + np.pi) % (2 * np.pi) - np.pi          # angle principal de U0^-1 U(G)
        nG = np.round((thG - th0 - red) / (2 * np.pi))
        nsaut += int((nG != 0).sum())
        pred = Q[c] @ th - 2 * np.pi * np.sum(W * nG)
        dphi = (phib - pred + np.pi) % (2 * np.pi) - np.pi
        if amp == 0.1:
            eb = max(eb, abs(dphi), float(np.abs(u[1:3]).max()))
        else:
            ebr = max(ebr, abs(dphi), float(np.abs(u[1:3]).max()))
        liens, J, _ = ops.bloc.jac(q, z, mu, tab)
        Qc = Q[c, liens]
        ec = max(ec, float(np.abs(J[:, 2, 2] - Qc).max()))
        ed = max(ed, float(np.abs(J[:, 2, :2]).max()), float(np.abs(J[:, :2, 2]).max()))
ligne('PASS' if eb < 1e-13 else 'FAIL', 'FIB-U1-CARTE-PETITS', f'configurations diagonales exp(i th s3), amplitude 0,1 : phibar = Q th, max écart {eb:.1e}')
ligne('PASS' if ebr < 1e-12 else 'FAIL', 'FIB-U1-CARTE-BRANCHES', f'amplitude 1,0 : phibar = Q th - 2 pi sum w_G n_G (n_G entiers de branche, {nsaut} chemins sautés sur 3 x 10473) : max écart {ebr:.1e}')
ligne('PASS' if max(ec, ed) < 1e-12 else 'FAIL', 'FIB-U1-JACOBIEN-CONSTANT', f'bloc (3,3) de J = Q (amplitudes 0,1 et 1,0) : {ec:.1e} ; blocs mixtes U(1)/chargés nuls : {ed:.1e}')

# ---------------------------------------------------------------- (d) loi gaussienne exacte
G = O.Gauss(ops)
ligne('PASS' if G.err_jauge_dans_kerQ < 1e-13 and G.dimS == 3 * (ops.V - ops.Vc) else 'FAIL', 'FIB-U1-DIMENSIONS',
      f'dim(ker Q ∩ tranche) = {G.dimS} (attendu 3 (L^4 - L_c^4) = {3 * (ops.V - ops.Vc)}) ; cond(M_V) = {np.linalg.cond(np.linalg.inv(G.MVi)):.3e}')
M = G.Mmat
equi = 0.5 * np.trace(M @ G.Gam)                       # beta <S> = Tr(M Gam)/2 = dim S / 2  (phibar = 0)
ligne('PASS' if abs(equi - G.dimS / 2) < 1e-8 else 'FAIL', 'FIB-U1-EQUIPARTITION', f'<S> = Tr(M Gam)/2 = {equi:.10f} = dim S/2 = {G.dimS / 2}')
varp = G.cov_F_beta()                                   # beta <th_p^2> (phibar = 0), 6V plaquettes
# invariance par translation de 3
cls = {}
for p in range(6 * ops.V):
    k, s = divmod(p, ops.V)
    r = tuple(ops.X[s] % 3)
    cls.setdefault((k, r), []).append(varp[p])
etr = max(max(v) - min(v) for v in cls.values())
moy = float(varp.mean())
vals = np.array(sorted({round(float(np.mean(v)), 12) for v in cls.values()}))
ligne('PASS' if etr < 1e-11 and abs(moy - 40 / 81) < 1e-11 else 'FAIL', 'FIB-U1-PLAQUETTES',
      f'beta <th_p^2> : invariance par translation de 3 : {etr:.1e} ; moyenne = {moy:.12f} (40/81 = {40/81:.12f}) ; {len(cls)} classes (r, plan), {len(vals)} valeurs distinctes dans [{vals.min():.6f}, {vals.max():.6f}]')
# boucle 3x3 des centres (c = (1,1,1,1), plan 01)
def boucle_centres(mu, nu, c0=(1, 1, 1, 1)):
    w = np.zeros(ops.N); e = np.eye(4, dtype=int); c0 = np.array(c0)
    for k in range(3):
        w[mu * ops.V + ops.site(c0 + k * e[mu])] += 1
        w[nu * ops.V + ops.site(c0 + 3 * e[mu] + k * e[nu])] += 1
        w[mu * ops.V + ops.site(c0 + 3 * e[nu] + k * e[mu])] -= 1
        w[nu * ops.V + ops.site(c0 + k * e[nu])] -= 1
    return w
wW = boucle_centres(0, 1)
vW_cond = 0.5 * wW @ G.Gam @ wW
Mp = np.linalg.pinv(M, rcond=1e-10, hermitian=True)
vW_libre = 0.5 * wW @ Mp @ wW
pl_libre = 0.5 * np.einsum('pi,pi->p', ops.rot(Mp), ops.rot(np.eye(ops.N)))
ligne('INFO', 'FIB-U1-BOUCLE-CENTRES', f'beta Var(th_W), boucle 3x3 des centres (plan 01) : conditionnelle {vW_cond:.12f} ; libre (sans contrainte) {vW_libre:.12f} ; moyenne beta <th_p^2> libre {pl_libre.mean():.12f} ((3V-3)/(6V) = {(3*ops.V-3)/(6*ops.V):.12f})')
# extension harmonique : Q H = 1, M_c
eH = float(np.abs(Q @ G.H - np.eye(ops.n)).max())
Mc = G.Mc
dMc = np.diag(Mc)
ligne('PASS' if eH < 1e-12 and np.ptp(dMc) < 1e-12 else 'FAIL', 'FIB-U1-KKT', f'Q H = 1 : {eH:.1e} ; M_c[c,c] = {dMc.mean():.12f} (tous égaux à {np.ptp(dMc):.1e} près) ; S_min = beta phibar^T M_c phibar')

# ---------------------------------------------------------------- (e) route A gaussienne
kappas = {'b8_e0.2': 8 * 0.04, 'b8_e0.1': 8 * 0.01, 'b8_e0.05': 8 * 0.0025, 'b2.8_e0.2': 2.8 * 0.04, 'b2.8_e0.1': 2.8 * 0.01,
          'b2.8_e0.05': 2.8 * 0.0025, 'b1000_e0.2': 1000 * 0.04, 'b1000_e0.1': 1000 * 0.01, 'b1000_e0.05': 1000 * 0.0025,
          'k1e-6': 1e-6, 'k1e-5': 1e-5, 'k1e-4': 1e-4, 'k3e-4': 3e-4, 'k1e-3': 1e-3, 'k3e-3': 3e-3, 'k1e-2': 1e-2, 'k3e-2': 3e-2,
          'k1e-1': 1e-1, 'k3e-1': 3e-1, 'k1': 1.0, 'k3': 3.0, 'k10': 10.0}
evMc = np.linalg.eigvalsh(G.Mc)
ligne('INFO', 'FIB-U1-SPECTRE-MC', f'spectre de M_c (64 valeurs) : min {evMc.min():.6f}, max {evMc.max():.6f} ; la pénalité domine un mode grossier ssi 2 kappa lambda << 1 (kappa = beta eps^2)')
routeA = {}
R_eye = ops.rot(np.eye(ops.N))
for nom, kap in kappas.items():
    Ci = G.penalite(kap)                                 # beta cov(theta) (partie invariante)
    S_eps = float(np.trace(M @ Ci))                      # beta <S>/beta... = Tr(M beta cov) : <S> exact
    vW = float(wW @ Ci @ wW)
    pl = float(np.einsum('pi,pi->p', ops.rot(Ci), R_eye).mean())
    routeA[nom] = dict(kappa=kap, S_moy=S_eps, beta_var_boucle_centres=vW, beta_th2_moy=pl,
                       ecart_relatif_boucle=vW / vW_cond - 1)
ecv = abs(routeA['k1e-6']['beta_var_boucle_centres'] / vW_cond - 1)
ligne('PASS' if ecv < 1e-4 else 'FAIL', 'FIB-U1-ROUTEA-LIMITE', f'kappa -> 0 : beta Var(th_W) (pénalité, kappa = 1e-6) / conditionnelle - 1 = {ecv:.2e} (deux calculs indépendants : inverse pénalisée / base de la tranche)')
for nom in ('b8_e0.2', 'b8_e0.1', 'b8_e0.05', 'b2.8_e0.2', 'b2.8_e0.1', 'b2.8_e0.05', 'b1000_e0.05', 'k1e-4', 'k1e-3', 'k1e-2'):
    r = routeA[nom]
    ligne('INFO', f'FIB-U1-ROUTEA-{nom}', f'kappa = {r["kappa"]:.4g} : <S> = {r["S_moy"]:.6f} (exact {G.dimS/2}) ; beta Var(th_W) = {r["beta_var_boucle_centres"]:.6f} (exact {vW_cond:.6f}, écart {100*r["ecart_relatif_boucle"]:+.2f} %)')

# ---------------------------------------------------------------- (e2) vecteur de 10 moments (FIB-U1) : 9 classes + boucle des centres
CLES = sorted(cls.keys())
def classes_de(diag_beta_th2):
    acc = {}
    for p in range(6 * ops.V):
        k, s_ = divmod(p, ops.V)
        acc.setdefault((k, tuple(ops.X[s_] % 3)), []).append(diag_beta_th2[p])
    return {c: float(np.mean(v)) for c, v in acc.items()}
# regroupement des 486 classes en 9 types (côtés franchis, transverses au centre)
def type_classe(c):
    k, r = c; mu, nu = O.PAIRES[k]
    dans = sum(1 for m in (mu, nu) if r[m] == 2)
    trans = sum(1 for m in range(4) if m not in (mu, nu) and r[m] == 1)
    return (dans, trans)
TYPES = [(2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]
def vecteur_m(diag_beta_th2, var_W):
    cl = classes_de(diag_beta_th2); par = {}
    for c, v in cl.items():
        par.setdefault(type_classe(c), []).append(v)
    etal = max(max(v) - min(v) for v in par.values())
    return [float(np.mean(par[t])) for t in TYPES] + [float(var_W)], etal
m_exact, etal0 = vecteur_m(varp, vW_cond)
ligne('PASS' if etal0 < 1e-11 else 'FAIL', 'FIB-U1-TYPES', f'les 486 classes se réduisent à 9 types (côtés franchis, transverses au centre) : dispersion interne {etal0:.1e} ; m (contrainte dure, beta-réduit) = {[round(x, 10) for x in m_exact]}')
m_eps = {}
for nom, kap in (('b8_e0.2', 0.32), ('b8_e0.1', 0.08), ('b8_e0.05', 0.02), ('b2.8_e0.2', 0.112), ('b2.8_e0.1', 0.028), ('b2.8_e0.05', 0.007),
                 ('k1e-3', 1e-3), ('k2e-3', 2e-3), ('k4e-3', 4e-3)):
    Ci = G.penalite(kap)
    d = np.einsum('pi,pi->p', ops.rot(Ci), R_eye)
    m_k, et = vecteur_m(d, wW @ Ci @ wW)
    m_eps[nom] = dict(kappa=kap, m=m_k, dispersion_types=et)
ligne('INFO', 'FIB-U1-M-EPS', ' ; '.join(f'{k} (kappa {v["kappa"]:g}) : m1 = {v["m"][0]:.6f}, m10 = {v["m"][9]:.6f}' for k, v in m_eps.items()))
# mutant du brouillon v4 (pilote T2) : V décalé de exp(0,05 i s3) sur UN lien grossier, références inchangées
phi = np.zeros(ops.n); phi[0] = 0.05
th_star = G.H @ phi
Fst = ops.rot(th_star)
dW = float(wW @ th_star)
mut = {}
for beta in (8.0, 2.8):
    dm, _ = vecteur_m(beta * Fst**2, beta * dW**2)      # décalage de beta <th_p^2> et de beta <th_W^2> (la variance ne change pas)
    mut[str(beta)] = dict(delta_m=dm, delta_relatif=[a / b for a, b in zip(dm, m_exact)], max_beta_F2_plaquette=float(beta * np.max(Fst**2)))
    ligne('INFO', f'FIB-U1-MUTANT-V-b{beta:g}', f'phibar_0 = 0,05 : décalage relatif des 10 moments (moyennes de classes sur les 16 blocs) max {100*max(abs(x) for x in mut[str(beta)]["delta_relatif"]):.4f} % ; plaquette la plus touchée beta F*^2 = {mut[str(beta)]["max_beta_F2_plaquette"]:.3e} (contre beta Var ~ 0,49)')
# ---------------------------------------------------------------- (f) valeurs gelées
gel = dict(
    reseau='L = 6 (L_c = 2), disposition kvu, S = beta th^T M th = (beta/2) sum_p th_p^2 (U(1) gaussienne ; = chaque couleur de SU(2) en coordonnées demi-angle w à beta -> infini)',
    dim_tranche=G.dimS, S_moy_conditionnel=G.dimS / 2, beta_th2_moy=moy,
    beta_th2_par_classe={f'{O.PAIRES[k][0]}{O.PAIRES[k][1]}_r{"".join(map(str, r))}': round(float(np.mean(v)), 14) for (k, r), v in sorted(cls.items())},
    beta_var_boucle_centres_conditionnelle=vW_cond, beta_var_boucle_centres_libre=vW_libre,
    Mc_diag=float(dMc.mean()), Mc_premiere_ligne=[round(float(x), 14) for x in Mc[0]], Mc_spectre=[round(float(x), 12) for x in evMc],
    routeA=routeA,
    vecteur_m=dict(definition='m = [beta<th_p^2> moyen des 9 types (côtés franchis, transverses au centre) dans l\'ordre ' + str(TYPES) + ', beta Var(th_W) boucle 3x3 des centres plan 01] ; loi U(1) gaussienne non compacte, phibar = 0 ; moments physiques = m/beta', exact=m_exact, route_A=m_eps, mutant_V_decale_0_05=mut))
json.dump(gel, open(os.path.join(SORTIES, 'valeurs_u1.json'), 'w'), indent=1, ensure_ascii=False)
ligne('INFO', 'FIB-U1-DUREE', f'{time.time() - t0:.0f} s')
print('BILAN FIB-U1', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
