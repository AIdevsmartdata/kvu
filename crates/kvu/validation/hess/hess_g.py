"""G1, M-b, G2, G3, M-c (GATES v4-T2 §1.3), calculés avec numpy/lat.py, indépendamment de kvu pour les références.
G1  : différences secondes de s -> S(chart(s zeta)) et croisées (zeta, xi), Richardson (h, h/2, h/4 ; h = 1e-3), contre <xi, H zeta> de `kvu hess hv`.
      chart gauche : U_l -> exp(i a_l.sigma/2) U_l ; chart additif : U_l = exp(i (A_l + a_l).sigma/2).
      Directions à support sur 6 liens (4 d'une plaquette + 2 voisins aléatoires) : seules les plaquettes touchant le support entrent dans les différences
      (même valeur exacte de la différence d'action, sans l'erreur d'arrondi de la somme globale). 10 paires, 2 fonds.
      erreur relative = |FD - an| / max(|an|, |<z,Hz>|, |<x,Hx>|)   (échelle écrite dans PREVOL.md).
M-b : mêmes calculs avec `--mutant M-b`.
G2  : H(U=1) = Q0 (base s2, beta = 1), 4^4 et 6^4, assemblée par colonnes.
G3  : jauge pure U_l = Om_x Om_{x+mu}^+ (Om Haar, graine 26102613021), 4^4 : T-cov (gauche, qcov, sans fixation) ; T-fix (additif, kerB, q0, après landau_fix de lat.py).
M-c : T-fix sans fixation de Landau."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
lat = O.charger_lat()
sc = O.SCRATCH + '/hess_g'; os.makedirs(sc, exist_ok=True)
L = 4; geo = lat.Geo(L); n = 12 * L ** 4
tout = True

# ---------------------------------------------------------------------------- G1
LD = np.longdouble

def vers_quat(U):
    """liens 2x2 -> quaternions (nl,4) en précision étendue (conversion exacte)."""
    return np.stack([U[..., 0, 0].real, U[..., 0, 1].imag, U[..., 0, 1].real, U[..., 0, 0].imag], axis=-1).astype(LD)

def qmul(a, b):
    return np.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3],
                     a[..., 0] * b[..., 1] + b[..., 0] * a[..., 1] - (a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2]),
                     a[..., 0] * b[..., 2] + b[..., 0] * a[..., 2] - (a[..., 3] * b[..., 1] - a[..., 1] * b[..., 3]),
                     a[..., 0] * b[..., 3] + b[..., 0] * a[..., 3] - (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1])], axis=-1)

def qdag(a):
    return a * np.array([1, -1, -1, -1], dtype=LD)

def qexp(a):
    """exp(i a.sigma/2) en quaternion, précision étendue (a (n,3))."""
    th = np.sqrt((a.astype(LD) ** 2).sum(-1)) / 2
    s = np.where(th > 0, np.sin(th) / np.where(th > 0, th, 1), 1) / 2
    return np.concatenate([np.cos(th)[:, None], a.astype(LD) * s[:, None]], axis=-1)

def S_plaq(Q, plist):
    Lp = geo.plaq[plist]
    P = qmul(qmul(Q[Lp[:, 0]], Q[Lp[:, 1]]), qmul(qdag(Q[Lp[:, 2]]), qdag(Q[Lp[:, 3]])))
    return np.sum(1 - P[:, 0], dtype=LD)   # précision étendue conservée

def chart(Q, A, mode, a):
    """a (nl,3) -> quaternions des liens perturbés (précision étendue)."""
    if mode == 'gauche':
        return qmul(qexp(a), Q)
    return qexp(A.astype(LD) + a.astype(LD))

def d2_rich(Q, A, mode, d, plist, h=1e-3):
    """dérivée seconde de s -> S(chart(s d)) en 0 : Richardson (h, h/2, h/4), différences secondes centrées, précision étendue.
    Le point central est chart(0) (et non Q) pour que les trois évaluations partagent exactement la même conversion."""
    d = d.astype(LD)
    c0 = S_plaq(chart(Q, A, mode, 0 * d), plist)
    def D(hh):
        hh = LD(hh)
        return float((S_plaq(chart(Q, A, mode, hh * d), plist) - 2 * c0 + S_plaq(chart(Q, A, mode, -hh * d), plist)) / hh ** 2)
    R1 = (4 * D(h / 2) - D(h)) / 3
    R2 = (4 * D(h / 4) - D(h / 2)) / 3
    return (16 * R2 - R1) / 15

def g1(nomfond, Ul, A, beta_unused, graine_dir, mutant=None):
    rng = np.random.default_rng(graine_dir)
    paires = []
    for k in range(10):
        p = rng.integers(geo.np)
        liens = list(geo.plaq[p]) + list(rng.choice(geo.nl, size=2, replace=False))
        liens = sorted(set(int(x) for x in liens))
        def dirv():
            z = np.zeros((geo.nl, 3))
            z[liens] = rng.standard_normal((len(liens), 3))
            return z / np.linalg.norm(z)
        paires.append((dirv(), dirv(), liens))
    plists = []
    for z, x, liens in paires:
        pl = sorted(set(pp for l in liens for pp, _ in geo.plaq_of_link[l]))
        plists.append(pl)
    cfg = sc + '/%s.npy' % nomfond
    O.ecrire_npy(cfg, O.liens_lat_vers_champ(Ul, L))
    Qd = vers_quat(Ul)
    res = {}
    for mode in ('gauche', 'additive'):
        Z = np.array([z.reshape(-1) for z, _, _ in paires]); X = np.array([x.reshape(-1) for _, x, _ in paires])
        O.ecrire_npy(sc + '/%s_Z.npy' % nomfond, Z)
        args = ['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/%s_Z.npy' % nomfond, '--sortie', sc + '/%s_HZ_%s.npy' % (nomfond, mode),
                '--base', 's2', '--coord', mode, '--fils', O.FILS]
        if mutant:
            args += ['--mutant', mutant]
        O.kvu(args)
        HZ = np.load(sc + '/%s_HZ_%s.npy' % (nomfond, mode))
        # <x, H x> par un second appel
        O.ecrire_npy(sc + '/%s_X.npy' % nomfond, X)
        args2 = list(args); args2[args2.index(sc + '/%s_Z.npy' % nomfond)] = sc + '/%s_X.npy' % nomfond; args2[args2.index(sc + '/%s_HZ_%s.npy' % (nomfond, mode))] = sc + '/%s_HX_%s.npy' % (nomfond, mode)
        O.kvu(args2)
        HX = np.load(sc + '/%s_HX_%s.npy' % (nomfond, mode))
        errs = []; errs_fd = []
        for k, (z, x, liens) in enumerate(paires):
            pl = plists[k]
            qz = d2_rich(Qd, A, mode, z, pl); qx = d2_rich(Qd, A, mode, x, pl); qzx = d2_rich(Qd, A, mode, z + x, pl)
            fd = 0.5 * (qzx - qz - qx)
            an = float(X[k] @ HZ[k]); anz = float(Z[k] @ HZ[k]); anx = float(X[k] @ HX[k])
            assert all(np.isfinite([fd, an, anz, anx, qz, qx]))
            echelle = max(abs(an), abs(anz), abs(anx))
            echelle_fd = max(abs(fd), abs(qz), abs(qx))           # échelle indépendante de l'opérateur testé (avenant A2)
            errs.append(abs(fd - an) / echelle)
            errs_fd.append(abs(fd - an) / echelle_fd)
        res[mode] = max(errs); res[mode + '_fd'] = max(errs_fd)
    return res

fondsG1 = []
Ch = O.haar_c16(L, 26102613031); Ul = O.champ_vers_liens_lat(Ch)
fondsG1.append(('haar', Ul, lat.su2_log(Ul)))
gs = lat.Gauss(geo, 0.25)
A = gs.draw(np.random.default_rng(26102613032), 1.0 / 8.0, 3, coarse=False)
fondsG1.append(('gauss_b8', lat.su2_exp(A), A))
pire = {'gauche': 0.0, 'additive': 0.0, 'gauche_fd': 0.0, 'additive_fd': 0.0}; pire_mb = {'gauche': 1e9, 'additive': 1e9, 'gauche_fd': 1e9, 'additive_fd': 1e9}; pire_mb_max = {'gauche': 0.0, 'additive': 0.0, 'gauche_fd': 0.0, 'additive_fd': 0.0}
for nom, Ulx, Ax in fondsG1:
    r = g1(nom, Ulx, Ax, None, 26102613033)
    rm = g1(nom + '_mb', Ulx, Ax, None, 26102613033, mutant='M-b')
    for m in r:
        pire[m] = max(pire[m], r[m]); pire_mb[m] = min(pire_mb[m], rm[m]); pire_mb_max[m] = max(pire_mb_max[m], rm[m])
    print('  G1 fond %s : correct %s ; mutant M-b %s' % (nom, r, rm), flush=True)
ok = max(pire.values()) <= 1e-6
tout &= ok
O.verdict(f, 'G1', ok, 'max relatif (echelle de H) gauche=%.3e additive=%.3e ; (echelle FD independante, avenant A2) gauche=%.3e additive=%.3e ; seuil 1e-6 sur les quatre ; 2 fonds x 10 paires' % (pire['gauche'], pire['additive'], pire['gauche_fd'], pire['additive_fd']))
# M-b : minimum par fond (le plus strict) du maximum sur les paires, sur les deux charts et les deux échelles
okmb = min(pire_mb.values()) >= 0.1 and max(pire_mb_max.values()) >= 0.1
tout &= okmb
O.verdict(f, 'M-b', okmb, 'ecart relatif du mutant : max sur fonds/paires gauche=%.3e additive=%.3e ; min sur les fonds du max sur les paires (echelle H) gauche=%.3e additive=%.3e ; (echelle FD) gauche=%.3e additive=%.3e ; >= 0,1 exige (ROUGE)' % (pire_mb_max['gauche'], pire_mb_max['additive'], pire_mb['gauche'], pire_mb['additive'], pire_mb['gauche_fd'], pire_mb['additive_fd']))

# ---------------------------------------------------------------------------- G2
def g2(Ln):
    gg = lat.Geo(Ln); nn = 12 * Ln ** 4
    cfg = sc + '/froid_%d.npy' % Ln
    froid = np.zeros((4, Ln, Ln, Ln, Ln, 2, 2), complex); froid[..., 0, 0] = 1; froid[..., 1, 1] = 1
    O.ecrire_npy(cfg, froid)
    Q0 = 0.25 * (gg.R.T @ gg.R)
    import scipy.sparse as sp
    Q0 = sp.kron(Q0, sp.eye(3), format='csr')
    err = 0.0
    pas = 1024 if Ln == 4 else 1024
    for debut in range(0, nn, pas):
        fin = min(nn, debut + pas)
        B = np.zeros((fin - debut, nn)); B[np.arange(fin - debut), np.arange(debut, fin)] = 1.0
        O.ecrire_npy(sc + '/g2_id.npy', B)
        O.kvu(['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/g2_id.npy', '--sortie', sc + '/g2_out.npy', '--base', 's2', '--coord', 'gauche', '--beta', 1, '--fils', O.FILS])
        Y = np.load(sc + '/g2_out.npy')      # lignes = colonnes de H
        ref = Q0[:, debut:fin].toarray().T
        assert np.isfinite(Y).all()
        err = max(err, np.abs(Y - ref).max())
    assert np.isfinite(err)
    return err / np.abs(Q0.data).max()
e4 = g2(4); e6 = g2(6)
ok = e4 <= 1e-10 and e6 <= 1e-10
tout &= ok
O.verdict(f, 'G2', ok, 'max|dH|/max|Q0| : 4^4 = %.3e, 6^4 = %.3e (seuil 1e-10)' % (e4, e6))

# ---------------------------------------------------------------------------- G3, M-c
Chaar = O.haar_c16(L, 26102613021)  # Om de Haar : une matrice par site, première direction de ce tirage
Om = Chaar[0].reshape(-1, 2, 2)
cold = np.broadcast_to(np.eye(2, dtype=complex), (geo.nl, 2, 2)).copy()
Upg = lat.gauge_transform(geo, cold, Om)
beta = 2.3
cfg = sc + '/pg.npy'; O.ecrire_npy(cfg, O.liens_lat_vers_champ(Upg, L))
def lmin(cfgp, espace, coord, ref, nom):
    rj = sc + '/%s.json' % nom
    O.kvu(['hess', 'lmin', '--config', cfgp, '--espace', espace, '--coord', coord, '--reference', ref, '--beta', beta, '--graine', 1, '--tol', '1e-10', '--sortie', rj, '--fils', O.FILS])
    return O.rapport_lmin(rj)
R = lmin(cfg, 'tcov', 'gauche', 'qcov', 'g3_tcov')
eps_cov = 1 - R['lambda_min']
ok = abs(eps_cov) <= 1e-9
tout &= ok
O.verdict(f, 'G3-tcov', ok, "eps' = %.3e (seuil 1e-9) ; dim X = %s ; residu projete %.2e" % (eps_cov, R['dim'], R['residu_projete_rel']))
Ufx, theta, it = lat.landau_fix(geo, Upg, tol=1e-26, maxit=20000)
cfgf = sc + '/pg_fix.npy'; O.ecrire_npy(cfgf, O.liens_lat_vers_champ(Ufx, L))
Rf = lmin(cfgf, 'kerB', 'additive', 'q0', 'g3_tfix')
eps_fix = 1 - Rf['lambda_min']
ok = abs(eps_fix) <= 1e-6
tout &= ok
O.verdict(f, 'G3-tfix', ok, "eps' = %.3e (seuil 1e-6) ; landau theta=%.2e it=%d ; max|A|=%.2e" % (eps_fix, theta, it, np.abs(lat.su2_log(Ufx)).max()))
Rm = lmin(cfg, 'kerB', 'additive', 'q0', 'mc')
eps_mc = 1 - Rm['lambda_min']
O.fin(eps_cov, eps_fix, eps_mc)
ok = abs(eps_mc) >= 1e-3
tout &= ok
O.verdict(f, 'M-c', ok, "eps' sans fixation de Landau = %.3e (>= 1e-3 exige : ROUGE)" % eps_mc)
sys.exit(0 if tout else 1)
