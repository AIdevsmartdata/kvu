#!/usr/bin/env python3
"""lecture_spectro.py : lecteur spectroscopique des sorties de `kvu su2 spectro gpu` (et `spectro neutre`), 20/11/2026.
numpy + scipy (scipy seulement pour l'ajustement multi-exponentiel).

Formats lus (identiques pour la carte et le processeur ; manifeste.json donne nops, tau_max, cotes, temps, niveaux_ape, formes) :
  DOSSIER/neutre.bin           : par configuration, paires (a<=b) x (tau_max+1) f64 puis nops moyennes <S_op> f64   (canal 0, A1++ : vide soustrait)
  DOSSIER/neutre_p<k>.bin      : paires x (tau_max+1) f64, impulsion k.2pi/l (canal k >= 1 ; pas de soustraction)
  DOSSIER/reps/<REP>/neutre.bin: representations cubiques (REP = Ep, T1m, ... ; dossier lu avec SON manifeste ; pas de soustraction)
Canal : 0 | 1,2,3,... (impulsion) | rep:NOM.

  lecture_spectro.py reduire REP_DOSSIER SORTIE.npz [--canal C] [--B 10]      (flux : un replica -> moyennes par blocs)
  lecture_spectro.py lecture --canal C [--B 10] [--json f] DOSSIER_OU_NPZ ... (lecture complete, voir ci-dessous)
  lecture_spectro.py autotest                                                  (donnees synthetiques a energies connues)

LECTURE COMPLETE (tout est FIGE ICI, avant de regarder une energie) :
 (1) bloc = B configurations (defaut 10 ; avertissement si 2*tau_int > B) ; jackknife par bloc ; soustraction du vide par echantillon (canal 0).
 (2) base PRINCIPALE : bases emboitees B15 > B9 > B6 > B3 (regle de analyse_neutre.py du depot), GEVP a tau0 = 1, plateau regle N
     (fenetres de 2 points, chi2 < 2, erreur relative < 10 %, fenetre suivante a moins de 1 sigma), sinon « pas de plateau ».
 (3) base ELAGUEE (secondaire) : operateurs classes par rapport signal/bruit de C_oo(1) decroissant (moyenne / erreur) ; on garde les m premiers,
     m = le plus grand m >= 2 tel que (i) toutes les diagonales de C(1) soient > 0 et (ii) la plus petite valeur propre de C(1) normalisee a
     diagonale 1 soit > 5 ecarts-types jackknife (conditionnement de C(tau0) : on indique aussi le rapport lambda_max/lambda_min). Le critere ne
     depend d'aucune energie.
 (4) BORNES RIGOUREUSES : pour un operateur diagonal C_oo(tau) = sum_n |c_n|^2 cosh(E_n (T/2 - tau)) (positivite de reflexion), le rapport
     C(tau)/C(tau+1) = cosh(E_eff (T/2-tau))/cosh(E_eff (T/2-tau-1)) definit E_eff(tau) >= E_0 : borne SUPERIEURE de l'etat fondamental du canal.
     Valable seulement si C(tau), C(tau+1) > 0 et le rapport > 1 sur TOUS les echantillons jackknife (sinon : « non definie »).
     On imprime E_eff(tau) +- sigma et E_eff + 2 sigma.
 (5) AJUSTEMENT MULTI-EXPONENTIEL (secondaire) du diagonal pre-designe (dernier niveau, forme S22 sinon premiere forme) : n = 2 exponentielles
     (cosh periodique T), tau de 1 a 4 (dof 0, poids diagonaux) puis 1 a 5 (poids diagonaux, covariance complete) ; contraintes E_{i+1}-E_i dans
     [0,3 ; DEMAX = 3], A_1 > 0, A_i >= 0 ; jackknife par 200 groupes. Option --nexp 3 : trois exponentielles (dE dans [0,3 ; 3]).
"""
import sys, os, json, math, glob, argparse
import numpy as np

# ------------------------------------------------------------------ statistique (conventions de analyse_spectro.py / analyse_neutre.py du depot)
def tau_int(x, c=6.0):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 20 or np.var(x) == 0:
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    rho = ac / ac[0]
    t = 0.5
    for m in range(1, n // 2):
        t += rho[m]
        if m >= c * t:
            return max(t, 0.5)
    return max(t, 0.5)

def jk_err(x):
    Nb = len(x)
    m = x.mean(axis=0)
    return m, np.sqrt((Nb - 1) / Nb * ((x - m) ** 2).sum(axis=0))

def matrice(Cp, paires, nn):
    M = np.zeros((Cp.shape[1], nn, nn))
    for k, (a, b) in enumerate(paires):
        M[:, a, b] = Cp[k]; M[:, b, a] = Cp[k]
    return M

def gevp(M, tau0, idx):
    A = M[tau0][np.ix_(idx, idx)]
    w, V = np.linalg.eigh(A)
    if w.min() <= 0:
        return None
    W = V @ np.diag(w ** -0.5) @ V.T
    out = []
    for t in range(M.shape[0]):
        B = W @ M[t][np.ix_(idx, idx)] @ W
        out.append(np.sort(np.linalg.eigvalsh((B + B.T) / 2))[::-1])
    return np.array(out)

def e_eff(lam, T):
    """lam [tau, n] -> E_eff [tau-1, n] par lam(tau)/lam(tau+1) = cosh(E(T/2-tau))/cosh(E(T/2-tau-1)) ; nan si impossible."""
    ntau, n = lam.shape
    E = np.full((ntau - 1, n), np.nan)
    for t in range(ntau - 1):
        a = T / 2 - t
        for k in range(n):
            r = lam[t, k] / lam[t + 1, k] if lam[t + 1, k] != 0 else np.nan
            if not np.isfinite(r) or r <= 1 or lam[t, k] <= 0 or lam[t + 1, k] <= 0:
                continue
            lo, hi = 1e-9, 40.0
            f = lambda e: math.cosh(min(e * a, 700)) / math.cosh(min(e * (a - 1), 700))
            if f(hi) < r:
                continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < r: lo = mid
                else: hi = mid
            E[t, k] = 0.5 * (lo + hi)
    return E

def ajuste_constante(Es, cov):
    Ci = np.linalg.inv(cov)
    one = np.ones(Es.shape[1])
    wgt = Ci @ one / (one @ Ci @ one)
    mu_s = Es @ wgt
    E0 = Es.mean(axis=0)
    mu = E0 @ wgt
    r = E0 - mu
    return mu_s, float(r @ Ci @ r), float(mu)

def plateau_N(Es, tau0, rel_max=0.10):
    Nb, nt = Es.shape
    res = {}
    for ta in range(tau0, nt - 1):
        W = Es[:, ta:ta + 2]
        if not np.all(np.isfinite(W)): continue
        Eb = W.mean(0); cov = (Nb - 1) / Nb * (W - Eb).T @ (W - Eb)
        try: res[ta] = ajuste_constante(W, cov)
        except np.linalg.LinAlgError: continue
    for ta in sorted(res):
        mu_s, chi2, mu = res[ta]
        m, e = jk_err(mu_s)
        if chi2 >= 2.0 or e / abs(mu) >= rel_max or (ta + 1) not in res: continue
        d = res[ta + 1][0] - mu_s
        _, dsig = jk_err(d)
        if abs(res[ta + 1][2] - mu) >= dsig: continue
        return dict(valeur=float(mu), erreur=float(e), fenetre=[ta, ta + 1], chi2=float(chi2))
    return None

def echantillons(Cb, Mb, soustraire=True):
    """echantillons jackknife du correlateur connexe [Nb, paires, tau]"""
    Nb = len(Cb)
    Cj = (Cb.sum(0)[None] - Cb) / (Nb - 1)
    if not soustraire: return Cj
    Mj = (Mb.sum(0)[None] - Mb) / (Nb - 1)
    nops = Mb.shape[1]; k = 0
    for a in range(nops):
        for b in range(a, nops):
            Cj[:, k, :] -= (Mj[:, a] * Mj[:, b])[:, None]; k += 1
    return Cj

# ------------------------------------------------------------------ lecture
def _canal(canal):
    canal = str(canal)
    if canal.startswith('rep:'): return ('rep', canal[4:])
    return ('p', int(canal))

def resoudre(dossier, canal):
    """-> (dossier du manifeste, fichier binaire, soustraire_vide, avec_M)"""
    kind, v = _canal(canal)
    if kind == 'rep':
        d = os.path.join(dossier, 'reps', v)
        return d, os.path.join(d, 'neutre.bin'), False, True
    if v == 0:
        return dossier, os.path.join(dossier, 'neutre.bin'), True, True
    return dossier, os.path.join(dossier, 'neutre_p%d.bin' % v), False, False

def lire_manifeste(dossier):
    return json.load(open(os.path.join(dossier, 'manifeste.json')))

def lire_plaq(dossier):
    return np.loadtxt(os.path.join(dossier, 'plaq.csv'), delimiter=',', skiprows=1, ndmin=2)[:, 1]

def reduire(dossier, canal=0, B=10, rep_plaq=None, morceau=2000):
    """Un replica, un canal -> moyennes par blocs de B configurations, EN FLUX (memmap : memoire ~ morceau * taille d'enregistrement).
    Retourne dict(Cb, Mb, tau_int, ncfg, B, man). Memes definitions que analyse_neutre.charger/taille_bloc/blocs du depot :
    k = min(n enregistrements, lignes de plaq.csv) ; tau_int = max(tau_int(plaq), tau_int de C_oo(1) pour chaque operateur)."""
    dm, fichier, sous, avecM = resoudre(dossier, canal)
    man = lire_manifeste(dm); nops = man['nops']; nt = man['tau_max'] + 1; npair = nops * (nops + 1) // 2
    rl = npair * nt + (nops if (_canal(canal)[0] == 'rep' or _canal(canal)[1] == 0) else 0)
    a = np.memmap(fichier, dtype='<f8', mode='r'); n = len(a) // rl
    a = a[:n * rl].reshape(n, rl)
    plaq_dir = dossier
    pl = lire_plaq(plaq_dir); k = min(n, len(pl))
    nb = k // B
    paires = [(i, j) for i in range(nops) for j in range(i, nops)]
    dg = [paires.index((o, o)) for o in range(nops)]
    Cb = np.empty((nb, npair, nt)); Mb = np.zeros((nb, nops)); serie_diag = np.empty((k, nops))
    for s in range(0, k, morceau):
        e = min(k, s + morceau)
        x = np.asarray(a[s:e])
        c = x[:, :npair * nt].reshape(e - s, npair, nt)
        serie_diag[s:e] = c[:, dg, 1]
        lo, hi = s // B, min(nb, e // B)
        if hi > lo:
            m = (hi - lo) * B
            Cb[lo:hi] = c[:m].reshape(hi - lo, B, npair, nt).mean(axis=1)
            if rl > npair * nt:
                Mb[lo:hi] = x[:m, npair * nt:].reshape(hi - lo, B, nops).mean(axis=1)
    tm = max(0.5, tau_int(pl[:k]), max(tau_int(serie_diag[:, o]) for o in range(nops)))
    return dict(Cb=Cb, Mb=Mb, tau_int=float(tm), ncfg=int(k), B=B, man=man)

def sauver_reduit(r, chemin):
    np.savez(chemin, Cb=r['Cb'], Mb=r['Mb'], tau_int=r['tau_int'], ncfg=r['ncfg'], B=r['B'], man=json.dumps(r['man']))

def charger_reduit(chemin):
    z = np.load(chemin)
    return dict(Cb=z['Cb'], Mb=z['Mb'], tau_int=float(z['tau_int']), ncfg=int(z['ncfg']), B=int(z['B']),
                man=json.loads(str(z['man'])) if 'man' in z.files else None)

def combiner(reductions):
    """Concatenation des blocs de meme taille = moyenne ponderee des correlateurs (poids = nombre de configurations)."""
    Bs = {r['B'] for r in reductions}
    if len(Bs) != 1: raise ValueError('tailles de bloc differentes : %s' % sorted(Bs))
    return dict(Cb=np.concatenate([r['Cb'] for r in reductions]), Mb=np.concatenate([r['Mb'] for r in reductions]),
                tau_int=max(r['tau_int'] for r in reductions), ncfg=sum(r['ncfg'] for r in reductions), B=Bs.pop(),
                man=reductions[0]['man'], nrep=len(reductions))

# ------------------------------------------------------------------ bases
def noms_ops(man):
    return [(lv, nm) for lv in man['niveaux_ape'] for nm in man['formes']]

def bases(man):
    ops = noms_ops(man); lv = man['niveaux_ape']
    idx = lambda pred: [i for i, (l, n) in enumerate(ops) if pred(l, n)]
    B = {'B15': list(range(len(ops))), 'B9': idx(lambda l, n: n in ('P11', 'R12', 'S22')),
         'B6': idx(lambda l, n: n in ('P11', 'R12')), 'B3': idx(lambda l, n: n in ('P11', 'R12', 'S22') and l == lv[len(lv) // 2])}
    return {k: v for k, v in B.items() if len(v) >= 1}

def analyser(Cj, nops, T, tau0, idx):
    paires = [(a, b) for a in range(nops) for b in range(a, nops)]
    Ls = []
    for c in Cj:
        l = gevp(matrice(c, paires, nops), tau0, idx)
        if l is None: return dict(base=idx, erreur="C(tau0) non defini positif sur au moins un echantillon")
        Ls.append(l)
    Es = np.array([e_eff(l, T) for l in np.array(Ls)])
    out = dict(base=idx, tau0=tau0, nb_blocs=len(Cj))
    for n in range(min(2, len(idx))):
        em, ee = jk_err(Es[:, :, n])
        out[f'etat{n}'] = dict(E_eff=[(int(t), float(em[t]), float(ee[t])) for t in range(len(em))], plateau=plateau_N(Es[:, :, n], tau0))
    return out

def elaguer(Cj, nops, Cb, tau0=1, sous_ech=4000):
    """Critere (3) de l'en-tete. -> (idx | None, journal). Journal : rapports S/B, ordre, et pour chaque m : (m, min-eig moyenne, sigma, critere, cond)."""
    paires = [(a, b) for a in range(nops) for b in range(a, nops)]
    dg = [paires.index((o, o)) for o in range(nops)]
    sn = np.array([Cb[:, d, tau0].mean() / (Cb[:, d, tau0].std(ddof=1) / math.sqrt(len(Cb))) for d in dg])
    ordre = list(np.argsort(-sn)); jr = {'sn': [float(x) for x in sn], 'ordre': [int(o) for o in ordre], 'essais': []}
    for m in range(nops, 1, -1):
        idx = sorted(int(o) for o in ordre[:m]); mins = []; conds = []
        for c in Cj[::max(1, len(Cj) // sous_ech)]:
            M = matrice(c, paires, nops)[tau0][np.ix_(idx, idx)]
            d = np.sqrt(np.diag(M))
            if not np.all(np.isfinite(d)) or np.any(np.diag(M) <= 0): mins.append(-1.0); conds.append(float('inf')); continue
            w = np.linalg.eigvalsh(M / np.outer(d, d)); mins.append(w.min()); conds.append(w.max() / w.min() if w.min() > 0 else float('inf'))
        mins = np.array(mins); n = len(mins)
        sd = math.sqrt((len(Cj) - 1) / len(Cj) * len(Cj) / n * ((mins - mins.mean()) ** 2).sum())
        crit = mins.mean() / sd if sd > 0 else float('inf')
        jr['essais'].append((m, float(mins.mean()), float(sd), float(crit), float(np.median(conds))))
        if mins.min() > 0 and crit > 5: return idx, jr
    return None, jr

def bornes_diagonales(Cj, nops, T, man):
    """Critere (4) : E_eff diagonal de chaque operateur (borne superieure rigoureuse de l'etat fondamental), definie ou non."""
    paires = [(a, b) for a in range(nops) for b in range(a, nops)]
    ops = noms_ops(man); out = {}
    for o in range(nops):
        k = paires.index((o, o))
        Es = np.array([e_eff(c[k, :, None], T)[:, 0] for c in Cj])
        definie = [bool(np.all(np.isfinite(Es[:, t]))) for t in range(Es.shape[1])]
        m, e = jk_err(Es)
        out[o] = dict(nom='niv%s_%s' % ops[o], E_eff=[(t, float(m[t]), float(e[t]), definie[t]) for t in range(len(m))])
    return out

# ------------------------------------------------------------------ ajustement multi-exponentiel (cosh periodique)
DEMAX = float(os.environ.get('DEMAX', '3.0'))   # borne superieure des ecarts E_{i+1}-E_i (fixee avant les energies : sans borne l'ajustement degenere)
DEMIN = 0.3

def modele_n(x, tau, T, n):
    E = np.cumsum(x[:n]); A = x[n:2 * n]
    f = lambda e: np.cosh(e * (T / 2 - tau)) / math.cosh(e * (T / 2 - 1))
    return sum(A[i] * f(E[i]) for i in range(n))

def ajuste_n(y, taus, T, W, n=2):
    """y [m] ; W [m,m] poids (inverse de covariance) ; -> (E1, chi2, params) ou None. Parametres : E1, dE2.., A1..An. n = 2 : memes
    departs et bornes que analyse_1120.ajuste2 du depot."""
    from scipy.optimize import least_squares
    L = np.linalg.cholesky(W); best = None; c1 = y[0]
    departs = {2: [(0.5,), (1.0,), (2.0,)], 3: [(0.5, 0.5), (1.0, 1.0), (2.0, 0.7)]}[n]
    amp0 = {2: [0.9, 0.1], 3: [0.8, 0.15, 0.05]}[n]
    for dE0 in departs:
        e1 = max(0.1, min(3.0, math.log(max(y[0] / y[1], 1.0001)) if len(y) > 1 else 1.0))
        x0 = [e1, *dE0, *[a * c1 for a in amp0]]
        lb = [1e-3] + [DEMIN] * (n - 1) + [1e-12] + [0.0] * (n - 1)
        ub = [8] + [DEMAX] * (n - 1) + [10 * abs(c1)] * n
        try:
            r = least_squares(lambda x: L.T @ (modele_n(x, taus, T, n) - y), x0, bounds=(lb, ub), x_scale=[1] * n + [abs(c1)] * n)
        except Exception:
            continue
        ch = float((r.fun ** 2).sum())
        if best is None or ch < best[1]: best = (float(r.x[0]), ch, [float(v) for v in r.x])
    return best

def predesigne(man):
    ops = noms_ops(man)
    s22 = [o for o in range(len(ops)) if ops[o][1] == 'S22']
    return s22[-1] if s22 else len(ops) - len(man['formes'])

def multi_exp(Cb, Mb, op, T, soustraire, taus, G=200, correle=False, n=2):
    """Critere (5) : jackknife par G groupes de blocs du diagonal `op`."""
    Nb = len(Cb); gid = np.arange(Nb) * G // Nb
    nops = Mb.shape[1]; paires = [(a, b) for a in range(nops) for b in range(a, nops)]; k = paires.index((op, op))
    S = np.array([Cb[gid == g].sum(0) for g in range(G)]); nn_ = np.array([(gid == g).sum() for g in range(G)], float)
    SM = np.array([Mb[gid == g].sum(0) for g in range(G)])
    def vec(mask):
        nx = nn_[mask].sum(); c = S[mask].sum(0)[k] / nx; m = SM[mask].sum(0)[op] / nx
        return (c - m * m) if soustraire else c
    full = vec(np.ones(G, bool)); jk = np.array([vec(np.arange(G) != g) for g in range(G)])
    out = {}
    for nom, tl in taus.items():
        t = np.arange(tl[0], tl[1] + 1)
        sub = jk[:, t]; cov = (G - 1) / G * (sub - sub.mean(0)).T @ (sub - sub.mean(0))
        W = np.linalg.inv(cov) if correle else np.diag(1 / np.diag(cov))
        b = ajuste_n(full[t], t, T, W, n)
        if b is None: out[nom] = None; continue
        Es = np.array([r[0] for r in (ajuste_n(jk[g][t], t, T, W, n) for g in range(G)) if r is not None])
        m = Es.mean(); e = math.sqrt((len(Es) - 1) / len(Es) * ((Es - m) ** 2).sum())
        out[nom] = dict(E1=b[0], err=e, chi2=b[1], dof=len(t) - 2 * n, params=b[2], echecs=G - len(Es), taus=[int(t[0]), int(t[-1])])
    return out

# ------------------------------------------------------------------ lecture complete
def lecture(red, canal=0, tmaxb=5, nexp=2, G=200, faire_multi=True):
    man = red['man']; T = man['cotes'][man['temps']]; nops = man['nops']
    kind, v = _canal(canal); soustraire = (kind == 'p' and v == 0)
    Cb, Mb = red['Cb'], red['Mb']
    out = dict(canal=str(canal), T=T, nops=nops, nb_blocs=len(Cb), ncfg=red['ncfg'], B=red['B'], tau_int=red['tau_int'], cotes=man['cotes'], beta=man['beta'])
    if 2 * red['tau_int'] > red['B']:
        out['avertissement'] = 'bloc trop court : 2*tau_int = %.2f > B = %d' % (2 * red['tau_int'], red['B'])
    Cj = echantillons(Cb, Mb, soustraire)
    out['principale'] = None; out['bases_refusees'] = []
    for nom, idx in bases(man).items():
        r = analyser(Cj, nops, T, 1, idx)
        if 'erreur' not in r:
            out['principale'] = dict(nom=nom, **{k: r[k] for k in ('etat0', 'etat1') if k in r}); break
        out['bases_refusees'].append(nom)
    idx, jr = elaguer(Cj, nops, Cb)
    out['elagage'] = dict(idx=idx, **jr)
    if idx is not None:
        r = analyser(Cj, nops, T, 1, idx)
        out['elaguee'] = {k: r[k] for k in ('etat0', 'etat1') if k in r} if 'erreur' not in r else dict(erreur=r['erreur'])
    out['diagonaux'] = bornes_diagonales(Cj, nops, T, man)
    pre = predesigne(man); out['pre_designe'] = int(pre)
    if faire_multi:
        out['multi_exp'] = multi_exp(Cb, Mb, pre, T, soustraire, {'b1_1-4': (1, 4), 'b2_1-%d' % tmaxb: (1, tmaxb)}, G, False, nexp)
        out['multi_exp_cov'] = multi_exp(Cb, Mb, pre, T, soustraire, {'b2c_1-%d' % tmaxb: (1, tmaxb)}, G, True, nexp)
    return out

def imprimer(r):
    f = lambda L, n=4: ' '.join('%d:%.4f(%.4f)' % (t, v, e) if math.isfinite(v) else '%d:--' % t for t, v, e in [x[:3] for x in L[:n]])
    print('canal %s : cotes %s beta %s, %d configurations, %d blocs de %d, tau_int max %.2f' % (r['canal'], r['cotes'], r['beta'], r['ncfg'], r['nb_blocs'], r['B'], r['tau_int']))
    if 'avertissement' in r: print('[!] ' + r['avertissement'])
    p = r['principale']
    if p is None: print('[!] aucune base principale definie positive (refusees : %s)' % r['bases_refusees'])
    else:
        for et in ('etat0', 'etat1'):
            if et in p: print('principale %s %s : E_eff %s | plateau %s' % (p['nom'], et, f(p[et]['E_eff']), p[et]['plateau']))
    e = r['elagage']
    if e['idx'] is None: print('[!] elagage : aucune base m >= 2 ne satisfait le critere')
    else:
        print('elagage : m = %d operateurs gardes %s ; retires %s' % (len(e['idx']), e['idx'], sorted(set(range(r['nops'])) - set(e['idx']))))
        el = r.get('elaguee', {})
        for et in ('etat0',):
            if et in el: print('elaguee %s : E_eff %s | plateau %s' % (et, f(el[et]['E_eff']), el[et]['plateau']))
    dg = r['diagonaux'][r['pre_designe']]
    print('borne rigoureuse (diagonal pre-designe %s) E_eff : %s' % (dg['nom'], ' '.join('%d:%.4f(%.4f)%s' % (t, v, e, '' if ok else '[non definie]') for t, v, e, ok in dg['E_eff'][:4])))
    for cle in ('multi_exp', 'multi_exp_cov'):
        for nom, x in r.get(cle, {}).items():
            print('multi-exp %s : %s' % (nom, 'echec' if x is None else 'E1 = %.4f +- %.4f (chi2 %.2f, dof %d, echecs %d)' % (x['E1'], x['err'], x['chi2'], x['dof'], x['echecs'])))

# ------------------------------------------------------------------ donnees synthetiques a energies connues
def synthetique(dossier, T=16, nops=5, nt=7, ncfg=40000, E=(0.9, 1.7, 2.6), graine=3, bruit=4e-4, doublon=True, vide=0.55, impulsion=False):
    """Ecrit un replica au format de `spectro neutre` : C_ij(tau) = sum_n V_in V_jn cosh(E_n (tau - T/2)) + <S_i><S_j> + bruit.
    Si doublon : l'operateur nops-1 est une copie quasi exacte de l'operateur 0 (base mal conditionnee a elaguer)."""
    rng = np.random.default_rng(graine)
    nm = len(E); tau = np.arange(nt)
    V = rng.uniform(0.4, 1.0, size=(nops, nm)) * 0.05 * np.array([1.0, 0.8, 0.5])[:nm]
    V[:, 1] *= rng.choice([-1, 1], size=nops); V[:, 2] *= rng.choice([-1, 1], size=nops)
    if doublon: V[nops - 1] = V[0] * (1 + 1e-4 * rng.normal(size=nm))
    paires = [(i, j) for i in range(nops) for j in range(i, nops)]
    exact = np.array([sum(V[i, n] * V[j, n] * np.cosh(E[n] * (tau - T / 2)) for n in range(nm)) for (i, j) in paires])
    c = np.full(nops, vide)
    sd = bruit * (1 + 0.5 * tau)
    C = exact[None] + np.array([c[i] * c[j] for (i, j) in paires])[None, :, None] * (0 if impulsion else 1) + rng.normal(size=(ncfg, len(paires), nt)) * sd
    M = c[None] + rng.normal(size=(ncfg, nops)) * 1e-3
    os.makedirs(dossier, exist_ok=True)
    formes = ['P11', 'R12', 'S22', 'CH1', 'CH2'][:nops]
    man = dict(alpha=0.5, beta=2.3, cotes=[8, 8, 8, T], formes=formes, niveaux_ape=[2], nops=nops, tau_max=nt - 1, temps=3,
               schema='kvu-spectro-neutre/1', statut='synthetique', E_vraies=list(E))
    json.dump(man, open(os.path.join(dossier, 'manifeste.json'), 'w'))
    if impulsion:
        C.astype('<f8').tofile(os.path.join(dossier, 'neutre_p1.bin'))
    else:
        np.concatenate([C.reshape(ncfg, -1), M], axis=1).astype('<f8').tofile(os.path.join(dossier, 'neutre.bin'))
    with open(os.path.join(dossier, 'plaq.csv'), 'w') as f:
        f.write('cfg,plaq\n'); f.write('\n'.join('%d,%.6f' % (i, 0.6 + 1e-4 * rng.normal()) for i in range(ncfg)) + '\n')
    return man

def cmd_autotest(a):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'syn'); man = synthetique(d)
        red = reduire(d, 0, 10)
        red['man'] = lire_manifeste(d)
        r = lecture(red, 0, faire_multi=False)
        imprimer(r)
        p = r['principale']['etat0']['plateau']
        assert p is not None and abs(p['valeur'] - 0.9) <= max(4 * p['erreur'], 0.01), 'E0 = 0,90 doit etre retrouvee'
        ix = r['elagage']['idx']
        assert ix is None or not (0 in ix and 4 in ix), 'les deux copies ne doivent pas rester ensemble'
        for o, dg in r['diagonaux'].items():     # borne rigoureuse : E_eff diagonal >= E0 (a 3 sigma pres)
            for t, v, e, ok in dg['E_eff'][1:3]:
                assert not ok or v + 3 * e >= 0.9, 'borne diagonale violee'
        print('AUTOTEST OK')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('reduire'); r.add_argument('dossier'); r.add_argument('sortie'); r.add_argument('--canal', default='0'); r.add_argument('--B', type=int, default=10)
    l = sp.add_parser('lecture'); l.add_argument('sources', nargs='+'); l.add_argument('--canal', default='0'); l.add_argument('--B', type=int, default=10)
    l.add_argument('--json'); l.add_argument('--nexp', type=int, default=2); l.add_argument('--tmaxb', type=int, default=5); l.add_argument('--sans-multi', action='store_true')
    sp.add_parser('autotest')
    a = ap.parse_args()
    if a.cmd == 'reduire':
        red = reduire(a.dossier, a.canal, a.B); sauver_reduit(red, a.sortie)
        print(a.dossier, 'canal', a.canal, 'ncfg', red['ncfg'], 'tau_int %.3f' % red['tau_int'], 'blocs', len(red['Cb']))
    elif a.cmd == 'lecture':
        reds = []
        for s in a.sources:
            reds.append(charger_reduit(s) if s.endswith('.npz') else reduire(s, a.canal, a.B))
            if reds[-1]['man'] is None: reds[-1]['man'] = lire_manifeste(resoudre(s, a.canal)[0])
        res = lecture(combiner(reds), a.canal, a.tmaxb, a.nexp, faire_multi=not a.sans_multi)
        imprimer(res)
        if a.json: json.dump(res, open(a.json, 'w'), indent=1, default=float)
    else:
        cmd_autotest(a)
