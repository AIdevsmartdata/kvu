#!/usr/bin/env python3
"""Analyse des sorties de `kvu su2 spectro run` : jackknife par blocs, GEVP, énergie effective (cosh), plateau (règle D4 de
PREDICTIONS_SPECTRO.md, figée avant les données).  Dépend de numpy seulement.

  analyse_spectro.py torelon DOSSIER [DOSSIER ...] [--tau0 1] [--mom 0:0] [--sans-niveau0] [--json sortie.json]
  analyse_spectro.py fort    DOSSIER [DOSSIER ...] --beta B --L L        (test de couplage fort, observable ponctuel de niveau 0)
  analyse_spectro.py autotest                                            (jeu synthétique à spectre connu)
"""
import sys, json, math, argparse, os
import numpy as np

# ------------------------------------------------------------------ lecture
def lire_csv(chemin):
    with open(chemin) as f:
        ent = f.readline().strip().split(',')
        lignes = [l for l in f.read().split('\n') if l.strip()]
    # une dernière ligne tronquée (processus tué entre deux vidanges) est écartée ; jamais une ligne du milieu
    if lignes and (len(lignes[-1].split(',')) != len(ent) or not lignes[-1].rstrip().endswith(tuple('0123456789'))):
        lignes = lignes[:-1]
    d = np.array([[float(x) for x in l.split(',')] for l in lignes], ndmin=2)
    return ent, d

def charger(dossier, mom=(0, 0)):
    man = json.load(open(os.path.join(dossier, 'manifeste.json')))
    ent, d = lire_csv(os.path.join(dossier, 'corr.csv'))
    sel = (d[:, 1] == mom[0]) & (d[:, 2] == mom[1])
    d = d[sel]
    nn = len(man['niveaux_ape']); tau_max = man['tau_max']
    paires = [(a, b) for a in range(nn) for b in range(a, nn)]
    C = d[:, 3:].reshape(len(d), len(paires), tau_max + 1)
    plaq = lire_csv(os.path.join(dossier, 'plaq.csv'))[1][:, 1]
    n = min(len(C), len(plaq))
    return dict(C=C[:n], plaq=plaq[:n], man=man, paires=paires, nn=nn)

# ------------------------------------------------------------------ statistique
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

def taille_bloc(runs):
    tmax = 0.5
    for r in runs:
        tmax = max(tmax, tau_int(r['plaq']))
        for s in range(1, r['nn']):  # niveaux lissés
            k = r['paires'].index((s, s))
            tmax = max(tmax, tau_int(r['C'][:, k, 1]))
    return max(10, int(math.ceil(2 * tmax))), tmax

def blocs(runs, B):
    out = []
    for r in runs:
        nb = len(r['C']) // B
        if nb == 0:
            continue
        out.append(r['C'][:nb * B].reshape(nb, B, *r['C'].shape[1:]).mean(axis=1))
    return np.concatenate(out, axis=0)  # [Nb, paires, tau]

def jackknife_echantillons(Cb):
    Nb = len(Cb)
    tot = Cb.sum(axis=0)
    return (tot[None] - Cb) / (Nb - 1)  # Nb échantillons

def jk_err(x):
    """x : [Nb, ...] échantillons jackknife -> (moyenne jk, erreur)."""
    Nb = len(x)
    m = x.mean(axis=0)
    return m, np.sqrt((Nb - 1) / Nb * ((x - m) ** 2).sum(axis=0))

# ------------------------------------------------------------------ GEVP
def matrice(Cp, paires, nn):
    """Cp : [paires, tau] -> [tau, nn, nn] symétrique."""
    M = np.zeros((Cp.shape[1], nn, nn))
    for k, (a, b) in enumerate(paires):
        M[:, a, b] = Cp[k]
        M[:, b, a] = Cp[k]
    return M

def gevp(M, tau0, idx):
    """valeurs propres décroissantes de C(τ0)^{-1/2} C(τ) C(τ0)^{-1/2}, base idx ; [tau, n] ; None si C(τ0) non défini positif."""
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
    """lam [tau, n] -> E_eff [tau-1, n] par λ(τ)/λ(τ+1) = cosh(E(T/2−τ))/cosh(E(T/2−τ−1)) ; nan si impossible."""
    ntau, n = lam.shape
    E = np.full((ntau - 1, n), np.nan)
    for t in range(ntau - 1):
        a = T / 2 - t
        for k in range(n):
            r = lam[t, k] / lam[t + 1, k] if lam[t + 1, k] != 0 else np.nan
            if not np.isfinite(r) or r <= 1 or lam[t, k] <= 0 or lam[t + 1, k] <= 0:
                continue
            lo, hi = 1e-9, 40.0
            # CORRECTION 22/11 : la garde min(., 700) donnait NaN des que 40 (T/2 - tau - 1) >= 700 (T > 32).
            # cosh inchange (au bit) tant qu'il ne deborde pas ; forme sans debordement sinon.
            def f(e, a=a):
                if e * max(abs(a), abs(a - 1)) <= 700:
                    return math.cosh(e * a) / math.cosh(e * (a - 1))
                return math.exp(e * (abs(a) - abs(a - 1))) * (1 + math.exp(-2 * e * abs(a))) / (1 + math.exp(-2 * e * abs(a - 1)))
            if f(hi) < r:
                continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < r:
                    lo = mid
                else:
                    hi = mid
            E[t, k] = 0.5 * (lo + hi)
    return E

# ------------------------------------------------------------------ plateau (règle D4)
def ajuste_constante(Es, cov):
    """Es [Nb, w] échantillons jk, cov [w,w] (jackknife de l'échantillon complet) -> (mu_samples [Nb], chi2 complet, mu complet)."""
    Ci = np.linalg.inv(cov)
    one = np.ones(Es.shape[1])
    wgt = Ci @ one / (one @ Ci @ one)
    mu_s = Es @ wgt
    E0 = Es.mean(axis=0)
    mu = E0 @ wgt
    r = E0 - mu
    return mu_s, float(r @ Ci @ r), float(mu)

def plateau(Es, tau_first, taille=3):
    seuil = 3.0 if taille == 3 else float(taille - 1) + 1.0  # dof = taille-1 ; seuil 3 (dof 2), 2 (dof 1)
    """Es [Nb, ntau_eff] échantillons jk de E_eff(τ) pour τ = 0.. ; fenêtres [τa, τa+2], τa >= tau_first.
    Retourne dict (valeur, erreur, fenêtre, chi2) ou None."""
    Nb, nt = Es.shape
    res = {}
    for ta in range(tau_first, nt - taille + 1):
        W = Es[:, ta:ta + taille]
        if not np.all(np.isfinite(W)):
            continue
        Eb = W.mean(axis=0)
        cov = (Nb - 1) / Nb * (W - Eb).T @ (W - Eb)
        try:
            mu_s, chi2, mu = ajuste_constante(W, cov)
        except np.linalg.LinAlgError:
            continue
        res[ta] = (mu_s, chi2, mu)
    for ta in sorted(res):
        mu_s, chi2, mu = res[ta]
        if chi2 >= seuil:
            continue
        if ta + 1 in res:
            d = res[ta + 1][0] - mu_s
            dm, dsig = jk_err(d)
            if abs(res[ta + 1][2] - mu) >= dsig:
                continue
        else:
            continue
        m, e = jk_err(mu_s)
        return dict(valeur=float(mu), erreur=float(e), fenetre=[ta, ta + taille - 1], chi2=float(chi2))
    return None

# ------------------------------------------------------------------ analyse torelon
def analyser(Cb, paires, nn, T, tau0, niveaux_exclus=0, etat=0, taille=3, premier=None):
    idx = list(range(niveaux_exclus, nn))
    Cj = jackknife_echantillons(Cb)
    Ccomp = Cb.mean(axis=0)
    while True:
        lam_c = gevp(matrice(Ccomp, paires, nn), tau0, idx)
        if lam_c is not None or len(idx) <= 1:
            break
        idx = idx[1:]  # D2 : la base perd d'abord le niveau le moins lissé
    if lam_c is None:
        return None
    Ls = [gevp(matrice(c, paires, nn), tau0, idx) for c in Cj]
    if any(l is None for l in Ls):
        return dict(erreur="un échantillon jackknife n'est pas défini positif", base=idx)
    Ls = np.array(Ls)  # [Nb, tau, n]
    Es = np.array([e_eff(l, T) for l in Ls])  # [Nb, tau-1, n]
    Ecomp = e_eff(lam_c, T)
    out = dict(base=idx, tau0=tau0, nb_blocs=len(Cb))
    for n in range(min(2, len(idx))):
        em, ee = jk_err(Es[:, :, n])
        p = plateau(Es[:, :, n], tau0 + 1 if premier is None else premier, taille)
        out[f"etat{n}"] = dict(E_eff=[(int(t), float(em[t]), float(ee[t])) for t in range(len(em))], plateau=p)
    return out

def cmd_torelon(a):
    mom = tuple(int(x) for x in a.mom.split(':'))
    runs = [charger(d, mom) for d in a.dossiers]
    T = runs[0]['man']['cotes'][runs[0]['man']['temps']]
    B, tint = taille_bloc(runs)
    Cb = blocs(runs, B)
    paires, nn = runs[0]['paires'], runs[0]['nn']
    rap = dict(dossiers=a.dossiers, T=T, bloc=B, tau_int_max=tint, nb_blocs=len(Cb), ncfg=[len(r['C']) for r in runs],
               niveaux=runs[0]['man']['niveaux_ape'], beta=runs[0]['man']['beta'], cotes=runs[0]['man']['cotes'])
    rap['principal'] = analyser(Cb, paires, nn, T, a.tau0, 1 if a.sans_niveau0 else 0, taille=a.taille, premier=a.premier)
    if a.systematiques:
        rap['tau0=2'] = analyser(Cb, paires, nn, T, 2, 1 if a.sans_niveau0 else 0)
        rap['sans_niveau0'] = analyser(Cb, paires, nn, T, a.tau0, 1)
        rap['sans_niveau0_tau0=2'] = analyser(Cb, paires, nn, T, 2, 1)
    # plancher constant (partie déconnectée) : C_ss(T/2) du dernier niveau et ⟨φ⟩
    kd = paires.index((nn - 1, nn - 1))
    m, e = jk_err(jackknife_echantillons(Cb)[:, kd, :])
    rap['C_dernier_niveau'] = [(t, float(m[t]), float(e[t])) for t in range(len(m))]
    imprimer(rap)
    if a.json:
        json.dump(rap, open(a.json, 'w'), indent=1)

def imprimer(rap):
    print(f"T={rap['T']} bloc={rap['bloc']} (tau_int max {rap['tau_int_max']:.2f}) blocs={rap['nb_blocs']} cfg={rap['ncfg']}")
    for k in ('principal', 'tau0=2', 'sans_niveau0', 'sans_niveau0_tau0=2'):
        r = rap.get(k)
        if r is None:
            continue
        print(f"[{k}]", {x: r[x] for x in ('base', 'tau0') if x in r}, r.get('erreur', ''))
        for et in ('etat0', 'etat1'):
            if et in r:
                print("  ", et, "E_eff:", " ".join(f"{t}:{v:.4f}({e:.4f})" for t, v, e in r[et]['E_eff']))
                print("  ", et, "plateau:", r[et]['plateau'])

# ------------------------------------------------------------------ couplage fort
def bessel_I(n, x):
    return sum((x / 2) ** (2 * k + n) / (math.factorial(k) * math.factorial(k + n)) for k in range(60))

def cmd_fort(a):
    u = bessel_I(2, a.beta) / bessel_I(1, a.beta)
    sorties = {}
    data = []
    for d in a.dossiers:
        ent, v = lire_csv(os.path.join(d, 'ponc.csv'))
        man = json.load(open(os.path.join(d, 'manifeste.json')))
        assert man['niveaux_ape'] == [0]
        ncfg = int(v[:, 0].max()) + 1
        data.append(v[:, 3].reshape(ncfg, -1))  # [cfg, tau]
    ok = True
    for tau in (1, 2):
        # blocs : 50 blocs par run
        bl = []
        for x in data:
            nb = 50 // len(data)
            b = len(x) // nb
            bl.append(x[:nb * b, tau].reshape(nb, b).mean(axis=1))
        bl = np.concatenate(bl)
        R = 4 * bl / u ** (a.L * tau)
        Rm = R.mean(); Rs = R.std(ddof=1) / math.sqrt(len(R))
        dth = 16 * (tau + 1) * u ** (2 * a.L)
        passe = abs(Rm - 1) <= 3 * Rs + dth
        ok &= passe
        print(f"beta={a.beta} L={a.L} tau={tau} u={u:.5f} R={Rm:.5f} +- {Rs:.5f}  delta_th={dth:.5f}  |R-1|={abs(Rm-1):.5f} -> {'PASSE' if passe else 'ECHEC'}")
    print("verdict", "PASSE" if ok else "ECHEC")

# ------------------------------------------------------------------ autotest
def cmd_autotest(a):
    rng = np.random.default_rng(12345)
    T = 32; nn = 4; tau_max = 10
    E = [0.40, 0.90, 1.4, 2.0]
    V = np.array([[1.0, 0.5, 0.2, 0.1], [1.0, 0.0, -0.4, 0.3], [1.0, -0.6, 0.3, -0.2], [1.0, -1.0, -0.5, 0.0]]) * 0.3
    paires = [(i, j) for i in range(nn) for j in range(i, nn)]
    exact = np.zeros((len(paires), tau_max + 1))
    for k, (i, j) in enumerate(paires):
        for n in range(nn):
            exact[k] += V[i, n] * V[j, n] * np.cosh(E[n] * (np.arange(tau_max + 1) - T / 2))
    ncfg = 4000
    noise = rng.normal(size=(ncfg, len(paires), tau_max + 1)) * 0.02 * (1 + 0.15 * np.arange(tau_max + 1))
    C = exact[None] * 0.0 + exact[None] + noise
    run = dict(C=C, plaq=C[:, 0, 1], man=dict(cotes=[8, 16, 16, T], temps=3, niveaux_ape=list(range(nn)), tau_max=tau_max, beta=0), paires=paires, nn=nn)
    B, tint = taille_bloc([run])
    Cb = blocs([run], B)
    r = analyser(Cb, paires, nn, T, 1)
    p = r['etat0']['plateau']
    print("bloc", B, "tau_int", tint, "blocs", len(Cb), "plateau", p)
    assert p is not None, "pas de plateau trouvé sur le jeu synthétique"
    assert abs(p['valeur'] - E[0]) <= max(4 * p['erreur'], 0.002 * E[0]), f"E0 {p['valeur']} != {E[0]}"
    print("AUTOTEST OK : E0 =", p['valeur'], "+-", p['erreur'], "(vrai 0.40)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('torelon'); t.add_argument('dossiers', nargs='+'); t.add_argument('--tau0', type=int, default=1)
    t.add_argument('--mom', default='0:0'); t.add_argument('--sans-niveau0', action='store_true'); t.add_argument('--json')
    t.add_argument('--systematiques', action='store_true'); t.add_argument('--taille', type=int, default=3); t.add_argument('--premier', type=int, default=None)
    f = sp.add_parser('fort'); f.add_argument('dossiers', nargs='+'); f.add_argument('--beta', type=float, required=True); f.add_argument('--L', type=int, required=True)
    sp.add_parser('autotest')
    a = ap.parse_args()
    {'torelon': cmd_torelon, 'fort': cmd_fort, 'autotest': cmd_autotest}[a.cmd](a)
