#!/usr/bin/env python3
"""Analyse des sorties de `kvu su2 spectro neutre` : soustraction du vide par échantillon jackknife, GEVP, E_eff (cosh),
règle de plateau N (figée dans PREDICTIONS_NEUTRE.md).  numpy seulement ; réutilise gevp/e_eff/jackknife de analyse_spectro.py.

  analyse_neutre.py mesure DOSSIER [DOSSIER ...] [--tau0 1] [--json f] [--sans-soustraction]
  analyse_neutre.py autotest
Règle N : fenêtres de DEUX points [ta, ta+1], ta >= tau0 ; première fenêtre telle que chi2 < 2 (dof 1) ET erreur relative de
la constante < 10 % ET la fenêtre suivante diffère de moins de 1 sigma de la différence ; sinon « pas de plateau ».
Bases emboîtées (ordre fixe) : B15 = tous les opérateurs ; B9 = formes P11,R12,S22 ; B6 = P11,R12 ; B3 = P11,R12,S22 au niveau médian.
Base principale = la première dont le GEVP est défini (C(tau0) défini positif) sur TOUS les échantillons jackknife.
"""
import sys, os, json, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_spectro import gevp, e_eff, jk_err, tau_int, ajuste_constante, matrice

def charger(dossier):
    man = json.load(open(os.path.join(dossier, 'manifeste.json')))
    nops = man['nops']; nt = man['tau_max'] + 1
    npair = nops * (nops + 1) // 2
    rl = npair * nt + nops
    a = np.fromfile(os.path.join(dossier, 'neutre.bin'), dtype='<f8')
    n = len(a) // rl
    a = a[:n * rl].reshape(n, rl)
    C = a[:, :npair * nt].reshape(n, npair, nt)
    M = a[:, npair * nt:]
    pl = np.loadtxt(os.path.join(dossier, 'plaq.csv'), delimiter=',', skiprows=1, ndmin=2)[:, 1]
    k = min(n, len(pl))
    return dict(C=C[:k], M=M[:k], plaq=pl[:k], man=man)

def noms_ops(man):
    f = man['formes']; return [(lv, nm) for lv in man['niveaux_ape'] for nm in f]

def bases(man):
    ops = noms_ops(man); nf = len(man['formes']); lv = man['niveaux_ape']
    idx = lambda pred: [i for i, (l, n) in enumerate(ops) if pred(l, n)]
    B = {'B15': list(range(len(ops))), 'B9': idx(lambda l, n: n in ('P11', 'R12', 'S22')),
         'B6': idx(lambda l, n: n in ('P11', 'R12')), 'B3': idx(lambda l, n: n in ('P11', 'R12', 'S22') and l == lv[len(lv) // 2])}
    return {k: v for k, v in B.items() if len(v) >= 1}

def taille_bloc(runs):
    tm = 0.5
    for r in runs:
        tm = max(tm, tau_int(r['plaq']))
        nops = r['M'].shape[1]
        pairs = [(a, b) for a in range(nops) for b in range(a, nops)]
        for o in range(nops):
            tm = max(tm, tau_int(r['C'][:, pairs.index((o, o)), 1]))
    return max(10, int(math.ceil(2 * tm))), tm

def blocs(runs, B):
    Cs, Ms = [], []
    for r in runs:
        nb = len(r['C']) // B
        if nb == 0: continue
        Cs.append(r['C'][:nb * B].reshape(nb, B, *r['C'].shape[1:]).mean(axis=1))
        Ms.append(r['M'][:nb * B].reshape(nb, B, -1).mean(axis=1))
    return np.concatenate(Cs), np.concatenate(Ms)

def echantillons(Cb, Mb, soustraire=True):
    """échantillons jackknife du corrélateur connexe [Nb, paires, tau]"""
    Nb = len(Cb)
    Cj = (Cb.sum(0)[None] - Cb) / (Nb - 1)
    Mj = (Mb.sum(0)[None] - Mb) / (Nb - 1)
    if not soustraire: return Cj
    nops = Mb.shape[1]
    out = Cj.copy()
    k = 0
    for a in range(nops):
        for b in range(a, nops):
            out[:, k, :] -= (Mj[:, a] * Mj[:, b])[:, None]
            k += 1
    return out

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

def analyser(Cj, nops, T, tau0, idx):
    paires = [(a, b) for a in range(nops) for b in range(a, nops)]
    Ccomp = Cj.mean(0)
    # échantillon complet (moyenne des échantillons jk = estimateur jk) ; GEVP doit être défini partout
    Ls = []
    for c in Cj:
        l = gevp(matrice(c, paires, nops), tau0, idx)
        if l is None: return dict(base=idx, erreur="C(tau0) non défini positif sur au moins un échantillon")
        Ls.append(l)
    Es = np.array([e_eff(l, T) for l in np.array(Ls)])
    out = dict(base=idx, tau0=tau0, nb_blocs=len(Cj))
    for n in range(min(2, len(idx))):
        em, ee = jk_err(Es[:, :, n])
        out[f'etat{n}'] = dict(E_eff=[(int(t), float(em[t]), float(ee[t])) for t in range(len(em))], plateau=plateau_N(Es[:, :, n], tau0))
    return out

def cmd_mesure(a):
    runs = [charger(d) for d in a.dossiers]
    man = runs[0]['man']; T = man['cotes'][man['temps']]; nops = man['nops']
    B, tint = taille_bloc(runs)
    Cb, Mb = blocs(runs, B)
    Cj = echantillons(Cb, Mb, not a.sans_soustraction)
    rap = dict(dossiers=a.dossiers, T=T, bloc=B, tau_int_max=tint, nb_blocs=len(Cb), ncfg=[len(r['C']) for r in runs], beta=man['beta'],
               cotes=man['cotes'], niveaux=man['niveaux_ape'], formes=man['formes'])
    print(f"cotes={man['cotes']} beta={man['beta']} T={T} bloc={B} tau_int={tint:.2f} blocs={len(Cb)} cfg={rap['ncfg']}")
    BB = bases(man); principal = None
    for nom, idx in BB.items():
        for tau0 in ([a.tau0] if not a.systematiques else [a.tau0, a.tau0 + 1]):
            r = analyser(Cj, nops, T, tau0, idx)
            rap[f'{nom}_tau0={tau0}'] = r
            if 'erreur' in r:
                print(nom, 'tau0', tau0, r['erreur']); continue
            for et in ('etat0', 'etat1'):
                if et in r:
                    print(f"[{nom} tau0={tau0}] {et} E_eff:", " ".join(f"{t}:{v:.4f}({e:.4f})" for t, v, e in r[et]['E_eff']), '| plateau', r[et]['plateau'])
            if principal is None and tau0 == a.tau0:
                principal = nom
    rap['base_principale'] = principal
    print('base principale :', principal)
    if a.json: json.dump(rap, open(a.json, 'w'), indent=1)

def cmd_autotest(a):
    """jeu synthétique : S = c + somme de modes cosh ; avec soustraction on retrouve E0, sans soustraction on ne la retrouve PAS."""
    rng = np.random.default_rng(7)
    T = 16; nops = 3; tm = 6; E = [0.90, 1.7, 2.6]; c = np.array([0.6, 0.55, 0.5])
    V = np.array([[1.0, 0.6, 0.2], [1.0, -0.3, 0.5], [1.0, -0.8, -0.4]]) * 0.05
    tau = np.arange(tm + 1)
    paires = [(i, j) for i in range(nops) for j in range(i, nops)]
    exact = np.array([sum(V[i, n] * V[j, n] * np.cosh(E[n] * (tau - T / 2)) for n in range(3)) for (i, j) in paires])
    ncfg = 60000
    # corrélateur « brut » = connexe + c_i c_j + bruit ; moyennes ⟨S⟩ = c + bruit
    bruit = rng.normal(size=(ncfg, len(paires), tm + 1)) * 4e-4 * (1 + 0.5 * tau)
    C = exact[None] + np.array([c[i] * c[j] for (i, j) in paires])[None, :, None] + bruit
    M = c[None] + rng.normal(size=(ncfg, nops)) * 1e-3
    run = dict(C=C, M=M, plaq=C[:, 0, 1])
    B, _ = taille_bloc([run]); Cb, Mb = blocs([run], B)
    res = {}
    for sous in (True, False):
        Cj = echantillons(Cb, Mb, sous)
        r = analyser(Cj, nops, T, 1, [0, 1, 2])
        res[sous] = r['etat0']['plateau']
        print('soustraction' if sous else 'SANS soustraction', res[sous])
    p = res[True]
    assert p is not None and abs(p['valeur'] - E[0]) <= max(4 * p['erreur'], 0.01 * E[0]), 'la masse vraie doit être retrouvée avec la soustraction'
    q = res[False]
    assert q is None or abs(q['valeur'] - E[0]) > 5 * max(q['erreur'], 1e-3), 'le mutant « vide oublié » doit être détecté'
    print('AUTOTEST OK (E0 vrai 0.90)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('mesure'); m.add_argument('dossiers', nargs='+'); m.add_argument('--tau0', type=int, default=1)
    m.add_argument('--json'); m.add_argument('--sans-soustraction', action='store_true'); m.add_argument('--systematiques', action='store_true')
    sp.add_parser('autotest')
    a = ap.parse_args()
    {'mesure': cmd_mesure, 'autotest': cmd_autotest}[a.cmd](a)
