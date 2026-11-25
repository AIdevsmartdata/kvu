#!/usr/bin/env python3
"""lecture_comb.py etiquette=RACINE [etiquette=RACINE ...] [options]

Generalisation de lecture_comb_n8.py : K sources etiquelees quelconques, toute taille n
et toute liste de noeuds LUES DANS LES MANIFESTES (rien de codé en dur qui depende de n8),
controle de positivite fige d'avance.

Usage :
  python3 lecture_comb.py gamer=donnees/gamer_n8 radeon=donnees/radeon_n8
  python3 lecture_comb.py gamer=donnees/gamer_n8 radeon=donnees/radeon_n8 \
      --max-tranches gamer=83 radeon=75            # rejoue la lecture de 07h
  python3 lecture_comb.py g=.. r=.. c=..           # 3 sources : paires 2 a 2 + moyenne ponderee

Options :
  --max-tranches etiquette=N   n'utilise que les N premieres tranches (ordre T### croissant)
  --max-tranches=N             meme chose pour toutes les sources
  --pref=F                     facteur de normalisation de l'integrale (defaut 393216.0, regle n8)
  --seuil=F                    seuil c*4 de la regle beta* (defaut 0.0231257)
  --predictions=FICHIER.json   predictions a3(b)-a3(b_ref) pour une liste de noeuds hors table
                               interne (liste de meme longueur que les noeuds, ou dict b->valeur)
  --forme-globale              ajoute, APRES la sortie historique, l'estimateur SECONDAIRE a forme globale (fit 3 parametres c + a*exp(b*(beta-beta0))
                               des d par noeud, beta* et sigma par bootstrap, critere fige d'avance de l'etude de variance 19/11)
  --sans-controle              ne pas imprimer le bloc de controle (sortie historique seule)

Sortie : exactement celle de lecture_comb_n8.py (memes nombres, meme ordre), suivie du
bloc de controle (contexte lu dans les manifestes, controle de positivite par source et
pour la combinaison, annexes de rappel d'une lecture en source unique).

Taille, noeuds, volumes et nombre de repliques viennent de manifest_thermo.json
(champs argv/--paire-n, betas, cotes) ; les predictions du modele sont cherchees dans le
manifeste (champ `predictions`), sinon dans la table interne, sinon dans --predictions.
Pur python (pas de numpy) : c'est le seul moyen de retrouver les Flottants de
lecture_comb_n8.py au bit pres.
"""
import sys, csv, glob, os, math, json, re

PREF = 393216.0
SEUIL = 0.0231257                      # c*4, regle principale preregistree
A3REF_MODELE = 0.000457                # a3(2,275) du modele (PREDICTIONS.md) : variante de lecture_n8
SEUIL_SIGMA = 2.0                      # CONTROLE DE POSITIVITE, fige avant lecture : a3-a3(ref) < -2 sigma_appariee
N_REFFERENCES = 8                      # taille dont on connait les predictions du modele
PRED_N8 = [0.0, 0.00029, 0.00068, 0.00126, 0.00215, 0.00352, 0.00572, 0.00932, 0.015288, 0.02522]
NOEUDS_N8 = (2.275, 2.3, 2.3125, 2.325, 2.3375, 2.35, 2.3625, 2.375, 2.3875, 2.4)
PRED_TABLE = {(N_REFFERENCES, NOEUDS_N8): PRED_N8, (None, NOEUDS_N8): PRED_N8}
BETA_PRED_TABLE = {N_REFFERENCES: (2.397, 0.062)}   # n : (beta* predit, extrapolation err*sqrt(h))
Paire_Cle = ('gamer', 'radeon')        # ordre canonique de la paire (compatibilite lecture_comb_n8.py)


# ---------------------------------------------------------------- helpers
def fr(x):
    """ecriture francaise d'un nombre court, fidele a la sortie historique :
    2.275 -> '2,275' ; 2.4 -> '2,40' (deux decimales minimum) ; 0.000457 -> '0,000457'."""
    if abs(x) < 0.01:
        return ('%g' % x).replace('.', ',')
    s = f'{x:.4f}'.rstrip('0')
    entier, _, dec = s.partition('.')
    if len(dec) < 2:
        dec = (dec + '00')[:2] if abs(x) >= 1 else dec
    return (entier + '.' + dec).replace('.', ',')


def cle_naturelle(p):
    d = os.path.basename(p.rstrip('/'))
    m = re.match(r'^(.*?)(\d+)$', d)
    return (m.group(1), int(m.group(2))) if m else (d, 0)


def tri_tranches(manifestes):
    """Ordre numerique des tranches (T001, T002, ..., T1000) a partir des chemins .../T###/A/manifest_thermo.json."""
    return sorted(manifestes, key=lambda t: cle_naturelle(os.path.dirname(os.path.dirname(t))))


def lis(d, V):
    P = {}
    for f in sorted(glob.glob(d + '/points/*.csv')):
        for r in csv.DictReader(open(f)):
            P.setdefault(round(float(r['beta']), 4), []).append(float(r['tot']) / (6 * V))
    return P


def lis_manifest(chemin):
    """(noeuds, taille n, volume, nombre de repliques, predictions) declares par le manifeste."""
    try:
        m = json.load(open(chemin))
    except Exception as ex:
        raise SystemExit(f'lecture_comb : manifeste illisible {chemin} ({ex})')
    betas = tuple(round(float(b), 4) for b in m.get('betas', []))
    argv = m.get('argv') or []
    n = None
    if '--paire-n' in argv:
        try:
            n = int(argv[argv.index('--paire-n') + 1])
        except Exception:
            n = None
    if n is None and m.get('cotes'):
        n = int(m['cotes'][0])
    V = 1
    for c in m.get('cotes', []):
        V *= int(c)
    return betas, n, V, m.get('repliques'), m.get('predictions')


# ---------------------------------------------------------------- lecture d'une source
def source(racine, max_tranches=None):
    """-> dict(noeuds, d, e, nt, nr, n, VA, VB, temps_s, av, preds_manifest)"""
    av = []
    # CORRECTIF (20/11) : la cle de tri porte sur le NOM DE LA TRANCHE (T001, T002, ...), pas sur le nom du manifeste (sinon ordre du systeme de fichiers)
    tdir = tri_tranches(glob.glob(racine + '/T*/A/manifest_thermo.json'))
    if not tdir:
        raise SystemExit(f'lecture_comb : aucune tranche complete (A/manifest_thermo.json) dans {racine}')
    if max_tranches is not None:
        if max_tranches < 0:
            raise SystemExit(f'lecture_comb : --max-tranches negatif ({max_tranches})')
        if max_tranches < len(tdir):
            av.append(f'{len(tdir) - max_tranches} tranches ignorees au-dela de {max_tranches} (--max-tranches)')
        elif max_tranches > len(tdir):
            av.append(f'--max-tranches demande {max_tranches} tranches : {len(tdir)} seulement disponibles')
        tdir = tdir[:max_tranches]
    TA, TB = {}, {}
    nA = nB = VA = VB = nr_manif = None
    betas_manif = set()
    preds_manif = None
    temps = 0.0
    nt = 0
    for t in list(tdir):
        b = os.path.dirname(os.path.dirname(t))
        if not os.path.exists(b + '/B/manifest_thermo.json'):
            av.append(f'{os.path.basename(b)} : manifest B absent, tranche incomplete ignoree')
            tdir.remove(t)
            continue
        bo, nn, vv, rr, pp = lis_manifest(t)
        betas_manif.add(bo)
        if nA is None:
            nA, VA, nr_manif = nn, vv, rr
        elif (nA, VA) != (nn, vv):
            av.append(f'tailles/volumes incoherents dans {b}/A ({nA}/{VA} puis {nn}/{vv})')
        bb, nb, vb, rb, pb = lis_manifest(b + '/B/manifest_thermo.json')
        betas_manif.add(bb)
        if nB is None:
            nB, VB = nb, vb
        elif (nB, VB) != (nb, vb):
            av.append(f'tailles/volumes incoherents dans {b}/B ({nB}/{VB} puis {nb}/{vb})')
        if pp is not None:
            preds_manif = pp
        A = lis(b + '/A', VA); B = lis(b + '/B', VB)
        nt += 1
        rc = b + '.rc'
        if os.path.exists(rc):
            try:
                temps += float(open(rc).read().split()[1])
            except Exception:
                pass
        for be in A:
            TA.setdefault(be, []).extend(A[be])
            TB.setdefault(be, []).extend(B[be])
    if len(betas_manif) > 1:
        av.append(f'liste de noeuds non identique d\'une tranche a l\'autre : {len(betas_manif)} variantes')
    noeuds_manif = sorted(set().union(*betas_manif)) if betas_manif else sorted(TA)
    noeuds_data = sorted(TA)
    noeuds = [b for b in noeuds_manif if b in TA and b in TB]
    if len(noeuds) != len(noeuds_manif):
        av.append(f'{len(noeuds_manif) - len(noeuds)} noeud(s) annonce(s) par le manifeste sans donnees : '
                  + ' '.join(f'{b:.4f}' for b in noeuds_manif if b not in noeuds))
    if len(noeuds) != len(noeuds_data):
        av.append(f'{len(noeuds_data) - len(noeuds)} noeud(s) present(s) dans les points mais pas dans le manifeste : '
                  + ' '.join(f'{b:.4f}' for b in noeuds_data if b not in noeuds))
    if not noeuds:
        raise SystemExit('lecture_comb : aucun noeud exploitable')
    d, eapp, eind = [], [], []
    for be in noeuds:
        la, lb = TA[be], TB[be]
        if len(la) != len(lb):
            av.append(f'noeud {be:.4f} : {len(la)} valeurs en A, {len(lb)} en B (appariement tronque)')
        dd = [x - y for x, y in zip(la, lb)]
        n = len(dd)
        m = sum(dd) / n
        v = sum((x - m) ** 2 for x in dd)
        d.append(m)
        eapp.append(math.sqrt(v / (n - 1) / n))
        ma = sum(la) / len(la); mb = sum(lb) / len(lb)
        ea = math.sqrt(sum((x - ma) ** 2 for x in la) / (len(la) - 1) / len(la))
        eb = math.sqrt(sum((x - mb) ** 2 for x in lb) / (len(lb) - 1) / len(lb))
        eind.append(math.hypot(ea, eb))
    nr = len(TA[noeuds[0]])
    if any(len(TA[b]) != nr for b in noeuds):
        av.append('nombre de repliques different selon le noeud')
    carte = os.path.join(os.path.dirname(racine.rstrip('/')), f'n{nA}_carte_s.txt')
    if os.path.exists(carte):
        try:
            temps = float(open(carte).read())
            av.append(f'temps de carte pris de {carte}')
        except Exception:
            pass
    return dict(noeuds=noeuds, d=d, e=eapp, eind=eind, nt=nt, nr=nr, n=nA, VA=VA, VB=VB,
                temps_s=temps, av=av, preds_manifest=preds_manif, racine=racine)


# ---------------------------------------------------------------- integration, erreur, beta*
def integ(nodes, y, pref, lg=True):
    out = [0.0]; acc = 0.0
    for i in range(len(nodes) - 1):
        h = nodes[i + 1] - nodes[i]; a, c = y[i], y[i + 1]
        acc += h * (c - a) / math.log(c / a) if (lg and a > 0 and c > 0 and abs(c / a - 1) > 1e-9) else h * (a + c) / 2
        out.append(pref * acc)
    return out


def a3err(nodes, d, e, pref, lg=True):
    """a3-a3(premier noeud) et son erreur appariee propagee (Jacobian par perturbation de e*1e-3)."""
    a = integ(nodes, d, pref, lg); J = []
    for j in range(len(nodes)):
        yp = list(d); yp[j] += e[j] * 1e-3; ap = integ(nodes, yp, pref, lg)
        J.append([(ap[i] - a[i]) / 1e-3 for i in range(len(nodes))])
    return a, [math.sqrt(sum(J[j][i] ** 2 for j in range(len(nodes)))) for i in range(len(nodes))]


def croise(nodes, a, seuil):
    for i in range(1, len(a) - 1):
        if a[i] < seuil <= a[i + 1] and a[i] > 0:
            t = (math.log(seuil) - math.log(a[i])) / (math.log(a[i + 1]) - math.log(a[i]))
            return nodes[i] + t * (nodes[i + 1] - nodes[i])
    return None


# ---------------------------------------------------------------- predictions du modele
def prediction(srcs, fichier):
    """-> (liste ou None, description). Manifeste d'abord, table interne ensuite, fichier ensuite."""
    noeuds = srcs[0]['noeuds']; n = srcs[0]['n']
    for s in srcs:
        p = s['preds_manifest']
        if p is not None:
            if isinstance(p, dict):
                vals = [p.get(f'{b:g}', p.get(b)) for b in noeuds]
                if all(v is not None for v in vals):
                    return [float(v) for v in vals], 'champ predictions du manifeste'
            elif isinstance(p, (list, tuple)) and len(p) == len(noeuds):
                return [float(v) for v in p], 'champ predictions du manifeste'
            else:
                return None, 'champ predictions du manifeste inexploitable'
    if fichier:
        try:
            j = json.load(open(fichier))
        except Exception as ex:
            raise SystemExit(f'lecture_comb : --predictions illisible ({ex})')
        if isinstance(j, dict):
            vals = [j.get(f'{b:g}', j.get(b)) for b in noeuds]
            if all(v is not None for v in vals):
                return [float(v) for v in vals], fichier
            raise SystemExit('lecture_comb : --predictions ne couvre pas tous les noeuds')
        if len(j) == len(noeuds):
            return [float(v) for v in j], fichier
        raise SystemExit(f'lecture_comb : --predictions a {len(j)} valeurs pour {len(noeuds)} noeuds')
    cle = PRED_TABLE.get((n, tuple(noeuds))) or PRED_TABLE.get((None, tuple(noeuds)))
    if cle is not None:
        return list(cle), f'table interne (n={N_REFFERENCES}, predictions hachees)'
    return None, f'aucune (taille n={n}, {len(noeuds)} noeuds : passer --predictions)'


# ---------------------------------------------------------------- lignes de rapport
def rapport(nom, nodes, d, e, pred, seuil, pref):
    a, ea = a3err(nodes, d, e, pref)
    b = croise(nodes, a, seuil)
    print(f'-- {nom} : a3-a3({fr(nodes[0])}) par noeud (trapeze log) : '
          + ' '.join(f'{x:+.5f}' for x in a))
    zs = ' '.join(f'{(x - p) / s:+.1f}' if s > 0 and pred is not None else ' --'
                  for x, p, s in zip(a, pred or [None] * len(a), ea))
    print('   erreur appariee : ' + ' '.join(f'{x:.5f}' for x in ea) + f' ; (mes-pred)/err : {zs}')
    sans = 'pas de croisement dans [' + fr(nodes[0]) + ' ; ' + fr(nodes[-1]) + '] : beta* > ' + fr(nodes[-1])
    print('   beta* = ' + (str(b) if b else sans))
    return a, ea


def liste_z(a, ea):
    return [a[i] / ea[i] if ea[i] > 0 else None for i in range(len(a))]


# ---------------------------------------------------------------- lignes de controle
def controle(S, ordre, comb, pred, desc_pred, seuil, pref, maxt):
    n0 = S[ordre[0]]
    nodes = n0['noeuds']
    print('== BLOC DE CONTROLE ==')
    print('(a) contexte lu dans les manifestes')
    print('  taille n = ' + ' ; '.join(sorted({str(S[k]["n"]) for k in ordre}))
          + f' ; noeud de reference = {fr(nodes[0])} (premier noeud du manifeste) ; {len(nodes)} noeuds : '
          + ' '.join(f'{b:.4f}' for b in nodes))
    print('  volumes (produit des cotes) : ' + ' ; '.join(f'{k}: A={S[k]["VA"]} B={S[k]["VB"]}' for k in ordre))
    print(f'  pref = {pref} ; seuil c*4 = {seuil} ; predictions : {desc_pred}')
    if pred is not None:
        print('  prediction du modele (a3-a3(' + fr(nodes[0]) + ')) : '
              + ' '.join(f'{x:+.5f}' for x in pred))
    if maxt:
        print('  max-tranches appliques : ' + ' '.join(f'{k}={v}' for k, v in sorted(maxt.items())))
    for k in ordre:
        s = S[k]
        if s['av']:
            print(f'  {k} : ' + ' ; '.join(s['av']))
    print('(b) CONTROLE DE POSITIVITE (regle figee avant lecture : a3(b)-a3(' + fr(nodes[0])
          + ') < -' + f'{SEUIL_SIGMA:g}' + ' sigma_appariee ; le noeud de reference a sigma nul par construction, il est exclu)')
    alertes = []
    if comb is None and len(ordre) == 1:
        print('  combinaison : une seule source, la combinaison est identique a ' + ordre[0])
    for k in ordre + (['COMBINEE'] if comb else []):
        if k == 'COMBINEE':
            a, ea, nt, nom = comb['a'], comb['ea'], comb['nt'], 'combinaison (' + comb['lbl'] + ')'
        else:
            a, ea, nt, nom = S[k]['a'], S[k]['ea'], S[k]['nt'], k
        z = liste_z(a, ea)
        flags = [i for i in range(1, len(nodes)) if ea[i] > 0 and a[i] < -SEUIL_SIGMA * ea[i]]
        plus = ' ; '.join(f'{fr(nodes[i])} (a3={a[i]:+.5f}, sigma={ea[i]:.5f}, z={z[i]:+.2f})' for i in flags)
        mini = min((i for i in range(1, len(nodes)) if ea[i] > 0), key=lambda i: z[i] if ea[i] > 0 else 0.0)
        print(f'  {nom} : {nt} tranches ; z par noeud : '
              + ' '.join('--    ' if zz is None else f'{zz:+6.2f}' for zz in z))
        if flags:
            alertes.append((nom, flags))
            print(f'    ALERTE positivite : {len(flags)} noeud(s) sous -{SEUIL_SIGMA:g} sigma -> {plus}')
        else:
            print(f'    aucun noeud sous -{SEUIL_SIGMA:g} sigma (z le plus defavorable {z[mini]:+.2f} a {fr(nodes[mini])})')
    print('  VERDICT : ' + ('POSITIVITE NON MENACEE' if not alertes
                           else 'ALERTE : ' + ', '.join(f'{n} a {len(f)} noeud(s)' for n, f in alertes)))
    print('(c) rappel colonnes de lecture_n8 (source unique)')
    for k in ordre:
        s = S[k]
        aord, eaord = a3err(nodes, s['d'], s['e'], pref, lg=False)
        eind = s['eind']
        _, eindp = a3err(nodes, s['d'], eind, pref, lg=True)
        b1 = croise(nodes, s['a'], seuil - A3REF_MODELE)
        print(f'  {k} : d=tA-tB : ' + ' '.join(f'{x:+.3e}' for x in s['d']))
        print(f'  {k} : err independante (trapeze log) : ' + ' '.join(f'{x:.5f}' for x in eindp)
              + ' ; trapeze ordinaire : ' + ' '.join(f'{x:+.5f}' for x in aord)
              + ' ; err appariee (ord) : ' + ' '.join(f'{x:.5f}' for x in eaord))
        print(f'  {k} : beta* variante (a3({fr(nodes[0])})={fr(A3REF_MODELE)} du modele ajoute) : '
              + (f'{b1}' if b1 else 'pas de croisement'))
        h = s['temps_s'] / 3600
        extra = ''
        if s['n'] in BETA_PRED_TABLE:
            bp, ex = BETA_PRED_TABLE[s['n']]
            extra = f' ; beta* predit (modele) : {fr(bp)} ; extrapolation {fr(ex)}/sqrt(h)'
        print(f'  {k} : temps de carte (noyau) : {h:.2f} h ; erreur a3({fr(nodes[-1])})-a3({fr(nodes[0])}) = '
              f'{s["ea"][-1]:.5f} appariee ({eindp[-1]:.5f} independante) ;'
              f' x sqrt(h) = {s["ea"][-1] * math.sqrt(h):.4f}{extra}')
    tot = sum(S[k]['temps_s'] for k in ordre) / 3600
    print('  temps de carte cumule : ' + ' ; '.join(f'{k} {S[k]["temps_s"] / 3600:.2f} h' for k in ordre)
          + f' ; total {tot:.2f} h')



# ---------------------------------------------------------------- estimateur a forme globale (SECONDAIRE)
def _moindres_carres_exp(nodes, d, e, b):
    """d_j ~ c + a*exp(b*(beta_j - beta_0)) : (c, a) par moindres carres ponderes (1/e^2) a b fixe ; -> (chi2, c, a)."""
    x = [math.exp(b * (n - nodes[0])) for n in nodes]
    w = [1.0 / (s * s) for s in e]
    S = sum(w); Sx = sum(wi * xi for wi, xi in zip(w, x)); Sxx = sum(wi * xi * xi for wi, xi in zip(w, x))
    Sy = sum(wi * yi for wi, yi in zip(w, d)); Sxy = sum(wi * xi * yi for wi, xi, yi in zip(w, x, d))
    det = S * Sxx - Sx * Sx
    if abs(det) < 1e-300:
        return float('inf'), 0.0, 0.0
    c = (Sxx * Sy - Sx * Sxy) / det; a = (S * Sxy - Sx * Sy) / det
    chi = sum(wi * (yi - c - a * xi) ** 2 for wi, xi, yi in zip(w, x, d))
    return chi, c, a


def ajuste_forme_globale(nodes, d, e):
    """Ajustement a 3 parametres (c, a, b) de d(beta) = c + a*exp(b*(beta-beta_0)) : recherche en b (grille puis section doree),
    (c, a) lineaires. -> dict(c, a, b, chi2)."""
    grille = [-30 + 0.5 * i for i in range(0, 401)]           # b de -30 a +170
    vals = [_moindres_carres_exp(nodes, d, e, b)[0] for b in grille]
    i0 = min(range(len(grille)), key=lambda i: vals[i])
    lo, hi = grille[max(i0 - 1, 0)], grille[min(i0 + 1, len(grille) - 1)]
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = _moindres_carres_exp(nodes, d, e, x1)[0], _moindres_carres_exp(nodes, d, e, x2)[0]
    for _ in range(80):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = _moindres_carres_exp(nodes, d, e, x1)[0]
        else:
            lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = _moindres_carres_exp(nodes, d, e, x2)[0]
    b = (lo + hi) / 2
    chi, c, a = _moindres_carres_exp(nodes, d, e, b)
    return dict(c=c, a=a, b=b, chi2=chi)


def a3_forme(nodes, fit, pref, beta):
    """a3(beta) - a3(beta_0) = pref * integrale exacte de c + a*exp(b*(t-beta_0))."""
    c, a, b = fit['c'], fit['a'], fit['b']
    u = beta - nodes[0]
    return pref * (c * u + (a * (math.exp(b * u) - 1.0) / b if abs(b) > 1e-12 else a * u))


def croise_forme(nodes, fit, pref, seuil):
    """beta* : a3(beta) = seuil par bissection sur [beta_0, beta_max] (premiere traversee par le bas)."""
    lo, hi = nodes[0], nodes[-1]
    pas = (hi - lo) / 400
    prev = lo; fprev = a3_forme(nodes, fit, pref, lo) - seuil
    for i in range(1, 401):
        t = lo + i * pas; ft = a3_forme(nodes, fit, pref, t) - seuil
        if fprev < 0 <= ft:
            l, h = prev, t
            for _ in range(60):
                m = (l + h) / 2
                if a3_forme(nodes, fit, pref, m) - seuil < 0:
                    l = m
                else:
                    h = m
            return (l + h) / 2
        prev, fprev = t, ft
    return None


def forme_globale(nodes, d, e, pref, seuil, n_tirages=400, graine=1):
    """Bloc SECONDAIRE : ajustement global 3 parametres des d par noeud + beta*, bootstrap parametrique (memes tirages pour le trapeze).
    Non preregistre ; la regle principale (trapeze log nœud par nœud) reste le temoin. Criteres de l'etude de variance (fixes d'avance) :
    l'ecart-type de beta* du fit doit etre <= 0,85 x celui du trapeze ET l'ecart central <= 1 sigma, sinon garder le trapeze."""
    import random
    fit = ajuste_forme_globale(nodes, d, e)
    bf = croise_forme(nodes, fit, pref, seuil)
    a_t = integ(nodes, d, pref, True)
    bt = croise(nodes, [x - a_t[0] for x in a_t], seuil)
    rnd = random.Random(graine); BF, BT = [], []
    for _ in range(n_tirages):
        dp = [x + rnd.gauss(0, s) for x, s in zip(d, e)]
        f = ajuste_forme_globale(nodes, dp, e)
        x = croise_forme(nodes, f, pref, seuil)
        if x is not None:
            BF.append(x)
        a = integ(nodes, dp, pref, True)
        y = croise(nodes, [u - a[0] for u in a], seuil)
        if y is not None:
            BT.append(y)

    def sd(v):
        if len(v) < 11:
            return None
        m = sum(v) / len(v)
        return math.sqrt(sum((u - m) ** 2 for u in v) / (len(v) - 1))
    sf, st = sd(BF), sd(BT)
    ddl = len(nodes) - 3
    print('-- FORME GLOBALE (SECONDAIRE, non preregistre ; la regle du trapeze reste le temoin) : '
          'd(beta) = c + a*exp(b*(beta-beta_0)), 3 parametres, moindres carres ponderes')
    print(f'   c = {fit["c"]:+.4e} ; a = {fit["a"]:+.4e} ; b = {fit["b"]:.3f} ; chi2 = {fit["chi2"]:.1f} pour {ddl} ddl')
    print('   a3-a3(' + fr(nodes[0]) + ') par noeud (forme globale) : ' + ' '.join(f'{a3_forme(nodes, fit, pref, n):+.5f}' for n in nodes))
    print('   beta* (forme globale) = ' + (str(bf) if bf is not None else f'pas de croisement dans [{fr(nodes[0])} ; {fr(nodes[-1])}]'))
    print('   beta* (trapeze log, meme combinaison) = ' + (str(bt) if bt is not None else 'pas de croisement'))
    if sf is not None and st is not None:
        print(f'   sigma(beta*) par {n_tirages} tirages : forme globale {sf:.4f} ({100 * len(BF) / n_tirages:.0f} % croisent) ;'
              f' trapeze {st:.4f} ({100 * len(BT) / n_tirages:.0f} % croisent) ; rapport {sf / st:.2f}')
        if bf is not None and bt is not None:
            z = abs(bf - bt) / max(sf, st)
            ok = sf <= 0.85 * st and z <= 1.0
            print(f'   critere fixe d\'avance : rapport <= 0,85 ET ecart central <= 1 sigma ({z:.2f}) : '
                  + ('REMPLI (estimateur secondaire utilisable)' if ok else 'NON REMPLI : garder le trapeze'))
    else:
        print('   sigma(beta*) : trop peu de tirages croisent pour un ecart-type')


# ---------------------------------------------------------------- ligne de commande
def parse(argv):
    srcs, maxt = [], {}
    opts = dict(pref=PREF, seuil=SEUIL, predictions=None, controle=True, forme=False)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help', '--aide'):
            print(__doc__); raise SystemExit(0)
        elif a == '--forme-globale':
            opts['forme'] = True; i += 1
        elif a == '--sans-controle':
            opts['controle'] = False; i += 1
        elif a.startswith('--max-tranches='):
            for morceau in a.split('=', 1)[1].split(','):
                _max(maxt, morceau, srcs)
            i += 1
        elif a == '--max-tranches':
            i += 1
            while i < len(argv) and re.fullmatch(r'([A-Za-z0-9_.\-]+=\d+|\d+)([ ,;]+[A-Za-z0-9_.\-]+=(\d+|\*))*',
                                                 argv[i].strip()):
                for morceau in re.split(r'[ ,;]+', argv[i].strip()):
                    if morceau:
                        _max(maxt, morceau, srcs)
                i += 1
        elif a.startswith('--pref='):
            opts['pref'] = float(a.split('=', 1)[1]); i += 1
        elif a.startswith('--seuil='):
            opts['seuil'] = float(a.split('=', 1)[1]); i += 1
        elif a.startswith('--predictions='):
            opts['predictions'] = a.split('=', 1)[1]; i += 1
        elif a.startswith('-'):
            raise SystemExit(f'lecture_comb : option inconnue {a}')
        elif '=' in a:
            k, v = a.split('=', 1)
            if not k or not v:
                raise SystemExit(f'lecture_comb : source mal formee {a} (attendu etiquette=dossier)')
            srcs.append((k, v)); i += 1
        else:
            raise SystemExit(f'lecture_comb : argument mal forme {a} (attendu etiquette=dossier)')
    for k in maxt:
        if k != '__global__' and k not in [s[0] for s in srcs]:
            raise SystemExit(f'lecture_comb : --max-tranches cite {k}, source inconnue ({", ".join(s[0] for s in srcs)})')
    if not srcs:
        print(__doc__); raise SystemExit('lecture_comb : donnez au moins une source etiquette=dossier')
    return srcs, maxt, opts


def _max(maxt, morceau, srcs):
    if re.fullmatch(r'\d+', morceau):
        for k, _ in srcs:
            maxt[k] = int(morceau)
        maxt.setdefault('__global__', int(morceau))
    elif re.fullmatch(r'[A-Za-z0-9_.\-]+=\d+', morceau):
        k, v = morceau.split('=')
        maxt[k] = int(v)
    else:
        raise SystemExit(f'lecture_comb : --max-tranches mal forme {morceau} (attendu etiquette=N ou N)')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args, maxt, opts = parse(argv)
    if '__global__' in maxt:
        g = maxt.pop('__global__')
        for k, _ in args:
            maxt.setdefault(k, g)
    ordre = []
    S = {}
    for k, v in args:
        if k in S:
            raise SystemExit(f'lecture_comb : etiquette {k} donnee deux fois')
        S[k] = source(v, maxt.get(k))
        ordre.append(k)
    # predictions : liste commune (les sources doivent partager la liste de noeuds)
    for k in ordre[1:]:
        if S[k]['noeuds'] != S[ordre[0]]['noeuds']:
            raise SystemExit(f'lecture_comb : liste de noeuds differente entre {ordre[0]} et {k} '
                             f'({len(S[ordre[0]]["noeuds"])} vs {len(S[k]["noeuds"])} noeuds) : '
                             'combinaison impossible, filtrer les tranches')
    pred, desc_pred = prediction([S[k] for k in ordre], opts['predictions'])
    for k in ordre:
        print(f'{k} : {S[k]["nt"]} tranches, {S[k]["nr"]} repliques par noeud')
    for k in ordre:
        a, ea = rapport(k, S[k]['noeuds'], S[k]['d'], S[k]['e'], pred, opts['seuil'], opts['pref'])
        S[k]['a'], S[k]['ea'] = a, ea
    comb = None
    if len(ordre) == 2:
        k1, k2 = ordre
        if set(ordre) == set(Paire_Cle):
            k1, k2 = Paire_Cle
        n1, d1, e1 = S[k1]['noeuds'], S[k1]['d'], S[k1]['e']
        n2, d2, e2 = S[k2]['noeuds'], S[k2]['d'], S[k2]['e']
        assert n1 == n2
        print('noeud    d_' + k1 + '      d_' + k2 + '     diff/sigma')
        chi = 0.0; dc = []; ec = []
        for i, b in enumerate(n1):
            s = math.hypot(e1[i], e2[i]); z = (d1[i] - d2[i]) / s; chi += z * z
            w1, w2 = 1 / e1[i] ** 2, 1 / e2[i] ** 2
            dc.append((d1[i] * w1 + d2[i] * w2) / (w1 + w2)); ec.append((w1 + w2) ** -.5)
            print(f'{b:7.4f} {d1[i]:+.3e} {d2[i]:+.3e} {z:+.2f}')
        print(f'chi2 de compatibilite {k1}-{k2[0].upper() + k2[1:]} : {chi:.1f} pour {len(n1)} ddl '
              f'(attendu {len(n1)} +- {math.sqrt(2 * len(n1)):.1f})')
        comb = dict(d=dc, e=ec)
        comb['lbl'] = 'moyenne ponderee des d par noeud, deux sources homogenes separees : etiquette'
        comb['a'], comb['ea'] = rapport('COMBINEE (' + comb['lbl'] + ')', n1, dc, ec, pred, opts['seuil'], opts['pref'])
        comb['nt'] = f'{S[k1]["nt"]}+{S[k2]["nt"]}'
    elif len(ordre) > 2:
        dc = []; ec = []
        noeuds = S[ordre[0]]['noeuds']
        for a, b in [(ordre[i], ordre[j]) for i in range(len(ordre)) for j in range(i + 1, len(ordre))]:
            d1, e1, d2, e2 = S[a]['d'], S[a]['e'], S[b]['d'], S[b]['e']
            print('noeud    d_' + a + '      d_' + b + '     diff/sigma')
            for i, be in enumerate(noeuds):
                s = math.hypot(e1[i], e2[i]); z = (d1[i] - d2[i]) / s
                print(f'{be:7.4f} {d1[i]:+.3e} {d2[i]:+.3e} {z:+.2f}')
            print(f'chi2 de compatibilite {a}-{b[0].upper() + b[1:]} : '
                  f'{sum(((d1[i] - d2[i]) / math.hypot(e1[i], e2[i])) ** 2 for i in range(len(noeuds))):.1f} '
                  f'pour {len(noeuds)} ddl (attendu {len(noeuds)} +- {math.sqrt(2 * len(noeuds)):.1f})')
        for i in range(len(noeuds)):
            W = 0.0; acc = 0.0
            for k in ordre:
                w = 1 / S[k]['e'][i] ** 2; W += w; acc += S[k]['d'][i] * w
            dc.append(acc / W); ec.append(W ** -.5)
        chi = 0.0
        for i, be in enumerate(noeuds):
            for k in ordre:
                chi += ((S[k]['d'][i] - dc[i]) / S[k]['e'][i]) ** 2
        ddl = len(noeuds) * (len(ordre) - 1)
        print(f'chi2 d\'homogeneite des {len(ordre)} sources : {chi:.1f} pour {ddl} ddl '
              f'(attendu {ddl} +- {math.sqrt(2 * ddl):.1f})')
        comb = dict(d=dc, e=ec, lbl=f'moyenne ponderee des d par noeud, {len(ordre)} sources homogenes separees : etiquette',
                    nt='+'.join(str(S[k]['nt']) for k in ordre))
        comb['a'], comb['ea'] = rapport('COMBINEE (' + comb['lbl'] + ')', noeuds, dc, ec, pred, opts['seuil'], opts['pref'])
    if opts['controle']:
        controle(S, ordre, comb, pred, desc_pred, opts['seuil'], opts['pref'], maxt)
    if opts['forme']:
        src = comb if comb is not None else S[ordre[0]]
        forme_globale(S[ordre[0]]['noeuds'], src['d'], src['e'], opts['pref'], opts['seuil'])


if __name__ == '__main__':
    main()
