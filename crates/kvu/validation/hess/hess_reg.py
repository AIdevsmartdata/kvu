"""HESS-REG : `kvu hess hv` (gauche, base ym4) contre ym4_hess.Hessienne(C,(1,0)).hvp ; base s2 contre 1/2 x la référence ;
mutant « facteur 2 de la base s2 omis » => écart >= 0,1. Fonds, graines et seuils : GATES v4-T2 §1.3.
Avenant WP13-A1 (PREVOL.md) : la norme est ||D||_inf / (Schur(H_op) ||v||_inf) où H_op est l'opérateur COMPARÉ : Schur(H_ym4) en base ym4, Schur(H_s2) = Schur(H_ym4)/2 en base s2 ;
statistique PASS = max sur fonds et vecteurs ; mutant : min sur les fonds du max sur les vecteurs >= 0,1 (lecture stricte) ET diagnostic indépendant ||H_mut v - H_ref v||/||H_ref v|| >= 0,9."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

SORTIE = sys.argv[1] if len(sys.argv) > 1 else None
f = open(SORTIE, 'a') if SORTIE else None
H = O.charger_ym4()
sc = O.SCRATCH + '/hess_reg'; os.makedirs(sc, exist_ok=True)
H.set_threads(int(O.FILS))

def fond(nom):
    if nom == 'L8_haar':
        return 8, O.haar_c16(8, 26102613001)
    if nom == 'L8_wilson':
        d = sc + '/gen_L8_b2.3'
        if not os.path.isdir(d):
            O.kvu(['su2', 'gen', '--backend', 'cpu', '--L', 8, '--beta', 2.3, '--graine', 26102613002, '--chaines', 1, '--therm', 500,
                   '--nmeas', 1, '--skip', 1, '--dump-every', 1, '--outdir', d])
        import glob
        p = sorted(glob.glob(d + '/**/cfg_*.npy', recursive=True))
        assert p, 'aucun dump dans ' + d
        return 8, np.load(p[-1])
    if nom == 'L12_b2.3':
        return 12, np.load(O.GRANDL + '/L12_b2.3_c1/cfg_002000.npy')
    if nom == 'L12_b16':
        return 12, np.load(O.GRANDL + '/L12_b16_c1/cfg_002000.npy')
    if nom == 'L12_haar':
        return 12, O.haar_c16(12, 26102613005)

fonds = ['L8_haar', 'L8_wilson', 'L12_b2.3', 'L12_b16', 'L12_haar']
tout_ok = True
pire = 0.0; pire_s2 = 0.0; pire_mut = 1e9; pire_diag = 1e9; pire_ancien = 1e9
for nom in fonds:
    L, C = fond(nom)
    cfg = sc + '/%s.npy' % nom
    np.save(cfg, C)
    n = 12 * L ** 4
    rng = np.random.default_rng(26102613100)
    vs = rng.standard_normal((10, n))
    idx = rng.choice(n, size=12, replace=False)
    base = np.zeros((12, n)); base[np.arange(12), idx] = 1.0
    V = np.vstack([vs, base])
    op = H.Hessienne(C, (1.0, 0.0))
    schur = op.schur()
    ref = np.array([op.hvp(v) for v in V])
    np.save(sc + '/%s_V.npy' % nom, V)
    # base ym4
    O.kvu(['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/%s_V.npy' % nom, '--sortie', sc + '/%s_Y.npy' % nom, '--base', 'ym4', '--coord', 'gauche', '--fils', O.FILS])
    Y = np.load(sc + '/%s_Y.npy' % nom)
    norm = lambda D, sch: max(np.abs(D[i]).max() / (sch * np.abs(V[i]).max()) for i in range(len(V)))
    assert np.isfinite(Y).all() and np.isfinite(ref).all() and np.isfinite(schur) and schur > 0
    e = norm(Y - ref, schur)
    # base s2
    perm = O.perm_ym4_vers_s2(L)
    Vs = np.ascontiguousarray(V[:, perm])
    np.save(sc + '/%s_Vs2.npy' % nom, Vs)
    O.kvu(['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/%s_Vs2.npy' % nom, '--sortie', sc + '/%s_Ys2.npy' % nom, '--base', 's2', '--coord', 'gauche', '--fils', O.FILS])
    Ys = np.load(sc + '/%s_Ys2.npy' % nom)
    # H_s2 v_s2 = 1/2 P (H_ym4 v_ym4) : on compare dans la disposition s2 ; ordre des vecteurs identique, la norme sup est invariante par permutation
    assert np.isfinite(Ys).all()
    es = norm(Ys - 0.5 * ref[:, perm], 0.5 * schur)
    # mutant facteur 2 omis
    O.kvu(['hess', 'hv', '--config', cfg, '--vecteurs', sc + '/%s_Vs2.npy' % nom, '--sortie', sc + '/%s_Ym.npy' % nom, '--base', 's2', '--coord', 'gauche', '--mutant', 'facteur2', '--fils', O.FILS])
    Ym = np.load(sc + '/%s_Ym.npy' % nom)
    assert np.isfinite(Ym).all()
    em = norm(Ym - 0.5 * ref[:, perm], 0.5 * schur)          # lecture stricte, dénominateur Schur(H_s2)
    em_ancien = norm(Ym - 0.5 * ref[:, perm], schur)          # dénominateur Schur(H_ym4) (information : essais antérieurs)
    diag = min(np.linalg.norm(Ym[i] - 0.5 * ref[i, perm]) / np.linalg.norm(0.5 * ref[i, perm]) for i in range(len(V)))
    pire = max(pire, e); pire_s2 = max(pire_s2, es); pire_mut = min(pire_mut, em); pire_diag = min(pire_diag, diag); pire_ancien = min(pire_ancien, em_ancien)
    print('  fond %s L=%d schur=%.4f  ym4 %.3e  s2 %.3e  mutant(facteur2) %.3e (ancien denominateur %.3e, ratio relatif min %.3f)' % (nom, L, schur, e, es, em, em_ancien, diag), flush=True)
ok = pire <= 1e-12 and pire_s2 <= 1e-12
O.verdict(f, 'HESS-REG', ok, 'ym4 max=%.3e s2 max=%.3e (seuil 1e-12), 5 fonds, 22 vecteurs' % (pire, pire_s2))
okm = pire_mut >= 0.1 and pire_diag >= 0.9
O.verdict(f, 'HESS-REG-MUTANT-FACTEUR2', okm, 'min sur les fonds du max sur les vecteurs = %.3e (>= 0,1 exige : ROUGE ; avenant A1 : denominateur Schur(H_s2)) ; ratio relatif independant min = %.3f (>= 0,9) ; [information : avec le denominateur Schur(H_ym4) : %.3e]' % (pire_mut, pire_diag, pire_ancien))
sys.exit(0 if ok and okm else 1)
