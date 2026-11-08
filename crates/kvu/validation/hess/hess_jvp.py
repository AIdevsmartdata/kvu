"""HESS-JVP : `kvu hess jvp` (nombres duaux) contre le Richardson de `kvu rg bloc` (carte complète), L = 6, 3 fonds, 20 directions.
f(h) = 2 vec(log(Ubar(h) Ubar(0)^+)) en coordonnées a ; différence centrée (f(h) - f(-h))/(2h), Richardson (h, h/2, h/4 ; h = 2,5e-4).
Perturbation : U_l -> exp(i (h zeta_l/2).sigma) U_l.  Critère : max ||Delta||_2 / ||J v||_2 <= 1e-8 ; h = 2,5e-4 (fixé au PREVOL).
Contrôle annexe : Ubar de `hess jvp --ubar-sortie` = Ubar de `rg bloc` (1e-13) ; J v de kvu contre l'implémentation analytique indépendante (balaban_np.py de WP11, information)."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outils as O

f = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
sc = O.SCRATCH + '/hess_jvp'; os.makedirs(sc, exist_ok=True)
L = 6; V = L ** 4; nl = 4 * V; n = 3 * nl
LD = np.longdouble

def c16_vers_quat(C):
    return np.stack([C[..., 0, 0].real, C[..., 0, 1].imag, C[..., 0, 1].real, C[..., 0, 0].imag], axis=-1)

def qmul(a, b):
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    out[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return out

def qexp(s):
    th = np.linalg.norm(s, axis=-1)
    fct = np.where(th > 1e-12, np.sin(th) / np.where(th > 0, th, 1), 1.0)
    return np.concatenate([np.cos(th)[..., None], fct[..., None] * s], axis=-1)

def qlog(q):
    v = q[..., 1:]; s = np.linalg.norm(v, axis=-1); th = np.arctan2(s, q[..., 0])
    fct = np.where(s > 1e-12, th / np.where(s > 0, s, 1), 1.0)
    return fct[..., None] * v

def fond(nom):
    if nom == 'haar':
        return O.haar_c16(L, 26102613061)
    beta, graine = (2.8, 26102613062) if nom == 'b2.8' else (8.0, 26102613063)
    d = sc + '/gen_%s' % nom
    if not os.path.isdir(d):
        O.kvu(['su2', 'gen', '--backend', 'cpu', '--L', L, '--beta', beta, '--graine', graine, '--chaines', 1, '--therm', 500, '--nmeas', 1, '--skip', 1, '--dump-every', 1, '--outdir', d])
    return np.load(sorted(glob.glob(d + '/**/cfg_*.npy', recursive=True))[-1])

def bloc(q_champ_ym4, nom):
    """q (4,L,L,L,L,4) -> Ubar quaternions (4,Lc,Lc,Lc,Lc,4) via `kvu rg bloc` (c16)."""
    C = O.quat_vers_c16(q_champ_ym4)
    O.ecrire_npy(sc + '/in_%s.npy' % nom, C)
    O.kvu(['rg', 'bloc', '--entree', sc + '/in_%s.npy' % nom, '--sortie', sc + '/out_%s.npy' % nom, '--fils', O.FILS])
    return c16_vers_quat(np.load(sc + '/out_%s.npy' % nom))

perm = O.perm_ym4_vers_s2(L)
Lc = L // 3; Vc = Lc ** 4
tout = True; pire = 0.0; pire_ub = 0.0
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reference', 'hess'))
import balaban_np as B
bl = B.Bloc(L)
for nom in ('haar', 'b2.8', 'b8'):
    C = fond(nom)
    q = c16_vers_quat(C)                      # (4,L,L,L,L,4), lien (mu, x) : ordre « champ »
    O.ecrire_npy(sc + '/cfg_%s.npy' % nom, C)
    rng = np.random.default_rng(26102613064 if nom == 'haar' else 26102613064 + (1 if nom == 'b2.8' else 2))
    Z = rng.standard_normal((20, n))          # disposition s2 : 3(4x+mu)+a
    O.ecrire_npy(sc + '/Z_%s.npy' % nom, Z)
    O.kvu(['hess', 'jvp', '--config', sc + '/cfg_%s.npy' % nom, '--vecteurs', sc + '/Z_%s.npy' % nom, '--sortie', sc + '/JZ_%s.npy' % nom, '--ubar-sortie', sc + '/ubar_%s.npy' % nom, '--fils', O.FILS])
    JZ = np.load(sc + '/JZ_%s.npy' % nom)     # (20, 3*4*Vc) disposition s2 grossière
    Ub0 = bloc(q, nom)
    # Ubar de kvu jvp vs rg bloc
    from_npy = np.load(sc + '/ubar_%s.npy' % nom)
    ub_kvu = from_npy if from_npy.shape[-1] == 4 else c16_vers_quat(from_npy)
    O.fin(ub_kvu, Ub0, JZ)
    pire_ub = max(pire_ub, np.abs(ub_kvu - Ub0).max())
    errs = []
    for k in range(20):
        zeta = Z[k].reshape(nl, 3)             # lien s2 l = 4x + mu
        # vers disposition champ (mu, x)
        zc = np.zeros((4, V, 3))
        for mu in range(4):
            zc[mu] = zeta[mu::4]
        def f_h(h):
            qq = qmul(qexp(0.5 * h * zc).reshape(4, L, L, L, L, 3 + 1), q)
            Ub = bloc(qq, nom)
            dag = Ub0 * np.array([1, -1, -1, -1])
            return 2 * qlog(qmul(Ub, dag))      # (4,Lc,Lc,Lc,Lc,3)
        def C(h):
            return (f_h(h) - f_h(-h)) / (2 * h)
        h = 2.5e-4          # fixé au PREVOL
        Ch, Ch2, Ch4 = C(h), C(h / 2), C(h / 4)
        R1 = (4 * Ch2 - Ch) / 3; R2 = (4 * Ch4 - Ch2) / 3
        fd = (16 * R2 - R1) / 15                # (4,Lc^4,3) ordre champ grossier
        # JZ en s2 grossier : 3(4z+mu)+a -> champ grossier
        j = JZ[k].reshape(4 * Vc, 3)
        jc = np.zeros((4, Vc, 3))
        for mu in range(4):
            jc[mu] = j[mu::4]
        fdc = fd.reshape(4, Vc, 3)
        errs.append(np.linalg.norm(fdc - jc) / np.linalg.norm(jc))
    assert np.isfinite(errs).all()
    e = max(errs); pire = max(pire, e)
    # supplément (non gelé, ID INFO-JVP-FORME-FERMEE) : DU.v dual contre la forme fermée de WP11 (balaban_np.py), 3 directions
    ea = 0.0
    for k in range(3):
        zeta = Z[k].reshape(nl, 3); zc = np.zeros((nl, 3))
        for mu in range(4):
            zc[mu * V:(mu + 1) * V] = zeta[mu::4]
        qf = q.reshape(nl, 4)
        out = np.zeros((4 * Vc, 3))
        for mu in range(4):
            for zs in range(Vc):
                zz = [(zs // Lc ** (3 - m)) % Lc for m in range(4)]
                liens, Jb, _ = bl.jac(qf, zz, mu)
                out[mu * Vc + zs] = sum(Jb[i] @ zc[l] for i, l in enumerate(liens))
        j = JZ[k].reshape(4 * Vc, 3); jc = np.zeros((4 * Vc, 3))
        for mu in range(4):
            jc[mu * Vc:(mu + 1) * Vc] = j[mu::4]
        ea = max(ea, np.linalg.norm(out - jc) / np.linalg.norm(jc))
    O.verdict(f, 'INFO-JVP-FORME-FERMEE-%s' % nom, ea <= 1e-10, 'DU.v dual contre la forme fermee de WP11 (balaban_np.py), 3 directions : %.3e (supplement, seuil 1e-10)' % ea)
    ok = e <= 1e-8
    tout &= ok
    O.verdict(f, 'HESS-JVP-%s' % nom, ok, 'max ||Richardson - dual||/||Jv|| = %.3e (seuil 1e-8), 20 directions ; Ubar dual - rg bloc = %.2e' % (e, np.abs(ub_kvu - Ub0).max()))
O.verdict(f, 'HESS-JVP', tout, 'L = 6, 3 fonds x 20 directions ; pire = %.3e ; pire |Ubar_dual - Ubar_rg| = %.2e (info)' % (pire, pire_ub))
sys.exit(0 if tout else 1)
