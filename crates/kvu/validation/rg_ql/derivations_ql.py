#!/usr/bin/env python3
"""Calculs qui portent les affirmations de docs/BLOCAGE_QUASI_LOCAL.md (WP22). Aucune donnée de configuration.

Usage : /usr/bin/python3 -I derivations_ql.py   (SymPy + numpy ; RC 0 si toutes les vérifications passent)

D1. Le schéma RK3 de Lüscher (arXiv:1006.4518, éq. (C.2)-(C.3)), appliqué à une équation LINÉAIRE abélienne y' = λ y
    (Z_i = ε λ W_i, exponentielles commutatives), a pour facteur d'amplification R(z) = 1 + z + z²/2 + z³/6, z = ελ.
D2. Intervalle réel de stabilité : |R(z)| ≤ 1 pour z ∈ [z*, 0], z* racine réelle de R(z) = -1 ; on rend z*.
    Le laplacien de réseau en 4D a pour spectre p̂² ∈ [0, 16] ⇒ stabilité linéaire du flot ssi ε ≤ -z*/16.
D3. Linéarisation abélienne d'un pas APE (α) : θ'_μ = θ_μ + (α/6) Σ_ν (∂_ν ∇_ν θ_μ − ∂_μ ∇_ν θ_ν)
    (∂ avant, ∇ arrière) : vérifiée symboliquement en développant les 6 agrafes au premier ordre.
D4. Sur une onde plane transverse θ_μ(x) = a cos(p x_ρ) (ρ ≠ μ), ce pas multiplie l'amplitude par 1 − (α/6) p̂²,
    p̂² = 4 sin²(p/2) ; le flot linéarisé (Z = −Ta(U·A), A = agrafes fermantes) la multiplie par e^{−t p̂²}.
D5. Second moment du noyau de chaleur de réseau en 4D : Σ_z |z|² K_t(z) = 8 t EXACTEMENT (K_t = e^{tΔ}) ;
    n pas d'Euler de pas ε du même laplacien : second moment 8 n ε ; donc rayon quadratique moyen √(8 t_APE),
    t_APE = α n / 6, pour la partie transverse linéarisée de n pas APE (noyau positif ssi 1 − 8ε ≥ 0, soit α ≤ 3/4).
"""
import sys

import numpy as np
import sympy as sp

ok = True


def verifier(cond, msg):
    global ok
    print(("OK   " if cond else "ECHEC ") + msg)
    ok = ok and bool(cond)


# ---------------------------------------------------------------- D1
z = sp.symbols("z")
w0 = sp.Integer(1)
w1 = w0 + sp.Rational(1, 4) * z * w0
w2 = w1 + sp.Rational(8, 9) * z * w1 - sp.Rational(17, 36) * z * w0
w3 = w2 + sp.Rational(3, 4) * z * w2 - sp.Rational(8, 9) * z * w1 + sp.Rational(17, 36) * z * w0
R = sp.expand(w3)
verifier(sp.simplify(R - (1 + z + z**2 / 2 + z**3 / 6)) == 0, f"D1 R(z) = {R}")

# ---------------------------------------------------------------- D2
racines = np.roots([1 / 6, 1 / 2, 1, 2])  # R(z) + 1 = z³/6 + z²/2 + z + 2
reelles = sorted(r.real for r in racines if abs(r.imag) < 1e-12)
zstar = reelles[0]
verifier(len(reelles) == 1 and -2.52 < zstar < -2.50, f"D2 racine réelle de R(z) = -1 : z* = {zstar:.6f}")
zz = np.linspace(zstar, 0, 20001)
Rz = 1 + zz + zz**2 / 2 + zz**3 / 6
verifier(np.max(np.abs(Rz)) <= 1 + 1e-12, f"D2 max |R| sur [z*, 0] = {np.max(np.abs(Rz)):.15f}")
eps_max = -zstar / 16
verifier(abs(eps_max - 0.15704) < 1e-4, f"D2 ε_max = -z*/16 = {eps_max:.6f} (p̂² ≤ 16 en 4D)")

# ---------------------------------------------------------------- D3
# Champ abélien générique : θ_{ν}(x + k) symboles indépendants ; on vérifie l'identité de l'opérateur en un site.
h = sp.symbols("h")
cache = {}


def th(nu, dx):
    """θ_ν(x + dx), dx tuple de 4 entiers."""
    cle = (nu, tuple(dx))
    if cle not in cache:
        cache[cle] = sp.Symbol(f"t{nu}_{'_'.join(str(d) for d in dx)}")
    return cache[cle]


def e(m):
    v = [0, 0, 0, 0]
    v[m] = 1
    return v


def add(*vs):
    return [sum(c) for c in zip(*vs)]


def neg(v):
    return [-c for c in v]


alpha = sp.symbols("alpha", positive=True)
ecart_max = 0
for mu in range(4):
    o = [0, 0, 0, 0]
    # angles des 6 agrafes orientées de x à x+μ (abélien : la phase s'additionne)
    som = 0
    for nu in range(4):
        if nu == mu:
            continue
        a = th(nu, o) + th(mu, e(nu)) - th(nu, e(mu))  # U_ν(x) U_μ(x+ν) U_ν(x+μ)†
        b = -th(nu, neg(e(nu))) + th(mu, neg(e(nu))) + th(nu, add(neg(e(nu)), e(mu)))  # U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ)
        som += a + b
    # au premier ordre en h (θ → hθ) : s = (1−α) e^{ihθ} + (α/6) Σ e^{ihφ} ⇒ arg s = h[(1−α)θ + (α/6)Σφ] + O(h³)
    lin = (1 - alpha) * th(mu, o) + alpha / 6 * som
    # opérateur annoncé : θ + (α/6) Σ_ν (∂_ν∇_ν θ_μ − ∂_μ∇_ν θ_ν)
    lap = 0
    for nu in range(4):
        lap += th(mu, e(nu)) - 2 * th(mu, o) + th(mu, neg(e(nu)))  # ∂_ν∇_ν θ_μ
        # ∂_μ ∇_ν θ_ν(x) = θ_ν(x+μ) − θ_ν(x+μ−ν) − θ_ν(x) + θ_ν(x−ν)
        lap -= th(nu, e(mu)) - th(nu, add(e(mu), neg(e(nu)))) - th(nu, o) + th(nu, neg(e(nu)))
    annonce = th(mu, o) + alpha / 6 * lap
    ecart_max = max(ecart_max, len(sp.expand(lin - annonce).free_symbols))
verifier(ecart_max == 0, "D3 pas APE linéarisé = θ + (α/6) Σ_ν (∂_ν∇_ν θ_μ − ∂_μ∇_ν θ_ν) (4 directions μ)")
# phase exacte au premier ordre : arg((1−α)e^{ihθ} + (α/6)Σ e^{ihφ_k}) — dérivée en h=0, cas numérique aléatoire
rng = np.random.default_rng(22)
for _ in range(5):
    al = rng.uniform(0.05, 0.95)
    th0, ph = rng.normal(), rng.normal(size=6)
    f = lambda hh: np.angle((1 - al) * np.exp(1j * hh * th0) + al / 6 * np.exp(1j * hh * ph).sum())
    d = (f(1e-5) - f(-1e-5)) / 2e-5
    verifier(abs(d - ((1 - al) * th0 + al / 6 * ph.sum())) < 1e-8, f"D3 dérivée de la phase projetée (α = {al:.3f}) : écart {abs(d - ((1 - al) * th0 + al / 6 * ph.sum())):.1e}")

# ---------------------------------------------------------------- D4
p, x = sp.symbols("p x", real=True)
onde = sp.cos(p * x)
lap1d = onde.subs(x, x + 1) - 2 * onde + onde.subs(x, x - 1)
facteur = sp.simplify(sp.expand_trig(lap1d) / onde)
verifier(sp.simplify(facteur + 4 * sp.sin(p / 2) ** 2) == 0, f"D4 Δ cos(px) = −p̂² cos(px), p̂² = 4 sin²(p/2) (facteur {facteur})")

# ---------------------------------------------------------------- D5
t = sp.symbols("t", positive=True)
q = sp.symbols("q0:4", real=True)
phat2 = sum(4 * sp.sin(qi / 2) ** 2 for qi in q)
Kt = sp.exp(-t * phat2)  # transformée de Fourier du noyau de chaleur de réseau
moment2 = -sum(sp.diff(Kt, qi, 2) for qi in q)
m2 = sp.simplify(moment2.subs({qi: 0 for qi in q}))
verifier(sp.simplify(m2 - 8 * t) == 0, f"D5 Σ |z|² K_t(z) = {m2} (4D)")
eps_, n_ = sp.symbols("epsilon n", positive=True)
Ke = (1 - eps_ * phat2) ** n_
m2e = sp.simplify((-sum(sp.diff(Ke, qi, 2) for qi in q)).subs({qi: 0 for qi in q}))
verifier(sp.simplify(m2e - 8 * n_ * eps_) == 0, f"D5 n pas d'Euler : second moment {m2e}")
# poids central du noyau d'un pas d'Euler 1 + εΔ (8 voisins à ε) : 1 − 8ε ; ε = α/6 ⇒ positif ssi α ≤ 3/4
verifier(sp.solve(sp.Eq(1 - 8 * alpha / 6, 0), alpha) == [sp.Rational(3, 4)], "D5 noyau positif ssi α ≤ 3/4")
# rayons des réglages de BLOC2 (α = 0,5) : n = 6, 13, 40
for n in (6, 13, 14, 40):
    r = float(sp.sqrt(sp.Rational(8, 6) * sp.Rational(1, 2) * n))
    print(f"INFO rayon APE(0,5 ; {n}) = √(4αn/3) = {r:.4f} ; dans la famille (4αn ≤ 27) : {4 * 0.5 * n <= 27}")
print(f"INFO rayon flot : √(8t) ≤ 3 ⇔ t ≤ {sp.Rational(9, 8)}")

print("STATUT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
