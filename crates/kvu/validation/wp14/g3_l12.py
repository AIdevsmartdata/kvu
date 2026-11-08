"""WP14 : fond de jauge pure à L = 12 (réponse connue : ε′ = 0), pour la gate WP14-G3-L12.
U_{x,μ} = Ω_x Ω_{x+μ}† avec Ω de Haar (graine fixée), disposition kvu (μ, x0, x1, x2, x3, i, j) complexe128, lien μ de x vers x + ê_μ.
Usage (python3 -I -B) : g3_l12.py SORTIE.npy [GRAINE]   (défaut 2611011421 ; refuse d'écraser)"""
import os
import sys

import numpy as np

L = 12
GRAINE = 2611011421


def main():
    sortie = sys.argv[1]
    graine = int(sys.argv[2]) if len(sys.argv) > 2 else GRAINE
    if os.path.exists(sortie):
        print(f"g3_l12 : {sortie} existe (jamais écrasé)", file=sys.stderr)
        return 3
    rng = np.random.default_rng(graine)
    q = rng.normal(size=(L, L, L, L, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    om = np.zeros((L, L, L, L, 2, 2), complex)   # convention kvu/ym4 : [[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]]
    om[..., 0, 0] = q[..., 0] + 1j * q[..., 3]
    om[..., 0, 1] = q[..., 2] + 1j * q[..., 1]
    om[..., 1, 0] = -q[..., 2] + 1j * q[..., 1]
    om[..., 1, 1] = q[..., 0] - 1j * q[..., 3]
    u = np.zeros((4, L, L, L, L, 2, 2), complex)
    for mu in range(4):
        om_plus = np.roll(om, -1, axis=mu)   # Ω_{x+μ}
        u[mu] = om @ np.conj(np.swapaxes(om_plus, -1, -2))
    # contrôle : unitarité et plaquettes triviales
    assert np.abs(u @ np.conj(np.swapaxes(u, -1, -2)) - np.eye(2)).max() < 1e-13
    mu, nu = 0, 1
    p = u[mu] @ np.roll(u[nu], -1, axis=mu) @ np.conj(np.swapaxes(np.roll(u[mu], -1, axis=nu), -1, -2)) @ np.conj(np.swapaxes(u[nu], -1, -2))
    assert np.abs(p - np.eye(2)).max() < 1e-13, "plaquette de jauge pure non triviale"
    np.save(sortie, np.ascontiguousarray(u))
    print(f"g3_l12 : fond de jauge pure L = {L}, graine {graine}, écrit dans {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
