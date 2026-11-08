#!/usr/bin/env python3
"""G-H04 du kevinotron unifié : test EXACT du tirage de x0 (bain thermique SU(2)) en F64.

Adaptation de reference/su2_cuda/test_heatbath_cuda.py (gate P1 de gpu_port). Même densité cible
p(x) ∝ sqrt(1-x²) exp(αx) sur [-1,1], mêmes moments exacts (quadrature de Gauss-Tchebychev de 2e espèce,
n = 4000), même seuil GELÉ avant exécution : |z| < 4 pour m1 et m2 à chaque α, plafond = 0, N = 2·10^7,
7 valeurs de α (Creutz α < 2, Kennedy-Pendleton α ≥ 2). Ajouts : le tirage vient de `su2 test-hb` du kevinotron
unifié (HIP ou CUDA selon la machine) et un MUTANT (Metropolis naïf, K1_MUTANT_METRO) doit être ROUGE à α ≥ 5.

Usage (SOUS le verrou GPU de la machine, fenêtre ≤ 10 min) :
    KVU_SU2_CMD="<binaire de test su2_cli>" python3 test_heatbath_unifie.py [--backend hip|cuda|cpu]
Par défaut KVU_SU2_CMD désigne le binaire de test `su2_cli` (passerelle `shim_cli`) ; quand le binaire `kvu` sera
câblé (T9), poser KVU_SU2_CMD="kvu su2" (les arguments sont alors passés tels quels après la commande).
La gate Rust équivalente est `su2_gpu::g_h04` (mêmes seuils, même mutant).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

ALPHAS = [0.3, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0]
N = 20_000_000
N_MUTANT = 4_000_000
SEUIL_Z = 4.0
SEP = "\x1f"


def exact(alpha, n=4000):
    k = np.arange(1, n + 1)
    x = np.cos(k * np.pi / (n + 1))
    w = np.pi / (n + 1) * np.sin(k * np.pi / (n + 1)) ** 2
    f = np.exp(alpha * (x - 1))
    z = np.sum(w * f)
    return [np.sum(w * f * x ** p) / z for p in (1, 2, 3, 4)]


def lancer(cmd, args, env_extra=None):
    """Exécute `su2 <args>` ; cmd est une liste (binaire de test + passerelle, ou ["kvu", "su2"])."""
    env = dict(os.environ)
    if len(cmd) == 1:  # binaire de test : passerelle shim_cli
        env["KVU_SU2_ARGV"] = SEP.join(args)
        full = [cmd[0], "--ignored", "--exact", "shim_cli", "--nocapture", "--test-threads=1"]
    else:
        full = cmd + args
    if env_extra:
        env.update(env_extra)
    return subprocess.run(full, capture_output=True, text=True, env=env)


def testhb(cmd, backend, alpha, n, graine, mutant=False):
    args = ["test-hb", repr(alpha), str(n), "--graine", str(graine), "--precision", "f64", "--backend", backend,
            "--options", "ref-k1"]
    if mutant:
        args += ["--mutant", "metro"]
    out = lancer(cmd, args)
    # libtest préfixe la première ligne par « test shim_cli ... » : on cherche TESTHB n'importe où
    lignes = [l[l.index("TESTHB"):] for l in out.stdout.splitlines() if "TESTHB" in l]
    if not lignes:
        print("ECHEC exécution", alpha, out.returncode, out.stderr[-500:])
        sys.exit(1)
    kv = dict(t.split("=") for t in lignes[0].split()[1:])
    return out.returncode, kv


def zscores(kv, alpha, n):
    m1, m2 = float(kv["m1"]), float(kv["m2"])
    e1, e2, e3, e4 = exact(alpha)
    sd1, sd2 = np.sqrt(e2 - e1 ** 2), np.sqrt(e4 - e2 ** 2)
    return (m1 - e1) / (sd1 / np.sqrt(n)), (m2 - e2) / (sd2 / np.sqrt(n)), e1, e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--json", default=None)
    ap.add_argument("--n", type=int, default=N, help="nombre de tirages (2e7 = seuil gelé ; réduire pour un essai)")
    a = ap.parse_args()
    n_tir = a.n
    cmd = shlex.split(os.environ.get("KVU_SU2_CMD", ""))
    if not cmd:
        print("KVU_SU2_CMD absent : indiquer le binaire de test su2_cli ou « kvu su2 »")
        sys.exit(2)
    res = {"n": n_tir, "seuil_z": SEUIL_Z, "alphas": {}, "mutant": {}}
    ok_all = True
    for i, alpha in enumerate(ALPHAS):
        rc, kv = testhb(cmd, a.backend, alpha, n_tir, 9000 + i)
        z1, z2, e1, e2 = zscores(kv, alpha, n_tir)
        ok = abs(z1) < SEUIL_Z and abs(z2) < SEUIL_Z and int(kv["plafond"]) == 0 and rc == 0
        ok_all &= ok
        res["alphas"][repr(alpha)] = dict(branche="Creutz" if alpha < 2 else "Kennedy-Pendleton", z1=z1, z2=z2,
                                          m1=float(kv["m1"]), m1_exact=e1, m2=float(kv["m2"]), m2_exact=e2,
                                          plafond=int(kv["plafond"]), ok=bool(ok))
        print(f"alpha={alpha:8.3f} m1={float(kv['m1']):.8f} exact={e1:.8f} z1={z1:+.2f} m2={float(kv['m2']):.8f} "
              f"exact={e2:.8f} z2={z2:+.2f} ok={ok}", flush=True)
    # contrôle négatif : le mutant doit être ROUGE à α = 5 et 20
    for alpha in (5.0, 20.0):
        n_mut = min(N_MUTANT, n_tir)
        rc, kv = testhb(cmd, a.backend, alpha, n_mut, 777, mutant=True)
        z1, z2, _, _ = zscores(kv, alpha, n_mut)
        rouge = abs(z1) >= SEUIL_Z or abs(z2) >= SEUIL_Z
        ok_all &= rouge
        res["mutant"][repr(alpha)] = dict(z1=z1, z2=z2, rouge=bool(rouge))
        print(f"MUTANT alpha={alpha}: z1={z1:+.1f} z2={z2:+.1f} rouge={rouge}", flush=True)
    res["ok"] = bool(ok_all)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("G-H04", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
