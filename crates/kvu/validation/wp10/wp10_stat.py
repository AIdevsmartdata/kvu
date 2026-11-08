# SPDX-License-Identifier: Apache-2.0
"""Statistiques et géométrie SU(2) de WP10, en double précision."""

import csv
import hashlib
import math

import numpy as np

NB_BOUCLES = 10473


def tau_int(x, c=6.0):
    """Fenêtre automatique de Sokal, port de rg::flot::tau."""
    return tau_int_diag(x, c)[:3]


def tau_int_diag(x, c=6.0):
    """Même estimateur, avec témoin de l'arrêt effectif de Sokal."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        return 0.5, 0, 0.0, False
    centres = x - np.mean(x)
    variance = float(np.sum(centres * centres) / n)
    if variance == 0:
        return 0.5, 0, 0.0, False
    tau, fenetre = 0.5, 0
    converge = False
    for w in range(1, n // 4):
        tau += float(np.sum(centres[:-w] * centres[w:]) / (n - w)) / variance
        fenetre = w
        if w >= c * tau:
            converge = True
            break
    tau = max(tau, 0.5)
    return tau, fenetre, tau * math.sqrt(2 * (2 * fenetre + 1) / n), converge


def stat_serie(x):
    """Moyenne, écart-type à n−1 et erreur corrigée de l'autocorrélation."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    moy = float(np.mean(x)) if n else math.nan
    variance = float(np.sum((x - moy) ** 2) / (n - 1)) if n > 1 else math.nan
    tau, fenetre, dtau, converge = tau_int_diag(x)
    return dict(moy=moy, err=math.sqrt(2 * tau * variance / n) if n else math.nan,
                tau=tau, fenetre=fenetre, dtau=dtau, n=n, sd=math.sqrt(variance),
                converge=converge, n_eff=n / (2 * tau))


def combiner_chaines(stats):
    """Combiner équipondérément des chaînes indépendantes de même longueur.

    Les diagnostics τ, fenêtre et sd restent propres aux chaînes : le résultat
    agrégé expose la moyenne, son erreur, l'effectif total et le nombre de chaînes.
    """
    if not stats:
        raise ValueError("aucune chaîne à combiner")
    if len({s['n'] for s in stats}) != 1:
        raise ValueError("les chaînes doivent avoir la même longueur")
    if len(stats) == 1:
        return stats[0]
    k = len(stats)
    return dict(moy=sum(s['moy'] for s in stats) / k,
                err=math.sqrt(sum(s['err'] ** 2 for s in stats)) / k,
                n=sum(s['n'] for s in stats), chaines=k,
                converge=all(s['converge'] for s in stats),
                n_eff=sum(s['n_eff'] for s in stats))


def z_score(m1, e1, m2, e2):
    denominateur = math.hypot(e1, e2)
    return (m1 - m2) / denominateur if denominateur else math.nan


def lire_q64(chemin):
    """Lire exclusivement le format q64 little-endian, C-order."""
    q = np.load(chemin, allow_pickle=False)
    if not isinstance(q, np.ndarray):
        q.close()
        raise ValueError("tableau NPY q64 attendu")
    if (q.dtype.str != '<f8' or q.ndim != 6 or q.shape[0] != 4
            or q.shape[-1] != 4 or q.shape[1] < 1
            or len(set(q.shape[1:5])) != 1 or not q.flags.c_contiguous):
        raise ValueError("q64 attendu : <f8 C-order de forme (4,L,L,L,L,4)")
    return q, q.shape[1]


def qmul(a, b):
    """Produit vectorisé avec la convention de heritage::su2_hb."""
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack((a0*b0 - a1*b1 - a2*b2 - a3*b3,
                     a0*b1 + b0*a1 - (a2*b3 - a3*b2),
                     a0*b2 + b0*a2 - (a3*b1 - a1*b3),
                     a0*b3 + b0*a3 - (a1*b2 - a2*b1)), axis=-1)


def qdag(a):
    return a * np.array([1., -1., -1., -1.])


def plaquette_q64(q):
    """Moyenne sur les sites et les six plans, sans recopier tout le champ."""
    somme = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            produit = qmul(q[mu], np.roll(q[nu], -1, axis=mu))
            produit = qmul(produit, qdag(np.roll(q[mu], -1, axis=nu)))
            # Seule la partie scalaire du dernier produit est nécessaire.
            somme += float(np.sum(produit * q[nu]))
    return somme / (6 * np.prod(q.shape[1:5]).item())


def unitarite_q64(q):
    return float(np.max(np.abs(np.sum(q * q, axis=-1) - 1)))


def sha256_fichier(chemin):
    hachage = hashlib.sha256()
    with open(chemin, 'rb') as fichier:
        for bloc in iter(lambda: fichier.read(8 * 1024 * 1024), b''):
            hachage.update(bloc)
    return hachage.hexdigest()


def lire_csv(chemin):
    with open(chemin, newline='', encoding='utf-8') as fichier:
        return [{cle: valeur if cle in ('phase', 'sha256_cfg') else float(valeur)
                 for cle, valeur in ligne.items()} for ligne in csv.DictReader(fichier)]
