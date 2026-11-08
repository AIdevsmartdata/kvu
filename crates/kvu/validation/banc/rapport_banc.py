#!/usr/bin/env python3
"""Produit BANC_UNIFIE.md à partir des JSON de `kevinotron bench` des deux machines.

Usage : rapport_banc.py DOSSIER_RESULTATS [--sortie BANC_UNIFIE.md]
  DOSSIER_RESULTATS contient acemagic/ et gamer/ (su2_<machine>.json, cond_…, fibre_…, obs_…).
Aucune valeur n'est inventée : une case absente reste « — ». Les références externes sont (a) su2_hip / su2_cuda
mesurés dans la même fenêtre (champ `reference` du JSON) et (b) les durées de cb.py du README de noyau2
(acemagic, 1 fil, machine chargée) pour cond et fibre.
"""
import argparse
import json
import os

# README noyau2 : ms par opération (acemagic, 1 fil, machine CHARGÉE).
CB = {
    'balayage_ref_rng_ms': {8: 5.2, 12: 36.0, 16: 107.0, 24: 630.0, 32: 1970.0},
    'balayage_rapide_ms': {8: 2.5, 12: 17.0, 16: 72.0, 24: 370.0, 32: 1250.0},
    'mesure_ms': {8: 7.6, 12: 42.0, 16: 124.0, 24: 780.0, 32: 2040.0},
}


def charger(dossier, fam, machine):
    p = os.path.join(dossier, machine, f'{fam}_{machine}.json')
    doc = json.load(open(p)) if os.path.exists(p) else None
    if doc and doc.get('schema') not in (None, 'kvu-banc/2'):
        raise ValueError('version de document de banc inconnue')
    if doc and doc.get('schema') == 'kvu-banc/2':
        for m in doc.get('mesures', []):
            if m.get('famille') == 'su2' and m.get('schema_mesure') != 'kvu-banc-su2/2':
                raise ValueError('document version2 avec mesure SU2 non versionnée')
    return doc


def micro(m):
    """Les rapports historiques et la référence utilisent toujours les microbancs."""
    version = m.get('schema_mesure')
    if version is None:
        if any(k in m for k in ('microbench', 'mode_mesure', 'profilage')):
            raise ValueError('mesure SU2 nouvelle sans version explicite')
        return m
    if version != 'kvu-banc-su2/2' or not isinstance(m.get('microbench'), dict):
        raise ValueError('mesure SU2 versionnée sans microbench compatible')
    if m.get('mode_mesure') not in ('apparie', 'micro_seul'):
        raise ValueError('mode de mesure SU2 inconnu')
    return m['microbench']


def f(x, fmt='{:.3g}'):
    return '—' if x is None else fmt.format(x)


def mesures_ok(doc, fam):
    return [m for m in (doc or {}).get('mesures', []) if m.get('famille') == fam]


def non_mesures(doc):
    return [m for m in (doc or {}).get('mesures', []) if m.get('statut') in ('non_mesure', 'echec')]


def table_su2(doc):
    L = ['| L | précision | K | liens/s HB | liens/s OR | pas/s (lot) | pas·chaîne/s | champ (Mo) | réf. pas/s | unifié / réf. |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for m in mesures_ok(doc, 'su2'):
        ref = m.get('reference', {})
        mb = micro(m)
        L.append(f"| {m['L']} | {m['precision']} | {m['chaines']} | {f(m['liens_par_s_hb'], '{:.3e}')} | {f(m['liens_par_s_or'], '{:.3e}')} | "
                 f"{f(mb['pas_par_s'])} | {f(mb['pas_chaines_par_s'])} | {f(m['memoire_champ_mo'], '{:.0f}')} | "
                 f"{f(ref.get('pas_par_s'))} | {f(m.get('ratio_pas_unifie_sur_reference'), '{:.3f}')} |")
    return '\n'.join(L)


def gains_f32(doc):
    par = {}
    for m in mesures_ok(doc, 'su2'):
        par[(m['L'], m['precision'], m['chaines'])] = micro(m)['pas_chaines_par_s']
    out = ['| L | K | F32 / F64 (pas·chaîne/s) |', '|---|---|---|']
    for (L, p, k), v in sorted(par.items()):
        if p == 'f32' and (L, 'f64', k) in par:
            out.append(f'| {L} | {k} | ×{v / par[(L, "f64", k)]:.2f} |')
    return '\n'.join(out)


def table_profilage(doc):
    lignes = ['| L | précision | K | D (ns) | S (ns) | N | N/S (pas/s) | N/D (pas/s) | S/D |',
              '|---|---|---|---|---|---|---|---|---|']
    for m in mesures_ok(doc, 'su2'):
        if m.get('mode_mesure') != 'apparie':
            continue
        micro(m)
        p = m['profilage']
        lignes.append(f"| {m['L']} | {m['precision']} | {m['chaines']} | {p['duree_totale_ns']} | {p['duree_pas_ns']} | "
                      f"{p['pas_executes']} | {f(m['pas_par_s'])} | {f(m['mur']['pas_par_s'])} | {f(m['rapport_mur_sur_noyaux'])} |")
    return '\n'.join(lignes) if len(lignes) > 2 else '_profil apparié non mesuré (JSON historique ou --sans-mur)._'


def table_cond(doc):
    L = ['| L | mode | balayage (ms) | balayages/s | cb.py (ms) | gain |', '|---|---|---|---|---|---|']
    for m in mesures_ok(doc, 'cond'):
        cle = 'balayage_ref_rng_ms' if m['mode'] == 'ref_rng' else 'balayage_rapide_ms'
        cb = CB[cle].get(m['L'])
        ms = m['sweep_s'] * 1e3
        L.append(f"| {m['L']} | {m['mode']} | {ms:.3g} | {f(m['sweeps_par_s'])} | {f(cb)} | {('×%.1f' % (cb / ms)) if cb else '—'} |")
    return '\n'.join(L)


def table_fibre(doc):
    L = ['| L | mesure complète (ms) | mesure F seule (ms) | cb.py mesure (ms) | gain |', '|---|---|---|---|---|']
    for m in mesures_ok(doc, 'fibre'):
        ms = m['mesure_complete_s'] * 1e3
        cb = CB['mesure_ms'].get(m['L'])
        L.append(f"| {m['L']} | {ms:.3g} | {m['mesure_f_s'] * 1e3:.3g} | {f(cb)} | {('×%.1f' % (cb / ms)) if cb else '—'} |")
    return '\n'.join(L)


def table_obs(doc):
    L = ['| L | boucles de Wilson (ms) | pas de flot/s |', '|---|---|---|']
    for m in mesures_ok(doc, 'obs'):
        L.append(f"| {m['L']} | {m['wilson_s'] * 1e3:.3g} | {f(m['pas_de_flot_par_s'])} |")
    return '\n'.join(L)


def gate_non_regression(docs):
    lignes, ok_all, n = [], True, 0
    for mach, doc in docs.items():
        for m in mesures_ok(doc, 'su2'):
            if 'gate_non_regression' in m:
                n += 1
                ok = m['gate_non_regression']
                ok_all &= ok
                lignes.append(f"| {mach} | {m['L']} | {m['reference']['binaire'].split('/')[-1]} | {m['ratio_pas_unifie_sur_reference']:.3f} | "
                              f"{'PASS' if ok else 'FAIL'} |")
    verdict = 'PASS' if n and ok_all else ('FAIL' if n else 'NON MESURÉE')
    tete = ['| machine | L | référence | pas/s unifié ÷ référence | gate (≥ 0,9) |', '|---|---|---|---|---|']
    return verdict, '\n'.join(tete + lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dossier')
    ap.add_argument('--sortie', default='BANC_UNIFIE.md')
    a = ap.parse_args()
    machines = [m for m in ('acemagic', 'gamer') if os.path.isdir(os.path.join(a.dossier, m))]
    docs = {m: {fam: charger(a.dossier, fam, m) for fam in ('su2', 'cond', 'fibre', 'obs')} for m in machines}
    out = ['# BANC_UNIFIE — kevinotron unifié, bancs des deux machines', '',
           'Produit par `validation/banc/rapport_banc.py` à partir des JSON de `kevinotron bench` (verrou GPU, fenêtres ≤ 600 s,',
           'GPU rendu entre deux fenêtres). Rapporté, non gaté, sauf la gate de non-régression : **F64 unifié ≥ 0,9 × su2_hip / su2_cuda**',
           '(référence lancée dans la même fenêtre, mêmes L, β = 2,5, 4 OR par pas, une chaîne).', '']
    verdict, tab = gate_non_regression({m: docs[m]['su2'] for m in machines})
    out += [f'## Gate de non-régression F64 : **{verdict}**', '', tab, '']
    for m in machines:
        d = docs[m]
        meta = next((x for x in d.values() if x), {})
        out += [f"## {m} (backend {meta.get('backend', '?')}, {meta.get('arch', '?')})", '']
        for fam, titre, tab in (('su2', 'Générateur SU(2)', table_su2), ('cond', 'Bain conditionnel', table_cond),
                                ('fibre', 'Mesure de fibre', table_fibre), ('obs', 'Observables', table_obs)):
            out += [f'### {titre}', '']
            doc = d[fam]
            if doc is None:
                out += ['_banc non exécuté sur cette machine._', '']
                continue
            out += [tab(doc), '']
            if fam == 'su2':
                out += ['Gain du chemin rapide :', '', gains_f32(doc), '']
                out += ['Profilage apparié version2 (horloges/synchronisations incluses) :', '', table_profilage(doc), '',
                        'S/D vérifie la cohérence des intervalles ; ce profil instrumenté n\'est pas une mesure de performance relative au microbanc.', '']
            nm = non_mesures(doc)
            if nm:
                out += ['Cas non mesurés ou en échec :', ''] + [f"- {x.get('cas')} : {x.get('statut')} ({x.get('raison') or x.get('erreur')})" for x in nm] + ['']
            sv = doc.get('sous_verrou_s')
            sv_txt = '' if sv is None else f", dont {sv:.0f} s sous verrou"
            out += [f"_fenêtres : {doc.get('fenetres')}, durée {doc.get('duree_s', 0):.0f} s{sv_txt}, budget {doc.get('budget_s'):.0f} s._", '']
    out += ['## Lecture', '',
            '- `liens/s` = 4·V·K ÷ durée d\'un sweep ; un pas = 1 bain thermique + 4 OR.',
            '- Les colonnes de débit et la gate0,9 ci-dessus comparent les microbancs HB/OR, y compris avec JSON version2.',
            '- Le couple apparié version2 décrit un run instrumenté : S/D contrôle sa cohérence, pas sa performance contre les microbancs.',
            '- Références cond/fibre : durées de cb.py (numba, 1 fil, acemagic chargée) du README de noyau2, donc un gain « contre l\'existant »,',
            '  pas un banc de même fenêtre.',
            '- Un résultat du gamer est de la RECHERCHE ; ce qui entre dans un verdict est revérifié sur acemagic.', '']
    comp = os.path.join(a.dossier, 'complements.md')
    if os.path.exists(comp):   # mesures de contrôle ajoutées à la main, rapportées telles quelles
        out += ['', open(comp).read().rstrip('\n'), '']
    open(a.sortie, 'w').write('\n'.join(out))
    print(f'{a.sortie} écrit (gate de non-régression : {verdict})')


if __name__ == '__main__':
    main()
