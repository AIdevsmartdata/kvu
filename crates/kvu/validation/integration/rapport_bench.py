#!/usr/bin/env python3
"""Produit BENCH.md (intégration) à partir des JSON de `kevinotron bench` des deux machines.

Usage : rapport_bench.py DOSSIER_RESULTATS [--sortie BENCH.md] [--commit SHA]
  DOSSIER_RESULTATS contient acemagic/ et gamer/ : su2_<m>.json, cond_<m>.json, cond_cpu_<m>.json, fibre_<m>.json, obs_<m>.json
  (plus, facultatif, des compléments su2_<…>_<m>.json et cond_<…>_<m>.json pour des cas non mesurés du banc principal).
Aucune valeur n'est inventée : une case absente reste « — ». Références externes : su2_hip / su2_cuda mesurés dans la MÊME
fenêtre (champ `reference`) ; cb.py = durées du README de noyau2 (acemagic, numba 1 fil, machine chargée).
"""
import argparse
import glob
import json
import os

MACHINES = ('acemagic', 'gamer')
# README noyau2 : ms par opération (acemagic, 1 fil, machine CHARGÉE).
CB = {
    'ref_rng': {16: 107.0, 24: 630.0, 32: 1970.0},
    'rapide': {16: 72.0, 24: 370.0, 32: 1250.0},
    'mesure': {16: 124.0, 24: 780.0, 32: 2040.0},
}


def charger(dossier, fam, m):
    p = os.path.join(dossier, m, f'{fam}_{m}.json')
    return verifier_document(json.load(open(p))) if os.path.exists(p) else None


def verifier_document(doc):
    if doc.get('schema') not in (None, 'kvu-banc/2'):
        raise ValueError('version de document de banc inconnue')
    if doc.get('schema') == 'kvu-banc/2':
        for m in doc.get('mesures', []):
            if m.get('famille') == 'su2' and m.get('schema_mesure') != 'kvu-banc-su2/2':
                raise ValueError('document version2 avec mesure SU2 non versionnée')
    return doc


def micro(m):
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


def mesures(doc):
    return [x for x in (doc or {}).get('mesures', []) if x.get('statut') is None]


def f(x, fmt='{:.3g}'):
    return '—' if x is None else fmt.format(x)


def ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


def complements(dossier, fam, m):
    """Bancs complémentaires <fam>_<quelque chose>_<m>.json (cas non mesurés du banc principal), triés."""
    motif = os.path.join(dossier, m, f'{fam}_*_{m}.json')
    exclus = {os.path.join(dossier, m, f'{fam}_cpu_{m}.json')}
    return [verifier_document(json.load(open(p))) for p in sorted(glob.glob(motif)) if p not in exclus and '_budget_mur_' not in p]


def su2_index(*docs):
    return {(x['L'], x['precision'], x['chaines']): x for doc in docs for x in mesures(doc)}


def entete_doc(doc):
    if not doc:
        return '_absent_'
    sv = doc.get('sous_verrou_s')
    sv = '' if sv is None else f", {sv:.0f} s sous verrou"
    return f"_{doc['backend']} {doc.get('arch', '')}, {doc.get('fenetres')} fenêtre(s), durée {doc.get('duree_s', 0):.0f} s{sv}._"


def section_su2(d, out):
    docs = {m: charger(d, 'su2', m) for m in MACHINES}
    extra = {m: complements(d, 'su2', m) for m in MACHINES}
    idx = {m: su2_index(docs[m], *extra[m]) for m in MACHINES}
    ls = sorted({k[0] for m in MACHINES for k in idx[m]})
    out += ['## 1. Générateur SU(2) : liens/s, pas/s (un pas = 1 bain thermique + 4 OR)', '']
    for m in MACHINES:
        comp = ''.join(f' Complément : {entete_doc(e)}' for e in extra[m])
        out += [f'### {m}', '', entete_doc(docs[m]) + comp, '',
                '| L | précision | K | liens/s HB | liens/s OR | pas/s (lot) | pas·chaîne/s | champ (Mo) |',
                '|---|---|---|---|---|---|---|---|']
        for (l, p, k), x in sorted(idx[m].items()):
            out.append(f"| {l} | {p} | {k} | {f(x.get('liens_par_s_hb'), '{:.3e}')} | {f(x.get('liens_par_s_or'), '{:.3e}')} | "
                       f"{f(micro(x).get('pas_par_s'))} | {f(micro(x).get('pas_chaines_par_s'))} | {f(x.get('memoire_champ_mo'), '{:.0f}')} |")
        mesures_faites = {f"su2 L={k[0]} {k[1]} K={k[2]}" for k in idx[m]}
        nm = [x for x in (docs[m] or {}).get('mesures', []) if x.get('statut') and x.get('cas') not in mesures_faites]
        if nm:
            out += ['', 'Non mesurés : ' + ' ; '.join(f"{x['cas']} ({x.get('raison') or x.get('erreur')})" for x in nm)]
        out.append('')

    out += ['### 1.1 Ancien contre nouveau : F64 unifié contre su2_hip / su2_cuda (même fenêtre, une chaîne, β = 2,5)', '',
            'Gate de non-régression (§6.7) : unifié ≥ 0,9 × référence.', '',
            '| machine | L | référence | pas/s réf. | pas/s unifié | unifié / réf. | gate |', '|---|---|---|---|---|---|---|']
    for m in MACHINES:
        for l in ls:
            x = idx[m].get((l, 'f64', 1))
            if not x:
                continue
            ref = x.get('reference') or {}
            r = x.get('ratio_pas_unifie_sur_reference')
            g = x.get('gate_non_regression')
            nom = os.path.basename(ref.get('binaire', '')) or '—'
            out.append(f"| {m} | {l} | {nom} | {f(ref.get('pas_par_s'))} | {f(micro(x).get('pas_par_s'))} | {f(r, '{:.3f}')} | "
                       f"{'—' if g is None else ('PASS' if g else 'FAIL')} |")
    out.append('')

    out += ['### 1.2 FP64 contre FP32 (même machine, même L, même K)', '',
            '| machine | L | K | pas·chaîne/s F64 | pas·chaîne/s F32 | F32 / F64 | HB F32/F64 | OR F32/F64 |', '|---|---|---|---|---|---|---|---|']
    for m in MACHINES:
        for l in ls:
            for k in (1, 16):
                a, b = idx[m].get((l, 'f64', k)), idx[m].get((l, 'f32', k))
                if not a or not b:
                    continue
                out.append(f"| {m} | {l} | {k} | {f(micro(a)['pas_chaines_par_s'])} | {f(micro(b)['pas_chaines_par_s'])} | "
                           f"×{f(ratio(micro(b)['pas_chaines_par_s'], micro(a)['pas_chaines_par_s']), '{:.2f}')} | "
                           f"×{f(ratio(b.get('liens_par_s_hb'), a.get('liens_par_s_hb')), '{:.2f}')} | "
                           f"×{f(ratio(b.get('liens_par_s_or'), a.get('liens_par_s_or')), '{:.2f}')} |")
    out.append('')

    out += ['### 1.3 AMD contre CUDA (même L, même précision, même K)', '',
            '| L | précision | K | acemagic pas·chaîne/s | gamer pas·chaîne/s | gamer / acemagic |', '|---|---|---|---|---|---|']
    for l in ls:
        for p in ('f64', 'f32'):
            for k in (1, 16):
                a, b = idx['acemagic'].get((l, p, k)), idx['gamer'].get((l, p, k))
                if not a or not b:
                    continue
                out.append(f"| {l} | {p} | {k} | {f(a['pas_chaines_par_s'])} | {f(b['pas_chaines_par_s'])} | "
                           f"×{f(ratio(b['pas_chaines_par_s'], a['pas_chaines_par_s']), '{:.2f}')} |")
    out.append('')


def cond_index(*docs):
    r = {}
    for doc in docs:
        for x in mesures(doc):
            r[(x['L'], x['mode'])] = x
    return r


def section_cond(d, out):
    out += ['## 2. Bain conditionnel « fin sachant grossier » : GPU contre CPU (moteur Rust) et contre cb.py', '',
            '`ref_rng` rejoue à l\'hôte le générateur numpy de cb.py (loi de référence imposée par G-LOI(a)) ; `rapide` = Philox sur GPU.',
            'CPU = même moteur Rust (rayon, fils de la politique : 8). cb.py = README de noyau2 (numba 1 fil, acemagic chargée).', '',
            '| machine | L | mode | GPU (ms/balayage) | CPU Rust (ms) | cb.py (ms) | GPU / CPU Rust | GPU / cb.py |',
            '|---|---|---|---|---|---|---|---|']
    for m in MACHINES:
        g = cond_index(charger(d, 'cond', m), *complements(d, 'cond', m))
        c = cond_index(charger(d, 'cond_cpu', m))
        for (l, mode) in sorted(set(g) | set(c)):
            sg = g.get((l, mode), {}).get('sweep_s')
            sc = c.get((l, mode), {}).get('sweep_s')
            cb = CB[mode].get(l)
            out.append(f"| {m} | {l} | {mode} | {f(None if sg is None else 1e3 * sg)} | {f(None if sc is None else 1e3 * sc)} | "
                       f"{f(cb)} | ×{f(ratio(sc, sg), '{:.1f}')} | ×{f(ratio(cb, None if sg is None else 1e3 * sg), '{:.1f}')} |")
    out.append('')


def section_fibre_obs(d, out):
    out += ['## 3. Mesure de fibre et observables (GPU)', '',
            '| machine | L | fibre complète (ms) | F seule (ms) | cb.py mesure (ms) | gain fibre | Wilson temporelles R,T ≤ min(L/2, 8) (ms) | pas de flot/s |',
            '|---|---|---|---|---|---|---|---|']
    for m in MACHINES:
        fi = {x['L']: x for x in mesures(charger(d, 'fibre', m))}
        ob = {x['L']: x for x in mesures(charger(d, 'obs', m))}
        for l in sorted(set(fi) | set(ob)):
            a, b = fi.get(l, {}), ob.get(l, {})
            mc = a.get('mesure_complete_s')
            out.append(f"| {m} | {l} | {f(None if mc is None else 1e3 * mc)} | {f(None if a.get('mesure_f_s') is None else 1e3 * a['mesure_f_s'])} | "
                       f"{f(CB['mesure'].get(l))} | ×{f(ratio(CB['mesure'].get(l), None if mc is None else 1e3 * mc), '{:.1f}')} | "
                       f"{f(None if b.get('wilson_s') is None else 1e3 * b['wilson_s'])} | {f(b.get('pas_de_flot_par_s'))} |")
    out.append('')


def charger_cb(dossier, m):
    """cb.py mesuré dans la même session (noyau2/bench.py, numba 1 et 4 fils) : {fils: {L: {...}}}."""
    r = {}
    for fils in (1, 4):
        p = os.path.join(dossier, m, f'cbpy_{fils}fil_{m}.json')
        if os.path.exists(p):
            r[fils] = {int(k[1:]): v for k, v in json.load(open(p))['res'].items()}
    return r


def section_correctifs(d, out):
    """Bancs rejoués après les correctifs du vérificateur (banc_correctifs.sh) : temps mur de su2 gen, bain
    conditionnel sur champ thermalisé, cb.py mesuré dans la même session."""
    out += ['## 5. Correctifs du 04/10 : temps mur de `su2 gen` et bain conditionnel sur champ thermalisé', '']
    out += ['### 5.1 Générateur : noyaux seuls contre temps mur de `su2 gen` (une chaîne)', '',
            'Version2 appariée : noyaux=N/S et mur=N/D du même run instrumenté, synchronisations/horloges incluses.',
            'S/D contrôle la cohérence du profilage ; ce n\'est ni une comparaison aux microbancs ni le débit exact de production non instrumentée.',
            'Historique : estimation sur runs séparés, conservée comme telle. La référence et la gate0,9 utilisent toujours les microbancs.', '',
            '| machine | L | précision | mode | pas/s noyaux | pas/s mur | mur / noyaux | D (ns) | S (ns) | N | réf. micro pas/s | gate F64 |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    idx = {}
    for m in MACHINES:
        doc = charger(d, 'su2_mur', m)
        idx[m] = su2_index(doc)
        for (l, p, k), x in sorted(idx[m].items()):
            mur = (x.get('mur') or {}).get('pas_par_s')
            ref = (x.get('reference') or {}).get('pas_par_s')
            g = x.get('gate_non_regression')
            micro(x)  # Refus des schémas nouveaux incomplets, sans repli sur l'alias apparié.
            profil = x.get('profilage') or {}
            mode = x.get('mode_mesure', 'historique')
            out.append(f"| {m} | {l} | {p} | {mode} | {f(x.get('pas_par_s'))} | {f(mur)} | {f(ratio(mur, x.get('pas_par_s')), '{:.2f}')} | "
                       f"{f(profil.get('duree_totale_ns'), '{:d}')} | {f(profil.get('duree_pas_ns'), '{:d}')} | {f(profil.get('pas_executes'), '{:d}')} | "
                       f"{f(ref)} | {'—' if g is None else ('PASS' if g else 'FAIL')} |")
    out.append('')
    out += ['### 5.2 Gain F32 / F64 au temps mur (une chaîne)', '',
            '| machine | L | F64 mur (pas/s) | F32 mur (pas/s) | F32 / F64 mur | F32 / F64 noyaux |', '|---|---|---|---|---|---|']
    for m in MACHINES:
        for l in sorted({k[0] for k in idx[m]}):
            a, b = idx[m].get((l, 'f64', 1)), idx[m].get((l, 'f32', 1))
            if not a or not b:
                continue
            if a.get('mode_mesure', 'historique') != b.get('mode_mesure', 'historique'):
                out.append(f'| {m} | {l} | — | — | modes différents, non comparés | — |')
                continue
            ma, mb = (a.get('mur') or {}).get('pas_par_s'), (b.get('mur') or {}).get('pas_par_s')
            out.append(f"| {m} | {l} | {f(ma)} | {f(mb)} | ×{f(ratio(mb, ma), '{:.2f}')} | "
                       f"×{f(ratio(b.get('pas_par_s'), a.get('pas_par_s')), '{:.2f}')} |")
    out.append('')
    out += ['### 5.3 Bain conditionnel sur champ thermalisé (300 pas HB + 4 OR, β = 2,3), cb.py dans la même session', '',
            'cb.py = `noyau2/bench.py` (configurations de production de noyau2 ; L = 32 = pavage 2⁴ de L = 16), numba 1 et 4 fils,',
            'mesuré sur acemagic dans la même session que les bancs (machine partagée avec la production).', '',
            '| machine | L | mode | GPU (ms) | CPU Rust (ms) | cb.py 1 fil (ms) | cb.py 4 fils (ms) | GPU / CPU Rust | GPU / cb.py 1 fil | GPU / cb.py 4 fils | plaquette de départ |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    cb = charger_cb(d, 'acemagic')
    cle = {'ref_rng': 'cb_sweep_ref_rng', 'rapide': 'cb_sweep_rapide'}
    for m in MACHINES:
        g = cond_index(charger(d, 'cond_therm', m))
        c = cond_index(charger(d, 'cond_therm_cpu', m))
        for (l, mode) in sorted(set(g) | set(c)):
            sg = g.get((l, mode), {}).get('sweep_s')
            sc = c.get((l, mode), {}).get('sweep_s')
            mg = None if sg is None else 1e3 * sg
            c1 = (cb.get(1, {}).get(l) or {}).get(cle[mode])
            c4 = (cb.get(4, {}).get(l) or {}).get(cle[mode])
            c1, c4 = (None if c1 is None else 1e3 * c1), (None if c4 is None else 1e3 * c4)
            pl = g.get((l, mode), {}).get('plaquette_depart')
            out.append(f"| {m} | {l} | {mode} | {f(mg)} | {f(None if sc is None else 1e3 * sc)} | {f(c1)} | {f(c4)} | "
                       f"×{f(ratio(sc, sg), '{:.1f}')} | ×{f(ratio(c1, mg), '{:.1f}')} | ×{f(ratio(c4, mg), '{:.1f}')} | {f(pl, '{:.4f}')} |")
    prod = []
    for l in (16, 24):
        g = cond_index(charger(d, f'cond_prod_L{l}', 'acemagic'))
        c = cond_index(charger(d, f'cond_prod_L{l}_cpu', 'acemagic'))
        for mode in ('rapide', 'ref_rng'):
            sg, sc = g.get((l, mode), {}).get('sweep_s'), c.get((l, mode), {}).get('sweep_s')
            if sg is not None or sc is not None:
                prod.append(f"| acemagic | {l} | {mode} | {f(None if sg is None else 1e3 * sg)} | {f(None if sc is None else 1e3 * sc)} | "
                            f"{f(g.get((l, mode), {}).get('plaquette_depart'), '{:.4f}')} |")
    if prod:
        out += ['', 'Contrôle : mêmes bancs sur les configurations de production de noyau2 (lues en place) :', '',
                '| machine | L | mode | GPU (ms) | CPU Rust (ms) | plaquette |', '|---|---|---|---|---|---|'] + prod
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dossier')
    ap.add_argument('--sortie', default='BENCH.md')
    ap.add_argument('--commit', default='?')
    ap.add_argument('--correctifs', default=None, help='dossier des bancs de banc_correctifs.sh (acemagic/, gamer/)')
    a = ap.parse_args()
    out = ['# BENCH — kevinotron unifié (`unifie/integ`), acemagic (HIP gfx1151) et gamer (CUDA sm_120)', '',
           f'Produit par `crates/kvu/validation/integration/rapport_bench.py` à partir des JSON de `kevinotron bench` '
           f'(`crates/kvu/validation/integration/resultats/`), binaires du commit {a.commit}. Chaque banc prend le verrou GPU de sa machine '
           'par fenêtres de 600 s au plus et le rend entre deux fenêtres ; le GPU était partagé avec la production '
           '(b10_lite sur acemagic, desert_v3_pilote sur le gamer) : les mesures sont faites sous verrou, donc sans GPU concurrent, '
           'mais la charge CPU des machines n\'était pas nulle. Rapporté, non gaté, sauf la gate de non-régression F64.', '']
    MARQUE = '<!-- LECTURE -->'
    section_su2(a.dossier, out)
    section_cond(a.dossier, out)
    section_fibre_obs(a.dossier, out)
    if a.correctifs:
        section_correctifs(a.correctifs, out)
    out += [MARQUE, '']
    with open(a.sortie, 'w') as fh:
        fh.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
