#!/usr/bin/env python3
"""Juge les sorties du chemin FP32 : LOI-0 à LOI-4, LOI-τ, LOT-1-F32, REPRISE-1-F32 (tâche T4).

Usage : juge_f32.py --dir DOSSIER_MACHINE [--f64-l16 DOSSIER_FP64] [--sans-p5] [--json SORTIE.json]

DOSSIER_MACHINE contient (écrit par les pilotes Rust, tests/su2_f32_pilote.rs) :
  loi0/loi0.json, loi1/{nominal,mutant}/res.jsonl + ks_*.f32, chaines/{f32,f32_beta1002}/L*_b*_c*,
  lot1.json, reprise.json, (bench.json).
Les seuils sont ceux de loi.SEUILS (gelés). Code de sortie 0 si tout est PASS, 8 sinon (validation).
Chaque ligne finale est « PASS <gate> » ou « FAIL <gate> » (ou « SKIP <gate> » : ne vaut pas vert).
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loi  # noqa: E402

S = loi.SEUILS


def lire_json(chemin):
    return json.load(open(chemin)) if os.path.exists(chemin) else None


def verdict(nom, ok, details=''):
    ligne = f"{'PASS' if ok else 'FAIL'} {nom}" + (f' ({details})' if details else '')
    print(ligne, flush=True)
    return ligne


def sans_prive(d):
    return {k: sans_prive(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith('_')}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dir', required=True)
    ap_.add_argument('--f64-l16')
    ap_.add_argument('--sans-p5', action='store_true', help='saute LOI-3 (permutations) : SKIP, ne vaut pas vert')
    ap_.add_argument('--json')
    a = ap_.parse_args()
    D = a.dir
    ap = loi.charger_ap()
    out, lignes = {}, []

    # ---- LOI-0
    l0 = lire_json(os.path.join(D, 'loi0', 'loi0.json'))
    if l0:
        ok = l0['defaut_final'] <= S['loi0_defaut_norme'] and l0['defaut_max'] <= S['loi0_defaut_norme'] and l0['plafond'] == 0
        print(f"LOI-0 : L={l0['l']} β={l0['beta']} {l0['pas']} pas : max||q|-1| final {l0['defaut_final']:.3e}, "
              f"max sur les points de contrôle {l0['defaut_max']:.3e} (seuil {S['loi0_defaut_norme']:.0e}), plafond {l0['plafond']}")
        out['LOI-0'] = l0
        lignes.append(verdict('LOI-0', ok))
    else:
        lignes.append('SKIP LOI-0 (loi0.json absent)')

    # ---- LOI-1 (nominal) et LOI-4b (mutant metro)
    r1 = None
    if os.path.exists(os.path.join(D, 'loi1', 'nominal', 'res.jsonl')):
        r1 = loi.juge_loi1(os.path.join(D, 'loi1', 'nominal'))
        for r in r1:
            print(f"LOI-1 α={r['alpha']:<10.7g} z1={r['z1']:+6.2f} z2={r['z2']:+6.2f} KS D={r['ks_D']:.2e} p={r['ks_p']:.3f} "
                  f"plafond={r['plafond']} min={r['minimum']} -> {'ok' if r['ok'] else 'ROUGE'}")
        out['LOI-1'] = r1
        lignes.append(verdict('LOI-1', all(r['ok'] for r in r1) and len(r1) == len(S['loi1_alphas']),
                              f"{len(r1)} α, |z|max={max(max(abs(r['z1']), abs(r['z2'])) for r in r1):.2f}, "
                              f"KS p min={min(r['ks_p'] for r in r1):.3g}"))
    else:
        lignes.append('SKIP LOI-1 (res.jsonl absent)')
    r1m = None
    if os.path.exists(os.path.join(D, 'loi1', 'mutant', 'res.jsonl')):
        r1m = loi.juge_loi1(os.path.join(D, 'loi1', 'mutant'))
        for r in r1m:
            print(f"LOI-4b mutant metro α={r['alpha']:<10.7g} z1={r['z1']:+9.2f} z2={r['z2']:+9.2f} KS p={r['ks_p']:.3g} "
                  f"-> {'vert (!)' if r['ok'] else 'rouge'}")
        out['LOI-4b'] = loi.juge_loi4b(r1m)
        out['LOI-4b_detail'] = r1m

    # ---- LOI-2, LOI-τ
    base32 = os.path.join(D, 'chaines', 'f32')
    l2 = None
    if os.path.isdir(base32):
        l2 = loi.juge_loi2(ap, base32, a.f64_l16)
        for k, c in l2['cellules'].items():
            print(f"LOI-2 {k:10s} F32 {c['m32']:.7f}({c['e32'] * 1e7:.0f}e-7) FP64 {c['m64']:.7f}({c['e64'] * 1e7:.0f}e-7) "
                  f"z={c['z']:+.2f} tau32={[round(t, 2) for t in c['tau32']]} tau64={[round(t, 2) for t in c['tau64']]}")
        print(f"LOI-2 χ²({l2['ndl']}) = {l2['chi2']:.2f}, p = {l2['p_chi2']:.4f}")
        ok2 = l2['ok_6cellules']
        det = f"6 cellules : |z|max={max(abs(c['z']) for c in l2['cellules'].values()):.2f}, chi2={l2['chi2']:.2f}, p={l2['p_chi2']:.3f}"
        if 'L16' in l2:
            for k, c in l2['L16'].items():
                print(f"LOI-2 {k:10s} F32 {c['m32']:.7f}({c['e32'] * 1e7:.0f}e-7) FP64 {c['m64']:.7f}({c['e64'] * 1e7:.0f}e-7) z={c['z']:+.2f}")
            ok2 = ok2 and l2['ok_L16']
            det += ', L16 ' + ', '.join(f"{k} z={c['z']:+.2f}" for k, c in l2['L16'].items())
        else:
            ok2 = None
        out['LOI-2'] = sans_prive(l2)
        lignes.append(verdict('LOI-2', ok2, det) if ok2 is not None else 'SKIP LOI-2 (comparateurs FP64 L16 absents ; les 6 cellules : '
                              + ('PASS' if l2['ok_6cellules'] else 'FAIL') + ' ' + det + ')')
        tau = loi.juge_loitau(l2, ap)
        print(f"LOI-τ L=8 β=2.3 : τ32={tau['tau32']:.3f} τ64={tau['tau64']:.3f} |Δτ|/τ={tau['ecart_rel']:.3f} "
              f"seuil 0,2+2σ={tau['seuil']:.3f} (σ_rel={tau['sigma_rel']:.3f})")
        out['LOI-tau'] = tau
        lignes.append(verdict('LOI-τ', tau['ok']))
    else:
        lignes.append('SKIP LOI-2 et LOI-τ (chaînes absentes)')

    # ---- LOI-3
    if os.path.isdir(base32) and not a.sans_p5:
        p5 = loi.juge_loi3_plaquettes(base32)
        for k in ('L6', 'L8'):
            r = p5[k]
            print(f"LOI-3 plaquettes locales {k} β=2.8 : D={r['D_exact']:.2e} p_perm={r['p_perm']:.3f} "
                  f"(p naïf {r['p_asympt_naive']:.3g}, D_perm q50 {r['D_perm_q50']:.2e}) cfgs {r['n_cfg_f32']}/{r['n_cfg_ref']}")
        pol = loi.juge_loi3_polyakov(ap, base32)
        print(f"LOI-3 |P_3| L=8 β=2.3 : amincissement {pol['pas_amincissement']}, n={pol['n32']}/{pol['n64']}, "
              f"D={pol['D']:.3e}, p={pol['p']:.3f}, moy {pol['moy32']:.4f}/{pol['moy64']:.4f}")
        out['LOI-3'] = dict(plaquettes=p5, polyakov=pol)
        lignes.append(verdict('LOI-3', p5['ok'] and pol['ok'],
                              f"p_perm L6={p5['L6']['p_perm']:.3f} L8={p5['L8']['p_perm']:.3f}, p(|P|)={pol['p']:.3f}"))
    else:
        lignes.append('SKIP LOI-3')

    # ---- LOI-4a (β x 1,002) puis verdict LOI-4
    base_mut = os.path.join(D, 'chaines', 'f32_beta1002')
    ok4a = None
    if os.path.isdir(base_mut):
        l2m = loi.juge_loi2(ap, base_mut)
        for k, c in l2m['cellules'].items():
            print(f"LOI-4a β×{loi.FACTEUR_MUTANT_BETA} {k:10s} z={c['z']:+9.2f}")
        v4a = loi.juge_loi4a(l2m)
        print(f"LOI-4a : {v4a['n_cellules_rouges']}/6 cellules |z|>3, p(χ²)={v4a['p_chi2']:.3g} -> {'ROUGE (comme exigé)' if v4a['rouge'] else 'VERT (la puissance du test est insuffisante)'}")
        out['LOI-4a'] = dict(v4a, cellules=sans_prive(l2m)['cellules'])
        ok4a = v4a['rouge']
    v4b = out.get('LOI-4b')
    if ok4a is not None and v4b is not None:
        print(f"LOI-4b : mutant metro rouge à α={v4b['alphas']} : {v4b['rouges']}")
        lignes.append(verdict('LOI-4', ok4a and v4b['rouge'], f"(a) β×1,002 rouge={ok4a}, (b) metro rouge={v4b['rouge']}"))
    else:
        lignes.append('SKIP LOI-4')

    # ---- LOT-1-F32, REPRISE-1-F32 (calculés par les tests Rust, rapportés ici)
    for nom, f in (('LOT-1-F32', 'lot1.json'), ('REPRISE-1-F32', 'reprise.json')):
        j = lire_json(os.path.join(D, f))
        if j:
            print(f"{nom} : {j.get('detail', '')}")
            out[nom] = j
            lignes.append(verdict(nom, bool(j['ok'])))
        else:
            lignes.append(f'SKIP {nom} ({f} absent)')

    bench = lire_json(os.path.join(D, 'bench.json'))
    if bench:
        out['bench'] = bench
        for b in bench.get('lignes', []):
            print('BENCH', json.dumps(b))

    print('---- RÉCAPITULATIF')
    for l in lignes:
        print(l)
    out['lignes'] = lignes
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, default=float)
    ko = [l for l in lignes if l.startswith('FAIL') or l.startswith('SKIP')]
    sys.exit(0 if not ko else 8)


if __name__ == '__main__':
    main()
