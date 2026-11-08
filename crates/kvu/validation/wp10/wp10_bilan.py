# SPDX-License-Identifier: Apache-2.0
"""Bilan automatique de la campagne WP10 : temps réel par machine, gates (verdicts rejoués par script), disque (SSD-1), emplacement des sorties.

Écrit (ou remplace) le bloc `<!-- BILAN-AUTO:DEBUT --> … <!-- BILAN-AUTO:FIN -->` du rapport (`rapport` de la configuration, défaut `<racine>/BILAN.md`) ;
le reste du rapport n'est jamais touché. Usage : wp10_bilan.py --config E/orch_config.json [--sortie FICHIER] [--sans-analyse]
Aucun verdict n'est rédigé à la main : les lignes PASS|FAIL|SKIP|INFO viennent de wp10_analyse.py, de wp10_post.py (état des tâches) et du calcul de disque ci-dessous.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEBUT, FIN = "<!-- BILAN-AUTO:DEBUT -->", "<!-- BILAN-AUTO:FIN -->"
PLAFOND_GO = 30.0


def hms(s):
    s = int(s)
    return f"{s // 3600:d} h {s % 3600 // 60:02d} min"


def sha(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def taille_gardee(dossier):
    """Octets des fichiers d'un répertoire de tâche (hors fichiers de reprise et journaux de fenêtre)."""
    total = 0
    n = 0
    for p in Path(dossier).iterdir():
        if p.is_file() and not (p.name == "reprise.json" or p.name.startswith("reprise.champs")):
            total += p.stat().st_size
            n += 1
    return total, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sortie")
    ap.add_argument("--sans-analyse", action="store_true")
    ap.add_argument("--constat", help="constat.json produit par wp10_reconstat.py")
    ap.add_argument("--analyse-sortie", help="répertoire de sortie de l'analyse")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    E = Path(cfg["racine"])
    plan = json.loads((E / "plan.json").read_text(encoding="utf-8"))["taches"]
    rapport = Path(a.sortie or cfg.get("rapport") or (E / "BILAN.md"))
    L = [f"### Bilan automatique (généré par `wp10_bilan.py` le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", ""]

    # --- état et temps réel par machine
    etats = {}
    for t in plan:
        p = E / "etat" / f"{t['id']}.json"
        etats[t["id"]] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"statut": "absente", "machine": None, "fenetres": []}
    constat = None
    if a.constat:
        constat = json.loads(Path(a.constat).read_text(encoding="utf-8"))
        if constat.get("racine") != str(E.resolve()) or set(constat['taches']) != {t['id'] for t in plan}:
            raise ValueError("constat incompatible avec la campagne")
        for ident, e in etats.items():
            fenetres = constat['taches'][ident]
            origine = e.get('fenetres', [])
            if len(fenetres) != len(origine) or any(
                    any(f[k] != w[k] for k in ('k', 'rc', 't0', 't1')) or
                    f['machine'] != (w.get('machine') or e.get('machine'))
                    for f, w in zip(fenetres, origine)):
                raise ValueError(f"constat incomplet ou périmé : {ident}")
            e['alertes'] = max(e.get('alertes', 0), sum(bool(f['fautes_noyau'] or f['fautes_log']) for f in fenetres))
            e['journal_noyau_illisible'] = sum(not (f['lisible'] and f.get('log_lisible', True)) for f in fenetres)
    L += ["#### Tâches et temps réel", "",
          "| tâche | machine | statut | fenêtres | durée des fenêtres (min) | calcul rapporté par kvu (min) | attente + mise en route (min) | tentatives | alertes GPU |",
          "|---|---|---|---|---|---|---|---|---|"]
    par_machine = {"ace": {"fen": 0, "duree": 0.0, "calcul": 0.0, "taches": 0, "t0": None, "t1": None},
                   "gam": {"fen": 0, "duree": 0.0, "calcul": 0.0, "taches": 0, "t0": None, "t1": None}}
    for t in plan:
        e = etats[t["id"]]
        f = e.get("fenetres", [])
        d = sum(w["duree_s"] for w in f)
        c = sum((w.get("calcul_s") or 0.0) for w in f)
        L.append(f"| {t['id']} | {e.get('machine') or '-'} | {e['statut']} | {len(f)} | {d / 60:.1f} | {c / 60:.1f} | {(d - c) / 60:.1f} | {e.get('tentatives', 0)} | {e.get('alertes', 0)} |")
        m = e.get("machine")
        if m in par_machine and f:
            pm = par_machine[m]
            pm["fen"] += len(f)
            pm["duree"] += d
            pm["calcul"] += c
            pm["taches"] += 1 if e["statut"] in ("terminee", "post", "post_ok", "rapatriement") else 0
            pm["t0"] = min(x for x in (pm["t0"], f[0]["t0"]) if x is not None)
            pm["t1"] = max(x for x in (pm["t1"], f[-1]["t1"]) if x is not None)
    L += ["", "| machine | tâches terminées | fenêtres | temps sous verrou, attente comprise | calcul rapporté par kvu | première fenêtre | dernière fenêtre | temps mur (première → dernière) |", "|---|---|---|---|---|---|---|---|"]
    for m, nom in (("ace", "acemagic (HIP)"), ("gam", "gamer (CUDA)")):
        pm = par_machine[m]
        if pm["t0"] is None:
            L.append(f"| {nom} | 0 | 0 | — | — | — | — | — |")
            continue
        L.append(f"| {nom} | {pm['taches']} | {pm['fen']} | {hms(pm['duree'])} | {hms(pm['calcul'])} | "
                 f"{datetime.fromtimestamp(pm['t0']).strftime('%m-%d %H:%M')} | {datetime.fromtimestamp(pm['t1']).strftime('%m-%d %H:%M')} | {hms(pm['t1'] - pm['t0'])} |")
    ts = [pm["t0"] for pm in par_machine.values() if pm["t0"]]
    te = [pm["t1"] for pm in par_machine.values() if pm["t1"]]
    if ts and te:
        L += ["", f"Temps mur de la campagne (première fenêtre → dernière fenêtre, deux machines) : **{hms(max(te) - min(ts))}**.", ""]

    # --- WP10-FENETRE, WP10-ALERTE-IGPU, verrous (verdicts rejoués sur l'état des tâches)
    calc, inconnues, modes, illisibles, alertes, echecs = [], 0, {}, 0, 0, []
    logs_illisibles = 0
    for t in plan:
        e = etats[t["id"]]
        alertes += e.get("alertes", 0)
        # Un journal de fenêtre illisible peut masquer une faute absente du noyau.
        # Conserver le compteur historique et compter chaque fenêtre au plus une fois.
        logs_illisibles += sum(not w.get("log_lisible", True) for w in e.get("fenetres", []))
        illisibles += max(e.get("journal_noyau_illisible", 0), sum(
            not (w.get("journal_lisible", True) and w.get("log_lisible", True))
            for w in e.get("fenetres", [])))
        if e["statut"] in ("echec", "post_echec", "echec_rapatriement"):
            echecs.append(t["id"])
        for w in e.get("fenetres", []):
            modes[w.get("verrou", "?")] = modes.get(w.get("verrou", "?"), 0) + 1
            if w.get("calcul_s") is None:
                inconnues += 1
            else:
                calc.append((w["calcul_s"], t["id"], w["k"]))
    if calc:
        mx = max(calc)
        verdict_f = "FAIL" if mx[0] > 600 else ("SKIP" if inconnues else "PASS")
        L.append("```text")
        L.append(f"{verdict_f} WP10-FENETRE {len(calc)} fenêtres mesurées, plus longue sous verrou {mx[0]:.0f} s ({mx[1]} fenêtre {mx[2]}) pour 600 s exigés ; {inconnues} fenêtre(s) sans durée lisible ; verrous : {modes}")
    else:
        L += ["```text", "SKIP WP10-FENETRE aucune fenêtre mesurée"]
    repli = 0
    try:
        repli = sum(1 for l in (E / "orch.log").read_text(encoding="utf-8").splitlines() if "REPLI-PARTAGE" in l)
    except OSError:
        pass
    L.append(f"INFO WP10-VERROU REPLI-PARTAGE : {repli} événement(s) (verrou exclusif non obtenu en {cfg.get('attente_exclusif_s', 600)} s)")
    libelle_journal = "journal noyau ou de fenêtre" if logs_illisibles else "journal noyau"
    L.append(f"{'PASS' if alertes == 0 and illisibles == 0 else ('FAIL' if alertes else 'SKIP')} WP10-ALERTE-IGPU {alertes} alerte(s) de faute GPU ; {libelle_journal} illisible pour {illisibles} fenêtre(s)")
    if constat is not None:
        details = [f"{ident} fenêtre {f['k']} rc={f['rc']} première ligne noyau : "
                   + (f['fautes_noyau'][0] if f['fautes_noyau'] else '(aucune ; log : ' + f['fautes_log'][0] + ')')
                   for ident, fenetres in constat['taches'].items() for f in fenetres
                   if f['fautes_noyau'] or f['fautes_log']]
        L.append(f"INFO WP10-ALERTE-IGPU-CONSTAT sha256={sha(a.constat)} ; " + (' ; '.join(details) or 'aucune fenêtre fautive'))
    L.append(f"{'PASS' if not echecs else 'FAIL'} WP10-ECHECS tâches en échec : {', '.join(echecs) if echecs else 'aucune'}")
    L += ["```", ""]

    # --- post-traitements et gates par tâche
    L += ["#### Post-traitements par tâche (verdicts de `wp10_post.py`)", "", "| tâche | statut | bloc2 | G-H13 | manifeste kvu-regen/1 | élagage | G-H13 après élagage | rapatriement (s) |", "|---|---|---|---|---|---|---|---|"]
    for t in plan:
        po = etats[t["id"]].get("post", {})
        def c(nom):
            x = po.get(nom)
            if not x:
                return "—"
            return ("PASS" if x["rc"] == 0 else f"FAIL(rc {x['rc']})") + f" ({x['s']:.0f} s)"
        L.append(f"| {t['id']} | {etats[t['id']]['statut']} | {c('bloc2')} | {c('h13')} | {c('manifeste')} | {c('elaguer')} | {c('h13_apres')} | {po.get('rapatriement_s', '—')} |")
    L.append("")

    # --- analyse statistique
    if not a.sans_analyse:
        sortie_an = Path(a.analyse_sortie) if a.analyse_sortie else E / "analyse"
        argv_an = [sys.executable, "-I", str(E / "outils" / "wp10_analyse.py"), "--racine", str(E), "--plan", str(E / "plan.json"), "--sortie", str(sortie_an), "--bloc3b"]
        r = subprocess.run(argv_an, capture_output=True, text=True)
        L += [f"#### Analyse statistique (`wp10_analyse.py`, rc {r.returncode}) : gates LOI-2, LOI-2-MUT, RECERT, COMPLETUDE, domaine, τ", "", "```text"]
        L += r.stdout.strip().splitlines() or ["(aucune sortie)"]
        L += ["```", ""]
        if (sortie_an / "tables.md").is_file():
            L += ["<details><summary>Tables (statistiques par chaîne, cellules, z, domaine, τ)</summary>", "", (sortie_an / "tables.md").read_text(encoding="utf-8"), "", "</details>", ""]

    # --- disque : SSD-1
    L += ["#### SSD-1 : disque", ""]
    total = 0
    n_fichiers = 0
    par_tache = []
    for t in plan:
        d = E / t["id"]
        if d.is_dir():
            o, n = taille_gardee(d)
            total += o
            n_fichiers += n
            par_tache.append((t["id"], o))
    du = os.statvfs(str(E))
    libre_go = du.f_bavail * du.f_frsize / 1e9
    ok = total / 1e9 <= PLAFOND_GO
    L.append(f"{'PASS' if ok else 'FAIL'} SSD-1 sorties conservées : {total / 1e9:.3f} Go (plafond {PLAFOND_GO:.0f} Go, D6) dans {n_fichiers} fichiers sous `{E}` ; libre sur l'acemagic : {libre_go:.0f} Go (≥ 30 Go exigés)"
             + ("" if libre_go >= 30 else " ; FAIL libre < 30 Go"))
    if not ok:
        L.append("règle de repli enregistrée (PREREG §5) : retirer les configurations L = 12 de mesure impaire des chaînes P36 en listant sha256 et taille dans `ELAGAGE.json` : à appliquer par le pilote, non exécutée automatiquement.")
    L += ["", "| tâche | octets conservés |", "|---|---|"] + [f"| {i} | {o / 1e6:.1f} Mo |" for i, o in par_tache] + [""]

    # --- provenance
    L += ["#### Provenance (empreintes)", "", "| objet | sha256 |", "|---|---|"]
    for nom, chemin in (("binaire acemagic", cfg.get("kvu_ace")),):
        if chemin and Path(chemin).is_file():
            L.append(f"| {nom} `{chemin}` | `{sha(chemin)}` |")
    if cfg.get("sha_gam"):
        L.append(f"| binaire gamer `{cfg.get('kvu_gam')}` (attendu au lancement) | `{cfg['sha_gam']}` |")
    for f in sorted((E / "outils").glob("*")):
        if f.is_file():
            L.append(f"| outil `{f.name}` | `{sha(f)}` |")
    L.append(f"| plan | `{sha(E / 'plan.json')}` |")
    L.append("")
    L += ["#### Emplacement des sorties", "",
          f"- acemagic : `{E}/<tâche>/` (configurations grossières `bloc3_cfg_*.npy` L/3, `bloc3b_cfg_*.npy` L/9, CSV, manifestes `regen_bloc3.json` et `regen_bloc3b.json`, `h13.json`, `H13_PASS`) ; analyse : `{Path(a.analyse_sortie) if a.analyse_sortie else E / 'analyse'}/` ; état et journaux : `{E}/etat/`, `{E}/orch.log`.",
          f"- gamer (originaux de classe B, conservés) : `{cfg.get('gam_racine')}/E/<tâche>/`.", ""]
    bloc = "\n".join([DEBUT] + L + [FIN]) + "\n"
    if rapport.is_file():
        # Le rejeu doit conserver même les CRLF hors bloc. Le mode historique
        # conserve son comportement de lecture, pour une sortie par défaut identique.
        texte = rapport.read_bytes().decode("utf-8") if a.constat else rapport.read_text(encoding="utf-8")
        if DEBUT in texte and FIN in texte:
            avant, reste = texte.split(DEBUT, 1)
            _, apres = reste.split(FIN, 1)
            texte = avant + bloc.rstrip("\n") + apres
        else:
            texte = texte.rstrip("\n") + "\n\n" + bloc
    else:
        texte = bloc
    rapport.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(rapport) + ".part")
    part.write_text(texte, encoding="utf-8")
    os.replace(part, rapport)
    print(f"bilan écrit : {rapport}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
