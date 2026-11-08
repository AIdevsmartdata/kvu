# SPDX-License-Identifier: Apache-2.0
"""Re-constat en lecture seule des fenêtres WP10, toutes valeurs de rc comprises.

--journal ace=FICHIER / gam=FICHIER remplace entièrement journalctl/ssh pour
la machine concernée. Les fichiers doivent être au format short-iso daté.
Les extraits produits ont un fichier .lectures.json : il conserve les lectures
illisibles au rejeu et lie les intervalles contrôlés au sha256 de l'extrait.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
from wp10_orch import detecter_fautes


def empreinte(data):
    return hashlib.sha256(data).hexdigest()


def hors_sources(sortie, sources):
    """Refuser une sortie sous une source protégée, y compris via un lien."""
    sortie = Path(sortie).resolve()
    for source in sources:
        source = Path(source).resolve()
        if sortie == source or source in sortie.parents:
            raise ValueError(f"sortie interdite dans la source : {sortie}")


def journaux_arguments(valeurs):
    journaux = {}
    for valeur in valeurs:
        machine, sep, fichier = valeur.partition('=')
        if not sep or machine not in ('ace', 'gam') or not fichier or machine in journaux:
            raise ValueError(f"--journal invalide ou répété : {valeur}")
        journaux[machine] = Path(fichier)
    return journaux


def filtrer_journal(data, t0, t1):
    lignes = []
    for ligne in data.decode('utf-8', errors='replace').splitlines(keepends=True):
        if not ligne.strip():
            continue
        try:
            instant = datetime.fromisoformat(ligne.split()[0]).timestamp()
        except (ValueError, IndexError) as exc:
            raise ValueError('horodatage short-iso absent du journal fourni') from exc
        if t0 - 5 <= instant <= t1 + 5:
            lignes.append(ligne)
    return ''.join(lignes)


def reconstater(config, sortie, journaux):
    fichiers = {}

    def lire(chemin):
        chemin = Path(chemin)
        data = chemin.read_bytes()
        fichiers[str(chemin.resolve())] = empreinte(data)
        return data

    cfg = json.loads(lire(config))
    E = Path(cfg['racine'])
    hors_sources(sortie, [E, config, *journaux.values()])
    sortie = Path(sortie)
    plan = json.loads(lire(E / 'plan.json'))['taches']
    resultat = {'version': 1, 'racine': str(E.resolve()), 'taches': {}, 'sha256_fichiers_lus': fichiers}
    extraits = {}
    fichiers_journaux = {}
    lectures_fournies = {}
    erreurs_journaux = {}
    for machine, chemin in journaux.items():
        try:
            fichiers_journaux[machine] = lire(chemin)
            metadata = Path(str(chemin) + '.lectures.json')
            if metadata.is_file():
                lectures = json.loads(lire(metadata))
                if lectures['machine'] != machine or lectures['sha256'] != empreinte(fichiers_journaux[machine]):
                    raise ValueError(f'extrait incompatible avec ses métadonnées : {chemin}')
                lectures_fournies[machine] = lectures['lectures']
        except OSError as exc:
            erreurs_journaux[machine] = str(exc)
    controlees = fautives = illisibles = 0
    infos = []
    for t in plan:
        ident = t['id']
        etat = json.loads(lire(E / 'etat' / f'{ident}.json'))
        fenetres = resultat['taches'][ident] = []
        for w in etat.get('fenetres', []):
            machine = w.get('machine') or etat.get('machine')
            if machine not in ('ace', 'gam'):
                raise ValueError(f"machine inconnue : {ident} fenêtre {w['k']} : {machine}")
            t0, t1 = float(w['t0']), float(w['t1'])
            if t1 < t0:
                raise ValueError(f"intervalle inversé : {ident} fenêtre {w['k']}")
            erreur, brut, lisible = '', '', False
            try:
                if machine in journaux:
                    if machine in erreurs_journaux:
                        raise OSError(erreurs_journaux[machine])
                    brut = filtrer_journal(fichiers_journaux[machine], t0, t1)
                    lisible = True
                    if machine in lectures_fournies:
                        lecture = [l for l in lectures_fournies[machine]
                                   if (l['tache'], l['k'], l['t0'], l['t1']) == (ident, w['k'], t0, t1)]
                        lisible = len(lecture) == 1 and lecture[0]['lisible']
                        if not lisible:
                            erreur = 'lecture source illisible ou intervalle absent des extraits sauvegardés'
                else:
                    commande = ['journalctl', '-k', '--since', f'@{t0 - 5:.6f}', '--until', f'@{t1 + 5:.6f}',
                                '--no-pager', '-q', '-o', 'short-iso']
                    if machine == 'gam':
                        commande = ['ssh', '-o', 'BatchMode=yes', cfg.get('ssh', 'gamer'), shlex.join(commande)]
                    r = subprocess.run(commande, capture_output=True, text=True, timeout=90)
                    brut, lisible, erreur = r.stdout, r.returncode == 0, r.stderr.strip()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                erreur = str(exc)
            # Extraits bruts, sans préfixe ajouté ; dédoublonnage des intervalles chevauchants.
            extraits.setdefault(machine, {}).update(dict.fromkeys(brut.splitlines(keepends=True)))
            log_lisible = True
            try:
                log = lire(E / 'etat' / 'logs' / ident / f"fenetre_{w['k']:03d}.log").decode('utf-8', errors='replace')
            except OSError as exc:
                log, log_lisible = '', False
                erreur += f'; journal de fenêtre : {exc}'
            fautes = detecter_fautes(brut.splitlines(), log)
            f = {k: w[k] for k in ('k', 'rc', 't0', 't1')}
            f.update(machine=machine, lisible=lisible, log_lisible=log_lisible, **fautes)
            if erreur:
                f['erreur'] = erreur
            fenetres.append(f)
            controlees += 1
            fautives += bool(fautes['fautes_noyau'] or fautes['fautes_log'])
            illisibles += not (lisible and log_lisible)
            infos.append(f"INFO {ident} fenêtre {w['k']} rc={w['rc']} machine={machine} lisible={lisible} "
                         f"log_lisible={log_lisible} fautes_noyau={len(fautes['fautes_noyau'])} fautes_log={len(fautes['fautes_log'])}")
    resultat['totaux'] = {'fenetres_controlees': controlees, 'fenetres_fautives': fautives, 'illisibles': illisibles}
    sortie.mkdir(parents=True, exist_ok=True)
    resultat['sha256_extraits'] = {}
    for machine, lignes in extraits.items():
        data = ''.join(lignes).encode('utf-8')
        nom = f'journal_noyau_{machine}.txt'
        (sortie / nom).write_bytes(data)
        resultat['sha256_extraits'][nom] = empreinte(data)
        lectures = [dict(tache=ident, k=f['k'], t0=f['t0'], t1=f['t1'], lisible=f['lisible'])
                    for ident, fenetres in resultat['taches'].items() for f in fenetres if f['machine'] == machine]
        metadata = {'machine': machine, 'sha256': empreinte(data), 'lectures': lectures}
        (sortie / (nom + '.lectures.json')).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    verdict = 'FAIL' if fautives else ('SKIP' if illisibles else 'PASS')
    finale = (f'{verdict} WP10-ALERTE-IGPU-RECONSTAT {fautives} fenêtre(s) fautive(s) ; '
              f'{illisibles} illisible(s) ; {controlees} contrôlée(s)')
    (sortie / 'constat.json').write_text(json.dumps(resultat, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (sortie / 'constat.txt').write_text('\n'.join(infos + [finale]) + '\n', encoding='utf-8')
    (sortie / 'sha256_fichiers_lus.json').write_text(json.dumps(fichiers, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(finale)
    return resultat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--sortie', required=True)
    ap.add_argument('--journal', action='append', default=[])
    a = ap.parse_args(argv)
    try:
        reconstater(a.config, a.sortie, journaux_arguments(a.journal))
    except (OSError, ValueError, KeyError) as exc:
        print(f'FAIL ECHEC_OUTIL WP10-RECONSTAT {exc}', file=sys.stderr)
        return 1
    return 0  # Le verdict scientifique est dans constat.txt ; rc = succès de l'outil.


if __name__ == '__main__':
    sys.exit(main())
