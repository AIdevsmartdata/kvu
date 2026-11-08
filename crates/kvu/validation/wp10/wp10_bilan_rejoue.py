# SPDX-License-Identifier: Apache-2.0
"""Rejoue le bilan WP10 dans --travail ; publication seulement avec --ecrire --commit SHA.

Les options --journal machine=fichier permettent un rejeu entièrement hors ligne.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
from wp10_bilan import DEBUT, FIN
from wp10_reconstat import empreinte, hors_sources, journaux_arguments

ICI = Path(__file__).resolve().parent


def decouper(data):
    debut, fin = DEBUT.encode(), FIN.encode()
    if data.count(debut) != 1 or data.count(fin) != 1:
        raise ValueError('un unique bloc BILAN-AUTO est requis')
    i, j = data.index(debut), data.index(fin) + len(fin)
    if j <= i:
        raise ValueError('bornes BILAN-AUTO inversées')
    return data[:i], data[i:j], data[j:]


def rejouer(a):
    config = Path(a.config).resolve()
    cfg = json.loads(config.read_bytes())
    E = Path(cfg['racine'])
    rapport = Path(cfg.get('rapport') or E / 'BILAN.md')
    travail = Path(a.travail).resolve()
    journaux = journaux_arguments(a.journal)
    hors_sources(travail, [E, config, rapport, *journaux.values()])
    if a.ecrire and not a.commit:
        raise ValueError('--commit est obligatoire avec --ecrire')
    if (travail / 'analyse').exists():
        raise ValueError('analyse déjà présente : choisir un nouveau --travail pour exclure les verdicts périmés')
    original = rapport.read_bytes()
    avant, ancien, apres = decouper(original)
    travail.mkdir(parents=True, exist_ok=True)
    date = datetime.now()
    archive = travail / f"BILAN-AUTO_avant_{date:%Y%m%d-%H%M%S}.md"
    with archive.open('xb') as f:
        f.write(ancien)
    Path(str(archive) + '.sha256').write_text(empreinte(ancien) + '\n', encoding='utf-8')
    copie = travail / 'rapport_rejoue.md'
    copie.write_bytes(original)
    commandes = [
        [sys.executable, '-B', str(ICI / 'wp10_reconstat.py'), '--config', str(config), '--sortie', str(travail / 'reconstat')],
        [sys.executable, '-B', str(ICI / 'wp10_bilan.py'), '--config', str(config), '--constat', str(travail / 'reconstat/constat.json'),
         '--analyse-sortie', str(travail / 'analyse'), '--sortie', str(copie)],
    ]
    for machine, fichier in journaux.items():
        commandes[0] += ['--journal', f'{machine}={fichier.resolve()}']
    for commande in commandes:
        r = subprocess.run(commande, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise ValueError(f"commande échouée rc={r.returncode} : {commande}\n{r.stdout}\n{r.stderr}")
    reference = (E / 'analyse/verdicts.txt').read_bytes()
    verdicts = (travail / 'analyse/verdicts.txt').read_bytes()
    if verdicts != reference:
        raise ValueError('verdicts.txt différent : publication interdite')
    nouveau_rapport = copie.read_bytes()
    avant2, nouveau, apres2 = decouper(nouveau_rapport)
    if avant != avant2 or apres != apres2:
        raise ValueError('rapport modifié hors du bloc : publication interdite')
    diff = ''.join(difflib.unified_diff(ancien.decode('utf-8').splitlines(keepends=True),
                                      nouveau.decode('utf-8').splitlines(keepends=True),
                                      fromfile=str(archive), tofile=str(copie)))
    (travail / 'diff_bloc.txt').write_text(diff, encoding='utf-8')
    changements = [l for l in diff.splitlines() if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
    verdict = re.compile(r'^[+-](PASS|FAIL|SKIP)\s')
    outils = list(ICI.glob('wp10_*.py')) + list((E / 'outils').glob('*.py'))
    provenance = {
        'commandes': commandes, 'commit_correctif': a.commit,
        'sha256_outils': {str(p.resolve()): empreinte(p.read_bytes()) for p in sorted(set(outils))},
        'sha256_constat': empreinte((travail / 'reconstat/constat.json').read_bytes()),
        'sha256_extraits': {p.name: empreinte(p.read_bytes()) for p in sorted((travail / 'reconstat').glob('journal_noyau_*'))},
        'ancien_bloc': {'chemin': str(archive), 'sha256': empreinte(ancien)},
        'sha256_rapport_avant': empreinte(original), 'sha256_rapport_rejoue': empreinte(nouveau_rapport),
        'sha256_hors_bloc_avant': empreinte(avant + apres), 'sha256_hors_bloc_apres': empreinte(avant2 + apres2),
        'sha256_verdicts_reference': empreinte(reference), 'sha256_verdicts_rejoues': empreinte(verdicts),
    }
    lignes = [f"## Avenant WP10-SUITE du {date:%Y-%m-%d %H:%M:%S} — bilan rejoué après le défaut D1", '',
              f"D1 : la ligne 369 de wp10_orch.py omettait le contrôle GPU pour rc ≠ 0. Commit du correctif : {a.commit or '(non fourni : préparation)'}.",
              'Re-constat de chaque fenêtre, puis bilan avec constat et analyse figée rejouée dans un dossier distinct.',
              f"Ancien bloc conservé : `{archive}` ; sha256 `{empreinte(ancien)}`.",
              'Les autres verdicts statistiques sont reproduits à l’octet : verdicts.txt identique.', '',
              'Commandes et empreintes :', '', '```json', json.dumps(provenance, ensure_ascii=False, indent=2), '```', '',
              'Lignes du diff qui changent un verdict :', '', '```diff',
              *[l for l in changements if verdict.match(l)], '```', '',
              'Toutes les autres lignes modifiées (date, table, place libre, provenance, emplacements…) :', '', '```diff',
              *[l for l in changements if not verdict.match(l)], '```', '']
    avenant = '\n'.join(lignes).encode('utf-8')
    (travail / 'avenant.md').write_bytes(avenant)
    (travail / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if a.ecrire:
        if rapport.read_bytes() != original:
            raise ValueError('rapport modifié pendant le rejeu : publication interdite')
        part = Path(str(rapport) + '.part')
        with part.open('xb') as f:
            f.write(nouveau_rapport + b'\n\n' + avenant)
        os.replace(part, rapport)
    print(f"PASS WP10-BILAN-REJOUE verdicts.txt identique ; hors bloc identique ; publication={a.ecrire} ; {travail}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--travail', required=True)
    ap.add_argument('--ecrire', action='store_true')
    ap.add_argument('--commit')
    ap.add_argument('--journal', action='append', default=[])
    a = ap.parse_args(argv)
    try:
        rejouer(a)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'FAIL ECHEC_OUTIL WP10-BILAN-REJOUE {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
