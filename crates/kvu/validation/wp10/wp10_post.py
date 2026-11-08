# SPDX-License-Identifier: Apache-2.0
"""Post-traitement CPU, intégrité G-H13 et conservation WP10."""

import sys
sys.dont_write_bytecode = True
from pathlib import Path
# Le mode isolé -I n'ajoute pas le répertoire du script aux imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from wp10_stat import NB_BOUCLES, lire_csv, lire_q64, plaquette_q64, sha256_fichier, unitarite_q64

COLONNES = ('pas', 'l_grossier', 'n_boucles', 'fraction_hors', 'theta_max',
            'n_ambigues', 'plaq_grossiere', 'sha256_cfg')
MOTIF = re.compile(r'bloc3(b?)_cfg_(\d+)\.npy\Z')


class ErreurUsage(ValueError):
    """Arguments incompatibles avec les données de la tâche."""


class Parseur(argparse.ArgumentParser):
    def error(self, message):
        raise ErreurUsage(message)


def lire_json(chemin):
    with open(chemin, encoding='utf-8') as fichier:
        return json.load(fichier)


def date_iso():
    return datetime.now(timezone.utc).isoformat()


def ecrire_json(chemin, contenu):
    chemin = Path(chemin)
    part = Path(str(chemin) + '.part')
    try:
        part.write_text(json.dumps(contenu, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(part, chemin)
    finally:
        part.unlink(missing_ok=True)


def ecrire_csv(chemin, colonnes, lignes):
    part = Path(str(chemin) + '.part')
    try:
        with part.open('w', encoding='utf-8', newline='') as fichier:
            auteur = csv.DictWriter(fichier, fieldnames=colonnes)
            auteur.writeheader()
            for ligne in lignes:
                auteur.writerow({k: f'{v:.17e}' if isinstance(v, float) else v for k, v in ligne.items()})
        os.replace(part, chemin)
    finally:
        part.unlink(missing_ok=True)


def configurations(rep, deuxieme=False):
    prefixe = 'bloc3b' if deuxieme else 'bloc3'
    return sorted((p for p in rep.glob(prefixe + '_cfg_*.npy') if MOTIF.fullmatch(p.name)),
                  key=lambda p: int(MOTIF.fullmatch(p.name)[2]))


def nom_cfg(pas, deuxieme=False):
    return f"bloc3{'b' if deuxieme else ''}_cfg_{int(pas):08d}.npy"


def entier(valeur):
    nombre = float(valeur)
    if not math.isfinite(nombre) or not nombre.is_integer():
        raise ValueError(f'entier attendu : {valeur}')
    return int(nombre)


def metadonnees(chemin):
    return dict(nom=chemin.name, sha256=sha256_fichier(chemin), octets=chemin.stat().st_size)


def bloc2(args):
    rep = args.repertoire
    if not (rep / 'bloc3.csv').is_file():
        raise ErreurUsage('bloc3.csv absent')
    entrees = configurations(rep)
    lignes = lire_csv(rep / 'bloc3.csv')
    if len(entrees) != len(lignes):
        raise ErreurUsage('nombre de configurations différent du nombre de lignes de bloc3.csv')
    if [entier(r['pas']) for r in lignes] != [int(MOTIF.fullmatch(p.name)[2]) for p in entrees]:
        raise ErreurUsage('pas de bloc3.csv incompatibles avec les configurations')
    sorties, histogrammes, etiquettes = [], [], set()
    for entree in entrees:
        pas = int(MOTIF.fullmatch(entree.name)[2])
        sortie = rep / nom_cfg(pas, True)
        manifeste = Path(str(sortie) + '.manifeste.json')
        reutiliser = False
        if args.garder_existants and sortie.is_file() and manifeste.is_file():
            try:
                m = lire_json(manifeste)
                reutiliser = (sha256_fichier(sortie) == m['sha256_sortie']
                              and sha256_fichier(entree) == m['sha256_entree'])
            except (OSError, ValueError, KeyError):
                pass
        if not reutiliser:
            resultat = subprocess.run([str(args.kvu), 'rg', 'bloc', '--entree', str(entree),
                                       '--sortie', str(sortie), '--fils', str(args.fils),
                                       '--backend', 'cpu'], capture_output=True, text=True)
            if resultat.returncode:
                raise RuntimeError(f'blocage au pas {pas}, rc={resultat.returncode} : {resultat.stderr.strip()}')
        m = lire_json(manifeste)
        theta = m['theta']
        q, cote = lire_q64(sortie)
        hachage = sha256_fichier(sortie)
        if hachage != m['sha256_sortie'] or sha256_fichier(entree) != m['sha256_entree']:
            raise ValueError(f'hachage incohérent au pas {pas}')
        if cote != m['l_sortie'] or len(theta['histogramme']) != 64:
            raise ValueError(f'manifeste incohérent au pas {pas}')
        sorties.append(dict(pas=pas, l_grossier=cote,
                            n_boucles=theta['n_boucles'],
                            fraction_hors=float(theta['fraction_hors']), theta_max=float(theta['theta_max']),
                            n_ambigues=theta['n_ambigues'], plaq_grossiere=plaquette_q64(q), sha256_cfg=hachage))
        histogrammes.append(dict(pas=pas, **{f'h{i}': v for i, v in enumerate(theta['histogramme'])}))
        etiquettes.add(theta['etiquette'])
    ecrire_csv(rep / 'bloc3b.csv', COLONNES, sorties)
    ecrire_csv(rep / 'bloc3b_hist.csv', ('pas', *(f'h{i}' for i in range(64))), histogrammes)
    print(f"INFO WP10-BLOC2 {rep.name} {len(sorties)} configurations, fraction_hors max="
          f"{max((r['fraction_hors'] for r in sorties), default=0):.6g}, étiquette(s)={','.join(sorted(etiquettes))}")
    return 0


def denominateur_valide(ligne):
    """Total des boucles et compteurs du CSV, y compris après élagage."""
    try:
        cote = entier(ligne['l_grossier'])
        return (cote > 0 and entier(ligne['n_boucles']) == NB_BOUCLES * 4 * cote**4
                and entier(ligne['n_ambigues']) >= 0
                and 0 <= ligne['fraction_hors'] <= 1)
    except (ValueError, KeyError, TypeError, OverflowError):
        return False


def verifier_configuration(chemin, ligne=None):
    """Contrôles partagés par H13 et ses quatre contre-épreuves."""
    erreurs = []
    hachage = sha256_fichier(chemin)
    if ligne is not None and hachage != ligne['sha256_cfg']:
        erreurs.append('sha256')
    if ligne is not None and not denominateur_valide(ligne):
        erreurs.append('dénominateur')
    q, cote = lire_q64(chemin)
    if not np.isfinite(q).all():
        erreurs.append('non fini')
        return dict(erreurs=erreurs, unitarite=math.nan, ecart=None, sha256=hachage)
    unite = unitarite_q64(q)
    if not math.isfinite(unite) or unite > 1e-13:
        erreurs.append('unitarité')
    ecart = None
    if ligne is not None:
        ecart = abs(plaquette_q64(q) - ligne['plaq_grossiere'])
        if not math.isfinite(ecart) or ecart > 1e-12:
            erreurs.append('plaquette')
        if cote != ligne['l_grossier']:
            erreurs.append('côté')
    return dict(erreurs=erreurs, unitarite=unite, ecart=ecart, sha256=hachage)


def verifier_h13(rep):
    """Tout recalculer ; un retrait documenté dispense seulement du fichier absent."""
    erreurs, informations = [], []
    hachages, elagues = {}, {}
    n_sha = n_sha_ok = n_cfg = 0
    max_unite = max_ecart = 0.0

    def exiger(condition, detail):
        if not condition:
            erreurs.append(detail)

    def hachage(nom):
        chemin = rep / nom
        if chemin.is_file():
            if nom not in hachages:
                hachages[nom] = sha256_fichier(chemin)
            return hachages[nom]
        return elagues.get(nom, {}).get('sha256')

    def comparer_sha(nom, attendu):
        nonlocal n_sha, n_sha_ok
        n_sha += 1
        ok = hachage(nom) is not None and hachage(nom) == attendu
        n_sha_ok += int(ok)
        exiger(ok, f'sha256 : {nom}')

    def operation(description, fonction):
        try:
            return fonction()
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            erreurs.append(f'{description} : {exc}')
            return None

    exiger((rep / 'COMPLETE').is_file(), 'COMPLETE absent')
    if (rep / 'ELAGAGE.json').exists():
        journal = operation('élagage', lambda: lire_json(rep / 'ELAGAGE.json'))
        if journal is not None:
            for fiche in journal['fichiers']:
                nom = fiche['nom']
                exiger(bool(MOTIF.fullmatch(nom)), f'retrait interdit : {nom}')
                exiger(nom not in elagues, f'retrait dupliqué : {nom}')
                elagues[nom] = fiche
    regen = operation('regen_bloc3.json', lambda: lire_json(rep / 'regen_bloc3.json'))
    if regen is not None:
        for fiche in regen['fichiers']:
            nom = fiche['nom']
            chemin = rep / nom
            exiger(chemin.is_file() or nom in elagues, f'fichier absent : {nom}')
            taille = chemin.stat().st_size if chemin.is_file() else elagues.get(nom, {}).get('octets')
            exiger(taille == fiche['octets'], f'taille : {nom}')
            comparer_sha(nom, fiche['sha256'])
    config = operation('config_run.json', lambda: lire_json(rep / 'config_run.json'))
    attendus = None
    if config is not None:
        attendus = [config['therm'] + m * config['skip'] for m in range(1, config['nmeas'] + 1)]
    lignes_par_pas = {}
    for deuxieme in (False, True):
        nom_csv = 'bloc3b.csv' if deuxieme else 'bloc3.csv'
        fichiers = configurations(rep, deuxieme)
        if deuxieme and not (rep / nom_csv).exists():
            exiger(not fichiers, 'bloc3b.csv absent malgré des configurations du deuxième pas')
            informations.append('bloc3b absent')
            continue
        lignes = operation(nom_csv, lambda: lire_csv(rep / nom_csv))
        if lignes is None:
            continue
        pas_lus = operation(nom_csv + ' pas', lambda: [entier(r['pas']) for r in lignes])
        if pas_lus is None:
            continue
        exiger(pas_lus == attendus, f'{nom_csv} : nombre de lignes ou pas incorrects')
        table = {p: r for p, r in zip(pas_lus, lignes)}
        lignes_par_pas[deuxieme] = table
        for pas, ligne in zip(pas_lus, lignes):
            nom = nom_cfg(pas, deuxieme)
            exiger(denominateur_valide(ligne), f'{nom_csv} pas {pas} : dénominateur')
            exiger((rep / nom).is_file() or nom in elagues, f'configuration absente : {nom}')
            comparer_sha(nom, ligne['sha256_cfg'])
            if deuxieme:
                def verifier_manifeste():
                    m = lire_json(rep / (nom + '.manifeste.json'))
                    comparer_sha(nom, m['sha256_sortie'])
                    comparer_sha(nom_cfg(pas), m['sha256_entree'])
                operation(nom + ' manifeste', verifier_manifeste)
        noms_attendus = {nom_cfg(p, deuxieme) for p in pas_lus}
        for chemin in fichiers:
            exiger(chemin.name in noms_attendus, f'configuration sans ligne CSV : {chemin.name}')
            pas = int(MOTIF.fullmatch(chemin.name)[2])
            bilan = operation(chemin.name, lambda: verifier_configuration(chemin, table.get(pas)))
            n_cfg += 1
            if bilan:
                hachages[chemin.name] = bilan['sha256']
                max_unite = max(max_unite, bilan['unitarite'])
                if bilan['ecart'] is not None:
                    max_ecart = max(max_ecart, bilan['ecart'])
                erreurs.extend(f"{chemin.name} : {e}" for e in bilan['erreurs'])
    mesures = operation('measurements.csv', lambda: lire_csv(rep / 'measurements.csv'))
    if mesures is not None and config is not None:
        exiger(len(mesures) == config['nmeas'] + 1, 'trajectoire : nombre de lignes incorrect')
        for indice, ligne in enumerate(mesures):
            exiger(ligne.get('measurement_index') == indice, 'trajectoire : indice incorrect')
            exiger(ligne.get('total_steps') == config['therm'] + indice * config['skip'], 'trajectoire : pas incorrect')
            exiger(ligne.get('phase') == ('initial' if indice == 0 else 'sample'), 'trajectoire : phase incorrecte')
            for cle in ('measurement_index', 'total_steps', 'plaquette', 'P0', 'P1', 'P2', 'P3'):
                valeur = ligne.get(cle, math.nan)
                exiger(math.isfinite(valeur), f'trajectoire : {cle} non fini')
                if cle not in ('measurement_index', 'total_steps'):
                    exiger(-1 - 1e-5 <= valeur <= 1 + 1e-5, f'trajectoire : {cle} hors bornes')
    return dict(ok=not erreurs, erreurs=erreurs, informations=informations, n_cfg=n_cfg,
                max_unitarite=max_unite, max_ecart_plaquette=max_ecart,
                sha256_ok=n_sha_ok, sha256_total=n_sha,
                elagues=sum(not (rep / nom).exists() for nom in elagues))


def mutants_h13(rep):
    lignes = lire_csv(rep / 'bloc3.csv')
    disponibles = [r for r in lignes if (rep / nom_cfg(r['pas'])).is_file()][:2]
    if len(disponibles) < 2:
        raise ValueError('deux configurations conservées sont nécessaires aux mutants H13')
    resultats = []
    with tempfile.TemporaryDirectory(prefix='wp10-mut-') as temporaire:
        copie = Path(temporaire)
        for ligne in disponibles:
            shutil.copy2(rep / nom_cfg(ligne['pas']), copie)
        ecrire_csv(copie / 'bloc3.csv', COLONNES, disponibles)
        for mutant, controle in (('m1', 'sha256'), ('m2', 'unitarité'),
                                 ('m3', 'plaquette'), ('m4', 'dénominateur')):
            ligne = dict(disponibles[0 if mutant != 'm3' else 1])
            chemin = copie / nom_cfg(ligne['pas'])
            shutil.copy2(rep / chemin.name, chemin)
            if mutant == 'm1':
                with chemin.open('r+b') as fichier:
                    fichier.seek(-1, 2)
                    octet = fichier.read(1)[0]
                    fichier.seek(-1, 2)
                    fichier.write(bytes([octet ^ 1]))
            elif mutant == 'm2':
                q, _ = lire_q64(chemin)
                q[0, 0, 0, 0, 0] *= 1 + 1e-9
                np.save(chemin, q)
                ligne['sha256_cfg'] = sha256_fichier(chemin)
            elif mutant == 'm3':
                ligne['plaq_grossiere'] += 1e-9
            elif mutant == 'm4':
                ligne['n_boucles'] -= 1
            indice = 1 if mutant == 'm3' else 0
            lignes_copie = [dict(r) for r in disponibles]
            lignes_copie[indice] = ligne
            ecrire_csv(copie / 'bloc3.csv', COLONNES, lignes_copie)
            ligne = lire_csv(copie / 'bloc3.csv')[indice]
            erreurs = verifier_configuration(chemin, ligne)['erreurs']
            rouge = controle in erreurs if mutant == 'm1' else erreurs == [controle]
            resultats.append(dict(mutant=mutant, ok=rouge, controle=controle, erreurs=erreurs))
    return resultats


def h13(args):
    rep = args.repertoire
    marqueur = rep / 'H13_PASS'
    # Un échec ou une interruption ne doit pas laisser une ancienne autorisation.
    marqueur.unlink(missing_ok=True)
    bilan = verifier_h13(rep)
    for info in bilan['informations']:
        print(f'INFO WP10-H13 {rep.name} {info}')
    for erreur in bilan['erreurs']:
        print(f'FAIL WP10-H13 {rep.name} {erreur}')
    print(f"{'PASS' if bilan['ok'] else 'FAIL'} WP10-H13 {rep.name} {bilan['n_cfg']} configurations, "
          f"max unitarité={bilan['max_unitarite']:.6g}, max écart plaquette={bilan['max_ecart_plaquette']:.6g}, "
          f"sha256 : {bilan['sha256_ok']}/{bilan['sha256_total']}, élagués : {bilan['elagues']}")
    if args.mutants:
        bilan['mutants'] = mutants_h13(rep)
        for mutant in bilan['mutants']:
            print(f"{'PASS' if mutant['ok'] else 'FAIL'} WP10-H13-MUT-{mutant['mutant']} {rep.name} "
                  f"{mutant['controle']} {'ROUGE' if mutant['ok'] else 'ECHEC_OUTIL'}")
        bilan['ok'] &= all(m['ok'] for m in bilan['mutants'])
    if args.json:
        ecrire_json(args.json, bilan)
    if bilan['ok']:
        marqueur.write_text(date_iso() + '\n', encoding='utf-8')
    return 0 if bilan['ok'] else 1


def manifeste(args):
    rep = args.repertoire
    original = rep / 'regen_bloc3.json'
    source = lire_json(original)
    sorties = configurations(rep, True)
    if not sorties or not (rep / 'bloc3b.csv').is_file() or not (rep / 'bloc3b_hist.csv').is_file():
        raise ErreurUsage('sorties du deuxième pas absentes ou incomplètes')
    cles = ('parametres', 'machine', 'graines', 'backend', 'arch', 'versions', 'git_head',
            'source_hash', 'noyaux_embarques', 'modules_compiles')
    m = {cle: source[cle] for cle in cles}
    hachage = sha256_fichier(args.kvu)
    # Les manifestes restent conservés même si leurs configurations ont été élaguées.
    fichiers = sorties + sorted(rep.glob('bloc3b_cfg_*.npy.manifeste.json'))
    fichiers += [rep / 'bloc3b.csv', rep / 'bloc3b_hist.csv']
    m.update(schema='kvu-regen/1', origine=dict(schema='kvu-regen/1', manifeste=str(original),
                                             sha256_manifeste=sha256_fichier(original)),
             binaire_sha256=hachage,
             binaire_provenance='manifeste' if hachage == source['binaire_sha256'] else 'inconnu',
             commande=['rg', 'bloc', '--entree', '<bloc3_cfg_*.npy>', '--sortie', '<bloc3b_cfg_*.npy>', '--backend', 'cpu'],
             entrees=[dict(nom=p.name, sha256=sha256_fichier(p)) for p in configurations(rep)],
             fichiers=[metadonnees(p) for p in sorted(fichiers)], fichiers_hachage='disque',
             duree_s=None, statut='COMPLETE', classe_regen='C',
             raisons_classe=["sortie dérivée hors ligne d'un run natif classé C ; rejeu REGEN-2 non fait"],
             regen_verifiee=dict(date=None, machine=None, resultat='non_verifiee'), ecrit_le=date_iso())
    destination = rep / 'regen_bloc3b.json'
    part = Path(str(destination) + '.part')
    try:
        part.write_text(json.dumps(m, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        resultat = subprocess.run([sys.executable, '-I', str(args.depot / 'scripts/regen_valider.py'), str(part)],
                                  capture_output=True, text=True)
        if resultat.returncode:
            print(f'FAIL WP10-MANIFESTE {rep.name} validation refusée : {resultat.stdout.strip()} {resultat.stderr.strip()}')
            return 1
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)
    print(f'PASS WP10-MANIFESTE {rep.name} kvu-regen/1 validé')
    return 0


def elaguer(args):
    rep = args.repertoire
    config = lire_json(rep / 'config_run.json')
    if config['skip'] <= 0:
        raise ErreurUsage('skip doit être strictement positif')
    if args.execute and not (rep / 'H13_PASS').is_file():
        raise ErreurUsage('H13_PASS absent : exécuter h13 avec succès avant élagage')
    candidats = []
    for p in sorted(configurations(rep) + configurations(rep, True)):
        pas = int(MOTIF.fullmatch(p.name)[2])
        delta = pas - config['therm']
        if delta <= 0 or delta % config['skip']:
            raise ErreurUsage(f'pas incompatible : {p.name}')
        if (delta // config['skip']) % args.un_sur:
            candidats.append(metadonnees(p))
    retires = {f['nom'] for f in candidats}
    indices_gardes = sorted({(int(MOTIF.fullmatch(p.name)[2]) - config['therm']) // config['skip']
                            for p in configurations(rep) + configurations(rep, True)
                            if p.name not in retires})
    journal = dict(date=date_iso(), N=args.un_sur, un_sur=args.un_sur,
                   indices_gardes=indices_gardes, fichiers=candidats)
    if args.execute:
        # Recalculer avant toute suppression protège contre un marqueur périmé.
        if not verifier_h13(rep)['ok']:
            raise ErreurUsage('H13_PASS périmé : intégrité désormais en échec')
        if (rep / 'ELAGAGE.json').exists():
            precedents = lire_json(rep / 'ELAGAGE.json')['fichiers']
            journal['fichiers'] = sorted({f['nom']: f for f in precedents + candidats}.values(), key=lambda f: f['nom'])
        ecrire_json(rep / 'ELAGAGE.json', journal)
        for fiche in candidats:
            (rep / fiche['nom']).unlink()
    else:
        ecrire_json(rep / 'ELAGAGE_PROPOSE.json', journal)
    print(f"INFO WP10-ELAGAGE {rep.name} {len(candidats)} fichiers {'retirés' if args.execute else 'proposés'}, un sur {args.un_sur}")
    return 0


def positif(texte):
    valeur = int(texte)
    if valeur < 1:
        raise argparse.ArgumentTypeError('entier strictement positif attendu')
    return valeur


def main(argv=None):
    try:
        p = Parseur(description=__doc__)
        sous = p.add_subparsers(dest='commande', required=True, parser_class=Parseur)
        b = sous.add_parser('bloc2')
        b.add_argument('--kvu', required=True, type=Path)
        b.add_argument('--fils', type=positif, default=1)
        b.add_argument('--garder-existants', action='store_true')
        h = sous.add_parser('h13')
        h.add_argument('--mutants', action='store_true')
        h.add_argument('--json', type=Path)
        m = sous.add_parser('manifeste')
        m.add_argument('--kvu', required=True, type=Path)
        # Racine effective du dépôt (le script est dans crates/kvu/validation/wp10).
        m.add_argument('--depot', type=Path, default=Path(__file__).resolve().parents[4])
        e = sous.add_parser('elaguer')
        e.add_argument('--un-sur', type=positif, required=True)
        e.add_argument('--execute', action='store_true')
        for parseur in (b, h, m, e):
            parseur.add_argument('repertoire', type=Path)
        args = p.parse_args(argv)
        return dict(bloc2=bloc2, h13=h13, manifeste=manifeste, elaguer=elaguer)[args.commande](args)
    except ErreurUsage as exc:
        print(f'INFO WP10-USAGE erreur : {exc}')
        return 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as exc:
        print(f'FAIL WP10-POST erreur : {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
