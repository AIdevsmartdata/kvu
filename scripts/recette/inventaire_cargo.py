# SPDX-License-Identifier: Apache-2.0
"""inventaire_cargo.py : inventaire DYNAMIQUE des tests, pris sur la sortie de cargo (gate T4-INTEGRITE-CARGO de la recette de fusion, lot C).

Pourquoi. L'inventaire STATIQUE (inventaire.py) lit les sources avec git seul. Il ne voit pas ce qui désactive des tests SANS toucher à un test : un attribut posé sur l'invocation d'une
macro génératrice (`#[cfg(any())] cas!(x);`), sur une déclaration de module externe (`#[cfg(any())] mod x;`), un `include!` retiré, une macro définie dans un AUTRE fichier dont le
gabarit perd son `#[test]`, un `harness = false` ou un `[lib] test = false` de Cargo.toml. Ici l'inventaire est celui que CARGO compte : ce que chaque exécutable de test contient (`cargo test -- --list`),
ce qui est ignoré (`-- --list --ignored`) et ce qui a été exécuté (la sortie de `cargo test`, authentifiée par le lanceur). Il vient EN PLUS de l'inventaire statique, qui reste utile (il tourne dès l'étape
candidat, sans compilation) ; rien n'est retiré à celui-ci.

Modèle. Une SUITE est {cible: {nom: état}} : la cible est l'exécutable de test tel que cargo le désigne (`unittests src/lib.rs`, `tests/boite_pilote.rs`, `doc-tests kvu`), le nom est le nom
qualifié que libtest imprime (`cli::bench::tests::cible_par_cas_bornee`), l'état est `actif` (compilé, non ignoré : cargo l'exécute) ou `ignore` (compilé, `#[ignore]` : listé mais jamais exécuté).
Comparer une base et un candidat (`comparer`) dit ce que le candidat a RETIRÉ de la suite compilée ou DÉSACTIVÉ : un test de la base absent du candidat, ou actif dans la base et ignoré dans le candidat.
Un test AJOUTÉ, ou un test ignoré qui redevient actif, n'est jamais une raison (même règle que l'inventaire statique). Le contenu d'un test n'est pas jugé (corps vidé, `#[should_panic]` : c'est le rôle de
l'inventaire statique et de la relecture).

Trois sources, de confiance décroissante, jamais mêlées sans contrôle :
  `--list` et `--list --ignored`  sortie de libtest elle-même, une ligne `nom: test` par test et, par exécutable, un RÉSUMÉ « N tests, M benchmarks » : le nombre de lignes doit concorder, sinon la sortie est
                                  refusée (jamais lue de travers). C'est la source de la suite du candidat.
  la sortie de `cargo test`       lignes `test nom ... ok|FAILED|ignored`. Elle est LUE AVEC TOLÉRANCE : des tests de kvu relancent leur propre exécutable en enfant avec un stdout hérité, et les lignes des
                                  enfants s'entrelacent avec celles du parent (`test enfant ... test autre ... ok`). On ne retient donc que les NOMS (`test <nom> ... ` est écrit d'un seul bloc) ; elle ne sert qu'à
                                  CONFIRMER que chaque test actif de la liste a été exécuté (`anomalies_execution`), jamais à définir la suite.
  une référence versionnée        la suite de la base (`ecrire_reference` / `lire_reference`), décision du pilote ou tirée d'une campagne authentifiée de la base.
"""

import re


class SortieIncoherente(ValueError):
    """La sortie de cargo n'a pas la forme attendue (section sans résumé, comptes qui ne concordent pas, nom en double, exécutable inconnu...) : la gate échoue FERMÉE, avec ce motif."""


ENTETE_EXECUTABLE = re.compile(r"^\s*Running (?P<cible>(?:unittests )?\S+\.rs) \(.*\)\s*$")   # le chemin d'un exécutable de test se termine par .rs : un texte « Running … (…) » écrit par un test n'ouvre pas de section
ENTETE_DOCTESTS = re.compile(r"^\s*Doc-tests (?P<crate>\S+)\s*$")
LIGNE_LISTE = re.compile(r"^(?P<nom>.+): (?P<genre>test|benchmark)$")
RESUME_LISTE = re.compile(r"^(?P<tests>\d+) tests?, (?P<bancs>\d+) benchmarks?$")
# `test <nom> ... ` : écrit d'un seul bloc par libtest ; un test `#[should_panic]` s'imprime `test <nom> - should panic ... ok` (et un doctest `- compile fail`, `- compile`) ;
# le nom d'un doctest peut être rembourré d'espaces. (?<!\w) : le mot « test » n'est pas la fin d'un autre mot ; il peut suivre une espace OU un texte d'enfant sans saut de ligne (`error:test x ... ok`, vu dans une vraie sortie), y compris au milieu d'une ligne.
NOM_VU = re.compile(r"(?<!\w)test (?P<nom>\S.*?)(?: - (?:should panic|compile fail|compile))? +\.\.\.(?=\s|$)(?: +(?P<etat>ok|FAILED|ignored)\b)?")
LIGNE_DOCTEST = re.compile(r"^(?P<base>.*) \(line (?P<ligne>\d+)\)$")
ENTETE_REFERENCE = "# kvu-noms-tests/1"


def _cible_de(ligne):
    """Cible annoncée par une ligne d'en-tête de cargo (`Running unittests src/lib.rs (…)`, `Doc-tests kvu`) ; None si ce n'en est pas une."""
    m = ENTETE_EXECUTABLE.match(ligne)
    if m:
        return m.group("cible")
    m = ENTETE_DOCTESTS.match(ligne)
    return f"doc-tests {m.group('crate')}" if m else None


def _normaliser(cible, noms):
    """{nom d'origine: nom normalisé} des noms d'une cible. Un doctest s'appelle `src/lib.rs - item (line 12)` : le numéro de ligne bouge à chaque édition du fichier, on l'ôte et on numérote
    les homonymes (`#2`, `#3`, dans l'ordre numérique des lignes) ; les autres noms sont rendus tels quels."""
    if not cible.startswith("doc-tests"):
        return {nom: nom for nom in noms}
    triables = []
    for nom in noms:
        m = LIGNE_DOCTEST.match(nom)
        triables.append(((m.group("base"), int(m.group("ligne"))) if m else (nom, 0), nom))
    table, vus = {}, {}
    for (base, _ligne), nom in sorted(triables):
        vus[base] = vus.get(base, 0) + 1
        table[nom] = base if vus[base] == 1 else f"{base} #{vus[base]}"
    return table


def parser_liste(texte):
    """{cible: [noms triés]} d'une sortie `cargo test -- --list` (format par défaut, résumés compris). SortieIncoherente si une section n'a pas son résumé « N tests, M benchmarks », si les comptes
    ne concordent pas, si un nom ou une cible est en double, si une ligne de test tombe hors d'une section, ou s'il n'y a aucune section (cargo n'a rien listé : jamais « suite vide = rien à perdre »)."""
    resultat, courant, noms, bancs = {}, None, [], 0
    for ligne in texte.splitlines():
        cible = _cible_de(ligne)
        if cible is not None:
            if courant is not None:
                raise SortieIncoherente(f"la section « {courant} » n'a pas de résumé « N tests, M benchmarks » (sortie tronquée, ou format terse)")
            if cible in resultat:
                raise SortieIncoherente(f"la cible « {cible} » apparaît deux fois")
            courant, noms, bancs = cible, [], 0
            continue
        r = RESUME_LISTE.match(ligne)
        if r and courant is not None:
            if int(r.group("tests")) != len(noms) or int(r.group("bancs")) != bancs:
                raise SortieIncoherente(f"« {courant} » : le résumé annonce {r.group('tests')} tests et {r.group('bancs')} benchmarks, {len(noms)} et {bancs} lignes lues")
            if len(set(noms)) != len(noms):
                raise SortieIncoherente(f"« {courant} » : nom de test en double")
            resultat[courant] = sorted(_normaliser(courant, noms).values())
            courant = None
            continue
        m = LIGNE_LISTE.match(ligne)
        if m:
            if courant is None:
                raise SortieIncoherente(f"ligne de test « {ligne[:80]} » hors de toute section d'exécutable")
            if m.group("genre") == "benchmark":
                bancs += 1
            else:
                noms.append(m.group("nom"))
    if courant is not None:
        raise SortieIncoherente(f"la section « {courant} » n'a pas de résumé « N tests, M benchmarks » (sortie tronquée, ou format terse)")
    if not resultat:
        raise SortieIncoherente("aucune section d'exécutable de test dans la sortie de cargo (rien n'a été listé)")
    return resultat


def parser_execution(texte):
    """({cible: {noms vus}}, {cible: {noms vus « ignored »}}) d'une sortie `cargo test` : tout `test <nom> ... ` écrit dans la section d'un exécutable, y compris au milieu d'une ligne entrelacée
    par un processus enfant. Un nom est « vu » dès que son préfixe est écrit (le statut peut avoir été séparé de lui par la sortie d'un enfant) ; le statut `ok`/`FAILED` n'est PAS jugé ici :
    c'est le rôle de T4-TESTS (rc 0, 0 échec, passés >= NMIN). Les cibles sans test figurent avec un ensemble vide."""
    vus, ignores, courant, bruts, bruts_ignores = {}, {}, None, {}, {}
    for ligne in texte.splitlines():
        cible = _cible_de(ligne)
        if cible is not None:
            courant = cible
            bruts.setdefault(cible, set())
            bruts_ignores.setdefault(cible, set())
            continue
        if courant is None:
            continue
        for m in NOM_VU.finditer(ligne):
            bruts[courant].add(m.group("nom").rstrip())
            if m.group("etat") == "ignored":
                bruts_ignores[courant].add(m.group("nom").rstrip())
    for cible, noms in bruts.items():
        table = _normaliser(cible, noms)
        vus[cible] = set(table.values())
        ignores[cible] = {table[n] for n in bruts_ignores[cible]}
    return vus, ignores


def suite_de(liste, ignores):
    """Suite {cible: {nom: actif|ignore}} d'après `parser_liste(--list)` et `parser_liste(--list --ignored)` : les deux sorties doivent porter les mêmes exécutables, et tout test ignoré doit être listé.
    Les exécutables sans test n'y figurent pas."""
    if set(liste) != set(ignores):
        raise SortieIncoherente("--list et --list --ignored ne portent pas les mêmes exécutables de test : " + ", ".join(sorted(set(liste) ^ set(ignores))))
    suite = {}
    for cible, noms in liste.items():
        a_ignorer = set(ignores[cible])
        inconnus = a_ignorer - set(noms)
        if inconnus:
            raise SortieIncoherente(f"« {cible} » : test(s) ignoré(s) absents de la liste : {', '.join(sorted(inconnus)[:3])}")
        if noms:   # un exécutable sans test (le binaire `kvu`, les doctests d'un crate qui n'en a pas) ne porte rien à comparer : il ne figure pas dans la suite
            suite[cible] = {nom: ("ignore" if nom in a_ignorer else "actif") for nom in noms}
    return suite


def suite_depuis_execution(texte):
    """Suite d'après la SEULE sortie d'exécution de cargo : `ignored` -> ignore, tout autre test vu -> actif. Moins fiable que `suite_de` (la sortie peut être entrelacée) : réservé au contrôle
    croisé d'une référence contre un journal ancien."""
    vus, ignores = parser_execution(texte)
    return {cible: {nom: ("ignore" if nom in ignores[cible] else "actif") for nom in sorted(noms)} for cible, noms in vus.items() if noms}


def actifs(suite):
    """{(cible, nom)} des tests actifs d'une suite."""
    return {(cible, nom) for cible, tests in suite.items() for nom, etat in tests.items() if etat == "actif"}


def resume(suite):
    """(nombre de tests, nombre d'actifs, nombre d'ignorés, nombre d'exécutables) d'une suite."""
    total = sum(len(t) for t in suite.values())
    nb_actifs = len(actifs(suite))
    return total, nb_actifs, total - nb_actifs, len(suite)


def anomalies_execution(suite, vus):
    """[(cible, nom, motif)] : incohérences entre la suite listée du candidat et sa sortie d'exécution authentifiée. (1) un test ACTIF de la liste dont le nom n'a jamais été écrit par cargo : il n'a pas été
    exécuté ; (2) un nom écrit par cargo qui n'est pas dans la liste (sortie falsifiée ou entrelacée par autre chose qu'un test de la suite). Ces anomalies ne s'approuvent pas : elles disent que les
    preuves ne concordent pas."""
    sortie = []
    for cible, tests in sorted(suite.items()):
        for nom, etat in sorted(tests.items()):
            if etat == "actif" and nom not in vus.get(cible, ()):
                sortie.append((cible, nom, "listé actif mais absent de la sortie d'exécution"))
    for cible, noms in sorted(vus.items()):
        for nom in sorted(noms):
            if nom not in suite.get(cible, {}):
                sortie.append((cible, nom, "écrit par cargo à l'exécution mais absent de la liste"))
    return sortie


def comparer(base, candidat):
    """[(cible, nom, motif)] : ce que `candidat` a RETIRÉ ou DÉSACTIVÉ par rapport à `base`. Vide si le candidat ne fait qu'ajouter ou réactiver. Motifs : « retiré de la suite compilée »
    (le test n'est plus compilé : `#[cfg]`, module ou macro désactivés, fichier retiré, renommage...), « cible disparue » (tout l'exécutable a disparu), « devenu ignoré » (actif dans la base)."""
    sortie = []
    for cible, tests in sorted(base.items()):
        c = candidat.get(cible)
        for nom, etat in sorted(tests.items()):
            if c is None:
                sortie.append((cible, nom, "retiré de la suite compilée (cible disparue)"))
            elif nom not in c:
                sortie.append((cible, nom, "retiré de la suite compilée"))
            elif etat == "actif" and c[nom] == "ignore":
                sortie.append((cible, nom, "devenu ignoré"))
    return sortie


def evaluer(base, candidat, vus, approuves, sha):
    """(anomalies, non_approuves, nb_approuves) : le jugement de la gate sur une suite de base, la suite listée du candidat et ses noms vus à l'exécution. `approuves` = {(jeton, sha)} lus dans le fichier
    des approbations du pilote : un retrait ou une désactivation n'est levé que par la ligne `<jeton du test> <sha du commit candidat>` (une approbation ne couvre QUE le commit dont elle porte le sha).
    Les anomalies d'exécution ne se lèvent pas."""
    anomalies = anomalies_execution(candidat, vus)
    retraits = comparer(base, candidat)
    non_approuves = [r for r in retraits if (jeton(r[0], r[1]), sha) not in approuves]
    return anomalies, non_approuves, len(retraits) - len(non_approuves)


def jeton(cible, nom):
    """Identifiant d'un test dans le fichier des approbations du pilote (UN mot, sans espace : le fichier est lu par champs) : `cargo:<cible>::<nom>`, `%` et espaces codés en %25 et %20."""
    codé = (cible + "::" + nom).replace("%", "%25").replace(" ", "%20").replace("\t", "%09")
    return "cargo:" + codé


def formater(raisons, limite=4):
    """Texte court de raisons [(cible, nom, motif)] : les `limite` premières, puis le nombre des autres."""
    morceaux = [f"{cible}::{nom} ({motif})" for cible, nom, motif in raisons[:limite]]
    return " | ".join(morceaux) + (f" (+{len(raisons) - limite} autre(s))" if len(raisons) > limite else "")


# ---------------------------------------------------------------- référence versionnée
def ecrire_reference(suite, commit, etape, source):
    """Texte canonique de la suite d'une base : en-tête (commit, étape, provenance, comptes), puis une ligne `cible<TAB>nom<TAB>actif|ignore` par test, triée. Les exécutables sans test sont omis."""
    total, nb_actifs, nb_ignores, _ = resume({c: t for c, t in suite.items() if t})
    cibles = sorted(c for c, t in suite.items() if t)
    lignes = [ENTETE_REFERENCE, f"# commit {commit}", f"# etape {etape}", "# source " + " ".join(source.split()),
              f"# tests {total} actifs {nb_actifs} ignores {nb_ignores} cibles {len(cibles)}"]
    for cible in cibles:
        for nom, etat in sorted(suite[cible].items()):
            if any(ch in cible + nom for ch in "\t\n\r"):
                raise ValueError(f"tabulation ou saut de ligne dans « {cible} » / « {nom} »")
            lignes.append(f"{cible}\t{nom}\t{etat}")
    return "\n".join(lignes) + "\n"


def lire_reference(texte, commit=None, etape=None):
    """(suite, méta) d'une référence écrite par `ecrire_reference`. SortieIncoherente si l'en-tête n'est pas celui du format, si `commit` ou `etape` (quand ils sont donnés) ne sont pas ceux du fichier,
    si une ligne est mal formée ou en double, ou si les comptes de l'en-tête ne sont pas ceux des lignes (fichier tronqué ou édité à la main)."""
    lignes = texte.split("\n")
    if lignes and lignes[-1] == "":
        lignes.pop()
    if not lignes or lignes[0] != ENTETE_REFERENCE:
        raise SortieIncoherente("référence de tests : en-tête « " + ENTETE_REFERENCE + " » absent")
    meta, suite = {}, {}
    for ligne in lignes[1:]:
        if ligne.startswith("# "):
            cle, _, valeur = ligne[2:].partition(" ")
            meta[cle] = valeur
            continue
        champs = ligne.split("\t")
        if len(champs) != 3 or champs[2] not in ("actif", "ignore") or not champs[0] or not champs[1]:
            raise SortieIncoherente(f"référence de tests : ligne mal formée « {ligne[:80]} »")
        cible, nom, etat = champs
        if nom in suite.setdefault(cible, {}):
            raise SortieIncoherente(f"référence de tests : « {cible} » / « {nom} » en double")
        suite[cible][nom] = etat
    for cle, attendu in (("commit", commit), ("etape", etape)):
        if attendu is not None and meta.get(cle) != attendu:
            raise SortieIncoherente(f"référence de tests : {cle} « {meta.get(cle)} » au lieu de « {attendu} »")
    total, nb_actifs, nb_ignores, nb_cibles = resume(suite)
    if total == 0:
        raise SortieIncoherente("référence de tests : aucun test (une base sans test ne protège rien : la gate ne compare plus rien)")
    if meta.get("tests") != f"{total} actifs {nb_actifs} ignores {nb_ignores} cibles {nb_cibles}":
        raise SortieIncoherente(f"référence de tests : l'en-tête annonce « {meta.get('tests')} », les lignes en portent {total} actifs {nb_actifs} ignores {nb_ignores} cibles {nb_cibles}")
    return suite, meta
