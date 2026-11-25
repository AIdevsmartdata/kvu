<!-- SPDX-License-Identifier: Apache-2.0 -->
# `kvu experience` : préenregistrements, pièces et carnet

Le manifeste `EXPERIENCE.json` décrit une expérience documentaire, avec le
schéma `kvu-experience/1`. L'outil lit des fichiers existants ; les textes de
question, de verdict, de réserve et de proposition restent des déclarations.
Il ne lance aucun calcul. `verifier` est en lecture seule et rend 8 dès qu'un
contrôle est ROUGE ; une écriture refusée par une règle rend aussi 8. Une
erreur d'usage rend 2 ; les erreurs d'entrée/sortie gardent le code commun 4.

## Schéma commenté

Le schéma JSON est embarqué dans le binaire et recopié à l'identique dans
`schemas/kvu-experience-1.schema.json`. Il utilise le sous-ensemble de JSON
Schema validé par `regen::schema::valider`. `valider_experience` complète les
contraintes entre type de proposition, statut et préfixe d'identifiant.
Les champs inconnus sont refusés dans les structures fermées du manifeste.

| Champs | Contenu et provenance |
| --- | --- |
| `schema`, `id`, `titre`, `question`, `carnet_id` | Identité ; `carnet_id` commence par `EXP-`. |
| `chemin_projet` | Racine de l'expérience relative à la racine du projet utilisée par le carnet ; défaut `.`. |
| `machine_principale` | Nom déclaré à `init`, avec le profil principal du dépôt comme défaut. |
| `prereg` | Au moins P0 : chemin, sha256, `gele_le`, `gele_source`, `change_un_critere`, `remplace`, sections éventuelles. |
| `anteriorite` | Pièces optionnelles `garde` et `sha256sums`, chacune `{chemin, sha256}`. |
| `code` | Commit et branche déclarés ; `binaires` dédoublonnés à partir des runs. |
| `machines` | Hôte, backend et architecture lus ; classe A si le nom est celui de la machine principale, B sinon. |
| `runs` | Références des manifestes et sorties, empreintes, dates, rc, graines, géométrie, paramètres et ids des PREREG. |
| `graines` | Règle textuelle optionnelle et `toutes_distinctes` recalculé. |
| `portes` | Id neuf, énoncé, seuil, valeur, statut, source hachée et mutant éventuel. |
| `verdict` | Statut déclaré, caractère provisoire, classe calculée, texte, chiffres sourcés et historique des états remplacés. |
| `carnet` | Propositions typées avec id, titre, statut restreint, énoncé, sources, liens, date et `tomberait` éventuel. |
| `reserves`, `incidents`, `donnees`, `index`, `auteurs` | Contexte ; les relecteurs peuvent figurer dans `auteurs.relecteurs`. |
| `horodatage` | `cree_le` et `modifie_le` en UTC. |

`hachage_consigne` reste un chemin optionnel du PREREG. Son empreinte propre
est enregistrée dans `hachage_consigne_sha256` : ce champ supplémentaire
permet de vérifier le fichier de consignation lui-même, et pas seulement son
premier mot. Ces deux champs doivent être présents ensemble. Le premier mot
du fichier doit être le sha256 du PREREG.

Tous les chemins enregistrés sont relatifs à l'expérience, avec `/`, sans
`..` ; les liens symboliques sortant de cette racine sont refusés. Un chemin
de fichier absolu passé en argument peut être accepté s'il désigne une pièce
à l'intérieur de la racine ; il est enregistré sous forme relative.
Les identifiants utilisent lettres ASCII, chiffres, points, tirets et `_`.

Pour une géométrie anisotrope, `cotes` suit l'ordre `[Lt, Ls, Ls, Ls]` ;
`--L` donne quatre côtés égaux. Les côtés ne sont pas répétés dans `parametres`.

Les dates acceptent `YYYY-MM-DDTHH:MM:SSZ` et un décalage `±hh:mm` en lecture.
Les écritures sont normalisées en UTC, à la seconde. Les secondes Unix des
runs peuvent être fractionnaires ; leur partie entière est conservée. Les
dates représentables vont de 1970 à 9999. L'horloge ne sert qu'aux champs
`horodatage`, à `historique.remplace_le` et à la date d'une proposition ;
`KVU_EXPERIENCE_HORLOGE` permet de les fixer pour les tests.

## Exemple de bout en bout

On suppose un dossier `experience-demo` contenant un `PREREG.md` déjà rédigé,
et `runs/r0.manifest.json`, un manifeste `kevinotron-unifie/1` terminé après
le gel. L'outil ne fabrique ni ce PREREG ni les résultats de simulation.

```sh
kvu experience init --racine experience-demo \
  --id demo --titre "Contrôle de l'observable" \
  --question "Le témoin satisfait-il le seuil gelé ?" \
  --prereg PREREG.md --gele-le 2026-11-11T10:00:00Z \
  --machine-principale principale --chemin-projet chantiers/experience-demo \
  --commit revision-declaree --branche travail

kvu experience ajouter-run --racine experience-demo \
  --machine principale --prereg P0 runs/r0.manifest.json

kvu experience porte G0 --racine experience-demo \
  --enonce "Contrôle documentaire du témoin" --seuil 1 --valeur 0 \
  --statut vert --source PREREG.md

kvu experience verdict --racine experience-demo \
  --statut compatible --provisoire --texte "Compatible sur le témoin lu." \
  --chiffre 'seuil=1@PREREG.md:2:[lecture]' \
  --reserve "Une seule famille de calcul ; portée limitée."

kvu experience proposer resultat RES-demo --racine experience-demo \
  --titre "Lecture du témoin" --statut numerique \
  --enonce "Observation limitée aux pièces citées." \
  --source PREREG.md:1-2 --lien EXP-demo \
  --tomberait "Si les pièces ou le critère gelé sont invalides."

kvu experience verifier --racine experience-demo
kvu experience verifier --racine experience-demo --json
kvu experience fiche --racine experience-demo
kvu experience carnet --racine experience-demo --vers copie-carnet
```

Sans `--gele-le`, `gele_le` est le **mtime** du fichier de hachage consigné
s'il est fourni, sinon celui du PREREG. Ce mtime est converti en secondes
Unix puis formaté en UTC. `gele_source` distingue `mtime` et `declare`.
Un gel déclaré sert notamment à une reconstruction documentée ; il ne
constitue pas une preuve indépendante d'antériorité : `verifier` rend alors un
AVERTISSEMENT V4 (rc 0 mais jamais un « OK » plein), que la fiche et l'entrée
de carnet exportée reprennent.

Pour un addendum changeant un critère, créer un autre fichier puis appeler :

```sh
kvu experience ajouter-prereg ADDENDUM.md --racine experience-demo \
  --change-un-critere --remplace P0 --sections critere \
  --gele-le 2026-11-11T11:00:00Z
kvu experience ajouter-run --racine experience-demo \
  --machine secondaire --prereg P0,P1 runs/r1.manifest.json
```

`--prereg` accepte des ids séparés par des espaces, des virgules ou plusieurs
occurrences de l'option. `--` peut séparer explicitement les ids des fichiers.
Un dossier d'entrée est parcouru directement, sans récursion,
pour ses `*.manifest.json`. Les autres schémas sont ignorés et comptés.
Le nom du fichier donne l'id du run. Le voisin `<id>.json`, s'il existe,
est haché. Une réimportation identique ne modifie aucun octet, même si
l'horloge a changé ; un même id avec un autre contenu est refusé.
Un nom de machine désigne un hôte/backend/architecture cohérent ; utiliser
un autre nom si ces caractéristiques diffèrent.

## Contrôles et carnet

| Contrôle | Obligation |
| --- | --- |
| V1 | Schéma, vocabulaire fermé, préfixes et dates des métadonnées. |
| V2 | sha256 de toutes les pièces hachées citées. |
| V3 | Ids uniques et premier mot du hachage consigné. |
| V4 | Début de chaque run postérieur ou égal au gel de chacun de ses PREREG ET du PREREG racine P0 (même non cité) ; un run postérieur à un PREREG qui change un critère doit le citer ; un addendum ne peut être gelé avant P0 ; AVERTISSEMENT (jamais « OK » plein) si un gel est seulement déclaré. |
| V5 | Un critère changé désigne le PREREG remplacé dans un fichier différent. |
| V6 | Graine effective de chaque run lue (alias `--seed`, défaut 1 et chaînes `--graine-pas`/`--chaines` pour `su2 gen`) ; un run sans graine lue est ROUGE ; graines distinctes entre runs ; indicateur cohérent. Les listes, `--chaines` et `--graine-pas` sont résolus par `su2::cli::chaines_depuis_listes` elle-même (arithmétique ENVELOPPANTE de la production : graine + i·pas modulo 2^64, figée par un test à la frontière) ; le nombre effectif de chaînes est borné à 1 000 000. Limite : pour les commandes autres que `su2 gen`, seule la graine de base est lue. |
| V7 | Classes, machines et binaires cohérents avec les pièces lues. |
| V8 | Une porte `rouge_attendu` possède un mutant. |
| V9 | Chaque source de proposition existe et sa plage est dans les bornes. |
| V10 | Dates et rc des runs valides et conformes aux manifestes ; fin ≥ début. |
| V11 | Historique du verdict chaîné : chaque entrée porte l'empreinte (sha256 canonique) de la précédente ; `verdict.chaine` désigne la dernière ; dates croissantes. Une troncature ou une édition de l'historique est ROUGE. Un historique partiellement chaîné est ROUGE. Un manifeste ANTÉRIEUR au chaînage (aucun lien) est migré au chargement ; la provenance reste dans `verdict.migre_le` et `verifier`, `fiche`, `carnet` (stderr et corps de l'entrée) portent un AVERTISSEMENT V11 permanent : l'historique antérieur à la migration n'est pas attesté. Limite : quelqu'un qui réécrit tout le manifeste de bout en bout peut recalculer la chaîne ; seul un ancrage externe l'empêche. |

Le contrôle documentaire d'une porte rouge n'est pas un nouveau jugement sur
son seuil. `verifier` vérifie les contraintes et les pièces, sans recalculer
une observable ni évaluer la physique.

L'export crée les sept dossiers du carnet. Les chaînes de l'en-tête sont
échappées avec la syntaxe JSON, compatible YAML. Une plage `a-b` devient
deux sources, aux lignes `a` et `b`, préfixées par `chemin_projet`.
La date du carnet est tronquée à la minute. L'expérience dérivée utilise
le gel du premier PREREG et cite ce PREREG (`<chemin>:1`, fichier qui existe toujours dans le chantier) ; son corps est la fiche (résumé des runs par machine et PREREG, portes, verdict, avertissements, réserves) ; ses liens désignent
les ids des propositions. Les liens déclarés doivent désigner une entrée
de l'export ou un fichier déjà présent dans le carnet cible.
Pour le vérificateur réel, `chemin_projet` doit être relatif à sa racine
de projet et les sources doivent y être disponibles.

Un contenu identique à la dernière version, hors seule ligne
`manifeste_sha256` de l'en-tête, laisse tous les fichiers intacts. Un
changement crée `<ID>-v2.md`, puis `-v3`, etc., avec un id adapté et un
champ `remplace` ainsi qu'un lien vers la version précédente. Les liens
déclarés conservent leurs ids d'origine. `porte` et `proposer` refusent
les ids déjà présents ; `verdict` archive même son état initial lors
de la première modification.

## Horloge imposée

`KVU_EXPERIENCE_HORLOGE` (date ISO) impose la date de toutes les écritures ; elle sert aux tests déterministes.
Toute écriture faite sous cette variable pose `horodatage.horloge_imposee = true`, jamais remis à faux ; `verifier`,
`fiche` et l'entrée du carnet portent alors un AVERTISSEMENT permanent : les dates écrites ne sont pas probantes.

## Écritures concurrentes

Toute commande qui écrit prend un verrou exclusif (`flock` sur `.EXPERIENCE.lock`,
créé dans la racine) avant de lire le manifeste et le tient jusqu'au renommage :
deux `porte`, `ajouter-run` ou `init` simultanés ne se perdent pas. Le fichier est ouvert sans suivre de lien
symbolique ni bloquer (O_CREAT|O_EXCL, O_NOFOLLOW, O_NONBLOCK sous Linux x86_64/aarch64), puis le descripteur est
contrôlé (fichier régulier, même inode que le chemin) ; un lien ou un fichier spécial est refusé. `verifier` ne verrouille pas.
L'export du carnet crée ses fichiers par `hard_link` et n'écrase jamais.

## Limites et antériorité

L'outil ne produit **aucun statut « établi »**, ni `lean`, ni promotion
automatique. Il ne juge pas la physique et ne détecte pas une modification
de critère dans le texte d'un PREREG : `change_un_critere` est déclaré.
Une empreinte protège les octets par rapport au manifeste conservé ; elle
ne prouve pas que celui-ci a été publié ou gelé auprès d'un tiers. En particulier,
antidater un run en éditant son manifeste de run et en recopiant le sha256 dans
`EXPERIENCE.json` ne se voit pas ; il faudrait un ancrage externe (commit git ou dépôt
tiers contenant l'empreinte de `EXPERIENCE.json` et des PREREG, horodatage d'un service
indépendant, ou journal d'un lanceur qui écrit lui-même les manifestes de run). L'outil
ne fournit pas cet ancrage.

Le cas réel `PREREG_ADDENDUM_1.md` contient les addenda 1, 1b et 2 édités
en place : seul le dernier mtime, 13:57:12Z, est observable. Les anciens
runs de 12:55Z et ceux à β=2,3/2,5 antérieurs à 13:57:12Z ne sont rattachés qu'à P0 ; les
runs postérieurs à chaque gel citent tous les PREREG gelés avant eux (P0, P1 pour β=2,7 dès
14:04Z, P0, P1, P2 après l'addendum 3 de 15:40:24Z, qui est un fichier séparé). Le rétro-manifeste
doit conserver cette limite dans `reserves`. Un addendum = un fichier.

Ce module met en œuvre la conception déjà retenue dans
`registre/carnet/decisions/DEC-conception-kvu-v02.md:14` et
`qlab/registre/REVUES_PILOTE.md:788`. Les exemples du carnet réel et son
vérificateur ont servi de contrat d'export. Les tests E-SCHEMA, E-ANTIDATE,
E-SHA, E-IDEMP et les invariants utilisent des dossiers temporaires et des
manifestes synthétiques. E-RETRO est un test `#[ignore]` visible dans la sortie de `cargo test` : la recette le lance avec
`--ignored` et `KVU_EXP_RETRO_DIR` (copie du cas réel, jamais modifiée) ; sans la variable il
ÉCHOUE au lieu de passer à vide.
