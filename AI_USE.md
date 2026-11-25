# Usage de l'IA dans ce dépôt

Le code a été écrit par des agents de codage IA sous la direction de l'auteur. Il n'a pas fait l'objet d'une relecture humaine ligne à ligne. (Taille de l'archive : 1 133 fichiers, tous types confondus (code, tests, documents, scripts).)

## Qui a écrit quoi (déclaration de l'auteur)

Ce dépôt public est un instantané sans l'historique de travail : les attributions ci-dessous, les noms de modèles et leurs rôles ne
peuvent pas être vérifiés fichier par fichier à partir de lui ; ce sont des déclarations de l'auteur.

- Le code (Rust, noyaux HIP/CUDA, scripts Python et shell), les tests et la documentation ont été écrits par des **agents de
  codage IA** : des grands modèles de langage travaillant dans des environnements de codage agentiques, avec accès à un shell,
  à un compilateur et aux GPU des deux machines de l'auteur. Plusieurs familles de modèles ont été employées entre septembre et
  novembre 2026, comme auteurs ou comme relecteurs ; les messages de commit du dépôt de travail le consignent, fichier par
  fichier (non publié).
- **Préparation des versions 0.1.0 et 0.1.1** (novembre 2026 : correctif zstd relu et fusionné, neutralisation des chemins privés,
  README, guide, CHANGELOG, métadonnées de citation) : un modèle Claude d'Anthropic (Claude Opus 5.5) dans Claude Code.
- **Versions 0.2 à 0.5** (novembre 2026 : tube tordu, spectroscopie, méthode intégrale, intrication, carnet d'expérience,
  coutures décalées, impulsions hors axe, lien intégré, transport), internes et non publiées séparément : écrites de la même façon, par des agents de
  codage IA de plusieurs familles (modèles Claude d'Anthropic et modèles d'autres éditeurs, dans des environnements agentiques),
  comme auteurs ou comme relecteurs.
- **Préparation de la publication de la 0.5.1** (fusion des branches, retrait des campagnes internes, neutralisation, documents,
  correctifs d'arrêt brutal, corrections de style de clippy, rejeu depuis l'archive, réponses à une relecture par un modèle d'une autre famille) : un modèle Claude
  (Claude Sonnet 5.5) dans Claude Code.
- L'auteur, Kévin Rémondière, a dirigé le travail : il a fixé les buts et les exigences (référence F64 bit-identique,
  provenance de chaque exécution, critères gelés avant les calculs) et la décision de publier, et il est responsable de cette
  version.

## Contrôles effectués et limites

La validation repose sur des tests reproductibles pour des cas documentés. Elle ne constitue ni une preuve générale de correction ni une validation physique.
Rien dans ce dépôt ne repose sur le fait qu'un modèle a lu, relu ou « vérifié » le code. Trois régimes sont à distinguer : les contrôles **publics** de la suite par défaut (`cargo test --release --locked -p kvu`, ou `scripts/recette_locale.sh` : tests ignorés exclus) ;
les tests **ignorés** (dont les tests de carte : ventilation et résultats dans le `README.md`, section « État de la validation sur carte (0.5.1) ») ; et les contrôles **internes** qui exigent des données non publiques (certaines gates de `crates/kvu/validation/`). Exemples conservés (commande, fichier de test, critère) :

1. **réponses connues** : vecteurs de Philox (`kvu su2 selftest`), moments exacts du tirage du bain thermique, limites de couplage fort et faible,
   invariance de jauge, groupes à réponse connue (`kvu groupe gates`) ; chaque test porte sa référence et sa tolérance dans son code (la table commande / fichier / référence / critère n'est pas rédigée : documentation longue reportée) ;
2. **implémentations de référence écrites séparément** : scripts Python écrits séparément du code Rust (par exemple pour le flux de bloc ou le bain
   conditionnel), comparés à 1e-12 ou au bit selon le test ; « écrit séparément » ne veut pas dire que la référence est
   mathématiquement indépendante de la méthode testée ;
3. **identité au bit** : la comparaison du mode F64 `compat-prod` aux ancres de `su2-heatbath-gpu` (DOI 10.5281/zenodo.23644087) est décrite, avec ses limites, dans le README, section « État de la validation sur carte (0.5.1) » ; elle ne vaut que pour les cas qui y sont listés ;
   les tests cités `cond_cpu`, `boite_masque` et la gate G-DIR-FILS comparent les sorties du bain conditionnel et des boîtes de Dirichlet pour certains nombres de fils (les cas et les nombres de fils sont dans ces tests) ; aucune indépendance générale vis-à-vis du nombre de fils n'en est déduite ;
4. **mutants** : plusieurs batteries de gates (par exemple `rg gates`, option `--sans-mutants` pour les écarter) contiennent des variantes délibérément fausses qui doivent être refusées ;
5. **intégrité de la suite** : la liste des tests telle que cargo la compte est versionnée
   (`scripts/recette/noms_tests/`), et la recette refuse un test retiré ou désactivé.

Une relecture par un modèle peut signaler un problème ; l'accord entre modèles ne le confirme pas. Les cas retenus doivent être étayés par un test, un calcul ou une trace vérifiable (c'est ainsi que des défauts signalés par des relectures ont été confirmés ou écartés avant la 0.5.1).

## Relecture humaine

L'auteur relit lui-même les textes de présentation (README, ce fichier, CHANGELOG, notes de version) avant chaque dépôt ; cela ne constitue pas une relecture du code.
Une relecture des textes et d'un écart de code par un modèle d'une autre famille a été faite avant le dépôt de la 0.5.1 ; elle ne prouve rien (voir ci-dessus).
