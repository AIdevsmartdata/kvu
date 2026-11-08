# Usage de l'IA dans ce dépôt

## Qui a écrit quoi

- Le code (Rust, noyaux HIP/CUDA, scripts Python et shell), les tests et la documentation ont été écrits par des **agents de
  codage IA** : des grands modèles de langage travaillant dans des environnements de codage agentiques, avec accès à un shell,
  à un compilateur et aux GPU des deux machines de l'auteur. Plusieurs familles de modèles ont été employées entre septembre et
  novembre 2026, comme auteurs ou comme relecteurs ; les messages de commit du dépôt de travail le consignent, fichier par
  fichier. Ce dépôt public est un instantané sans cet historique.
- La préparation de la publication (novembre 2026 : correctif zstd relu et fusionné, neutralisation des chemins privés,
  README, guide, CHANGELOG, métadonnées de citation) a été faite par un modèle Claude d'Anthropic (Claude Opus 5.5) dans
  Claude Code.
- L'auteur, Kévin Rémondière, a dirigé le travail : il a fixé les buts et les exigences (référence F64 bit-identique,
  provenance de chaque exécution, critères gelés avant les calculs) et la décision de publier, et il est responsable de cette
  version.

## Comment la justesse est établie

Rien dans ce dépôt ne repose sur le fait qu'un modèle a lu, relu ou « vérifié » le code. Les preuves sont mécaniques et se
rejouent :

1. **réponses connues** : vecteurs de Philox, moments exacts du tirage du bain thermique, limites de couplage fort et faible,
   invariance de jauge, groupes à réponse connue (`kvu groupe gates`) ;
2. **références indépendantes** : scripts Python écrits séparément du code Rust (par exemple pour le flux de bloc ou le bain
   conditionnel), comparés à 1e-12 ou au bit ;
3. **bit-identité** : le générateur F64 reproduit au bit les sorties du générateur publié `su2-heatbath-gpu` ; les sorties
   ne dépendent pas du nombre de fils ;
4. **mutants** : chaque batterie de gates contient des variantes délibérément fausses qui doivent être refusées ;
5. **intégrité de la suite** : la liste des tests telle que cargo la compte est versionnée
   (`scripts/recette/noms_tests/`), et la recette refuse un test retiré ou désactivé.

L'expérience du projet est que l'accord de deux modèles n'est pas une preuve : les erreurs ont été trouvées par un calcul
indépendant, un cas plus grand ou un mutant, pas par un second modèle qui relit.

## Relecture humaine

L'auteur relit le README, ce fichier et le CHANGELOG avant chaque version publiée.
