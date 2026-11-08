# PREREG FINAL — WP14 : mesure pilote de ε′(φ) sur fonds Monte-Carlo (grandL L = 12), 01/11/2026

Rédigé AVANT toute donnée de verdict (le projet de l'Opus du 26/10 est `briefs/WP14_PREREG_projet.md`). GO de Kévin : 01/11/2026. Procédure allégée v6 (GATES.md).
Relecture par une autre famille (tour 1 : NON GELABLE, 1 objection bloquante sur la certification des gates et 1 non bloquante sur les tentatives .a2, corrigées ; tour 2 : NON GELABLE, 5 objections bloquantes toutes sur le DURCISSEMENT de verdict.py contre un pipeline falsifié (reprises .a2 limitées aux refus, liste de fonds authentifiée par sha256 et sans doublons, contrôle S2 recalculé sur exactement 112 identifiants, commande exacte comparée au plan, gates auxiliaires et g3_fond certifiés) : TOUTES corrigées et testées par `test_verdict.py` (12 falsifications), mais SANS tour 3, la règle des 2 corrections étant épuisée : réserve déclarée ; aucune objection ne portait sur la règle de lecture, l'estimateur ni les seuils) : **Kimi K3.1 Pro à la place de GPT-6.1-Sol** (quota Sol épuisé jusqu'au 04/11) — substitution déclarée.
Aucune conclusion sur le mass gap n'est tirée de WP14 : un pas, L = 12, un seul schéma de blocs, fonds bruts.

## 1. Question
ε′(φ) = 1 − λ_min(H(φ)|_X, Q|_X) (Hessienne de Wilson SU(2) rapportée à sa référence) se comporte-t-elle sur des fonds de la VRAIE mesure comme sur les fonds gaussiens de S2
(ε′ ∝ g, α ≈ 1 ; S2 VERDICT §3.2 : α = 1,020 [0,988 ; 1,053], 4⁴, T-fix-fluct grossier) ? g = √(4/β).

## 2. Périmètre et écarts au projet (déclarés)
- **Fonds** : configurations MC brutes, ensembles grandL L12 (lecture seule), liste `fonds.py` (règle ci-dessous), 36 fonds, sha256 de la liste : `19b584e6006b8a67a16f39053079bb3202e3894e755c77f6e4da7bc0468f6d04`.
  β ∈ {4 ; 8 ; 16} (ajustement) : 8 fonds par β = chaînes c1 et c2 (départ chaud) dumps 600, 1200, 1800 ; c3 (froid) dumps 900, 1800. β ∈ {2,3 ; 2,5 ; 2,8} (rapportés à part, hors ajustement) : 4 fonds
  (c1, c2 : dumps 600, 1800). Espacement ≥ 600 pas dans une chaîne (≥ 2 τ_Pol,max = 289 ; τ_int plaquette ≤ 2,3). Les dumps 100 servent aux essais de coût et ne sont dans aucune liste.
  Ce sont les analogues des fonds « plein » de S2 : les fonds « grossier » (espérance conditionnelle) n'ont pas d'analogue MC sans échantillonneur de fibre.
- **(d) fibre : EXCLU.** WP12b n'est pas VERT (ni WP12a admis : ETAT_KVU_FINAL, AUDIT_REPRISE §6) ; aucun rendu WP12a/WP12b n'existe.
- **(b) ker DŪ : NON EXÉCUTÉ, pour cause de coût mesuré.** `kvu hess espace --espace kerdu` : construction à L = 9 (Haar) 14 min 46 s sur 4 fils ; l'échelle en L (jacobien dense 12 L⁴ colonnes, SVD de Jacobi en m²n)
  donne ≈ 31× à L = 12 (≈ 7 h par fond, 6 Go de jacobien), soit des centaines d'heures-cœur pour 36 fonds. Le code (WP13, fusionné) n'est pas modifié. (b) reste à faire après un jacobien creux.
- **Espaces exécutés** : (a) `tcov` (coordonnées gauche, référence Q_cov(U), fond brut) ; (c) `kerB` = T-fix-fluct de S2 (coordonnées additives, référence Q0, fond fixé en jauge de Landau par `kvu hess landau`).
  **Les deux sont DÉCISIFS** (le projet nommait (a) et (b) ; (c), l'espace principal de S2, remplace (b) absent).
- Écart au projet : le mutant M-a (SU(2) abélianisé) n'existe pas dans kvu ; il est remplacé par WP14-MUT (mutant compilé M-b) + WP14-ANA (estimateur à réponse connue) + WP14-G3-L12.

## 3. Calcul (gelé)
`kvu hess lmin --espace E --coord C --reference R --beta β --graine 1 --tol 1e-10 --iter-max 3000 --bloc 4 --fils 1` (projecteur auto = CG à L = 12), un calcul par (espace, fond), via le lanceur unique
`scripts/executer.py` (manifeste par calcul, créneaux ≤ 12, nice 10, MemoryMax 6 Go par calcul, 6 h max). Un refus AVANT lancement est réessayé ≤ 2 fois ; un calcul démarré n'est jamais relancé.
Codes (commit des scripts : `ae9e88c5f84574feb3a58cacdad4cee99da81648`) : driver.py `5996ac3a69888637dd05415a04231fdf13cb346b4ed05899ac95c8f1210227d7`, estimateur.py `edcb6641bf0e9f0765c34ffa3a9f45d471c83d09fce2fdca594e236edb30af0f`, etape.py `0feffcb70ad89aba0c4ffb66524056e6a231898a42a863607f0d31cafe109052`, fonds.py `7d50f11c762b053dc343e0575f1f9dae3966049c8aed2ad65fae9ab8ae99d44d`, g3_l12.py `f1eb66c359a63e904292aae62150401118af768f3a903462085f563667c153fe`, plan.py `b7aa0d808698aa03953532b0cbab704c1c79188d94888ee3a5a9d8bb0124cd0e`, s2_controle.py `6d363fc7c1a639e14b6ce2fad4b83a86929c9fcd439f03fa1414ec7aed02d0e7`, test_verdict.py `80d7c23a82822c17ccd973b4b141874bfd52938c9f5bd0f5518abb7a6d9b4311`, verdict.py `baeaf7de9abd660d79576cd5bd4eeb6e86f1be5c58f3888d42dc7a9edc48bbf6`. Outils WP13 : `outils.py` `894220f2fa34cbc44f6c64e9f38a313e3afed1a1b44d94e21fc1f0cff616218a`, `lat.py` `a29c19be8eb6d07685068de0ae2eade0594cbc893f0ae86150f506446f4871af`. Code Rust inchangé : branche wp14-2026-11 issue de `pret-a-fusionner/wp10` (f787e7f) ;
le candidat ajoute seulement des fichiers Python et ce PREREG. Bancs de coût (hors liste) : kerB 601 itérations, 22 min de mur, λ_min stable à 1e-12 dès 120 itérations ; tcov 40 itérations 2 min 50.

## 4. Estimateur et règle de lecture (gelés ; constantes dans estimateur.py)
Pour chaque espace : m(β) = médiane de ε′ sur les fonds ; α = pente OLS (poids égaux) de ln m contre ln g sur β ∈ {4, 8, 16} ; IC 95 % = percentiles 2,5/97,5 de 2000 rééchantillons bootstrap
stratifiés par β (graine numpy `[2611011401, indice]`, indice tcov 0, kerB 2). Rapporté, jamais une gate : ε′/g par β, β bas à part.
Lecture d'un espace (IC = [lo, hi]) : INDÉTERMINÉ si largeur > 0,4, ou médiane ≤ 0, ou ≠ 8 fonds par β d'ajustement ; DIFFÉRENT− si hi < 0,8 ; DIFFÉRENT+ si lo > 1,2 ;
COMME-S2 si lo ≥ 0,9 et hi ≤ 1,1 ; sinon INDÉTERMINÉ.
Verdict global : INDÉTERMINÉ si une gate d'instrument ou le contrôle positif n'est pas vert ; DIFFÉRENT si au moins un espace décisif est DIFFÉRENT± ; COMME-S2 si tous les espaces décisifs sont COMME-S2 ; sinon INDÉTERMINÉ.
**Prédiction écrite avant les données** : S2 lui-même donne sur la fenêtre {4,8,16} des α de 0,97 (kerB, grossier) à 1,15 (kerB, plein) selon le type de fond ; si les fonds MC se comportent comme « plein », un résultat INDÉTERMINÉ avec α entre 1,1 et 1,2 est attendu ; la lecture COMME-S2 est exigeante par construction.

## 5. Gates (toutes lues dans les fichiers, aucune écrite en dur ; `verdict.py`)
| ID | énoncé | seuil |
|---|---|---|
| HESS-S2 (contrôle positif, d'abord) | fonds gaussiens grossiers 4⁴ régénérés par `lat.py` (mêmes graines 1-4 × 4 fonds, β ∈ {4…256}) passés par `kvu hess lmin --espace kerB` | α(kvu) sur β ∈ {8…256} ∈ [0,99 ; 1,05] ; médianes kvu = médianes lat.py à 1e-8 par β ; les trois lignes MED, ALPHA, FENETRE et MANIFESTES (112 calculs avec manifeste conforme, LOBPCG convergé, résidu ≤ 1e-9, Q positive) EXIGÉES en PASS, aucun FAIL ni SKIP ; fenêtre {4,8,16} : \|α(kvu) − α(lat)\| ≤ 1e-7. Constat avant gel : les fonds régénérés ne sont pas ceux de S2 (base propre dégénérée de eigh), α(lat.py) sur eux = 1,030 ; HESS-S2-REGEN = INFO |
| WP14-INSTR | gates de WP13 rejouées sur le binaire de la campagne (`validation/hess/lancer.sh`, manifeste `instr`, VERDICT.txt en sortie hachée) | `VERDICT WP13 VERT` |
| WP14-EXEC-CONV | chaque calcul (tentative terminée rc 0 : `nom`, sinon `nom.a2`, `nom.a3` après un refus avant lancement) : manifeste du lanceur, dépôt propre au SHA candidat, binaire kvu et entrées attendues (fond ; fond fixé de Landau pour kerB, dont le travail `landau_*` est lui-même certifié) déclarés avec le sha256 ACTUEL du fichier, entrées inchangées, paramètres de la commande conformes (graine 1, bloc 4, tol 1e-10, iter-max 3000), rapport = sortie hachée du manifeste ; LOBPCG convergé (tol 1e-10, 4 vecteurs), résidu projeté ≤ 1e-9, positivité de Q restreinte vérifiée | tous les calculs prévus |
| WP14-STAB | graine 2, bloc 6 sur le fond c1/600, β ∈ {4,8,16}, espaces tcov et kerB (mêmes exigences de manifeste) | convergé et \|Δλ_min\| ≤ 1e-6 |
| WP14-G3-L12 | fond de jauge pure L = 12 (graine 2611011421) | \|ε′\| ≤ 1e-6 (kerB après Landau), ≤ 1e-9 (tcov) |
| WP14-MUT | mutant compilé M-b sur kerB, fonds b8_c1_600 et b16_c1_600 | ε′ s'écarte d'au moins 0,1 du calcul sain (sinon ECHEC_OUTIL) |
| WP14-ANA | `verdict.py autotest` : α de S2 retrouvé à 1e-9 ; synthétique : α=1 → COMME-S2 ≥ 80 % ; α=2 → DIFFÉRENT+ ≥ 95 % ; α=0,5 → DIFFÉRENT− ≥ 95 % ; bruit 30 % → INDÉTERMINÉ ≥ 90 % ; combinaison ; `test_verdict.py` : campagne fabriquée α=1 → COMME-S2, α=2 → DIFFÉRENT, six falsifications (rapport périmé, Landau absent, mutant sans --mutant, S2 sans ALPHA, fond remplacé) → INDÉTERMINÉ, refus puis .a2 réussi accepté | PASS |
| WP14-2MACH | sans objet : aucun calcul sur le gamer (tout sur l'acemagic) | — |
Aucun seuil ne change après observation. Un calcul non certifié rend la gate EXEC-CONV rouge et donc le verdict INDÉTERMINÉ ; pas de repli.

## 6. Machines et limites
Acemagic seulement, CPU, aucun GPU. Régulateur : classe P0-kvu à nice 15 ; la recette de fusion garde la priorité. Calculs > 5 min lancés détachés (`systemd-run --unit=wp14-… -p MemoryMax=40G -p Nice=15`).
Limites : un pas, L = 12, MC brut, ε′ est un sup sur le volume (dépend du volume) ; 36 fonds ; (b) et (d) absents ; τ_Pol à β = 8, 16 ≈ 200-290 pas : les fonds d'une chaîne peuvent rester corrélés.
