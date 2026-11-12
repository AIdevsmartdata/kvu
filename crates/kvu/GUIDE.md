# kvu — guide d'utilisation

`kvu` est un binaire unique qui **reconnaît la machine** et choisit le bon code : GPU AMD par HIP (hiprtc), GPU NVIDIA par CUDA
(NVRTC, en CUBIN), sinon le CPU. Les noyaux GPU sont écrits une fois (`crates/kvu/noyaux/`) et compilés à l'exécution. Le CPU
calcule en F64 et sert de référence. Les deux machines de référence (celles des tableaux ci-dessous) sont une station AMD
Ryzen AI Max+ 395 (Radeon 8060S, gfx1151, profil `acemagic`) et un PC à GeForce RTX 5060 Ti (sm_120, profil `gamer`).

Référence de conception : `ARCHITECTURE_UNIFIEE.md` (même dossier). Ce guide dit COMMENT s'en servir. Les commandes des blocs de
code de ce guide sont vérifiées par le test `cli_readme` (chaque sous-commande et chaque option existe ; les lignes `--dry-run`
sont exécutées).

## 1. Installation

```bash
cargo build --release --locked -p kvu                    # binaire target/release/kvu (backend HIP toujours compilé)
cargo build --release --locked -p kvu --features cuda    # avec le backend CUDA (cudarc, chargement dynamique de libcuda/NVRTC)
scripts/build_auto.sh --crate kvu                        # build réglé par machine (cible dédiée, dépôt propre exigé)
scripts/build_auto.sh --dry-run                          # affiche le plan sans rien construire
```

Aucune bibliothèque GPU n'est nécessaire à la compilation : `libamdhip64`/`libhiprtc` (ROCm) et `libcuda`/`libnvrtc` (pilote et
CUDA ≥ 12.8) sont chargées à l'exécution, et leur absence n'est une erreur que si l'on demande ce backend. `scripts/build_auto.sh`
exige un dépôt propre (le binaire doit correspondre à un commit), exporte `KEV_GIT_HEAD`, refuse sous (seuil SSD + 10 Go) libres,
puis écrit `<binaire>.sha256` et `<binaire>.commit` à côté du binaire ; `--lto` donne la build finale (LTO fat, plus lente).
Configuration des machines : `kvu.toml` (modèle à la racine du dépôt) et `docs/MACHINES.md`.

## 2. Choix du backend

La détection est automatique et se lit avec `kvu doctor` (profils de la fixture `crates/kvu/kvu.toml` ; sans profil qui
corresponde, une machine est `inconnue`) :

| machine | reconnue par | backend auto | verrou GPU | garde SSD |
|---|---|---|---|---|
| acemagic | hôte, CPU « AMD RYZEN AI MAX+ 395 » ou gfx1151 | HIP (gfx1151) | `/tmp/kevinotron_gpu.lock` | 100 Go |
| gamer | hôte, CPU « i5-14600KF » ou sm_120 | CUDA (sm_120) | `/tmp/gamer_gpu.lock` (sur le gamer) | 50 Go |
| autre | — | premier GPU utilisable, sinon CPU | `/tmp/kevinotron_gpu.lock` | 20 Go |

Surcharges d'environnement : `KEVINOTRON_BACKEND=auto|hip|cuda|cpu`, `KEVINOTRON_MACHINE=acemagic|gamer|inconnue`,
`KEVINOTRON_SSD_MIN_GO`, `KEVINOTRON_RAM_MIN_GO` (RAM minimale d'admission des jobs de la file), `KEVINOTRON_STORE`, `KEVINOTRON_FILE`, `KEVINOTRON_CACHE_DIR`, `KEVINOTRON_VERROU=ignorer`
(tests CPU seulement, consigné dans le manifeste).

**Un backend forcé mais absent est une erreur, jamais un repli silencieux** :

```bash
KEVINOTRON_BACKEND=cuda kvu doctor      # sur acemagic : "GPU : …", code de sortie 3
```

Les sous-commandes `su2`, `cond`, `fibre`, `obs` acceptent aussi `--backend auto|hip|cuda|cpu`. Les variables gardent le préfixe
`KEVINOTRON_` du moteur dont kvu est issu.

Le CPU ne calcule qu'en **F64** (référence) ; la précision **F32** exige un GPU.

## 3. Sous-commandes

`kvu <sous-commande> …` ; `--help` partout ; `kvu --version` donne la version, le `source_hash` de la recette des sources
(`kvu-source/1`) et le `git_head`.

| commande | rôle |
|---|---|
| `doctor [--json] [--test]` | machine, GPU, backend et pourquoi, versions, politique, SSD, store, file, cache ; `--test` compile et exécute un noyau d'essai sous verrou |
| `etat --json` | UNE ligne JSON d'état (< 200 ms, sans contexte GPU) : voir §8, pour les mods |
| `queue add\|list\|run\|annuler` | file de jobs de la machine (voir §7) |
| `su2 gen` | générateur SU(2) : lots de chaînes, F64 de référence ou F32 rapide, bord gelé R2-3, reprise, mesures en ligne |
| `su2 test-hb ALPHA N` | test du tirage de x₀ (ligne `TESTHB` de su2_hip) ; `su2 selftest` : vecteurs Philox |
| `cond chaine` | bain conditionnel « fin sachant grossier » (mode `ref_rng` ou `rapide`) |
| `fibre echantillonner` | Échantillonnage de fibre SU(2), pénalité ou RATTLE, et jouets |
| `cond facteur3` | fibre dure de DEC de facteur 3 et noyau de réponse de la moyenne conditionnelle (T3-D) : `repliques`, `effectifs`, `tube-classe`, `tube-f`, `tube-hist` ; voir `docs/COND_FACTEUR3.md` |
| `fibre mesurer` | F (54), D (54×54), blk d'une configuration |
| `obs wilson\|polyakov\|flot` | boucles de Wilson, corrélateur de Polyakov, flot de Wilson (E, t₀, w₀, Q) |
| `store du\|gc\|verifier\|lister\|exporter` | place disque, éclaircissement (simulation par défaut), vérification |
| `store classer` | **simulation** : classe A/B/C la régénérabilité de chaque chaîne des ensembles (aucune écriture) |
| `regen M.json` | rejoue une chaîne depuis son manifeste (`kvu-regen/1` de `su2 boite` et `fibre echantillonner` ; historiques `su2_hip/1`, `su2_gen/1`) et compare les sha256 fichier par fichier ; ne rejoue pas un run `su2 gen` (§ Régénération des ensembles) |
| `groupe liste\|gates\|certificat\|derive\|plan\|chaine` | registre des groupes de jauge (SU(2), Sp(4)) et batterie de gates à réponse connue (G-ALG, G-HAAR, G-SC, G-WC, G-UNIT, G-REG, mutants), certificat G-CERT ; voir `docs/CONVENTIONS.md` |
| `gen --groupe sp4` | générateur d'ensembles d'un groupe générique (Sp(4), CPU) : Metropolis + sur-relaxation, refuse un groupe sans certificat ; `su2 gen` reste le chemin SU(2) bit-identique à (A) |
| `rg bloc\|swendsen\|domaine\|gates` | module RG par blocs : configuration grossière (L/3)⁴ par la moyenne non linéaire de Bałaban L_b = 3 (SU(2), CPU), témoin de Swendsen b = 2, rapport de domaine, gates RG-* |
| `rg flot\|etalonner\|flot-gates` | estimateurs de couplage effectif (SD à un lien, appariement de plaquette, MCRG à deux opérateurs), courbes d'étalonnage de Wilson au volume grossier, gates RG-SD / RG-PLAQ / RG-N3 / RG-TAU |
| `fetch gamer:<chemin\|store/ID> [DEST]` | rapatriement par le câble 2,5 Gbit/s, sha256 vérifiés |
| `valide [--rapide\|--complet\|--gate NOM] [--json]` | suite de validation (§9) |
| `bench su2\|cond\|fibre\|obs` | bancs sous verrou (§10) |

### rg (module RG par blocs, WP6 ; GPU et mesure en ligne, WP7)

```text
kvu rg bloc --entree X.npy --sortie Y.npy [--fils N] [--geometrie 3centre|3coin] [--backend cpu|hip|cuda|auto]
kvu rg swendsen --entree X.npy --sortie Y.npy [--c 0.5] [--pas 1]
kvu rg domaine --entree X.npy [--rapport R.json] [--fils N] [--backend cpu|hip|cuda|auto]
kvu rg gates [--fixtures D] [--donnees D] [--grandl D] [--rapports D] [--fils N] [--seul RG-PUR,RG-ABEL] [--sans-mutants] [--json] [--certifier [FICHIER]]
kvu rg gates-gpu --backend hip|cuda [--fixtures D] [--grandl D] [--sortie D] [--fils N] [--repetitions N] [--sans-mutants] [--json]
kvu rg comparer --a DIR --b DIR [--min N] [--id ID] [--sans-theta]
```

`bloc` calcule Ū(z, μ) = U(Γ₀) exp(Σ w log(U(Γ₀)⁻¹ U(Γ ∪ [x, x+3e_μ] ∪ Γ′⁻¹))) (définition gelée S1, 10 473 boucles par lien grossier, quaternions SU(2) exacts) :
L doit être un multiple de 3 ; la sortie a le côté L/3 et le format de l'entrée (c16 ou q64), plus un manifeste `<sortie>.manifeste.json` (`kvu-rg-bloc/1` : pas_de_bloc,
géométrie, histogramme de θ, fraction hors domaine, sha256 de l'entrée et de la sortie). Le parallélisme est par lien grossier et la sortie, manifeste compris,
est identique à l'octet quel que soit `--fils`. `--geometrie 3coin` est le MUTANT de S1 (blocs de coin, non invariant par réflexion) : il ne sert qu'à vérifier que les gates le rejettent.
`domaine` rapporte la distribution de |θ| (RG-DOMAINE : `DANS_DOMAINE` ssi f = #(θ > π/2)/#boucles ≤ 1e-3 et aucune branche ambiguë) sans jamais échouer.
`swendsen` est le témoin de schéma b = 2 (port de `block_config`). `gates` émet une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate (RG-REG-PY, RG-JAUGE, RG-HYPER,
RG-PUR, RG-ABEL, RG-DOMAINE, RG-SWENDSEN, RG-FILS, et leurs mutants) ; code de sortie 0 si aucune ligne FAIL ni SKIP, 8 sinon. Fixtures : `validation/rg/` (script
`gen_fixtures.py`, venv épinglé, généré une fois) ; données de test : `tests/donnees/rg/`.

**GPU (WP7).** `--backend hip|cuda|auto` calcule le même bloc sur le GPU (noyau `noyaux/rg_bloc.cu`, source unique HIP/CUDA, F64 sans contraction, un fil par lien grossier,
somme séquentielle dans l'ordre du CPU, aucune atomique flottante ; tables de chemins en mémoire globale en lecture seule, ≈ 9 Ko, car le runtime n'expose pas la mémoire
constante). Le GPU est sous le verrou GPU de la machine ; la sortie est déterministe au bit sur un même backend et à 1e-12 de celle du CPU (gates RG-GPU, RG-DOMAINE-GPU,
G-DET) ; un manifeste GPU porte en plus le bloc `calcul` (backend, arch, sha256 du code objet). `gates-gpu` joue ces gates (+ mutant `3coin`, + banc de temps L = 24 → 8 et
extrapolation L = 54 → 18, rapportés, jamais une gate) ; avec `--sortie D` il écrit `D/<backend>/<nom>.npy` et `<nom>.theta.json`, que `comparer` (RG-HIP-CUDA, ≤ 1e-12) juge entre deux
backends ; `comparer --sans-theta --id ENLIGNE-1` ne compare que les quaternions. Script de rejeu : `validation/rg/gates_rg_gpu.sh`.
**Mesure en ligne.** `su2 gen … --mesures bloc:3:<période>` calcule, à chaque mesure, la configuration grossière (L/3)⁴ sur le GPU (ou le CPU) et écrit dans le répertoire de la
chaîne `bloc3_cfg_<pas>.npy` (q64), `bloc3.csv` (pas, côté, n boucles, fraction hors domaine, θ max, ambiguës, plaquette grossière, sha256) et `bloc3_hist.csv` (histogramme de θ) ; aucun
dump fin n'est requis (L multiple de 3, champ résident F64). Un run terminé écrit aussi `regen_bloc3.json` (`kvu-regen/1`, `parametres.mesures`, classe C : classement A/B laissé à `regen`) ; en 0.1.1,
`kvu regen` ne sait pas encore le rejouer (la commande enregistrée garde `--outdir` : FAIL REGEN, code 8 ; PUB-15) : rejouer le run par son `argv`
(§ Régénération des ensembles).
Un run sans `--mesures bloc:3` est inchangé à l'octet.

### hess (Hessienne de l'action de Wilson SU(2) sur fonds réels, WP13)

Sous-commande `hess` :

```text
kvu hess hv   --config X.npy --vecteurs V.npy --sortie Y.npy [--base ym4|s2] [--coord gauche|additive] [--beta B] [--fils N]
kvu hess jvp  --config X.npy --vecteurs V.npy --sortie Y.npy [--ubar-sortie U.npy] [--fils N]
kvu hess espace --config X.npy --espace libre|kerB|landau|tcov|kerdu [--projecteur svd|cg|auto] [--projeter V.npy --sortie Y.npy] [--rapport R.json]
kvu hess lmin --config X.npy --espace libre|kerB|landau|tcov|kerdu [--reference q0|qcov] [--coord gauche|additive] [--beta B] [--tol 1e-10] [--dense] [--vecteur-sortie V.npy] [--sortie R.json]
kvu hess landau --config X.npy --sortie Y.npy [--tol 1e-24] [--iter-max 5000]
```
`hv` : H·v de S (β inclus), port de `ym4_hess` (identique au bit) ; base `ym4` (X_a = iσ_a/√2, indice 3(μV+x)+a) ou `s2` (iσ_a/2, indice 3(4x+μ)+a, H_s2 = H_ym4/2) ; coordonnées `gauche` (U → exp(iζσ/2)U) ou `additive` (U = exp(iA·σ/2), A → A+ζ, convention de S2).
`jvp` : DŪ·v par nombres duaux sur la carte de Bałaban (3centre) ; entrée en disposition s2 fine, sortie en s2 grossière. `lmin` : ε′ = 1 − λ_min(H|_X, Q|_X) par LOBPCG généralisé F64 (graine fixée), `landau` : fixation de Landau réseau (port de `lat.py::landau_fix`).
Toutes les sorties sont identiques à l'octet quel que soit `--fils`. Validation : `crates/kvu/validation/hess/` (`PREVOL.md`, `lancer.sh`), GATES v4-T2 partie 1 §1.3.

### groupe (registre des groupes, WP5)

```text
kvu groupe gates --groupe su2 [--backend cpu|auto|hip|cuda] [--json] [--seul G-ALG,G-SC] [--sans-mutants] [--tirages N] [--certifier [DIR]]
kvu groupe certificat --groupe su2 [--dossier validation/groupes]
kvu groupe derive --groupe su2 [--tirages N]
kvu groupe gates --groupe sp4 [--chaines DIR] [--sans-calcul] [--json] [--seul G-ALG,G-SC] [--sans-mutants] [--tirages N] [--certifier [DIR]]
kvu groupe plan --groupe sp4 --dossier DIR [--binaire CHEMIN] [--sans-mutants] [--sans-info]
kvu groupe chaine --groupe sp4 --id ID --dossier DIR [--mesures N] [--reprise N]
```

`gates` émet une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate ; code de sortie 0 si aucune ligne FAIL ni SKIP, 8 sinon (sur CPU `G-UNIT-F32` est SKIP :
gate propre au GPU, règle 0.2 de `GATES.md`), 2 si le groupe n'est pas au registre. `--certifier` écrit `validation/groupes/<groupe>.json` (une entrée par backend :
horodatage, sha256 du binaire, verdicts) et exige la batterie complète et gelée. `--groupe g2` est refusé (code 2) : ni fiche ni certificat (Sp(4) : fiche et certificat du lot WP16). Sur un backend GPU,
la batterie prend le verrou GPU de la machine. Conventions, valeurs dérivées et protocoles : `docs/CONVENTIONS.md`.

**Sp(4) (WP16, CPU).** `gates --groupe sp4` rejoue la batterie de GATES v4.1/v4.4 (Fiche, G-ALG-1/2/3/DIM, G-HAAR-1/2/3/-CHAMP, G-SC, G-WC, G-UNIT, G-REG, mutants) ; les chaînes
Monte Carlo (G-SC, G-WC, G-REG et leurs mutants) sont des **séries de plaquettes** lues dans `--chaines DIR`. `plan` imprime une commande `chaine` par ligne (à passer à
`gamer_lot` ou `calc_borne`), `chaine --id ID` simule une chaîne avec points de reprise (`DIR/<id>.serie`, `DIR/<id>.reprise`) et la prolonge avec `--mesures N′` (extension unique
de v4.0 : mêmes graine et ε) ; sans `--sans-calcul`, `gates` calcule en série ce qui manque. Une ligne SKIP n'affiche aucune valeur physique. `--certifier` écrit
`validation/groupes/sp4.json` (entrée `cpu`, aucun SKIP admis).

### gen (groupes génériques)

```text
kvu gen --groupe sp4 --L 8 --beta 8 --graine S [--start hot|cold] [--therm N] [--nmeas N] [--skip K] [--dump-every N] [--dump-initial] [--certificats DIR] --outdir DIR
```

Metropolis (10 propositions par lien, ε₀ = 0,3 adapté pendant la thermalisation seulement) puis sur-relaxation, RNG Philox à compteur (pas, lien, tirage, étiquette) ; écrit
`measurements.csv`, `manifest.json`, `COMPLETE` et des `cfg_NNNNNN.npy` (complex128, (μ, x₀…x₃, i, j)). Le binaire refuse (code 2) un groupe sans certificat `cpu` vert.

### su2 gen

```text
kvu su2 gen --L 16 --beta 2.3[,2.8…] --graine S[,S2…] [--chaines K] [--graine-pas P] --start hot|cold
        --therm N --nmeas N --skip 1 --nor 4 [--precision f64|f32] [--backend auto|hip|cuda|cpu]
        [--options ref-k1|compat-prod|rapide] [--dump-every N --dump-initial --format c16|q64 --zstd]
        [--mesures SPEC] [--reprise-every N]
        [--bord --bc cold|hot|eq --bc-graine G --jauge G --pre N --bloc N] --outdir DIR | --reprendre DIR
```

- **Lots** : `--beta` et `--graine` prennent des listes (une valeur par chaîne) ; `--graine S --graine-pas P` donne `S + i·P`. Un lot
  partage L, la précision et le départ (`--start` est unique par lancement). La chaîne `c` d'un lot donne **les mêmes octets** que la
  même chaîne lancée seule (gate LOT-1).
- **Options numériques** : `ref-k1` (défaut F64) = su2_gen K1, `-ffp-contract=off` ; `compat-prod` = les binaires de production su2_hip
  (hipcc) et su2_cuda (nvcc) ; `rapide` (défaut F32). Les noms historiques `--seed` et `--bc-seed` sont acceptés.
- **Reprise à l'octet** : `--reprise-every N` écrit des points de reprise atomiques ; `su2 gen --reprendre DIR` continue, dans un
  nouveau processus, avec la même torsion et les mêmes mesures en ligne (rechargées de `torsion.json` et `mesures_en_ligne.json` ;
  une option différente est refusée, code 2 ; les fichiers des mesures en ligne sont ramenés au point de reprise), et donne les
  mêmes octets qu'un run ininterrompu (gate REPRISE-1). Un run créé par la 0.1.0 AVEC des mesures n'a pas
  `mesures_en_ligne.json` : pour le reprendre, repasser la même `--mesures` (sinon refus, code 2, avec la commande à relancer) ; un
  run sans mesures se reprend comme avant. `--bord` n'accepte pas `--mesures` (code 2). Les en-têtes des CSV (`therm.csv`,
  `measurements.csv` ; avec `--bord`, `therm_blocs.csv` et `blocs.csv`) sont sur le disque dès leur création : après un arrêt du
  PROCESSUS (kill -9, plantage), y compris pendant la thermalisation, `--reprendre` repart du dernier point de reprise valide
  (0.1.1, PUB-14 ; parade pour la 0.1.0 : `docs/DEFAUTS_CONNUS.md`). Avant le premier point de reprise, il n'y a rien à reprendre :
  relancer le run. Une coupure de COURANT peut perdre des lignes vidées mais non synchronisées (PUB-05, limite connue).
- **Mesures en ligne** : `--mesures wilson:R8xT8:10,flot:0.01:3:20` mesure pendant la génération (type:paramètres:période) : aux
  grands L on ne stocke plus les configurations.

### su2 boite (boîtes de Dirichlet à bord fixé)

```text
kvu su2 boite --ell 8 --beta 2.5 --bords cold,z2full --repliques 16 --graine 26102611000 --graine-pas 1
        --therm 1000 --nmeas 30000 --skip 1 --nor 2 --profil-tous 10 --backend auto --outdir DIR
kvu su2 boite --ell 12 --beta 2.5 --chaines cold:1,gauge:2,flux:3,haar:4,therm:5 --therm 1000 --nmeas 5000 --dump-final --outdir DIR2
kvu su2 boite-obs --cfg DIR2/c000/config_finale.npy --ell 12 --backend cpu
kvu su2 boite --ell 12 --beta 2.5 --bord-npy DIR2/c003/config_finale.npy --bords npy --repliques 8 --graine 26102800001 --outdir DIR3
```

Simule SU(2) Wilson dans des boîtes ℓ⁴ (ℓ pair) dont les liens du bord (deux extrémités sur le bord du cube) sont **fixés** : un lot de chaînes
(une par couple bord/graine, même ℓ et même β) par lancement, sur GPU (HIP, CUDA) ou sur le CPU de référence. Mesures à chaque pas :
plaquette centrale `P_c` (24 plaquettes de coin au centre), boucle de Wilson 2×2 centrale `W₂` (ℓ ≥ 4), et, tous les `--profil-tous` pas, le profil de
plaquette par couche de profondeur. Un pas = 1 bain thermique + `--nor` surrelaxations ; départ intérieur froid, `--therm` pas sans mesure.

- **Bords** : `cold`, `gauge` (jauge pure g(x)g(x+μ)†, témoin), `z2full` (toutes les plaquettes du bord = −1), `z2_01`, `flux` (rotations non commutantes, φ = π/2),
  `haar` (liens fixés de Haar), `therm` (liens fixés d'une configuration d'équilibre périodique de même β, produite par le même moteur : `--therm-source` pas).
  Avec `--bords`, la chaîne c = indice_bord·R + réplique porte la graine G + c·P ; `--chaines BORD:GRAINE,…` donne les graines une à une.
- **Bord arbitraire** (`--bord-npy FICHIER`, répétable) : un champ NPY du réseau de calcul (q64 de `--dump-final` ou c16, côté ℓ + 2) dont seuls les
  liens fixés sont lus (validés SU(2) à 1e-12, copiés au bit) ; le bord s'appelle `npy:<8 premiers hexadécimaux du sha256>` (`npy` s'il est seul), le
  manifeste consigne chemin, format et sha256, et `kvu regen` refuse de rejouer si le fichier a changé. Spécification, refus et codes : `docs/BORD_NPY.md`.
- **Masque de liens gelés** : la boîte est plongée dans un réseau périodique de côté ℓ + 2 ; tout lien non libre (fixé, inexistant, de remplissage) est gelé par un masque
  (un octet par lien) que le bain thermique et la surrelaxation respectent, en CPU, HIP et CUDA (même source de noyau). Sans masque, les sorties de `su2 gen` sont inchangées à l'octet
  (G-BIT-HIP-PROD, G-BIT-CUDA, G-BIT-K1). Le F32 n'a pas de masque : `--precision f32` est refusé (code 2).
- **Sorties** (répertoire NEUF) : `cNNN/mesures.csv` (`pas,pc,w2`), `cNNN/profil.csv` (`pas,d0,d1,…`), `cNNN/config_finale.npy` (avec `--dump-final`, format q64, réseau de calcul),
  `manifest.json`, `manifest_regen.json` (schéma `kvu-regen/1`, rejouable : `kvu regen DIR/manifest_regen.json --sortie DIR_REJEU`), `COMPLETE` en dernier ; dernière ligne de la sortie
  standard : `PERF boite …` (temps par pas et par chaîne). En fin de run, les liens gelés sont comparés à l'octet à leur état de départ : un écart rend le code 8.
- **Verrou GPU** pris par la commande elle-même (comme `su2 gen`) ; sous `gpu_partage`, en mode partagé (`verrou: partage`, `docs/MACHINES.md` §8) ; chaque lot doit tenir dans une fenêtre de 10 minutes (voir `scripts/pilote_boites.py`, le pilote de balayage
  qui découpe une liste de tâches `ell,beta,bord,graine` en lots et les répartit sur les deux GPU). Le pilote exige le verrou de chaque cible ssh (`--verrou gamer=/tmp/gamer_gpu.lock`) et
  enveloppe chaque lot dans `flock` : un `kvu` lancé à distance sans `kvu.toml` ne reconnaît pas le poste et prendrait le verrou par défaut, pas celui du poste.
- `su2 boite-obs` relit une configuration NPY de réseau de calcul (ℓ + 2) et rend `P_c`, `W₂` et le profil, sur l'hôte ou sur le GPU (mêmes ordres de sommation, mêmes octets).
- Références, définitions exactes, gates et mesures de performance : `docs/BOITES.md`, `GATES-DIR.md`.

### cond, fibre, obs

```text
kvu cond chaine --cfg X.npy --beta b --graine s [--mode ref_rng|rapide] [--ntherm 100 --nsweep 6000 --every 2 --deach 5
                       --ckpt 500] --out DIR | --reprendre DIR
kvu fibre mesurer --cfg X.npy [--libre] --out DIR
kvu obs wilson   (--cfg X.npy | --ensemble ID) --out DIR [--rmax 4 --tmax 4 --dir-temps 3 --toutes]
kvu obs polyakov (--cfg X.npy | --ensemble ID) --out DIR [--dir-temps 3]
kvu obs flot     (--cfg X.npy | --ensemble ID) --out DIR [--dt 0.01 --tmax 3.0 --chaque 1 --cible-t2e 0.3 --cible-w 0.3]
kvu obs topo --ensemble DIR --out DIR --ens NOM --chaine K --backend-gen hip|cuda|cpu \
             [--backend auto|hip|cuda|cpu --flot-complet-chaque 1 --dt 0.01 --balayages 20 --pas-max 5000 --reprendre]
kvu obs topo-gates analyse --ens q|obs4|lit|lit_mut --jsonl F1,F2 [--lignes OUT.txt]
kvu obs topo-gates inst [--backend cpu|hip|cuda|auto]
kvu obs topo-gates cool-id [--backend cpu|hip|cuda|auto]
kvu obs topo-gates gpu --cfg F1,F2,F3 [--backend hip|cuda|auto]
kvu obs topo-gates certificat --lignes F1,F2 --out validation/topo/topo.json [--cmd "commande exécutée"]
kvu obs topo --ensemble DIR --out DIR --ens NOM --chaine K --backend-gen hip|cuda|cpu --sans-flot [--max-cfg N --reprendre]
kvu obs topo-gates analyse-v5 --ens q5|lit5|lit5_mut --jsonl F1,F2 [--lignes OUT.txt]
```

Avenant v5.WP19 (tranche 4) : `--sans-flot` mesure le cooling seul (aucun pas de flot ; `flot_a2` vide, `t0` et `flot_t0` absents) ;
`--max-cfg N` mesure au plus N configurations nouvelles puis s'arrête proprement (dernière ligne : `topo : M configurations mesurées
par cette passe, R restantes`), la suite par `--reprendre`. `obs topo-gates analyse-v5` émet TOPO-V5-INTEGRITE-<ens> puis TOPO-Q-V5,
TOPO-Q-MUT-V5, TOPO-COOL-V5 (`q5`), TOPO-LIT-V5, RECERT-GEN-lit5 et TOPO-LIT-MUT-OP (`lit5`), TOPO-LIT-MUT-V5 et RECERT-GEN-lit5_mut
(`lit5_mut`), avec leurs lignes INFO (GATES.md, bloc v5.WP19).

`obs topo` écrit et vide `topo.jsonl` après chaque configuration, dans l'ordre des numéros `cfg_NNNNNN.npy[.zst]`
(un sous-répertoire `c000/` est accepté). Le flot est complet par défaut ; utiliser `--flot-complet-chaque 10`
pour `lit` et `lit_mut`. Sans `--reprendre`, un JSONL existant est refusé. La reprise vérifie `ens`, `chaine` et
`backend_gen`, conserve les configurations déjà mesurées et récupère une éventuelle dernière ligne interrompue.
Le manifeste initial est conservé ; chaque reprise crée sa provenance dans `reprises/rNNNNNN/`.
Le backend `auto` suit `obs flot` (GPU à partir de L = 16, sinon CPU), avec admission mémoire et verrou GPU.
`obs topo-gates` émet une ligne par verdict : rc 1 si une ligne est `FAIL`, rc 0 pour `PASS`, `SKIP` et `INFO`.
Le certificat recopie ces lignes et liste les `FAIL` et `SKIP` dans `non_vert` ; `--cmd` peut être répété.

`cond chaine --mode ref_rng` rejoue à l'hôte les tirages numpy de cb.py (PCG64 + ziggurat) : à 1e-12 de cb.py par balayage, c'est la
loi de référence imposée par G-LOI(a). `--mode rapide` (Philox sur GPU) a la même loi, pas les mêmes tirages.

## 4. Codes de sortie (stables)

| code | sens |
|---|---|
| 0 | succès |
| 1 | échec (ou erreur interne d'un module) |
| 2 | usage, option ou configuration invalide |
| 3 | GPU indisponible (backend forcé absent, compilation en échec) ou faute d'exécution du GPU (voir ci-dessous) |
| 4 | entrée/sortie ou format de fichier |
| 5 | ressources refusées (RAM, charge, moniteur en alerte) : rien n'est tué, on refuse |
| 6 | verrou GPU non obtenu dans le délai |
| 7 | garde SSD : l'écriture ferait passer l'espace libre sous le seuil |
| 8 | validation ROUGE (une gate FAIL ou SKIP) |
| 9 | non implémenté |

Les erreurs sont écrites en français sur stderr (`kvu <sous-commande> : <message>`). Les erreurs d'usage de clap sont suivies de
`usage invalide : …`.

### Fautes d'exécution du GPU : jamais un code 0

Un noyau GPU qui fait une faute (accès mémoire illégal, échec de lancement) ne le dit pas quand on le lance : l'erreur
(`an illegal memory access was encountered (700)`) n'est rendue qu'à la synchronisation ou à la copie suivante. kvu la rend alors comme
toute erreur du GPU : message sur stderr (`kvu <sous-commande> : GPU : …`, suivi de `faute d'exécution GPU (code de sortie 3) : …`)
et **code de sortie 3**. En plus :

- tout code d'erreur du runtime reçu par `lancer`, `synchroniser`, `lire`, `ecrire`, `copier` ou `zero` est consigné dans un registre du
  processus (`gpu::sante`), même si l'appelant l'avale ; un appareil qu'on détruit fait une dernière synchronisation, dont l'erreur est
  consignée aussi ; à la sortie, **un registre non vide interdit le code 0** (message `FAUTE D'EXÉCUTION GPU relevée pendant la
  commande, qui rendait pourtant 0`, code forcé à 3 ; un code déjà non nul est conservé) ;
- **jamais de libération sous un noyau en vol** : chaque tampon et chaque module attend l'appareil (`hipDeviceSynchronize`, code de retour vérifié
  et consigné ; CUDA : `cuCtxSynchronize`, puis relecture des erreurs que cudarc a consignées dans ses `Drop`) AVANT d'être libéré ou déchargé,
  quel que soit l'ordre de destruction, y compris quand une erreur court-circuite la fin d'une commande (le déroulement des `Drop` libérait alors
  ses tampons sans avoir attendu les noyaux déjà lancés : faute de page côté données, ou, pour un module déchargé, côté instructions) ; la
  destruction de l'appareil fait de même ;
- 4096 lancements de noyaux de suite sans passer par `Appareil::synchroniser` en forcent une (`gpu::sante::LANCEMENTS_MAX_SANS_CONTROLE`) : une
  faute est relevée au plus tard après ce nombre de lancements, quelle que soit la commande ;
- `su2 gen` fait une dernière synchronisation avant de clore son manifeste (une faute des derniers noyaux donne `statut` = `echec`, pas
  `termine`) ; à la clôture réussie d'un run **repris** après une fenêtre en échec, la raison de l'échec passe de `raison_echec` à la liste
  `fautes_anterieures` : le manifeste ne dit plus « termine, rc 0 » à côté de la raison d'un échec.

Le code de sortie d'un lanceur est celui de la commande : `echo "$(date) rc=$?"` écrit le code de `date`, pas celui de kvu, car le shell
développe le mot de gauche à droite et le `$(…)` remet `$?` à 0 avant l'expansion de `$?`. Échapper les `$` dans un `bash -c "…"` n'y change
rien : le shell interne développe le même mot dans le même ordre. Lire le code dans une variable (`kvu … ; rc=$? ; echo "… rc=$rc"`) ou
dans le manifeste de l'exécuteur.

**Injection de pannes** (essais des lanceurs et de la recette ; elle ne peut que faire échouer, jamais simuler un succès) :
`KVU_INJECTION_FAUTE_GPU=<point>[:<rang>]` fait échouer le `<rang>`-ième appel (défaut 1) du point `lancement`, `synchronisation`,
`copie` ou `liberation` (l'attente de l'appareil qui précède une libération ou une destruction), ou, avec `sortie`, fait relever une faute par
le contrôle de sortie du processus (y compris pour une commande sans GPU). Le
message porte le texte du runtime, donc les détecteurs de journal des lanceurs se déclenchent comme sur une vraie faute. Une valeur
invalide donne le code 2 sans exécuter la commande (une faute de frappe ne désactive jamais l'injection en silence).

## 5. Formats

| fichier | format |
|---|---|
| `therm.csv` | `step,plaquette,P0,P1,P2,P3` puis `%ld,%.17g×5` (identique à su2_hip) |
| `measurements.csv` | `measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3` (phase `initial` ou `sample`) |
| `cfg_NNNNNN.npy[.zst]` | **c16** : complex128 `(4,L,L,L,L,2,2)` (compat su2_hip) ; **q64** : float64 `(4,L,L,L,L,4)`, 2× plus petit |
| `blocs.csv`, `therm_blocs.csv` | mode bord : `bloc,S0..S{L-1},T0..T{L-1}`, `%.17g` |
| `reprise.json` + `reprise.champs` | état de reprise (octets bruts, sha256) |
| `manifest.json`, `COMPLETE` | provenance (§6) ; `COMPLETE` est écrit en dernier, et seulement si rc = 0 |
| `F.npy`, `D.npy`, `blk.npy`, `controles.json` | sorties de `cond chaine` (`F (nmes,54)`, `D (nD,54,54)`, `blk (nmes,6,(L/2)^4)` float32) |
| `wilson.npy`, `flot.csv`, `flot.json` | sorties de `obs` |

Un lot de K > 1 chaînes écrit `manifest.json` et `COMPLETE` à la racine du lot et un sous-répertoire `c000/ … c{K-1}/` par chaîne
(`therm.csv`, `measurements.csv`, dumps). Taille d'une configuration : c16 = 256·L⁴ octets (L = 32 : 268 Mo ; L = 64 : 4,3 Go), q64 la
moitié. Tout se lit avec numpy (`np.load`) ; `reference/` conserve les sources d'origine.

## 6. Provenance et garde SSD

- **Manifeste** : argv, machine, backend, versions, précision, graines, `binaire_sha256`, `git_head`, `source_hash` (recette
  Unifiee, reproductible en Python : `crates/kvu/validation/historique/source_hash_unifiee.py`), `noyaux_embarques`, sources compilées
  avec leurs options et le sha256 du code objet, fichiers produits (nom, sha256, octets).
- **Garde SSD** : avant CHAQUE écriture lourde, et pour la projection totale d'un run au lancement. L'espace libre après écriture doit
  rester au-dessus du seuil (acemagic 100 Go, gamer 50 Go) ; sinon rien n'est écrit et le code de sortie est 7.
  `kvu store du --projeter L:CHAINES:DUMPS:FORMAT[:REPRISES]` dit AVANT de partir si la garde accepterait (ex. `64:1:11:q64:21`).
- Aux grands L, l'espace disque est la contrainte : d'où les trois défauts du grand L : `--format q64`, peu de dumps, mesures en
  ligne.

## 7. Store, fetch, file de jobs

```bash
kvu store du [--chemin DIR…] [--projeter 64:1:11:q64]   # occupation par ensemble, Go libres, marge, projection
kvu store gc [--appliquer] [--age-h 24] [--garder-chaque K] [--convertir-q64] [--zstd]
kvu store verifier ID ;  kvu store lister [--json] ;  kvu store exporter ID DEST [--format c16|q64]
kvu fetch gamer:/home/utilisateur/kevinotron_store/validation_unifie/gh06 ./gh06     # câble gamer-eth (192.0.2.2)
kvu fetch gamer:store/ID                                                 # un ensemble du store du gamer
```

- `store gc` **simule par défaut** ; `--appliquer` agit, jamais hors de la racine du store ; un ensemble contenant `.garder` est
  intouchable. Ordre : runs sans COMPLETE dont le pid est mort (> 24 h), reprises d'un run COMPLETE, un dump sur k, conversion c16 → q64
  (aller-retour exact vérifié).
- `fetch` : `du -sb` distant, garde SSD locale sur cette taille, `rsync -a --partial` sur `gamer-eth`, puis contrôle des sha256 du
  manifeste distant (un écart supprime le fichier fautif et rend une erreur). **Règle deux machines** : un résultat du gamer est de la
  RECHERCHE ; ce qui entre dans un verdict est rapatrié et revérifié sur acemagic.
- **File** : `kvu queue add [--machine acemagic|gamer] [--gpu] [--mem 8G] [--fils N] [--priorite P] [--sortie-go X] -- CMD…`,
  `queue list [--json]`, `queue run [--une|--boucle]`, `queue annuler ID`. La file respecte les verrous GPU (sans interblocage avec un
  `flock` externe), `nice`, `taskset`, `MemoryMax`, la RAM disponible, le moniteur de charge et la garde SSD. Elle ne tue jamais rien.

### Régénération des ensembles (`regen`, `store classer`)

```bash
kvu regen M.json --sortie DIR [--comparer ORIG] [--ancres anchors.sha256] [--preuve-gbit PREUVES] [--delai 600]
kvu regen M.json --adapter-seulement --vers M_regen.json [--hacher]   # traduit en kvu-regen/1 sans rejouer
kvu regen --adapter-ensemble ENSEMBLE --vers DIR                      # un manifeste kvu-regen/1 par chaîne
kvu store classer --chemin ENSEMBLE [--chemin …] [--preuve-gbit PREUVES] [--detail] [--json]
```

- **Ce que `kvu regen` rejoue (0.1.1, vérifié par exécution sur CPU)** : les manifestes `kvu-regen/1` de `su2 boite` et de
  `fibre echantillonner` (`DIR/manifest_regen.json` : `PASS REGEN`, fichiers identiques à l'octet). Il lit aussi, par adaptateur,
  les manifestes historiques `su2_hip/1` et `su2_gen/1` (rejeu sur GPU avec un binaire admis, non rejoué pour cette version). Il ne
  rejoue PAS un run `su2 gen` : `DIR/manifest.json` n'est pas reconnu (code 4) et `DIR/regen_bloc3.json` (`--mesures bloc:3`)
  échoue (`FAIL REGEN`, code 8 : la commande enregistrée garde `--outdir`, que `regen` ajoute une seconde fois) ; défaut PUB-15 de
  `docs/DEFAUTS_CONNUS.md`.
- **Parade pour `su2 gen` (vérifiée)** : un run est déterministe ; relancer la commande enregistrée dans son manifeste (clé `argv`),
  avec le même binaire `kvu` (clé `binaire`) et le même backend sur la même machine, dans un dossier neuf (`--outdir`) et sans
  `--arret-apres` ni `--fenetre-s` (un run repris garde l'`argv` de sa création), puis comparer aux sha256 consignés. Le script
  ci-dessous traite les deux syntaxes (`--outdir RUN` et `--outdir=RUN`, de même pour `--arret-apres` et `--fenetre-s`) et ne lance
  `kvu` qu'après une transformation réussie (sinon il s'arrête, code 1, sans rien lancer ; il refuse de même un `argv` de reprise,
  `--reprendre`, et une de ces options placée en fin d'`argv` sans sa valeur). `RUN` est le run d'origine, `NEUF` un
  dossier qui n'existe pas ; la seconde commande n'affiche que des `OK` (code 0) si tout est identique :

```bash
python3 - RUN NEUF <<'FIN'
import json, os, sys
run, neuf = sys.argv[1], sys.argv[2]
a = json.load(open(run + "/manifest.json"))["argv"][1:]
b, n, i = [], 0, 0
while i < len(a):
    nom, colle = a[i].split("=", 1)[0], "=" in a[i]
    if nom == "--reprendre":
        sys.exit("argv d'une reprise (--reprendre) : rien n'est lancé")
    if nom in ("--arret-apres", "--fenetre-s", "--outdir"):
        if not colle and i + 1 >= len(a):
            sys.exit(nom + " sans valeur : rien n'est lancé")
        if nom == "--outdir":
            b, n = b + ["--outdir", neuf], n + 1
        i += 1 if colle else 2
    else:
        b, i = b + [a[i]], i + 1
if n != 1 or os.path.exists(neuf):
    sys.exit("transformation impossible (--outdir absent ou répété, ou NEUF existe) : rien n'est lancé")
os.execvp("kvu", ["kvu"] + b)
FIN
python3 -c 'import json,sys;[print(f["sha256"]+"  "+sys.argv[2]+"/"+f["nom"]) for f in json.load(open(sys.argv[1]+"/manifest.json"))["fichiers"]]' RUN NEUF | sha256sum -c
```

- Manifeste `kvu-regen/1` (schéma : `schemas/kvu-regen-1.schema.json`, validateur `scripts/regen_valider.py`) : binaire, source,
  noyaux, graines, paramètres, commande exacte, fichiers (sha256, octets), classe A/B/C, `regen_verifiee`. Les manifestes existants
  `su2_hip/1` et `su2_gen/1` sont lus **tels quels** (adaptateurs), jamais modifiés.
- `regen` exige le binaire attendu par le manifeste, ou un binaire dont la G-BIT correspondante est PASS **pour lui** (fichier de
  preuves, `validation/regen/gbit_preuves.sh`) ; il rejoue **sous verrou GPU** dans un répertoire de sortie **neuf**, jamais sous
  l'ensemble d'origine, compare les sha256 fichier par fichier et écrit `regen_verifiee` dans une **copie** du manifeste.
  Code de sortie 0 (identique) ou 8 (différence) ; 2 si le binaire n'est pas admis ou la sortie interdite.
- Classes : **A** régénérable au bit (manifeste complet, binaire ancré ou G-BIT PASS, produit sur acemagic) ; **B** produit hors
  acemagic ; **C** binaire ou graines inconnus, adaptateur non reconnu, ou référence de la publication (à garder). Dans le doute, C.
- `store classer` ne modifie, ne déplace et ne supprime **rien** (simulation seulement).

## 8. Mods Claude Code : `etat --json`

`kvu etat --json` est le contrat de la bande d'état (verrou GPU, SSD, file) : une ligne, en moins de 200 ms.

```json
{"alerte_moniteur":false,"backend":"hip","charge1":14.72,"file":{"attente":0,"echec":0,"en_cours":0},
 "gpu_verrou":{"pid":2626162,"pris":true},"horodatage":"2026-10-04T01:11:18Z","machine":"acemagic",
 "ram_dispo_go":96.4,"ssd_libre_go":130.0,"ssd_seuil_go":100.0}
```

`gpu_verrou.pris` dit si un processus tient le GPU (production S2/S3 comprise), `ssd_libre_go` contre `ssd_seuil_go` donne la marge, et
`alerte_moniteur` repasse à vrai si le fichier `alerte` du profil (moniteur de charge externe) date de moins de 10 minutes.

## 9. Validation : `kvu valide`

```bash
kvu valide --rapide  [--json]    # CPU seulement, sans verrou ni gamer : G-H01, G-H02 et TESTS-CPU (tous les tests non ignorés de kvu)
kvu valide --complet [--json]    # toutes les gates de §6 sur les DEUX machines (partie gamer par ssh gamer-eth)
kvu valide --gate G-H06          # une gate (ou un préfixe : G-H0, LOI-)
kvu valide --complet --dry-run   # affiche le plan et l'environnement, ne lance rien
```

`valide --complet` et la plupart des gates de `crates/kvu/validation/` supposent les deux machines de référence de l'auteur
(ssh entre elles, binaires de production, ensembles sur disque) : leurs chemins par défaut sont neutralisés
(`/chemin/vers/projet/...`, voir `docs/PUBLICATION.md`). Ailleurs, la recette est `cargo test` ou `scripts/recette_locale.sh`.

La suite est `crates/kvu/validation/gates_unifie.sh`, qui orchestre `validation/<tâche>/gates_*.sh` (T1 à T8) et
`validation/historique/G-H01 … G-H13` (les 13 gates de `gates_gpu1.sh`, rejouées avec le moteur unifié). Chaque script écrit une
ligne `PASS|FAIL|SKIP <gate> [détail]` ; l'orchestrateur les agrège et ajoute **un FAIL pour chaque gate attendue qui n'a pas été
émise** (catalogue de `cli/catalogue.rs`). **Un SKIP n'est pas vert** : code de sortie 0 seulement s'il n'y a ni FAIL ni SKIP, sinon 8.
Sortie `--json` : `{"gates":[{"nom","machine","statut":"PASS|FAIL|SKIP","detail"}],"fail":n,"skip":n}`.

Verrou GPU : chaque script tourne dans UNE fenêtre (≤ 10 min) sous le verrou de sa machine, pris par l'orchestrateur ou par le script lui-même (s'il déclare « prend lui-même le verrou »), puis rendu. La partie gamer s'exécute EN PARALLÈLE (son verrou est distinct) ; G-H08 à G-H13, qui rapatrient et analysent ses ensembles, passent après sa fin. Attente du verrou : `KVU_VERROU_ATTENTE_S` (défaut 7200 s). Lancée sous `flock`, la suite ne se bloque
pas (un faux `flock` laisse passer les prises internes tenues par un ancêtre). `CARGO_TARGET_DIR` est fixé hors dépôt
(`…/kevinotron_dev/targets/T9`) ; le travail va sur le SSD (`…/kevinotron_dev/scratch`), jamais dans `/tmp` (RAM sur acemagic).

## 10. Bancs : `kvu bench`

```bash
kvu bench su2 --L 8,16,32,48 --precision f64,f32 --chaines 1,16 --budget-s 600 [--json] [--sortie F.json]
kvu bench cond  --L 16,24,32 [--mode rapide,ref_rng] [--budget-s 600]
kvu bench fibre --L 16,24,32 [--budget-s 600]
kvu bench obs   --L 16,24,32 [--budget-s 600]
kvu bench su2 --L 16 --dry-run         # plan, sans toucher au GPU
bash crates/kvu/validation/banc/banc_unifie.sh   # les quatre bancs sur la machine courante, puis rapport_banc.py → BANC_UNIFIE.md
```

Chaque banc se déroule sous le verrou GPU de la machine, par **fenêtres de 600 s au plus** (le verrou est rendu entre deux fenêtres) ;
`--budget-s` borne le total, et un cas non mesuré faute de budget est listé comme tel. **Gate de non-régression** : le débit F64
unifié (pas/s, une chaîne, options `compat-prod`) doit valoir au moins 0,9 × celui de su2_hip (acemagic) ou de su2_cuda (gamer),
lancé dans la même fenêtre au même L ; le code de sortie est 8 si elle est rouge. Le reste est rapporté, non gaté.

## 11. Exemples

`crates/kvu/exemples/` : chaque script a trois modes (`EXEMPLE_MODE=dry|petit|prod`) ; `dry` (défaut) affiche les commandes et les
estimations, `petit` les exécute à L ≤ 8 sur le CPU dans un répertoire jetable, `prod` lance les vraies commandes.

| script | contenu |
|---|---|
| `01_r2_3_s2_repliques.sh` | une cellule de prod_s2 (β = 3,0, L = 16, bords cold/hot/eq/eqg) en lot de 4 répliques, `su2 gen --bord` |
| `02_saturation_v3.sh` | génération de V (`su2 gen`, F64) puis `cond chaine --mode ref_rng` ; lecture de `F.npy`/`D.npy` pour torsion.py |
| `03_desert_L48_64.sh` | L = 64 en F32, dumps q64, `--mesures`, projection SSD (`store du --projeter`) et durées estimées |
| `04_balayage_beta.sh` | 16 valeurs de β en un seul lot |

## 12. Pièges connus

- **GPU partagé** : sur une machine où d'autres calculs utilisent le GPU, ne lancez pas un banc ou une gate sans le verrou du
  profil (`flock`), et ne le tenez pas plus de 10 minutes à la fois.
- **/tmp en RAM** : si `/tmp` est un tmpfs, la garde SSD y refuse les grosses écritures ; écrire sur un disque.
- Un run F32 ne s'exécute pas sur le CPU (F64 seulement).

### rg flot (estimateurs de couplage effectif, WP8)

```text
kvu rg flot --ensemble DIR [--filtre L12_b2.3_c*] [--schema balaban|swendsen|ape|dec] [--pas 1] [--estimateurs sd,plaq,mcrg2] [--courbe etal_L4.json] [--sortie DIR] [--mutant-u1] [--fils N]
kvu rg etalonner --L 4 [--independante] [--facteur-beta 1.05] --sortie etal_L4.json [--fils N]
kvu rg flot-gates [--donnees D] [--grandl D] [--etalonnages D] [--seul RG-SD,RG-TAU] [--sans-mutants] [--json] [--certifier [FICHIER]] [--fils N]
```

`flot` lit des chaînes (dossiers `cfg_*.npy` + `manifest.json`, regroupées par β), bloque chaque configuration `--pas` fois (Bałaban L_b = 3, ou les schémas de facteur 2 : `swendsen` = témoin de
`block_config`, `ape` et `dec` = ceux de `n3_blocage_beta_eff.py`, qui sont des schémas DIFFÉRENTS) et écrit `<sortie>/flot.json` (`kvu-rg-flot/1`, refus d'écraser, identique à l'octet quel que soit
`--fils`) : par cellule (β, L), les estimateurs `sd` (Schwinger–Dyson à un lien, rapport I₂/I₁ exact, bissection), `plaq` (β′ tel que ⟨P⟩_Wilson(β′ ; L′) = ⟨P̄⟩ sur la courbe d'étalonnage, PCHIP monotone,
**jamais d'extrapolation** : « hors étalonnage » sinon) et `mcrg2` (plaquette 1×1 et rectangle 1×2 ; écart des deux β′ = systématique de schéma), avec erreurs jackknife par chaîne, τ_int (fenêtre de Sokal, c = 6),
fraction hors domaine et la quantité cible `a_j = β′ − β − 0,4081` (rapportée, jamais une gate). `etalonner` produit les courbes d'étalonnage (protocole gelé : 14 β, 3 chaînes, mesures en ligne ; `--independante` :
chaîne de ré-estimation de RG-PLAQ ; `--facteur-beta` : mutant). `flot-gates` émet une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate (RG-BESSEL, RG-TAU, RG-N3, RG-SD, RG-PLAQ, RG-SCHEMA rapporté, RG-FILS-FLOT, et leurs mutants).
Données et protocole : `validation/rg_flot/README.md` ; données de test : `tests/donnees/rg_flot/`.

### fibre echantillonner

Échantillonne une fibre de Bałaban (réseau fin L multiple de 3, au moins 6) ou un jouet.
Le départ est projeté sur la cible avant les deux routes. `--kappa inf` retire
la pénalité et permet un réseau pair sans cible. Les sorties CSV, NPY q64,
`resume.json` et `manifest_regen.json` sont déterministes ; `manifest.json`
porte la provenance datée et `COMPLETE` est écrit en dernier. La sortie doit
être neuve. `--outdir` est synonyme de `--sortie` pour le rejeu par `kvu regen`.

```sh
kvu fibre echantillonner --methode rattle --entree-v V.npy --entree-u0 U0.npy \
    --L 6 --beta 2.3 --dt 0.02 --duree 0.2 --therm 10 --nmeas 100 --skip 1 \
    --chaines 1 --graine 26102612001 --fils 4 --dump-tous 10 \
    --tol-newton 1e-13 --max-newton 50 --tol-rev 1e-9 --sortie DIR
kvu fibre echantillonner --jouet deux-liens --t 0.3 --beta 2.3 \
    --methode penalite --kappa 2e-3 --delta 0.05 --nmeas 100 --sortie JOUET
kvu fibre echantillonner --jouet courbe --masse 1,2 --methode rattle --nmeas 100 --sortie COURBE
```

Les chaînes de pénalité se répartissent sur les fils CPU ; les chaînes RATTLE
sont séquentielles et chaque modèle utilise les fils demandés. La recette de
rejeu fixe `--fils 1` : ce réglage ne change aucun résultat numérique. Dans le
CSV pénalité, `accepte` vaut 1 si le balayage a accepté au moins un lien ; le
taux d'acceptation du résumé compte chaque proposition locale. Les résumés
portent sur toutes les propositions après thermalisation, y compris celles
entre les mesures ; les ΔH non définis des rejets techniques sont exclus de
la moyenne. Les durées sont affichées seulement sur stderr.

## Extensions WP9 du flot et de l'étalonnage

`rg flot` accepte `--backend cpu|hip|cuda|auto` (défaut `cpu`). Les backends GPU
exigent le schéma `balaban` et utilisent le même verrou de machine que `rg bloc`.
Le GPU est ouvert une fois ; seuls les blocs sont calculés sur GPU, les estimateurs
restent sur CPU. La provenance GPU figure dans `calcul`.

`--bins K` (ou `--bins 4,8`, tailles ≥ 1 sans doublon) ajoute un jackknife par blocs de configurations consécutives dans
chaque chaîne. Le reste rejoint le dernier bloc ; aucune configuration n'est jetée.
Le jackknife par chaîne et les valeurs centrales restent inchangés. Les erreurs
supplémentaires figurent dans `jackknife_bins`, `erreur_jackknife_bins`,
`erreur_totale_bins` et `a_j.erreur_bins`. Un réplicat indéfini donne `null`.
La première taille conserve ces clés ; `bins.tailles` et
`jackknife_bins_tailles` donnent toutes les tailles dans l'ordre demandé,
avec effectifs, erreurs, moyennes et réplicats (aussi pour l'écart de schéma).
`--detail-configs` ajoute P, R, sommes SD et domaine du premier pas par fichier.
Sans ces options, aucun champ supplémentaire n'est écrit.

```sh
kvu rg flot --ensemble ensembles/production --schema balaban --backend auto --bins 10 --detail-configs --courbe etal_wp9_L4.json --sortie flot_wp9
```

`rg etalonner --protocole wp9` étend la grille à 31 β entre 0,25 et 64, pour
L = 4 (20 000 mesures) ou L = 8 (10 000 mesures), avec trois chaînes,
2 000 pas de thermalisation et quatre OR. Les bases de graines sont respectivement
26 102 620 000 et 26 102 624 000 ; la formule reste base + 100 i + c.
Le protocole par défaut `wp8` conserve sa grille et son JSON. Une courbe WP9
porte `protocole: wp9` et s'utilise avec `rg flot --courbe`.
`--milieux` produit une chaîne c = 8 à chacun des 30 milieux de la grille,
avec les mêmes graines de base, thermalisation et mesures. Son JSON porte
`milieux: true` et ne peut pas servir de `--courbe` ; ce mode est incompatible
avec `--independante` et toute option explicite `--facteur-beta`.

```sh
kvu rg etalonner --protocole wp9 --L 4 --sortie etal_wp9_L4.json
kvu rg etalonner --protocole wp9 --L 4 --independante --sortie indep_wp9_L4.json
kvu rg etalonner --protocole wp9 --L 4 --independante --facteur-beta 1.05 --sortie mutant_wp9_L4.json
kvu rg etalonner --protocole wp9 --L 4 --milieux --sortie milieux_wp9_L4.json
```

`rg etalonnage-gates` vérifie les protocoles, les 31 points indépendants en P et R
(max |z| < 3,59 et p du χ² > 0,01, survie exacte avec 31 degrés de liberté),
et le rejet du mutant facultatif. Les lignes INFO rapportent les écarts
PCHIP/linéaire, l'interpolation laisser-un-de-côté (en P et en β) et, avec
`--comparer`, le recouvrement P et R avec un autre protocole au même L.
Le code de retour est 0 sans FAIL, sinon le code de validation 8.
Avec `--milieux M.json`, les lignes `RG-PLAQ-WP9-MILIEU-P-L<L>` et
`RG-PLAQ-WP9-MILIEU-R-L<L>` valident les 30 mesures contre PCHIP en sens direct.
L'erreur de courbe est le maximum des erreurs des deux nœuds voisins ; les
seuils sont 3,59 et p > 0,01 (30 degrés de liberté), sous protocole vérifié.

```sh
kvu rg etalonnage-gates --courbe etal_wp9_L4.json --independante indep_wp9_L4.json --mutant mutant_wp9_L4.json --milieux milieux_wp9_L4.json --comparer etal_L4.json --json
```

## Blocage quasi-local (WP22)

Les opérateurs `dec`, `ape:α:n`, `flot:t:ε` et `agr:c` bloquent par un facteur 3.
Leurs définitions, points d'attache, règles de famille et supports exacts figurent dans
[docs/BLOCAGE_QUASI_LOCAL.md](../../docs/BLOCAGE_QUASI_LOCAL.md).

```sh
kvu rg bloc --entree validation/rg/fixtures/cfg_s7_q64.npy --sortie bloc_ape6.npy --operateur ape:0.5:6 --fils 6
kvu rg bloc --entree validation/rg/fixtures/cfg_s7_q64.npy --sortie bloc_ape40.npy --operateur ape:0.5:40 --hors-famille --fils 6
kvu rg flot --ensemble ensembles/production --operateur ape:0.5:6 --estimateurs sd --detail-configs --sortie flot_ape6 --fils 6
kvu rg ql-gates --fixtures validation/rg/fixtures --references validation/rg_ql/references --seul RG-QL-NORM --fils 6 --json
kvu rg ql-gates --fixtures validation/rg/fixtures --references validation/rg_ql/references --grandl ensembles/production --fils 6 --certifier validation/rg_ql/rg_ql.json
```

Un opérateur hors famille exige `--hors-famille`, consigné au manifeste ; une spécification
invalide reste refusée. Le manifeste quasi-local porte `kvu-rg-bloc-ql/1`. Le flot avec
un opérateur quasi-local contient `operateur`, sans `domaine` ni `a_j`.
Le défaut `3centre` conserve les sorties historiques.

`ql-gates` émet 51 verdicts (`--sans-mutants` omet les mutants, `--seul` filtre par
préfixes séparés par des virgules). Sans `--grandl`, les cinq gates L12 sont SKIP et
le code de retour vaut 8. Le certificat `kvu-rg-ql-cert/1` exige la liste complète,
chaque identifiant exactement une fois avec PASS ; `--seul` et `--sans-mutants`
sont incompatibles avec `--certifier`. PASS sur un mutant signifie que son défaut
est détecté. Les verdicts ne contiennent ni durée ni chemin temporaire ; seule la
ligne `INFO RG-QL-BILAN` rapporte le temps d'exécution.

La liste fermée A.3 de l'avenant v5.WP22 (GATES.md) contient cinq identifiants L12 sur 51,
soit 46 verdicts hors L12.

Instrument de réponse WP22-B (CPU) : après validation complète du livrable A
(`rg ql-gates --grandl … --certifier …`, 51 PASS), les commandes suivantes calculent
les réponses par différences centrées et exécutent les sept contrôles gelés :

```sh
kvu rg reponse --entree crates/kvu/validation/rg/fixtures/cfg_s7_q64.npy --operateur ape:0.5:6 --echantillon 16 --pas 0.001 --fils 4 --detail --sortie reponse.json
kvu rg reponse-gates --fixtures crates/kvu/validation/rg/fixtures --fils 4 --json
```

Sans `--echantillon`, les 324 liens du bloc sont utilisés. Le JSON
`kvu-rg-reponse/1` contient les agrégats par distance de plaquettes et par classe
euclidienne de largeur 0,5, avec l'erreur estimée ; `--detail` ajoute toutes les
paires. Il ne dépend pas du nombre de fils. Une sortie existante est refusée.
Les opérateurs hors famille exigent `--hors-famille`, comme `rg bloc`.
