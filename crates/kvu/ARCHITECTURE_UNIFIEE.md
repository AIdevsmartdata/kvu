# kevinotron unifié — architecture (04/10/2026)

> **Note de publication (novembre 2026).** Document de conception d'origine, conservé tel quel : il a été écrit quand `kvu`
> était un crate du moteur interne de l'auteur (`kevinotron`), dont le binaire lui déléguait ses sous-commandes. Dans ce
> dépôt autonome, il n'y a que le binaire `kvu` ; les mentions du moteur historique, de ses worktrees et de ses lots (K1…K8)
> décrivent l'origine du code, pas ce dépôt. Les chemins privés y sont neutralisés (`docs/PUBLICATION.md`). Mode d'emploi
> actuel : `GUIDE.md` ; vue d'ensemble : `README.md` à la racine.

Architecte : Opus 5.5. Branche de base : `unifie/base` (depuis main d31d8ab), worktree
`/chemin/vers/projet/kevinotron_dev/wt/base`. Ce document fait foi pour les 9 tâches T1-T9.
Copie identique : `crates/kvu/ARCHITECTURE_UNIFIEE.md` dans chaque worktree.

---

## 0. En dix lignes

1. **Un crate neuf `crates/kvu`** (membre du workspace), léger à compiler. Le binaire `kevinotron` lui délègue
   ses nouvelles sous-commandes (`doctor`, `su2`, `cond`, `obs`, `store`, `fetch`, `queue`, `valide`, `bench`…).
   Le moteur historique (196 kLOC, 49 noyaux) reste intact.
2. **La machine est reconnue à l'exécution** : acemagic → HIP gfx1151, gamer → CUDA sm_120, sinon CPU.
   `KEVINOTRON_BACKEND` force le choix, et un backend forcé mais absent est une erreur, jamais un repli silencieux.
3. **Les noyaux GPU ont une source unique** (`crates/kvu/noyaux/`). Chaque source = prélude du backend + fichiers,
   compilée à l'exécution par hiprtc ou NVRTC. Côté CUDA, on produit du **CUBIN** et non du PTX : le pilote 13.2
   du gamer est plus ancien que NVRTC 13.3.
4. **FP64 de référence**, bit-identique à su2_hip / su2_cuda / su2_gen (K1) à options numériques égales.
   **FP32 rapide**, qui prouve la même loi par des tests gelés (LOI-1 à LOI-4).
5. **Profil mesuré** (§2) : le bain thermique est limité par le **calcul FP64** sur les deux GPU. Passer tout en FP32
   donne ×6,4 sur acemagic et ×8,8 sur le gamer à L = 32. Stocker en FP32 sans changer l'arithmétique ne rapporte rien.
6. **Bain conditionnel « fin sachant grossier » sur GPU.** Le mode `ref_rng` rejoue à l'hôte, à l'identique, le
   générateur numpy de cb.py (PCG64 + ziggurat) pendant que le GPU calcule les agrafes : on reste à 1e-12 de cb.py.
   Le mode `rapide` est un Philox à compteur entièrement sur GPU, de même loi.
7. **Lots multi-chaînes** : une chaîne d'un lot donne les mêmes octets que seule. S'y ajoutent le bord gelé R2-3,
   la reprise à l'octet et les mesures en ligne (Wilson, flot), qui évitent de stocker les configurations des grands L.
8. **Provenance** : le manifeste hache le binaire, les 11 noyaux embarqués, chaque source assemblée et compilée,
   ses options et le sha256 du code objet. Une **garde SSD** refuse d'écrire sous un seuil (acemagic 100 Go, gamer 50 Go).
9. **`store du/gc`** (simulation par défaut), **`fetch`** par le câble 2,5 Gbit/s avec vérification des sha256, et
   **file de jobs** sur les deux machines. Celle-ci respecte les verrous GPU (sans interblocage avec un `flock` externe),
   nice, taskset, MemoryMax et le moniteur de charge.
10. **Suite `kevinotron valide`** : les 13 gates historiques (G-H01..G-H13) plus ~40 gates neuves, aux seuils gelés ici (§6).

---

## 1. État des lieux (lu le 04/10)

| brique | où | état | limite |
|---|---|---|---|
| moteur historique | `/chemin/vers/projet/kevinotron` (main d31d8ab) | 15 groupes, CPU + CUDA (cudarc/NVRTC, `src/gpu`) + HIP (`src/hip`, 23 pilotes) ; `Backend::detect()` Cuda→Hip→Cpu | dispatch en 3 couches dans un `main.rs` de 8 kLOC ; aucun pilote SU(2) quaternionique hors K1 |
| su2_gen (K1) | `src/bin/su2_gen.rs`, `src/hip/su2_sweep.hip`, `src/su2_hb.rs` | HIP bit-identique à su2_hip.cpp compilé `-ffp-contract=off` (gate g1c), référence CPU | HIP seulement, une chaîne, FP64 |
| su2_hip | chantier wilson_gpu_grandL_20260926 | production AMD (hipcc par défaut : contraction fast-honor-pragmas) | autonome, hors dépôt |
| su2_cuda | gamer `~/ym4_calcul/su2_cuda` | portage mécanique (5 lignes), 13/13 gates (gpu_port) | autonome |
| su2_bord | R2_3_influence_bord/calc (binaire 5af3ea49, prod S2) | bord gelé, jauge, tranches ; C0 bit-identique à su2_cuda | CUDA seulement |
| cb.py (noyau2) | `/chemin/vers/projet/outils/ym4_noyaux/noyau2` | numba CPU, `ref_rng` à 1e-12 de lib_Cb, `rapide` même loi | CPU seulement ; saturation v3 impose `ref_rng` (G-LOI(a)) |
| manifeste K8 | `src/manifest.rs` | recettes BuildSh (aveugle aux 49 noyaux) et Complete | ne sait rien des sources RÉELLEMENT compilées ni des options |

Les deux générateurs SU(2) calculent tout en FP64 (`struct Q {double x,y,z,w}`), avec Philox4x32-10 à compteur
`(lien, sweep, tirage, étiquette)` et la clé `(graine basse, graine haute)`.

## 2. Profil (mesuré le 04/10, avant toute conception)

Microbanc `prof.cpp` (copié ici) : noyaux OR et bain thermique (KP + Philox) du port K1, en templates
stockage S × arithmétique C, L⁴, damier, blocs de 256. Chaque banc a tourné sous le verrou de sa machine
(fenêtre < 2 min). Journaux : `prof_acemagic.log`, `prof_gamer.log`. Valeurs en liens/s.

| L | noyau | machine | S64C64 | S64C32 | S32C64 | S32C32 | C32 seul | S32 seul | tout FP32 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | HB | acemagic | 1,58e8 | 4,59e8 | 1,51e8 | 1,01e9 | ×2,90 | ×0,95 | **×6,40** |
| 32 | OR | acemagic | 3,64e8 | 4,61e8 | 3,14e8 | 1,07e9 | ×1,27 | ×0,86 | ×2,95 |
| 32 | HB | gamer | 3,04e8 | 1,56e9 | 2,66e8 | 2,67e9 | ×5,13 | ×0,88 | **×8,79** |
| 32 | OR | gamer | 5,35e8 | 1,44e9 | 4,24e8 | 2,67e9 | ×2,68 | ×0,79 | ×4,99 |
| 16 | HB | acemagic | 1,36e8 | 5,19e8 | 1,28e8 | 8,90e8 | ×3,83 | ×0,95 | ×6,57 |
| 16 | HB | gamer | 2,59e8 | 1,24e9 | 2,28e8 | 2,03e9 | ×4,80 | ×0,88 | ×7,86 |

Contrôle de cohérence : le HB FP64 du microbanc (1,58e8 liens/s à L = 32 sur acemagic) concorde avec le banc K1
(`liens_par_s_hb = 1,74e8`, su2_hip). Sur le gamer, 3,04e8 contre 2,60e8 pour su2_cuda.

**Verdict** :
- le bain thermique est **limité par le calcul FP64** sur les deux machines. FP32 seul en stockage ne gagne rien
  (×0,86-0,95 : conversions) ; FP32 en arithmétique gagne ×2,9 (acemagic) à ×5,1 (gamer) ;
- une fois l'arithmétique en FP32, la bande passante devient visible : tout FP32 donne ×6,4 à ×8,8 sur le HB et
  ×3 à ×5 sur l'OR ;
- **décision** : le chemin rapide est **tout FP32** (float4, FMA permise). Il n'y a pas de variante
  « stockage FP32 + calcul FP64 ». Les observables sont accumulées en FP64 (lecture float promue) ;
- estimation pour un pas (1 HB + 4 OR) à L = 64 : acemagic FP64 ≈ 1,2 s, acemagic FP32 ≈ 0,32 s, **gamer FP32 ≈ 0,13 s**
  (16 Go suffisent : le champ FP32 à L = 64 pèse 1,07 Go).

## 3. Machines et politique

| | acemagic | gamer | inconnue |
|---|---|---|---|
| détection | hôte `acemagic` ou CPU « AMD RYZEN AI MAX+ 395 » ou gfx1151 visible (ROCm : /opt/rocm714) | hôte `pc-nvidia-exemple` ou CPU « i5-14600KF » ou sm_120 visible | sinon |
| backend auto | HIP (gfx1151, FP64 1/16, mémoire unifiée : GTT) | CUDA (sm_120, FP64 1/64, 16 Go) | premier GPU utilisable, sinon CPU |
| verrou GPU | `/tmp/kevinotron_gpu.lock` | `/tmp/gamer_gpu.lock` (sur le gamer) | `/tmp/kevinotron_gpu.lock` |
| attente verrou / fenêtre de banc | 7200 s / 600 s | 7200 s / 600 s | idem |
| nice / affinité | 10 / — | 5 / `taskset -c 12-19` (cœurs E, OBLIGATOIRE) | 10 / — |
| fils rayon max | 8 | 8 | 4 |
| RAM dispo minimale avant job | 25 Go | 6 Go | 8 Go |
| MemoryMax (jobs > 4 Go) | 24G | 12G | 8G |
| garde SSD (libre minimal APRÈS écriture) | **100 Go** | **50 Go** | 20 Go |
| store par défaut | `/chemin/vers/projet/kevinotron_store` | `~/kevinotron_store` | `~/kevinotron_store` |
| file par défaut | `~/.local/state/kevinotron/file` | idem | idem |
| moniteur | `/chemin/vers/projet/outils/machines/ALERTE.txt` (récent ⇒ refus) | — | — |
| câble vers l'autre | `gamer-eth` (192.0.2.2) | — (acemagic tire) | — |
| précision conseillée en production | F32 (désert, saturation) ; F64 pour les gates | F32 | F64 |

Surcharges : `KEVINOTRON_MACHINE`, `KEVINOTRON_BACKEND`, `KEVINOTRON_SSD_MIN_GO`, `KEVINOTRON_STORE`,
`KEVINOTRON_FILE`, `KEVINOTRON_CACHE_DIR`, `KEVINOTRON_HIP_ARCH` (historique), `KEVINOTRON_VERROU=ignorer`
(tests CPU seulement, consigné dans le manifeste). **Règle deux machines** : un résultat du gamer est de la
RECHERCHE. Ce qui entre dans un verdict est revérifié sur acemagic (intégrité des dumps, lois, gates G-H08..G-H13).

## 4. Architecture logicielle

### 4.1 Dépôt et crates

```
kevinotron/ (workspace)
├── Cargo.toml            racine : + dépendance kvu, feature `unifie-cuda = ["gpu", "kvu/cuda"]`     [T9]
├── build.rs, src/manifest.rs   + recette `Unifiee` (Complete + crates/kvu/{src,noyaux}/**, Cargo.toml, build.rs)  [T8]
├── src/main.rs           + 6 lignes : `argv[1] ∈ kvu::cli::SOUS_COMMANDES` ⇒ `kvu::cli::principal(argv)`   [T9]
├── src/hip/backend.rs    `Backend::detect()` honore KEVINOTRON_BACKEND (sinon cascade inchangée)      [T9]
├── scripts/build_auto.sh build selon la machine                                                   [T1]
└── crates/kvu/
    ├── Cargo.toml, build.rs (source_hash Unifiee injecté, T8)
    ├── ARCHITECTURE_UNIFIEE.md, SOURCES_IMPORTEES.md, GUIDE.md (T9), exemples/ (T9)
    ├── reference/        copies à l'octet (su2_hip.cpp, su2_cuda.cu, hip2cuda.h, su2_bord.cu, cb.py, gates)  [gelé]
    ├── noyaux/           11 sources GPU (registre gelé, contenu par tâche)
    ├── src/{commun,heritage,noyaux}.rs                                                              [gelé]
    ├── src/machine/ [T1]  src/gpu/ [T2]  src/su2/ [T3, rapide32.rs T4]  src/cond/ [T5]
    ├── src/fibre/ [T6]    src/obs/ [T7]  src/stockage/ [T8]  src/cli/ + src/bin/kvu.rs [T9]
    ├── tests/<tâche>_*.rs, tests/donnees/<tâche>/ (L ≤ 8, ≤ 5 Mo par tâche, SHA listés)
    └── validation/<tâche>/ (scripts de gates) ; validation/{gates_unifie.sh,historique,banc}/ [T9]
```

`heritage.rs` réutilise PAR CHEMIN (`#[path]`, sans copie) `src/hip/{ffi,wrap}.rs`, `src/su2_hb.rs` et
`src/manifest.rs`. Leurs gates (K1, K8, M1) restent donc valables, et aucun code n'est dupliqué.
Phase 2, hors lot : migrer les 49 noyaux historiques sur `kvu::gpu`.

**Dépendances** (toutes déjà dans Cargo.lock, après l'ajout de serde/serde_json par l'architecte) : rayon,
libloading, clap, serde, serde_json, cudarc 0.19 en option (mêmes features que la racine, obligatoire car le
workspace unifie les features). **Aucune nouvelle dépendance** sans accord. zstd passe par le binaire système
(présent sur les deux machines).

### 4.2 Interface de ligne de commande

Chaque module expose `cli::commandes() -> Vec<clap::Command>` et `cli::executer(nom, &ArgMatches) -> Resultat<i32>`.
`cli::principal(argv)` les assemble, affiche les erreurs en français et rend les codes stables de `commun::code` :
0 ok, 1 échec, 2 usage, 3 GPU, 4 E/S, 5 ressources, 6 verrou, 7 SSD, 8 validation, 9 non implémenté.

```
kevinotron doctor [--json] [--test]          machine, GPU, backend + raison, versions, politique, SSD, store, file, cache
kevinotron etat --json                       une ligne JSON (mods, supervision) — §4.11
kevinotron su2 gen --L 16 --beta 2.3[,2.8…] --graine S[,S2…] [--chaines K] --start hot|cold --therm N --nmeas N
           --skip 1 --nor 4 [--precision f64|f32] [--backend auto|hip|cuda|cpu] [--options ref-k1|compat-prod|rapide]
           [--dump-every N --dump-initial --format c16|q64 --zstd] [--mesures SPEC] [--reprise-every N]
           [--bord --bc cold|hot|eq --bc-graine G --jauge G --pre N --bloc N] --outdir DIR | --reprendre DIR
kevinotron su2 test-hb ALPHA N --graine S [--precision …]       ligne TESTHB (format su2_hip)
kevinotron su2 selftest                                          KAT Philox, bornes u01
kevinotron cond chaine --cfg X.npy --beta b --graine s --mode ref_rng|rapide [--ntherm 100 --nsweep 6000 --every 2
           --deach 5 --ckpt 500] --out DIR | --reprendre DIR
kevinotron fibre mesurer --cfg X.npy [--libre] --out DIR
kevinotron obs wilson|polyakov|flot --cfg X.npy|--ensemble ID [paramètres] --out DIR
kevinotron store du|gc|verifier|lister|exporter …     kevinotron fetch gamer:<chemin|store/ID> [DEST]
kevinotron queue add [--machine acemagic|gamer] [--gpu] [--mem 8G] [--fils N] [--priorite P] [--sortie-go X] -- CMD…
kevinotron queue list|run [--une|--boucle]|annuler ID
kevinotron valide [--rapide|--complet|--gate NOM] [--json]       kevinotron bench su2|cond|fibre|obs [--budget-s 600]
```

Compatibilité : `su2 gen` sans les options neuves écrit EXACTEMENT les fichiers de su2_hip : `therm.csv`,
`measurements.csv` (en-têtes et `%.17g` identiques), `cfg_NNNNNN.npy` (c16), `manifest.json` (schéma unifié, voir
§4.9) et `COMPLETE` en dernier. Un lot de K chaînes écrit `c000/ … c{K-1}/`, un sous-répertoire par chaîne.

### 4.3 Runtime GPU (T2)

- `gpu::ouvrir(backend, index) -> Appareil`, puis `compiler(&SourceAssemblee, &OptionsCompil) -> Module`,
  `fonction`, `alloc::<T: Pod>` (mise à zéro), `ecrire/lire/copier/zero`, `lancer(&Fonction, &Lancement, &[Arg])`,
  `synchroniser`. API **synchrone par défaut** et flux unique (même discipline que K1).
- **HIP** : `heritage::wrap` (hiprtc), options `--offload-arch=<arch> -O3 -ffp-contract=<…> -std=c++17`.
- **CUDA** : cudarc (driver) ; NVRTC via `cudarc::nvrtc::sys` avec `-arch=sm_120 --fmad=<…> -O3 -std=c++17`,
  `nvrtcGetCUBIN`, puis `Ptx::from_binary(cubin)`. La bibliothèque NVRTC est cherchée dans
  `LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64` (cudarc tente `libnvrtc.so` en premier).
- `assembler(backend, nom, &[&Fichier], defines)` = prélude + `#define` + fichiers, dans l'ordre. Le texte est
  déterministe, et son sha256 entre dans le manifeste et dans la clé du cache.
- `OptionsCompil` : `reference_k1()` (HIP `-ffp-contract=off`, CUDA `--fmad=false`), `compat_production(b)` (HIP
  fast-honor-pragmas comme hipcc, CUDA `--fmad=true` comme nvcc) et `rapide()`. `arguments(backend, arch)` est pur.
- Cache des codes objets : `~/.cache/kevinotron/kvu/<nom>-<arch>-<sha16>.{hsaco,cubin}` (+ `.json`), plafond
  256 Mo (LRU), écriture atomique. Une entrée corrompue est évincée et recompilée.

### 4.4 Noyaux à source unique et bit-identité (T3, T4)

- `su2_commun.h` : Philox4x32-10, `u01`, `struct Q`, `qmul/qdag/qadd/qnorm/qscale`, `Geo/mkgeo`, `nb_up/nb_dn`,
  `coords`, `site_of`, `staple`. Le texte est **recopié caractère pour caractère de `src/hip/su2_sweep.hip`** (sha da154635),
  sauf l'`#include` (passé dans le prélude).
- `su2_ref64.cu` : noyaux K1 étendus, sans changer l'arithmétique :
  - **lots** : `blockIdx.y` = chaîne, le champ de la chaîne c commence à `c·4V` ;
  - **compteurs PAR CHAÎNE** : l'indice de lien du compteur est l'indice LOCAL `li = mu·V + s`, la clé et β sont
    lus dans des tableaux par chaîne ;
  - **gel** : argument `int gel`, avec `if (gel && mu != 0 && x[0] == 0) return;`, placé comme dans su2_bord.cu ;
  - les mesures gardent l'arbre fixe `k_reduce`/`k_final` de su2_hip.
- Les expressions flottantes sont celles de su2_hip.cpp, dans le même ordre d'évaluation. La bit-identité se juge
  **à options égales** :
  - **HIP `reference_k1`** = su2_gen K1 = su2_hip.cpp compilé `-ffp-contract=off` (g1c) ;
  - **HIP `compat_production`** = su2_hip de production (hipcc, fast-honor-pragmas) : à vérifier, G-BIT-HIP-PROD ;
  - **CUDA `compat_production`** (`--fmad=true`, CUBIN NVRTC 13.3 sm_120) = su2_cuda (nvcc 13.3) : à vérifier,
    G-BIT-CUDA. Même NVRTC et même ptxas, donc même code attendu.
- **Repli gelé** si une identité à l'octet échoue :
  - le départ `cfg_000000` reste identique à l'octet ;
  - après un pas, max|Δq| ≤ 1e-12 avec **0 décision divergente** (diagnostic `diag_ntry`) ;
  - l'écart est expliqué : contraction FMA ou ULP de la libm device ;
  - l'affirmation devient alors « bit-identique à K1/fpoff ; 1e-12 par pas face au binaire de production ».
  C'est la tolérance « 1e-15 justifiée » demandée, appliquée au départ (les tirages Philox sont des entiers identiques).
- `su2_bord.cu` : `k_set_bord`, `k_gen_jauge`, `k_app_jauge`, `k_plaq_st`, `k_tranche` de su2_bord.cu (sha 0a01a43e),
  mêmes expressions et mêmes étiquettes (`TAG_BORD`, `TAG_JAUGE`, slots 0xFFFFFFFE / 0xFFFFFFFD).
- `su2_f32.cu` (T4) : `struct QF {float x,y,z,w}`, même Philox (entiers identiques), `u01f` sur 24 bits =
  `((a >> 8) + 0.5f) · 2⁻²⁴` (jamais 0 ni 1), KP / Creutz en float (`logf`, `cosf`, `__expf` interdit : précision IEEE),
  réunitarisation de chaque lien écrit, mesures promues en double et réduites à arbre fixe.

### 4.5 Générateur SU(2) (T3 ; FP32 : T4)

`su2::construire(app, ConfigLot) -> Box<dyn NoyauxSu2>` choisit `Ref64Gpu`, `Rapide32Gpu` ou `Cpu64`. Le trait
(signatures gelées, `src/su2/mod.rs`) est : `init`, `fixer_gel`, `sweep_hb`, `sweep_or`, `mesurer` (plaquette + P0..P3),
`mesurer_tranches` (S_t, T_t), `historique*`, `telecharger/televerser`, `champ_gpu`, `compteurs`,
`sweep_id/fixer_sweep_id`, `provenance`.

- **Invariant des lots (LOT-1)** : la chaîne c d'un lot donne les mêmes octets que la même chaîne lancée seule.
- **Reprise (REPRISE-1)** : l'état d'une chaîne Philox se résume à (champ, sweep_id, compteurs, position). Point de
  reprise atomique = `reprise.json` + `reprise.champs` (bits bruts, sha256 vérifié). 2N pas d'un coup = N pas,
  reprise dans un NOUVEAU processus, puis N pas : identiques à l'octet.
- **Bord (R2-3)** : séquence de su2_bord : pré-thermalisation périodique, pose du bord, jauge éventuelle, gel,
  thermalisation, mesures des tranches à chaque pas, blocs. Fichiers `therm_blocs.csv`, `blocs.csv` (`%.17g`) et
  manifeste avec `gel_intact`, empreintes FNV et `jauge_ecart_max`.
- **Mesures en ligne** : `su2::run::MesureEnLigne` (implémentée par `obs`, T7) est appelée tous les `periode` pas.
  Aux grands L, on mesure sur le GPU au lieu de stocker.

### 4.6 Bain conditionnel « fin sachant grossier » (T5)

- `GeometrieCond` = `cb.Geometry` (ordres `flatnonzero` gelés : p1, p1pm, fr[mu][p], free_all, fib_free/e1/e2).
- Un balayage = `cb.sweep` :
  1. réunitarisation de tous les liens ;
  2. pour mu = 0..3 : **paires** (A_eff = A₁ + A₂†V†, k = √det, V = A/k, α = βk ; tirage ; U₁ = W V†, U₂ = U₁†V) ;
  3. pour mu = 0..3 et p = 0, 1 : **liens libres**.
- **`ref_rng`** : un noyau GPU de préparation calcule α (et Vn), qu'on rapatrie. L'hôte tire `(x0, r, v)` avec
  `GenerateurNumpy`, le rejeu exact de `np.random.default_rng(seed)` : SeedSequence → PCG64 XSL-RR, `random()` sur
  53 bits, `standard_normal()` par le ziggurat de numpy (tables BSD-3, licence citée dans le fichier), et la boucle
  vectorisée `_hb_numpy` (u pour tous les « todo », puis l'acceptation, puis les rejetés, puis `normal(size=(n,3))`
  normalisé). On renvoie ces tirages au GPU, et un noyau de fin écrit U. Les tirages sont les mêmes que cb.py ; seuls
  les arrondis des agrafes diffèrent (quaternions contre 2×2 complexes), d'où **≤ 1e-12 par balayage** (EQC-1).
  **Preuve exacte que toutes les décisions de rejet coïncident** : l'état PCG64 après 5 balayages est identique à
  celui de cb (EQC-1b). Sur acemagic, les transferts passent par la mémoire unifiée, donc coûtent peu.
- **`rapide`** : un seul noyau par (mu) pour les paires et par (mu, p) pour les libres. Philox à compteur
  `(li, sweep_id, tirage, ETIQUETTE_COND)`, li = indice LOCAL du premier lien de la paire ou du lien libre.
  - x₀ : KP si α ≥ 2 ; si α < 2, proposition exponentielle de cb (`1 + log1p(-u·(-expm1(-2α)))/α`) et rejet ;
    proposition uniforme si α < 1e-12 (β = 0 admis) ;
  - direction par (cos θ, φ) comme K1.
- **Chaîne** (`cond::chaine`) = chemin principal de `chaine_v3.py` :
  - thermalisation ntherm, puis nsweep balayages ; F tous les `every` balayages, D toutes les `deach` mesures ;
  - contrôles Vdev et unitarité ;
  - points de reprise atomiques : champ, état de l'aléa (PCG64 ou sweep_id), accumulateurs, index ;
  - sorties NPY : `F.npy (nmes, 54)`, `D.npy (nD, 54, 54)`, `blk.npy (nmes, 6, (L/2)⁴) f32`, `controles.json`.
  C'est un remplaçant direct du cœur de chaine_v3 (la partie « tordue » reste en Python, lisible par-dessus).

### 4.7 Mesure de fibre (T6)

- `poids_fourier(L, 1)` = `fourier_weights_sparse` (CSR, 54 lignes). Plaquettes et gradients sur GPU (une passe).
- F par somme compensée de Neumaier, une ligne par fil.
- D = Σ lignes · lignesᵀ : chaque bloc accumule ses lignes de fibre (creuses, ≤ 27 colonnes non nulles) dans une
  matrice 54×54 en LDS (23 Ko), écrit son partiel, puis une seconde passe somme les partiels **dans l'ordre des blocs**.
  Le résultat est déterministe, donc identique d'un run à l'autre. Il diffère de BLAS à l'arrondi près.
- blk = `block_sums(pv).astype(float32)` (somme 2⁴ dans l'ordre numpy, puis arrondi en float).
- `mesurer_libre` (témoin G10) et `mesurer_f` (léger).

### 4.8 Observables (T7)

- Boucles de Wilson R×T (T selon `dir_temps` = 3, ou les 12 orientations), corrélateur de Polyakov.
- Flot de Wilson RK3 de Lüscher (coefficients de `observables::wilson_flow`) : Z = projection antihermitienne de
  U·agrafe, exponentielle exacte de su(2) en quaternions (cos|θ| + sin|θ| θ̂·iσ), `E_plaq` normalisé comme
  `action_density` (correctif du 19/06), `E_trefle`, `Q_top` (trèfle), t₀ et w₀ par interpolation linéaire comme
  `wilson_flow_curve` (cible 0,3 par défaut, à fixer par le PREREG pour SU(2)).
- Toujours sur une COPIE du champ, avec des accumulations FP64 à arbre fixe.

### 4.9 Stockage et provenance (T8)

- **NPY** :
  - `c16` (complex128 (4,L,L,L,L,2,2), à l'octet comme `dump_npy` de su2_hip) ;
  - `q64` (float64 (4,L,L,L,L,4), 2× plus petit, format interne de ym4_noyaux) ;
  - tableaux f64, f32 et u32 ;
  - écriture atomique (.part, fsync, rename, fsync du répertoire) ;
  - lecture d'un `.npy.zst` par décompression via `zstd -dc` ;
  - refus d'une matrice hors SU(2) à 1e-12.
- **zstd facultatif** (`--zstd`, niveau 3, `zstd -t` vérifié avant de supprimer l'original). Les quaternions
  double précision compressent peu (~5-10 %) : le vrai levier est `q64` et l'éclaircissement.
- **Manifeste unifié** (schéma `kevinotron-unifie/1`, au-dessus du contrat K8 : répertoire NEUF, `en_cours`, COMPLETE
  en dernier si rc = 0). Il contient :
  - argv, machine, backend, arch, versions (hip_runtime, hiprtc, cuda_pilote, nvrtc) ;
  - précision, mode, graines ;
  - `binaire_sha256` (/proc/self/exe), `git_head` (KEV_GIT_HEAD embarqué), `source_hash` (recette `Unifiee`,
    reproductible en Python : `sha256(json.dumps({chemin: sha256}, sort_keys=True))`) ;
  - `noyaux_embarques` [(nom, sha256)] ;
  - `modules_compiles` [ProvenanceCompil : nom, sha256 de la source assemblée, parties, options, arguments,
    sha256 du code objet] ;
  - politique (verrou pris ici ou par un ancêtre, nice, cpus, mem) ;
  - garde SSD (seuil, Go libres au début) ;
  - fichiers produits [(nom, sha256, octets)].
- **Garde SSD** : `GardeSsd::verifier(chemin, octets)` avant CHAQUE écriture lourde. Au lancement d'un run, la
  **projection totale** (dumps × taille + reprises) est refusée si elle passe le seuil. Rien n'est créé :
  rc = 7 et le refus est consigné.
- **Politique d'éclaircissement** (`store gc`, simulation par défaut, `--appliquer` pour agir, jamais hors de la
  racine du store, un ensemble contenant `.garder` est intouchable) :
  1. runs sans COMPLETE dont le pid est mort, plus vieux que 24 h ;
  2. reprises d'un run COMPLETE ;
  3. un dump sur k (premier et dernier gardés) ;
  4. conversion c16 → q64, avec vérification de l'aller-retour exact.
- `store du [--chemin X…]` : occupation par ensemble, total, Go libres, marge au seuil, projection.
- `fetch gamer:<chemin|store/ID> [dest]` :
  1. `du -sb` distant, puis garde SSD locale ;
  2. `rsync -a --partial -e "ssh -o BatchMode=yes"` sur `gamer-eth` ;
  3. contrôle des sha256 du manifeste distant ; un écart est une erreur et le fichier fautif est supprimé.

### 4.10 File de jobs (T1)

- Une file par machine, en fichiers : `<file>/{attente,en_cours,fini,echec,annule}/<id>.json`. Chaque transition
  est un renommage atomique, donc deux workers ne prennent jamais le même job.
- Avant de lancer, le worker vérifie que la RAM disponible dépasse `ram_min_go`, que le moniteur n'est pas en
  alerte (ALERTE.txt de moins de 10 min) et que la garde SSD passe (`sortie_estimee_go`). Sinon le job reste en
  attente, avec la raison au journal. Il ne tue jamais rien.
- Commande lancée (fonction pure `construire_commande`) :
  `[systemd-run --scope -p MemoryMax=M --quiet] nice -n N [taskset -c CPUS] [flock -w T VERROU] CMD…`.
  `systemd-run` est obligatoire si `mem_max` est donné ou si le job est estimé à plus de 4 Go.
- `queue add --machine gamer` exécute `kevinotron queue add …` sur le gamer par `ssh gamer-eth`. `queue run --boucle`
  est un worker. Ce lot n'installe aucun service systemd ; la proposition d'unité est documentée.

### 4.11 doctor, etat, mods

- `doctor` donne :
  - machine, CPU, RAM, GPU (arch, ratio FP64, mémoire) ;
  - backend choisi et pourquoi ;
  - versions, politique ;
  - SSD libre contre seuil, taille du store, file, cache des noyaux ;
  - recommandations (précision conseillée, L maximal en mémoire) ;
  - avec `--test` : compilation + exécution d'un noyau d'essai sous verrou, en moins de 5 s.
- `etat --json` (contrat gelé, pour les **mods Claude Code**, bande ou volet d'état posés par la session
  principale) : une ligne
  `{"machine","backend","gpu_verrou":{"pris":bool,"pid":int|null},"ssd_libre_go","ssd_seuil_go","ram_dispo_go",
  "charge1","file":{"attente","en_cours","echec"},"alerte_moniteur":bool,"horodatage"}`.
  Elle est produite en moins de 200 ms, sans contexte GPU.

### 4.12 Verrou GPU sans interblocage (T1)

Tout run GPU de `kvu` prend le verrou de sa machine (`VerrouGpu::acquerir`), SAUF s'il est déjà tenu par un
**ancêtre**. C'est le cas de `flock -w 7200 /tmp/kevinotron_gpu.lock kevinotron …` (usage actuel des chantiers) et
de `cargo test` lancé sous flock. On le détecte avec /proc/locks (inode du fichier → pid), comparé à la chaîne des
parents. Sans cette détection, un `flock` externe suivi d'un `flock` interne bloquerait jusqu'à expiration.
Sous `gpu_partage` (GPU_PARTAGE=1 et GPU_PARTAGE_VERROU_TENU = même fichier, lot WP21), le verrou est pris en LOCK_SH (`partage`) ;
les bancs exigent l'exclusivité et refusent alors (rc 6). Règle complète : `docs/MACHINES.md` §8.
Le manifeste note `verrou: pris_ici|ancetre|partage|ignore`.

---

## 5. Formats

| fichier | format |
|---|---|
| `therm.csv` | `step,plaquette,P0,P1,P2,P3` puis `%ld,%.17g×5` (su2_hip) |
| `measurements.csv` | `measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3` (phase `initial`/`sample`) |
| `cfg_NNNNNN.npy[.zst]` | c16 (défaut, compat) ou q64 |
| `blocs.csv`, `therm_blocs.csv` | `bloc,S0..S{L-1},T0..T{L-1}`, `%.17g` (su2_bord) |
| `reprise.json` + `reprise.champs` | EtatReprise (serde) + octets bruts des champs (F64 ou F32), sha256 |
| `manifest.json`, `COMPLETE` | §4.9 |
| sorties `cond chaine` | `F.npy`, `D.npy`, `blk.npy`, `controles.json`, `alea.json` (reprise) |
| sorties `obs` | `wilson.npy (r_max, t_max)`, `flot.csv` (`t,E_plaq,E_trefle,t2E,Q`), `flot.json` (t0, w0) |

Taille d'une configuration : c16 = 256·L⁴ octets (L = 32 : 268 Mo ; L = 48 : 1,36 Go ; **L = 64 : 4,3 Go**),
q64 = la moitié. Avec 160 Go libres et un seuil de 100 Go, il reste **60 Go** utiles sur acemagic, soit 14 configs
c16 à L = 64. D'où trois défauts : `q64` à partir de L ≥ 32, refus anticipé par la projection, mesures en ligne.

---

## 6. Gates (seuils GELÉS ici, avant tout code)

Règle commune :
- une gate GPU imprime `PASS <nom>` ou `FAIL <nom>` ; sans GPU, elle imprime `SKIP <nom>`, qui **ne vaut pas vert** ;
- les gates sont des tests `#[ignore = "GPU"]` lancés sous verrou avec `-- --ignored --nocapture --test-threads=1` ;
- chaque fichier de test GPU tourne en moins de 10 min ;
- les données de référence lourdes sont lues EN PLACE (SOURCES_IMPORTEES.md) ;
- un contrôle négatif (mutant) accompagne toute gate statistique.

### 6.1 Bit-identité et déterminisme (T3)
| gate | machine | critère |
|---|---|---|
| G-BIT-K1 | acemagic | `kvu su2 gen --backend hip --precision f64 --options ref-k1` contre `su2_gen --hip` (bâti depuis main d31d8ab dans la cible de T3), L = 12 et 16, β = 2,3, départ chaud, therm 300, nmeas 1000, dump tous les 250, graines 926193012 / 926193016 : `cmp` identique (4 NPY, therm.csv, measurements.csv) |
| G-BIT-HIP-PROD | acemagic | `--options compat-prod`, arguments de `commandes_amd.txt` (p2 L6/L8 : 20 pas, dump à chaque pas) contre `gpu_port/ref_amd/p2` : `cmp` identique ; sinon repli gelé §4.4 |
| G-BIT-CUDA | gamer | idem contre `gpu_port/cuda/p2` (et `~/ym4_calcul/su2_cuda/p2`) ; plus C0 : L8 β2,5 graine 28092800001 therm 50 nmeas 200 contre `reference/fixtures_bord/c0_ref` |
| G-BIT-BORD | gamer | mode bord, arguments de `tests_code.sh` (t_cold, t_hot, t_eq, t_eqg) : `blocs.csv` et `therm_blocs.csv` identiques aux fixtures (cmp seulement) |
| G-CPU-GPU | les deux | 1 sweep HB depuis la même configuration, CPU contre GPU F64 : max\|Δq\| ≤ 1e-12, 0 décision divergente ; avec et sans gel ; L = 6, β ∈ {2,3 ; 2,5 ; 8} |
| G-DET | les deux | double run identique à l'octet ; graine + 1 ⇒ différent |
| LOT-1 | les deux | lot de 4 chaînes (β et graines différents, F64 et F32) = 4 runs seuls, à l'octet |
| REPRISE-1 | les deux | 2N pas = N + reprise (nouveau processus) + N, à l'octet (F64, F32, bord) |
| BORD-C1/C2a | les deux | empreinte FNV des liens gelés identique début/fin ; jauge du bord : écart des tranches ≤ 1e-14 |

### 6.2 Loi du chemin FP32 (T4) — tests pré-écrits
| gate | critère |
|---|---|
| LOI-0 | après 10⁴ pas à L = 8 β = 2,3 : max\| \|q\| − 1 \| ≤ 2e-6 (pas de dérive) ; plafond = 0 |
| LOI-1 | `k_test_x0` FP32, 7 α ∈ {0,3 ; 1 ; 1,999 ; 2 ; 5 ; 20 ; 100}, N = 2·10⁷ : \|z\| < 4 pour m₁ et m₂ (moments exacts de Gauss-Tchebychev n = 4000, comme test_heatbath_cuda.py), plafond = 0 ; KS contre la CDF exacte sur 4·10⁵ tirages : p > 1e-4. Les deux backends |
| LOI-2 | 6 cellules (L ∈ {6, 8} × β ∈ {2,3 ; 2,8 ; 3,5}), 3 chaînes F32 par cellule, therm 2000, nmeas 20000, protocole P3 : ⟨P⟩ contre les ensembles FP64 AMD de même protocole (`gpu_port/ref_amd/ens`), erreur de Sokal (c = 6, `analyse_plaquette.stat`) : \|z\| < 3 par cellule et χ²(6) p > 0,01. En plus, L = 16 aux β ∈ {2,5 ; 16}, F32 contre F64 unifié de même protocole : \|z\| < 4 |
| LOI-3 | KS par permutation sur les plaquettes locales (protocole P5) à β = 2,8, L = 6 et 8 : p > 0,01 ; distribution de \|P\| (Polyakov) à L = 8, β = 2,3 : KS p > 0,01 |
| LOI-4 (puissance) | contrôles négatifs ROUGES obligatoires : (a) chaînes F32 à β × 1,002 ⇒ LOI-2 ROUGE (\|z\| > 3 dans ≥ 4 cellules sur 6, ou p(χ²) < 1e-3) ; (b) `mutant_metro` F32 ⇒ LOI-1 ROUGE à α ≥ 5 |
| LOI-τ | τ_int(P) F32 contre F64, à L = 8, β = 2,3 : \|Δτ\|/τ ≤ 0,2 + 2σ |

### 6.3 Bain conditionnel (T5) et fibre (T6)
| gate | critère |
|---|---|
| RNG-NP | `GenerateurNumpy` : 10⁵ `random()` et 10⁵ `standard_normal()` identiques À L'OCTET à numpy 2.5.3 pour les graines {0, 1, 2709, 2⁶³−1} ; `hb_numpy` identique à `cb._hb_numpy` (α de 0,01 à 200) |
| EQC-1 | ≥ 100 cas, 1 balayage `ref_rng` (CPU Rust, HIP, CUDA) contre `cb.sweep(mode='ref_rng')` : L ∈ {4, 8}, configs {plate, π-flux, Haar×3, production L8}, β ∈ {2,3 ; 2,8 ; 16}, graines {0, 1, 2} : max\|ΔC\| ≤ 1e-12 ; Vdev, unitarité ≤ 1e-12 ; β = 0 en ref_rng : erreur Config |
| EQC-1b | après 5 balayages, état PCG64 identique à `rng.bit_generator.state` de cb (toutes les décisions de rejet coïncident) |
| EQC-2 | mode `rapide` GPU contre les chaînes de production de référence `noyau2/data/L8` (10 V × β ∈ {2,3 ; 16}, 250 mesures), protocole `test_statistique.py` : par β, \|z\|max < 4,5 et fraction \|z\| > 3 ≤ 1 % (540 z) ; \|z̄\| < 4/√nV ; C_P^var apparié \|t\| < 3 ; médianes à 3σ ; tr/54 apparié \|t\| < 3 ; loi du tirage α ∈ {0 ; 0,5 ; 1,9 ; 2 ; 5 ; 30 ; 200} : \|z\| < 4 et KS p > 1e-4 (4·10⁵) ; mutant (β × 1,01) ROUGE |
| EQC-3 | `rapide` CPU Rust contre GPU, mêmes clés : 1 balayage ≤ 1e-12, 0 décision divergente |
| EQC-4 | reprise de chaîne (les deux modes) : identique à l'octet |
| FIB-1 | contre `cb.mesure` sur 14 cas (L ∈ {4, 8, 12, 16} CPU ; GPU L ≤ 16) : F (Wf dense, CPU) absolu ≤ 1e-13 ; F Neumaier contre `math.fsum` relatif ≤ 1e-14 ; D Frobenius relatif ≤ 1e-12 ; blk identique (bits float32) ; mesure_libre D ≤ 1e-12, F ≤ 1e-13 ; plaquettes et gradients ≤ 1e-14 ; plate / π-flux : D = 0 exactement |
| FIB-2 | D GPU identique à l'octet sur 2 runs et pour 2 tailles de grille |

### 6.4 Observables (T7)
| gate | critère |
|---|---|
| OBS-1 | Wilson GPU contre CPU quaternion ≤ 1e-13 relatif ; champ froid : W = 1 exactement ; champ abélien à flux constant (plongement U(1)) : W(R,T) = cos(R·T·θ) ≤ 1e-13 ; invariance de jauge (jauge Haar aléatoire) ≤ 1e-13 ; CPU quaternion contre `observables::wilson::avg_wilson_loop` historique (test racine `tests/unifie_obs_vs_historique.rs`) ≤ 1e-12 |
| OBS-2 | Polyakov GPU = mesure de Ref64 (P0..P3) ≤ 1e-15 ; corrélateur GPU contre CPU ≤ 1e-13 |
| OBS-3 | flot GPU contre CPU quaternion, 50 pas à dt = 0,01 : liens ≤ 1e-12, E(t) relatif ≤ 1e-12 ; contre `wilson_flow_step` historique sur SU2Group ≤ 1e-10 (exponentielles différentes) ; ordre RK3 : rapport des erreurs entre dt et dt/2 dans [6, 10] ; jauge pure ⇒ E ≡ 0 (≤ 1e-14) ; t₀ égal à `wilson_flow_curve` sur la même courbe (≤ 1e-12) |
| OBS-4 | Q_top après flot (t = 3, L = 8, β = 2,5, 20 configs) : \|Q − round(Q)\| < 0,1 pour ≥ 90 % (rapporté, gate souple) |

### 6.5 Infrastructure (T1, T2, T8)
| gate | critère |
|---|---|
| RT-1 | noyaux d'essai (remplissage, saxpy FP64, réduction à arbre fixe) corrects sur HIP et CUDA ; réduction identique à l'octet à la référence CPU au même arbre |
| RT-2 | CUBIN NVRTC sur le gamer : chargement OK sans PTX ; `arguments()` golden (HIP, CUDA) ; cache : miss, hit, entrée empoisonnée évincée, plafond respecté |
| MACH-1 | `doctor --json` : acemagic ⇒ machine acemagic, backend hip, gfx1151 ; gamer ⇒ gamer, cuda, sm_120 ; KEVINOTRON_BACKEND=cuda sur acemagic ⇒ rc 3 ; =cpu ⇒ cpu |
| VERROU-1 | `flock` externe puis `acquerir` interne : pas d'interblocage (ancêtre détecté) ; deux processus indépendants : sérialisés ; attente bornée ⇒ rc 6 |
| FILE-1 | `construire_commande` golden (gamer : taskset 12-19, nice 5 ; mem ⇒ systemd-run) ; 2 workers et 1 job ⇒ exécuté une fois ; RAM, moniteur et SSD simulés ⇒ job laissé en attente avec la raison |
| PROV-1 | manifeste : `binaire_sha256` = sha256 du fichier exécuté ; `source_hash` recalculé en Python identique ; `noyaux_embarques` = sha des fichiers du dépôt ; `sha256_source` = sha du texte assemblé recalculé ; options présentes |
| PROV-2 | modifier un fichier de `noyaux/` dans un arbre temporaire ⇒ source_hash et empreinte changent ; les recettes BuildSh et Complete restent INCHANGÉES (tests K8 verts) |
| PROV-3 | kill -9 en cours de run ⇒ pas de COMPLETE ; rc ≠ 0 ⇒ pas de COMPLETE |
| SSD-1 | seuil forcé au-dessus du libre ⇒ refus AVANT tout octet écrit, rc 7, aucun COMPLETE ; projection d'un run c16 L = 64 × 20 dumps refusée sur acemagic |
| SSD-2 | `store gc` simulé : rien supprimé ; `--appliquer` : seulement sous la racine, `.garder` respecté, liens symboliques non suivis ; c16 → q64 vérifié par l'aller-retour |
| FETCH-1 | fetch d'un petit ensemble du gamer : sha vérifiés ; un octet altéré à distance ⇒ erreur et fichier fautif supprimé ; garde SSD appliquée sur la taille distante |

### 6.6 Les 13 gates historiques (gpu_port, `gates_gpu1.sh`), rejouées avec le moteur unifié (T9)
| ancienne | neuve | contenu |
|---|---|---|
| s0 | G-H01 | binaire = source : `kevinotron --provenance`/manifeste (git_head = HEAD, sha du binaire, source_hash) |
| s1 | G-H02 | fichiers de `reference/` = SOURCES_IMPORTEES.md (sha256 -c) |
| P0 | G-H03 | `build_auto.sh` RC = 0 et zéro avertissement sur les deux machines |
| P1 | G-H04 | bain thermique exact, 7 α, F64, HIP et CUDA (\|z\| < 4, plafond 0) |
| P7 | G-H05 | reproductibilité à l'octet sur chaque machine (= G-DET) |
| runs CUDA 26/26 | G-H06 | 18 chaînes (L ∈ {6, 8} × β ∈ {2,3 ; 2,8 ; 3,5} × 3), therm 2000, nmeas 20000, EN UN LOT sur le gamer : rc 0, COMPLETE |
| runs AMD 20/20 | G-H07 | idem sur acemagic |
| P2 | G-H08 | départs chauds HIP et CUDA de mêmes graines : max\|Δ\| < 1e-13 (dumps rapatriés par `fetch`) |
| P3 | G-H09 | plaquette CUDA unifié contre références AMD, 6 cellules : \|z\| < 3, χ² p > 0,01 |
| P4 | G-H10, G-H11 | critères C1-C4 de PREREG_CRITERES (wilson_gpu_grandL) sur les runs CUDA unifiés, puis HIP unifiés |
| P5 | G-H12 | KS par permutation, plaquettes locales, β = 2,8, L = 6 et 8 : p > 0,01 |
| P6 | G-H13 | intégrité des dumps unifiés (recalculée SUR ACEMAGIC) : \|UU† − 1\| ≤ 1e-13, \|det − 1\| ≤ 1e-13, plaquette numpy = CSV ≤ 1e-12 |

### 6.7 Banc (T9)
`kevinotron bench su2 --L 8,16,32,48 --precision f64,f32 --chaines 1,16 --budget-s 600` sur chaque machine (sous
verrou), puis `bench cond` et `bench fibre` (L = 16, 24, 32), avec en référence les bancs de su2_hip et su2_cuda
(liens/s) et ceux de cb.py (README de noyau2). Tableau `BANC_UNIFIE.md` : liens/s, pas/s, mémoire, et gain contre
l'existant. Rapporté, non gaté, sauf une gate de non-régression : **F64 unifié ≥ 0,9 × su2_hip / su2_cuda au même L**.

---

## 7. Tâches et intégration

| id | titre | branche | dépend de |
|---|---|---|---|
| T1 | Machine : détection, politique, verrou, doctor/etat, file de jobs, build_auto.sh | unifie/machine | — |
| T2 | Runtime GPU unifié HIP/CUDA (CUBIN), assemblage, cache | unifie/gpu | — |
| T3 | Générateur SU(2) FP64 de référence : lots, bord, reprise, run, CLI su2 | unifie/su2 | T2 (exécution GPU) |
| T4 | Chemin FP32 rapide et tests de loi | unifie/f32 | T2, T3 (trait, su2_commun.h) |
| T5 | Bain conditionnel GPU + générateur numpy + chaîne | unifie/cond | T2, T3 ; T6 pour EQC-2 |
| T6 | Mesure de fibre GPU | unifie/fibre | T2, T5 (géométrie) |
| T7 | Observables : Wilson, Polyakov, flot, mesures en ligne | unifie/obs | T2, T3 |
| T8 | Stockage : NPY, provenance, garde SSD, store, fetch | unifie/stockage | — |
| T9 | Intégration : CLI, kevinotron, valide, bench, docs | unifie/integration | tous |

**Ordre d'intégration** (fusions dans `unifie/integration`, créée depuis `unifie/base` par T9 ; fichiers disjoints,
donc aucun conflit attendu) :
T2 → T8 → T1 → T3 → T4 → T5 (géométrie d'abord) → T6 → T7 → T9 (câblage final, puis gates complètes sur les deux
machines). Une tâche qui dépend d'une autre fusionne sa branche LIVRÉE (tag `unifie/<id>-livre`) avec
`git merge --no-edit unifie/<id>` dès qu'elle en a besoin.

## 8. Règles communes aux codeurs

1. **Worktree** :
   `git -C /chemin/vers/projet/kevinotron worktree add /chemin/vers/projet/kevinotron_dev/wt/<id> -b unifie/<id> unifie/base`.
   Ne jamais commiter sur main. Commits petits et clairs, en français. Tag `unifie/<id>-livre` à la livraison.
2. **Propriété des fichiers** : on ne modifie QUE ses fichiers (liste de la tâche).
   - `commun.rs`, `heritage.rs`, `noyaux.rs` (liste), `Cargo.toml` de kvu et `Cargo.lock` sont gelés : une demande
     de changement passe par l'architecte ;
   - une signature gelée du squelette ne change pas sans accord ; l'ajout de fonctions publiques est libre ;
   - retirer les `#![allow(unused_variables, dead_code)]` de squelette à la livraison.
3. **SSD** :
   - `df --output=avail -BG /` AVANT chaque build ou écriture lourde ; sous **110 Go** sur acemagic (50 sur le
     gamer), on s'ARRÊTE et on le signale ;
   - build dans `CARGO_TARGET_DIR=/chemin/vers/projet/kevinotron_dev/targets/<id>` (gamer :
     `~/kevinotron_dev/targets/<id>`), et **`cargo clean` à la fin** ;
   - données de test : L ≤ 8 seulement, au plus 5 Mo par tâche.
4. **CPU et RAM** :
   - acemagic : `nice -n 10`, `cargo -j 8` ;
   - gamer : `nice -n 10 taskset -c 12-19`, `-j 6`, cargo en `~/.cargo/bin/cargo`,
     `LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64` ;
   - jobs de plus de 4 Go sous `systemd-run --scope -p MemoryMax=…` ; garder au moins 25 Go libres sur acemagic.
5. **GPU** :
   - tout passe sous verrou : acemagic `flock -w 7200 /tmp/kevinotron_gpu.lock …`, gamer (sur le gamer)
     `flock -w 7200 /tmp/gamer_gpu.lock …` ;
   - fenêtres de 10 min au plus, puis on rend le GPU ;
   - on ne touche jamais à `~/ym4_calcul/su2_cuda/` ni à `/chemin/vers/projet/chantiers/*/bin/` et `*/kev/target`
     (lecture et exécution courte de référence seulement, si une gate l'exige explicitement) ;
   - on ne tue aucun processus qu'on n'a pas lancé ; pas de modèle LLM local.
6. **Gamer** :
   - copie du worktree par `rsync -a --exclude target --exclude .git` sur `gamer-eth:~/kevinotron_dev/wt/<id>/` ;
   - exécution détachée si elle dure plus de 2 min ;
   - résultats = RECHERCHE.
7. **Style** :
   - Rust idiomatique, `cargo clippy -p kvu -- -D warnings` propre, zéro avertissement, `rustfmt` (largeur 120) ;
   - noyaux GPU commentés en français comme l'existant ;
   - **tests unitaires et de régression pour TOUT** : CPU toujours, GPU en `#[ignore = "GPU"]` avec lignes PASS/FAIL/SKIP.
8. **Gates** : seuils de §6 GELÉS. Un seuil ne se règle jamais après avoir vu une donnée. Un écart se signale tel quel.
9. **Rendu** : texte final en français. Il contient les commandes exactes des gates et leurs sorties (RC,
   PASS/FAIL), ce qui est vert, ce qui ne l'est pas, et la place disque rendue.

## 9. Risques et parades

| risque | parade |
|---|---|
| NVRTC 13.3 contre pilote 13.2 : PTX refusé | CUBIN (`nvrtcGetCUBIN`), testé par RT-2 |
| hiprtc fast-honor-pragmas ≠ hipcc : production su2_hip pas reproduite à l'octet | repli gelé §4.4 ; G-BIT-K1 (fpoff) reste exigée |
| numpy SIMD (SVML) pour log1p/expm1 : ULP différents | tolérance 1e-12 ; une décision ne bascule qu'avec une probabilité ~1e-16 par tirage ; EQC-1b le vérifie exactement |
| ziggurat de numpy mal rejoué | RNG-NP compare 10⁵ tirages à l'octet ; tables copiées (BSD-3, licence citée), pas recalculées |
| `flock` externe + verrou interne ⇒ interblocage | détection de l'ancêtre par /proc/locks (VERROU-1) |
| SSD (91 %) | garde à chaque écriture, projection au lancement, q64 par défaut aux grands L, mesures en ligne, `store gc` |
| GPU partagé avec la production S2/S3 | fenêtres ≤ 10 min, verrou, priorité aux chantiers ; aucun binaire de production touché |
| gamer instable (cœurs P) | taskset 12-19 imposé par la politique, résultats = RECHERCHE, revérification sur acemagic |
| `#[path]` vers le moteur historique : couplage | fichiers gelés ; une modification ultérieure de su2_hb.rs relance les gates T3 |
| dérive FP32 à grand β | réunitarisation à chaque écriture ; LOI-2 à β = 16 ; LOI-4 prouve la puissance des tests |

## 10. Anticiper les usages

- **R2-3, suite S2** (Y_L(k, β), L ∈ {16, 24, 32}, β ∈ {3,0 ; 3,9}) :
  - `su2 gen --bord` en F64 compat-prod est bit-identique à su2_bord (G-BIT-BORD), donc continuable sans changer la loi ;
  - les répliques passent en lot (`--chaines`, graines communes par réplique ⇒ nombres aléatoires communs) ;
  - un passage en F32 n'est possible qu'après un AVENANT au PREREG (la loi est prouvée, mais le protocole change).
- **Saturation v3** (verrou (b)) :
  - générateur de V = `su2 gen` F64 (remplace su2_gen K1 en continuité de bits) ;
  - chaînes C-b en `cond chaine --mode ref_rng`, qui RESTE la loi de référence imposée par G-LOI(a), sur GPU ;
  - les mesures F et D sortent en NPY, lisibles par `torsion.py` ;
  - le gain attendu dépasse ×10 face à numba à L = 32, où les tirages numpy d'hôte dominent.
- **Ensembles du désert jusqu'à L = 48-64** (acemagic, 115 Go unifiés) :
  - `su2 gen --precision f32 --format q64 --mesures wilson:R8xT8:10,flot:0.01:3:20 --dump-every 200 --reprise-every 500` ;
  - L = 64 : champ F32 1,07 Go, pas ≈ 0,32 s, donc 10⁴ pas en ≈ 55 min ;
  - un dump q64 pèse 2,1 Go ; la projection l'impose avant le départ.
  - Le gamer est ≈ 2,5× plus rapide en F32 ; on rapatrie par `fetch` et on revérifie sur acemagic (G-H13).
- **Balayages en β aux petits L** : lots de 16-64 chaînes (L = 6-12 est limité par la latence de lancement ; un lot
  amortit les 40 lancements par pas).
- **Python** : `reference/` et les NPY c16/q64 restent lisibles par numpy ; `cond chaine` peut remplacer le cœur de
  chaine_v3.
- **Mods Claude Code** : `kevinotron etat --json` (§4.11) alimente une bande d'état (verrou GPU, SSD, file).
  La session principale les pose (plugin-authoring) ; ce n'est pas un lot Sonnet.

## 11. Hors périmètre (phase 2)
Migration des 49 noyaux historiques sur `kvu::gpu` et suppression des doublons cudarc/HIP ; unité systemd du worker
de file ; FP32 pour les groupes autres que SU(2) ; noyau H·v (ym4_hess) sur GPU ; multi-GPU.
