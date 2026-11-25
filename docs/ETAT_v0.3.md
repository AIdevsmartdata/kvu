# kvu v0.3 : état, carte des modules, backends validés, limites

Fusion du 22/11/2026 (agent `fusion`) : `integration/v0.3` contient en plus `spectro/xi0-cpu-1122` (option `--xi0` de `spectro run|neutre`, processeur) et
`spectro/profil-tube-1122` (option `--profil` de `spectro gpu-torelon`, noyau `k_rho`), et le lecteur `analyse_spectro.py` corrigé (voir plus bas). Les passages
ci-dessous datés « qualification » décrivent l'état AVANT cette fusion ; la section « Fusion du 22/11 » donne les rejeux.

Qualification du 22/11/2026 (agent `qualif`), branche `integration/v0.3` = `spectro/integ-1121` (b153994) + trois commits de scripts
de `spectro/aniso-1121` (`accord_xi.*`, `balayage_xi2.sh`, `analyse_point_xi.py`). La lignée est linéaire :
`integration/v0.2` ⊂ `integ/1119` ⊂ `kvu/impairs-1120` ⊂ `spectro/integ-1121`. Un seul binaire `kvu` ; les noyaux GPU
(`crates/kvu/noyaux/*.cu`) sont compilés à l'exécution pour HIP (AMD) ou CUDA (NVIDIA) ; le backend HIP est toujours compilé
(dlopen), le backend CUDA exige `--features cuda` (`cargo build --release --locked -p kvu --features cuda`).
La version du paquet reste 0.1.0 (`kvu --version` affiche 0.1.0 + empreinte des sources) : « v0.3 » est une étiquette de dépôt.

## Carte des modules (commande, backends, tests)
Légende : cpu / hip (Radeon gfx1151) / cuda (RTX 5060 Ti) ; « fumée » = essai court de la qualification (`qualif/fumee/`).
« — » = non applicable (le code refuse avec un message clair, rc 2).

| Module | Commande | cpu | hip | cuda | Tests (`crates/kvu/tests/`, `src/*/tests`) |
|---|---|---|---|---|---|
| Générateur (périodique, f32/f64, dumps, reprise) | `kvu su2 gen` | oui (f64) | oui (f32, f64) | oui (f32, f64) | su2_cpu, su2_gpu, su2_f32_gpu, su2_f32_hote, su2_reprise_*, su2_archive |
| Torsion de 't Hooft | `su2 gen --torsion` | oui | oui | oui | su2_torsion, geo_gpu* |
| Bord gelé R2-3 | `su2 gen --bord` | oui | oui | oui | su2_cli |
| Mesure en ligne (wilson, flot, corr) | `su2 gen --mesures` | oui | oui | oui | su2_corr_enligne, rg_gpu_enligne |
| Tube anisotrope (générateur) | `su2 gen --cotes [--xi0 X]` | oui (f64) | oui (f64, `--backend hip`, moteur `une_boite`) | oui (idem `cuda`) | su2_tube, su2_tube_philox |
| Tube tordu | `su2 tube smin / apparie` | oui | — | — | su2_tube |
| Boîtes de Dirichlet | `su2 boite`, `boite-obs` | oui | oui | oui | boite_*, boite_gpu |
| Méthode intégrale (a₃, côtés impairs) | `su2 thermo mesure / analyse` | analyse seule | mesure + analyse | mesure + analyse | une_boite, impairs_1120 |
| Spectroscopie torelon / neutre (processeur) | `su2 spectro run / neutre` | oui | — | — | su2::spectro::tests |
| Spectroscopie sur carte (impulsions, blocage, reps, --xi0, --ops-bruts) | `su2 spectro gpu` | — | oui | oui | su2_spectro_gpu |
| Torelons sur carte | `su2 spectro gpu-torelon` | — | oui | oui | su2_spectro_gpu |
| Algorithme à deux niveaux (Lüscher–Weisz ; masque de liens gelés, cumul sur la carte) | `--deux-niveaux D,N1[,S1]` de `su2 spectro gpu` et `gpu-torelon` (`--dn-hote` : ancienne voie hôte) | — | oui (carte = hôte au bit en f64) | oui (voir fusion-2) | deux_niveaux (unitaires), su2_spectro_gpu |
| Scanner de champs (côtés quelconques l0,l1,l2,l3) | `obs champs` | oui (seul) | — | — | obs_champs |
| Observables / flot / topologie | `obs wilson / polyakov / flot / topo / topo-gates` | oui | oui | oui | obs_cpu, obs_gpu, obs_topo_* |
| Corrélateur de densité d'action | `rg correlateur [--mesure gpu]` | oui | oui (--mesure gpu) | oui (--mesure gpu) | rg_gpu_enligne, rg_* |
| Blocage RG (Bałaban, Swendsen), MCRG | `rg bloc / swendsen / apparier / flot / action / …` | oui | bloc oui | bloc oui | rg_* |
| Gates GPU du bloc | `rg gates-gpu` | — | oui (SKIP sans `--grandl`) | oui (idem) | rg_gpu_tables |
| Géométrie (GeoSpec, torsion et masque GPU) | via `su2 gen`, `su2 boite` | oui | oui | oui | geom_*, geo_gpu* |
| Bain conditionnel, facteur 3 | `cond chaine / sweep / tirage / facteur3` | oui | oui | oui | cond_*, cond3_* |
| Intrication (Rényi-2, EXPÉRIMENTAL) | `intrication portes / balayage / courbe / doublement / debit` | oui (f64) | — | — | intrication_* |
| Expérience (carnet, portes) | `experience …` | oui | — | — | experience_* |
| Machine (profil, verrou, file) | `doctor`, `etat`, `queue`, `valide`, `store`, `fetch`, `regen` | oui | oui | oui | machine_*, cli_*, regen_cpu, stockage_* |
| Autres | `fibre`, `hess`, `groupe`, `gen` (sp4) | oui | fibre oui | fibre oui | fibre_*, hess_*, groupes_* |

## Commandes types
```
kvu su2 gen --L 8 --beta 2.3 --graine 1 --therm 200 --nmeas 50 --backend auto --precision f32 --outdir D
kvu su2 thermo mesure --paire-n 3 --paire-m 2 --impairs oui --betas 2.30 --repliques 4 --therm 50 --nmeas 100 --skip 2 --nor 1 --graine G --backend cuda --precision f32 --sortie D
kvu su2 spectro gpu --cotes 8,8,8,16 --beta 2.39 --xi0 2 --impulsions 1 --blocage 1 --tau-max 6 --backend hip --sortie D
kvu su2 spectro gpu-torelon --cotes 6,6,6,24 --beta 2.7 --moments 0:0,1:0 --backend cuda --sortie D
kvu su2 spectro run --cotes 4,4,4,8 --beta 2.3 --tau-max 3 --sortie D          # processeur
kvu rg correlateur --ens DIR --mesure gpu --sortie out.json ; kvu obs champs --cfg X.npy --out DIR
```
Sur la Radeon : sous `flock -w 3600 /tmp/kevinotron_gpu.lock` ; sur la RTX du gamer : `carte-gamer -- …`.

## Ce qui a été validé (22/11/2026)
- Tests processeur (acemagic, `RAYON_NUM_THREADS=4`, `cargo test --release -p kvu`) : 1 798 tests, 0 échec, 185 ignorés
  (26 unitaires + 159 d'intégration, presque tous « GPU »). Piège : la garde SSD refuse `/tmp` (tmpfs de 50 Go libres < seuil 100 Go) :
  16 tests (intrication_*, regen_cpu, rg_gates) échouent sans `KEVINOTRON_SSD_MIN_GO=10` ; avec, tous passent.
- Tests « ignorés » GPU sur la Radeon (HIP) : 141 passent, 3 échouent, aucun par régression : `obs_4_l16_quantification` (FAIL réel
  5/20 documenté, `docs/ARBITRAGES_D3.md`), `g_bit_hip_prod` (référence externe absente : chemin `/chemin/vers/projet/…`),
  `f3_verdict` (statistique trop courte avec les valeurs par défaut de `KVU_F3_*` : « NON CONCLUANTE »).
- Identité au bit CUDA (gamer, sous `carte-gamer`, même graine) contre les binaires de production : voir `RAPPORT` de la qualification
  (`sprint_2026-11-22/qualif/RAPPORT.md`) ; fichiers de données identiques, seuls les manifestes diffèrent par la provenance
  (chemin du binaire, dossier, secondes, session, sha256 des sources) et `bilan_gpu.json` par les temps.
- Identité au bit processeur contre le binaire `spectro/aniso-1121` : gen (périodique, mesures en ligne, torsion, `--cotes`), boite,
  tube apparie, spectro run/neutre (+ impulsions), intrication balayage : données identiques.

## Fusion du 22/11 (xi0 processeur + profil du tube + lecteur) : rejeux
- Fusions sans conflit textuel (76040bc, b18478d). Un conflit sémantique : `ParamTorelon` gagne `profil` et `profil_niveau` ; `tests/su2_spectro_gpu.rs` (2 endroits) ne compilait plus (corrigé, c5ef034).
- Tests processeur (`RAYON_NUM_THREADS=4 KEVINOTRON_SSD_MIN_GO=10 cargo test --release -p kvu --no-fail-fast`) : 1 800 verts, 0 échec, 189 ignorés (qualification : 1 798 / 185 ; les nouveaux : `profil_tests::regle_de_somme_cpu`, `identite_somme_delta_et_symetrie`, 4 ignorés carte, tests `--xi0` processeur). Un échec isolé au premier passage (`rg_wp9_invariant_cli_api_et_cpu_explicite`, tests/rg_flot_wp9.rs:143, comparaison d'octets CLI contre API avec `--backend` par défaut) ; seul : 12/12 ; au second passage complet : vert ; non lié à la fusion (le module `rg` n'est pas touché) : à surveiller (soupçon : le backend « auto » choisit la carte quand elle est libre).
- Tests propres : `profil_tests::carte_contre_cpu_f32/f64` verts sur HIP (2 sur 2) et sur CUDA (2 sur 2 ; les deux autres, du backend absent, échouent comme attendu sur chaque machine) ; contrôle croisé `--xi0 2` (6³×12, β = 2,4, f64, 1 500 configurations par chaîne) : processeur (3 chaînes, `spectro neutre`) contre HIP (4 répliques, `spectro gpu`) : plaquette spatiale 0,51881(13) contre 0,51894(10) (−0,8 σ), temporelle 0,76251(5) contre 0,76257(4) (−1,0 σ) ; `sprint_2026-11-22/fusion/xi0chk/RESULTAT.txt`. Règle de somme du profil : `regle_de_somme_cpu` (processeur) et carte contre processeur (ci-dessus). Fumée HIP de `gpu-torelon --profil 1,2` : rc 0, `profil.bin` écrit.
- Identité au bit CUDA sans les nouvelles options (gamer, `carte-gamer`, binaire fusionné contre `kvu_integ_1121`) : `spectro gpu` (f32, f64, `--impulsions 1`, `--blocage 1`) : tous les fichiers de données identiques (seul `bilan_gpu.json`, les temps, diffère) ; `gpu-torelon` (sans et avec `--moments`) : identique ; critère pair et impair : ligne suivante.
- Fumée (34 commandes chacune ; `qualif/fumee.sh`) : processeur : mêmes statuts que la qualification (modules GPU refusés rc 2, `correlateur_gpu` rc 124 sous charge) ; HIP : 32 OK, 2 KO ; CUDA : 32 OK, 2 KO ; les 2 KO sont ceux de la qualification (`gen --cotes` : chemin processeur F64 seulement ; `rg gates-gpu` : fixtures absentes rc 4 / 3 SKIP). Dont `spectro gpu --xi0 2 --ops-bruts` et `gpu-torelon` OK sur les deux cartes ; `gpu-torelon --profil 1,2` : OK (HIP et CUDA, `profil.bin` écrit). Détails : `sprint_2026-11-22/fusion/{fumee/,fumee_cuda_RESULTATS.txt,profchk/}`.
- Critère pair et impair (`thermo mesure`, f32 et f64, `--paire-n 2|3 --paire-m 2 [--impairs oui]`, deux répliques, mêmes graines) : CUDA, binaire fusionné contre `kvu_integ_1121` : 4 cas sur 4 identiques hors manifeste et `bilan_gpu.json`.
- Lecteur `scripts/spectro/analyse_spectro.py` (`e_eff`) : la garde `min(e·a, 700)` donnait NaN dès que 40·(T/2 − τ − 1) ≥ 700, soit une période temporelle > 32. Corrigé (b14d7a0) ; test `scripts/spectro/test_e_eff.py` (périodes 16, 32, 48, 64 : énergie retrouvée à 1e-6 ; périodes ≤ 32 : mêmes bits que l'ancien code). Copies corrigées : cet arbre, un arbre de travail de la seconde machine, deux dossiers de travail internes (même empreinte). Copies NON corrigées (figées ou production) : `outils_1118/lecture_graine/lecture_spectro.py` (autre fichier, même garde) et tous les `scripts/spectro/analyse_spectro.py` des autres arbres et des dossiers de travail de la seconde machine.

## Fusion n°2 du 22/11 (agent `fusion-2` : deux niveaux + scanner)
- `spectro/deux-niveaux-1122` (803143d) : deux conflits textuels dans `spectro/cli.rs` (liste des fichiers vidés en fin de boucle : on garde `fprof` ET `fcd`) et `spectro/gpu_torelon.rs` ; un conflit de sens : `calc_phi` (deux niveaux) et `--profil` (cartes Φ) se disputaient le corps de `mesurer`. Résolution : `calc_phi` rend la carte Φ du profil (`Option<Vec<f64>>`), `mesurer` la reprend. Compile CPU et CUDA.
- `scanner/v03-1122` (7da02aa) : fusion sans conflit (ETAT et `su2/cli.rs` se recollent seuls).
- Deux niveaux : D = côté de la tranche épaisse (T multiple de D), N1 = échantillons internes, S1 = balayages entre eux (défaut 1) ; conseillé D = 8, N1 = 50–100 ; gain 1,4 à 3 sur E⁺ et T2⁺ (rapport deux-niveaux), nul sur A1⁺ et le torelon ; mesuré ×1,03 seulement en temps sur 16³×32 (cumul carte). Éviter les τ multiples de D.
- Les résultats des essais de cette fusion (tests, CUDA, AMD) sont dans la section « Rejeux fusion-2 » ci-dessous.

## Rejeux fusion-2 (22/11, binaire fusionné 32dad6b)
- Tests processeur (`RAYON_NUM_THREADS=4 KEVINOTRON_SSD_MIN_GO=10 cargo test --release -p kvu`) : 1 808 verts, 0 échec, 189 ignorés (fusion 1 : 1 800 / 189) ; `rg_wp9_invariant_cli_api_et_cpu_explicite` vert au premier passage.
- Identité au bit CUDA sans les nouvelles options (`ident_cuda.sh`, binaire fusionné contre `kvu_integ_1121`) : `spectro gpu` f32, f64, `--impulsions 1`, `--blocage 1` : données identiques (seul `bilan_gpu.json`, les temps, diffère) ; `gpu-torelon` sans et avec `--moments` : identique (8 fichiers sur 8).
- Deux niveaux, CUDA (8³×16, β = 2,39, 4 répliques, `--impulsions 1,2 --blocage 1,2 --reps E+,T2+`, D = 8, N1 = 10, graine 7) : option éteinte contre binaire fusion 1 : identique f32 et f64 (hors `bilan_gpu.json`) ; voie carte contre voie hôte (`--dn-hote`) : identique AU BIT en f64 ET en f32 sur CUDA (sur la Radeon, f32 : 20 fichiers `deux_niveaux*` sur 40 diffèrent à l'ulp, f64 identique) ; contrôle du gel : 3 072 liens gelés, 0 modifié ; anisotrope `--xi0 2` 12³×48 f64 : D = 8 et D = 16 carte = hôte au bit, D = 16 : 15 552 gelés, 0 modifié. Le cumul sur la carte compile sous NVRTC et marche.
- `su2 gen --cotes` et `--xi0` sur CUDA (RTX 5060 Ti, f64, 4 chaînes, 300 + 300 pas, binaire fusionné ; processeur en référence, mêmes réglages) : 6×8×8×10, β = 2,3 : plaquette CUDA 0,60205 ± 0,00020, processeur 0,60219 ± 0,00017 (0,5 σ) ; 6⁴, β = 2,4, ξ₀ = 2 : ss CUDA 0,51861 ± 0,00045, processeur 0,51906 ± 0,00032 (0,8 σ), st 0,76414 ± 0,00020 contre 0,76408 ± 0,00011 (0,3 σ). Compatibles ; flux aléatoires non identiques au bit (attendu).
- Deux niveaux sur `gpu-torelon`, CUDA (6³×16, β = 2,7, D = 8, N1 = 5) : marche ; 1 296 liens gelés, 0 modifié ; sorties `corr_deux_niveaux.csv` et `bilan_deux_niveaux.json`.
- Radeon (HIP) : mêmes essais : éteinte identique à la fusion 1 (f64, f32, gpu-torelon, `--profil` identique) ; carte = hôte au bit en f64, à l'ulp en f32 ; `gpu-torelon --deux-niveaux` marche ; `--dn-hote` n'existe que pour `spectro gpu` (le torelon cumule toujours sur l'hôte).
- Fumée (`qualif/fumee.sh`, 34 commandes) : HIP 33 OK, 1 KO (`rg gates-gpu` : 3 SKIP, rc 8, déjà connu) ; `su2 gen --cotes` sur HIP passe maintenant ; processeur : 27 OK, 7 refus attendus (commandes qui exigent une carte, `thermo analyse` sans mesure préalable).

## Limites connues
- Pas de feature `hip` : HIP est toujours compilé ; seul CUDA est optionnel.
- `rg gates-gpu` : sans `--grandl` (ensembles L24) trois portes SKIP ; sans les fixtures `validation/rg/fixtures/cfg_s7_q64.npy`
  (absentes du dépôt de travail et du gamer) la commande échoue avec rc 4.
- `su2 tube`, `su2 spectro run|neutre`, `intrication`, `obs champs` : processeur F64 seulement.
- `su2 gen --cotes` : processeur, ou carte (`--backend hip|cuda`, F64 seulement, ajouté le 22/11 par la branche scanner/v03-1122) ; le backend doit être explicite
  (`auto` refusé). Sur carte : réseau périodique sans torsion, sans mutant, sans `--mesures tube:…`, départ hot|cold (pas `vide`) ; therm.csv n'a que
  les colonnes `step,plaquette` ; plaquette et Polyakov de measurements.csv sont calculés sur l'hôte à partir du champ téléchargé (une copie par mesure et par chaîne).
  Les flux aléatoires carte et processeur ne sont pas identiques au bit : mêmes moyennes dans les erreurs (voir scanner-v03/RAPPORT.md).
- `su2 gen --cotes … --xi0 X` : réseau anisotrope (axe du temps = 3, plaquettes temporelles ξ₀, spatiales 1/ξ₀), sans torsion ni mutant ; ξ₀ dans le manifeste (`xi0`) et poids de plans dans
  `geometrie.json` (`beta_plans`, chemin C1) ; `plaq_ss_st.csv` par chaîne. Sur le processeur, la colonne `plaquette` de measurements.csv est alors la moyenne PONDÉRÉE (z·½ReTr) ; sur carte elle est non pondérée.
  Absente : sorties identiques au bit. `su2 gen --xi0` sans `--cotes` est refusé.
- `obs champs` accepte des côtés quelconques (l0,l1,l2,l3 ≥ 2, NPY `(4,l0,l1,l2,l3,…)` + `geometrie.json` voisin ; périodique, sans torsion ni masque). Boîte cubique : chemin historique inchangé.
  Boîte non cubique : portes de somme (références indépendantes : `tube::reseau::plaquette` et `polyakov_lignes`), localité, jauge (option) ; pas de porte de charge indépendante (la somme de q est recalculée
  par le même trèfle ; seul le test de permutation des axes la contrôle). Le flot est celui de l'action de Wilson isotrope même pour une configuration anisotrope.
  Même sur une boîte anisotrope-ξ₀, les champs exportés ne tiennent pas compte des poids de la génération.
- `su2 thermo mesure`, `su2 spectro gpu|gpu-torelon`, `rg gates-gpu` refusent le processeur (rc 2).
- `su2 spectro gpu` exige `--tau-max < ℓ_temps` (défaut 6 : refus sur un côté temporel de 4).
- Modules intrication, tube, spectro : EXPÉRIMENTAUX, relus par une seule famille de modèles.
