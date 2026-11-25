# Noyaux GPU du socle `geom` : chemin C1 (B10, kvu v0.2)

Fragment de documentation (à assembler par l'agent de documentation). Portés : torsion de 't Hooft sur GPU (lot A) et masque
de liens gelés (lot C), côtés égaux, F64. Révision c09b (relecture finale de Sol, O1 à O7).

## Ce qui existe
- `noyaux/su2_geo.cu` : noyaux du chemin C1 (réseau régulier à 4 côtés pairs, coefficient par plaquette `coef[p·V + s]`, masque
  de liens gelés), compilés dans le module `su2_geo64` = `su2_commun.h` + `su2_ref64.cu` + `su2_geo.cu` (réutilise `sample_x0`,
  `k_reduce`, `k_final` sans copie). Embarqué par `su2::geo_gpu` HORS du registre gelé `noyaux::TOUS`, comme `su2_boite.cu` ;
  son nom, son empreinte ET son texte intégral figurent au manifeste d'un run C1 (`noyaux_c1`) : la source exacte se
  relit dans les sorties, sans le dépôt (l'empreinte `kvu-source/1` du binaire ne conserve pas le texte).
- `su2::geo_gpu::Geo64Gpu` : générateur `NoyauxSu2` construit depuis une `geom::GeoSpec` (`Tables::compiler`), F64 seulement,
  côtés égaux à `ConfigLot.l` : torsion quelconque ; masque de la géométrie (`Bord::Fixe`, boîte), `fixer_masque` (WP-DIR, avec
  `AppMasque`), `fixer_gel` (R2-3 traduit en masque). `geo_gpu::valider_c1` : validation PURE, avant toute ouverture de GPU.
- CONTRAT des valeurs de bord : `Geo64Gpu` gèle des liens mais n'en POSE aucune valeur. Un bord `Fixe` ou une boîte doivent
  déclarer `SourceBord::ParChaine` ; l'appelant pose les liens gelés (`televerser`) avant le premier balayage. Une valeur
  prescrite (froid, chaud, copie, npy) est refusée (code 2).
- Refus (tous `Erreur::Config`, code 2) : côtés inégaux, β par plan, C2, torsion avec masque, gel ou bord fixé (aucune référence
  CPU), L impair ou hors de 2..180, F32, β = 0 avec torsion.
- Le texte assemblé des noyaux C0 (`su2_ref64`, `su2_f32`, `boite`, `cond`, `cond3`, `obs`, `rg_bloc`, HIP et CUDA) est inchangé
  à l'octet : porte P1 (`tests/geo_gpu_sources.rs`, valeurs d'or calculées par un script Python indépendant).

## Conventions
- Contrat consommé : en-tête de `geom/tables.rs`, cité point par point en tête de `su2/geo_gpu.rs`.
- Géométrie transmise aux noyaux par un tampon de 9 entiers copiés des `Tables` : ℓ0..ℓ3, V, pas0..pas3 (aucune formule de pas sur
  GPU) ; voisins et énumération des parités vérifiés exhaustivement contre `Tables` (porte G-GEO).
- Agrafe : A = Σ_{ν≠μ} (z_t·t + z_b·b), z_t au coin x, z_b au coin x − ν̂ (= `torsion::agrafe_tordue`, = agrafe du tube).
- Plaquette mesurée (colonne `plaquette`) : P_tw = (1/6V) Σ_p z_p ½ReTr U_p ; Polyakov bruts (comme le CPU, docs/TORSION.md).
- Compteurs Philox identiques au CPU et au chemin C0 : (li, sweep_id, tirage, TAG_HB), départ (li, 0xFFFFFFFF, 0|1, TAG_INIT).

## Trois niveaux de comparaison (à ne pas confondre)
- ÉGALITÉ BINAIRE (`to_bits`) : C1 avec une spécification C0 contre C0, même carte, en `ref-k1` (P2) ET en `compat-prod`
  (P2-COMPAT : chaque balayage, depuis un état commun, doit rester identique au bit) ; C1 avec masque contre C0 avec le même
  masque (C-MASQUE-C0) ; déterminisme (P5) ; reprise = run non interrompu.
- PAR BALAYAGE depuis un état commun : GPU contre CPU (P3, C-MASQUE-CPU-GPU) : 0 décision divergente (nombre d'essais par lien)
  et max|Δq| ≤ 1e-12 ; mesures sur un état identique (P4).
- TRAJECTOIRE LIBRE : un pas (1 bain + 4 surrelaxations) ≤ 1e-10 (P3-TRAJ-1) ; au-delà, la dynamique amplifie les écarts
  d'arrondi GPU/libm jusqu'à O(1) en ~10 pas, chemin C0 compris (`diagnostic_derive_libre` : information, pas une porte ; une erreur d'exécution le fait échouer).

## Interface en ligne de commande
- `kvu su2 gen --torsion PLANS --backend hip|cuda` (F64) : le run passe par `Geo64Gpu` ; il écrit `torsion.json`, `geometrie.json`
  et le bloc `geometrie` du manifeste (`kvu-geometrie/1`, temps = axe 3), `moteur_su2` = Geo64Gpu ; ses dumps se relisent par
  `stockage::npy::lire_champ_geo`. Un run tordu sur GPU n'est PAS égal à l'octet au run CPU (voir ci-dessus).
- Restent REFUSÉS avec une torsion non triviale (code 2, sans ouvrir de GPU) : `--backend auto`, `--precision f32`, `--bord`,
  `--mesures`, β = 0, et un L que le chemin C1 refuse (validation pure avant `ouvrir_backend`).
- Reprise : le point de reprise consigne les invariants du moteur (`reprise.json`, champ `invariants` : moteur
  « Geo64Gpu/<backend> » ou « Cpu64+M4 », géométrie, gel, empreinte du masque externe, `AppMasque`). Un run se reprend sur son
  moteur d'origine : CPU→GPU et GPU→CPU refusés (code 2, avant d'ouvrir le GPU, rien d'écrit). Un point de reprise SANS
  invariants (binaire antérieur) : C0, repris comme avant ; tordu, repris sur CPU SEULEMENT si le manifeste du run prouve une
  provenance CPU (backend « cpu », aucun module `su2_geo64`) ; tout autre cas (run tordu GPU d'un binaire antérieur, manifeste
  absent) est refusé (code 2) : relancer le run.
- Recette des portes GPU : `cargo test --release --test geo_gpu --test geo_gpu_masque -- --ignored --nocapture --test-threads=1`
  sous le verrou GPU ; sans GPU, ces portes ÉCHOUENT (KVU_GPU_SAUT=1 pour les sauter explicitement : SKIP, jamais vert).

## Ce qui n'est PAS encore là
- Lot B (côtés inégaux pairs, référence = moteur CPU du tube), β par plan, lot D (F32 sous porte de loi).
- `su2 gen` ne connaît que des côtés égaux (`--L`) : les géométries anisotropes passent par le tube (`su2 tube`), CPU seulement.
