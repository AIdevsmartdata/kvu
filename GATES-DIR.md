# GATES-DIR — gates du lot WP-DIR (boîtes SU(2) à bord fixé sur GPU), version v1

- **Version** : v1, gelée le 2026-10-25 par l'implémenteur du lot WP-DIR, **avant** tout run de gate du lot (le sha256 de ce fichier est
  consigné dans `journal_wp_dir.log`, ligne `WP-DIR sha256 GATES-DIR.md`, antérieure à tout lancement). Toute modification ultérieure
  crée un bloc **v2** daté en fin de fichier (ancien seuil, nouveau seuil, raison, nouveau sha256 au journal) ; une modification non
  consignée invalide les verdicts qui en dépendent. Les erreurs de protocole découvertes en cours de route sont corrigées par un bloc
  v2, jamais en réécrivant ce texte.
- **Rattachement** : complète `GATES.md` (v1 à v3.1 + errata, sha256 `b5dbad56…`, inchangé par ce lot). Les règles 0.1 à 0.9 de `GATES.md`
  s'appliquent sans changement : une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate, un SKIP n'est pas vert, un mutant vert est un
  `ECHEC_OUTIL`, le vérificateur ré-exécute, hygiène RC, verrous GPU (`flock -w 7200`, fenêtres ≤ 10 min), aucun binaire de production touché.
- **Cible du lot** : un masque de liens gelés dans le bain thermique SU(2) (CPU, HIP, CUDA, source unique), des boîtes ℓ⁴ de Dirichlet,
  la mesure de la plaquette centrale, de la boucle 2×2 centrale et du profil par couche, la commande `kvu su2 boite`, un pilote de balayage.
- **Références externes (lecture seule)** : QW1 = `chantiers/vision_20261025/calculs/QW1_filtre_dobrushin/` (`box_su2.py` sha256
  `d7363d65…`, `gamma.py` `439ed2f3…`, `logs/analyse.json` `fc7893a8…`, `PREREG.md` `a67d2b57…`, `verification/VERIFICATION.md`).
  Copies à l'octet dans `crates/kvu/reference/qw1/` (sha256 dans `SOURCES_IMPORTEES.md`).

## 1. Définitions gelées

**Boîte.** ℓ pair ≥ 2, sites {0,…,ℓ}⁴, lien `U_μ(x)` de `x` à `x+e_μ` pour `x_μ ≤ ℓ−1`. Un lien est **fixé** (donnée de bord) si ses deux
extrémités sont sur le bord du cube (une coordonnée vaut 0 ou ℓ) ; tout autre lien de la boîte est **libre** (tiré). Nombre de liens libres
`n_libres(ℓ) = 4ℓ(ℓ−1)³` (ℓ = 8 : 10 976, valeur du PREREG de QW1). Action `S = β Σ_p (1 − ½ Re Tr U_p)`, β = 4/g² (aucun N dans β).
**Réseau de calcul** : la boîte est plongée, mêmes coordonnées, dans un réseau périodique de côté **L_c = ℓ + 2** (pair, exigé par le damier
des noyaux) ; tous les liens du réseau de calcul qui ne sont pas libres sont **gelés par le masque** (liens fixés, liens inexistants `x_μ = ℓ`,
liens de la couche de remplissage). Un lien libre ne lit jamais un lien de remplissage ni ne voit le repliement périodique (un lien libre a une
extrémité intérieure : `x±e_ν` reste dans [0, ℓ]).
**Masque.** Un octet par lien (disposition `li = μ·V + s` du champ), partagé par toutes les chaînes d'un lot ; octet non nul = lien gelé : ni le
bain thermique ni la surrelaxation n'y touchent. Absence de masque ⇒ comportement antérieur inchangé.
**Pas.** 1 bain thermique (Kennedy–Pendleton si α ≥ 2, Creutz sinon) + `nor` surrelaxations, damier 4 μ × 2 parités.
**Observables** (centre c = ℓ/2 dans chaque coordonnée ; `P(x; μν)` = ½ Tr de la plaquette de coin x dans le plan (μ,ν) ; plans (μ<ν) dans l'ordre
01 02 03 12 13 23) :
- `P_c` = moyenne des 24 valeurs `P(c − a e_μ − b e_ν ; μν)`, plan puis a puis b ∈ {0,1} (mêmes 24 plaquettes que `measure` de QW1) ;
- `W₂` (ℓ ≥ 4) = moyenne sur les 6 plans de ½ Tr de la boucle 2×2 de coin inférieur `c − e_μ − e_ν` (2 pas +μ, 2 pas +ν, 2 pas −μ, 2 pas −ν) ;
- profil par couche : classe de profondeur d'une plaquette de base x dans le plan (μ,ν) valide (`x_μ, x_ν ≤ ℓ−1`) = min sur ses 4 coins de
  `min_i min(x_i, ℓ − x_i)` ; `profil_d` = somme des ½ Tr des plaquettes de la classe d, divisée par leur nombre, d = 0 … ℓ/2 − 1
  (la classe ℓ/2 est vide). Les sommes se font dans l'ordre d'une liste fixe et un arbre à 256 fils fixe (aucune atomique flottante).
**Bords** (chargés sur l'hôte, identiques sur les trois backends ; seuls les liens fixés sont posés, l'intérieur part froid = 1) :
`cold` (1) ; `gauge` (`g(x) g(x+μ)†`, g de Haar par site, jauge pure) ; `z2full` (`U_μ(x) = (−1)^{x₀+…+x_{μ−1}}`, toutes les plaquettes du bord
valent −1) ; `z2_01` (`U₁(x) = (−1)^{x₀}`) ; `flux` (`U₁(x) = exp(iσ₃ φ x₀)`, `U₃(x) = exp(iσ₁ φ x₂)`, φ = π/2, valeurs trigonométriques
**exactes** (0, ±1) : écart de ≤ 6,2e-17 à `box_su2.py`, qui écrit `cos(π/2) = 6,1e-17`, voir §5) ; `haar` (liens fixés de Haar indépendants) ;
`therm` (liens fixés découpés dans une configuration d'équilibre périodique de même β, L ≥ ℓ + 1, produite par le même moteur sans masque).
**Statistique** (commune à C1, C2, C4) : une série par chaîne ; erreur par la méthode Γ de Wolff, `S = 1,5` (port de `gamma.py`, `uwerr`) ; M chaînes
indépendantes (graines distinctes) par cellule ; estimation = moyenne des moyennes de chaîne, `σ² = Σ σ_c² / M²` ; écart de deux cellules
`z = (m₁ − m₂)/√(σ₁² + σ₂²)`. Les analyses sont faites par un script **Python indépendant** du moteur (`validation/boite/analyse_dir.py`, numpy).

## 2. Constantes de référence gelées (QW1, β = 2,5, `logs/analyse.json` sha256 `fc7893a88bbf201f…`)

| ℓ | ⟨P_c⟩ cold | ⟨P_c⟩ z2full | ΔP_c = cold − z2full | ⟨W₂⟩ cold | ⟨W₂⟩ z2full | ΔW₂ |
|---|---|---|---|---|---|---|
| 2 | 0,82223624(33) | 0,21400910(24) | **0,60822714(41)** | — | — | — |
| 4 | 0,67115281(38) | 0,58980331(46) | **0,08134950(59)** | 0,32542229(1037) | 0,13830734(1194) | **0,18711495(158)** |
| 6 | 0,65700349(38) | 0,64048753(42) | **0,01651597(57)** | 0,27718084(1068) | 0,22617022(1166) | **0,05101062(158)** |
| 8 | 0,65435088(39) | 0,64833275(40) | **0,00601814(56)** | 0,26605584(1086) | 0,24730973(1136) | **0,01874611(157)** |

Témoin de jauge de QW1 (z) : −0,90 / 1,82 / −0,26 / −0,73 pour ℓ = 2 / 4 / 6 / 8. Valeurs du vérificateur indépendant de QW1 (Metropolis,
autre code, `verification/VERIFICATION.md` §3), **INFO seulement** : ΔP_c(ℓ=4) = 0,08226(31), ΔW₂(ℓ=4) = 0,18616(77), ΔP_c(ℓ=2) = 0,60823(40).
Série de couplage fort (SU(2), `GATES.md` v3.1) : `P = u + 4u⁵ + O(u⁹)`, `u = I₂(β)/I₁(β)`.

## 3. Gates

### A. Non-régression (octets)
| ID | énoncé | seuil | méthode |
|---|---|---|---|
| G-DIR-BIT-HIP-PROD | ancre HIP de (A), binaire du commit final | `cfg_002000.npy`, `measurements.csv`, `therm.csv` = lignes `hip` de `anchors.sha256` : **3/3** | commande de `GATES.md` WP1 (compat-prod, L = 12, β = 2,5, graine 26095612011), acemagic, verrou |
| G-DIR-BIT-CUDA | ancre CUDA de (A) | **3/3** (L = 8, β = 2,8, graine 27092708011) | gamer, `/tmp/gamer_gpu.lock` |
| G-DIR-BIT-K1 | `ref-k1` contre `su2_gen --hip` | `cmp` identique : 4 NPY + 2 CSV à L = 12 et 16 (**6/6** chacun) | `validation/regen/gbit_preuves.sh`, acemagic |
| G-DIR-GPU-EXISTANTS | les gates GPU existants de `tests/su2_gpu.rs` (G-CPU-GPU avec gel, G-DET, LOT-1, REPRISE-1, BORD-C1/C2a, G-H04…) | tous PASS, sans modification des tests | `cargo test --release --test su2_gpu -- --ignored`, par fenêtres ≤ 10 min, acemagic (HIP) et gamer (CUDA) |

### B. Masque, géométrie, bords, mesures (mécanique, déterministe)
| ID | énoncé | seuil | méthode / mutant |
|---|---|---|---|
| G-DIR-COMPTE | nombres de liens libres, fixés et inexistants | `n_libres = 4ℓ(ℓ−1)³` pour ℓ ∈ {2,4,6,8,10,12,16,24} (ℓ = 8 : 10 976) ; fixés = `4ℓ(ℓ+1)³ − n_libres` ; masque calculé par une formule **indépendante** (test, géométrie des coordonnées) = masque du code, octet par octet | test CPU ; mutant : masque décalé d'un site ⇒ ROUGE |
| G-DIR-BORDS | définitions des bords (réponses connues ; plaquettes dont les 4 liens sont fixés) | `cold` : toutes = 1 ; `z2full` : toutes = −1 exactement ; `z2_01` : plan 01 = −1, tous les autres plans = +1 ; `gauge` : toutes valent 1 à ≤ 4e-16 ; `flux` : plan 01 : ½ Tr = 0 exactement, plan 13 : ½ Tr = −1 si x₀ et x₂ sont tous deux impairs et +1 sinon (commutateur de iσ₃ et iσ₁), liens de norme 1 à ≤ 1e-16 ; `haar` : ‖q‖ = 1 à ≤ 4e-16 et deux graines différentes donnent des liens différents pour ≥ 99 % des liens fixés ; `therm` : égal à la tranche de la source, à l'octet | test CPU ; mutant : `z2full` remplacé par `z2_01` ⇒ ROUGE |
| G-DIR-MASQUE-GEL | le masque généralise le gel R2-3 | masque ≡ `BordX0Spatial` : après 5 pas (1 bain + 4 surrelax.), L = 6, β = 2,5, graine 26102610001, chaîne chaude : octets **identiques** (`to_bits`) à `fixer_gel(BordX0Spatial)`, sur CPU, HIP, CUDA | test GPU (chaque backend contre lui-même) |
| G-DIR-MASQUE-NUL | masque de zéros = pas de masque | octets identiques après 5 pas (mêmes paramètres, graine 26102610002), CPU, HIP, CUDA | idem |
| G-DIR-GELES-INTACTS | un lien gelé ne bouge jamais | pour ℓ ∈ {4,6,8}, bords {cold, z2full, flux, haar, gauge}, 50 pas (1 bain + 2 surrelax.) : max sur les liens gelés de max\|Δq\| = **0,0 exactement** (octets) ; fraction de liens libres modifiés après 5 pas ≥ 0,99 ; CPU, HIP, CUDA | **mutants** : (M-DIR-2) surrelaxation sans masque, (M-DIR-3) bain thermique sans masque ⇒ max\|Δq\| > 0,1, ROUGES |
| G-DIR-CPU-GPU | CPU contre GPU F64 `ref-k1`, avec masque | ℓ ∈ {4,6}, bords {cold, flux, haar, z2full}, β ∈ {2,5 ; 8}, états de départ = **(a)** le départ froid (intérieur 1 + bord ; pour `flux` à ℓ = 6, des agrafes exactement nulles, observées par QW1, traversent la branche de Haar des noyaux : leur nombre est rapporté (INFO) et doit être > 0, sinon le cas ne teste pas la branche et la gate est SKIP, non verte) et **(b)** l'état GPU après 3 pas ; mêmes états transmis au CPU, même `sweep_id` ; un bain thermique : **0 décision divergente** (essais par lien identiques) et max\|Δq\| ≤ 1e-12, puis une surrelaxation : ≤ 1e-12 ; liens gelés identiques à l'octet entre CPU, GPU et état initial ; `P_c`, `W₂`, profils recalculés des deux états : écart ≤ 1e-12. **INFO** : nombre de liens libres à octets différents après le bain et après la surrelaxation. | HIP contre CPU (acemagic), CUDA contre CPU (gamer) ; **mutant** (M-DIR-7) : CPU qui ignore le masque ⇒ ROUGE |
| G-DIR-DET | déterminisme | deux runs `su2 boite` (ℓ = 6, bords cold,z2full,haar, 2 répliques, β = 2,5, 200 pas, graine 26102610003), même backend : `mesures.csv`, `profil.csv`, `config_finale` identiques à l'octet ; CPU, HIP, CUDA | `cmp` |
| G-DIR-LOT | une chaîne d'un lot = la même chaîne seule | ℓ = 4, lot de 5 chaînes (bords cold, z2full, flux, haar, gauge ; graines 26102610011…15), 100 pas ; chaque chaîne seule : séries et configuration finale identiques à l'octet ; CPU, HIP, CUDA | `cmp` |
| G-DIR-FILS | indépendance au nombre de fils (CPU) | 1 fil et 8 fils : sorties identiques à l'octet | `RAYON_NUM_THREADS` |
| G-DIR-OBS | mesures contre l'implémentation de QW1 | sur 8 configurations de boîte (ℓ ∈ {4, 6} × {froide, Haar de bord + 20 pas, z2full + 20 pas, flux + 20 pas}, β = 2,5), configurations écrites en NPY q64 par `su2 boite --dump-final`, relues par la sous-commande `su2 boite-obs` (CPU Rust et GPU) et par `box_su2.measure`/`depth_classes` (Python, QW1) : `P_c`, `W₂`, `profil_d` Rust CPU contre Python ≤ 1e-12 ; GPU contre Rust CPU ≤ 1e-14 ; champ froid : `P_c = W₂ = profil_d = 1` exactement | **mutant** (M-DIR-8) : la référence Python évaluée au centre c+1 ⇒ écart > 1e-3, ROUGE |
| G-DIR-F32-REFUS | pas de masque ignoré en silence | le générateur F32 refuse `fixer_masque` (erreur, jamais un succès sans effet) ; `su2 boite --precision f32` : rc 2 avec message ; masque de mauvaise longueur : rc 2 | test unitaire ; mutant : F32 qui accepte sans appliquer ⇒ ROUGE |
| G-DIR-BORD-INTACT | liens de bord inchangés en fin de run (G2 de QW1), masque géométrique indépendant | `su2 boite` sur 10 bords (cold, gauge, z2full, z2_01, flux, haar×3, therm×2) à ℓ = 4, 100 pas : max\|ΔU_bord\| = 0 ; la commande elle-même refuse (rc 8) tout écart | mutants M-DIR-2, M-DIR-3 ⇒ ROUGES |

### C. Physique (statistique)
Chaque gate de ce groupe est jouée sur **HIP (acemagic) et sur CUDA (gamer)**, chaque backend jugé séparément (jamais deux backends combinés) ; graines CUDA = graines HIP + 500.
| ID | énoncé | seuil | mutant |
|---|---|---|---|
| **G-DIR-QW1** | reproduction de QW1, β = 2,5, bords cold et z2full, M = 16 chaînes par bord, 1000 pas de thermalisation, 30 000 pas mesurés, nor = 2 (protocole de QW1), départ intérieur froid. **ℓ = 8** : `ΔP_c = 0,00601814(56)` et `ΔW₂ = 0,01874611(157)` | **\|z\| < 3** pour chacune des deux quantités (z = (Δ_nous − Δ_QW1)/√(σ_nous² + σ_QW1²)) ; mêmes tests aux ℓ = 2, 4, 6 (ΔP_c ; ΔW₂ pour ℓ = 4, 6), \|z\| < 3 aussi ; en INFO : z de chaque ⟨P_c⟩, ⟨W₂⟩ de bord et des références du vérificateur indépendant | **M-DIR-1** (masque ignoré dans le bain et la surrelaxation, `--mutant masque-ignore`) : la même analyse doit donner \|z\| ≥ 3 pour ΔP_c(ℓ = 8) **et** \|ΔP_c(ℓ = 4)\| < 0,02 (l'influence du bord disparaît) |
| G-DIR-JAUGE | témoin de jauge : bord `gauge` contre `cold`, β = 2,5, ℓ ∈ {2, 4, 6, 8, 12, 16}, M = 8 chaînes par cellule, 1000 + 20 000 pas (ℓ ≤ 8) ou 1000 + 5 000 pas (ℓ = 12, 16) | **\|z\| ≤ 3** pour ⟨P_c⟩ à chaque ℓ | **M-DIR-5** : bord `gauge` remplacé par `haar` ⇒ \|z\| > 3 à ℓ = 2 ou 4 |
| G-DIR-AGIT | le bord agit (G4 de QW1) | ℓ = 2, β = 2,5 : ⟨P_c⟩(z2full) ≠ ⟨P_c⟩(cold) à **> 5σ** (M = 8, 1000 + 5 000 pas) | M-DIR-1 : ROUGE (non > 5σ) |
| G-DIR-SC | réponse connue (couplage fort) : β = 0,2, ℓ = 6, bords {cold, z2full, flux, haar}, M = 8 chaînes par bord, 1000 + 10 000 pas | `⟨P_c⟩ = u + 4u⁵` (u = I₂(0,2)/I₁(0,2), calculé par un script indépendant, erreur de troncature O(u⁹) < 1e-10) à **\|z\| < 3** pour chaque bord | **M-DIR-6** : simulé à 1,05 β, analysé à β = 0,2 ⇒ \|z\| > 3 |

### D. Manifeste, régénération, pilote
| ID | énoncé | seuil | méthode / mutant |
|---|---|---|---|
| G-DIR-MANIFESTE | `su2 boite` écrit `manifest_regen.json` au schéma `kvu-regen/1` | validation par `scripts/regen_valider.py` (`python3 -I`) **et** `kvu::regen::schema` : 0 erreur ; les paramètres de la boîte (ℓ, β, bords, graines, pas, masque) y sont ; `fichiers` = séries, profils, configurations finales avec sha256 | **mutant** : un champ obligatoire retiré ⇒ rejeté |
| G-DIR-REGEN | `kvu regen manifest_regen.json` rejoue la boîte | ℓ = 4, 3 bords, 200 pas, HIP (acemagic) et CUDA (gamer) : tous les fichiers listés **identiques à l'octet**, rc 0 | **mutant** : une graine modifiée dans la commande du manifeste ⇒ rc 8 |
| G-DIR-PILOTE | pilote de balayage | (a) `--dry-run` sur un fichier de 12 lignes : plan déterministe (deux exécutions identiques), aucun lot dont l'estimation dépasse la fenêtre, rc 0 ; (b) un lot trop long pour la fenêtre : refus rc 2 avec message ; (c) exécution réelle de 4 tâches (2 cellules ℓ = 4, β ∈ {2,5 ; 3,0} × 2 bords, donc 2 lots) sur **les deux GPU en parallèle** : chaque GPU exécute ≥ 1 lot, tous COMPLETE, `index.json` cohérent, manifeste `verrou` = `pris_ici` ou `ancetre` ; (d) **verrou par GPU** : un `flock -w 7200 <verrou du GPU> sleep 8` tenu pendant que le pilote lance un lot sur ce GPU ⇒ le lot ne démarre pas avant la libération (horodatages) | mutant : pilote avec `--sans-verrou` n'existe pas ; le test (d) échoue si le lot démarre avant la libération |

### E. Qualité et production
| ID | énoncé | seuil |
|---|---|---|
| G-DIR-TESTS | `cargo test --release --no-fail-fast` (acemagic `--no-default-features` ; gamer `--features cuda`) | rc 0, **0 échec**, nombre de tests passés ≥ 812 (base `main` a15cd8c) + tests ajoutés par le lot ; ignorés ≥ 73 |
| G-DIR-CLIPPY | `cargo clippy --all-targets -- -D warnings` (acemagic ; gamer `--features cuda`) | rc 0 |
| G-DIR-TESTS-INTEGRITE | aucun test existant modifié, supprimé ni ignoré | `git diff --stat main -- crates/kvu/tests` = ajouts de fichiers seulement ; `#[cfg(test)]` des fichiers modifiés : ajouts seulement |
| G-DIR-PROD | T0-PROD | `7098f98f…`, `d62ff7d2…`, `cb29307d…` identiques au début et à la fin |
| G-DIR-GREP | WP3-GREP | `validation/machine/wp3_grep.py` : code 0 (aucun hôte, chemin de machine ni seuil dans `src/`) |
| G-DIR-H01 | binaire = source | `G-H01.sh` PASS rc 0 sur le binaire final |

### F. Rapportés (jamais des gates)
- **G-DIR-GAIN** (INFO) : temps mesuré par pas (1 bain + 2 surrelaxations + mesure centrale + profil tous les 10 pas, en régime établi, hors compilation et E/S) pour ℓ ∈ {8, 12, 16} (et ℓ = 24 si la mémoire le permet), lots de K ∈ {1, 16, 64} chaînes, sur HIP (acemagic) et CUDA (gamer), sous verrou, fenêtres ≤ 10 min ; référence : `box_su2.py` de QW1 (un cœur, `nice`, temps CPU utilisateur par pas) mesuré le même jour aux mêmes ℓ, plus l'équivalent « 8 processus » de QW1. Le gain = temps CPU de QW1 par pas et par chaîne ÷ temps GPU par pas et par chaîne (K saturant).
- Nombre de liens à octets différents entre CPU et GPU (G-DIR-CPU-GPU, INFO), avec la justification : mêmes compteurs Philox donc mêmes nombres aléatoires ; seules `log`, `cos`, `sin`, `exp` (libm de l'hôte contre bibliothèques du périphérique) peuvent différer d'un ULP ; `sqrt` et `1/k` sont corrects au dernier bit sur les trois ; d'où un critère à 1e-12 sur les statistiques déterministes plutôt que l'égalité des octets.

## 4. Graines et protocoles chiffrés (gelés)
- Tests B : graines 26102610001 … 26102610099 (indiquées ci-dessus ; les autres cas de B prennent 26102610020 + indice de cas).
- G-DIR-QW1 : chaîne c (0 à 31 : bord cold = 0…15, z2full = 16…31) de la cellule ℓ → graine `26102611000 + 1000·j + c`, j = 0, 1, 2, 3 pour ℓ = 8, 6, 4, 2.
- G-DIR-JAUGE / G-DIR-AGIT : graine `26102612000 + 100·j + c` (j indice de ℓ dans (2,4,6,8,12,16), c = 0…15 : cold 0…7, gauge 8…15).
- G-DIR-SC : graine `26102613000 + c` (c = 0…31, bords dans l'ordre cold, z2full, flux, haar).
- Les mutants réutilisent les graines nominales de la gate qu'ils attaquent.
- Un run de gate n'est **jamais** relancé avec d'autres graines pour verdir ; un FAIL est consigné tel quel et une v2 écrit l'arbitrage.
- Fenêtre GPU ≤ 10 min : si un protocole chiffré dépasse la fenêtre, il est coupé en plusieurs lancements (cellules séparées), jamais allégé.

## 5. Écarts assumés, avant les runs
1. **Flux, valeurs trigonométriques exactes.** `box_su2.py` pose `cos(φ x₀)` en flottant (6,1e-17 pour φ x₀ impair multiple de π/2), ce qui rend certaines
   agrafes non nulles (≈ 1e-16) : son tirage par rejet pour α < 1 les traite. Ici, les valeurs 0, ±1 sont posées exactes : l'agrafe est alors **exactement nulle**
   et traverse la branche de Haar des noyaux (`k ≤ 1e-300`), qui est précisément testée par G-DIR-CPU-GPU (a). Écart de la donnée de bord : ≤ 6,2e-17.
2. **Bord `gauge`/`haar`/`therm`** : mêmes lois que QW1, générateur Philox (compteurs `TAG_JAUGE`/`TAG_BORD`) au lieu de numpy : égalité en loi, pas en réalisation.
3. **Sources thermiques** : produites par le moteur (sans masque), clé = graine de la chaîne XOR `0x9E3779B97F4A7C15`, 2000 pas de thermalisation, L_source = max(10, ℓ+2) ;
   QW1 prenait L = 10 et 2000 pas.
4. **Réseau de calcul L_c = ℓ + 2** : seul le gaspillage de fils change (liens gelés = sortie immédiate), pas le tirage des liens libres (compteur Philox = indice de lien du réseau de calcul).
5. **Limites** : ce lot livre l'instrument ; il ne conclut rien sur la contraction de Dobrushin. Le verdict de QW1 (VIVANT) reste celui de QW1.

## 6. Mise en œuvre (fichiers ; leur contenu n'est pas haché ici, seuls les seuils de ce document le sont)
`crates/kvu/validation/boite/gates_dir.sh` (lance les gates, une ligne `PASS|FAIL|SKIP|INFO` par gate, verrous et fenêtres), `analyse_dir.py` (statistique, `python3 -I`,
numpy), `ref_obs.py` (mesures de QW1 sur des configurations NPY), `crates/kvu/tests/boite_*.rs` (tests CPU, GPU en `#[ignore = "GPU"]`), `scripts/pilote_boites.py`.

---

## v2 — arbitrage de l'implémenteur, 2026-10-25 vers 21h40, AVANT tout run de gate GPU ou statistique

**Nature** : un seuil numérique de la gate G-DIR-BORDS, écrit en v1 sans analyse d'erreur d'arrondi, est corrigé. Relevé par le premier test unitaire du module
hôte (`su2::boite::tests::bord_jauge_pure`), avant tout autre run. Aucun autre seuil, aucune graine, aucun protocole ne change.

| ID | texte v1 | texte v2 | raison |
|---|---|---|---|
| G-DIR-BORDS, bord `gauge` | « toutes les plaquettes entièrement fixées valent 1 à ≤ 4e-16 » | **≤ 3e-15** | une plaquette du bord de jauge pure est le produit de 8 facteurs quaternioniques (4 liens, chacun produit de deux g de Haar renormalisé) : l'écart mesuré par le test est de **1,11e-15** (5 ULP de 1), du bruit d'arrondi sans partie systématique ; la borne v1 (4e-16 ≈ 2 ULP) ne tenait compte d'aucune accumulation. Le mutant de la gate (`z2full` pris pour `z2_01`) reste rouge à O(1), donc la gate garde son pouvoir de détection. |

Honnêtement : c'est un **relâchement** d'un seuil numérique (4e-16 → 3e-15), de facteur 7,5 ; il est à relire par Kévin comme les relâchements de GATES v2.

---

## v3 — constat de l'implémenteur, 2026-10-25 vers 23h10, APRÈS la 1re et la 2e passe de G-DIR-PILOTE (c, d)

**Nature** : aucun seuil, aucune graine, aucun protocole ne change ; les textes des gates v1 et v2 restent ceux qui ont été exécutés. La 1re passe réelle du pilote (acemagic en local, gamer par ssh ;
graines 26102610501 à 504 et 26102610510 ; sortie `scratch/wp-dir/g_pilote/verdicts_pilote.txt`) a donné **deux FAIL, deux défauts du pilote** `scripts/pilote_boites.py`, corrigés dans le code et non dans la gate.

| passe 1 | cause | correction (code du pilote) |
|---|---|---|
| (c) FAIL : le lot gamer a tourné (COMPLETE, rc 0) mais le rapatriement `rsync` a échoué (rc 23) | chemin distant cité avec `"$HOME"/…` ; `rsync` à arguments protégés n'a pas de shell distant | `chemin_rsync` et `rsync -s` ; test `chemin_de_rapatriement_sans_shell_distant` |
| (d) gamer FAIL : le lot a démarré **7,05 s avant** la libération du verrou du gamer | un `kvu` lancé par ssh depuis le répertoire personnel, sans `kvu.toml`, ne reconnaît pas le poste et prend le verrou par défaut `/tmp/kevinotron_gpu.lock`, pas `/tmp/gamer_gpu.lock` | `--verrou NOM=CHEMIN` (ou `NOM=aucun`) obligatoire pour une cible ssh ; le pilote enveloppe chaque lot dans `flock -E 75 -w ATTENTE CHEMIN` ; 4 tests nouveaux ; mutant « enveloppe désactivée » ROUGE sur 3 tests |

Conséquence consignée : pendant la 1re passe, le lot gamer de (c) (0,33 s de GPU, ℓ = 4) a tourné pendant qu'une de mes fenêtres statistiques tenait le verrou du gamer : le pilote a violé le verrou,
sans effet sur les données (chaque chaîne est déterministe par sa graine) ni sur les verdicts des fenêtres.

**2e passe** (mêmes graines, mêmes seuils ; `scratch/wp-dir/g_pilote2/verdicts_pilote.txt`) : (c) les deux cibles déclarent leur verrou, `index.json` le consigne ; (d) en trois variantes, acemagic sans déclaration
(le `kvu` local prend lui-même son verrou), acemagic avec déclaration (enveloppe `flock`, `kvu` voit `ancetre`), gamer avec déclaration. Le test (d) reste « le lot ne démarre pas avant la libération du verrou tenu 8 s ».
