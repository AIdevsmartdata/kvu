# Fragment v0.2 : module tube (`kvu su2 gen --cotes`, `kvu su2 tube smin | apparie`), EXPÉRIMENTAL

Statut : **proposition**, sous étiquette EXPÉRIMENTALE : l'aide commence par « [EXPÉRIMENTAL] », les manifestes portent
`parametres.statut = "experimental"`, et chaque lancement écrit sur stderr « module expérimental : interface et formats
susceptibles de changer ; résultats à confirmer indépendamment ». Le module ne porte aucun résultat de physique ; ses portes
n'ont été relues que par une famille de modèles (Claude) : rien n'est acquis avant leur rejeu par une autre famille.

## À quoi sert le module
SU(2) de Wilson sur un tore ANISOTROPE (ℓ0, ℓ1, ℓ2, ℓ3), typiquement deux directions longues et une petite fibre (ℓ, ℓ, L, L),
avec torsion de 't Hooft (pile de coin : convention de `su2/torsion.rs`, la même que `lib_tube.py`). Calcul CPU F64.
- `su2 gen --cotes l0,l1,l2,l3 --backend cpu …` : génération (bain thermique + surrelaxation), mêmes fichiers que `su2 gen`
  (`therm.csv`, `measurements.csv`, `cfg_NNNNNN.npy` de forme (4, ℓ0, ℓ1, ℓ2, ℓ3, 2, 2)) ; avec `--mesures tube:ab:RMAX:TMAX[:N]`,
  `tube.csv` (plaquettes par plan, V = Σ_p (1 − z_p ½Tr U_p), boucles de Wilson brutes et corrigées de la pile, Polyakov,
  fraction de colonnes de vortex du plan ab) et `resume.json` (moyennes et erreurs). Manifeste commun `manifest.json` avec le
  bloc `geometrie` (`kvu-geometrie/1`), `geometrie.json` dans chaque dossier de chaîne, COMPLETE en dernier. À côtés égaux et
  pairs, la chaîne est égale à l'octet à celle de `su2 gen --backend cpu` (tests `egal_a_cpu64_*`).
- `su2 tube smin` : **candidat minimum** de S/β d'un secteur tordu par refroidissement (U ← A†/|A|) jusqu'à
  max|∇S/β| < `--gtol` (fini, > 0) ; une ligne JSON sur la sortie standard, avec `convergence` (atteinte ou non, gradient
  final, balayages) et `nature` ; une convergence non atteinte est aussi dite sur stderr. Un gradient nul ne certifie pas la
  minimalité (point stationnaire : minimum local ou selle). `--bruit` dans [0, 2π], > 0 depuis les départs stationnaires
  `mince` et `abelien`. `--cfg F.npy` écrit la configuration, `F.npy.manifest.json` et le `geometrie.json` du dossier.
- `su2 tube apparie` : deux secteurs appariés (torsion de fibre seule ; fibre + plan long), ⟨V⟩₁ − ⟨V⟩₀ par β avec erreur de
  blocs, répliques indépendantes ; `apparie.json` et, sauf `--series non`, les séries `V_bII_rRR_sS.txt`. Départ `minimum`,
  secteur 0 : vide U*(+,+) si la fibre 23 est tordue, identité sinon ; l'action initiale n'est nulle que si `--fibre` vaut
  exactement 23 ou `aucune`, et NON nulle sinon (par exemple `--fibre 01` sur 4,4,2,2 : S/β = 8 ; `--fibre 01,23` : le vide
  ignore la torsion 01, S/β = 8 aussi) ; `apparie.json` consigne l'action initiale RÉELLE, calculée sur la configuration de
  départ (`action_initiale_secteur0`, S/β) et la décrit en clair
  (`depart_secteur0`). Secteur 1 : candidat minimum refroidi (`candidat_minimum_secteur1`, avec sa convergence).

## Domaine admis (table : `crates/kvu/src/su2/tube/domaine.rs`, reprise par `kvu su2 tube --help`)
Chaque option numérique est confrontée à son intervalle dans l'analyse de la ligne de commande, AVANT les grosses allocations
(champs, piles, chaînes) et AVANT toute écriture ; hors du domaine : refus d'usage, code 2, rien d'écrit, aucune allocation
de champ. Ce qui n'est PAS garanti : les petites listes issues des arguments (par exemple la liste des β) sont construites
avant d'être comptées et refusées, et la table n'est pas l'unique source des plafonds (balayages, chaînes, nombre de β,
répliques sont aussi codés dans `cli.rs`, et ce tableau est une copie manuelle : ils concordent à la relecture c16c, sans que
le code l'impose). Viennent ensuite l'admission mémoire sur des MAJORANTS (vecteurs réservés à leur taille exacte, copies statistiques,
catalogue des fichiers écrits ; code 5) et la projection disque (code 7). Le test `domaine_admis_parcouru_borne_par_borne`
parcourt la table (bornes, juste dehors, NaN, infinis, zéro, maximum du type).

| option | commandes | intervalle | raison |
|---|---|---|---|
| `--cotes` | gen, smin, apparie | chaque côté entier dans [2, 4096] | tore périodique ; avec le volume, garde tables et estimations de mémoire représentables |
| `volume` | gen, smin, apparie | ℓ0ℓ1ℓ2ℓ3 dans [2⁴, 2²⁴] sites | 4V < 2³² (indice de lien du compteur Philox) ; au plus ~3,6 Gio de champ et de tables par chaîne |
| `--beta` | gen, apparie | réel fini dans [10⁻³, 10³] | couplages du module (production : 1 à 48) ; sous 10⁻³, α = β|A| → 0 où le tirage partagé de x₀ peut épuiser ses 10⁶ essais ; au-dessus, marge contre les débordements de β·V |
| `nombre de β` | apparie | 1 à 256 valeurs | tâches β × répliques × secteurs bornées (clés de tâche k < 2¹⁹) |
| `--therm` | gen | entier dans [0, 2³² − 2] | therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne (compteur Philox des balayages sur 32 bits) |
| `--therm` | apparie | entier dans [0, 2³² − 21] | therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne (compteur Philox des balayages sur 32 bits) |
| `--nmeas` | gen | entier dans [1, 2³² − 1] | au moins une mesure ; therm + nmeas·skip ≤ 2³² − 1 |
| `--nmeas` | apparie | multiple de 20 dans [20, 2³² − 16] | moyenne et erreur sur les mêmes 20 blocs égaux ; therm + nmeas·skip ≤ 2³² − 1 |
| `--skip` | gen | entier dans [1, 2³² − 1] | au moins un pas entre deux mesures ; therm + nmeas·skip ≤ 2³² − 1 |
| `--skip` | apparie | entier dans [1, ⌊(2³² − 1)/20⌋ = 214748364] | au moins un pas entre deux mesures ; therm + nmeas·skip ≤ 2³² − 1 avec nmeas ≥ 20 |
| `--nor` | gen, apparie | entier dans [0, 100] | sur-relaxations par pas (production : 1 à 4) ; au-delà, coût sans usage |
| `--chaines` | gen | 1 à 4096 chaînes (listes --beta et --graine comprises) | un lot borné ; graines des chaînes deux à deux distinctes |
| `--graine` | gen, smin, apparie | entier dans [0, 2⁶⁴ − 1] | toute graine 64 bits : clé Philox = graine (gen, smin) ou graine + 7919·k (apparie) |
| `--graine-pas` | gen | entier dans [0, 2⁶⁴ − 1], graines du lot distinctes | graine + i·pas modulo 2⁶⁴ ne doit retomber sur aucune graine du lot |
| `--dump-every` | gen | 0 (jamais) ou entier dans [1, nmeas] | configurations écrites comptées dans la projection disque et la mémoire du catalogue |
| `--mesures tube:…:N (période)` | gen | entier dans [1, ⌊nmeas/2⌋] (au plus 2³¹ − 1) | au moins deux mesures d'observables (moyenne et erreur de blocs) |
| `--mesures tube:ab:RMAX` | gen | entier dans [1, ℓ_a/2] | boucles de Wilson sans recouvrement sur le tore |
| `--mesures tube:ab:RMAX:TMAX` | gen | entier dans [1, ℓ_b/2] | boucles de Wilson sans recouvrement sur le tore |
| `--gtol` | smin | réel fini dans [10⁻¹⁴, 10⁻²] | sous 10⁻¹⁴, plancher d'arrondi du gradient en F64 ; au-dessus de 10⁻², ce n'est plus un point stationnaire |
| `--bruit` | smin | réel fini dans [0, 2π], > 0 pour --depart mince ou abelien | angle de rotation ; mince et abelien sont stationnaires : sans bruit le refroidissement n'en bouge pas |
| `--balayages` | smin | entier dans [0, 2³² − 1] | plafond du refroidissement, exact (dernier paquet tronqué) |
| `--repliques` | apparie | entier dans [1, 1024] | tâches bornées ; clés de tâche k = ib·max(1000, 2·répliques) + 2·réplique + secteur < 2¹⁹ |

Contraintes croisées, vérifiées au même endroit : volume ℓ0ℓ1ℓ2ℓ3 ≤ 2²⁴ ; therm + nmeas·skip ≤ 2³² − 1 ; `--dump-every` ≤
nmeas ; période N ≤ nmeas avec ⌊nmeas/N⌋ ≥ 2 ; R ≤ ℓ_a/2 et T ≤ ℓ_b/2 ; `--nmeas` d'`apparie` multiple de 20 ; `--bruit` > 0
depuis les départs mince et abelien ; graines d'un lot deux à deux distinctes.

Filet final : aucun nombre non fini n'est écrit dans un résultat (CSV d'observables, `resume.json`, `apparie.json`, sortie de
`smin`) ; sinon erreur de validation (code 8), manifeste clos en échec, pas de COMPLETE. Une chaîne dont un lien épuise les
10⁶ essais du bain thermique s'arrête aussitôt (code 8) ; le cumul des essais est saturant.

## Trois exemples (exécutés tels quels, mêmes arguments, par `tests/su2_tube.rs`, test `exemples_de_la_note_tournent`)
```sh
kvu su2 gen --cotes 8,8,2,2 --backend cpu --torsion 01,23 --start vide --beta 2.4 --therm 20 --nmeas 40 --nor 2 --mesures tube:01:2:2:2 --outdir tube_gen
kvu su2 tube smin --cotes 4,4,2,2 --torsion 01,23 --cfg smin.npy
kvu su2 tube apparie --cotes 4,4,2,2 --beta 8,16 --repliques 2 --therm 100 --nmeas 400 --series non --outdir tube_apparie
```
Le premier écrit `tube_gen/` (fichiers ci-dessus, `tube.csv` toutes les 2 mesures) et une ligne `PERF` ; le deuxième imprime
`S_sur_beta` (comparée par le test à la référence de la porte G1 du module, 6,740116866) et écrit `smin.npy` ; le troisième
imprime une ligne `APPARIE` par β et écrit `tube_apparie/apparie.json`.

## Limites
- **CPU F64 seulement** : `--backend cpu` exigé, ni GPU ni F32. À côtés impairs (coloriage glouton), le coût par site est
  d'environ deux fois celui des côtés pairs.
- **Non reprenable** : aucun point de reprise. `su2 gen --cotes … --reprendre` et `su2 gen --reprendre DIR` sur un run du
  module sont refusés (code 2, « non reprenable ») ; un run interrompu (sans COMPLETE) se relance dans un répertoire neuf.
- **Erreurs de blocs SANS temps d'autocorrélation** : 20 blocs égaux, moyenne et erreur sur les mêmes mesures
  (`gen --cotes` : les 20·⌊n/20⌋ premières ; `apparie` exige `--nmeas` multiple de 20). Si les mesures sont corrélées au-delà
  d'un bloc, l'erreur est sous-estimée ; aucune estimation de τ_int n'est faite.
- **Au plus 2³² − 1 balayages de bain thermique par chaîne** (therm + nmeas·skip, calculé en arithmétique vérifiée) : le
  numéro de balayage du compteur Philox tient sur 32 bits ; au-delà, deux balayages réemploieraient les mêmes nombres
  aléatoires. Refus d'usage (code 2) avant toute écriture, pour `gen --cotes` et `apparie` ; `smin --balayages` ≤ 2³² − 1.
- **Tailles et réservations bornées avant toute écriture et avant les grosses allocations de champ** : toutes les estimations d'octets (côtés, chaînes ou tâches,
  séries, R et T des boucles, configurations écrites) en arithmétique vérifiée (débordement : code 2) ; mémoire confrontée à
  l'admission de kvu (`machine::admission`, plancher `KEVINOTRON_RAM_MIN_GO`, code 5), écriture projetée à la garde SSD à
  l'ouverture du manifeste (code 7). `gen --cotes` : `--nmeas` ≥ 1, au plus 4096 chaînes, période des observables ≤ nmeas
  avec au moins 2 mesures d'observables.
- Refus d'usage (code 2, rien d'écrit) : avec `--cotes`, les options de `su2 gen` sans effet sur le tube (`--L`, `--options`,
  `--bc`, `--bc-graine`, `--jauge`, `--pre`, `--bloc`, `--source-hash`, `--bord`, `--zstd`, `--archive`, `--arret-apres`,
  `--fenetre-s`, `--reprise-every` ≠ 0, `--format` ≠ c16, `--precision` ≠ f64, `--mutant metro`) et la période 0 de
  `--mesures tube:…:0` ; pour `apparie`, `--repliques` < 1, `--skip` < 1, `--nmeas` non multiple de 20 (validés avant tout
  calcul) ; pour `smin --cfg`, un dossier dont le `geometrie.json` décrit une autre géométrie (un `geometrie.json` décrit toutes
  les configurations de son dossier).
- Graines d'`apparie` : la tâche (β n° ib, réplique, secteur) a la clé Philox g + 7919·k (mod 2⁶⁴), avec
  k = ib·max(1000, 2·répliques) + 2·réplique + secteur ; K, l'ensemble des k d'un run, est inclus dans [0, 2¹⁹) dans le
  domaine admis (256 β, 1024 répliques). Dans un run, les clés sont deux à deux distinctes par construction (k injectif,
  7919 impair) ; identiques aux versions précédentes jusqu'à 500 répliques.
  **Entre deux runs** de graines g et g′ (ensembles d'indices K et K′) : ils partagent un flux SI ET SEULEMENT SI
  (g′ − g)·7919⁻¹ mod 2⁶⁴ ∈ K − K′ (différences des indices réellement utilisés, prises modulo 2⁶⁴). Exemple : deux β, une
  réplique (K = {0, 1, 1000, 1001}) : les graines 1 et 1 + 7919·1000 partagent des flux. Pour des jeux de clés DISJOINTS,
  prendre les graines g_j = g₀ + 7919·2²⁰·j (j < 2⁴³) : (g_j − g_i)·7919⁻¹ = 2²⁰·(j − i) n'est jamais dans K − K′ ⊂ ]−2¹⁹, 2¹⁹[.
  Lots de `gen --cotes` : clé = graine de la chaîne ; au plus 4096 chaînes, de graines deux à deux distinctes (sinon refus) ;
  deux runs partagent un flux ssi une graine est commune.
- Mutants cachés (`--mutant ordre-couleurs | pile-incomplete | mauvais-plan`) : contrôles négatifs des portes, jamais en
  production.
- Défauts connus du code PARTAGÉ (non modifié, bornés ici par le domaine admis) : `heritage/su2_hb.rs:200-204` (quand
  α = β|A| → 0, exp(−2α) s'arrondit à 1 et la branche de Creutz épuise ses 10⁶ essais) ; cumuls non saturants des essais
  (`heritage/su2_hb.rs:347`, `su2/cpu.rs:230`) ; numéro de balayage sur 32 bits sans borne dans `su2 gen` hors module.

## Fichiers et tests
Code : `crates/kvu/src/su2/tube/` (`reseau.rs`, `obs.rs`, `cli.rs`) sur le socle `geom` (`docs/v0.2/notes/geom.md`). Fichiers
partagés : `su2/mod.rs` (`pub mod tube`), `su2/cli.rs` (sous-commande `tube`, option `--cotes`, aiguillages de `--cotes` et de
`--reprendre`, message de `su2` sans sous-commande). Tests : `tests/su2_tube.rs` (intégration, domaine admis, coloriage contre un
balayage en place, erreurs de blocs, mutants, refus d'usage, exemples de cette note), `tests/su2_tube_philox.rs` (les couples
(clé, compteur) réellement passés à Philox sont deux à deux distincts entre flux) et les tests unitaires `su2::tube::*`.
