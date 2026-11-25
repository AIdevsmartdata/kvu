# Fragment v0.2 : module intrication (`kvu intrication`), à assembler par l'agent de documentation

**Statut : proposition, EXPÉRIMENTAL.** Le module a été écrit et relu par une seule famille de modèles ; rien de ce qu'il
calcule n'est un acquis avant un rejeu indépendant et la relecture d'une autre famille. L'aide de `kvu intrication` commence
par « [EXPÉRIMENTAL] », chaque manifeste porte `parametres.statut = "experimental"`, et chaque lancement écrit sur stderr :
« module expérimental : interface et formats susceptibles de changer ; résultats à confirmer indépendamment ».

## À quoi sert le module
Entropie de Rényi-2 par la méthode des répliques (interpolation en α de Buividovich et Polikarpov, arXiv:0802.4247), SU(2)
Wilson 4D : deux répliques d'un réseau L³ × T recollées sur la direction 0 (le temps), région A = tranche x1 ∈ [0, l) ; bain
thermique et surrelaxation de `heritage::su2_hb`, Philox par réplique. La géométrie est le bloc `kvu-geometrie/1` du socle
`geom` (`docs/v0.2/notes/geom.md`), inscrit au manifeste, et c'est CE bloc que le calcul construit. Sous-commandes :
- `portes [--rapide]` : portes (a) à (f) du module et leurs mutants, une ligne JSON par porte ;
- `balayage` : F(l1) − F(l0) par interpolation en α (Simpson, `--nalpha` points impairs ≥ 3) entre deux largeurs ;
- `courbe` : ΔS₂(l) = S₂(l+1) − S₂(l) et C(l+½) = (l+½)³ ΔS₂ / (2L²) pour chaque largeur de `--largeurs` ;
- `doublement` : S₂ de la tranche ENTIÈRE, 2 log Z(T) − log Z(2T), par deux routes indépendantes (interpolation en α de
  A vide à A plein, et intégration thermodynamique en β sur `--nbeta` points) ;
- `debit` : secondes par pas sur la géométrie interpolée (aucun fichier écrit).

## Trois exemples (exécutés tels quels par `tests/intrication_cli.rs`, quelques secondes sur 8 fils)
```text
kvu intrication portes --rapide --sortie runs/portes
kvu intrication courbe --l 4 --t 4 --beta 2.3 --largeurs 1,2 --nalpha 5 --therm 50 --mes 400 --graine 7 --sortie runs/courbe
kvu intrication doublement --l 2 --t 2 --beta 1.0 --nalpha 5 --nbeta 5 --therm 100 --mes 2000 --or 2 --sortie runs/doublement
```

## Lire les sorties
- ΔS₂ est une différence de LOGARITHMES (népériens, sans unité) : `delta_s2` = S₂(l+1) − S₂(l) = F(l+1) − F(l), avec
  S₂(A) = −log Tr ρ_A² et F = −log Z de la géométrie recollée ; `erreur` est son erreur statistique, `delta_s2_trapezes` la
  même intégrale par trapèzes (indicateur d'erreur de quadrature). Pour l + 1 = L, A₁ est toute la tranche spatiale (aire
  de bord nulle) : C n'a plus de sens, un avertissement est écrit sur stderr et dans la ligne (`avertissement`).
- `doublement` rend le code **8** si les deux routes sont en désaccord, |z| ≥ 4 (z = écart / erreur combinée) : la ligne de
  résultat est écrite, le manifeste est clos avec rc 8 (statut `echec`, pas de COMPLETE). Code 0 sinon.
- `portes` rend 0 si toutes les portes comptées sont vertes, 8 sinon.
- FILET, indépendant du domaine : si une chaîne atteint le plafond d'essais du bain thermique (10⁶ essais pour un lien,
  laissé tel quel : tirage dégénéré), ou si une erreur est nulle ou non finie (série constante, configurations répétées),
  la ligne est écrite pour diagnostic puis le run est clos en échec, code 8, sans COMPLETE (`debit` rend 8). Une valeur non
  finie (écrite `null`) dans une ligne de résultat, hors des blocs de géométrie, fait de même.
- Toute option hors du domaine ci-dessous est refusée, code 2, avant l'admission mémoire et toute écriture (aucun
  répertoire créé) ; mémoire insuffisante : code 5.

## Domaine admis des options (`cli::DOMAINE`, contrôlé en un seul endroit, `cli::valider` ; repris par `--help`)
Avec V = 2·L³·T.
| option | intervalle admis | raison |
|---|---|---|
| `--l`, `--t` | pairs, ≥ 2 ; 4·V ≤ 2³¹ et 2·(6·V + 6·L³) ≤ 2³⁰ − 1 | capacités C2 des liens et des plaquettes (geom::tables), volume sans débordement |
| `--l0`, `--l1` | 0 ≤ l ≤ L, --l0 ≠ --l1 | région ; l0 = l1 : ΔF = 0 par construction, erreur nulle |
| `--largeurs` | 0 ≤ l < L, distinctes | l + 1 ≤ L sans débordement ; deux largeurs égales tireraient les mêmes nombres |
| `--beta` | fini, 0,001 ≤ β ≤ 1000 | Creutz : 1 − e^(−2β\|A\|) garde ≥ 12 chiffres pour \|A\| ≥ 0,1 ; réseau gelé au-delà ; aucun infini intermédiaire |
| `--nalpha`, `--nbeta` | impairs, 3 ≤ n ≤ 32767 | quadrature de Simpson ; champ de 15 bits du flux |
| `--therm`, `--mes`, `--sep` | --sep ≥ 1 ; therm + mes × sep < 2³² | le compteur Philox de balayage ne reboucle jamais |
| `--blocs` | 2 ≤ blocs ≤ mes, mes multiple de blocs | moyenne et erreur sur les mêmes observations |
| `--or` | 0 ≤ or ≤ 1000 | surrelaxations par pas |
| `--pas` | pas ≥ 1, 1 + pas < 2³² | debit : pas préparatoire compris |
| `--graine` | tout u64 | seule dans la clé Philox |
| `--sortie` | répertoire qui n'existe pas | un run = un répertoire neuf |
Exemple : L = T = 80 est admis, L = T = 82 ne l'est pas (plafond des plaquettes). La borne basse de β tient à la
soustraction 1 − e^(−2β|A|) du tirage de Creutz (à β = 10⁻²⁰ elle vaut 0 et tout essai est rejeté) ; un |A|
exceptionnellement petit dans le domaine reste couvert par le filet. La route β de `doublement` simule aussi les points
de sa grille, β·i/(nbeta − 1) pour i ≥ 1 : le plus petit vaut au moins 10⁻³/32766 ≈ 3·10⁻⁸ ; 1 − e^(−2β|A|) y garde encore
au moins 7 chiffres pour |A| ≥ 0,1, et le filet couvre tout tirage dégénéré.

## Flux d'aléa
La graine (u64) remplit SEULE la clé Philox ; le 4e mot du compteur porte, sur des bits disjoints, l'usage (départ chaud ou
bain thermique), la route (α, β vide, β pleine, débit), la réplique, le point et la largeur ; lien, balayage et tirage
occupent les trois autres mots. Dans un run de `balayage`, `courbe`, `doublement` ou `debit`, aucun couple (clé, compteur)
n'est consommé deux fois (vérifié sur les couples RÉELLEMENT consommés, départs et rejets compris : `tests/intrication_flux.rs`).
La largeur étant un champ, la largeur l d'une `courbe` et le `balayage --l0 l --l1 l+1` de même graine tirent les mêmes
nombres. Exception VOLONTAIRE : la commande `portes` garde, pour (a) à (d), le compteur de `CpuSu2` à graines fixes, et
réemploie les mêmes flux entre une porte et son mutant ou entre deux portes de comparaison ((b) et f_b ; (c) mutant et (c)
sans identification ; (d1) aux largeurs 1 et 2 et son mutant ; (d3) et son mutant) ; ses portes (e) et (e2) n'en réemploient
aucun. La porte (a) vérifie au bit le tirage du module contre `CpuSu2`.

## Runs et reprise
- Un run = un répertoire NEUF (`--sortie`, refusé s'il existe ; parents créés) : `manifest.json` (`en_cours` à l'ouverture,
  puis `termine` ou `echec` avec rc et raison ; COMPLETE en dernier, seulement si rc = 0) et `resultats.jsonl` (une ligne par
  porte, balayage ou largeur, recopiée sur stdout). Garanties exactes : une erreur du calcul, une panique, une valeur non
  finie, le filet (plafond d'essais, erreur nulle), une erreur d'écriture, de synchronisation (`sync_all`) ou de
  consignation de `resultats.jsonl` clôt le manifeste en échec (statut `echec`, rc, raison ; jamais de COMPLETE). Restent
  hors de cette garantie : (i) un processus tué (signal, manque de mémoire) laisse `en_cours` ; (ii) si l'écriture du
  manifeste FINAL échoue, l'erreur est rendue et le manifeste reste `en_cours` ; (iii) si le manifeste final (`termine`,
  rc 0) est écrit mais que l'écriture de COMPLETE échoue ensuite, la commande rend une erreur alors que le manifeste dit
  `termine` : seul COMPLETE fait foi (code partagé `heritage/manifest.rs`, ordre manifeste puis COMPLETE).
- Pas de `--reprendre`. Reprise MANUELLE, par largeur seule : la largeur l est un champ du flux ; relancer `courbe
  --largeurs l` avec la même graine et les mêmes autres options, dans un nouveau répertoire, rejoue cette largeur à
  l'identique (même machine, même binaire). Une `courbe` interrompue garde ses lignes déjà écrites.

## Limites
- CPU seul (F64, rayon) ; aucun noyau GPU.
- 2 répliques seulement (tout autre nombre est refusé : aucune porte ne le couvre) ; tranche sur l'axe 1 ; côtés spatiaux
  égaux et pairs, T pair.
- Mémoire : `balayage`, `courbe`, `doublement` et `debit` passent l'admission commune de kvu (RAM disponible moins le
  plancher de la politique ; refus code 5) APRÈS `cli::valider` et AVANT toute allocation proportionnelle aux entrées et
  toute écriture ; `portes` (tailles fixes, L ≤ 6, quelques dizaines de Mio) ne la passe pas. Le besoin est calculé sans
  allocation (`besoin_memoire`) : par chaîne, majorant `octets_chaine` = régions + séries de mesures + moyennes de blocs
  (tampons proportionnels aux entrées, AJOUTÉS) + le maximum de la construction et de la marche (de 1,5 à 3,5 ko par site du
  réseau recollé) ; fois le nombre de chaînes simultanées, min(points, fils rayon) ; plus 2 Kio par point. Comme pour
  `su2 gen`, la garde lit /proc/meminfo et ne voit pas un plafond de groupe de contrôle (MemoryMax).
- Erreurs statistiques par moyennes de blocs (`--blocs`, 20 par défaut), SANS temps d'autocorrélation ni contrôle de
  stabilité en taille de bloc ; une erreur sous-estimée n'est pas détectée.
- Portes : (e) et (e2) (couplage fort) lisent la table de plaquettes du module ; elles valident l'échantillonneur et
  l'agrafe, PAS le câblage des largeurs intermédiaires. Celui-ci repose sur (b) (A plein contre un réseau 2T écrit à part),
  (c) (jauge ; mutant de coupure rouge seulement à largeur ≥ 3), (d1), et sur le contrôle non circulaire
  `table_des_largeurs_intermediaires_contre_action_selon_la_spec` (action réécrite depuis la spécification, toutes les
  largeurs, L = 4 et 6) : même famille, même lecture de la spécification que l'auteur.

## Changements de comportement (pour le CHANGELOG)
- Nouvelle sous-commande `kvu intrication` (EXPÉRIMENTAL) ; le binaire séparé `kvu_intrication` de la branche du module
  n'existe plus. Aucune autre sous-commande de kvu n'est modifiée.
