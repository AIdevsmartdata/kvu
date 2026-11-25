# Note v0.2 : le socle de géométrie `geom` et les conventions communes (fragment pour l'agent de documentation)

Candidat `integ/c04-geom` (socle B05 et conventions B01 de la feuille de route v0.2). Ce fragment est à assembler dans
`docs/CONVENTIONS.md` (§ v0.2), le `CHANGELOG.md` et le `GUIDE.md`. Statuts : [lu] = vu dans le code cité ; [règle] = convention v0.2.

## 1. Une seule description de la géométrie : `geom::GeoSpec`

`crates/kvu/src/geom/` décrit, pour TOUT kvu, la géométrie et les conditions au bord d'une configuration : côtés ℓ₀…ℓ₃, direction du
temps (toujours explicite), un bord par direction (`Periodique`, `Fixe` = hyperplan x_μ = 0 de liens transverses gelés, `Recolle` =
répliques recousues), torsion de 't Hooft par plan (`Torsion`, déplacée de `su2::torsion` et réexportée par lui), facteur de β par plan
(optionnel), masque (optionnel : boîte WP-DIR). Les modules l'IMPORTENT au lieu de la recopier.

- `GeoSpec::chemin()` : **C0** (côtés égaux et pairs, tout périodique, sans torsion, sans β par plan, sans masque : le code historique,
  `ChampHote`, `Cpu64`, `Ref64Gpu`, `Rapide32Gpu`), **C1** (régulier : côtés quelconques, torsion, masque, β par plan : `ChaineTube` du tube
  sur CPU, `Cpu64` + torsion M4 à côtés égaux), **C2** (explicite : répliques, `intrication::Reseau`).
- `Tables::compiler(&spec)` (C0, C1) : côtés, pas, V, coefficients 6V (β_plan × z), masque 4V, classes de couleur ; les plaquettes
  explicites de C2 sont un type DÉCLARÉ (`geom::Plaq`, `PlaquettesExplicites`, format de l'intrication) que l'intrication remplira (B07).
- `ChampGeo { geo, q }` : champ de liens de n'importe quelle géométrie ; `ChampHote` reste le cas C0 (`ChampHote::try_from(ChampGeo)`
  refuse un champ non C0, code 2).
- Préréglages : `GeoSpec::cubique`, `anisotrope`, `hyperplan` (R2-3 : `Fixe` sur la direction 0), `boite` (WP-DIR : ℓ⁴ dans L_c = ℓ + 2
  avec le masque `su2::boite::masque_boite`), `repliques` (intrication). Le code de R2-3 et des boîtes N'EST PAS déplacé en v0.2 :
  `GeoSpec` sait seulement le DÉCRIRE dans les manifestes.

## 2. Conventions v0.2 [règle]

1. **Direction du temps explicite.** Tout code neuf porte `dir_temps` (dans `GeoSpec`, dans le bloc `geometrie` de chaque manifeste) ;
   aucun « temps = axe 3 » ou « axe 0 » codé en dur sans constante nommée et consignée. Les commandes v0.1 GARDENT leurs défauts
   historiques (aucun comportement ne change), qu'il faut connaître [lu, base 27eac28] :

   | commande ou module | direction du temps | source |
   |---|---|---|
   | `su2 gen` (Polyakov mesurés dans les 4 directions), `heritage::su2_hb` | 3 (« direction temporelle 3 ») | `heritage/su2_hb.rs:9-10` |
   | `obs wilson`, `obs polyakov` | `--dir-temps`, défaut 3 | `obs/cli.rs:77,90` |
   | mesures en ligne `polyakov…` (`su2 gen --mesures`) | 3 | `obs/enligne.rs:108,119` |
   | `obs champs` (scanner) | `--dir-temps`, défaut 3, consignée seulement | `obs/champs_cli.rs:272` |
   | glueball WP20 (`obs/glueball.rs`, candidat c03) | 3 | `obs/glueball.rs:4` (e8d785f) |
   | `rg correlateur` (tranches 0++) | 0 (x₀, indice le plus lent) | `rg/correlateur.rs:6` ; `DIR_TEMPS_CORR` sur la branche mesure GPU |
   | intrication (répliques empilées sur l'axe 0) | 0 | `intrication/cli.rs` (470ba63) |
   | tube (`su2 gen --cotes`) | 3 déclaré ; physiquement plans longs (0, 1), fibre (2, 3) | `su2/tube/cli.rs` (32dfd27) |
   | bord gelé R2-3 | profondeur = x₀ (liens spatiaux de x₀ = 0 gelés) | `su2/mod.rs:48-53` |

2. **Disposition des sites et des liens.** s = ((x₀·ℓ₁ + x₁)·ℓ₂ + x₂)·ℓ₃ + x₃, pas = [ℓ₁ℓ₂ℓ₃, ℓ₂ℓ₃, ℓ₃, 1], lien li = μ·V + s, lien de x
   vers x + μ̂ ; quaternion (a₀, a₁, a₂, a₃), U = a₀ + i(a₁σ₁ + a₂σ₂ + a₃σ₃). C'est l'ordre C du NPY : forme (4, ℓ₀, ℓ₁, ℓ₂, ℓ₃, 2, 2) en c16,
   (4, ℓ₀, ℓ₁, ℓ₂, ℓ₃, 4) en q64 ; à côtés égaux, octets IDENTIQUES au format v0.1 (porte G-NPY). Répliques (C2) : empilées sur l'axe 0,
   réplique la plus lente ; `cotes` décrit UNE réplique, les côtés stockés sont (s·ℓ₀, ℓ₁, ℓ₂, ℓ₃).
3. **Torsion.** z_p = −1 sur la pile de coin x_μ = ℓ_μ − 1, x_ν = ℓ_ν − 1 de chaque plan tordu (convention M4, `docs/TORSION.md`), +1
   ailleurs ; S = β Σ_p (1 − z_p ½ Re Tr U_p).
4. **Normalisation des densités.** Deux objets distincts, à nommer sans ambiguïté :
   `plaquette` = P = ½ Re Tr U_p (P = 1 à l'identité ; colonne `plaquette` de `su2 gen`) ; `p` ou densité d'action = 1 − ½ Re Tr U_p
   (`rg correlateur` : p_μν(x) ; tube : V = Σ_p (1 − z_p ½Tr U_p)) ; toute autre normalisation porte son facteur dans sa définition écrite
   (scanner : e(x) = 4 Σ_{μ<ν} (1 − ½ Tr P_μν(x)), moyenne = E_plaq = 24(1 − ⟨P⟩)). Avec torsion, une densité qui ignore z_p est FAUSSE :
   un lecteur qui ne sait pas appliquer les z refuse la configuration (règle 6).
5. **Un seul nom de manifeste : `manifest.json`** (`stockage::provenance::Manifeste`). Le bloc `geometrie` y est écrit par
   `GeoSpec::bloc()` (ou `GeoSpec::inscrire(&mut man, "geometrie")`) : aucun module n'écrit ce bloc à la main. La clé
   `geometrie` décrit TOUTES les configurations du dossier du run et de ses sous-dossiers (elle est confrontée à chacune,
   règle 6). Un run dont les configurations ont plusieurs géométries n'écrit PAS de clé `geometrie` : seulement des clés
   `geometrie_<nom>` (ex. `geometrie_secteur0`, `geometrie_secteur1`), au schéma `kvu-geometrie/1`, non examinées par les
   lecteurs, et un `geometrie.json` dans chaque dossier de configurations. Un run SANS configuration (intrication) peut écrire
   `geometrie` (géométrie A₀) et `geometrie_alpha1`. La provenance de la géométrie (fichiers lus, déclaration supposée) va dans un champ séparé `geometrie_source`.
6. **Fichier voisin `geometrie.json` et examen du voisinage** (`geom::examiner_voisinage`, l'UNIQUE implémentation, que le
   scanner et tout lecteur importent). Toute configuration non C0 a un `geometrie.json` dans son dossier
   (`geom::ecrire_geometrie`). Dossiers examinés : celui du fichier et son parent, pris dans le chemin rendu ABSOLU puis dans le
   chemin RÉSOLU (liens symboliques) : la réponse ne dépend pas de l'écriture du chemin (porte G-CHEMINS). Tout y est lu :
   `geometrie.json`, le bloc `geometrie` de `manifest.json` s'il est CANONIQUE (il porte `schema` ; un bloc sans `schema` —
   « 3centre » de `rg`, texte de R2-3, dimensions de la boîte — est une description historique libre, non interprétée), et
   les déclarations historiques (`torsion.json` ; `torsion`, `bord`, `masque`, `mode` de `manifest.json`, de
   `config_run.json` et de leur objet `config_run`). Refus, code 2 : JSON illisible ; déclaration mal formée (une torsion se
   lit strictement : `{"plans": [...]}` avec au plus `pile` et `facteur` = −1, une liste de paires, 6 booléens ou « aucune » ;
   `{}`, `null` dans `torsion.json`, `{"plan": …}` sont REFUSÉS) ; deux déclarations canoniques différentes ; une torsion
   historique différente de celle de la géométrie ; un bord ou un mode « bord » (R2-3, hyperplan x0 = 0) sans bord `Fixe` sur la
   direction 0, un masque ou un mode
   « boite » sans masque ; sans géométrie canonique, toute déclaration historique non C0. Erreur d'E/S réelle (fichier absent,
   dossier illisible) : code 4. **Un lecteur refuse donc (code 2) toute configuration non C0 sans sa géométrie**
   (`stockage::npy::lire_champ_geo`, `ChampHote::try_from`). Conséquence pour les runs : le bloc `geometrie` d'un
   `manifest.json` décrit TOUTES les configurations du dossier du run et de ses sous-dossiers ; un run qui produit des
   configurations de plusieurs géométries n'écrit pas de clé `geometrie` au niveau du run (seulement `geometrie_<nom>`, non
   examinées) et met un `geometrie.json` dans chaque dossier de configurations.
7. **Schémas.** Nom `kvu-<module>/<n>` (champ `schema` du bloc), fichier `schemas/kvu-<module>-<n>.schema.json` (copie embarquée dans
   `crates/kvu/src/…`, égalité testée), écrit dans le sous-ensemble du validateur embarqué `regen/schema.rs` (type, required,
   properties, additionalProperties, items, enum, const, minimum, minItems, minLength, if/then, format sha256). Aucun schéma neuf sans
   l'accord de l'architecte ou du pilote.

8. **Contrat des tables** (doc du module `geom::tables`, à lire par les noyaux GPU de B10 et l'intrication de B07). C1 :
   S = β Σ_{s,p} (1 − c(p, s) ½ Re Tr U_p(s)), plaquette de COIN s du plan p (ordre 01, 02, 03, 12, 13, 23), c(p, s) =
   `coef[p·V + s]` (6V réels) = β_plan(p) × z(p, s), le β GLOBAL hors de `coef` ; agrafe du lien (μ, s) : plaquette haute de coin
   s, plaquette basse de coin s − ν̂ (`coef[plan·V + voisin_bas(s, ν)]`) ; masque 4V indexé par li. Classes : listes de SITES ;
   pour une direction μ FIXÉE, les liens (μ, s) d'une même classe sont indépendants (pas pour deux directions d'un même site) ;
   balayage `for μ { for classe { … } }` avec barrière entre classes et entre directions ; côtés impairs (glouton) : CPU
   seulement, refusés par les noyaux GPU. C2 : entrées `(li << 1) | dague`, incidences `(id << 2) | position` dans 6 cases par
   lien (`6·li + 2·i(μ,ν) + bas`, `MAXCASE` = 4 entrées, case vide `NUL` = u32::MAX) ; capacités vérifiées par
   `GeoSpec::valider` pour un bord recollé : 4V ≤ 2³¹ (`LIMITE_LIENS_C2`) ; majorant de l'union des plaquettes des deux
   géométries de l'interpolation, 2 × (6V + 3·s·ℓ₁ℓ₂ℓ₃) ≤ 2³⁰ − 1 (`LIMITE_PLAQUETTES_C2`) ; l'intrication contrôle aussi le
   nombre réel avant l'encodage.
9. **Un seul bord `Fixe`** en v0.2 (deux hyperplans gelés se coupent et leurs valeurs prescrites y seraient indéfinies) : refus,
   code 2.

## 3. Le bloc `kvu-geometrie/1` (exemple : tube 8 × 8 × 4 × 4, torsion 01,23)

```json
{
  "beta_plans": null,
  "bords": [{"type": "periodique"}, {"type": "periodique"}, {"type": "periodique"}, {"type": "periodique"}],
  "chemin": "C1",
  "coloriage": "parite",
  "cotes": [8, 8, 4, 4],
  "dir_temps": 3,
  "disposition": "s = ((x0*l1 + x1)*l2 + x2)*l3 + x3 ; li = mu*V + s ; …",
  "masque": null,
  "schema": "kvu-geometrie/1",
  "torsion": {"facteur": -1, "pile": "coin (l_mu - 1, l_nu - 1) de chaque plan tordu (convention kvu M4)", "plans": ["01", "23"]}
}
```
Bords : `{"type": "fixe", "valeurs": {"type": "froid" | "chaud" (+ "graine") | "copie" (+ "source") | "npy" (+ "chemin", "sha256") |
"par_chaine"}}` ; `{"type": "recolle", "repliques": 2, "region": {"type": "tranche", "axe": 1, "largeur": w}, "couture": "symetrique"}`.
Masque : `{"type": "boite", "ell": ℓ, "valeurs": {…}}`. `chemin`, `coloriage`, `disposition`, `torsion.pile` sont DÉRIVÉS : écrits par
kvu et vérifiés à la lecture (un bloc qui ment est refusé, code 2).

## 4. Divergences relevées le 11/11 (avant le socle) et réconciliation

| clé | tube (32dfd27) | intrication (470ba63) | scanner (c01, fb793fb) | `kvu-geometrie/1` |
|---|---|---|---|---|
| `schema` | `"kvu-geometrie/1"` | absent | absent | requis |
| côtés | `cotes` [ℓ₀..ℓ₃] | `cotes` [T, L, L, L] (une réplique) | `l` + `dimensions` [L, L, L, L] + `axes` | `cotes` (une réplique si `recolle`) |
| temps | `dir_temps` 3 (constante) | `dir_temps` 0 | `direction_temps` (défaut 3) + `note_direction_temps` | `dir_temps` requis, 0..3 |
| bords | `bords` : 4 chaînes `"periodique"` | `bords` : texte libre ; `repliques` 2 ; `empilement_repliques` (texte) ; `region` {type, axe, definition, largeurs [..]} | `conditions_aux_bords` (texte) | `bords` : 4 objets typés ; recollement : `repliques`, `region` {tranche, axe, **une** largeur}, `couture` |
| torsion | `Torsion::json()` {plans, pile « coins (L-1, L-1) », facteur} | `"aucune"` (texte) | `"aucune (z_p = +1 …)"` (texte) | objet {plans, facteur −1, pile (texte canonique ℓ_μ − 1)} |
| β par plan | `beta_plans` null | absent | absent | `beta_plans` null ou 6 réels > 0 |
| masque | `masque` null | absent | absent | `masque` null ou boîte |
| disposition | `disposition` (texte propre) | dans `empilement_repliques` | — | `disposition` (texte canonique, vérifié) |
| coloriage | `coloriage` parite/glouton | absent | absent | `coloriage` (dérivé, vérifié) |
| chemin | absent | absent | absent | `chemin` C0/C1/C2 (dérivé, vérifié) |
| autres | — | run `portes` : bloc réduit à {note, dir_temps} | `declaration` (fichiers lus) | provenance → champ séparé `geometrie_source` ; runs multi-géométries → `geometrie_<nom>` |

## 5. Marche à suivre pour un auteur de module

1. Construire une `GeoSpec` (préréglage ou champs), `spec.valider()?` ; choisir le moteur d'après `spec.chemin()` ; si le module ne
   sait pas traiter ce chemin (torsion, masque, répliques), REFUSER par `Erreur::Config` (code 2) avant tout calcul.
2. Écrire `spec.inscrire(&mut man, "geometrie")?` dans `manifest.json` ; écrire `geom::ecrire_geometrie(dossier, &spec)?` dans chaque
   dossier qui contient des configurations non C0 ; écrire les champs par `stockage::npy::ecrire_champ_geo` (ou `ecrire_champ` en C0).
3. Lire une configuration par `stockage::npy::lire_champ_geo(chemin, DIR_TEMPS_HISTORIQUE)` (le défaut historique de la commande est
   passé EXPLICITEMENT), ou, pour un lecteur cubique, `geom::geometrie_voisine(cfg)?` puis refus si la géométrie n'est pas C0.
4. Prendre côtés, pas, voisins, z et masque dans `Tables::compiler(&spec)` ; jamais de copie locale de `GeoA`, de la pile ou de
   l'indexation (la copie, c'est le risque R1).
5. Porte de raccord : avec une spec C0, le chemin C1 du module doit égaler C0 à l'octet (CPU).

## 6. Ce que chaque branche doit changer pour se conformer (liste remise au chef d'orchestre)

- **tube** (`module/tube-2026-11-11`, après rebase sur le socle) : dans `su2/tube/reseau.rs`, supprimer les lignes 21-180 (32dfd27 :
  `PLANS`, `indice_plan`, `GeoA` et son `impl`, `Pile` et son `impl`) et les tests `coords_site_voisins`, `coloriage_valide` (lignes
  568-609, désormais dans `geom::reseau::tests`) ; ajouter `pub use crate::geom::reseau::{indice_plan, GeoA, Pile, PLANS};` ; ligne 10 :
  retirer `Erreur, Resultat` de l'import de `crate::commun` ; ligne 13 : déplacer `use crate::su2::torsion::Torsion;` dans
  `mod tests` (il n'y sert plus qu'aux tests). `obs.rs` et `cli.rs` importent `super::reseau::{…}` : inchangés. Dans `cli.rs` :
  remplacer `bloc_geometrie`/`ecrire_geometrie`/`DIR_TEMPS` par `GeoSpec::anisotrope(geo.d, 3).avec_torsion(t)` + `.bloc()` /
  `.inscrire(…)` et `geom::ecrire_geometrie` ; remplacer `ecrire_npy` par `stockage::npy::ecrire_champ_geo` (en-tête et octets
  identiques, écriture atomique avec garde SSD et sha256 en plus) ; le texte `torsion.pile` change (ℓ_μ − 1 au lieu de L − 1).
  `su2 tube apparie` : pas de clé `geometrie` au niveau du run s'il produit des configurations des deux secteurs
  (`geometrie_secteur0`, `geometrie_secteur1` + un `geometrie.json` par dossier). Défaut HÉRITÉ du tube, corrigé dans le socle
  (O3) : `GeoA::nouvelle` calculait V sans contrôle de débordement (65536⁴ ⇒ panique ou V = 0) ; produits vérifiés désormais,
  code 2 ; rien à faire de plus que le `pub use`.
- **intrication** (`module/intrication-2026-11-11`) : remplacer `cli::geometrie(l, t, region)` par
  `GeoSpec::repliques(l, t, 2, 1, largeur)` + `.bloc()` (une géométrie par région : `geometrie` pour A₀, `geometrie_alpha1` pour A₁) ;
  run `portes` : un bloc par porte dans `resultats.jsonl` ou `geometrie` de la porte principale ; `torsion` devient l'objet canonique ;
  remplacer `Plaq` par `pub use crate::geom::Plaq` (mêmes champs e, mu, nu, w0, w1) et `MAXCASE` par `geom::MAXCASE` ; ajouter
  `Reseau::depuis_spec(&GeoSpec, …)` qui refuse tout ce qui n'est pas `Recolle` sur 0 avec ℓ₁ = ℓ₂ = ℓ₃ pairs (B07).
- **scanner** (`obs champs`, c01 puis c01b) : remplacer `dossiers_meta`, `torsion_non_triviale`, `bord_declare` et la détection de
  `verifier_geometrie` par `geom::examiner_voisinage(cfg)` (même dossiers que c01b 6cab9d0, lecture STRICTE des torsions,
  confrontation des déclarations, bloc canonique du manifeste) + refus si une géométrie est déclarée et `chemin() != C0` ;
  `fichiers_lus` va dans `geometrie_source` ; écrire `geometrie` par `GeoSpec::cubique(l, dir_temps).bloc()` et
  la provenance (`declaration`) dans `geometrie_source` ; renommer `direction_temps` → `dir_temps` (via le bloc commun).
- **glueball** (WP20, candidat c03) : remplacer « temps = axe 3 » codé en dur (`obs/glueball.rs:4` et ses boucles) par `dir_temps` d'une
  `GeoSpec`, défaut historique 3 passé explicitement ; côtés inégaux (16³ × 24, 20³ × 16) par `Tables` ; écrire le bloc `geometrie`
  dans son manifeste (l'en-tête `kvu-glu/1` garde `l`, `lt`).
- **mesure GPU / corrélateur** (`module/mesure-gpu-2026-11-11`, `rg correlateur`) : prendre côtés et `dir_temps` d'une `GeoSpec`
  (lue par `lire_champ_geo(…, DIR_TEMPS_CORR = 0)`) au lieu de `rg::bloc::{coordonnees, deplacer, site}` cubiques ; refuser une
  configuration non C0 tant que les noyaux ne savent pas appliquer z et le masque ; écrire le bloc `geometrie` (la constante
  `DIR_TEMPS_CORR` passe dans le bloc).
