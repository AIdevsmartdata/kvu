# Torsion de 't Hooft dans `kvu su2 gen` (CPU F64)

## Définition physique

La torsion de 't Hooft pour SU(2) modifie l'action de jauge en multipliant certaines plaquettes par un facteur de centre \( z_p \in \{+1, -1\} \) :
\[
S = \beta \sum_p \left(1 - \frac{1}{2} z_p \operatorname{Re}\operatorname{Tr} U_p\right), \qquad z_p = \begin{cases} -1 & \text{si } p \text{ appartient à la pile de torsion d'un plan tordu}, \\ +1 & \text{sinon}. \end{cases}
\]
La torsion est décrite par un tenseur antisymétrique modulo 2, \( n_{\mu\nu} \in \mathbb{Z}_2 \) (donc \( n_{\mu\nu} = n_{\nu\mu} \)). Pour SU(2), le centre est \(\{ \pm 1 \}\).

## Pile et convention d'emplacement

Pour un plan tordu \(\{ \mu, \nu \}\) (avec \( \mu < \nu \)), la **pile** est l'ensemble des plaquettes dont le coin \( y \) vérifie \( y_\mu = L-1 \) et \( y_\nu = L-1 \), toutes les autres coordonnées prenant toutes les valeurs possibles. Il y a donc \( L^2 \) plaquettes dans la pile. Le facteur \( z_p \) ne dépend pas de l'orientation.

Seule la classe \( n_{\mu\nu} \bmod 2 \) est physique : déplacer la pile est un changement de variables (multiplier tous les liens \( U_\nu(y) \) par \(-1\) sur une tranche). La convention de ce générateur est de placer la pile en \((L-1, L-1)\), comme écrit dans le manifeste et dans `torsion.json`.

## Agrafe tordue

La mise à jour locale (heatbath et overrelaxation) utilise l'agrafe tordue, calculée dans `torsion::agrafe_tordue`. Pour le lien \( U_\mu(x) \), pour chaque direction \( \nu \neq \mu \), on considère les deux plaquettes adjacentes :

- \( t = U_\nu(x+\hat\mu) U_\mu(x+\hat\nu)^\dagger U_\nu(x)^\dagger \) (coin \( x \)), facteur \( z(x; \mu, \nu) \) ;
- \( b = U_\nu(x+\hat\mu-\hat\nu)^\dagger U_\mu(x-\hat\nu)^\dagger U_\nu(x-\hat\nu) \) (coin \( x-\hat\nu \)), facteur \( z(x-\hat\nu; \mu, \nu) \).

L'agrafe est \( A = \sum_{\nu \neq \mu} (z_t \cdot t + z_b \cdot b) \). Avec une torsion triviale, cette agrafe est identique, au bit près, à `su2_hb::staple`.

## Plaquette mesurée

Dans les CSV (`plaquette` de `therm.csv` et `measurements.csv`), la colonne `plaquette` contient \( P_{\text{tw}} = \frac{1}{6V} \sum_p z_p \frac{1}{2} \operatorname{Re}\operatorname{Tr} U_p \), reliée à l'action par \(S/(6V)=\beta(1-P_{\text{tw}})\). Elle est invariante par déplacement de la pile. Les composantes \( P_0, P_1, P_2, P_3 \) (boucles de Polyakov) restent **brutes** (non modifiées par la torsion).

## Boucles de Polyakov

Pour \( n_{\mu\nu} = 1 \), la symétrie \( G = (\text{changement de variables qui ramène la pile}) \circ (\text{translation d'un pas selon } \mu)\) vérifie \( G^L P_\nu(x) = -P_\nu(x) \), et de même en échangeant \(\mu\) et \(\nu\). Les valeurs brutes dépendent de l'emplacement de la pile : son déplacement change le signe des boucles de la tranche traversée. Pour le déplacement de T5(a), ce sont les boucles \(P_1(x)\) de la tranche \(x_0=L-1\).

À volume fini, la symétrie de centre impose \(\langle P_\rho\rangle=0\) dans **toute** direction, à tout \(\beta\), sans sélection de secteur : multiplier les liens longitudinaux d'un hyperplan par −1 conserve les plaquettes et inverse \(P_\rho\). Les directions hors du plan sont libres de la contrainte supplémentaire de \(G\), mais gardent cette symétrie de centre. Une moyenne finie signée peut fluctuer ou rester dans un secteur.

Pour chaque configuration, la colonne \(P_\rho\) est la moyenne de \(\tfrac12\operatorname{ReTr}\prod U_\rho\) sur les \(L^3\) sites de base. T5(b) prend ensuite la valeur absolue de cette moyenne spatiale, puis moyenne sur la chaîne : \(\langle|P_\rho|\rangle\), distinct de \(|\langle P_\rho\rangle|\). À \(\beta=6,8\), les critères sont \(\langle|P_0|\rangle_{\rm tw},\langle|P_1|\rangle_{\rm tw}<0,2\), \(\langle|P_2|\rangle_{\rm tw},\langle|P_3|\rangle_{\rm tw}>0,5\), et \(\langle|P_0|\rangle_{\rm per},\langle|P_1|\rangle_{\rm per}>0,4\). L'étalonnage donne respectivement 0,08–0,09, 0,76–0,82 et 0,55–0,63. Les critères signés \(|\langle P_0\rangle_{\rm tw}|,|\langle P_1\rangle_{\rm tw}|<0,05\) sont aussi conservés ; ils ne supposent pas l'absence de passages entre secteurs.

## Énergie de flux — physique corrigée

On définit \( F(\beta) = -\ln(Z_{\text{tw}}/Z_{\text{per}}) \), reliée à la différence d'énergie \( \Delta E = 6V \langle P_{\text{per}} - P_{\text{tw}} \rangle = \mathrm{d}F/\mathrm{d}\beta \). À \(\beta=0\), \(F(0)=\Delta E(0)=0\) exactement par la mesure de Haar des deux ensembles. Ce point est posé analytiquement : aucun run n'est effectué à \(\beta=0\).

Avec une seule torsion \( n_{01} = 1 \) sur \( T^4 \), il existe des **configurations tordues d'action nulle** (« mangeurs de torsion »). Exemple : \( U_0(x) = i\sigma_1 \) pour \( x_0 = L-1 \), \( U_1(x) = i\sigma_2 \) pour \( x_1 = L-1 \), tous les autres liens à l'identité ; alors \( z_p \cdot \frac{1}{2}\operatorname{Tr} U_p = 1 \) pour toutes les plaquettes. Le fond abélien uniforme de flux \( 2\pi/L^2 \) n'est donc **pas** le minimum de l'action, et l'énergie de flux ne tend **pas** vers \( L^4(1-\cos(\pi/L^2)) \approx \pi^2/2 \).

Ce que l'on observe numériquement (étalonnage du chef de projet, CPU, \( L = 4 \), départ froid, \( nor = 1 \), 300 pas de thermalisation, 4 000 mesures, graines identiques pour les runs tordu et périodique, erreurs par blocs de 100 mesures sur la série appariée) :

| \( \beta \) | 0,5 | 1 | 1,5 | 2 | 2,5 | 3 | 3,5 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| \( \Delta E = 6V\langle P_{\text{per}} - P_{\text{tw}} \rangle \) | −0,11 ± 0,32 | 0,15 ± 0,30 | −0,24 ± 0,35 | 0,27 ± 0,60 | **3,49 ± 0,36** | **1,72 ± 0,24** | 0,98 ± 0,20 | 0,61 ± 0,13 | 0,24 ± 0,08 | 0,28 ± 0,07 |
| \( F(\beta) \) (trapèzes, \( F(0)=0 \)) | −0,03 | −0,02 | −0,04 | −0,04 | 0,90 | 2,21 | 2,88 | 3,28 | 4,13 | 4,65 |
| \( \langle P_0 \rangle, \langle P_1 \rangle \) tordu | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,01 / 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| \( \langle P_2 \rangle, \langle P_3 \rangle \) tordu | 0,00 | 0,00 | 0,00 | 0,00 | 0,05 / 0,01 | 0,47 / 0,07 | 0,57 / 0,57 | 0,63 / 0,63 | 0,76 / 0,76 | 0,82 / 0,82 |

Ce tableau est une **mesure d'étalonnage**, avec une graine par β et \(L=4\). À couplage fort, l'effet est non résolu statistiquement ; l'indépendance n'est pas exacte : à volume fixé et pour une torsion, \(F\sim2L^2(\beta/4)^{L^2}\) lorsque \(\beta\to0\). On observe un pic de \(\Delta E\) vers \(\beta=2,5\), puis une décroissance. Le comportement \(\Delta E\approx c/\beta\) et la croissance du coût libre \(F\) sont des interprétations de cet étalonnage, sans détermination analytique de \(c\). Les minima d'action nulle ne suffisent pas à fixer ce coefficient. Les valeurs signées des directions 2 et 3 dans le tableau reflètent les secteurs visités pendant ces chaînes finies ; les tests utilisent les modules configuration par configuration.

## Intégration et incertitudes statistiques (r3)

Les deux chaînes d'un même β partagent leur graine : l'erreur de \(\Delta E\) et celles des six différences par plan sont calculées sur les **différences appariées**, par blocs de 100 mesures. Les β distincts utilisent des graines distinctes et sont indépendants. Pour chaque préfixe de grille, avec \(h_i=\beta_i-\beta_{i-1}\), \(\beta_0=0\) :
\[
F(\beta_k)=\sum_{i=1}^k\frac{h_i}{2}(\Delta E_{i-1}+\Delta E_i)
=\sum_{j=1}^k c_j\Delta E_j,\quad
c_j=\begin{cases}(h_j+h_{j+1})/2&j<k,\\h_k/2&j=k,\end{cases}
\qquad \sigma_F^2=\sum_{j=1}^k c_j^2\sigma_j^2.
\]
Un incrément isolé a l'erreur \((h_i/2)\sqrt{\sigma_{i-1}^2+\sigma_i^2}\), avec \(\sigma_0=0\). Les incréments voisins partagent une estimation de \(\Delta E\) : sommer leurs variances séparément omettrait les termes croisés. T4 conserve les seuils \(F(4)>5\sigma_F\), \(F(8)>5\sigma_F\), et incréments \(>-3\sigma\).

Ces erreurs supposent des **blocs décorrélés** et une chaîne stationnaire. Le contrôle bon marché réutilise les mêmes séries pour des blocs de 200 mesures et exige \(0,5\le\sigma_{\Delta E,200}/\sigma_{\Delta E,100}\le2\), sinon échec « blocs non décorrélés ». Cette stabilité ne démontre pas à elle seule la décorrélation ; \(\sigma_F\) est une erreur statistique et n'inclut pas le biais de quadrature des trapèzes.

## Interface en ligne de commande

- Nouvelle option `--torsion PLANS` pour `su2 gen`, où `PLANS` vaut `aucune` ou une liste de paires (ex. `01,23`). L'ordre des chiffres dans une paire est indifférent (`10` ≡ `01`).
- Une torsion non triviale exige **`--backend cpu` explicite**, précision F64 et \(\beta>0\) pour chaque chaîne. `hip`, `cuda` et `auto` sont refusés sans ouvrir de backend, indépendamment du matériel. `--bord` et `--mesures` (mesures en ligne) sont aussi incompatibles.
- \(\beta=0\) avec torsion est refusé explicitement (`Erreur::Config`, code 2), à cause d'une singularité préexistante du bain thermique historique. Le code `src/heritage/` est conservé ; l'ancrage de Haar ci-dessus reste exact.
- Un run avec torsion écrit `torsion.json` et les champs `torsion`, `action` et `polyakov` du manifeste. Ce dernier décrit les valeurs brutes, leur dépendance à la pile et l'identité \(G^L P_\nu=-P_\nu\).
- `--reprendre DIR` relit la torsion depuis `torsion.json` ; une torsion explicitement différente est refusée (code 2). Les incompatibilités CLI sont validées contre la torsion effective, même sans `--torsion`, avant d'ouvrir le backend. Le chemin commun de l'API compare aussi les options fournies à celles du run avant toute écriture ; les anciens points d'entrée sans options refusent donc la reprise d'un run tordu.

## Tests

Les tests d'acceptation sont dans `tests/su2_torsion.rs` :

- **T1** : torsion triviale = périodique au bit (agrafe, sweeps réécrits, sorties CLI).
- **T2** : action tordue exacte (somme directe, identité d'agrafe corrigée, champ froid, fond abélien, mangeur de torsion).
- **T3** : couplage fort, effet non résolu statistiquement ; différences par plan appariées.
- **T4** : coût libre F positif et compatible avec une croissance, avec incertitudes cumulées corrigées ; ΔE décroît après son pic dans l'interprétation de l'étalonnage.
- **T5** : (a) déplacement de pile et boucles fermées périodiquement ; (b) modules des Polyakov tordus et contraste périodique à β = 6 et 8.
- **T6** : reprise avec torsion.
- **T7** : refus CLI et API, run interrompu avec checkpoint valide, diagnostic de torsion et inventaire des fichiers identique au bit après refus.
- **T8** : tests unitaires de `Torsion` (analyse, symétrie, z, JSON).
