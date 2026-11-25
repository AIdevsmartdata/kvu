# Transport à longueur physique commune

`transport.py` prépare hors ligne les arguments de `kvu su2 thermo mesure --beta-s` et lit leurs manifestes et CSV. Le binaire kvu doit fournir l’option des couplages séparés, issue de la branche `thermo/couplages-separes-1122` (652b0b3). La collecte à lien intégré est disponible depuis `e916814` et le format avec estimateurs séparés depuis `9611f6c`. En analyse, seul Python 3 et sa bibliothèque standard sont requis.

```bash
python3 -I -B scripts/thermo/transport.py planifier \
  --ell 2.40 --grille 4:0.19,0.6,1.2,1.8 --grille 5:0.19,0.6,1.2,1.8 \
  --grille 6:0.213,0.6,1.2,1.8 --grille 8:0.184,0.6,1.2,1.8 \
  --repliques 64 --therm 200 --nmeas 1000 --graine 50000 \
  --kvu kvu --backend hip --sortie ./mesures-transport > plan.json
```

Les paramètres de mesure et les nœuds de cet exemple servent à illustrer la syntaxe. La qualification du mélange et le devis de précision relèvent du pilote. On peut demander un sous-ensemble d’au moins deux tailles parmi 4, 5, 6 et 8. Pour chaque taille, le générateur inverse `ell = n*a*sqrt(sigma)` par interpolation linéaire de `log(a*sqrt(sigma))` entre deux lignes de la table 1 de LTW03 (hep-lat/0307017). Le JSON conserve les deux lignes utilisées et leurs erreurs publiées. Il fixe le couplage spatial à cette valeur de β et ajoute le même β comme dernier nœud temporel. La table est utilisée aux points finaux isotropes.

Les arguments `argv` de chaque section figurent dans le plan. Relancer la même commande avec `--commandes` pour les afficher comme lignes POSIX citées. L’acquisition s’effectue avec le lanceur GPU et le protocole du pilote. Chaque commande utilise `--cotes T,L,L,L --cotes-b 2T,L,L,L` et le producteur écrit `A` et `B` sous le dossier de la section. Le premier côté est la durée, conformément à la convention de `--beta-s`. Le plan fixe les mêmes graines aux deux bras, réserve des plages distinctes aux sections et ajoute `--impairs oui` quand nécessaire.

```bash
# Forces par nœud sur une paire de dossiers, y compris une grille pilote partielle.
python3 -I -B scripts/thermo/transport.py noeuds DOSSIER_COURT DOSSIER_LONG

# Avec les acquisitions du plan terminées, D vaut arrivée moins départ à durée commune 5.
python3 -I -B scripts/thermo/transport.py lire plan.json --de 4 --vers 5 --type D
```

Pour **E** (départ `n4_t4`, arrivée `n5_t5`), les durées sont 4/8 et 5/10 sur les deux sections. Pour **D** (départ `n4_t5`, même arrivée), elles sont 5/10 sur les deux sections. `lire` exige tous les nœuds du plan jusqu’à la cible isotrope. `--racine` permet de relocaliser les dossiers. Les deux dossiers d’une section peuvent provenir d’acquisitions séparées et doivent partager géométrie spatiale, couplages, conditions de mesure, précision et graines. Le lecteur valide les graines réellement inscrites dans les manifestes et CSV. Il accepte un changement de la graine de base du plan si elle est identique pour les deux bras d’une même section. Chaque empreinte source manquante ou vide est signalée par `null` pour le bras correspondant. Les empreintes connues doivent coïncider.

Les trois commandes acceptent `--estimateur brut|lien-integre|y0|ys`. Les trois estimateurs intégrés ajoutent `--lien-integre` aux commandes de collecte de `planifier`. Pour `noeuds` et `lire`, la force vaut `2*S_t(A) - S_t(B)` en sommes extensives. Le champ `colonne_temporelle` indique le choix effectué pour `S_t`.

| Estimateur | Valeur utilisée pour `S_t` |
| --- | --- |
| `brut` (défaut) | `st`, y compris pour une collecte à lien intégré |
| `lien-integre` | `li_t` dans le format historique, `(y0_t + ys_t) / 2` dans le format séparé |
| `y0` | `y0_t`, liens temporels |
| `ys` | `ys_t`, liens spatiaux |

Les estimateurs intégrés exigent que chaque manifeste annonce son format sous `lien_integre.colonnes_csv`. Les deux bras d’une paire doivent déclarer le même format. Le lecteur exige toutes les colonnes annoncées, sans doublon ni mélange des deux formats, et contrôle leur finitude. `y0` et `ys` requièrent le format séparé. Pour ce format, le lecteur vérifie `y0_t = 0` à `beta_t = 0` et contrôle `y0_tot = y0_t + zs_s` et `ys_tot = ys_t + zs_s` avec une tolérance d’arrondi F64 basée sur l’échelle des termes. Le producteur calcule les totaux en F64 pour les deux précisions de champ.

L’erreur-type par nœud utilise les différences par réplique. Pour les trapèzes signés, le lecteur regroupe les nœuds ayant exactement les mêmes identités de répliques et graines. Dans chaque bloc il combine les scores avant de calculer leur dispersion. Entre blocs, y compris entre tailles, il additionne par défaut les erreurs-types comme borne triangulaire sur l’écart-type. Avec `--justification-independance 'motif documenté'`, il les combine en quadrature si les ensembles de graines des blocs sont disjoints. L’indépendance entre blocs reste une hypothèse du pilote. Un chevauchement partiel de graines impose la borne triangulaire.

`--ancres ancres.json` accepte un objet indexé par section, par exemple `n4_t5`. Chaque borne donne `beta_s`, `beta_t` au premier nœud, `intervalle: [minimum, maximum]` et `source` (texte). Les ancres restent séparées des trapèzes et de leur incertitude statistique. Si les deux ancres sont disponibles, leur différence est encadrée par `[min_arrivee - max_depart, max_arrivee - min_depart]`. Les biais de quadrature, d’échelle, de précision et de mélange restent à contrôler avant toute conclusion physique.

```bash
python3 -I -B -m unittest discover -s scripts/tests -p test_transport_thermo.py -v
```
