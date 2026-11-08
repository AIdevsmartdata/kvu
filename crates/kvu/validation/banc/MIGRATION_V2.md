# JSON de banc version2 — microbanc et profil apparié

Le document porte `schema="kvu-banc/2"`. Une mesure SU(2) porte
`schema_mesure="kvu-banc-su2/2"` et `mode_mesure="apparie"` ou `"micro_seul"`.
Les JSON historiques sans version restent lisibles par les deux rapporteurs.
Une version inconnue ou une mesure version2 sans `microbench` est refusée.

Les taux HB/OR mesurés dans des boucles distinctes restent dans `microbench` :
`s_par_pas=hb_s_par_sweep+nor*or_s_par_sweep`, `pas_par_s=1/s_par_pas` et
`pas_chaines_par_s=K/s_par_pas`. Les anciens champs HB/OR restent des alias.
La référence externe, le ratio unifié/référence et la gate≥0,9 utilisent
exclusivement ce microbanc, exactement comme auparavant. Les tables historiques
et leurs gains F32/F64 lisent désormais cet objet explicitement.

Avec mur, un seul run instrumenté publie `profilage` (`kvu-profil-run/1`) :
`duree_totale_ns` D, `duree_pas_ns` S, `pas_executes` N, `hb_executes`,
`or_executes`, K, nor et terminé. Entiers de nanosecondes monotones :
0<S≤D, N>0, HB=N et OR=N* nor, contrôlés sans ajustement des valeurs.
`pas_par_s` signifie alors N/S ; `mur.pas_par_s` signifie N/D ;
`rapport_mur_sur_noyaux` signifie S/D. Le nombre de chaînes multiplie les
débits par chaîne, jamais les durées du lot.

D couvre l'appel interne complet du générateur, validation/admission,
construction, initialisation, mesures/CSV, contrôle final, fermeture du
manifeste/COMPLETE et destructeurs compris. Le verrou GPU est déjà acquis :
son attente est exclue, ainsi que le JSON de banc et le nettoyage ultérieur.
S somme les intervalles disjoints des vrais HB+nor OR de cet appel. Le travail
antérieur est drainé avant chaque intervalle ; une synchronisation termine
le travail avant la fin de l'intervalle. Les horloges/synchronisations ajoutées
restent dans D ; la synchronisation finale est aussi dans S. En CPU, les
préemptions font partie du temps écoulé. En GPU, S inclut le lancement et
l'attente hôte ; ce n'est pas le seul temps du dispositif.

Le rapport S/D vérifie la cohérence du profilage. Il n'est ni une nouvelle
preuve de performance par rapport au microbanc, ni le débit exact du chemin de
production non instrumenté. L'ancienne comparaison de runs séparés n'est plus
publiée et aucun clamp n'est utilisé. Les tests et le contre-exemple de cet
estimateur sont conservés comme histoire du défaut.

Avec `--sans-mur`, aucun run profilé n'est exécuté : les alias `pas_par_s` et
`pas_chaines_par_s` restent les microbancs, mode `micro_seul`, sans objets
`mur`, `profilage` ou `rapport_mur_sur_noyaux`. Pour une comparaison historique
uniforme, les consommateurs doivent lire `microbench` quel que soit ce mode.

`RapportRun`, les fichiers de données physiques et le chemin non profilé ne
changent pas de format. Un profil partiel après arrêt porte `termine=false` ;
il ne devient pas une mesure complète de banc. Une erreur native garde son
erreur et ne produit aucun profil valide.
