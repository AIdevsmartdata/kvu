# Fragment v0.2 : mesure en ligne `corr` de `su2 gen` (B08), à assembler par l'agent de documentation

## Ce que fait le module
`kvu su2 gen … --mesures corr:<lissages>:<période>` mesure, pendant la génération et SANS dump, les observables de
`kvu rg correlateur --rmax 0` (`moy_d`, `moy_g`, `c0`, `moy_pp`, `O_t[L]`, `D_t[L]`, `corr` vide) pour chaque lissage, tous les
`<période>` pas de production. Lissages séparés par `+` : `brut`, `flot/<eps>/<n>` (nom JSON `flot:<eps>:<n>`, exactement
l'argument de `rg correlateur --lissage`). Exemple (production du test VL) : `--mesures corr:brut+flot/0.02/50:1`.
- GPU : copie F64 de la chaîne du champ résident (`FlotGpu::tranches_resident`, noyaux `k_obs_copie`, `k_obs_z/maj`,
  `k_corr_plaq/dens/tranches`) ; le champ résident n'est que LU, aucun nombre aléatoire n'est consommé. Champ F32 : rapatrié et
  renormalisé exactement comme le dump. CPU : chemin CPU de `rg correlateur` (`correlateur::mesurer`).
- Direction du temps : axe 0 (`DIR_TEMPS_CORR`), écrite dans chaque ligne (colonne `dir_temps`).
- Fichiers par répertoire de chaîne : `correlateur_enligne.csv` (en ajout ; une ligne par mesure et lissage ; `pas` = pas TOTAL
  `therm + m·skip`, le dump hors ligne correspondant étant `cfg_{m:06}.npy` ; réels en `%.17g`) et, à la clôture d'un run
  terminé, `correlateur.json` au schéma `kvu-rg-correlateur/1` (mêmes clés que `rg correlateur`, `cfg` = `pas_NNNNNN`),
  reconstruit depuis le CSV avec contrôle du calendrier (ni trou ni doublon) et de la finitude ; les deux fichiers sont au
  manifeste du run (sha256). Une valeur non finie arrête le run (pas de COMPLETE).
- Reprise (c08b, générique pour TOUTES les mesures en ligne, sur le modèle de la torsion) : à la création, si `--mesures` est
  donné, la spécification normalisée (lissages, période, mutant compris) et le moteur effectif (hip, cuda ou cpu) sont écrits dans
  `<outdir>/mesures_en_ligne.json` (schéma `kvu-mesures-en-ligne/1`), inscrit au manifeste. À `--reprendre` : `--mesures` absente
  ⇒ RECHARGÉE ; présente et différente ⇒ refus (code 2) ; autre moteur ⇒ refus (code 2) ; ces contrôles précèdent toute écriture,
  troncature ou mesure. Run d'un binaire antérieur (sans ce fichier) : l'argv de création du manifeste fait foi (`--mesures X` ⇒ la
  reprise exige `--mesures X`, sinon refus avec la commande à relancer ; argv illisible ⇒ `--mesures` explicite exigée). Pour
  `corr` : mesures postérieures au point de reprise et ligne partielle retirées (troncature durable), calendrier des lignes
  conservées vérifié dès la reprise, ancien `correlateur.json` supprimé au démarrage de la reprise.
- Durabilité : CSV synchronisé (fsync) après chaque ajout, donc avant tout point de reprise ; JSON synchronisé avant COMPLETE ;
  toutes les chaînes sont validées avant la publication du premier JSON.
- Manifeste : `mesures_en_ligne.json` dans `fichiers` ; provenance du module GPU de mesure sous `modules_mesures` (celle du
  générateur, `modules_compiles`, est inchangée). Admission : le pic de mémoire de travail de la mesure est ajouté au besoin du run
  (formule dans `MesureCorr::besoin_memoire`).
- Mutants des portes (non documentés dans l'aide) : `:mutant=pas-en-moins`, `:mutant=direction` (GPU seulement).
- Branchements : méthode à défaut `MesureEnLigne::terminer` (`su2/run.rs`), `FlotGpu` possède un clone de l'Appareil.

## Portes (B08, 11/11/2026 ; HIP gfx1151 et CUDA RTX 5060 Ti ; `tests/su2_corr_enligne.rs`, `rg::corr_enligne`)
E1 non-perturbation (dumps, therm.csv, measurements.csv identiques à l'octet avec/sans `corr` et au binaire de base ; mutant
« recopie du flot dans le champ résident » rouge) ; E2 en ligne = `rg correlateur --mesure gpu` AU BIT (L = 8 à 32), = `--mesure cpu`
à ≤ 6e-14 ; E3 mutants rouges ; E4 déterminisme ; E5 reprise. Coût mesuré d'une mesure (brut + flot 50 pas) : 2,6 à 3,7 fois
10 balayages selon L et la carte (`ORCHESTRE/b08/RAPPORT.md`).

## Changements de comportement (pour le CHANGELOG)
- `su2 gen --bord … --mesures …` est REFUSÉ (code 2) avant toute écriture : `executer_bord` n'appelle aucune mesure en ligne, et
  la v0.1.0 rendait COMPLETE sans aucune sortie de mesure (défaut reproduit par le chef d'orchestre sur la version publique).
- `su2 gen --reprendre DIR` sans `--mesures` RECHARGE la spécification enregistrée ; pour un run créé par un binaire antérieur avec
  `--mesures X`, la reprise sans `--mesures X` est REFUSÉE (code 2) au lieu de se clore COMPLETE avec des mesures tronquées.
- Ajouter des mesures en ligne à la reprise d'un run créé sans `--mesures` est refusé (code 2).

## Phrase du GUIDE proposée (§ « Reprise à l'octet », à la place de la phrase actuelle)
« `--reprise-every N` écrit des points de reprise atomiques ; `su2 gen --reprendre DIR` continue, dans un nouveau processus, avec la
même torsion et les mêmes mesures en ligne (rechargées de `torsion.json` et `mesures_en_ligne.json` ; une option différente est
refusée), et donne les mêmes octets qu'un run ininterrompu (gate REPRISE-1). »
