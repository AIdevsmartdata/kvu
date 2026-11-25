# Réduction groupée des orbites d'impulsion

Les noyaux d'orbites ordinaires et bloquées partagent les traces et les coordonnées de 256 sites avec une barrière de bloc. Huit sous-groupes de 32 lanes traitent ensuite les vecteurs. Les phases réelle et imaginaire d'un même vecteur sont évaluées ensemble. Trois étages de réduction se font dans chaque lane et cinq par échanges de registres. La lane 0 reproduit l'ordre des additions de l'ancien arbre, y compris pour les blocs partiels complétés de zéros. Les calculs de phase restent en F64. En mode champ F32, les compilateurs HIP et CUDA peuvent contracter les multiplications-additions.

## Porte numérique

Construire depuis la racine du dépôt avec la feature `cuda` sur le gamer et les features par défaut sur acemagic. Lancer le binaire de test obtenu sous le verrou de la carte.

```bash
cargo test --release -p kvu --features cuda --test su2_spectro_gpu --no-run
KVU_F32_SOUS_VERROU=1 carte-gamer -- /chemin/su2_spectro_gpu-HASH \
  --ignored --nocapture --test-threads=1 spectro_gpu_orbites_hors_axe
```

```bash
cargo test --release -p kvu --test su2_spectro_gpu --no-run
KVU_F32_SOUS_VERROU=1 carte-acemagic -- /chemin/su2_spectro_gpu-HASH \
  --ignored --nocapture --test-threads=1 spectro_gpu_orbites_hors_axe
```

La ligne `ORBITES` doit apparaître pour chacun des quatre couples précision/blocage. Les écarts normalisés au maximum absolu de la référence doivent être inférieurs à `1e-12` pour F64 et `1e-5` pour F32, pour la comparaison CPU, le cran axial sur carte et les transformations cubiques. Le test compare également l'impulsion nulle et les permutations de la liste au bit, et contrôle la réponse du champ unité.

Pour la vérification des fichiers historiques, effectuer une petite série de mesures CLI avec le binaire de référence et le binaire candidat. Fixer le backend, la graine, les côtés et tous les paramètres, et éteindre `--impulsions-vecteurs`. Répéter en F32 et F64 pour une série sans impulsion et une série avec `--impulsions 3 --blocage 1`. Comparer les arbres de sortie avec `diff -rq --exclude=bilan_gpu.json`. Le fichier exclu contient le chronométrage.

## Chronométrage D8

Lancer séparément sur chaque matériel à partir du même commit et conserver le modèle de carte et la version du pilote avec les résultats.

```bash
KVU_F32_SOUS_VERROU=1 carte-gamer -- bash crates/kvu/validation/spectro/banc_orbites.sh \
  cuda /chemin/kvu /sortie/neuve-cuda
KVU_F32_SOUS_VERROU=1 carte-acemagic -- bash crates/kvu/validation/spectro/banc_orbites.sh \
  hip /chemin/kvu /sortie/neuve-hip
```

Le banc reprend la géométrie D8 (`16⁴`, β = 2,39, quatre répliques, champ F32, S22, deux niveaux APE et B1). Il compare zéro impulsion à la paire `2:2:1,3:0:0` sur 150 configurations par réplique, trois fois, en alternant l'ordre des deux variantes. Les 200 balayages de thermalisation précèdent le chronomètre `secondes_mesure` de la CLI. Le chronomètre inclut les balayages entre mesures, la spectroscopie, les transferts et les écritures.

Le fichier `resume.json` contient les débits, les temps et le rapport temps avec orbites sur temps sans impulsion pour chaque paire. La cible par matériel est une médiane des rapports inférieure ou égale à 1,2. Rejouer le même banc avec le binaire de base `ba75fd1` pour une comparaison directe des deux implémentations.
