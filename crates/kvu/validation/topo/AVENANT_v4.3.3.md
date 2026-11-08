# Avenant v4.3.3 (mesure de repli) — WP19 (responsable WP19, Opus), 2026-10-26

**Nature.** Précision OPÉRATOIRE de l'avenant v4.3.2 §« Lieu de mesure », **avant** le lancement des mesures concernées. Aucune graine, aucun seuil, aucune définition, aucune affectation de **génération** n'est changé ;
la génération est terminée pour les 69 chaînes (journal `WP19 GEN`, toutes `CHAINE_OK`). Aucune charge d'un ensemble `lit` ou `lit_mut` n'a été lue par le responsable ; seules les lignes de `q` (3 chaînes sur 4 mesurées) et d'`obs4` ont été analysées
(`OBS-4-L16-REJEU`, INFO), sans effet sur ce qui suit.
**Cause (constat opérationnel).** Les 3 places `gpu_partage` de l'acemagic sont tenues depuis plus d'une heure par d'autres lots (journal : `lit_16` et `lit_15` en attente de place depuis 02:08, aucune ligne produite à 03:00). La mesure des chaînes
**générées sur HIP** ne progresse donc plus, alors que le gamer, dont les 4 places sont à nous, travaille à 100 %.

## Règle
1. Une chaîne **HIP** dont la mesure n'a pas démarré (aucun `topo.jsonl` ni `MESURE_OK` dans `scratch/wp19/topo/<ens>/run_KK/` à l'instant du lancement du repli, liste consignée au journal) peut être mesurée par le **gamer en CUDA** : les dumps sont copiés
   (`rsync`) vers `~/kvu_unif/wp19/ens_hip/<ens>/run_KK`, sha256 des `cfg_*.npy` vérifiés des deux côtés (inventaire N dumps, source `COMPLETE`), puis `kvu obs topo --backend cuda --backend-gen hip` (mêmes options, mêmes paramètres, même binaire gamer
   `ac622a8a…`) ; le record porte `backend_gen = hip`, `backend_mesure = cuda` ; le `topo.jsonl` est rapatrié (sha256 vérifié, N lignes). Les chaînes sont prises **dans l'ordre inverse de la file** (lit 16 → 01, puis lit_mut 16 → 09), par réservation atomique
   du dossier de sortie ; le renfort HIP en ordre inverse est arrêté (ses réservations vides retirées). La mesure HIP d'origine (ordre direct) continue ; une chaîne déjà terminée n'est jamais remesurée ni écrasée.
2. Accord des deux backends de mesure : mêmes témoins que v4.3.2 (TOPO-GPU HIP et CUDA, OBS-3, référence numpy) ; le rendu donne X̂ par backend de **mesure** (INFO) en plus de par backend de génération.
3. Le choix « qui mesure quoi » dépend uniquement de l'état des files (places GPU), jamais d'une valeur de charge.
