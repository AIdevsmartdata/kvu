# Avenant v4.3.2 à l'avenant v4.3.1 — WP19 (responsable WP19, Opus), 2026-10-26

**Nature.** Corrige deux points OPÉRATOIRES de l'avenant v4.3.1 (§A « Backend », §A « Lieu de mesure ») **avant** le lancement des chaînes concernées. Aucune graine, aucun seuil,
aucune définition de v4.3 ni de v4.3.1 n'est changé ; aucune charge de configuration du lot n'a été lue. Haché au journal (ligne `WP19 GEL avenant v4.3.2`), relu par GPT-6.1-Sol (lecture seule).
**Cause.** Le gel de v4.3.1 s'appuyait sur un débit CUDA de 4,67 pas/s (banc `b1`, dominé par la compilation à froid et la contention). Les fenêtres réelles des quatre chaînes CUDA déjà exécutées (q 03, q 04, lit 32, lit_mut 32,
journal `WP19 GEN FIN`, identiques à l'avenant) donnent, avec quatre processus actifs simultanément : première fenêtre de q 03 = 1 500 pas en 28,4 s (≈ 53 pas/s) ; `lit` 32 = 12 000 pas en 207 s de mur total (≈ 58 pas/s) ; `q` 04 et `q` 03 = 4 500 pas en 91 s ; ordre de grandeur agrégé : ≈ 120 pas/s sur la durée du lot (4 processus, dont trois très courts) et ≈ 200 pas/s sur la première fenêtre, du même ordre que les 156 pas/s d'un processus HIP : la moitié CUDA du brief est réalisable.

## Changements
1. **Backend (remplace le paragraphe « Backend » de v4.3.1 §A).** On rétablit la prescription du brief « moitié HIP, moitié CUDA », affectation STATIQUE : **CUDA** = `q` chaînes 03, 04 ; `lit` chaînes 17…32 ; `lit_mut` chaînes 17…32 ;
   **HIP** = `q` 01, 02 ; `obs4` ; `lit` 01…16 ; `lit_mut` 01…16. Les quatre chaînes CUDA déjà terminées (q 03, q 04, lit 32, lit_mut 32) sont **conservées telles quelles** (même affectation, mêmes graines, même binaire, mêmes options,
   `CHAINE_OK`, plafond cumulé 0 : aucune régénération). Fenêtres (v4.3.1 §A) : CUDA W = 1 500, HIP W = 12 000 ; l'ordre de la file reste q, obs4, lit_mut, lit ; 4 processus CUDA simultanés (gamer), 2 HIP (acemagic). X̂ par backend rapporté en INFO
   (HIP 16 chaînes, CUDA 16 chaînes).
2. **Lieu de mesure (remplace « Lieu de mesure » de v4.3.1 §A, jamais exécuté).** Les dumps ne sont **pas copiés** : chaque chaîne est mesurée sur la machine qui la détient, avec son backend (`backend_mesure` = `hip` ou `cuda` dans le record, `backend_gen`
   identique). Les `topo.jsonl` sont ensuite rapatriés sur acemagic (`rsync`, sha256 des deux côtés vérifiés, chaîne `COMPLETE` et `MESURE_OK` exigés, inventaire attendu N lignes) pour l'analyse. L'accord GPU = CPU est contrôlé sur les témoins prescrits (TOPO-GPU : 3 configurations par machine ; OBS-3 rejouée ; référence numpy indépendante sur une configuration L = 16 hors gates, INFO) ; il n'est pas une borne universelle sur le lot, et le choix du backend de mesure ne dépend d'aucune valeur de charge.
3. Le reste de v4.3.1 est inchangé (fenêtres, plafond, extension par régénération sur le même backend, flot partiel, jackknife, seuil 3,26094…).
