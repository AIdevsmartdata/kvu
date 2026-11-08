# validation/f32 — preuve que le chemin FP32 a la même loi que le FP64 (tâche T4)

Seuils **gelés** dans `loi.py` (`SEUILS`) avant toute donnée ; ils reprennent ARCHITECTURE_UNIFIEE.md §6.2.
Un écart se rapporte tel quel, avec les nombres.

| fichier | rôle |
|---|---|
| `gates_f32.sh` | orchestre les fenêtres GPU (≤ 10 min, verrou rendu entre elles), les builds, le FP64 de référence et le jugement |
| `loi.py` | bibliothèque : moments exacts (Gauss-Tchebychev n = 4000), CDF exacte, LOI-2 (estimateurs de Sokal de `analyse_plaquette.py`), LOI-3 (protocole P5), LOI-4, LOI-τ |
| `juge_f32.py` | lit les sorties des pilotes Rust, imprime `PASS/FAIL/SKIP <gate>` et rend rc 0 si tout est PASS (8 sinon) |
| `../../tests/su2_f32_gpu.rs` | tests GPU (`#[ignore = "GPU"]`) et pilotes : LOI-0, LOI-1, chaînes, LOT-1-F32, REPRISE-1-F32, banc |
| `../../tests/su2_f32_hote.rs` | tests CPU : miroir hôte du tirage de x0 (LOI-1 sans GPU), Philox, assemblage, compilation hors GPU |

## Lancer

```
# acemagic (HIP) : tout, avec jugement (le verrou est pris par fenêtre ; ne PAS l'envelopper dans un flock externe)
bash crates/kvu/validation/f32/gates_f32.sh tout
# gamer (CUDA, cœurs E 12-19, verrou /tmp/gamer_gpu.lock SUR le gamer) : génération seulement
bash crates/kvu/validation/f32/gates_f32.sh gen          # puis rsync de $KVU_LOI_OUT vers acemagic
# acemagic : jugement des sorties du gamer (règle deux machines)
bash crates/kvu/validation/f32/gates_f32.sh juge --gamer-dir /chemin/rapatrie/gamer
bash crates/kvu/validation/f32/gates_f32.sh nettoyer     # supprime le répertoire de travail (scratch)
```

Variables : `KVU_F32_SCRATCH` (défaut `/chemin/vers/projet/kevinotron_dev/scratch/f32`), `CARGO_TARGET_DIR`
(défaut `/chemin/vers/projet/kevinotron_dev/targets/T4`), `PY` (venv `ym4_noyaux`).

## Choix de protocole figés avant les données (non fixés par §6.2)

- LOI-0 : on juge à la fois la fin des 10⁴ pas et le maximum sur les points de contrôle (tous les 1000 pas).
- LOI-2 : la cellule β = 2,3 n'existe pas dans `gpu_port/ref_amd/ens` ; on prend les chaînes AMD de même
  protocole de `wilson_gpu_grandL_20260926/validation/validation` (comme `analyse_gpu1.py`, `refs_amd`).
  Chaînes : c1, c2 départs chauds, c3 départ froid. Comparateurs FP64 à L = 16 : `su2_hip` de `reference/`
  (sha ed1e0fce), rebâti dans le scratch, mêmes options que `build.sh`.
- LOI-3 : P5 avec B = 2000 permutations, graine 27092026, 20000 bins (copie de `analyse_gpu1.p5`) ; la loi de |P|
  porte sur |P₃| à L = 8, β = 2,3, séries amincies de k = ⌈5·max τ_int(|P|)⌉ pas, KS à deux échantillons.
- LOI-τ : τ = moyenne des τ_int(P) des 3 chaînes, σ(τ) = √(Σ δτᵢ²)/3 (δτ de Sokal) ;
  critère |τ₃₂ − τ₆₄|/τ₆₄ ≤ 0,2 + 2·hypot(σ₃₂, σ₆₄)/τ₆₄.
- LOI-4 (b) : mutant Metropolis ROUGE à chacun des α ≥ 5 (5, 20, 100).
