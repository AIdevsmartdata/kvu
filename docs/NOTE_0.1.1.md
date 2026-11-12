# kvu 0.1.1 : note de version / release note

## Français
1. La 0.1.1 corrige trois défauts de la 0.1.0 qui touchaient les **mesures en ligne** (`su2 gen --mesures …`). Pour ces défauts, aucun nombre
   produit par la chaîne elle-même n'est en cause : `measurements.csv`, `therm.csv` et les dumps sont inchangés. Deux limites numériques
   aux marges, hors de tout usage connu, sont décrites à part : PUB-18 et PUB-19 de `docs/DEFAUTS_CONNUS.md`.
2. **Reprise sans `--mesures`** : en 0.1.0, `su2 gen --reprendre DIR` sans repasser `--mesures` clôturait le run (COMPLETE) avec des fichiers
   `wilson.csv`, `polyakov_corr.csv`, `flot.csv`, `flot_t0.csv`, `bloc3.csv`, `bloc3_hist.csv` tronqués, sans message.
3. **`--bord --mesures`** : la 0.1.0 clôturait (COMPLETE) sans écrire aucune mesure. La 0.1.1 refuse (code 2).
4. **Arrêt brutal du processus** (kill, plantage) puis reprise : avec `--mesures`, les mesures postérieures au dernier point de reprise étaient
   écrites deux fois ; pendant la thermalisation, la reprise était refusée (code 4, CSV vide ; parade en 0.1.0 : `docs/DEFAUTS_CONNUS.md`, PUB-14) ;
   la 0.1.1 repart du dernier point de reprise valide (avant le premier, rien à reprendre ; coupure de courant : PUB-05) ;
   un run repris puis terminé était déclaré rouge par `kvu store verifier` (code 8 ; PUB-16, même fichier).
5. Vérifier ses anciens runs : `python3 scripts/verifier_mesures_en_ligne.py DOSSIER` (lecture seule ; `DOSSIER` = un run ou un répertoire
   de runs). Il relit la commande de création dans `manifest.json` et contrôle, pour chaque run TERMINÉ lancé avec `--mesures`, chaque
   fichier de mesure de chaque chaîne (`c000`… pour un lot) : `TRONQUÉ` (défaut 2), `ABSENT` (défaut 3), `DOUBLON` (défaut 4), `INCOMPLET`.
6. Ce qu'il contrôle : une mesure de période `P > 0` est faite aux pas `therm + m·skip`, `m = P, 2P, … ≤ nmeas` ; chacun de ces pas doit
   figurer dans le fichier, sans clé répétée, avec le même nombre de lignes à chaque pas. Avec `P = 0` ou `P > nmeas`, aucun fichier n'est attendu.
7. À la main, pour un fichier : `tail -n +2 D/wilson.csv | cut -d, -f1-3 | sort | uniq -d` ne doit rien afficher (clés : champs 1-3 pour
   Wilson, 1-2 pour `polyakov_corr.csv` et `flot.csv`, 1 pour `flot_t0.csv`, `bloc3.csv`, `bloc3_hist.csv`).
8. Un fichier tronqué, en double ou absent : refaire les mesures, ou ne pas utiliser ce fichier. Pour ces défauts, rien d'autre n'est à refaire.
9. Un run créé par la 0.1.0 AVEC des mesures n'a pas `mesures_en_ligne.json` : pour le reprendre avec la 0.1.1, repasser la même
   `--mesures` (sinon refus, code 2, avec la commande à relancer). Un run sans mesures se reprend comme avant.
10. Autres points connus : `docs/DEFAUTS_CONNUS.md` §3 ; changements : `CHANGELOG.md`, section 0.1.1.

## English
1. 0.1.1 fixes three defects of 0.1.0 affecting **online measurements** (`su2 gen --mesures …`). For these defects, no number produced by the
   Markov chain is concerned: `measurements.csv`, `therm.csv` and the dumps are unchanged. Two numerical limits at the margins, outside
   any known use, are described separately: PUB-18 and PUB-19 in `docs/DEFAUTS_CONNUS.md`.
2. **Resume without `--mesures`**: in 0.1.0, `su2 gen --reprendre DIR` without repeating `--mesures` closed the run (COMPLETE) with truncated
   `wilson.csv`, `polyakov_corr.csv`, `flot.csv`, `flot_t0.csv`, `bloc3.csv`, `bloc3_hist.csv`, with no message.
3. **`--bord --mesures`**: 0.1.0 closed the run (COMPLETE) without writing any measurement. 0.1.1 refuses it (exit code 2).
4. **Process killed** (kill, crash) then resumed: with `--mesures`, measurements taken after the last checkpoint were written twice;
   during thermalization, the resume was refused (exit code 4, empty CSV; workaround for 0.1.0: `docs/DEFAUTS_CONNUS.md`, PUB-14);
   0.1.1 resumes from the last valid checkpoint (before the first one there is nothing to resume; power loss: PUB-05);
   a resumed run, once completed, failed `kvu store verifier` (exit code 8; PUB-16, same file).
5. To check your old runs: `python3 scripts/verifier_mesures_en_ligne.py DIR` (read-only; `DIR` = one run or a directory of runs). It reads
   the creation command from `manifest.json` and checks, for every COMPLETED run started with `--mesures`, each measurement file of each
   chain (`c000`… for a batch): `TRONQUÉ` (truncated, defect 2), `ABSENT` (missing, defect 3), `DOUBLON` (duplicated, defect 4), `INCOMPLET`.
6. What it checks: a measurement with period `P > 0` is taken at steps `therm + m·skip`, `m = P, 2P, … ≤ nmeas`; each of these steps must be
   in the file, with no repeated key and the same number of lines at every step. With `P = 0` or `P > nmeas`, no file is expected.
7. By hand, for one file: `tail -n +2 D/wilson.csv | cut -d, -f1-3 | sort | uniq -d` must print nothing (keys: fields 1-3 for Wilson, 1-2
   for `polyakov_corr.csv` and `flot.csv`, 1 for `flot_t0.csv`, `bloc3.csv`, `bloc3_hist.csv`).
8. A truncated, duplicated or missing file: redo the measurements, or do not use that file. For these defects, nothing else needs redoing.
9. A run created by 0.1.0 WITH measurements has no `mesures_en_ligne.json`: to resume it with 0.1.1, repeat the same `--mesures`
   (otherwise it is refused, exit code 2, with the command to rerun). A run without measurements resumes as before.
10. Other known issues: `docs/DEFAUTS_CONNUS.md` §3; changes: `CHANGELOG.md`, section 0.1.1.
