# Graines de WP12a (préfixe 261026120xx), écrites AVANT les runs de verdict — liste exhaustive

Flux Philox : `Flux::nouveau(graine)` (clé = graine basse, graine haute), compteurs `(chaîne, trajectoire|balayage, indice, étiquette)`. Les tags `haar_q(i, slot, tag, k0, k1)` sont des u32.

| usage | graine / tag |
|---|---|
| champ de Haar L = 6 (FIB-JAC, FIB-LOGDET, CT-9) | `haar_q(i, 0, 26102612, 17, 41)` |
| jauge de Haar de CT-9 | `haar_q(s, 5, 26102612, 29, 43)` |
| L = 6 Wilson β = 2,8 (FIB-JAC, FIB-LOGDET, CT-9) | 26102612102 (`CpuSu2`, init chaud, 2000 × [1 bain thermique + 4 OR]) |
| L = 6 Wilson β = 2,8 (FIB-CHOL) | 26102612103 (2000 pas) |
| directions FIB-JAC / FIB-LOGDET | `Flux(26102612200 + k)` / `Flux(26102612300 + k)`, k = 0..19, compteur (0, k, i, TAG_OUTIL) |
| FIB-RATTLE-POT | M = diag(1,1) : 26102612401 ; M = diag(1,4) : 26102612402 ; mutant « sans terme » : 26102612403 (M = 1), 26102612413 (M = diag(1,4)) ; mutant « signe opposé » : 26102612404, 26102612414 |
| V des jouets | `haar_q(0, 7, 2610250 + k, 5, 9)`, k = 0, 1 |
| FIB-JOUET route B | 26102612501 + 10·idx, idx = ((β ∈ {2 ; 8}, t ∈ {0,5 ; 0,3}, V ∈ {0, 1}) dans cet ordre d'imbrication, V le plus rapide) = 0..7 ; mutant : 26102612551 + 10·(2·i_t + i_V) |
| FIB-JOUET route A | 26102612601 + 10·idx + i_κ (i_κ = 0, 1, 2 pour κ = 1, 2, 4·10⁻³) |
| FIB-A-LIBRE | route A : 26102612701 + k ; référence `CpuSu2` : 26102612711 + k ; mutant : 26102612721 + k ; k = 0, 1, 2 pour β = 2,3 ; 2,8 ; 8 |
| FIB-UBAR-LOCAL | champ de Wilson 26102612801 (2000 pas) ; chaîne 26102612802 |
| FIB-CONTR | U0 : 26102612821 (L = 6 β = 2,8), 26102612822 (L = 6 β = 8), 26102612823 (L = 12 β = 8) ; chaînes : 26102612921, 26102612922, 26102612923 |
| FIB-REV | U0 26102612831 ; équilibrage 26102612832 ; impulsions 26102612833 (compteur (0, k, i)) |
| FIB-ORDRE | U0 26102612841 ; équilibrage 26102612842 ; impulsions 26102612843 |
| FIB-CREUTZ | U0 26102612851 ; chaîne 26102612852 |
| FIB-VOLUME | impulsions : courbe `Flux(26102612801)`, deux liens `Flux(26102612803)` ; quaternions relatifs du jouet : `haar_q(k, 9, 2610280, 3, 8)` |
| FIB-DET, FIB-FILS, FIB-MANIFESTE | graines des tests `tests/fibre_contrainte_cli.rs` (champ de test `Flux(26102612042)`, graines de chaînes écrites dans le fichier) |
| FIB-STAT-CALIB | 26102612990 (principal), 26102612991 (mutant) |
