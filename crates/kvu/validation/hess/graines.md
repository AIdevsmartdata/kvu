# Graines de WP13 (écrites et hachées AVANT les runs qui portent un verdict)

Graines fixées par le bloc GATES v4-T2 partie 1 §1.3 : 26102613001 (HESS-REG, Haar L = 8), 26102613002 (`kvu su2 gen` L = 8, β = 2,3, 500 pas), 26102613005 (HESS-REG, Haar L = 12),
26102613100 (vecteurs de HESS-REG), 26102613011 (HESS-EXACT, Haar), 26102613012 (HESS-EXACT, `lat.Gauss` β = 8), 26102613021 (G3, Ω de Haar).

Graines non fixées par le bloc (préfixe 261026130xx, choisies par le responsable de WP13 ; générateur : `numpy.random.default_rng(graine)` partout, sauf `kvu su2 gen` : `--graine`) :
| graine | usage |
|---|---|
| 26102613031 | G1 : fond 1, Haar 4⁴ |
| 26102613032 | G1 : fond 2, `lat.Gauss` β = 8 (plein), 4⁴ |
| 26102613033 | G1 : choix des 10 paires de directions (plaquette + 2 liens, directions gaussiennes normalisées) |
| 26102613041 | HESS-LMIN : fond 1, Haar 4⁴ (β = 2,3) |
| 26102613042 | HESS-LMIN : fond 2, `lat.Gauss` β = 8 |
| 26102613043 | HESS-LMIN : fond 3, `lat.Gauss` β = 64 |
| 26102613051 | HESS-ESPACES : fond gaussien β = 8, « grossier » (`coarse=True`) |
| 26102613052 | HESS-ESPACES : fond gaussien β = 64, « grossier » |
| 26102613053 | HESS-ESPACES : fond gaussien β = 8, « plein » (`coarse=False`) |
| 26102613061 | HESS-JVP : fond Haar L = 6 |
| 26102613062 | HESS-JVP : `kvu su2 gen` L = 6, β = 2,8, 500 pas |
| 26102613063 | HESS-JVP : `kvu su2 gen` L = 6, β = 8, 500 pas |
| 26102613064 (+1, +2) | HESS-JVP : 20 directions gaussiennes (fonds Haar, β = 2,8, β = 8) |
| 26102613071 | HESS-KERDU et ref_dim_kerdu.py : fond Haar L = 6 |
| 26102613072 | HESS-KERDU (ii) : 20 vecteurs gaussiens |
| 26102613081 | HESS-FILS : Haar L = 8 ; 26102613083 : 6 vecteurs L = 8 |
| 26102613082 | HESS-FILS : Haar L = 4 (lmin) ; graine LOBPCG 7 |
| 26102613084 | HESS-FILS : 4 vecteurs L = 6 (jvp, espace kerdu) |
| 26102613091 / 26102613092 | supplément `hess_landau.py` : fond Haar 4⁴ / `lat.Gauss` β = 8 (plein) |
Graine de LOBPCG : `--graine 1` (toutes les gates sauf HESS-FILS : 7). Vecteurs de base de HESS-REG : 12 indices distincts tirés par `rng.choice(n, 12, replace=False)` du générateur de graine 26102613100 après les 10 vecteurs gaussiens.
