# Témoin : run créé par le candidat public 0.1.1, interrompu proprement au point de reprise (11/11/2026, B08)

Binaire : branche `publication/v0.1.1-reprise`, commit 68ea1ec, construit depuis `git archive` sur l'acemagic,
sha256 4fd7865e2f252e8a82dfe6b1954c0d22a88073689a8eb82fed0adc9089a15df4.
Commande (dans l'unité bornée, dossier courant = parent de `run`) :
`kvu su2 gen --backend cpu --L 4 --beta 2.3 --graine 26111112 --start hot --therm 6 --nmeas 6 --skip 2 --reprise-every 2
--arret-apres 10 --mesures wilson:R2xT2:1 --outdir run` ⇒ point de reprise au pas 10, wilson.csv aux pas 8 et 10,
`mesures_en_ligne.json` consigné au manifeste. Ces fichiers sont copiés tels quels (lecture seule) ; le test
`tests/su2_reprise_0_1_1.rs` les recopie dans un répertoire jetable avant de reprendre.
