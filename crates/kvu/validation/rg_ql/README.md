# Validation du blocage quasi-local (WP22)

Définitions : [BLOCAGE_QUASI_LOCAL.md](../../../../docs/BLOCAGE_QUASI_LOCAL.md).
Protocoles et seuils des gates RG-QL-* : avenant v5.WP22, section A (liste fermée A.3).

- `derivations_ql.py` : calculs indépendants D1–D5 des normalisations APE/flot et des rayons.
- `support_py.py` : parcours en largeur indépendant pour les supports et leurs tailles.
- `gen_references.py` : génération des références Python BLOC2 ; les références sont déjà produites.
- `references/` : 20 champs grossiers (DEC, APE6, APE13, APE40, AGR1 ; trois fixtures 9⁴ et une entrée 12⁴),
  avec `manifeste.json` donnant les sha256, les entrées et la provenance du générateur.
- `rg_ql.json` : certificat à venir, écrit seulement après un run complet vert avec `--grandl --certifier`.

Depuis `crates/kvu`, contrôle ciblé :

```sh
kvu rg ql-gates --fixtures validation/rg/fixtures --references validation/rg_ql/references --seul RG-QL-NORM --fils 6
```

Le run complet de verdict est réservé au responsable après gel. Sans `--grandl`,
les cinq gates L12 sont SKIP et aucun certificat n'est délivré. Les lignes de verdict
sont déterministes ; le bilan seul contient une durée. Le certificat impose les
51 identifiants obligatoires, chacun une fois avec PASS, et refuse toute ligne FAIL
ou SKIP. Les mutants doivent être détectés pour obtenir PASS.

Les 51 identifiants de A.3 contiennent cinq lignes L12 (REF, JAUGE, DEC-ZERO, FLOT, UNIT) :
46 verdicts hors L12 et cinq SKIP sans `--grandl` (la spécification de code A2 disait « six » par
erreur ; l'avenant gelé et sa liste fermée font autorité).
