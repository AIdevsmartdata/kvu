#!/usr/bin/env bash
# Lancer sous carte-gamer ou carte-acemagic. Utilise le chronomètre de spectro gpu.
set -euo pipefail

if [[ $# != 3 || ( ${1:-} != cuda && ${1:-} != hip ) ]]; then
    echo "Usage : $0 cuda|hip /chemin/kvu /nouvelle/sortie" >&2
    exit 2
fi
if [[ ${KVU_F32_SOUS_VERROU:-} != 1 ]]; then
    echo "Lancer avec KVU_F32_SOUS_VERROU=1 sous le verrou de la carte." >&2
    exit 2
fi

backend=$1
bin=$2
out=$3
repetitions=${REPETITIONS_ORBITES:-3}
nmeas=${NMEAS_ORBITES:-150}
if [[ ! $repetitions =~ ^[1-9][0-9]*$ || ! $nmeas =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS_ORBITES et NMEAS_ORBITES doivent être des entiers strictement positifs." >&2
    exit 2
fi

mkdir -- "$out"
commun=(su2 spectro gpu --backend "$backend" --precision f32
    --cotes 16,16,16,16 --temps 3 --beta 2.39 --repliques 4 --graine 221126006
    --therm 200 --nmeas "$nmeas" --skip 5 --nor 4
    --niveaux 4,16 --alpha 0.5 --tau-max 8 --formes S22
    --blocage 1 --ape-bloc 1 --lissage-bloc 2 --p-bloc 1.0)

mesurer() {
    local repetition=$1 variante=$2
    shift 2
    local dossier="$out/${repetition}_${variante}"
    "$bin" "${commun[@]}" --sortie "$dossier" "$@" > "$dossier.log" 2>&1
}

for ((i=1; i<=repetitions; ++i)); do
    if ((i % 2 == 1)); then
        mesurer "$i" base
        mesurer "$i" paire --impulsions-vecteurs 2:2:1,3:0:0
    else
        mesurer "$i" paire --impulsions-vecteurs 2:2:1,3:0:0
        mesurer "$i" base
    fi
done

python3 - "$out" "$backend" "$bin" "$repetitions" <<'PY' | tee "$out/resume.json"
import json
import math
from pathlib import Path
from statistics import median
import sys

out = Path(sys.argv[1])


def mesure(repetition, variante):
    bilan = json.loads((out / f"{repetition}_{variante}" / "bilan_gpu.json").read_text())
    return {cle: bilan[cle] for cle in ("secondes_mesure", "cfg_par_seconde_total")}


essais = []
for i in range(1, int(sys.argv[4]) + 1):
    base = mesure(i, "base")
    paire = mesure(i, "paire")
    for resultat in (base, paire):
        if any(not math.isfinite(x) or x <= 0 for x in resultat.values()):
            raise ValueError(f"chronométrage invalide dans la répétition {i}")
    essais.append({
        "base": base,
        "paire": paire,
        "rapport_temps_paire_sur_base": paire["secondes_mesure"] / base["secondes_mesure"],
    })

rapport = median(e["rapport_temps_paire_sur_base"] for e in essais)
print(json.dumps({
    "backend": sys.argv[2],
    "binaire": sys.argv[3],
    "essais": essais,
    "mediane_rapport_temps_paire_sur_base": rapport,
    "cible_1_2_atteinte": rapport <= 1.2,
}, indent=2))
PY
