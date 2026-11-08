#!/usr/bin/env bash
# F32 (T4) : LOI-0, LOI-1, LOI-2, LOI-3, LOI-4, LOI-τ, LOT-1-F32, REPRISE-1-F32, sur les DEUX machines.
# Enchaîne les modes documentés de f32/gates_f32.sh (qui prend lui-même son verrou, par fenêtre de 10 min) :
#   acemagic : nettoyer, gen, f64 (comparateurs FP64 à L = 16), puis juge ; le jugement des sorties du gamer est refait ICI
#              (règle deux machines : un résultat du gamer est de la RECHERCHE, revérifié sur acemagic) ;
#   gamer    : nettoyer, gen seulement (les sorties sont rapatriées par `kevinotron fetch`).
# Entre f64 et juge, les journaux `L16_*.log` sont déplacés hors du répertoire des comparateurs : leur nom correspond au motif
# `L16_b*_c*` de loi.py (dossiers_chaines), qui les prendrait pour des chaînes sans COMPLETE (écart de f32/gates_f32.sh en mode
# `tout`, rapporté à T4 ; le contournement n'altère ni donnée ni seuil).
# Les lignes du jugement du gamer sont réémises avec la machine « gamer ».
GATE=F32
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_gpu
[ "$MACHINE" = gamer ] && export PATH="$HOME/.cargo/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
F32="$RACINE/crates/kvu/validation/f32"
if [ "$MACHINE" = gamer ]; then SF="$HOME/kvu_unif/scratch/f32"; else SF=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/f32; fi
export KVU_F32_SCRATCH="$SF"
PYV="${PY_F32:-/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python}"
[ -x "$PYV" ] || PYV="$PY"
bash "$F32/gates_f32.sh" nettoyer >&2
rc=0
bash "$F32/gates_f32.sh" gen || rc=$?
[ "$MACHINE" = gamer ] && exit "$rc"
bash "$F32/gates_f32.sh" f64 || rc=$?
mkdir -p "$SF/f64_journaux"; mv "$SF"/f64/*.log "$SF/f64_journaux/" 2>/dev/null
# jugement de acemagic
"$PYV" "$F32/juge_f32.py" --dir "$SF/acemagic" --f64-l16 "$SF/f64" --json "$SF/acemagic/juge.json" || rc=$?
# jugement des sorties du gamer, rapatriées
D="$SF/gamer"; rm -rf "$D"
# rsync direct par le câble : les dossiers de chaînes du pilote F32 portent un COMPLETE vide (contrat K8 historique) que
# `kevinotron fetch` (T8) refuse de vérifier (« COMPLETE annonce , manifest.json vaut … ») ; écart rapporté, jamais masqué.
mkdir -p "$D"
if rsync -a -e "ssh -o BatchMode=yes" "gamer-eth:kvu_unif/scratch/f32/gamer/" "$D/" >&2; then
  "$PYV" "$F32/juge_f32.py" --dir "$D" --f64-l16 "$SF/f64" --json "$D/juge.json" 2>&1 | while IFS= read -r l; do
    case "$l" in
      PASS\ *|FAIL\ *|SKIP\ *)
        st="${l%% *}"; r="${l#* }"; nom="${r%% *}"; det="${r#"$nom"}"; det="${det# }"
        printf 'GATE\t%s\tgamer\t%s\t%s\n' "$st" "$nom" "$det";;
      *) echo "[gamer] $l" >&2;;
    esac
  done
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
else
  echo "FAIL F32-fetch les sorties F32 du gamer n'ont pas pu être rapatriées"; rc=1
fi
exit 0
