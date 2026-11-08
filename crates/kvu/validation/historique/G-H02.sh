#!/usr/bin/env bash
# G-H02 (ancienne s1) : fichiers de reference/ = inventaire. Trois contrôles :
#  1. sha256 de chacun des fichiers de crates/kvu/reference/ contre SHA256_REFERENCE.txt (relevé à
#     l'intégration, après cmp avec les origines) ;
#  2. les trois empreintes de tête de SOURCES_IMPORTEES.md (su2_hip.cpp ed1e0fce…, su2_cuda.cu 510b7fe9…,
#     su2_bord.cu 0a01a43e…) ;
#  3. les sha256 que SOURCES_IMPORTEES.md liste (sources historiques réutilisées par chemin), s'il y en a.
GATE=G-H02
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
pb=""
( cd "$REF" && sha256sum -c --quiet "$HIST/SHA256_REFERENCE.txt" >"$SCRATCH/gh02.log" 2>&1 ) || pb="$pb reference/ diffère de SHA256_REFERENCE.txt ($(head -c 160 "$SCRATCH/gh02.log" | tr '\n' ' '));"
chk() { local f="$1" pre="$2"; [ "$(sha256sum "$REF/$f" 2>/dev/null | cut -c1-${#pre})" = "$pre" ] || pb="$pb $f sha != $pre…;"; }
chk su2_hip/su2_hip.cpp ed1e0fce
chk su2_cuda/su2_cuda.cu 510b7fe9
chk su2_bord/su2_bord.cu 0a01a43e
# copies listées « <sha>  <chemin> » dans crates/kvu/SOURCES_IMPORTEES.md (chemins relatifs à la racine du dépôt)
n=0
while read -r sha chemin _; do
  [ -f "$RACINE/$chemin" ] || continue
  # Depuis WP1 (dépôt autonome) : les 4 copies héritées sont dans crates/kvu/src/heritage/ et portent le sha256 de leur
  # origine à 0a7a2de, SANS exception (la copie de manifest.rs est à l'octet) ; su2_sweep.hip est lu par les tests à src/hip/.
  n=$((n + 1))
  [ "$(sha256sum "$RACINE/$chemin" | cut -d' ' -f1)" = "$sha" ] || pb="$pb $chemin diffère de SOURCES_IMPORTEES.md;"
done < <(grep -E '^[0-9a-f]{64}  ' "$RACINE/crates/kvu/SOURCES_IMPORTEES.md")
# le SOURCES_IMPORTEES.md de la racine (WP1) est la même table que celle de crates/kvu/ : les deux fichiers sont identiques
if [ -f "$RACINE/SOURCES_IMPORTEES.md" ] && ! cmp -s "$RACINE/SOURCES_IMPORTEES.md" "$RACINE/crates/kvu/SOURCES_IMPORTEES.md"; then
  pb="$pb SOURCES_IMPORTEES.md (racine) diffère de crates/kvu/SOURCES_IMPORTEES.md;"
fi
nref="$(wc -l <"$HIST/SHA256_REFERENCE.txt")"
if [ -n "$pb" ]; then fail "$GATE" "$pb"; else pass "$GATE" "$nref fichiers de reference/ + $n sources héritées = inventaire${note:-}"; fi
exit 0
