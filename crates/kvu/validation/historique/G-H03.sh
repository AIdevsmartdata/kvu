#!/usr/bin/env bash
# G-H03 (ancienne P0) : scripts/build_auto.sh RC = 0 et ZÉRO avertissement dans le code unifié, sur la machine.
# Le build va dans une cible jetable HORS /tmp (RAM), nettoyée à la fin. Les avertissements du moteur
# historique (hors crates/kvu/**) sont comptés et rapportés, jamais masqués : ils existent sur main.
GATE=G-H03
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
[ -f "$RACINE/scripts/build_auto.sh" ] || { fail "$GATE" "scripts/build_auto.sh absent"; exit 0; }
exiger_ssd "$GATE" 8
CIBLE="${KVU_TARGET_VALIDE:-$([ "$MACHINE" = gamer ] && echo "$HOME/kvu_unif/cibles/valide" || echo /chemin/vers/projet/chantiers/kevinotron_unification/cibles/valide)}"
case "$(basename "$CIBLE")" in valide*) ;; *) fail "$GATE" "cible de build « $CIBLE » refusée : son nom doit commencer par « valide » (elle est supprimée)"; exit 0;; esac
LOG="$SCRATCH/gh03_build.log"
trap 'rm -rf "$CIBLE"' EXIT
# build_auto.sh : construit le binaire `kevinotron` selon la machine (acemagic : --no-default-features ;
# gamer : --features unifie-cuda), dans la cible jetable, binaire final copié dans la cible aussi (jamais dans
# bin/ du programme : le binaire en cours de validation n'est pas remplacé). KEV_GIT_HEAD est fourni
# pour les copies rsync (sans .git). --sale-ok : l'arbre rsyncé du gamer n'a pas de .git (ignoré sinon).
( cd "$RACINE" && CARGO_TARGET_DIR="$CIBLE" KEVINOTRON_BIN_DIR="$CIBLE/bin" KEV_GIT_HEAD="${KVU_GIT_HEAD:-}" \
    bash scripts/build_auto.sh --sale-ok >"$LOG" 2>&1 )
rc=$?
"$PY" - "$LOG" "$rc" "$CIBLE/build_auto.log" <<'PYEOF'
import os, re, sys
log, rc = open(sys.argv[1], errors='replace').read(), int(sys.argv[2])
if os.path.exists(sys.argv[3]):
    log += '\n' + open(sys.argv[3], errors='replace').read()   # sortie complète de cargo, écrite par build_auto.sh
blocs = re.split(r'\n(?=warning: |error(?:\[|: ))', log)
kvu, autres = [], 0
for b in blocs:
    if not b.startswith('warning:'): continue
    if re.match(r'warning: `[^`]+` \(.*\) generated \d+ warning', b) or 'build failed' in b[:40]: continue   # lignes de bilan de cargo
    m = re.search(r'-->\s+(\S+?):\d+', b)
    chemin = m.group(1) if m else ''
    if chemin.startswith('crates/kvu/') or '/crates/kvu/' in chemin or chemin in ('src/hip/backend.rs',):
        kvu.append(b.split('\n')[0] + ' @ ' + chemin)
    else:
        autres += 1
if rc != 0:
    print(f"FAIL G-H03 build_auto.sh rc={rc} : " + ' '.join(log.strip().split('\n')[-2:]))
elif kvu:
    print(f"FAIL G-H03 {len(kvu)} avertissement(s) dans le code unifié : " + ' | '.join(kvu[:3]))
else:
    print(f"PASS G-H03 build_auto.sh rc=0, 0 avertissement dans crates/kvu ({autres} avertissements hérités du moteur historique, hors périmètre)")
PYEOF
exit 0
