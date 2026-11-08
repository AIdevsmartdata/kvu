#!/usr/bin/env bash
# test_installer.sh — régression de scripts/installer_kevinotron.sh, dans un répertoire temporaire, sans GPU.
#
# Faux binaire : un script qui répond à `doctor --json` et `etat --json` (machine, backend et file de jobs
# réglables par variables d'environnement). Un « occupant » est un exécutable placé dans DEST sous
# le nom kevinotron-<…> (copie de /bin/bash qui boucle : son /proc/<pid>/exe pointe dans DEST, comme un vrai run).
# Sortie : une ligne PASS|FAIL INSTALL-<cas> par cas, puis rc = nombre d'échecs.
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$(cd "$ICI/../../../.." && pwd)/scripts/installer_kevinotron.sh"
T=$(mktemp -d "${TMPDIR:-/tmp}/kvu_installer.XXXXXX")
OCC_PID=""
trap '[ -n "$OCC_PID" ] && kill "$OCC_PID" 2>/dev/null; rm -rf "$T"' EXIT
ECHECS=0

verdict() { # nom condition-vraie(0/1) détail
  if [ "$2" = 0 ]; then echo "PASS INSTALL-$1"; else echo "FAIL INSTALL-$1 : $3"; ECHECS=$((ECHECS + 1)); fi
}

# Faux binaire figé (version $1) dans $T/src$1/kevinotron, avec .sha256 et .commit comme build_auto.sh.
faux_binaire() {
  local d="$T/src$1"
  mkdir -p "$d"
  cat > "$d/kevinotron" <<EOF
#!/usr/bin/env bash
# version $1
case "\$1 \${2:-}" in
  "doctor --json") echo "{\"machine\":\"\${FAUX_MACHINE:-acemagic}\",\"backend\":{\"choisi\":\"\${FAUX_BACKEND:-hip}\"}}" ;;
  "etat --json") echo "{\"file\":{\"attente\":\${FAUX_ATTENTE:-0},\"en_cours\":\${FAUX_EN_COURS:-0},\"echec\":0,\"orphelins\":\${FAUX_ORPHELINS:-0}}}" ;;
  *) exit 1 ;;
esac
EOF
  chmod +x "$d/kevinotron"
  sha256sum "$d/kevinotron" | sed "s|  .*|  kevinotron|" > "$d/kevinotron.sha256"
  echo "commit$1" > "$d/kevinotron.commit"
  echo "$d/kevinotron"
}

DEST="$T/bin"
S1=$(faux_binaire 1)
S2=$(faux_binaire 2)
N1="kevinotron-$(sha256sum "$S1" | cut -c1-12)"
N2="kevinotron-$(sha256sum "$S2" | cut -c1-12)"
inst() { bash "$INSTALLER" --machine acemagic --dest "$DEST" "$@" > "$T/out" 2>&1; echo $?; }

# 1. simulation par défaut : rien n'est écrit
rc=$(inst --source "$S1")
[ "$rc" = 0 ] && [ ! -e "$DEST" ] && grep -q SIMULATION "$T/out"
verdict SIMULATION $? "rc=$rc, DEST existe ? $(ls "$DEST" 2>&1)"

# 2. installation : binaire versionné, .sha256, .commit, lien
rc=$(inst --source "$S1" --appliquer)
[ "$rc" = 0 ] && [ -x "$DEST/$N1" ] && [ "$(readlink "$DEST/kevinotron")" = "$N1" ] \
  && [ "$(cat "$DEST/$N1.commit")" = commit1 ] && cmp -s "$S1" "$DEST/$N1"
verdict PREMIERE $? "rc=$rc $(cat "$T/out")"

# 3. idempotence
rc=$(inst --source "$S1" --appliquer)
[ "$rc" = 0 ] && grep -q "déjà installé" "$T/out" && grep -q "vise déjà" "$T/out"
verdict IDEMPOTENT $? "rc=$rc $(cat "$T/out")"

# 4. empreinte altérée ⇒ refus rc 3, rien de neuf
cp "$S2" "$T/altere"; cp "$S2.sha256" "$T/altere.sha256"; cp "$S2.commit" "$T/altere.commit"; echo "# octet" >> "$T/altere"
rc=$(inst --source "$T/altere" --appliquer)
[ "$rc" = 3 ] && [ "$(readlink "$DEST/kevinotron")" = "$N1" ]
verdict SHA-ALTERE $? "rc=$rc"

# 5. mauvaise machine reconnue ⇒ rc 5, lien inchangé
rc=$(FAUX_MACHINE=gamer FAUX_BACKEND=cuda inst --source "$S2" --appliquer)
[ "$rc" = 5 ] && [ "$(readlink "$DEST/kevinotron")" = "$N1" ] && [ ! -e "$DEST/$N2" ]
verdict DOCTOR $? "rc=$rc $(cat "$T/out")"

# 6. file de jobs non vide ⇒ binaire installé, lien NON basculé (rc 4)
rc=$(FAUX_ATTENTE=1 inst --source "$S2" --appliquer)
[ "$rc" = 4 ] && [ -x "$DEST/$N2" ] && [ "$(readlink "$DEST/kevinotron")" = "$N1" ]
verdict FILE-OCCUPEE $? "rc=$rc $(cat "$T/out")"

# 7. un processus exécute un binaire installé ⇒ rc 4
cp /bin/bash "$DEST/kevinotron-occupant"
"$DEST/kevinotron-occupant" -c 'while :; do sleep 1; done' & OCC_PID=$!
sleep 0.2
rc=$(inst --source "$S2" --appliquer)
[ "$rc" = 4 ] && grep -q "pid $OCC_PID" "$T/out" && [ "$(readlink "$DEST/kevinotron")" = "$N1" ]
verdict PROCESSUS-OCCUPANT $? "rc=$rc $(cat "$T/out")"
kill "$OCC_PID" 2>/dev/null; wait "$OCC_PID" 2>/dev/null; OCC_PID=""; rm -f "$DEST/kevinotron-occupant"

# 8. machine libre (des jobs ORPHELINS ne bloquent pas : vérificateur du 04/10) ⇒ bascule, précédent noté
rc=$(FAUX_ORPHELINS=2 inst --source "$S2" --appliquer)
grep -q "2 job(s) orphelin(s)" "$T/out"
verdict ORPHELINS-SIGNALES $? "$(cat "$T/out")"

[ "$rc" = 0 ] && [ "$(readlink "$DEST/kevinotron")" = "$N2" ] && [ "$(cat "$DEST/kevinotron.precedent")" = "$N1" ]
verdict BASCULE $? "rc=$rc $(cat "$T/out")"

# 9. retour arrière
rc=$(inst --revenir --appliquer)
[ "$rc" = 0 ] && [ "$(readlink "$DEST/kevinotron")" = "$N1" ]
verdict REVENIR $? "rc=$rc $(cat "$T/out")"

# 10. un fichier ordinaire à la place du lien n'est jamais écrasé
rm "$DEST/kevinotron"; echo ordinaire > "$DEST/kevinotron"
rc=$(inst --source "$S2" --appliquer)
[ "$rc" = 2 ] && [ "$(cat "$DEST/kevinotron")" = ordinaire ]
verdict FICHIER-ORDINAIRE $? "rc=$rc $(cat "$T/out")"

exit "$ECHECS"
