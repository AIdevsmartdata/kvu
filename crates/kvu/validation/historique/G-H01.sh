#!/usr/bin/env bash
# G-H01 (ancienne s0) : binaire = source. Un petit run CPU (L = 4) du binaire à valider écrit son
# manifeste ; on exige : git_head = HEAD attendu ; binaire_sha256 = sha256 du fichier exécuté ;
# source_hash = recalcul INDÉPENDANT en Python de la recette `kvu-source/1` sur l'arbre source
# (validation/historique/source_hash_kvu.py, qui lit la liste de fichiers dans src/stockage/recette.rs).
# rc 0 si PASS, rc 1 si FAIL (la ligne reste le verdict).
GATE=G-H01
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
# `binaire_ok` sort avec 0 après `fail` : le drapeau conserve ici le verdict à la sortie.
ECHEC=0
fail() { ECHEC=1; echo "FAIL $1 ${2:-}"; }
trap 'if [ "$ECHEC" = 1 ]; then exit 1; fi' EXIT
binaire_ok "$GATE"
D="$SCRATCH/gh01"; rm -rf "$D"
if ! "$BIN" su2 gen --backend cpu --L 4 --beta 2.3 --graine 1 --start hot --therm 2 --nmeas 2 --skip 1 --nor 4 \
      --outdir "$D" >"$SCRATCH/gh01.log" 2>&1; then
  fail "$GATE" "run CPU de L = 4 en échec : $(tail -2 "$SCRATCH/gh01.log" | tr '\n' ' ')"; exit 1
fi
M="$(find "$D" -name manifest.json | sort | head -1)"
[ -z "$M" ] && { fail "$GATE" "aucun manifest.json produit sous $D"; exit 1; }
BINREEL="$(readlink -f "$(command -v "$BIN")")"
SHA_BIN="$(sha256sum "$BINREEL" | cut -d' ' -f1)"
if [ -f "$RACINE/crates/kvu/src/stockage/recette.rs" ]; then
  HASH_ATTENDU="$("$PY" -I "$HIST/source_hash_kvu.py" "$RACINE" 2>/dev/null | head -1)"
else   # arbre de forme historique (arbres jouets des tests de ce script, tests/cli_historique.rs) : recette Unifiee d'origine
  HASH_ATTENDU="$("$PY" "$HIST/source_hash_unifiee.py" "$RACINE" 2>/dev/null | head -1)"
fi
"$PY" - "$M" "${KVU_GIT_HEAD:-}" "$SHA_BIN" "$HASH_ATTENDU" <<'PYEOF'
import json, sys
man = json.load(open(sys.argv[1])); head, sha_bin, h = sys.argv[2:5]
def cle(m, nom):
    if nom in m: return m[nom]
    for v in m.values():
        if isinstance(v, dict) and nom in v: return v[nom]
    return None
pb = []
if not head: pb.append("git HEAD attendu inconnu (KVU_GIT_HEAD vide)")
elif cle(man, 'git_head') != head: pb.append(f"git_head {cle(man, 'git_head')} != HEAD {head[:12]}")
if cle(man, 'binaire_sha256') != sha_bin: pb.append(f"binaire_sha256 {str(cle(man, 'binaire_sha256'))[:12]} != sha du fichier {sha_bin[:12]}")
if not h: pb.append("recalcul du source_hash impossible")
elif cle(man, 'source_hash') != h: pb.append(f"source_hash {str(cle(man, 'source_hash'))[:12]} != recalcul {h[:12]}")
if pb:
    print("FAIL G-H01 " + " ; ".join(pb))
    sys.exit(1)
else: print(f"PASS G-H01 git_head={head[:12]} binaire={sha_bin[:12]} source_hash={h[:12]} (recalculé en Python)")
PYEOF
exit $?
