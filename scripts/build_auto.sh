#!/usr/bin/env bash
# build_auto.sh : construit le crate kvu avec les BONS réglages selon la machine.
# (copie adaptée de scripts/build_auto.sh de l'intégration `unifie/*` ; WP1, 25/10/2026 : dépôt kvu autonome)
# Binaire reproductible d'un chemin à l'autre par --remap-path-prefix (sauf --sans-remap).
#
#   acemagic : HIP seulement (dlopen, aucune dépendance de build), nice 10, -j 8
#   gamer    : --features cuda (CUDA), nice 10, taskset -c 12-19 (cœurs E), -j 6,
#              ~/.cargo/bin/cargo, LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
#
# Garde-fous : refuse si l'espace libre est sous (seuil SSD + 10 Go) — 110 Go sur acemagic, 60 Go sur le
# gamer —, refuse un dépôt sale (sauf --sale-ok), construit TOUJOURS dans un CARGO_TARGET_DIR dédié,
# ne touche JAMAIS aux binaires de production (chantiers/*/bin, ~/ym4_calcul/su2_cuda, kevinotron_dev/bin).
#
# Usage : scripts/build_auto.sh [--crate kvu] [--lto] [--sale-ok] [--nettoyer] [--dry-run]
#                               [--machine acemagic|gamer|inconnue] [--strict-warnings] [--en-ligne]
#                               [--sans-remap] (désactive le remappage des chemins, pour comparaison)
# Lot : variable KVU_LOT (défaut wp1) ; racine des sorties : KVU_SORTIES (défaut : dossier PARENT du dépôt, U).
# Sortie par défaut : <U>/bin/<lot>/<machine> (gamer : ~/kvu_unif/bin/<lot>) ; cible par défaut : <U>/cibles/<lot>
#          (gamer : ~/kvu_unif/cibles/<lot>) ; surcharges : KEVINOTRON_BIN_DIR, CARGO_TARGET_DIR.
#          Le binaire est écrit avec <binaire>.sha256 et <binaire>.commit.
set -u

CRATE=kvu
LTO=0
SALE_OK=0
NETTOYER=0
DRY=0
STRICT=0
ENLIGNE=0
REMAP=1
MACHINE_FORCEE=""

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --crate) CRATE=${2:-}; shift 2 ;;
    --crate=*) CRATE=${1#--crate=}; shift ;;
    --lto) LTO=1; shift ;;
    --sale-ok) SALE_OK=1; shift ;;
    --nettoyer) NETTOYER=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --strict-warnings) STRICT=1; shift ;;
    --en-ligne) ENLIGNE=1; shift ;;
    --sans-remap) REMAP=0; shift ;;
    --machine) MACHINE_FORCEE=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build_auto : option inconnue « $1 »" >&2; usage >&2; exit 2 ;;
  esac
done
case "$CRATE" in kvu) ;; *) echo "build_auto : --crate kvu (reçu « $CRATE » : seul le crate kvu existe dans ce dépôt)" >&2; exit 2 ;; esac

DEPOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
LOT=${KVU_LOT:-wp1}
SORTIES=${KVU_SORTIES:-$(dirname "$DEPOT")}

# ---- 1. machine : surcharge, hôte, puis nvidia-smi / rocminfo ------------------------------------
detecter_machine() {
  if [ -n "$MACHINE_FORCEE" ]; then echo "$MACHINE_FORCEE"; return; fi
  if [ -n "${KEVINOTRON_MACHINE:-}" ]; then echo "$KEVINOTRON_MACHINE"; return; fi
  local hote; hote=$(cat /proc/sys/kernel/hostname 2>/dev/null || hostname)
  case "$hote" in
    acemagic) echo acemagic; return ;;
    pc-nvidia-exemple) echo gamer; return ;;
  esac
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qi "RTX 5060"; then echo gamer; return; fi
  if command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q gfx1151; then echo acemagic; return; fi
  echo inconnue
}
MACHINE=$(detecter_machine)
case "$MACHINE" in acemagic|gamer|inconnue) ;; *) echo "build_auto : machine « $MACHINE » inconnue (acemagic|gamer|inconnue)" >&2; exit 2 ;; esac

# ---- 2. réglages par machine ---------------------------------------------------------------------
HOME_DIR=${HOME:-/home/utilisateur}
case "$MACHINE" in
  acemagic)
    CARGO=cargo; JOBS=8; NICE=10; AFFINITE=""; SEUIL=100; LDPATH=""
    BIN_DEFAUT=$SORTIES/bin/$LOT/acemagic
    CIBLE_DEFAUT=$SORTIES/cibles/$LOT
    FEATURES="" ;;
  gamer)
    CARGO=$HOME_DIR/.cargo/bin/cargo; JOBS=6; NICE=10; AFFINITE="12-19"; SEUIL=50; LDPATH=/usr/local/cuda-13.3/lib64
    BIN_DEFAUT=$HOME_DIR/kvu_unif/bin/$LOT
    CIBLE_DEFAUT=$HOME_DIR/kvu_unif/cibles/$LOT
    FEATURES="--features cuda" ;;
  *)
    CARGO=cargo; JOBS=4; NICE=10; AFFINITE=""; SEUIL=20; LDPATH=""
    BIN_DEFAUT=$SORTIES/bin/$LOT/inconnue
    CIBLE_DEFAUT=$SORTIES/cibles/$LOT
    FEATURES="" ;;
esac
# Surcharge du seuil SSD (même variable que la politique de kvu) ; entier de Go.
if [ -n "${KEVINOTRON_SSD_MIN_GO:-}" ]; then
  case "$KEVINOTRON_SSD_MIN_GO" in
    ''|*[!0-9]*) echo "build_auto : KEVINOTRON_SSD_MIN_GO=« $KEVINOTRON_SSD_MIN_GO » : entier de Go attendu" >&2; exit 2 ;;
    *) SEUIL=$KEVINOTRON_SSD_MIN_GO ;;
  esac
fi
SEUIL_BUILD=$((SEUIL + 10))
CIBLE=${CARGO_TARGET_DIR:-$CIBLE_DEFAUT}
# Les préfixes doivent être absolus, même avec une cible ou un CARGO_HOME relatifs.
CIBLE=$(readlink -m -- "$CIBLE")
CARGO_HOME_EFF=${CARGO_HOME:-$HOME_DIR/.cargo}
CARGO_HOME_EFF=$(readlink -m -- "$CARGO_HOME_EFF")
# Chemin PHYSIQUE du dépôt : CARGO_MANIFEST_DIR vient de getcwd (liens symboliques résolus).
DEPOT_P=$(readlink -f -- "$DEPOT")
SORTIE=${KEVINOTRON_BIN_DIR:-$BIN_DEFAUT}
PAQUET="-p kvu"; BINAIRE=kvu; MODE="paquet kvu"
if [ "$LTO" = 1 ]; then PROFIL="release (LTO fat, codegen-units 1 : build finale)"; LTO_ENV=true; CGU=1; else PROFIL="release (LTO off, codegen-units 8)"; LTO_ENV=off; CGU=8; fi
HORS_LIGNE="--offline"; [ "$ENLIGNE" = 1 ] && HORS_LIGNE=""

CMD="nice -n $NICE"
[ -n "$AFFINITE" ] && CMD="$CMD taskset -c $AFFINITE"
CMD="$CMD $CARGO build --release -j $JOBS${HORS_LIGNE:+ $HORS_LIGNE}${FEATURES:+ $FEATURES} $PAQUET --bin $BINAIRE"

# Production : jamais de sortie ni de cible dans ces dossiers.
interdit() {
  case "$1" in
    # sorties et cibles propres au programme d'unification (dépôt kvu) : autorisées (le reste de chantiers/ est de la production)
    */chantiers/kevinotron_unification/bin/*|*/chantiers/kevinotron_unification/cibles/*|*/chantiers/kevinotron_unification/arbres/*) return 1 ;;
    */chantiers/*|*/ym4_calcul/su2_cuda*|*/kev/target*|/chemin/vers/projet/kevinotron/target*|*/kevinotron_dev/bin|*/kevinotron_dev/bin/*) return 0 ;;
  esac
  return 1
}

plan() {
  echo "PLAN build_auto"
  echo "machine         : $MACHINE"
  echo "mode            : $MODE"
  echo "cargo           : $CARGO"
  echo "features        : ${FEATURES:-(aucune)}"
  echo "profil          : $PROFIL"
  echo "jobs            : -j $JOBS"
  echo "priorité        : nice -n $NICE"
  if [ -n "$AFFINITE" ]; then echo "affinité        : taskset -c $AFFINITE"; else echo "affinité        : aucune"; fi
  echo "LD_LIBRARY_PATH : ${LDPATH:-(inchangé)}"
  echo "cible           : $CIBLE"
  if [ "$REMAP" = 1 ]; then
    echo "remap chemins   : $DEPOT_P=/kvu, $CARGO_HOME_EFF=/cargo, $CIBLE=/cible"
  else
    echo "remap chemins   : désactivé (--sans-remap)"
  fi
  echo "binaire         : $BINAIRE"
  echo "sortie          : $SORTIE"
  echo "seuil SSD       : $SEUIL_BUILD Go libres minimum ($SEUIL + 10)"
  echo "dépôt           : propre exigé (sauf --sale-ok)"
  echo "KEV_GIT_HEAD    : exporté (git rev-parse HEAD)"
  echo "nettoyage       : $([ "$NETTOYER" = 1 ] && echo "cargo clean après la copie" || echo non)"
  echo "commande        : $CMD"
  echo "--- fin du plan ---"
}

libre_go() { # Go libres du système de fichiers qui contient $1 (ou son premier ancêtre existant)
  local p=$1
  while [ ! -e "$p" ] && [ "$p" != "/" ]; do p=$(dirname "$p"); done
  df --output=avail -BG "$p" 2>/dev/null | tail -1 | tr -dc '0-9'
}

if [ "$DRY" = 1 ]; then
  plan
  # Informatif (hors golden) : l'état réel des garde-fous, sans rien créer.
  echo "libre           : $(libre_go "$CIBLE") Go"
  if [ -d "$DEPOT/.git" ] || git -C "$DEPOT" rev-parse --git-dir >/dev/null 2>&1; then
    if [ -n "$(git -C "$DEPOT" status --porcelain --untracked-files=no)" ]; then echo "état du dépôt   : SALE"; else echo "état du dépôt   : propre"; fi
  else
    echo "état du dépôt   : hors git (copie rsync) : KEV_GIT_HEAD=${KEV_GIT_HEAD:-non fixé}"
  fi
  exit 0
fi

# ---- 3. garde-fous -------------------------------------------------------------------------------
case "$CIBLE" in ""|/|"$HOME_DIR"|"$HOME_DIR"/|"$DEPOT"|"$DEPOT"/) echo "REFUS : cible $CIBLE trop large (cargo clean la viderait)" >&2; exit 2 ;; esac
if interdit "$CIBLE"; then echo "REFUS : la cible $CIBLE est un dossier de production" >&2; exit 2; fi
if interdit "$SORTIE"; then echo "REFUS : la sortie $SORTIE est un dossier de production" >&2; exit 2; fi
LIBRE=$(libre_go "$CIBLE")
if [ -z "$LIBRE" ] || [ "$LIBRE" -lt "$SEUIL_BUILD" ]; then
  echo "REFUS : ${LIBRE:-?} Go libres sous $CIBLE, il en faut au moins $SEUIL_BUILD (seuil SSD $SEUIL + 10)." >&2
  echo "        Libérer de la place (kvu store du / gc) avant de construire." >&2
  exit 7
fi

cd "$DEPOT" || exit 2
if git rev-parse --git-dir >/dev/null 2>&1; then
  if [ "$SALE_OK" != 1 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "REFUS : dépôt $DEPOT sale (le binaire doit correspondre à un commit). --sale-ok pour passer outre." >&2
    git status --porcelain --untracked-files=no >&2
    exit 2
  fi
  KEV_GIT_HEAD=$(git rev-parse HEAD)
  [ -n "$(git status --porcelain --untracked-files=no)" ] && KEV_GIT_HEAD="$KEV_GIT_HEAD+sale"
elif [ -n "${KEV_GIT_HEAD:-}" ]; then
  echo "dépôt hors git (copie rsync) : état déclaré par KEV_GIT_HEAD=$KEV_GIT_HEAD"
  # Sans empreinte attendue, le binaire porte « +non_verifie » ; avec, « +sale » si la copie diffère du commit
  # (empreinte de la recette kvu-source/1 calculée sur l'arbre propre : python3 -I crates/kvu/validation/historique/source_hash_kvu.py <racine>).
  if [ -n "${KEV_SOURCE_HASH_ATTENDU:-}" ]; then
    export KEV_SOURCE_HASH_ATTENDU
    CALCULE=$(python3 -I "$DEPOT/crates/kvu/validation/historique/source_hash_kvu.py" "$DEPOT" 2>/dev/null | head -1)
    if [ "$CALCULE" != "$KEV_SOURCE_HASH_ATTENDU" ]; then
      echo "AVERTISSEMENT : la copie diffère du commit déclaré (empreinte $CALCULE) : git_head « +sale »" >&2
      KEV_GIT_HEAD="${KEV_GIT_HEAD%%+*}+sale"
    fi
  else
    echo "AVERTISSEMENT : KEV_SOURCE_HASH_ATTENDU non fixé : git_head embarqué « ${KEV_GIT_HEAD%%+*}+non_verifie »" >&2
    case "$KEV_GIT_HEAD" in *+sale) ;; *) KEV_GIT_HEAD="${KEV_GIT_HEAD%%+*}+non_verifie" ;; esac
  fi
else
  echo "REFUS : pas de dépôt git ici et KEV_GIT_HEAD non fixé (copie rsync : exporter KEV_GIT_HEAD=<commit>)." >&2
  exit 2
fi
export KEV_GIT_HEAD

# ---- 4. build ------------------------------------------------------------------------------------
if [ "$REMAP" = 1 ]; then
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$DEPOT_P=/kvu --remap-path-prefix=$CARGO_HOME_EFF=/cargo --remap-path-prefix=$CIBLE=/cible"
fi
mkdir -p "$CIBLE" "$SORTIE" || exit 4
export CARGO_TARGET_DIR=$CIBLE
export CARGO_PROFILE_RELEASE_LTO=$LTO_ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=$CGU
[ -n "$LDPATH" ] && export LD_LIBRARY_PATH=$LDPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ -x "$CARGO" ] || command -v "$CARGO" >/dev/null 2>&1 || { echo "REFUS : $CARGO introuvable (rustup en local utilisateur : curl https://sh.rustup.rs)" >&2; exit 2; }

plan
JOURNAL=$CIBLE/build_auto.log
echo "build en cours (journal : $JOURNAL)"
$CMD > "$JOURNAL" 2>&1
RC=$?
AVERT=$(grep -c '^warning' "$JOURNAL")
echo "cargo build rc=$RC, avertissements : $AVERT"
if [ $RC -ne 0 ] || grep -qE '^error' "$JOURNAL"; then echo "ECHEC build :"; tail -n 30 "$JOURNAL"; exit 1; fi
if [ -n "$(git rev-parse --git-dir 2>/dev/null)" ] && [ "$SALE_OK" != 1 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ECHEC : le build a modifié le dépôt"; exit 1
fi

# ---- 5. copie atomique du binaire, avec sha256 et commit ------------------------------------------
SRC=$CIBLE/release/$BINAIRE
[ -x "$SRC" ] || { echo "ECHEC : $SRC introuvable après le build"; exit 1; }
cp "$SRC" "$SORTIE/$BINAIRE.part" && mv -f "$SORTIE/$BINAIRE.part" "$SORTIE/$BINAIRE" || exit 4
( cd "$SORTIE" && sha256sum "$BINAIRE" > "$BINAIRE.sha256" && echo "$KEV_GIT_HEAD" > "$BINAIRE.commit" ) || exit 4
echo "$BINAIRE figé : $(cut -c1-16 "$SORTIE/$BINAIRE.sha256")… commit $KEV_GIT_HEAD -> $SORTIE/$BINAIRE"

if [ "$NETTOYER" = 1 ]; then
  ( cd "$DEPOT" && $CARGO clean ${HORS_LIGNE:+$HORS_LIGNE} >/dev/null 2>&1 ) && echo "cargo clean : fait ($CIBLE)"
fi
if [ "$STRICT" = 1 ] && [ "$AVERT" -gt 0 ]; then echo "ÉCHEC (--strict-warnings) : $AVERT avertissement(s), voir $JOURNAL"; exit 3; fi
exit 0
