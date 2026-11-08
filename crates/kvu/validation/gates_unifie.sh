#!/usr/bin/env bash
# gates_unifie.sh — orchestrateur de la suite de validation du kevinotron unifié (tâche T9).
#
# Lancé par `kevinotron valide` (cli/valide.rs) ; utilisable seul : KVU_MODE=complet bash gates_unifie.sh
#
# Rôle : exécuter, sur chaque machine demandée, les scripts de gates
#   - validation/<tâche>/gates_*.sh  (T1..T8, une ligne PASS|FAIL|SKIP <nom> [détail] par gate),
#   - validation/historique/G-H01..G-H13 (les 13 gates de gates_gpu1.sh, ARCHITECTURE_UNIFIEE.md §6.6),
# puis agréger. Le gamer est atteint par `ssh gamer-eth` (câble 2,5 Gbit/s) : l'arbre est copié par rsync,
# puis CE MÊME script y tourne en local (KVU_DISTANT=1).
#
# Contrat d'environnement (voir cli/valide.rs) :
#   KVU_MODE=rapide|complet  KVU_GATE=<motif>  KVU_MACHINES=acemagic,gamer  KVU_BIN=<kevinotron>
#   KVU_RACINE=<racine du dépôt>  KVU_SANS_GPU=1 (mode rapide : aucun GPU, aucun gamer)
#   KVU_GIT_HEAD (commit attendu ; sinon `git rev-parse HEAD`)
#   KVU_BIN_GAMER (binaire sur le gamer, défaut ~/kvu_unif/bin/${KVU_LOT:-wp1}/kvu, où build_auto.sh le dépose)
#   KVU_TIMEOUT_S (par script sous verrou, hors attente du verrou, défaut 1800) ; KVU_TIMEOUT_AUTONOME_S (scripts autonomes, 86400)
# Les scripts reçoivent en plus KVU_MACHINE (leur machine), KVU_SCRATCH (répertoire de travail, SUR LE SSD :
# KVU_SCRATCH_BASE, défaut <U>/scratch/<lot> (U = dossier parent du dépôt ; gamer : ~/kvu_unif/scratch/<lot>) ; supprimé à la fin sauf KVU_GARDER_SCRATCH=1).
#
# Sortie (stdout) : une ligne par gate, séparateur tabulation
#   GATE<TAB>PASS|FAIL|SKIP|INFO|FAIL-DOCUMENTE<TAB>machine<TAB>nom<TAB>détail
# puis  FIN<TAB>fail=N<TAB>skip=M<TAB>info=I<TAB>fail_documente=D . rc = nombre de FAIL (plafonné à 255). Un SKIP n'est pas vert.
# INFO (OBS-4-L8) et FAIL-DOCUMENTE (G-H11) sont les deux statuts de l'arbitrage D3 (GATES.md v1, lot WP2) : ils ne comptent
# ni dans le rc ni comme PASS, et le rapport les liste toujours.
# Le bruit des scripts va sur stderr.
#
# VERROU GPU, par fenêtre (un script = une fenêtre, ≤ 10 min) :
#   - un script qui déclare « prend lui-même le verrou » (et tout script de historique/) le gère seul, par fenêtre ;
#   - tout autre script (T1..T8 : « à lancer SOUS le verrou ») est lancé ICI sous `flock -w 7200 <verrou de la machine>`,
#     le verrou étant rendu à la fin du script ; en mode rapide (KVU_SANS_GPU=1) aucun verrou n'est pris.
#   Verrou : /tmp/kevinotron_gpu.lock (acemagic), /tmp/gamer_gpu.lock (gamer) ; surcharge KVU_VERROU ; attente
#   KVU_VERROU_ATTENTE_S (défaut 7200).
# Si cette suite est elle-même lancée sous `flock` (un ancêtre tient le verrou), un faux `flock` placé en tête du PATH
# laisse passer toutes les prises internes : pas d'interblocage.
# CARGO_TARGET_DIR : cible de développement hors dépôt (scripts qui compilent) ; jamais celle des productions.
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACINE="${KVU_RACINE:-$(cd "$ICI/../../.." && pwd)}"
MODE="${KVU_MODE:-complet}"
MOTIF="${KVU_GATE:-}"
SANS_GPU="${KVU_SANS_GPU:-0}"
TIMEOUT_S="${KVU_TIMEOUT_S:-1800}"
ATTENTE_VERROU_S="${KVU_VERROU_ATTENTE_S:-7200}"
TIMEOUT_AUTONOME_S="${KVU_TIMEOUT_AUTONOME_S:-86400}"   # scripts qui gèrent seuls leur verrou (attentes comprises)
PASS=0; FAIL=0; SKIP=0; INFO=0; FAILDOC=0   # INFO et FAIL-DOCUMENTE : arbitrages D3 (GATES v1), non bloquants, toujours listés
# Commit attendu dans le manifeste des runs (G-H01) ; sur le gamer l'arbre est copié SANS .git : il est transmis.
KVU_GIT_HEAD="${KVU_GIT_HEAD:-$(git -C "$RACINE" rev-parse HEAD 2>/dev/null || true)}"
export KVU_GIT_HEAD
# Empreinte de la recette kvu-source/1 de l'arbre propre : transmise aux copies sans .git (gamer) pour que build.rs
# vérifie la copie (git_head nu si identique, « +sale » sinon, « +non_verifie » sans elle).
KEV_SOURCE_HASH_ATTENDU="${KEV_SOURCE_HASH_ATTENDU:-$(python3 -I "$RACINE/crates/kvu/validation/historique/source_hash_kvu.py" "$RACINE" 2>/dev/null | head -1)}"
export KEV_SOURCE_HASH_ATTENDU

# ---------------------------------------------------------------------------------------------
# Machine locale
# ---------------------------------------------------------------------------------------------
machine_locale() {
  if [ -n "${KVU_MACHINE:-}" ]; then echo "$KVU_MACHINE"; return; fi
  if grep -q "AI MAX+ 395" /proc/cpuinfo 2>/dev/null; then echo acemagic
  elif grep -q "i5-14600KF" /proc/cpuinfo 2>/dev/null; then echo gamer
  else echo inconnue; fi
}
LOCALE="$(machine_locale)"
MACHINES="${KVU_MACHINES:-$LOCALE}"
if [ -z "${CARGO_TARGET_DIR:-}" ]; then
  if [ "$LOCALE" = gamer ]; then CARGO_TARGET_DIR="$HOME/kvu_unif/cibles/${KVU_LOT:-wp1}"
  else CARGO_TARGET_DIR="$(dirname "$RACINE")/cibles/${KVU_LOT:-wp1}"; fi
fi
export CARGO_TARGET_DIR
verrou_machine() { # machine
  if [ -n "${KVU_VERROU:-}" ]; then echo "$KVU_VERROU"; return; fi
  case "$1" in gamer) echo /tmp/gamer_gpu.lock;; *) echo /tmp/kevinotron_gpu.lock;; esac
}

# ---------------------------------------------------------------------------------------------
# Émission normalisée
# ---------------------------------------------------------------------------------------------
emettre() { # statut machine nom détail
  printf 'GATE\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4"
  case "$1" in PASS) PASS=$((PASS + 1));; FAIL) FAIL=$((FAIL + 1));; SKIP) SKIP=$((SKIP + 1));;
    INFO) INFO=$((INFO + 1));; FAIL-DOCUMENTE) FAILDOC=$((FAILDOC + 1));; esac
}

# Rend vrai si le nom de gate $1 passe le filtre KVU_GATE (insensible à la casse, motif = sous-chaîne).
passe_filtre() {
  [ -z "$MOTIF" ] && return 0
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    *"$(printf '%s' "$MOTIF" | tr '[:upper:]' '[:lower:]')"*) return 0;;
  esac
  return 1
}

# Convertit la sortie brute d'un script (stdin) en lignes GATE. Accepte :
#   « PASS nom détail », « test x ... PASS nom » (libtest --nocapture), et les lignes GATE déjà normalisées.
normaliser() { # machine
  local machine="$1" ligne rest statut nom detail
  while IFS= read -r ligne; do
    case "$ligne" in
      GATE$'\t'*) # déjà normalisée : on garde la machine du script s'il ne l'a pas fixée
        IFS=$'\t' read -r _ statut mach nom detail <<<"$ligne"
        passe_filtre "$nom" && emettre "$statut" "${mach:-$machine}" "$nom" "$detail" ;;
      *)
        rest="$ligne"
        case "$rest" in "test "*" ... "*) rest="${rest#* ... }";; esac
        case "$rest" in
          REPORT\ *) # gate souple « rapportée » (OBS-4) : PASS, sauf si elle dit elle-même « NON atteint »
            rest="${rest#REPORT }"; nom="${rest%% *}"; detail="${rest#"$nom"}"; detail="${detail# }"; detail="${detail# : }"
            case "$detail" in *[Nn][Oo][Nn]\ [Aa]tteint*|*NON\ ATTEINT*) statut=FAIL;; *) statut=PASS;; esac
            passe_filtre "$nom" && emettre "$statut" "$machine" "$nom" "(rapportée) $detail" ;;
          PASS\ *|FAIL\ *|SKIP\ *|INFO\ *|FAIL-DOCUMENTE\ *|PASS|FAIL|SKIP)
            statut="${rest%% *}"; rest="${rest#"$statut"}"; rest="${rest# }"
            nom="${rest%% *}"; detail="${rest#"$nom"}"; detail="${detail# }"
            [ -z "$nom" ] && nom="(sans nom)"
            passe_filtre "$nom" && emettre "$statut" "$machine" "$nom" "$detail" ;;
          *) printf '%s\n' "$ligne" >&2 ;;
        esac ;;
    esac
  done
}

# ---------------------------------------------------------------------------------------------
# Faux flock (transparent si l'ancêtre tient déjà le verrou)
# ---------------------------------------------------------------------------------------------
installer_faux_flock() {
  FAUX_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kvu_flock.XXXXXX")"
  local vrai; vrai="$(command -v flock || true)"
  [ -z "$vrai" ] && return 0
  cat >"$FAUX_DIR/flock" <<SHIM
#!/usr/bin/env bash
# faux flock : si le verrou visé est tenu par un ANCÊTRE de ce processus, exécute la commande
# directement (un second flock bloquerait jusqu'au délai). Sinon, flock réel.
VRAI="$vrai"
args=("\$@"); lock=""; cmd=()
i=0
while [ \$i -lt \${#args[@]} ]; do
  a="\${args[\$i]}"
  case "\$a" in
    -w|--timeout|-E|--conflict-exit-code) i=\$((i + 2));;
    -c|--command) cmd=(bash -c "\${args[\$((i + 1))]}"); i=\$((i + 2));;
    -n|-x|-s|-u|-o|--nonblock|--exclusive|--shared|--unlock|--close|--no-fork|--verbose) i=\$((i + 1));;
    --) i=\$((i + 1)); [ -z "\$lock" ] && { lock="\${args[\$i]}"; i=\$((i + 1)); }; [ \${#cmd[@]} -eq 0 ] && cmd=("\${args[@]:\$i}"); break;;
    -*) i=\$((i + 1));;
    *) if [ -z "\$lock" ]; then lock="\$a"; i=\$((i + 1)); else [ \${#cmd[@]} -eq 0 ] && cmd=("\${args[@]:\$i}"); break; fi;;
  esac
done
if [ -n "\$lock" ] && [ \${#cmd[@]} -gt 0 ] && [ -e "\$lock" ]; then
  ino=\$(stat -Lc %i "\$lock" 2>/dev/null)
  pids=\$(awk -v ino="\$ino" '\$2=="FLOCK" { n=split(\$6,a,":"); if (a[n]==ino) print \$5 }' /proc/locks 2>/dev/null)
  if [ -n "\$pids" ]; then
    p=\$\$
    while [ -n "\$p" ] && [ "\$p" -gt 1 ]; do
      for q in \$pids; do [ "\$p" = "\$q" ] && exec "\${cmd[@]}"; done
      p=\$(awk '{print \$4}' "/proc/\$p/stat" 2>/dev/null)
    done
  fi
fi
exec "\$VRAI" "\$@"
SHIM
  chmod +x "$FAUX_DIR/flock"
  export PATH="$FAUX_DIR:$PATH"
}
nettoyer() { [ -n "${FAUX_DIR:-}" ] && rm -rf "$FAUX_DIR"; [ -n "${SCRATCH:-}" ] && [ "${KVU_GARDER_SCRATCH:-0}" != 1 ] && rm -rf "$SCRATCH"; }
trap nettoyer EXIT

# ---------------------------------------------------------------------------------------------
# Un script de gates
# ---------------------------------------------------------------------------------------------
prend_son_verrou() { # script : vrai s'il gère seul son verrou (par fenêtre)
  case "$1" in */historique/*) return 0;; esac
  grep -qiE "prend lui-m[eê]me le verrou|prend le verrou lui-m[eê]me|prend elle-m[eê]me le verrou" "$1"
}
lancer_script() { # machine script
  local machine="$1" script="$2" rc nbf_avant="$FAIL" sortie
  sortie="$(mktemp "$SCRATCH/sortie.XXXXXX")"
  local verrou=() delai="$TIMEOUT_S"
  if prend_son_verrou "$script"; then
    delai="$TIMEOUT_AUTONOME_S"   # le délai englobe ses propres attentes de verrou, fenêtre après fenêtre
  elif [ "$SANS_GPU" != 1 ]; then
    verrou=(flock -w "$ATTENTE_VERROU_S" "$(verrou_machine "$machine")")
  fi
  # Particularités par script (T1..T8) : features CUDA du gamer.
  local args=() extra_env=()
  if [ "$machine" = gamer ]; then
    case "$(basename "$(dirname "$script")")" in
      cond) extra_env=(KVU_FEATURES=cuda);;
      obs) args=(--cuda);;
    esac
  fi
  ( cd "$RACINE" && env "${extra_env[@]}" KVU_MACHINE="$machine" KVU_MODE="$MODE" KVU_GATE="$MOTIF" KVU_SANS_GPU="$SANS_GPU" \
      KVU_BIN="${BIN_LOCAL:-}" KVU_RACINE="$RACINE" KVU_SCRATCH="$SCRATCH" \
      "${verrou[@]}" timeout "$delai" bash "$script" "${args[@]}" >"$sortie" 2>"$sortie.err" )
  rc=$?
  cat "$sortie.err" >&2
  normaliser "$machine" <"$sortie"
  if [ "$rc" -eq 124 ]; then
    emettre FAIL "$machine" "$(basename "$script")" "délai de ${delai} s dépassé"
  elif [ "$rc" -ne 0 ] && [ "$FAIL" -eq "$nbf_avant" ]; then
    emettre FAIL "$machine" "$(basename "$script")" "le script s'est terminé avec rc = $rc sans signaler de FAIL"
  fi
  rm -f "$sortie" "$sortie.err"
}

# Le script traite-t-il la gate visée ? (mots du motif dans son nom ou son texte ; sinon on l'écarte.)
concerne() { # script
  [ -z "$MOTIF" ] && return 0
  basename "$1" | grep -qi -- "$MOTIF" && return 0
  grep -qi -- "$MOTIF" "$1"
}

# Ordre : TESTS-CPU (sans GPU), puis les gates historiques (courtes, et leurs défauts se voient tôt), puis celles des tâches.
scripts_de_gates() {
  local f
  for f in "$ICI"/historique/gates_cpu.sh; do [ -f "$f" ] && echo "1 $f"; done
  for f in "$ICI"/historique/G-H[0-9][0-9]*.sh "$ICI"/historique/gates_rt.sh; do [ -f "$f" ] && echo "2 $f"; done
  for f in "$ICI"/*/gates_*.sh; do case "$f" in */historique/*|*/f32/gates_f32.sh) continue;; esac; [ -f "$f" ] && echo "3 $f"; done
  for f in "$ICI"/historique/gates_f32_t9.sh; do [ -f "$f" ] && echo "4 $f"; done
}

# Mode rapide : AUCUN GPU. Les scripts de T1..T8 mélangent gates CPU et GPU et supposent le verrou tenu par l'appelant : on
# ne les lance donc pas sans verrou. Seuls les scripts CPU de historique/ (G-H01, G-H02, gates_cpu = tous les tests
# non ignorés du crate kvu) tournent en mode rapide.
rapide_ok() { case "$(basename "$1")" in G-H01.sh|G-H02.sh|gates_cpu.sh) case "$1" in */historique/*) return 0;; esac;; esac; return 1; }

# Fenêtre de GROUPE : les gates historiques GPU (G-H04..G-H08) gèrent chacune leur verrou ; pour ne pas refaire la queue du
# verrou (production = ~1 h par chaîne) à chaque lancement, l'orchestrateur les exécute TOUTES sous UNE fenêtre
# (`flock <verrou> bash gates_unifie.sh` en mode KVU_GROUPE, quelques minutes en tout, < 10 min). Le processus qui tient le
# verrou est un ANCÊTRE des scripts : leurs `flock` (faux flock) et le VerrouGpu de kevinotron (détection d'ancêtre,
# /proc/locks) passent sans s'interbloquer.
est_groupe_gpu() { case "$1" in */historique/G-H0[4-8]*.sh|*/historique/gates_rt.sh) return 0;; esac; return 1; }

executer_groupe() { # machine script…
  local machine="$1"; shift
  local sortie t0=$SECONDS verrou; verrou="$(verrou_machine "$machine")"
  sortie="$(mktemp "$SCRATCH/groupe.XXXXXX")"
  echo "== [$machine] fenêtre de groupe ($(for x in "$@"; do basename "$x" .sh; done | tr '\n' ' ')) : attente du verrou $verrou" >&2
  if [ "$SANS_GPU" = 1 ]; then
    KVU_GROUPE="$*" KVU_SCRATCH_EXT="$SCRATCH" KVU_MACHINE="$machine" bash "$ICI/gates_unifie.sh" >"$sortie" 2>"$sortie.err"
  else
    flock -w "$ATTENTE_VERROU_S" "$verrou" env KVU_GROUPE="$*" KVU_SCRATCH_EXT="$SCRATCH" KVU_MACHINE="$machine" \
      bash "$ICI/gates_unifie.sh" >"$sortie" 2>"$sortie.err"
  fi
  local rc=$?
  cat "$sortie.err" >&2
  local n_avant="$FAIL"
  normaliser "$machine" <"$sortie"
  if [ "$rc" -ne 0 ] && [ "$FAIL" -eq "$n_avant" ]; then
    emettre FAIL "$machine" "groupe-gpu" "fenêtre de groupe : rc = $rc (verrou non obtenu en ${ATTENTE_VERROU_S} s ?) sans FAIL signalé"
  fi
  echo "== [$machine] fenêtre de groupe rendue après $((SECONDS - t0)) s" >&2
  rm -f "$sortie" "$sortie.err"
}

run_local() { # machine [tout|independants|dependants]
  local machine="$1" filtre="${2:-tout}" s groupe=()
  while IFS= read -r s; do
    concerne "$s" || continue
    if [ "$SANS_GPU" = 1 ] && ! rapide_ok "$s"; then continue; fi
    # stockage (T8) : PROV, SSD et FETCH-1 sont des gates de acemagic (FETCH-1 parle au gamer par le câble)
    if [ "$machine" = gamer ]; then case "$s" in */stockage/*) continue;; esac; fi
    case "$filtre" in
      independants) est_dependante "$s" && continue;;
      dependants) est_dependante "$s" || continue;;
    esac
    if est_groupe_gpu "$s"; then groupe+=("$s"); continue; fi
    if [ "${#groupe[@]}" -gt 0 ]; then executer_groupe "$machine" "${groupe[@]}"; groupe=(); fi
    echo "== [$machine] $(basename "$(dirname "$s")")/$(basename "$s")" >&2
    lancer_script "$machine" "$s"
  done < <(scripts_de_gates | LC_ALL=C sort -k1,1n -k2,2 | cut -d' ' -f2-)
  if [ "${#groupe[@]}" -gt 0 ]; then executer_groupe "$machine" "${groupe[@]}"; fi
}

# Dépendantes : G-H08..G-H13 et F32 rapatrient et analysent les ensembles produits sur le gamer.
est_dependante() { case "$(basename "$1")" in G-H08*|G-H09*|G-H1[0-3]*|gates_f32_t9.sh) return 0;; esac; return 1; }

# Jeu de fixtures de la mesure de fibre (T6) : généré UNE fois ici (venv numba, ~40 s) puis copié sur le gamer, qui n'a pas
# numpy. Les scripts locaux le réutilisent (KVU_FIBRE_FIXTURES).
preparer_fixtures_fibre() {
  local py="${KVU_PYTHON:-/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python}" gen="$ICI/fibre/gen_fixtures.py"
  local dest="${SCRATCH_BASE:-/tmp}/fibre_fixtures"
  [ -f "$gen" ] && [ -x "$py" ] || return 0
  if [ ! -f "$dest/index.json" ]; then
    echo "== génération du jeu de fixtures de la fibre ($dest)" >&2
    systemd-run --scope -p MemoryMax=14G --quiet nice -n 10 "$py" "$gen" --sortie "$dest" >&2 || { rm -rf "$dest"; return 0; }
  fi
  [ -f "$dest/index.json" ] && export KVU_FIBRE_FIXTURES="$dest"
  return 0
}

# Suite distante (gamer par le câble). S'exécute en ARRIÈRE-PLAN (aucune émission ici) : écrit dans
# $SCRATCH/distant.{out,err,rc,echec} ; `distant_collecter` normalise ensuite, dans le shell principal.
distant_lancer() { # machine
  local hote="gamer-eth" dest="kvu_unif/arbres/${KVU_LOT:-wp1}" sortie="$SCRATCH/distant.out"
  : >"$sortie"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$hote" true 2>/dev/null; then
    echo "ssh $hote impossible" >"$SCRATCH/distant.echec"; return
  fi
  ssh -o BatchMode=yes "$hote" "mkdir -p $dest" || { echo "mkdir distant impossible" >"$SCRATCH/distant.echec"; return; }
  if ! rsync -a --delete --exclude target --exclude .git --exclude '__pycache__' \
        -e "ssh -o BatchMode=yes" "$RACINE/" "$hote:$dest/" >&2; then
    echo "copie de l'arbre vers $hote impossible" >"$SCRATCH/distant.echec"; return
  fi
  # Environnement de référence de T5 (EQC-2, concordance grand L) : ses scripts lisent noyau2 et le venv numba à des chemins
  # codés en dur sous /chemin/vers/projet/outils/ym4_noyaux. On y copie noyau2 (164 Mo, lecture seule) et on pointe le venv du gamer
  # (~/ym4_calcul/ym4_noyaux_venv). Retiré à la fin du travail de T9 (voir RAPPORT_T9.md).
  local noy=/chemin/vers/projet/outils/ym4_noyaux
  if [ -d "$noy/noyau2" ]; then
    ssh -o BatchMode=yes "$hote" "mkdir -p $noy && ln -sfn \$HOME/ym4_calcul/ym4_noyaux_venv $noy/venv" \
      && rsync -a --exclude __pycache__ -e "ssh -o BatchMode=yes" "$noy/noyau2/" "$hote:$noy/noyau2/" >&2
  fi
  local fix_env=""
  if [ -n "${KVU_FIBRE_FIXTURES:-}" ] && [ -f "$KVU_FIBRE_FIXTURES/index.json" ]; then
    ssh -o BatchMode=yes "$hote" "mkdir -p kvu_unif/scratch/fibre_fixtures" \
      && rsync -a --delete -e "ssh -o BatchMode=yes" "$KVU_FIBRE_FIXTURES/" "$hote:kvu_unif/scratch/fibre_fixtures/" >&2 \
      && fix_env="KVU_FIBRE_FIXTURES=\$HOME/kvu_unif/scratch/fibre_fixtures"
  fi
  echo "== [gamer] suite distante sur $hote" >&2
  ssh -o BatchMode=yes -o ServerAliveInterval=30 "$hote" \
    "cd $dest && PATH=\$HOME/.cargo/bin:\$PATH $fix_env KVU_MACHINE=gamer KVU_MODE=$MODE KVU_GATE='$MOTIF' KVU_MACHINES=gamer KVU_DISTANT=1 KVU_GIT_HEAD='$KVU_GIT_HEAD' KEV_SOURCE_HASH_ATTENDU='$KEV_SOURCE_HASH_ATTENDU' \
       KVU_BIN=\"${KVU_BIN_GAMER:-\$HOME/kvu_unif/bin/${KVU_LOT:-wp1}/kvu}\" KVU_RACINE=\"\$PWD\" \
       KVU_TIMEOUT_S=$TIMEOUT_S KVU_VERROU_ATTENTE_S=$ATTENTE_VERROU_S KVU_TIMEOUT_AUTONOME_S=$TIMEOUT_AUTONOME_S LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:\${LD_LIBRARY_PATH:-} \
       nice -n 5 taskset -c 12-19 bash crates/kvu/validation/gates_unifie.sh" >"$sortie" 2>"$SCRATCH/distant.err"
  echo $? >"$SCRATCH/distant.rc"
}

distant_collecter() { # machine
  local machine="$1" rc n_avant="$FAIL"
  [ -f "$SCRATCH/distant.err" ] && cat "$SCRATCH/distant.err" >&2
  if [ -f "$SCRATCH/distant.echec" ]; then
    emettre FAIL "$machine" "distant" "$(cat "$SCRATCH/distant.echec")"; return
  fi
  # Les lignes GATE du gamer sont reprises telles quelles ; FIN est ignorée ici.
  normaliser "$machine" <"$SCRATCH/distant.out"
  rc="$(cat "$SCRATCH/distant.rc" 2>/dev/null || echo 255)"
  if [ "$rc" -eq 255 ]; then emettre FAIL "$machine" "ssh" "liaison perdue pendant la suite distante";
  elif [ "$rc" -gt 0 ] && [ "$FAIL" -eq "$n_avant" ]; then
    emettre FAIL "$machine" "distant" "suite distante rc = $rc sans FAIL signalé"; fi
}

# ---------------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------------
# Mode interne GROUPE : exécute les scripts listés (sous le verrou tenu par le processus parent) et émet leurs lignes GATE.
if [ -n "${KVU_GROUPE:-}" ]; then
  SCRATCH="$KVU_SCRATCH_EXT"
  nettoyer() { [ -n "${FAUX_DIR:-}" ] && rm -rf "$FAUX_DIR"; }   # le scratch appartient au processus parent
  installer_faux_flock
  BIN_LOCAL="${KVU_BIN:-}"
  for s in $KVU_GROUPE; do
    echo "== [$LOCALE] $(basename "$(dirname "$s")")/$(basename "$s") (fenêtre de groupe)" >&2
    lancer_script "$KVU_MACHINE" "$s"
  done
  exit 0
fi
# Travail sur le SSD, PAS dans /tmp (RAM sur acemagic, et la garde SSD des runs y refuserait toute écriture).
SCRATCH_BASE="${KVU_SCRATCH_BASE:-$([ "$LOCALE" = gamer ] && echo "$HOME/kvu_unif/scratch/${KVU_LOT:-wp1}" || echo "$(dirname "$RACINE")/scratch/${KVU_LOT:-wp1}")}"
mkdir -p "$SCRATCH_BASE" 2>/dev/null || SCRATCH_BASE="${TMPDIR:-/tmp}"
SCRATCH="$(mktemp -d "$SCRATCH_BASE/kvu_gates.XXXXXX")"
installer_faux_flock
BIN_LOCAL="${KVU_BIN:-}"
# Le gamer tourne EN PARALLÈLE (son verrou GPU est distinct) pendant que acemagic exécute ses gates indépendantes ;
# les gates qui rapatrient et analysent les ensembles du gamer (G-H08..G-H13) passent après sa fin.
GAMER_DISTANT=0; ACEMAGIC=0; INCONNUE=0
case ",$MACHINES," in *,gamer,*) GAMER_DISTANT=1;; esac
case ",$MACHINES," in *,acemagic,*) ACEMAGIC=1;; esac
case ",$MACHINES," in *,inconnue,*) INCONNUE=1;; esac
if [ "${KVU_DISTANT:-0}" = 1 ]; then GAMER_DISTANT=0; fi    # sur le gamer : tout est local
if [ "$GAMER_DISTANT" = 1 ] && [ "$LOCALE" = gamer ]; then GAMER_DISTANT=0; fi
local_nom="$LOCALE"
if [ "${KVU_DISTANT:-0}" = 1 ]; then
  run_local "$LOCALE"
else
  PID_DISTANT=""
  if [ "$GAMER_DISTANT" = 1 ]; then
    if [ "$SANS_GPU" = 1 ]; then
      emettre SKIP gamer "distant" "mode rapide : le gamer n'est pas sollicité"
    else
      preparer_fixtures_fibre
      distant_lancer gamer &
      PID_DISTANT=$!
    fi
  elif [ "$LOCALE" = gamer ] && case ",$MACHINES," in *,gamer,*) true;; *) false;; esac; then
    run_local gamer
  fi
  if [ "$ACEMAGIC" = 1 ]; then
    if [ "$LOCALE" = acemagic ]; then run_local acemagic independants
    else emettre FAIL acemagic "machine" "acemagic n'est pas atteignable depuis $LOCALE (suite à lancer sur acemagic)"; fi
  fi
  [ "$INCONNUE" = 1 ] && run_local "$LOCALE"
  if [ -n "$PID_DISTANT" ]; then wait "$PID_DISTANT"; distant_collecter gamer; fi
  [ "$ACEMAGIC" = 1 ] && [ "$LOCALE" = acemagic ] && run_local acemagic dependants
fi
printf 'FIN\tfail=%s\tskip=%s\tinfo=%s\tfail_documente=%s\n' "$FAIL" "$SKIP" "$INFO" "$FAILDOC"
[ "$FAIL" -gt 255 ] && FAIL=255
exit "$FAIL"
