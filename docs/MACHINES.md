# Machines : configuration par `kvu.toml`

Depuis WP3, le code de `crates/kvu/src` ne contient plus aucun nom d'hôte, chemin ou seuil propre à une machine. Tout vit dans un
fichier de configuration `kvu.toml`, lu au démarrage de chaque sous-commande. La détection matérielle reste dans le code (hôte, modèle
de CPU, GPU visible par sonde `dlopen`, capacité CUDA, nom `gfxNNNN`) ; la configuration dit seulement *quelle machine* c'est et
*quelles règles* s'appliquent.

## 1. Où est le fichier, et quelle valeur gagne

Recherche du fichier, dans cet ordre (le premier qui existe est lu, les suivants sont ignorés) :

1. `KEVINOTRON_CONFIG` : chemin explicite. S'il est posé et que le fichier manque ou est illisible : **erreur, code de sortie 2**
   (jamais de repli silencieux).
2. `./kvu.toml` (répertoire courant).
3. `~/.config/kvu/kvu.toml`.
4. Sinon : **défauts compilés du profil `inconnue` seulement**. Aucune machine n'est alors reconnue.

Priorité d'une valeur : **variable `KEVINOTRON_*` > clé du fichier > défaut du profil `inconnue`**. Une clé absente d'un profil
prend donc la valeur prudente de `inconnue` (garde SSD 20 Go, RAM minimale 8 Go, `MemoryMax` 8G, 4 fils).

Un fichier malformé arrête la commande avec le code **2** et un message `fichier:ligne:colonne : raison`, par exemple
`kvu.toml:3:1 : clé dupliquée dans la table` ou `kvu.toml:1:1 : table inconnue « [profils.cray] »`.

> Installation : placer le `kvu.toml` du dépôt dans `~/.config/kvu/kvu.toml` (ou lancer depuis la racine du dépôt, ou poser
> `KEVINOTRON_CONFIG`). Sans fichier, la machine principale est traitée comme une machine inconnue (garde SSD 20 Go au lieu de 100 Go,
> et, sur le second poste, ni verrou propre ni affinité) : `kvu doctor` l'écrit explicitement (« profil inconnue : aucun kvu.toml
> trouvé »).
>
> Machines de l'auteur (acemagic, gamer) : le `kvu.toml` de la racine est un MODÈLE neutre qui ne reconnaît aucune machine ; leurs
> profils complets sont la fixture `crates/kvu/kvu.toml`. ATTENTION à l'ordre de recherche : depuis la racine du dépôt, le modèle
> `./kvu.toml` est PRIORITAIRE sur `~/.config/kvu/kvu.toml` ; copier la fixture dans `~/.config` ne suffit donc que pour un lancement
> hors de la racine. Depuis la racine, poser explicitement `KEVINOTRON_CONFIG=<dépôt>/crates/kvu/kvu.toml` (ou lancer depuis
> `crates/kvu/`). Les chemins neutralisés de la fixture (`/chemin/vers/projet/...`, `/home/utilisateur/...`, `pc-nvidia-exemple`,
> `192.0.2.x`) sont à adapter à la machine avant usage. Tests et gates qui supposent une machine précise (MACH-1,
> `detection_reelle_sur_cette_machine`) lisent la machine attendue dans `KVU_TEST_MACHINE_ATTENDUE=acemagic|gamer` ; sans elle, ils
> se sautent en l'annonçant (SKIP).

## 2. Format

Sous-ensemble strict de TOML 1.0, analysé par `crates/kvu/src/config/toml_min.rs` (aucune dépendance ajoutée) :

- tables `[profils.<nom>]` ; clés nues ; commentaires `#` ;
- valeurs : chaîne `"..."` (échappements `\" \\ \n \t \r \b \f \uXXXX`) ou `'...'`, entier, flottant, booléen, tableau de chaînes
  (multiligne, virgule finale admise) ;
- rejetés avec ligne et colonne : clé ou table dupliquée, chaîne ou tableau non fermé, type inattendu, tableau mixte, `[[tableaux de
  tables]]`, tables en ligne, clés pointées ou entre guillemets, chaînes multilignes, `inf`/`nan`, dates, entiers hexadécimaux ;
- au niveau du schéma : table inconnue (seuls `[profils.acemagic]`, `[profils.gamer]`, `[profils.inconnue]` et `[profils.location]`
  existent), clé inconnue dans un profil, mauvais type, valeur hors plage ; la clé racine facultative `schema = "kvu-config/1"`.

Chemins : `~/` est le répertoire personnel. Un texte vide signifie « absent » (pas d'affinité, pas de câble…).

### Clés d'un profil

| clé | type | rôle (défaut de `inconnue`) |
|---|---|---|
| `hotes` | tableau de chaînes | noms d'hôte exacts qui reconnaissent la machine (aucun) |
| `cpu` | tableau de chaînes | sous-chaînes du modèle de CPU, sans casse (aucune) |
| `gpu_arch` | tableau de chaînes | architectures de GPU visibles : `gfxNNNN`, `sm_XY` (aucune) |
| `backend` | chaîne | `auto` (premier GPU utilisable), `hip`, `cuda`, `cpu` (`auto`) |
| `verrou` | chaîne | fichier de verrou GPU (`/tmp/kevinotron_gpu.lock`) |
| `attente_verrou_s`, `fenetre_banc_s` | entier | attente du verrou, fenêtre maximale d'un banc (7200, 600) |
| `nice` | entier | priorité des jobs, de -20 à 19 (10) |
| `cpus` | chaîne | affinité imposée, ex. `"12-19"` (aucune) |
| `fils_max` | entier | fils rayon maximum (4) |
| `mem_max` | chaîne | plafond `systemd-run MemoryMax`, ex. `"24G"` ; vide = aucun (`8G`) |
| `ram_min_go`, `ssd_min_go` | nombre | RAM disponible minimale pour lancer un job ; garde SSD après écriture (8, 20) |
| `alerte` | chaîne | fichier d'alerte du moniteur de charge (aucun) |
| `precision` | chaîne | précision conseillée pour un nouvel ensemble : `f32` ou `f64` (`f64`) |
| `store`, `file` | chaîne | racine du store d'ensembles, de la file de jobs (`~/kevinotron_store`, `~/.local/state/kevinotron/file`) |
| `cable` | chaîne | hôte du câble direct vers l'autre machine ; vide = aucun câble, donc aucun `ssh` (aucun) |
| `reference_bench` | chaîne | binaire de référence de `kvu bench` (aucun) |
| `arch_hip`, `arch_cuda` | chaîne | force l'architecture de compilation ; vide = lue sur le périphérique (aucune) |

## 3. Profils fournis (`kvu.toml` à la racine du dépôt)

- `acemagic` : machine principale (APU AMD, HIP). Verrou `/tmp/kevinotron_gpu.lock`, `nice` 10, 8 fils, `MemoryMax` 24G, RAM minimale
  25 Go, garde SSD 100 Go, moniteur de charge, précision conseillée `f32`, câble `gamer-eth`.
- `gamer` : second poste (GPU NVIDIA, CUDA). Verrou `/tmp/gamer_gpu.lock`, `nice` 5, affinité `12-19`, 8 fils, `MemoryMax` 12G,
  RAM minimale 6 Go, garde SSD 50 Go, aucun câble. Les valeurs de ces deux profils sont celles de `0a7a2de` (gate WP3-EQUIV).
- `inconnue` : politique prudente, aussi utilisée quand aucun fichier n'est trouvé : backend détecté, garde SSD 20 Go, RAM minimale
  8 Go, `MemoryMax` 8G, 4 fils, précision `f64`.
- `location` : machine louée (voir §5). Aucun nom d'hôte, aucun câble, aucun `ssh`, verrou `/tmp/kvu_gpu.lock`, store sous `$HOME`.

Reconnaissance : pour chaque profil qui porte des motifs (`hotes`, `cpu`, `gpu_arch`), la machine correspond si **l'hôte, ou le CPU, ou
un GPU** correspond. Aucun motif ne correspond : profil `inconnue`. `inconnue` et `location` sont des profils *génériques* : ils n'ont
pas de motifs et ne sont jamais reconnus d'eux-mêmes.

## 4. Variables d'environnement

| variable | effet |
|---|---|
| `KEVINOTRON_CONFIG` | chemin du `kvu.toml` (obligatoire s'il est posé) |
| `KEVINOTRON_PROFIL` (ancien nom : `KEVINOTRON_MACHINE`) | force le profil. Accepté : le profil du matériel, ou un profil générique (`inconnue`, `location`). Un profil de machine connue différent du matériel est **refusé** (rc 2) : le verrou GPU suit toujours le GPU réellement utilisé. Avec un profil générique sur un matériel connu, le verrou, l'affinité et le moniteur restent ceux du matériel, et les planchers (SSD, RAM) sont les plus stricts des deux |
| `KEVINOTRON_BACKEND` | `auto|hip|cuda|cpu` ; un backend forcé indisponible donne rc 3, jamais de repli |
| `KEVINOTRON_SSD_MIN_GO`, `KEVINOTRON_RAM_MIN_GO` | seuils (valeur invalide ignorée) |
| `KEVINOTRON_STORE`, `KEVINOTRON_FILE` | racines du store et de la file |
| `KEVINOTRON_CPUS` | restreint l'affinité du profil, ne l'élargit jamais |
| `KEVINOTRON_CABLE` | hôte du câble pour `fetch` |
| `KEVINOTRON_HIP_ARCH` | force l'architecture HIP (prime sur `arch_hip`) |

## 5. Machine louée (Vast.ai) : profil `location`

Principe : une machine louée est un conteneur inconnu, de passage. `KEVINOTRON_PROFIL=location` la configure sans écrire aucun nom
de machine nulle part.

- **Architecture lue sur le périphérique.** CUDA : `sm_<majeur><mineur>` de la capacité de calcul (7.5 → `sm_75`, 8.0 → `sm_80`,
  8.6 → `sm_86`, 8.9 → `sm_89`, 9.0 → `sm_90`, 12.0 → `sm_120`). HIP : le nom `gfxNNNN` de `gcnArchName`. Rien n'est écrit en dur ;
  `arch_cuda` / `arch_hip` (ou `KEVINOTRON_HIP_ARCH`) ne servent qu'à forcer.
- **CUDA ≥ 12.8** (NVRTC) pour une carte Blackwell (`sm_120`) ; une carte Ampère/Hopper s'accommode de versions plus anciennes.
- **NVRTC → CUBIN, pas de PTX.** Le code compile les noyaux avec NVRTC en CUBIN (`nvrtcGetCUBIN`, option `-arch=sm_XY`). Le PTX
  est refusé si le pilote est plus ancien que NVRTC. À vérifier sur chaque machine louée : `kvu doctor --json` doit montrer
  `versions.cuda_pilote` et `versions.nvrtc`, et `kvu doctor --test` doit compiler et exécuter le noyau d'essai (`"ok": true`).
- **Store sous `$HOME`** (`~/kevinotron_store`), file sous `~/.local/state/kevinotron/file`, verrou local `/tmp/kvu_gpu.lock`.
  **Pas de câble ni de `ssh`** : `kvu fetch` et `kvu queue add --machine <autre>` refusent (« aucun câble ») ; les résultats sont
  rapatriés à la main avec leur `sha256`.
- **Pas de `systemd-run` dans un conteneur** : `mem_max` est vide, la limite du conteneur s'applique ; la garde RAM
  (`ram_min_go`) et la garde SSD (`ssd_min_go`) restent actives.
- Construction : chaîne Rust épinglée par `rust-toolchain.toml` (`rustup` l'installe), puis
  `cargo build --release --features cuda` (la bibliothèque `cudarc` charge `libcuda`/`libnvrtc` à l'exécution ; il faut le pilote
  et le paquet NVRTC dans le conteneur). Un binaire reconstruit a son propre `sha256` : les gates d'ancrage (`G-BIT-*`) se rejouent
  sur la machine louée avant toute campagne.
- Matériel : le FP64 est le goulot (gfx1151 : 1/16 du FP32 ; GeForce : ~1/64). Seules les cartes de centre de calcul (A100, H100)
  accélèrent vraiment le chemin FP64 ; une carte grand public n'aide que le chemin F32 SU(2) à loi validée. Aucune graine, clé ni
  donnée privée sur une machine louée.

### Règle « hors la machine principale = classe B »

Tout résultat produit ailleurs que sur la machine principale (le second poste, une machine louée) est de **classe B** :
régénérable statistiquement, jamais à l'octet. Aucun verdict n'en dépend avant d'avoir été rejoué ou recertifié sur la machine
principale (réplication statistique entre machines, bit-identité seulement sur une même machine). Les manifestes de régénération
d'un résultat loué sont donc classés B, jamais A.

## 6. Ajouter une machine

Les profils sont les quatre noms du type `Machine` (le nom d'un profil est le nom de sa variante en minuscules). Pour une machine
louée, utiliser `location` ; pour un nouveau poste permanent, ajouter une variante à `Machine` (`crates/kvu/src/commun.rs`) et un
profil dans `kvu.toml` avec ses motifs de reconnaissance, sans toucher au reste du code.

## 7. WP3-GREP et la casse (réserve 2 de T0-VERIF, lot t1-prelude)

La gate WP3-GREP (GATES v1, exclusions v2) cherche `acemagic|gamer|10\.42\.0|/chemin/vers/projet|remondiere` dans
`crates/kvu/src` avec un motif **sensible à la casse**. Elle est verte (`code=0`) : aucun hôte, chemin, câble ni
seuil n'est écrit en dur hors de `kvu.toml`. Ce vert ne veut pas dire « aucun nom de machine dans le code » :

- l'**identité des profils** reste câblée dans le type `Machine` (`commun.rs` : variantes `Acemagic`, `Gamer`,
  `Inconnue`, `Location` ; noms de profil fabriqués à la compilation, en minuscules, par `nom_de_variante!`) ;
- `regen/classer.rs` compare la machine de production à `NOM_PROFIL_ACEMAGIC` pour la classe A, et
  `regen/adaptateurs.rs`, `cli/catalogue.rs`, `machine/{detection,politique}.rs` nomment les profils par ces variantes.

Décision (pilote de la tranche 1, GATES v3) : **documenté, pas corrigé** dans la tranche 1. Renommer ou
généraliser le type `Machine` toucherait les manifestes, le catalogue des gates et des tests figés ; c'est une
tâche de la publication (renommage public, DECISIONS D1). Pour que le résidu reste visible, `wp3_grep.py` imprime
en plus une ligne `INFO WP3-GREP-CASSE code=N fichier:n …` (même recherche, insensible à la casse, mêmes
exclusions). Elle n'est jamais un verdict ; sa valeur au commit du lot t1-prelude est `code=24`.

## 8. Verrou GPU partagé sous `gpu_partage` (lot WP21, tranche 4)

`gpu_partage` (enveloppe des petits jobs GPU, `/usr/local/bin/gpu_partage` v2.1) attend une place, prend le verrou GPU de la machine en mode
**partagé** (`flock -s` éphémère sur un descripteur hérité : le PID inscrit dans `/proc/locks` meurt aussitôt, l'ancêtre n'est donc pas reconnu),
puis exporte `GPU_PARTAGE=1` et `GPU_PARTAGE_VERROU_TENU=<chemin du verrou>` avant de lancer la commande. Jusqu'au lot WP21, `kvu` demandait alors
son verrou en exclusif sur le même fichier et attendait derrière son propre parent (incident n° 1 de WP9, contourné par `flock -s` ancêtre).

Règle de `VerrouGpu::acquerir` (spécification du lot WP21) :

1. `KEVINOTRON_VERROU=ignorer` ⇒ `ignore` (inchangé, prioritaire).
2. **Mode partagé** si et seulement si `GPU_PARTAGE` vaut `1` (espaces ignorés) **et** `GPU_PARTAGE_VERROU_TENU` désigne le **même fichier** que le
   verrou de `kvu` (même périphérique et même inode). Alors : si un **ancêtre vivant** tient déjà le verrou, quel que soit son mode (contournement
   des campagnes `gpu_partage -- flock -s <verrou> kvu …`, ou fenêtre exclusive), `ancetre` comme avant, sans seconde prise ; sinon `kvu` prend
   son verrou en `LOCK_SH` (flock BSD) : accordé tout de suite à côté du verrou partagé de `gpu_partage` et des autres jobs partagés ; une demande
   exclusive (`flock -w …`) attend la fin de tous. Le manifeste consigne alors `verrou: partage`. Un détenteur exclusif qui n'est pas un ancêtre
   fait attendre jusqu'au délai (code 6), comme avant.
3. Sinon, comportement **inchangé** : `LOCK_EX`, ou `ancetre` si un ancêtre tient le verrou, délai ⇒ code 6. Si `GPU_PARTAGE=1` est posé mais que le
   verrou tenu par `gpu_partage` n'est pas celui de `kvu` (variable absente, fichier absent, autre fichier), une ligne d'avertissement est écrite
   sur la sortie d'erreur et la prise est **exclusive**.

États consignés dans les manifestes : `verrou: pris_ici | ancetre | partage | ignore` (et `non_determine` sans GPU). En mode partagé, la garde
est tenue par CE processus (rendue à sa fin, même si `gpu_partage` meurt avant lui).

**Bancs** : `kvu bench …` exige l'exclusivité. Sous `gpu_partage` (condition 2 vraie), il refuse aussitôt (code 6, message « exclusivité requise »)
au lieu d'attendre un verrou exclusif que son propre parent empêche. Les gates à l'octet (G-BIT) restent lancées en fenêtre exclusive
(`flock -w 7200 <verrou>`) ; leurs sorties ne dépendent pas du mode de verrou.

**Hors du périmètre** : la file de jobs (`kvu queue`) enveloppe toujours ses jobs GPU dans un `flock` exclusif (FILE-1 inchangé) ; elle n'est pas
faite pour tourner sous `gpu_partage`. Un `kvu` lancé à distance sans `kvu.toml` prend le verrou du profil `inconnue` (`/tmp/kevinotron_gpu.lock`),
pas celui du poste : sous `gpu_partage` du gamer (`/tmp/gamer_gpu.lock`), il n'y a alors pas correspondance et la prise reste exclusive sur un
autre fichier (comportement d'avant, défaut connu de WP-DIR) ; poser `KEVINOTRON_CONFIG` vers le `kvu.toml` du dépôt l'évite.
