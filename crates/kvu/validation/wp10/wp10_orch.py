# SPDX-License-Identifier: Apache-2.0
"""Orchestrateur de la campagne WP10 (PREREG v1 §7) : deux files, une par GPU, tirent dans la même liste de tâches.

Une tâche = une chaîne (`kvu su2 gen`, K = 1). Elle est exécutée par FENÊTRES de `fenetre_s` secondes (≤ 10 min sous verrou GPU, GATES §0.7) :
la première fenêtre lance `su2 gen … --fenetre-s`, les suivantes `su2 gen --reprendre DIR --fenetre-s` ; un run s'arrête proprement à l'échéance en
écrivant un point de reprise, de sorte que le résultat est identique à l'octet à celui d'un run sans coupure (gate WP10-REPRISE). Entre deux fenêtres
la file attend `ecart_s` secondes (équité : les autres chantiers qui attendent le verrou passent).
Une tâche terminée sur le gamer est rapatriée par rsync vers l'acemagic (le rapatriement fait foi : G-H13 y recalcule tous les hachages), puis les
post-traitements tournent sur l'acemagic (nice) : deuxième pas du bloc (CPU), G-H13, manifeste kvu-regen/1, élagage des tâches de contrôle.
Tout est reprenable : l'état de chaque tâche est dans `<racine>/etat/<id>.json` ; relancer `run` continue là où l'on s'était arrêté.

  wp10_orch.py init   --racine E --plan plan.json [--gam-racine D] [--fenetre-s 540] …   écrit E/orch_config.json et copie les outils
  wp10_orch.py run    --config E/orch_config.json                                      lance les deux files + post-traitements + bilan
  wp10_orch.py status --config E/orch_config.json                                      tableau d'état
  touch E/STOP                                                                         arrêt propre : chaque file finit sa fenêtre puis s'arrête
"""

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
DEFAUT = {
    "racine": "/chemin/vers/projet/chantiers/kevinotron_unification/ensembles/wp10",
    "gam_racine": "/home/utilisateur/kvu_unif/wp10_ens",                     # chemin SUR le gamer
    "kvu_ace": "/chemin/vers/projet/chantiers/kevinotron_unification/bin/wp10/acemagic/kvu",
    "kvu_gam": "/home/utilisateur/kvu_unif/bin/wp10/kvu",                    # chemin SUR le gamer
    "sha_ace": "", "sha_gam": "",                                # sha256 attendus des binaires (vide = non vérifié)
    "fenetre_s": 540,
    "ecart_s": 15,
    "verrou": {"ace": "/tmp/kevinotron_gpu.lock", "gam": "/tmp/gamer_gpu.lock"},
    "ram_min_go": {"ace": 30, "gam": 24},                        # GATES v2 §0.7
    "disque_min_go": {"ace": 110, "gam": 60},                    # garde de l'outil (ssd_min_go : 100 Go acemagic, 50 Go gamer) + 10 Go ; SSD-1 : ≥ 30 Go libres en permanence
    "mem_scope": {"ace": "12G", "gam": "8G"},
    "gam_cpus": "2-19",
    "fils_hote": 8,                                              # RAYON_NUM_THREADS des fenêtres
    "post_fils": 3,
    "post_paralleles": 2,
    "tentatives": 3,
    "ssh": "gamer",
    "petit_si_L_max": 36,                                        # L ≤ 36 : gpu_partage ; au-delà : verrou exclusif
    "attente_exclusif_s": 600,                                   # verrou exclusif non obtenu en 10 min (jobs partagés continus des autres chantiers) : repli partagé
    "repli_collant_s": 7200,                                     # après un repli, les fenêtres suivantes de la machine restent en mode partagé pendant 2 h
}


# ------------------------------------------------------------------------------------------------------------------
# utilitaires
# ------------------------------------------------------------------------------------------------------------------
def maintenant():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detecter_fautes(lignes_noyau: list[str], texte_log_fenetre: str) -> dict:
    """Détection pure, commune au contrôle en ligne et au re-constat."""
    noyau = re.compile(r"amdgpu.*(page fault|GPU reset|ring .* timeout|VM_CONTEXT|hang)|NVRM: Xid", re.I)
    fenetre = re.compile(r"HSA_STATUS_ERROR|illegal memory access|hipErrorIllegalAddress|MEMORY_FAULT|Xid", re.I)
    return {"fautes_noyau": [l for l in lignes_noyau if noyau.search(l)],
            "fautes_log": [l for l in texte_log_fenetre.splitlines() if fenetre.search(l)]}


class Orch:
    def __init__(self, cfg_chemin):
        self.cfg_chemin = Path(cfg_chemin)
        self.cfg = {**DEFAUT, **json.loads(self.cfg_chemin.read_text(encoding="utf-8"))}
        self.E = Path(self.cfg["racine"])
        self.etat_dir = self.E / "etat"
        self.logs = self.etat_dir / "logs"
        self.outils = self.E / "outils"
        self.plan = json.loads((self.E / "plan.json").read_text(encoding="utf-8"))["taches"]
        self.par_id = {t["id"]: t for t in self.plan}
        self.verrou = threading.Lock()
        self.arret = threading.Event()
        self.pool = None
        self.posts = []
        self.vivantes = {"ace": True, "gam": True}
        self.repli_jusqua = {}
        self.reprise_autorisee = True

    # ---- journaux -------------------------------------------------------------------------------------------
    def log(self, lane, msg):
        ligne = f"{maintenant()} [{lane}] {msg}\n"
        with self.verrou:
            with open(self.E / "orch.log", "a", encoding="utf-8") as f:
                f.write(ligne)
        print(ligne, end="", flush=True)

    def journal(self, msg):
        j = self.cfg.get("journal")
        if j:
            with self.verrou:
                with open(j, "a", encoding="utf-8") as f:
                    f.write(f"{maintenant()} WP10 {msg}\n")

    # ---- état -----------------------------------------------------------------------------------------------
    def chemin_etat(self, ident):
        return self.etat_dir / f"{ident}.json"

    def etat(self, ident):
        p = self.chemin_etat(ident)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"id": ident, "statut": "attente", "machine": None, "fenetres": [], "tentatives": 0, "post": {}}

    def sauver(self, e):
        with self.verrou:
            p = self.chemin_etat(e["id"])
            part = Path(str(p) + ".part")
            part.write_text(json.dumps(e, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(part, p)

    def maj(self, ident, fonction):
        """Lecture-modification-écriture ATOMIQUE de l'état d'une tâche sous le verrou (les fils de post-traitement et les files n'écrasent pas leurs modifications)."""
        with self.verrou:
            p = self.chemin_etat(ident)
            e = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"id": ident, "statut": "attente", "machine": None, "fenetres": [], "tentatives": 0, "post": {}}
            fonction(e)
            part = Path(str(p) + ".part")
            part.write_text(json.dumps(e, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(part, p)
            return e

    # ---- exécution locale / distante --------------------------------------------------------------------------
    def ssh(self, cmd_str, entree=None, timeout=120):
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                               self.cfg["ssh"], cmd_str], input=entree, capture_output=True, text=True, timeout=timeout)

    def sh(self, machine, argv, entree=None, timeout=120):
        """Exécute une commande COURTE sur la machine ; rend (rc, stdout, stderr). Tolère un échec ssh transitoire (3 essais)."""
        if machine == "ace":
            r = subprocess.run(argv, input=entree, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        derniere = None
        for essai in range(3):
            try:
                r = self.ssh(shlex.join(argv), entree=entree, timeout=timeout)
                if r.returncode != 255:
                    return r.returncode, r.stdout, r.stderr
                derniere = (255, r.stdout, r.stderr)
            except subprocess.TimeoutExpired:
                derniere = (124, "", "timeout ssh")
            time.sleep(5 * (essai + 1))
        return derniere

    def aide(self, machine, *args):
        """wp10_remote.py (copie figée dans outils/) ; rend le JSON."""
        if machine == "ace":
            argv = [sys.executable, "-I", str(self.outils / "wp10_remote.py"), *map(str, args)]
        else:
            argv = ["python3", "-I", f"{self.cfg['gam_racine']}/outils/wp10_remote.py", *map(str, args)]
        rc, out, err = self.sh(machine, argv)
        try:
            return json.loads(out.strip().splitlines()[-1]) if out.strip() else {"erreur": f"rc={rc} {err.strip()[:200]}"}
        except ValueError:
            return {"erreur": f"rc={rc} sortie illisible : {out.strip()[:200]} {err.strip()[:200]}"}

    def dossier(self, machine, ident):
        return str(self.E / ident) if machine == "ace" else f"{self.cfg['gam_racine']}/E/{ident}"

    def kvu(self, machine):
        return self.cfg["kvu_ace"] if machine == "ace" else self.cfg["kvu_gam"]

    def toml(self, machine):
        return str(self.outils / "kvu.toml") if machine == "ace" else f"{self.cfg['gam_racine']}/outils/kvu.toml"

    # ---- ressources -------------------------------------------------------------------------------------------
    def attendre_ressources(self, lane, machine, job):
        """RAM (GATES v2 §0.7) et disque (SSD-1) avant chaque fenêtre ; attend jusqu'à 30 min puis rend False."""
        debut = time.time()
        while not self.arret.is_set():
            mem = self.aide(machine, "memoire").get("memavailable_octets")
            disq = self.aide(machine, "libre", self.dossier(machine, "") or "/").get("libre_octets")
            ok_mem = mem is not None and mem / 2**30 >= self.cfg["ram_min_go"][machine]
            ok_dis = disq is not None and disq / 2**30 >= self.cfg["disque_min_go"][machine]
            if ok_mem and ok_dis:
                return True
            if time.time() - debut > 1800:
                self.log(lane, f"ressources insuffisantes depuis 30 min (RAM {mem}, disque {disq}) : abandon de la fenêtre")
                return False
            self.log(lane, f"attente des ressources : MemAvailable {(mem or 0) / 2**30:.1f} Go (min {self.cfg['ram_min_go'][machine]}), "
                           f"disque libre {(disq or 0) / 2**30:.1f} Go (min {self.cfg['disque_min_go'][machine]})")
            time.sleep(60)
        return False

    # ---- commande kvu ------------------------------------------------------------------------------------------
    def commande_kvu(self, job, premiere, machine):
        d = self.dossier(machine, job["id"])
        backend = "hip" if machine == "ace" else "cuda"
        if premiere:
            c = [self.kvu(machine), "su2", "gen", "--backend", backend, "--precision", job["precision"], "--options", job["options"],
                 "--L", str(job["L"]), "--beta", repr(job["beta_run"]), "--graine", str(job["graine"]), "--start", job["start"],
                 "--therm", str(job["therm"]), "--nmeas", str(job["nmeas"]), "--skip", str(job["skip"]), "--nor", str(job["nor"]),
                 "--reprise-every", "0", "--mesures", job["mesures"], "--fenetre-s", str(self.cfg["fenetre_s"]), "--outdir", d]
        else:
            c = [self.kvu(machine), "su2", "gen", "--backend", backend, "--reprendre", d, "--mesures", job["mesures"],
                 "--fenetre-s", str(self.cfg["fenetre_s"])]
        return c

    def petit(self, job):
        if job.get("verrou") in ("partage", "exclusif"):          # surcharge explicite du plan (essais)
            return job["verrou"] == "partage"
        return job["L"] <= self.cfg["petit_si_L_max"]

    def mode_verrou(self, machine, job):
        """'part' (petit job, ou repli collant en cours) ou 'excl' (L = 54 : verrou exclusif, protocole du pilote)."""
        if self.petit(job) or time.time() < self.repli_jusqua.get(machine, 0):
            return "part"
        return "excl"

    def enveloppe_verrou(self, machine, job, limite, mode):
        """Verrou GPU de la machine. Partagé : `gpu_partage` (au plus K petits jobs) précédé d'un `flock -s` extérieur qui reste vivant et ancêtre de kvu
        (sinon /proc/locks consigne le pid mort du `flock` interne de gpu_partage et kvu attend le verrou exclusif jusqu'au délai : piège relevé par
        DESERT_GPU le 25/10, reproduit ici). Exclusif : `flock -w`, fenêtre ≤ 10 min ; code 75 si le verrou n'est pas obtenu en `attente_exclusif_s`
        (le flock de Linux n'est pas équitable envers les exclusifs : des jobs partagés continus l'affament)."""
        v = self.cfg["verrou"][machine]
        if mode == "part":
            return ["flock", "-E", "75", "-s", "-w", "7200", v, "gpu_partage", "-w", "7200", "--", "timeout", str(limite)]
        return ["flock", "-E", "75", "-w", str(int(self.cfg["attente_exclusif_s"])), v, "timeout", str(limite)]

    # ---- une fenêtre -------------------------------------------------------------------------------------------
    def fenetre(self, lane, machine, job, k, premiere):
        """Lance une fenêtre ; rend (rc, t0, t1, mode). Verrou exclusif d'abord (L = 54) ; s'il n'est pas obtenu en `attente_exclusif_s`, la même fenêtre
        repart en mode partagé et la machine y reste `repli_collant_s` (consigné : REPLI-PARTAGE)."""
        mode = self.mode_verrou(machine, job)
        t_att = time.time()
        rc, t0, t1 = self.fenetre_une(lane, machine, job, k, premiere, mode)
        if rc == 75 and mode == "excl" and self.cfg["repli_collant_s"] > 0:
            self.repli_jusqua[machine] = time.time() + self.cfg["repli_collant_s"]
            self.log(lane, f"{job['id']} : REPLI-PARTAGE : verrou exclusif non obtenu en {self.cfg['attente_exclusif_s']} s sur {machine} ; fenêtre {k} en mode partagé "
                           f"(gpu_partage), la machine y reste {self.cfg['repli_collant_s']} s")
            self.journal(f"REPLI-PARTAGE sur {machine} (tâche {job['id']}, fenêtre {k}) : verrou exclusif affamé par des jobs partagés continus")
            mode = "part"
            rc, t0, t1 = self.fenetre_une(lane, machine, job, k, premiere, mode)
        return rc, t_att if rc == 75 else t0, t1, mode

    def fenetre_une(self, lane, machine, job, k, premiere, mode):
        """Une tentative de fenêtre dans le mode de verrou donné ; rend (rc, t0, t1)."""
        limite = int(self.cfg["fenetre_s"] + 50)      # 590 s : fenêtre ≤ 10 min sous verrou, point de reprise compris
        cmd = self.commande_kvu(job, premiere, machine)
        logdir = self.logs / job["id"]
        logdir.mkdir(parents=True, exist_ok=True)
        env_txt = f"KEVINOTRON_CONFIG={shlex.quote(self.toml(machine))} RAYON_NUM_THREADS={self.cfg['fils_hote']}"
        if machine == "ace":
            full = ["calc_borne", "--mem", self.cfg["mem_scope"]["ace"], "--nice", "5", "--", "env",
                    f"KEVINOTRON_CONFIG={self.toml('ace')}", f"RAYON_NUM_THREADS={self.cfg['fils_hote']}",
                    *self.enveloppe_verrou("ace", job, limite, mode), *cmd]
            t_att = time.time()
            with open(logdir / f"fenetre_{k:03d}.log", "w", encoding="utf-8") as f:
                f.write("# " + shlex.join(full) + "\n")
                f.flush()
                p = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT)
            t1 = time.time()
            return p.returncode, t_att, t1
        # --- gamer : script détaché, sondage de l'état
        gdir = f"{self.cfg['gam_racine']}/_fenetres/{job['id']}"
        scr = f"{gdir}/fenetre_{k:03d}"
        corps = "\n".join([
            "#!/bin/bash",
            "echo 1000 > /proc/self/oom_score_adj 2>/dev/null",
            f"cd {shlex.quote(self.cfg['gam_racine'])}",
            f"export {env_txt}",
            f"date +%s.%N > {scr}.t0",
            f"systemd-run --scope --quiet --slice=ym4calc.slice -p MemoryMax={self.cfg['mem_scope']['gam']} -p MemorySwapMax=1G "
            f"nice -n 5 taskset -c {self.cfg['gam_cpus']} {shlex.join(self.enveloppe_verrou('gam', job, limite, mode))} {shlex.join(cmd)} > {scr}.log 2>&1",
            "rc=$?",
            f"date +%s.%N > {scr}.t1",
            f"echo $rc > {scr}.rc", ""])
        rc, _, err = self.sh("gam", ["mkdir", "-p", gdir])
        rc, _, err = self.sh("gam", ["bash", "-c", f"cat > {shlex.quote(scr)}.sh && rm -f {shlex.quote(scr)}.rc"], entree=corps)
        if rc != 0:
            raise RuntimeError(f"écriture du script distant : rc={rc} {err}")
        t_att = time.time()
        self.sh("gam", ["bash", "-c", f"setsid -f bash {shlex.quote(scr)}.sh > /dev/null 2>&1 < /dev/null"])
        injoignable = 0
        while True:
            time.sleep(5)
            rc, out, _ = self.sh("gam", ["bash", "-c", f"cat {shlex.quote(scr)}.rc 2>/dev/null"], timeout=60)
            if rc in (255, 124):
                injoignable += 1
                if injoignable > 240:     # 20 min
                    raise RuntimeError("gamer injoignable depuis 20 min pendant une fenêtre")
                continue
            injoignable = 0
            if out.strip():
                rcw = int(out.strip())
                break
        _, t0s, _ = self.sh("gam", ["bash", "-c", f"cat {shlex.quote(scr)}.t0 {shlex.quote(scr)}.t1"])
        tt = [float(x) for x in t0s.split()] if t0s.split() else [t_att, time.time()]
        subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{self.cfg['ssh']}:{scr}.log", str(logdir / f"fenetre_{k:03d}.log")], capture_output=True)
        return rcw, tt[0], tt[-1]

    def duree_calcul(self, ident, k):
        """Durée de calcul rapportée par kvu lui-même (dernière ligne du journal de la fenêtre : « … , 538.12 s ») ; None si absente."""
        try:
            lignes = (self.logs / ident / f"fenetre_{k:03d}.log").read_text(encoding="utf-8").strip().splitlines()
            m = re.search(r"([0-9]+(?:\.[0-9]+)?) s\s*$", lignes[-1]) if lignes else None
            return float(m.group(1)) if m else None
        except (OSError, ValueError, IndexError):
            return None

    # ---- contrôle des fautes GPU -------------------------------------------------------------------------------
    def fautes_gpu(self, machine, t0, t1):
        """(lisible, lignes) : fautes noyau sur [t0 - 5, t1 + 5] ; lisible = faux si le journal n'a pas pu être lu
        (le contrôle est alors INCONCLUANT, consigné, jamais compté comme « aucune faute »)."""
        try:
            rc, out, _ = self.sh(machine, ["journalctl", "-k", "--since", f"@{t0 - 5:.6f}",
                                          "--until", f"@{t1 + 5:.6f}", "--no-pager", "-q", "-o", "short-iso"], timeout=90)
        except (OSError, subprocess.SubprocessError):
            return False, []
        if rc != 0:
            return False, []
        return True, detecter_fautes(out.splitlines(), "")["fautes_noyau"]

    # ---- une tâche ---------------------------------------------------------------------------------------------
    def executer_tache(self, lane, machine, job):
        ident = job["id"]
        e = self.etat(ident)
        e["machine"] = machine
        e["statut"] = "en_cours"
        d = self.dossier(machine, ident)
        info = self.aide(machine, "info", d)
        if info.get("erreur"):
            self.log(lane, f"{ident} : sonde impossible ({info['erreur']})")
            e["statut"] = "attente"
            e["machine"] = None
            self.sauver(e)
            return False
        premiere = not info.get("existe")
        if premiere:
            rc, out, _ = self.sh("ace", ["ou_calculer", "8"], timeout=60)
            self.log(lane, f"{ident} : ou_calculer → {(out.strip().splitlines() or ['?'])[0]} (placement GPU imposé par le plan ; CPU des post-traitements : acemagic)")
            e["t_debut"] = time.time()
            self.journal(f"tâche {ident} démarrée sur {machine} (graine {job['graine']}, {job['precision']}, L={job['L']}, β={job['beta_run']})")
        else:
            self.log(lane, f"{ident} : reprise sur {machine} ({info.get('reprise')}, complet={info.get('complete')})")
        self.sauver(e)
        k = len(e["fenetres"])
        sans_progres = 0
        dernier_pas = (info.get("reprise") or {}).get("pas_faits", -1)
        while not self.arret.is_set():
            if (self.E / "STOP").exists():
                self.arret.set()
                break
            info = self.aide(machine, "info", d)
            if info.get("complete"):
                break
            if info.get("existe") and not info.get("reprise") and e["fenetres"]:
                # dossier créé mais aucun point de reprise : la première fenêtre a échoué avant son premier point
                self.log(lane, f"{ident} : dossier sans point de reprise après {len(e['fenetres'])} fenêtre(s) : mise de côté et nouveau départ")
                self.sh(machine, ["mv", d, f"{d}.echec{e['tentatives']}"])
                premiere = True
            if not self.attendre_ressources(lane, machine, job):
                e["statut"] = "attente"
                self.sauver(e)
                return False
            k += 1
            if info.get("reprise"):
                self.aide(machine, "sauvegarder", d, k)       # lien dur : coût nul, protège d'un point de reprise corrompu en silence (erratum E4 : les deux machines)
            premiere = premiere or not info.get("existe")
            self.log(lane, f"{ident} : fenêtre {k} ({'départ' if premiere else 'reprise'})")
            try:
                rc, t0, t1, mode = self.fenetre(lane, machine, job, k, premiere)
            except Exception as exc:                       # noqa: BLE001 : toute erreur d'infrastructure est consignée, jamais avalée
                self.log(lane, f"{ident} : fenêtre {k} : erreur d'infrastructure {exc!r}")
                rc, t0, t1, mode = 99, time.time(), time.time(), "?"
            premiere = False
            info = self.aide(machine, "info", d)
            lisible, fautes_noyau = self.fautes_gpu(machine, t0, t1)
            try:
                texte_log = (self.logs / ident / f"fenetre_{k:03d}.log").read_text(encoding="utf-8", errors="replace")
                log_lisible = True
            except OSError:
                texte_log, log_lisible = "", False
                self.log(lane, f"{ident} : INCONCLUANT : journal de fenêtre {k} illisible")
            fautes_log = detecter_fautes([], texte_log)["fautes_log"]
            fautes = fautes_noyau + fautes_log
            if not lisible:
                self.log(lane, f"{ident} : INCONCLUANT : journal noyau illisible sur {machine} (contrôle des fautes GPU non fait pour la fenêtre {k})")
                e["journal_noyau_illisible"] = e.get("journal_noyau_illisible", 0) + 1
            calcul = self.duree_calcul(ident, k)
            w = {"k": k, "rc": rc, "t0": t0, "t1": t1, "duree_s": round(t1 - t0, 2), "calcul_s": calcul, "machine": machine, "verrou": mode,
                 "reprise": info.get("reprise"), "complet": bool(info.get("complete")), "n_cfg": info.get("n_cfg"), "fautes_gpu": fautes, "journal_lisible": lisible,
                 "fautes_noyau": fautes_noyau, "fautes_log": fautes_log, "log_lisible": log_lisible}
            e["fenetres"].append(w)
            self.sauver(e)
            self.log(lane, f"{ident} : fenêtre {k} rc={rc} {w['duree_s']:.0f} s (calcul {calcul if calcul is None else round(calcul)} s), reprise={info.get('reprise')}, complet={info.get('complete')}, configurations={info.get('n_cfg')}")
            if fautes:
                self.log(lane, f"{ident} : ALERTE-IGPU {fautes[:2]} : retour au point de reprise précédent")
                self.journal(f"ALERTE-IGPU sur {machine}, tâche {ident}, fenêtre {k} : {fautes[:1]}")
                e["alertes"] = e.get("alertes", 0) + 1
                rc = 98
                if e["alertes"] > 1:                          # un seul rejeu par tâche (relecture de Sol, point 13) : récidive = échec de la tâche
                    e["statut"] = "echec"
                    self.sauver(e)
                    self.log(lane, f"{ident} : ALERTE-IGPU répétée : tâche en ÉCHEC, sorties suspectes conservées pour examen")
                    self.journal(f"tâche {ident} en ÉCHEC : ALERTE-IGPU répétée sur {machine}")
                    return False
            if rc == 75:
                # attente du verrou ou d'une place épuisée (flock -E 75, gpu_partage) : aucune erreur de calcul, ni point de reprise touché ;
                # on attend à nouveau, sans compter une tentative (erratum E4, fermeture de E3)
                self.log(lane, f"{ident} : fenêtre {k} : verrou ou place non obtenu (code 75) : nouvelle attente, sans tentative comptée")
                time.sleep(5)
                continue
            if rc != 0:
                e["tentatives"] += 1
                self.sauver(e)
                if e["tentatives"] > self.cfg["tentatives"]:
                    self.log(lane, f"{ident} : ÉCHEC après {e['tentatives']} tentatives (rc={rc})")
                    e["statut"] = "echec"
                    self.sauver(e)
                    self.journal(f"tâche {ident} en ÉCHEC sur {machine} (rc={rc}, {e['tentatives']} tentatives) : voir {self.logs / ident}")
                    return False
                if rc == 98:
                    self.aide(machine, "restaurer", d, k)
                if self.aide(machine, "info", d).get("reprise"):
                    self.log(lane, f"{ident} : réparation des sorties en ligne : {self.aide(machine, 'reparer', d)}")
                time.sleep(30)
                continue
            self.aide(machine, "noyer", d)                    # sauvegardes de reprise inutiles après une fenêtre saine
            pas = (info.get("reprise") or {}).get("pas_faits", -1)
            if not info.get("complete"):
                sans_progres = sans_progres + 1 if pas <= dernier_pas else 0
                dernier_pas = max(dernier_pas, pas)
                if sans_progres >= 2:
                    self.log(lane, f"{ident} : aucun progrès sur 2 fenêtres : ÉCHEC")
                    e["statut"] = "echec"
                    self.sauver(e)
                    return False
                time.sleep(self.cfg["ecart_s"])               # équité envers les autres demandeurs du verrou
        if self.arret.is_set() and not self.aide(machine, "info", d).get("complete"):
            e["statut"] = "en_cours"
            self.sauver(e)
            return False
        # --- terminée
        self.aide(machine, "nettoyer", d)
        e["statut"] = "terminee"
        e["t_fin"] = time.time()
        e.pop("reservee_par", None)
        self.sauver(e)
        self.log(lane, f"{ident} : terminée sur {machine} en {len(e['fenetres'])} fenêtres")
        self.journal(f"tâche {ident} terminée sur {machine} ({len(e['fenetres'])} fenêtres, "
                     f"{sum(w['duree_s'] for w in e['fenetres']):.0f} s de fenêtres)")
        self.pool.submit(self.post, lane, job)
        return True

    # ---- rapatriement et post-traitements ----------------------------------------------------------------------
    def rapatrier(self, lane, job):
        ident = job["id"]
        dest = self.E / ident
        src = f"{self.cfg['ssh']}:{self.cfg['gam_racine']}/E/{ident}/"
        for essai in range(5):
            r = subprocess.run(["rsync", "-a", "--checksum", "--exclude", "reprise.*", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                                src, str(dest) + "/"], capture_output=True, text=True)
            if r.returncode == 0:
                return True
            self.log(lane, f"{ident} : rapatriement rc={r.returncode} {r.stderr.strip()[:200]} (essai {essai + 1})")
            time.sleep(20)
        return False

    def etape(self, ident, nom, argv, nice=15):
        logdir = self.logs / ident
        logdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(logdir / f"post_{nom}.log", "w", encoding="utf-8") as f:
            f.write("# " + shlex.join(argv) + "\n")
            f.flush()
            p = subprocess.run(["nice", "-n", str(nice), *argv], stdout=f, stderr=subprocess.STDOUT)
        verdicts = [l for l in (logdir / f"post_{nom}.log").read_text(encoding="utf-8").splitlines() if re.match(r"(PASS|FAIL|INFO) ", l)]
        return p.returncode, round(time.time() - t0, 1), verdicts

    def post(self, lane, job):
        ident = job["id"]
        e = self.etat(ident)
        try:
            if e["machine"] == "gam":
                self.maj(ident, lambda x: x.update(statut="rapatriement"))
                t0 = time.time()
                if not self.rapatrier(lane, job):
                    self.maj(ident, lambda x: x.update(statut="echec_rapatriement"))
                    self.journal(f"tâche {ident} : rapatriement depuis le gamer en ÉCHEC")
                    return
                dt = round(time.time() - t0, 1)
                self.maj(ident, lambda x: x["post"].__setitem__("rapatriement_s", dt))
                self.log(lane, f"{ident} : rapatrié du gamer en {dt} s")
            self.maj(ident, lambda x: x.update(statut="post"))
            d = str(self.E / ident)
            py = [sys.executable, "-I"]
            post = str(self.outils / "wp10_post.py")
            kvu = self.cfg["kvu_ace"]
            etapes = [("bloc2", py + [post, "bloc2", d, "--kvu", kvu, "--fils", str(self.cfg["post_fils"]), "--garder-existants"]),
                      ("h13", py + [post, "h13", d, "--json", f"{d}/h13.json"]),
                      ("manifeste", py + [post, "manifeste", d, "--kvu", kvu, "--depot", str(self.outils / "depot")])]
            if job["garder"] != "tout":
                n = job["garder"]["un_sur"]
                etapes += [("elaguer", py + [post, "elaguer", d, "--un-sur", str(n), "--execute"]), ("h13_apres", py + [post, "h13", d])]
            for nom, argv in etapes:
                rc, dt, verdicts = self.etape(ident, nom, argv)
                self.maj(ident, lambda x: x["post"].__setitem__(nom, {"rc": rc, "s": dt, "verdicts": verdicts[-3:]}))
                self.log(lane, f"{ident} : post {nom} rc={rc} {dt:.0f} s {verdicts[-1] if verdicts else ''}")
                if rc != 0:
                    self.maj(ident, lambda x: x.update(statut="post_echec"))
                    self.journal(f"tâche {ident} : post-traitement {nom} rc={rc} : {verdicts[-1] if verdicts else ''}")
                    return
            self.maj(ident, lambda x: x.update(statut="post_ok"))
            self.journal(f"tâche {ident} : post-traitements verts (bloc2, G-H13, manifeste{', élagage' if job['garder'] != 'tout' else ''})")
        except Exception as exc:                            # noqa: BLE001
            self.maj(ident, lambda x: (x.update(statut="post_echec"), x["post"].__setitem__("erreur", repr(exc))))
            self.log(lane, f"{ident} : post-traitement : erreur {exc!r}")

    # ---- file --------------------------------------------------------------------------------------------------
    def prochaine(self, machine):
        """Première tâche à faire pour `machine` : d'abord une reprise de ses propres tâches, puis la première tâche libre du plan
        (ordre du plan = plus longues d'abord) compatible avec l'affinité."""
        with self.verrou:
            etats = {}
            for t in self.plan:
                p = self.chemin_etat(t["id"])
                etats[t["id"]] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
            for t in self.plan:
                s = etats[t["id"]]
                if s and s["statut"] == "en_cours" and s["machine"] == machine and not s.get("reservee_par"):
                    s["reservee_par"] = machine
                    self._ecrire_sans_verrou(s)
                    return t
            for t in self.plan:
                s = etats[t["id"]]
                if s is not None and s["statut"] not in ("attente",):
                    continue
                autre = "gam" if machine == "ace" else "ace"
                if t["affinite"] not in ("any", {"ace": "acemagic", "gam": "gamer"}[machine]):
                    # tâche réservée à l'autre machine : reprise ici seulement si la file de l'autre machine est définitivement arrêtée
                    if self.vivantes[autre] or self.reprise_autorisee is False:
                        continue
                if s and s.get("reservee_par"):
                    continue
                s = s or {"id": t["id"], "statut": "attente", "machine": None, "fenetres": [], "tentatives": 0, "post": {}}
                s["statut"] = "en_cours"
                s["machine"] = machine
                s["reservee_par"] = machine
                self._ecrire_sans_verrou(s)
                return t
        return None

    def _ecrire_sans_verrou(self, e):
        p = self.chemin_etat(e["id"])
        part = Path(str(p) + ".part")
        part.write_text(json.dumps(e, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(part, p)

    def file(self, machine):
        lane = "ace" if machine == "ace" else "gam"
        premier_echec = 0
        while not self.arret.is_set():
            if (self.E / "STOP").exists():
                self.log(lane, "fichier STOP présent : arrêt de la file")
                break
            job = self.prochaine(machine)
            if job is None:
                self.log(lane, "plus de tâche disponible pour cette machine")
                break
            try:
                ok = self.executer_tache(lane, machine, job)
            except Exception as exc:                        # noqa: BLE001
                ok = False
                self.log(lane, f"{job['id']} : erreur imprévue {exc!r}")
            e = self.maj(job["id"], lambda x: x.pop("reservee_par", None))
            if not ok and e["statut"] == "attente":
                premier_echec += 1
                if premier_echec >= 3:
                    self.log(lane, "trois abandons de ressources consécutifs : la file s'arrête (les tâches restent disponibles pour l'autre machine)")
                    break
        self.vivantes[machine] = False

    # ---- exécution ----------------------------------------------------------------------------------------------
    def verifier_binaires(self):
        for machine in ("ace", "gam"):
            attendu = self.cfg["sha_ace" if machine == "ace" else "sha_gam"]
            if not attendu:
                continue
            rc, out, err = self.sh(machine, ["sha256sum", self.kvu(machine)])
            if rc != 0 or out.split()[0] != attendu:
                raise SystemExit(f"binaire {machine} : sha256 {out.split()[:1]} ≠ attendu {attendu} (rc={rc} {err.strip()})")
            self.log("orch", f"binaire {machine} vérifié : {attendu[:16]}…")

    def run(self):
        self.etat_dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.E / "STOP").unlink(missing_ok=True)
        signal.signal(signal.SIGTERM, lambda *_: self.arret.set())
        signal.signal(signal.SIGINT, lambda *_: self.arret.set())
        self.verifier_binaires()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg["post_paralleles"])
        self.log("orch", f"démarrage : {len(self.plan)} tâches, fenêtres de {self.cfg['fenetre_s']} s, racine {self.E}")
        self.journal(f"orchestrateur démarré ({len(self.plan)} tâches, fenêtres {self.cfg['fenetre_s']} s) : {self.E}")
        # reprise : les tâches « terminee » dont le post-traitement n'a pas abouti sont relancées en post
        for t in self.plan:
            e = self.etat(t["id"])
            if e["statut"] in ("terminee", "rapatriement", "post", "post_echec") and not e.get("post", {}).get("manifeste", {}).get("rc") == 0:
                self.pool.submit(self.post, "post", t)
        for t in self.plan:                                  # réservations d'un `run` précédent interrompu : levées
            e = self.etat(t["id"])
            if e.pop("reservee_par", None) is not None:
                self.sauver(e)
        lanes = [threading.Thread(target=self.file, args=(m,), name=m) for m in ("ace", "gam")]
        for th in lanes:
            th.start()
        for th in lanes:
            th.join()
        self.pool.shutdown(wait=True)
        # redémarrer les tâches restées « attente »/« en_cours » n'est pas fait ici : un `run` ultérieur les reprend
        restantes = [t["id"] for t in self.plan if self.etat(t["id"])["statut"] not in ("post_ok", "echec", "post_echec", "echec_rapatriement")]
        if restantes and not self.arret.is_set() and not (self.E / "STOP").exists():
            self.log("orch", f"tâches non terminées : {restantes} (relancer `run`)")
        if not restantes:
            self.log("orch", "toutes les tâches sont traitées : bilan")
            self.bilan()
        else:
            self.log("orch", f"arrêt avec {len(restantes)} tâche(s) à poursuivre : {restantes}")

    def bilan(self):
        argv = [sys.executable, "-I", str(self.outils / "wp10_bilan.py"), "--config", str(self.cfg_chemin)]
        r = subprocess.run(argv, capture_output=True, text=True)
        self.log("orch", f"bilan rc={r.returncode} {r.stdout.strip()[-300:]} {r.stderr.strip()[-300:]}")
        self.journal(f"bilan écrit (rc={r.returncode}) : prereg/WP10_RAPPORT.md")

    def status(self):
        lignes = []
        for t in self.plan:
            e = self.etat(t["id"])
            f = e.get("fenetres", [])
            rep = f[-1]["reprise"] if f and f[-1].get("reprise") else None
            total = t["therm"] + t["nmeas"] * t["skip"]
            fini = e["statut"] in ("terminee", "post", "post_ok", "post_echec", "rapatriement", "echec_rapatriement")
            pas = total if fini else (rep["pas_faits"] if rep and "pas_faits" in rep else 0)
            temps = sum(w["duree_s"] for w in f)
            lignes.append(f"{t['id']:<12} {e['statut']:<17} {str(e['machine'] or '-'):<4} fenêtres={len(f):<3} pas={pas}/{total} ({100 * pas / total:5.1f} %) "
                          f"fenêtres_s={temps:7.0f} tentatives={e.get('tentatives', 0)}")
        return "\n".join(lignes)


# ------------------------------------------------------------------------------------------------------------------
def commande_init(args):
    E = Path(args.racine)
    (E / "outils" / "depot" / "scripts").mkdir(parents=True, exist_ok=True)
    (E / "etat" / "logs").mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    (E / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    depot = ICI.parents[3]
    for f in ("wp10_stat.py", "wp10_post.py", "wp10_analyse.py", "wp10_remote.py", "wp10_plan.py", "wp10_orch.py", "wp10_bilan.py", "wp10_allocation.py"):
        shutil.copy2(ICI / f, E / "outils" / f)
    shutil.copy2(depot / "kvu.toml", E / "outils" / "kvu.toml")
    shutil.copy2(depot / "scripts" / "regen_valider.py", E / "outils" / "depot" / "scripts" / "regen_valider.py")
    (E / "outils" / "depot" / "schemas").mkdir(exist_ok=True)
    shutil.copy2(depot / "schemas" / "kvu-regen-1.schema.json", E / "outils" / "depot" / "schemas" / "kvu-regen-1.schema.json")
    cfg = {k: v for k, v in {"racine": str(E), "gam_racine": args.gam_racine, "fenetre_s": args.fenetre_s, "journal": args.journal,
                             "kvu_ace": args.kvu_ace, "kvu_gam": args.kvu_gam, "sha_ace": args.sha_ace, "sha_gam": args.sha_gam,
                             "ecart_s": args.ecart_s, "rapport": args.rapport, "attente_exclusif_s": args.attente_exclusif_s,
                             "repli_collant_s": args.repli_collant_s}.items() if v not in (None, "")}
    chemin = E / "orch_config.json"
    chemin.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # outils côté gamer
    g = args.gam_racine
    ssh = ["ssh", "-o", "BatchMode=yes", "gamer"]
    subprocess.run([*ssh, f"mkdir -p {shlex.quote(g)}/outils {shlex.quote(g)}/E {shlex.quote(g)}/_fenetres"], check=True)
    subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(E / "outils" / "wp10_remote.py"), str(E / "outils" / "kvu.toml"),
                    f"gamer:{g}/outils/"], check=True)
    print(f"initialisé : {chemin}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    i = s.add_parser("init")
    i.add_argument("--racine", required=True)
    i.add_argument("--plan", required=True)
    i.add_argument("--gam-racine", default=DEFAUT["gam_racine"])
    i.add_argument("--fenetre-s", type=float, default=DEFAUT["fenetre_s"])
    i.add_argument("--ecart-s", type=float, default=DEFAUT["ecart_s"])
    i.add_argument("--attente-exclusif-s", type=float, default=DEFAUT["attente_exclusif_s"])
    i.add_argument("--repli-collant-s", type=float, default=DEFAUT["repli_collant_s"])
    i.add_argument("--journal", default="")
    i.add_argument("--rapport", default="")
    i.add_argument("--kvu-ace", default=DEFAUT["kvu_ace"])
    i.add_argument("--kvu-gam", default=DEFAUT["kvu_gam"])
    i.add_argument("--sha-ace", default="")
    i.add_argument("--sha-gam", default="")
    for nom in ("run", "status"):
        r = s.add_parser(nom)
        r.add_argument("--config", required=True)
    a = p.parse_args(argv)
    if a.cmd == "init":
        return commande_init(a)
    o = Orch(a.config)
    if a.cmd == "status":
        print(o.status())
        return 0
    o.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
