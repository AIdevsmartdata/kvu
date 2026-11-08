//! CLI topologique WP19 : mesures incrémentales, gates instrumentales et certificat fidèle.

use super::topo::{
    self,
    gpu::{Cible, EspaceTopo, ModuleTopo},
    ParamsTopo, RecordTopo, IDX_T3,
};
use super::topo_gates::{self, LigneGate, Verdict};
use crate::commun::{BackendKind, ChampHote, Erreur, Resultat};
use crate::heritage::{
    manifest::sha256_fichier,
    su2_hb::{qdag, qnorm, Q},
};
use crate::machine::{
    admission::{admettre, octets_champ, BesoinMemoire},
    choisir_backend, detecter,
    detection::arch_memoire_unifiee,
    politique, GardeVerrou, VerrouGpu,
};
use crate::stockage::{npy, GardeSsd, Manifeste};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::json;
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn requis(nom: &'static str) -> Arg {
    Arg::new(nom).long(nom).required(true)
}

fn defaut(nom: &'static str, valeur: &'static str) -> Arg {
    Arg::new(nom).long(nom).default_value(valeur)
}

fn backend(cpu: bool) -> Arg {
    let choix = if cpu {
        vec!["auto", "hip", "cuda", "cpu"]
    } else {
        vec!["auto", "hip", "cuda"]
    };
    defaut("backend", "auto").value_parser(choix)
}

/// Enregistrement séparé pour ne pas modifier les commandes historiques du module obs.
pub fn commandes() -> Vec<Command> {
    vec![
        Command::new("topo")
            .about("Mesures topologiques d'une chaîne SU(2), JSONL et provenance")
            .arg(requis("ensemble").value_name("DIR"))
            .arg(requis("out").value_name("DIR"))
            .arg(requis("ens").value_name("NOM"))
            .arg(requis("chaine"))
            .arg(requis("backend-gen").value_parser(["hip", "cuda", "cpu"]))
            .arg(backend(true))
            .arg(defaut("flot-complet-chaque", "1"))
            .arg(defaut("dt", "0.01"))
            .arg(defaut("balayages", "20"))
            .arg(defaut("pas-max", "5000"))
            .arg(
                Arg::new("reprendre")
                    .long("reprendre")
                    .action(ArgAction::SetTrue),
            )
            .arg(
                Arg::new("sans-flot")
                    .long("sans-flot")
                    .help("cooling seul, aucun flot (avenant v5.WP19)")
                    .action(ArgAction::SetTrue),
            )
            .arg(
                Arg::new("max-cfg")
                    .long("max-cfg")
                    .value_name("N")
                    .help("mesurer au plus N configurations nouvelles puis s'arrêter proprement (reprise par --reprendre)"),
            ),
        Command::new("topo-gates")
            .about("Gates et analyses topologiques (rc 1 si FAIL, SKIP accepté)")
            .subcommand_required(true)
            .flatten_help(true)
            .subcommand(
                Command::new("analyse")
                    .about("Analyse les records JSONL dans l'ordre donné")
                    .arg(requis("ens").value_parser(["q", "obs4", "lit", "lit_mut"]))
                    .arg(
                        requis("jsonl")
                            .value_delimiter(',')
                            .action(ArgAction::Append),
                    )
                    .arg(Arg::new("lignes").long("lignes")),
            )
            .subcommand(
                Command::new("analyse-v5")
                    .about("Analyse v5.WP19 (TOPO-Q-V5, TOPO-LIT-V5 et mutants) des records JSONL")
                    .arg(requis("ens").value_parser(["q5", "lit5", "lit5_mut"]))
                    .arg(
                        requis("jsonl")
                            .value_delimiter(',')
                            .action(ArgAction::Append),
                    )
                    .arg(Arg::new("lignes").long("lignes")),
            )
            .subcommand(
                Command::new("inst")
                    .about("TOPO-INST : charges gelées, jauge et réflexion")
                    .arg(backend(true)),
            )
            .subcommand(
                Command::new("cool-id")
                    .about("TOPO-COOL-ID et mutant sans dagger")
                    .arg(backend(true)),
            )
            .subcommand(
                Command::new("gpu")
                    .about("TOPO-GPU : comparaison CPU/GPU sur trois champs")
                    .arg(requis("cfg").value_delimiter(',').action(ArgAction::Append))
                    .arg(backend(false)),
            )
            .subcommand(
                Command::new("certificat")
                    .about("Recopie les verdicts existants dans un certificat")
                    .arg(
                        requis("lignes")
                            .value_delimiter(',')
                            .action(ArgAction::Append),
                    )
                    .arg(requis("out"))
                    .arg(Arg::new("cmd").long("cmd").action(ArgAction::Append)),
            ),
    ]
}

fn arg<T: std::str::FromStr>(m: &ArgMatches, nom: &str) -> Resultat<T> {
    let s = m
        .get_one::<String>(nom)
        .ok_or_else(|| Erreur::Config(format!("--{nom} manquant")))?;
    s.parse()
        .map_err(|_| Erreur::Config(format!("--{nom} : valeur « {s} » invalide")))
}

fn chemins(m: &ArgMatches, nom: &str) -> Vec<PathBuf> {
    m.get_many::<String>(nom)
        .into_iter()
        .flatten()
        .map(PathBuf::from)
        .collect()
}

fn io(p: &Path, e: std::io::Error) -> Erreur {
    Erreur::io(p.display().to_string(), e)
}

fn sha(p: &Path) -> Resultat<String> {
    sha256_fichier(p).map_err(|e| io(p, e))
}

fn backend_voulu(demande: &str, l: usize) -> Resultat<BackendKind> {
    match demande {
        "auto" if l < 16 => Ok(BackendKind::Cpu),
        "auto" => choisir_backend(&detecter()),
        autre => autre.parse(),
    }
}

// Le verrou est acquis avant l'ouverture de l'appareil, exactement comme obs flot.
// Son garde reste vivant jusqu'à la destruction du module et de ses espaces.
struct Calcul {
    module: Option<ModuleTopo>,
    _verrou: Option<GardeVerrou>,
}

impl Calcul {
    fn ouvrir(b: BackendKind) -> Resultat<Self> {
        if b == BackendKind::Cpu {
            return Ok(Self {
                module: None,
                _verrou: None,
            });
        }
        let pol = politique(&crate::machine::detection::detecter_rapide());
        let verrou = VerrouGpu::nouveau(&pol.verrou_gpu)
            .acquerir(Duration::from_secs(pol.attente_verrou_s))?;
        let app = crate::gpu::ouvrir(b, 0)?;
        let module = ModuleTopo::nouveau(&app)?;
        Ok(Self {
            module: Some(module),
            _verrou: Some(verrou),
        })
    }

    fn charges(&self, champ: &ChampHote) -> Resultat<topo::Charges> {
        match &self.module {
            Some(module) => {
                let e = module.espace(champ.l)?;
                module.charger(&e, champ)?;
                module.charges(&e, Cible::Source)
            }
            None => topo::charges(champ),
        }
    }
}

fn admission(b: BackendKind, l: usize, reference_cpu: bool) -> Resultat<()> {
    let champ = octets_champ(l, 8);
    let v = (l as u64).saturating_pow(4);
    // EspaceTopo : 3 champs 16V + 3 tableaux RK3 12V + réductions (3 ceil(V/256)+3).
    let espace = 84u64
        .saturating_mul(v)
        .saturating_add(3 * v.div_ceil(256) + 3)
        .saturating_mul(8);
    let gpu = b != BackendKind::Cpu;
    let unifie = b == BackendKind::Hip
        && detecter()
            .gpus
            .iter()
            .any(|g| g.backend == b && arch_memoire_unifiee(&g.arch));
    admettre(
        &format!("obs topo L={l} ({})", b.nom()),
        BesoinMemoire {
            // CPU : copies, étapes RK3, charges, géométrie et décodage ; marge conservatrice.
            hote: champ.saturating_mul(if gpu && !reference_cpu { 2 } else { 12 }),
            gpu: if gpu { espace } else { 0 },
        },
        unifie,
    )
}

fn configurations(dir: &Path) -> Resultat<Vec<(usize, PathBuf)>> {
    let mut fichiers = Vec::new();
    let mut dossiers = vec![dir.to_path_buf()];
    while let Some(d) = dossiers.pop() {
        for entree in std::fs::read_dir(&d).map_err(|e| io(&d, e))? {
            let entree = entree.map_err(|e| io(&d, e))?;
            let p = entree.path();
            let nom = entree.file_name();
            let nom = nom.to_string_lossy();
            if d == dir
                && p.is_dir()
                && nom.len() == 4
                && nom.starts_with('c')
                && nom[1..].bytes().all(|c| c.is_ascii_digit())
            {
                dossiers.push(p);
            } else if p.is_file() {
                let stem = nom
                    .strip_suffix(".npy.zst")
                    .or_else(|| nom.strip_suffix(".npy"));
                if let Some(numero) = stem.and_then(|s| s.strip_prefix("cfg_")) {
                    if numero.len() == 6 && numero.bytes().all(|c| c.is_ascii_digit()) {
                        fichiers.push((numero.parse::<usize>().expect("six chiffres"), p));
                    }
                }
            }
        }
    }
    fichiers.sort_by_key(|(cfg, _)| *cfg);
    if fichiers.is_empty() {
        return Err(Erreur::Config(format!(
            "aucune configuration cfg_NNNNNN.npy[.zst] dans {}",
            dir.display()
        )));
    }
    if fichiers.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(Erreur::Config(
            "ensemble : numéros cfg dupliqués (une seule chaîne requise)".into(),
        ));
    }
    Ok(fichiers)
}

struct Reprise {
    cfgs: BTreeSet<usize>,
    longueur: u64,
    nouvelle_ligne: bool,
}

/// Une queue incomplète après interruption est ignorée, puis tronquée seulement après validation.
fn lire_reprise(p: &Path, ens: &str, chaine: usize, backend_gen: &str) -> Resultat<Reprise> {
    let mut r = Reprise {
        cfgs: BTreeSet::new(),
        longueur: 0,
        nouvelle_ligne: false,
    };
    let mut lecteur = BufReader::new(File::open(p).map_err(|e| io(p, e))?);
    let mut texte = Vec::new();
    let mut numero = 0;
    loop {
        texte.clear();
        let n = lecteur
            .read_until(b'\n', &mut texte)
            .map_err(|e| io(p, e))?;
        if n == 0 {
            break;
        }
        numero += 1;
        let record: RecordTopo = match serde_json::from_slice(&texte) {
            Ok(record) => record,
            Err(_) if !texte.ends_with(b"\n") => break,
            Err(e) => {
                return Err(Erreur::Format(format!(
                    "{}: ligne {numero}: {e}",
                    p.display()
                )))
            }
        };
        if record.ens != ens || record.chaine != chaine || record.backend_gen != backend_gen {
            return Err(Erreur::Config(format!(
                "{}: ligne {numero}: ens, chaine ou backend_gen incompatible",
                p.display()
            )));
        }
        if !r.cfgs.insert(record.cfg) {
            return Err(Erreur::Config(format!(
                "{}: cfg {} dupliquée",
                p.display(),
                record.cfg
            )));
        }
        r.longueur += n as u64;
        r.nouvelle_ligne = !texte.ends_with(b"\n");
    }
    Ok(r)
}

fn dossier_manifeste(out: &Path) -> PathBuf {
    if !out.exists() {
        return out.to_path_buf();
    }
    // Chaque invocation garde sa provenance sans réécrire celle de la première passe.
    (1usize..)
        .map(|i| out.join("reprises").join(format!("r{i:06}")))
        .find(|p| !p.exists())
        .expect("un répertoire de provenance libre")
}

pub fn mesurer(m: &ArgMatches) -> Resultat<i32> {
    let dir = PathBuf::from(arg::<String>(m, "ensemble")?);
    let out = std::path::absolute(arg::<String>(m, "out")?)
        .map_err(|e| Erreur::io("répertoire de sortie", e))?;
    let ens: String = arg(m, "ens")?;
    let chaine: usize = arg(m, "chaine")?;
    let backend_gen: String = arg(m, "backend-gen")?;
    let chaque: usize = arg(m, "flot-complet-chaque")?;
    let mut params = ParamsTopo {
        dt: arg(m, "dt")?,
        balayages: arg(m, "balayages")?,
        pas_max: arg(m, "pas-max")?,
        sans_flot: m.get_flag("sans-flot"),
        ..ParamsTopo::default()
    };
    let max_cfg: Option<usize> = match m.get_one::<String>("max-cfg") {
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n > 0 => Some(n),
            _ => {
                return Err(Erreur::Config(format!(
                    "--max-cfg : entier > 0 attendu, « {s} » reçu"
                )))
            }
        },
        None => None,
    };
    if chaine == 0
        || chaque == 0
        || ens.is_empty()
        || !params.dt.is_finite()
        || params.dt <= 0.0
        || params.pas_max < 500
        || topo::TEMPS_A2.iter().any(|t| {
            let n = (t / params.dt).round();
            n < 1.0 || n > params.pas_max as f64
        })
    {
        return Err(Erreur::Config("topo : chaine et flot-complet-chaque > 0, ens non vide, dt fini > 0 et temps de flot accessibles (pas-max ≥ 500) requis".into()));
    }
    let jsonl = out.join("topo.jsonl");
    let existe = jsonl.exists();
    if existe && !m.get_flag("reprendre") {
        return Err(Erreur::Config(format!(
            "{} existe déjà ; utiliser --reprendre",
            jsonl.display()
        )));
    }
    let reprise = if existe {
        lire_reprise(&jsonl, &ens, chaine, &backend_gen)?
    } else {
        Reprise {
            cfgs: BTreeSet::new(),
            longueur: 0,
            nouvelle_ligne: false,
        }
    };
    let fichiers = configurations(&dir)?;
    let (_, l) = npy::inspecter_champ(&fichiers[0].1)?;
    let b = backend_voulu(&arg::<String>(m, "backend")?, l)?;
    admission(b, l, false)?;
    let mut entrees = Vec::new();
    for (_, p) in &fichiers {
        let (_, cote) = npy::inspecter_champ(p)?;
        if cote != l {
            return Err(Erreur::Config(
                "topo : toutes les configurations doivent avoir le même L".into(),
            ));
        }
        entrees.push(json!({"chemin": p, "sha256": sha(p)?, "octets": std::fs::metadata(p).map_err(|e| io(p, e))?.len(), "L": l}));
    }
    let argv: Vec<_> = std::env::args().collect();
    let mut man = Manifeste::ouvrir_avec_garde(
        &dossier_manifeste(&out),
        "kvu obs topo",
        &argv,
        &GardeSsd::machine(),
        (fichiers.len() as u64).saturating_mul(1 << 20),
    )?;
    let resultat = (|| -> Resultat<()> {
        man.champ("backend", &json!(b.nom()).to_string())?;
        man.champ("entrees", &json!(entrees).to_string())?;
        man.champ(
            "parametres",
            &json!({"ens": ens, "chaine": chaine, "backend_gen": backend_gen,
            "flot_complet_chaque": chaque, "max_cfg": max_cfg, "mesure": params})
            .to_string(),
        )?;
        if existe {
            man.champ(
                "reprise",
                &json!({"cfgs": reprise.cfgs, "sha256_avant": sha(&jsonl)?}).to_string(),
            )?;
        }
        let calc = Calcul::ouvrir(b)?;
        if let Some(v) = &calc._verrou {
            man.verrou(v.mode())?;
        }
        let espace: Option<EspaceTopo> = calc
            .module
            .as_ref()
            .map(|module| {
                man.ajouter_compil(&[module.provenance()])?;
                module.espace(l)
            })
            .transpose()?;
        let mut sortie = if existe {
            OpenOptions::new().append(true).open(&jsonl)
        } else {
            OpenOptions::new().write(true).create_new(true).open(&jsonl)
        }
        .map_err(|e| io(&jsonl, e))?;
        if existe {
            sortie
                .set_len(reprise.longueur)
                .map_err(|e| io(&jsonl, e))?;
            if reprise.nouvelle_ligne {
                sortie.write_all(b"\n").map_err(|e| io(&jsonl, e))?;
            }
        }
        let mut mesurees = 0usize;
        for (cfg, p) in &fichiers {
            if reprise.cfgs.contains(cfg) {
                continue;
            }
            if max_cfg.is_some_and(|n| mesurees >= n) {
                break;
            }
            let debut = Instant::now();
            let champ = npy::lire_champ(p)?;
            params.flot_complet = !params.sans_flot && (chaque == 1 || cfg % chaque == 0);
            let mut record = match (&calc.module, &espace) {
                (Some(module), Some(e)) => module.mesurer(e, &champ, &params)?,
                _ => topo::mesurer_cpu(&champ, &params)?,
            };
            record.ens.clone_from(&ens);
            record.chaine = chaine;
            record.cfg = *cfg;
            record.backend_gen.clone_from(&backend_gen);
            record.fichier = p.to_string_lossy().into_owned();
            let mut ligne =
                serde_json::to_vec(&record).map_err(|e| Erreur::Format(e.to_string()))?;
            ligne.push(b'\n');
            sortie
                .write_all(&ligne)
                .and_then(|()| sortie.flush())
                .map_err(|e| io(&jsonl, e))?;
            mesurees += 1;
            println!(
                "cfg={cfg:06} Q(t/a²=3)={:e} Q_I={:e} temps={:.3}s",
                record
                    .flot_a2
                    .q_imp
                    .get(IDX_T3)
                    .copied()
                    .unwrap_or(f64::NAN),
                record
                    .cooling
                    .q_imp
                    .last()
                    .copied()
                    .unwrap_or(f64::NAN)
                    .round(),
                debut.elapsed().as_secs_f64()
            );
        }
        let restantes = fichiers
            .iter()
            .filter(|(cfg, _)| !reprise.cfgs.contains(cfg))
            .count()
            - mesurees;
        if max_cfg.is_some() {
            println!(
                "topo : {mesurees} configurations mesurées par cette passe, {restantes} restantes"
            );
        }
        man.fichier(&jsonl)
    })();
    match resultat {
        Ok(()) => {
            man.terminer(0)?;
            Ok(0)
        }
        Err(e) => {
            let _ = man.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
    }
}

fn ligne(verdict: Verdict, id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate {
        verdict,
        id: id.into(),
        detail: detail.into(),
    }
}

fn decision(ok: bool) -> Verdict {
    if ok {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

fn emettre(lignes: &[LigneGate], fichier: Option<&Path>) -> Resultat<i32> {
    let texte: String = lignes.iter().map(|l| format!("{l}\n")).collect();
    if let Some(p) = fichier {
        ecrire(p, texte.as_bytes())?;
    }
    std::io::stdout()
        .lock()
        .write_all(texte.as_bytes())
        .map_err(|e| Erreur::io("sortie standard", e))?;
    Ok(i32::from(lignes.iter().any(|l| l.verdict == Verdict::Fail)))
}

fn ecrire(p: &Path, octets: &[u8]) -> Resultat<()> {
    GardeSsd::machine().verifier(p, octets.len() as u64)?;
    if let Some(parent) = p.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    }
    std::fs::write(p, octets).map_err(|e| io(p, e))
}

pub fn gates(m: &ArgMatches) -> Resultat<i32> {
    let (nom, sm) = m
        .subcommand()
        .ok_or_else(|| Erreur::Config("topo-gates : sous-commande requise".into()))?;
    match nom {
        "analyse" => {
            let mut records = Vec::new();
            for p in chemins(sm, "jsonl") {
                records.extend(topo_gates::lire_jsonl(&p)?);
            }
            let lignes = topo_gates::analyser(&arg::<String>(sm, "ens")?, &records);
            let out = sm.get_one::<String>("lignes").map(Path::new);
            emettre(&lignes, out)
        }
        "analyse-v5" => {
            let mut records = Vec::new();
            for p in chemins(sm, "jsonl") {
                records.extend(topo_gates::lire_jsonl(&p)?);
            }
            let lignes = topo_gates::analyser_v5(&arg::<String>(sm, "ens")?, &records);
            let out = sm.get_one::<String>("lignes").map(Path::new);
            emettre(&lignes, out)
        }
        "inst" | "cool-id" => {
            let b = backend_voulu(&arg::<String>(sm, "backend")?, 16)?;
            admission(b, if nom == "inst" { 16 } else { 2 }, false)?;
            let calc = Calcul::ouvrir(b)?;
            let lignes = if nom == "inst" {
                inst(&calc, b)?
            } else {
                cool_id(&calc, b)?
            };
            emettre(&lignes, None)
        }
        "gpu" => gate_gpu(sm),
        "certificat" => certificat(sm),
        _ => Err(Erreur::Config(format!(
            "topo-gates : sous-commande inconnue « {nom} »"
        ))),
    }
}

fn inst(calc: &Calcul, b: BackendKind) -> Resultat<Vec<LigneGate>> {
    // Littéraux arrondis une seule fois au f64 représentable, comme les valeurs gelées v4.3.
    #[allow(clippy::excessive_precision)]
    let references = [
        (8, 1.99358272808992157, 2.00152533639564373),
        (12, 1.99873108290073163, 2.00006061806441646),
        (16, 1.99959843702319396, 2.00000608755305483),
    ];
    let mut ok = true;
    let mut details = Vec::new();
    for (l, tre, imp) in references {
        let champ = topo::champ_abelien(l);
        let c = calc.charges(&champ)?;
        let g = calc.charges(&topo::transformer_jauge(&champ, 26102619001))?;
        let r = calc.charges(&topo::reflechir_axe3(&champ))?;
        let ecarts = [
            (c.q_tre - tre).abs(),
            (c.q_imp - imp).abs(),
            (g.q_tre - c.q_tre).abs(),
            (g.q_imp - c.q_imp).abs(),
            (r.q_tre + c.q_tre).abs(),
            (r.q_imp + c.q_imp).abs(),
        ];
        let gain = c.q_imp - c.q_tre;
        ok &= ecarts.iter().all(|&e| e <= 1e-10) && gain >= 4e-4;
        details.push(format!("L={l} gel_tre={:e} gel_imp={:e} jauge_tre={:e} jauge_imp={:e} reflexion_tre={:e} reflexion_imp={:e} ameliore_moins_trefle={gain:e}",
            ecarts[0], ecarts[1], ecarts[2], ecarts[3], ecarts[4], ecarts[5]));
    }
    Ok(vec![ligne(
        decision(ok),
        "TOPO-INST",
        format!("backend={} ; {}", b.nom(), details.join(" ; ")),
    )])
}

fn ecart_matrices(w: &[Q], u: &[Q]) -> f64 {
    w.iter()
        .zip(u)
        .map(|(&w, u)| {
            let d = qdag(w);
            let trace = 2.0 * (u[0] * d[0] + u[1] * d[1] + u[2] * d[2] + u[3] * d[3]);
            (trace - 2.0 * qnorm(w)).abs()
        })
        .fold(0.0, f64::max)
}

fn cool_id(calc: &Calcul, b: BackendKind) -> Resultat<Vec<LigneGate>> {
    let w = topo::cool_id_matrices(26102619002, 1000);
    let ecart = |dagger| -> Resultat<f64> {
        match &calc.module {
            Some(module) => Ok(ecart_matrices(&w, &module.cool_id_maj(&w, dagger)?)),
            None => Ok(topo::cool_id_ecart(&w, dagger)),
        }
    };
    let (normal, mutant) = (ecart(true)?, ecart(false)?);
    Ok(vec![
        ligne(
            decision(normal <= 1e-13),
            "TOPO-COOL-ID",
            format!("backend={} ecart_max={normal:e}", b.nom()),
        ),
        ligne(
            decision(mutant >= 1e-2),
            "TOPO-COOL-ID-MUT",
            format!(
                "{}backend={} ecart_max={mutant:e} mutant sans dagger",
                if mutant >= 1e-2 { "" } else { "ECHEC_OUTIL " },
                b.nom()
            ),
        ),
    ])
}

fn gate_gpu(m: &ArgMatches) -> Resultat<i32> {
    let fichiers = chemins(m, "cfg");
    if fichiers.len() < 3 {
        return emettre(
            &[ligne(
                Verdict::Skip,
                "TOPO-GPU",
                format!("{} configurations ; trois requises", fichiers.len()),
            )],
            None,
        );
    }
    if fichiers.len() != 3 {
        return Err(Erreur::Config(
            "TOPO-GPU : exactement trois configurations requises".into(),
        ));
    }
    let b = backend_voulu(&arg::<String>(m, "backend")?, 16)?;
    if b == BackendKind::Cpu {
        return emettre(
            &[ligne(
                Verdict::Skip,
                "TOPO-GPU",
                "backend=cpu ; GPU indisponible",
            )],
            None,
        );
    }
    for p in &fichiers {
        admission(b, npy::inspecter_champ(p)?.1, true)?;
    }
    let calc = Calcul::ouvrir(b)?;
    let module = calc
        .module
        .as_ref()
        .ok_or_else(|| Erreur::Config("TOPO-GPU : module absent".into()))?;
    let mut details = Vec::new();
    let mut ok = true;
    for p in fichiers {
        let champ = npy::lire_champ(&p)?;
        let e = module.espace(champ.l)?;
        let cpu_flot = topo::charges(&super::ref_cpu::flot_n_pas(&champ, 0.01, 300)?)?;
        let cpu_cool = topo::charges(&topo::cooling(&champ, 20, true)?.1)?;
        module.charger(&e, &champ)?;
        module.flot_vers(&e, 0.01, 300)?;
        let gpu_flot = module.charges(&e, Cible::Flot)?;
        for _ in 0..20 {
            module.balayage_cooling(&e, true)?;
        }
        let gpu_cool = module.charges(&e, Cible::Cooling)?;
        for (mode, cpu, gpu) in [
            ("flot", cpu_flot, gpu_flot),
            ("cooling", cpu_cool, gpu_cool),
        ] {
            let delta = (gpu.q_imp - cpu.q_imp).abs();
            ok &= delta <= 1e-12;
            details.push(ligne(Verdict::Info, "TOPO-GPU-DETAIL", format!("backend={} cfg={} mode={mode} delta_q_imp={delta:e} delta_q_tre={:e} delta_E_plaq={:e}",
                b.nom(), p.display(), (gpu.q_tre - cpu.q_tre).abs(), (24.0 * (1.0 - gpu.plaq) - 24.0 * (1.0 - cpu.plaq)).abs())));
        }
    }
    let mut lignes = vec![ligne(
        decision(ok),
        "TOPO-GPU",
        format!(
            "backend={} trois champs flotés et refroidis ; seuil_delta_q_imp={:e}",
            b.nom(),
            1e-12
        ),
    )];
    lignes.extend(details);
    emettre(&lignes, None)
}

/// Ne crée aucun verdict : seules les lignes au format Display de LigneGate sont recopiées.
fn lire_lignes(p: &Path) -> Resultat<Vec<LigneGate>> {
    let mut lignes = Vec::new();
    for texte in BufReader::new(File::open(p).map_err(|e| io(p, e))?).lines() {
        let texte = texte.map_err(|e| io(p, e))?;
        let Some((v, suite)) = texte.split_once(' ') else {
            continue;
        };
        let verdict = match v {
            "PASS" => Verdict::Pass,
            "FAIL" => Verdict::Fail,
            "SKIP" => Verdict::Skip,
            "INFO" => Verdict::Info,
            _ => continue,
        };
        let (id, detail) = suite.split_once(' ').ok_or_else(|| {
            Erreur::Format(format!(
                "{} : ligne de gate hors du format LigneGate",
                p.display()
            ))
        })?;
        if id.is_empty() {
            return Err(Erreur::Format(format!(
                "{} : identifiant de gate vide",
                p.display()
            )));
        }
        lignes.push(ligne(verdict, id, detail));
    }
    Ok(lignes)
}

fn certificat(m: &ArgMatches) -> Resultat<i32> {
    let mut lignes = Vec::new();
    for p in chemins(m, "lignes") {
        lignes.extend(lire_lignes(&p)?);
    }
    let non_vert: Vec<_> = lignes
        .iter()
        .filter(|l| matches!(l.verdict, Verdict::Skip | Verdict::Fail))
        .collect();
    let commandes: Vec<_> = m.get_many::<String>("cmd").into_iter().flatten().collect();
    let document = json!({"schema": "kvu-topo/1", "binaire_sha256": crate::stockage::provenance::sha256_binaire()?,
        "git_head": crate::stockage::provenance::git_head(), "lignes": lignes,
        "commandes": commandes, "non_vert": non_vert});
    let out = PathBuf::from(arg::<String>(m, "out")?);
    let mut octets =
        serde_json::to_vec_pretty(&document).map_err(|e| Erreur::Format(e.to_string()))?;
    octets.push(b'\n');
    ecrire(&out, &octets)?;
    emettre(&lignes, None)
}
