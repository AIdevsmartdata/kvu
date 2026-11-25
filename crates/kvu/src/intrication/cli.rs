//! Ligne de commande du module intrication, au contrat des modules kvu (`commandes()` / `executer(nom, &ArgMatches)`, voir
//! `cli/mod.rs`) : inscrite dans `cli::MODULES` et `cli::SOUS_COMMANDES`, c'est `kvu intrication …` (seule entrée ; le
//! binaire séparé `kvu_intrication` a disparu à la révision REV2 du 11/11/2026).
//!
//! Chaque sous-commande qui produit des résultats écrit un RÉPERTOIRE DE RUN neuf (`--sortie`, refus s'il existe) :
//! `manifest.json` (`stockage::provenance::Manifeste`, garde SSD, champs `geometrie` — direction du temps explicite — et
//! `parametres`) puis `resultats.jsonl` (une ligne JSON par porte, par largeur ou par balayage, recopiée sur la sortie
//! standard). Les fonctions `lancer_*` prennent des structures de paramètres (pas de `Parser`).

use super::portes::{self, ParamsChaine};
use super::{controler_specs, octets_chaine, octets_points, GeoRep, Mutant, Region, Reseau};
use crate::commun::{code, Erreur, Resultat};
use crate::geom::GeoSpec;
use crate::machine::admission::{self, BesoinMemoire};
use crate::stockage::{GardeSsd, Manifeste};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const NOM: &str = "intrication";

/// Statut du module (révision REV2 du 11/11/2026) : EXPÉRIMENTAL, mêmes mots que le module tube. L'aide de
/// `kvu intrication` commence par « [EXPÉRIMENTAL] », chaque manifeste porte `parametres.statut = "experimental"` et chaque
/// lancement écrit `AVERTISSEMENT_EXPERIMENTAL` sur stderr.
pub const STATUT: &str = "experimental";
pub const AVERTISSEMENT_EXPERIMENTAL: &str =
    "module expérimental : interface et formats susceptibles de changer ; résultats à confirmer indépendamment";

fn a_usize(nom: &'static str, defaut: Option<&'static str>, aide: &'static str) -> Arg {
    let a = Arg::new(nom)
        .long(nom)
        .value_parser(clap::value_parser!(usize))
        .help(aide);
    match defaut {
        Some(d) => a.default_value(d),
        None => a.required(true),
    }
}

fn a_f64(nom: &'static str, aide: &'static str) -> Arg {
    Arg::new(nom)
        .long(nom)
        .value_parser(clap::value_parser!(f64))
        .required(true)
        .help(aide)
}

fn a_graine() -> Arg {
    Arg::new("graine")
        .long("graine")
        .value_parser(clap::value_parser!(u64))
        .default_value("1")
        .help("graine u64 (seule dans la clé Philox ; point, largeur, réplique et usage sont des champs du compteur)")
}

fn a_sortie() -> Arg {
    Arg::new("sortie")
        .long("sortie")
        .value_parser(clap::value_parser!(PathBuf))
        .required(true)
        .help("répertoire de run NEUF (manifest.json + resultats.jsonl)")
}

fn args_chaine() -> Vec<Arg> {
    vec![
        a_usize(
            "therm",
            Some("100"),
            "pas de thermalisation (1 HB + n_or OR) ; therm + mes × sep < 2³² balayages par chaîne",
        ),
        a_usize(
            "mes",
            Some("400"),
            "mesures par point (multiple de --blocs)",
        ),
        a_usize("sep", Some("1"), "pas entre deux mesures (≥ 1)"),
        a_usize("or", Some("3"), "surrelaxations par pas (≤ 1000)"),
        a_usize(
            "blocs",
            Some("20"),
            "blocs pour l'erreur (2 ≤ blocs ≤ mes ; aucun temps d'autocorrélation)",
        ),
    ]
}

/// Les sous-commandes du module (une commande `intrication` et ses sous-commandes).
pub fn commandes() -> Vec<Command> {
    vec![commande()]
}

/// DOMAINE ADMIS des options (relecture c19b, méthode commune avec le module tube) : (option, intervalle admis, raison).
/// Contrôlé en UN endroit, `valider`, avant l'admission mémoire, toute allocation proportionnelle aux entrées et toute
/// écriture (hors domaine : code 2, rien créé) ; repris dans l'aide (après-texte de chaque sous-commande), dans la note
/// `docs/v0.2/notes/intrication.md` et dans le rapport ; parcouru par `tests/intrication_domaine.rs`. Avec V = 2·L³·T.
pub const DOMAINE: &[(&str, &str, &str)] = &[
    ("--l, --t", "pairs, ≥ 2 ; 4·V ≤ 2³¹ et 2·(6·V + 6·L³) ≤ 2³⁰ − 1", "capacités C2 des liens et des plaquettes (geom::tables), volume sans débordement"),
    ("--l0, --l1", "0 ≤ l ≤ L, --l0 ≠ --l1", "région ; l0 = l1 : ΔF = 0 par construction, erreur nulle"),
    ("--largeurs", "0 ≤ l < L, distinctes", "l + 1 ≤ L sans débordement ; deux largeurs égales tireraient les mêmes nombres"),
    ("--beta", "fini, 0,001 ≤ β ≤ 1000", "Creutz : 1 − e^(−2β|A|) garde ≥ 12 chiffres pour |A| ≥ 0,1 ; réseau gelé au-delà ; aucun infini intermédiaire"),
    ("--nalpha, --nbeta", "impairs, 3 ≤ n ≤ 32767", "quadrature de Simpson ; champ de 15 bits du flux"),
    ("--therm, --mes, --sep", "--sep ≥ 1 ; therm + mes × sep < 2³²", "le compteur Philox de balayage ne reboucle jamais"),
    ("--blocs", "2 ≤ blocs ≤ mes, mes multiple de blocs", "moyenne et erreur sur les mêmes observations"),
    ("--or", "0 ≤ or ≤ 1000", "surrelaxations par pas"),
    ("--pas", "pas ≥ 1, 1 + pas < 2³²", "debit : pas préparatoire compris"),
    ("--graine", "tout u64", "seule dans la clé Philox"),
    ("--sortie", "répertoire qui n'existe pas", "un run = un répertoire neuf"),
];

/// Le tableau `DOMAINE` en texte, pour l'aide.
pub fn texte_domaine() -> String {
    let mut t = String::from("Domaine admis (contrôlé avant toute allocation et toute écriture ; hors domaine : code 2) :\n");
    for (o, i, r) in DOMAINE {
        t.push_str(&format!("  {o:<22} {i}  [{r}]\n"));
    }
    t
}

/// Contrôle UNIQUE des options d'une sous-commande contre `DOMAINE` : pur, sans écriture ni allocation proportionnelle aux
/// entrées ; appelé par `executer` avant l'admission mémoire et l'ouverture du run. `--sortie` est contrôlé à l'ouverture.
pub fn valider(m: &ArgMatches) -> Resultat<()> {
    let communs = |s: &ArgMatches, points: &[&str]| -> Resultat<()> {
        for &o in points {
            portes::controler_points(o, u(s, o)).map_err(err)?;
        }
        portes::controler_beta(f(s, "beta")).map_err(err)?;
        portes::controler_chaine(&chaine(s)).map_err(err)
    };
    match m.subcommand() {
        Some(("portes", _)) => Ok(()),
        Some(("balayage", s)) => {
            communs(s, &["nalpha"])?;
            let (l0, l1) = (u(s, "l0"), u(s, "l1"));
            if l0 == l1 {
                return Err(Erreur::Config(format!(
                    "--l0 = --l1 = {l0} : aucune plaquette ne change, ΔF = 0 par construction (erreur nulle) ; largeurs différentes requises"
                )));
            }
            specs(u(s, "l"), u(s, "t"), l0, l1).map(|_| ())
        }
        Some(("courbe", s)) => {
            communs(s, &["nalpha"])?;
            let (l, t) = (u(s, "l"), u(s, "t"));
            let largeurs = || {
                s.get_many::<usize>("largeurs")
                    .into_iter()
                    .flatten()
                    .copied()
            };
            for (i, lw) in largeurs().enumerate() {
                if lw >= l {
                    return Err(Erreur::Config(format!("--largeurs : largeur {lw} ≥ L = {l} ; chaque largeur l doit vérifier l + 1 ≤ L")));
                }
                if largeurs().take(i).any(|x| x == lw) {
                    return Err(Erreur::Config(format!("--largeurs : largeur {lw} répétée")));
                }
                specs(l, t, lw, lw + 1)?;
            }
            Ok(())
        }
        Some(("doublement", s)) => {
            communs(s, &["nalpha", "nbeta"])?;
            let l = u(s, "l");
            specs(l, u(s, "t"), 0, l).map(|_| ())
        }
        Some(("debit", s)) => {
            let (l, t, pas, n_or) = (u(s, "l"), u(s, "t"), u(s, "pas"), u(s, "or"));
            if pas == 0 {
                return Err(Erreur::Config(
                    "debit : --pas = 0 ; au moins un pas chronométré requis".into(),
                ));
            }
            if portes::balayages_par_chaine(1, pas, 1).is_none() {
                return Err(Erreur::Config(format!(
                    "debit : 1 + --pas {pas} : au plus 2³² − 1 balayages par chaîne (le compteur Philox de balayage ne reboucle pas)"
                )));
            }
            if n_or > portes::OR_MAX {
                return Err(Erreur::Config(format!(
                    "--or = {n_or} : au plus {} surrelaxations par pas",
                    portes::OR_MAX
                )));
            }
            specs(l, t, l / 2, l / 2 + 1).map(|_| ())
        }
        _ => Err(Erreur::Config(
            "intrication : sous-commande manquante (portes, balayage, courbe, doublement, debit)"
                .into(),
        )),
    }
}

pub fn commande() -> Command {
    let mut c = commande_sans_domaine();
    for sous in ["portes", "balayage", "courbe", "doublement", "debit"] {
        c = c.mut_subcommand(sous, |x| x.after_help(texte_domaine()));
    }
    c.after_help(texte_domaine())
}

fn commande_sans_domaine() -> Command {
    Command::new(NOM)
        .about("[EXPÉRIMENTAL] Entropie de Rényi-2 par répliques, SU(2) Wilson 4D, CPU F64 : portes, balayage en α, courbe ΔS₂(l), doublement")
        .subcommand_required(true)
        .subcommand(
            Command::new("portes")
                .about("portes (a)-(f) ; rc 0 si toutes vertes, 8 sinon")
                .arg(Arg::new("rapide").long("rapide").action(ArgAction::SetTrue).help("tailles et statistiques réduites"))
                .arg(a_sortie()),
        )
        .subcommand(
            Command::new("balayage")
                .about("F(l1) − F(l0) par interpolation en α entre deux tranches")
                .args([
                    a_usize("l", None, "côté spatial L"),
                    a_usize("t", None, "durée T d'une réplique"),
                    a_usize("l0", None, "largeur de la tranche à α = 0"),
                    a_usize("l1", None, "largeur de la tranche à α = 1"),
                    a_f64("beta", "couplage β (fini, 0,001 ≤ β ≤ 1000)"),
                    a_usize("nalpha", Some("9"), "points α (impair, 3 à 32767, Simpson)"),
                    a_graine(),
                ])
                .args(args_chaine())
                .arg(a_sortie()),
        )
        .subcommand(
            Command::new("courbe")
                .about("ΔS₂(l) = S₂(l+1) − S₂(l) et C(l+½) = (l+½)³ ΔS₂ / (2L²) pour chaque largeur l")
                .args([
                    a_usize("l", None, "côté spatial L"),
                    a_usize("t", None, "durée T d'une réplique"),
                    a_f64("beta", "couplage β (fini, 0,001 ≤ β ≤ 1000)"),
                    Arg::new("largeurs")
                        .long("largeurs")
                        .value_parser(clap::value_parser!(usize))
                        .value_delimiter(',')
                        .required(true)
                        .help("largeurs l < L, distinctes (liste séparée par des virgules)"),
                    a_usize("nalpha", Some("9"), "points α (impair, 3 à 32767, Simpson)"),
                    a_graine(),
                ])
                .args(args_chaine())
                .arg(a_sortie()),
        )
        .subcommand(
            Command::new("doublement")
                .about("S₂(tranche entière) = 2 log Z(T) − log Z(2T), par interpolation en α et par intégration en β")
                .args([
                    a_usize("l", None, "côté spatial L"),
                    a_usize("t", None, "durée T d'une réplique"),
                    a_f64("beta", "couplage β (fini, 0,001 ≤ β ≤ 1000)"),
                    a_usize("nalpha", Some("9"), "points α (impair, 3 à 32767, Simpson)"),
                    a_usize("nbeta", Some("9"), "points β sur [0, β] (impair, 3 à 32767, Simpson)"),
                    a_graine(),
                ])
                .args(args_chaine())
                .arg(a_sortie()),
        )
        .subcommand(
            Command::new("debit").about("secondes par pas sur la géométrie interpolée (sans manifeste : aucune sortie écrite)").args([
                a_usize("l", None, "côté spatial L"),
                a_usize("t", None, "durée T"),
                a_usize("pas", Some("20"), "pas chronométrés (≥ 1 ; 1 + pas < 2³² balayages)"),
                a_usize("or", Some("3"), "surrelaxations par pas (≤ 1000)"),
            ]),
        )
}

fn u(m: &ArgMatches, k: &str) -> usize {
    *m.get_one::<usize>(k)
        .expect("argument requis ou par défaut")
}
fn f(m: &ArgMatches, k: &str) -> f64 {
    *m.get_one::<f64>(k).expect("argument requis")
}
fn g(m: &ArgMatches) -> u64 {
    *m.get_one::<u64>("graine").expect("graine par défaut")
}
fn sortie(m: &ArgMatches) -> PathBuf {
    m.get_one::<PathBuf>("sortie")
        .expect("--sortie requis")
        .clone()
}
fn chaine(m: &ArgMatches) -> ParamsChaine {
    ParamsChaine {
        therm: u(m, "therm"),
        mes: u(m, "mes"),
        sep: u(m, "sep"),
        n_or: u(m, "or"),
        blocs: u(m, "blocs"),
    }
}
fn chaine_json(p: &ParamsChaine) -> Value {
    json!({"therm": p.therm, "mes": p.mes, "sep": p.sep, "or": p.n_or, "blocs": p.blocs})
}

#[derive(Clone, Debug)]
pub struct ParamsPortes {
    pub rapide: bool,
}

#[derive(Clone, Debug)]
pub struct ParamsBalayage {
    pub l: usize,
    pub t: usize,
    pub l0: usize,
    pub l1: usize,
    pub beta: f64,
    pub nalpha: usize,
    pub graine: u64,
    pub chaine: ParamsChaine,
}

#[derive(Clone, Debug)]
pub struct ParamsCourbe {
    pub l: usize,
    pub t: usize,
    pub beta: f64,
    pub largeurs: Vec<usize>,
    pub nalpha: usize,
    pub graine: u64,
    pub chaine: ParamsChaine,
}

#[derive(Clone, Debug)]
pub struct ParamsDoublement {
    pub l: usize,
    pub t: usize,
    pub beta: f64,
    pub nalpha: usize,
    pub nbeta: usize,
    pub graine: u64,
    pub chaine: ParamsChaine,
}

/// Spécification d'une réplique recollée du socle `geom` : L³ × T, 2 répliques, tranche sur l'axe 1 de largeur `largeur`.
pub fn spec(l: usize, t: usize, largeur: usize) -> GeoSpec {
    super::spec_tranche(l, t, largeur)
}

/// Les deux spécifications d'une interpolation (A₀ à α = 0, A₁ à α = 1), contrôlées AVANT tout calcul (capacités C2 du
/// socle, recollement sur 0, ℓ₁ = ℓ₂ = ℓ₃ pairs, T pair).
pub fn specs(l: usize, t: usize, l0: usize, l1: usize) -> Resultat<(GeoSpec, GeoSpec)> {
    let (s0, s1) = (spec(l, t, l0), spec(l, t, l1));
    controler_specs(&s0, &s1).map_err(Erreur::Config)?;
    Ok((s0, s1))
}

/// Garde de mémoire (relecture intrication-claude, objection 2) : estimation AVANT toute allocation (`octets_chaine`, construction
/// comprise) fois le nombre de chaînes simultanées (`portes::chaines_simultanees`), soumise à l'admission commune de kvu
/// (`machine::admission` : RAM disponible moins le plancher de la politique) ; refus `Erreur::Ressources` (code 5), rien créé.
/// Comme l'admission de `su2 gen`, elle lit /proc/meminfo : un plafond de groupe de contrôle (cgroup) n'est pas vu.
/// Relecture c19 (majeure 6, mineure 9) : le besoin est une fonction PURE (`besoin_memoire`, sans allocation, testée contre un
/// plafond donné), qui compte aussi les tableaux proportionnels aux mesures, aux blocs et aux points.
pub fn garde_memoire(
    quoi: &str,
    l: usize,
    t: usize,
    simultanees: usize,
    p: &ParamsChaine,
    points: usize,
) -> Resultat<()> {
    let besoin = besoin_memoire(l, t, simultanees, p, points)?;
    let quoi = format!(
        "intrication {quoi} L = {l}, T = {t} ({simultanees} chaîne(s) simultanée(s), {points} point(s) : {:.2} Gio estimés)",
        besoin as f64 / (1u64 << 30) as f64
    );
    admission::admettre(
        &quoi,
        BesoinMemoire {
            hote: besoin,
            gpu: 0,
        },
        false,
    )
}

/// Besoin de mémoire (octets) d'un run, SANS rien allouer : `simultanees` chaînes (`octets_chaine`, mesures et blocs compris)
/// plus les tableaux proportionnels aux `points` (`octets_points`). Arithmétique saturée (un débordement donne u64::MAX, refusé).
pub fn besoin_memoire(
    l: usize,
    t: usize,
    simultanees: usize,
    p: &ParamsChaine,
    points: usize,
) -> Resultat<u64> {
    let geo = GeoRep::nouveau(l, t, 2).map_err(err)?;
    Ok(octets_chaine(&geo, p.mes, p.blocs)
        .saturating_mul(simultanees as u64)
        .saturating_add(octets_points(points)))
}

/// Blocs `kvu-geometrie/1` d'une porte (une paire si interpolation).
fn bloc_porte(l: usize, t: usize, l0: usize, l1: Option<usize>) -> Value {
    match l1 {
        None => json!({"geometrie": spec(l, t, l0).bloc()}),
        Some(l1) => {
            json!({"geometrie": spec(l, t, l0).bloc(), "geometrie_alpha1": spec(l, t, l1).bloc()})
        }
    }
}

/// Un run : répertoire neuf, manifeste « en_cours », resultats.jsonl écrit au fil de l'eau. Il garde les spécifications
/// inscrites au manifeste : le calcul les relit (`Run::spec`) au lieu de reconstruire sa géométrie à part (objection 8).
pub struct Run {
    man: Manifeste,
    fichier: std::fs::File,
    chemin: PathBuf,
    geometries: Vec<(String, GeoSpec)>,
}

impl Run {
    /// `geometries` : (clé, spécification) inscrites par `GeoSpec::inscrire` (bloc canonique du socle) ; un run sans
    /// configuration écrit `geometrie` (A₀) et `geometrie_alpha1` (A₁), ou des clés `geometrie_<nom>` s'il en a plusieurs.
    /// Une erreur APRÈS la création du manifeste (inscription, création de resultats.jsonl) le clôt en échec (objection 4).
    pub fn ouvrir(
        rep: &Path,
        sous: &str,
        geometries: &[(String, GeoSpec)],
        mut params: Value,
    ) -> Resultat<Run> {
        if let Some(o) = params.as_object_mut() {
            o.insert("statut".into(), Value::String(STATUT.into()));
        }
        let argv: Vec<String> = std::env::args().collect();
        let mut man = Manifeste::ouvrir_avec_garde(
            rep,
            &format!("kvu intrication {sous}"),
            &argv,
            &GardeSsd::machine(),
            1 << 22,
        )?;
        let suite = (|| -> Resultat<(std::fs::File, PathBuf)> {
            for (cle, spec) in geometries {
                spec.inscrire(&mut man, cle)?;
            }
            man.champ("parametres", &params.to_string())?;
            man.champ("fils_rayon", &rayon::current_num_threads().to_string())?;
            let chemin = rep.join("resultats.jsonl");
            let fichier = std::fs::File::create(&chemin)
                .map_err(|e| Erreur::io(format!("création de {}", chemin.display()), e))?;
            Ok((fichier, chemin))
        })();
        match suite {
            Ok((fichier, chemin)) => Ok(Run {
                man,
                fichier,
                chemin,
                geometries: geometries.to_vec(),
            }),
            Err(e) => {
                let _ = man.echouer(e.code_sortie(), &e.to_string());
                Err(e)
            }
        }
    }

    /// Spécification inscrite au manifeste sous la clé `cle`.
    pub fn spec(&self, cle: &str) -> Resultat<GeoSpec> {
        self.geometries
            .iter()
            .find(|(c, _)| c == cle)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| {
                Erreur::Config(format!(
                    "intrication : géométrie « {cle} » absente du manifeste du run"
                ))
            })
    }

    /// Écrit une ligne de résultat (et la recopie sur stdout). Relecture c19 (majeure 4) : une valeur NON FINIE (NaN ou infini,
    /// sérialisés en `null` par serde_json) hors des blocs de géométrie rend `Erreur::Validation` (code 8) APRÈS l'écriture
    /// de la ligne (gardée pour diagnostic) : `conduire` clôt alors le run en échec, jamais de code 0 ni de COMPLETE.
    pub fn ligne(&mut self, v: &Value) -> Resultat<()> {
        println!("{v}");
        writeln!(self.fichier, "{v}")
            .map_err(|e| Erreur::io(format!("écriture de {}", self.chemin.display()), e))?;
        self.fichier
            .flush()
            .map_err(|e| Erreur::io(format!("écriture de {}", self.chemin.display()), e))?;
        match chemin_non_fini(v, "") {
            Some(c) => Err(Erreur::Validation(format!(
                "résultat non fini (null) en « {c} » dans {}",
                self.chemin.display()
            ))),
            None => Ok(()),
        }
    }

    /// Clôture : resultats.jsonl synchronisé (`sync_all`) puis consigné, puis `Manifeste::terminer(rc)` (COMPLETE en dernier,
    /// seulement si rc = 0). Relecture c19 (majeure 8) : une erreur de synchronisation n'est plus ignorée ; elle, comme une
    /// erreur de consignation, clôt le manifeste en échec et est rendue. Une erreur d'écriture du manifeste FINAL lui-même
    /// est rendue, mais le manifeste peut alors rester `en_cours` (rien de plus n'est inscriptible).
    pub fn terminer(self, rc: i32) -> Resultat<i32> {
        self.terminer_avec(rc, |f| f.sync_all())
    }

    /// `terminer` avec une synchronisation donnée (les essais y injectent une erreur d'E/S).
    pub fn terminer_avec(
        mut self,
        rc: i32,
        synchroniser: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
    ) -> Resultat<i32> {
        let mut etape = synchroniser(&self.fichier)
            .map_err(|e| Erreur::io(format!("synchronisation de {}", self.chemin.display()), e));
        if etape.is_ok() {
            etape = self.man.fichier(&self.chemin);
        }
        if let Err(e) = etape {
            let _ = self.man.echouer(e.code_sortie(), &e.to_string());
            return Err(e);
        }
        self.man.terminer(rc)?;
        Ok(rc)
    }

    /// Clôture en échec (statut `echec`, rc et raison consignés, jamais de COMPLETE).
    pub fn echouer(self, rc: i32, raison: &str) -> Resultat<()> {
        self.man.echouer(rc, raison)
    }
}

/// Conduit un run ouvert (relecture intrication-claude, objection 4) : `f` calcule et écrit ; `Ok(rc)` ⟹ `terminer(rc)` ;
/// toute erreur de `f` ⟹ manifeste clos en échec (code de l'erreur) puis l'erreur est rendue ; une panique de `f` ⟹
/// manifeste clos en échec (code 1) puis la panique reprend (le dispatch de kvu la convertit en échec propre). Un run mort ne
/// reste donc jamais `en_cours`.
pub fn conduire(mut run: Run, f: impl FnOnce(&mut Run) -> Resultat<i32>) -> Resultat<i32> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut run))) {
        Ok(Ok(rc)) => run.terminer(rc),
        Ok(Err(e)) => {
            let _ = run.echouer(e.code_sortie(), &e.to_string());
            Err(e)
        }
        Err(panique) => {
            let _ = run.echouer(code::ECHEC, "panique pendant le calcul");
            std::panic::resume_unwind(panique)
        }
    }
}

/// Chemin du premier `null` d'une ligne de résultat, hors des blocs de géométrie du socle (clés `geometrie…`, dont certains
/// champs sont nuls par construction).
fn chemin_non_fini(v: &Value, chemin: &str) -> Option<String> {
    match v {
        Value::Null => Some(chemin.to_string()),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .find_map(|(i, x)| chemin_non_fini(x, &format!("{chemin}[{i}]"))),
        Value::Object(o) => o
            .iter()
            .filter(|(k, _)| !k.starts_with("geometrie"))
            .find_map(|(k, x)| chemin_non_fini(x, &format!("{chemin}.{k}"))),
        _ => None,
    }
}

fn err(e: String) -> Erreur {
    Erreur::Config(e)
}

#[allow(clippy::too_many_arguments)]
fn porte(
    run: &mut Run,
    tout: &mut bool,
    nom: &str,
    vert: bool,
    compte: bool,
    v: Value,
    geo: Value,
) -> Resultat<()> {
    if compte {
        *tout &= vert;
    }
    let mut ligne = json!({"porte": nom, "vert": vert, "compte": compte, "valeurs": v});
    if let (Some(o), Some(g)) = (ligne.as_object_mut(), geo.as_object()) {
        for (k, val) in g {
            o.insert(k.clone(), val.clone());
        }
    }
    run.ligne(&ligne)
}

fn sj<T: serde::Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap_or(Value::Null)
}

/// Portes (a)-(f). Rend le code de sortie (0 si toutes les portes comptées sont vertes, 8 sinon).
pub fn lancer_portes(p: &ParamsPortes, run: &mut Run) -> Resultat<i32> {
    let mut tout = true;
    let rapide = p.rapide;
    let (lp, tp) = if rapide { (4, 4) } else { (6, 6) };
    // (a)
    let a = portes::porte_a(
        4,
        2.3,
        if rapide { 3 } else { 10 },
        2,
        [0x1234_5678_9ABC, 0x0F0F_1111_2222],
    )
    .map_err(err)?;
    porte(
        run,
        &mut tout,
        "a_A_vide_bit_a_bit_CpuSu2",
        a.liens_differents == 0,
        true,
        sj(&a),
        bloc_porte(4, 4, 0, None),
    )?;
    // (b) + mutant
    let b = portes::porte_b(lp, tp, 77, Mutant::Aucun).map_err(err)?;
    let v = b.agrafes_differentes == 0
        && b.plaquettes_differentes == 0
        && b.ecart_action == 0.0
        && b.poids_non_unitaires == 0;
    let geo_b = || {
        let mut g = bloc_porte(lp, tp, lp, None);
        g["geometrie_reference"] = GeoSpec::anisotrope([2 * tp, lp, lp, lp], 0).bloc();
        g
    };
    porte(
        run,
        &mut tout,
        "b_A_plein_egal_reseau_2T_bit_a_bit",
        v,
        true,
        sj(&b),
        geo_b(),
    )?;
    let bm = portes::porte_b(lp, tp, 77, Mutant::AgrafeCoupure).map_err(err)?;
    porte(
        run,
        &mut tout,
        "f_mutant_agrafe_coupure_rougit_b",
        bm.agrafes_differentes > 0 || bm.plaquettes_differentes > 0,
        true,
        sj(&bm),
        geo_b(),
    )?;
    // (c) + mutant
    for larg in [1, 2, lp / 2 + 1, lp] {
        let c =
            portes::porte_c(lp, tp, larg, 99 + larg as u64, Mutant::Aucun, true).map_err(err)?;
        let v = c.ecart_max_plaquette < 1e-12 && c.ecart_action_relatif < 1e-12;
        porte(
            run,
            &mut tout,
            "c_invariance_jauge",
            v,
            true,
            sj(&c),
            bloc_porte(lp, tp, larg, None),
        )?;
    }
    // largeur 3 : la tranche a des sites intérieurs (tout A), où le raccord de la coupure n'est pas masqué par l'identification
    let cm = portes::porte_c(lp, tp, 3, 5, Mutant::AgrafeCoupure, true).map_err(err)?;
    porte(
        run,
        &mut tout,
        "f_mutant_agrafe_coupure_rougit_c",
        cm.ecart_max_plaquette > 1e-3,
        true,
        sj(&cm),
        bloc_porte(lp, tp, 3, None),
    )?;
    let cs = portes::porte_c(lp, tp, 2, 5, Mutant::Aucun, false).map_err(err)?;
    porte(
        run,
        &mut tout,
        "c_info_sans_identification_aux_sommets_de_branchement",
        cs.ecart_max_plaquette > 1e-3,
        false,
        sj(&cs),
        bloc_porte(lp, tp, 2, None),
    )?;
    // (d1) + mutant
    for (larg, alpha) in [(1usize, 0.37f64), (2, 0.0), (2, 1.0), (lp / 2, 0.81)] {
        let d =
            portes::porte_d1(lp, tp, larg, alpha, 3 + larg as u64, Mutant::Aucun).map_err(err)?;
        porte(
            run,
            &mut tout,
            "d1_agrafe_egale_variation_action",
            d.ecart_max < 1e-12,
            true,
            sj(&d),
            bloc_porte(lp, tp, larg, Some(larg + 1)),
        )?;
    }
    let dm = portes::porte_d1(lp, tp, 1, 0.37, 4, Mutant::PoidsIgnores).map_err(err)?;
    porte(
        run,
        &mut tout,
        "d1_mutant_poids_ignores_rouge",
        dm.ecart_max > 1e-3,
        true,
        sj(&dm),
        bloc_porte(lp, tp, 1, Some(2)),
    )?;
    // (d2) + mutant
    let geo = GeoRep::nouveau(lp, tp, 2).map_err(err)?;
    let res = Reseau::nouveau(
        geo,
        Region::tranche(&geo, 2),
        Region::tranche(&geo, 3),
        &[1, 2],
        Mutant::Aucun,
    )
    .map_err(err)?;
    let n = portes::porte_classes(&res);
    porte(
        run,
        &mut tout,
        "d2_classes_independantes",
        n == 0,
        true,
        json!({"plaquettes_en_conflit": n, "plaquettes": res.plaqs.len()}),
        bloc_porte(lp, tp, 2, Some(3)),
    )?;
    let gi = GeoRep::nouveau_sans_controle(lp, tp - 1, 2);
    let resi = Reseau::nouveau(
        gi,
        Region::tranche(&gi, 2),
        Region::tranche(&gi, 3),
        &[1, 2],
        Mutant::Aucun,
    )
    .map_err(err)?;
    let ni = portes::porte_classes(&resi);
    porte(
        run,
        &mut tout,
        "d2_mutant_T_impair_rouge",
        ni > 0,
        true,
        json!({"T": tp - 1, "plaquettes_en_conflit": ni}),
        bloc_porte(lp, tp - 1, 2, Some(3)),
    )?;
    // (d3) + mutant
    let (th, me) = if rapide { (100, 400) } else { (200, 2000) };
    let d3 = portes::porte_d3(4, 4, 2, 2.3, th, me, 21, Mutant::Aucun).map_err(err)?;
    porte(
        run,
        &mut tout,
        "d3_stationnarite_schwinger_dyson",
        d3.z.abs() < 4.0 && d3.z_temoin.abs() < 4.0,
        true,
        sj(&d3),
        bloc_porte(4, 4, 2, None),
    )?;
    let d3m = portes::porte_d3(4, 4, 2, 2.3, th, me, 21, Mutant::PoidsIgnores).map_err(err)?;
    porte(
        run,
        &mut tout,
        "d3_mutant_poids_ignores",
        d3m.z.abs() > 4.0,
        true,
        sj(&d3m),
        bloc_porte(4, 4, 2, None),
    )?;
    // (e) couplage fort, l = 1 → 2
    let beta_e = 0.3;
    let pc = ParamsChaine {
        therm: 50,
        mes: if rapide { 2000 } else { 20000 },
        sep: 1,
        n_or: 1,
        blocs: 20,
    };
    let bal = portes::balayage(4, 4, 1, 2, beta_e, 5, pc, 31).map_err(err)?;
    let mut zmax: f64 = 0.0;
    let mut pts = Vec::new();
    for pt in &bal.points {
        let (pred, ordre1) =
            portes::prediction_couplage_fort(4, 4, 1, 2, beta_e, pt.alpha).map_err(err)?;
        let z = (pt.moyenne - pred) / pt.erreur;
        zmax = zmax.max(z.abs());
        pts.push(json!({"alpha": pt.alpha, "mesure": pt.moyenne, "erreur": pt.erreur, "prediction": pred, "ordre1": ordre1, "z": z}));
    }
    let zint = bal.delta_f / bal.erreur;
    porte(
        run,
        &mut tout,
        "e_couplage_fort",
        zmax < 4.0 && zint.abs() < 4.0,
        true,
        json!({"beta": beta_e, "points": pts, "delta_f": bal.delta_f, "erreur": bal.erreur, "z_integrale": zint, "n_plaquettes_diff": bal.n_plaquettes_diff}),
        bloc_porte(4, 4, 1, Some(2)),
    )?;
    bal.sante().map_err(Erreur::Validation)?;
    // (e2) S₂(1) = F(1) − F(0) : non nul dès l'ordre β² (β² L²/2)
    let b01 = portes::balayage(4, 4, 0, 1, beta_e, 9, pc, 37).map_err(err)?;
    let (pred, o1) = portes::prediction_df_couplage_fort(4, 4, 0, 1, beta_e).map_err(err)?;
    let z01 = (b01.delta_f - pred) / b01.erreur;
    porte(
        run,
        &mut tout,
        "e2_couplage_fort_S2_de_1",
        z01.abs() < 4.0 && (o1 - beta_e * beta_e * 16.0 / 2.0).abs() < 1e-12,
        true,
        json!({"beta": beta_e, "mesure": b01.delta_f, "erreur": b01.erreur, "prediction_boucles_independantes": pred,
               "ordre1_main_beta2L2_sur_2": o1, "z": z01, "trapezes": b01.delta_f_trapezes, "profil": sj(&b01.points)}),
        bloc_porte(4, 4, 0, Some(1)),
    )?;
    b01.sante().map_err(Erreur::Validation)?;
    run.ligne(&json!({"toutes_vertes": tout}))?;
    Ok(if tout { code::OK } else { code::VALIDATION })
}

/// `--largeurs` (relecture c19, bloquantes 1 et 3) : chaque largeur l vérifie l < L, contrôlé AVANT toute addition (l + 1 ne
/// déborde jamais), et aucune n'est répétée (deux largeurs égales tireraient les mêmes nombres : la largeur est un champ du flux).
fn valider_largeurs(l: usize, largeurs: &[usize]) -> Resultat<()> {
    for (i, &lw) in largeurs.iter().enumerate() {
        if lw >= l {
            return Err(Erreur::Config(format!(
                "--largeurs : largeur {lw} ≥ L = {l} ; chaque largeur l doit vérifier l + 1 ≤ L"
            )));
        }
        if largeurs[..i].contains(&lw) {
            return Err(Erreur::Config(format!("--largeurs : largeur {lw} répétée")));
        }
    }
    Ok(())
}

/// Clés des deux spécifications d'une largeur de `courbe` : `geometrie` / `geometrie_alpha1` pour une seule largeur, sinon
/// `geometrie_l<w>` / `geometrie_l<w>_alpha1`.
fn cles_courbe(n_largeurs: usize, lw: usize) -> (String, String) {
    if n_largeurs == 1 {
        ("geometrie".into(), "geometrie_alpha1".into())
    } else {
        (
            format!("geometrie_l{lw}"),
            format!("geometrie_l{lw}_alpha1"),
        )
    }
}

/// Balayage sur les spécifications inscrites au manifeste du run (`geometrie`, `geometrie_alpha1`).
pub fn lancer_balayage(p: &ParamsBalayage, run: &mut Run) -> Resultat<i32> {
    let (s0, s1) = (run.spec("geometrie")?, run.spec("geometrie_alpha1")?);
    let b = portes::balayage_specs(&s0, &s1, p.beta, p.nalpha, p.chaine, p.graine).map_err(err)?;
    run.ligne(&sj(&b))?;
    // filet (relecture c19b, bloquante 10) : la ligne est écrite pour diagnostic, puis le run est clos en échec (code 8)
    b.sante().map_err(Erreur::Validation)?;
    Ok(code::OK)
}

/// Relecture intrication-claude, objection 11 : pour lw + 1 = L, A₁ est toute la tranche spatiale (aire de bord nulle) et
/// C = (l+½)³ ΔS₂ / (2L²) n'a plus de sens ; avertissement sur stderr au lancement et dans la ligne de résultat.
fn avertissement_tranche_entiere(lw: usize) -> String {
    format!(
        "largeur {lw} → {} = L : A₁ est toute la tranche spatiale (aire de bord nulle) ; C = (l+½)³ ΔS₂ / (2L²) n'a pas de sens pour ce point",
        lw + 1
    )
}

pub fn lancer_courbe(p: &ParamsCourbe, run: &mut Run) -> Resultat<i32> {
    // largeurs < L et distinctes : l + 1 ci-dessous ne déborde pas (appel de bibliothèque compris)
    valider_largeurs(p.l, &p.largeurs)?;
    for &lw in &p.largeurs {
        let (c0, c1) = cles_courbe(p.largeurs.len(), lw);
        let (s0, s1) = (run.spec(&c0)?, run.spec(&c1)?);
        let b =
            portes::balayage_specs(&s0, &s1, p.beta, p.nalpha, p.chaine, p.graine).map_err(err)?;
        let x = lw as f64 + 0.5;
        let aire = 2.0 * (p.l as f64) * (p.l as f64);
        let c = x * x * x * b.delta_f / aire;
        let ec = x * x * x * b.erreur / aire;
        let mut ligne = json!({"largeur": lw, "l_milieu": x, "delta_s2": b.delta_f, "erreur": b.erreur, "C": c, "erreur_C": ec,
            "delta_s2_trapezes": b.delta_f_trapezes, "balayage": sj(&b)});
        if lw + 1 == p.l {
            ligne["avertissement"] = Value::String(avertissement_tranche_entiere(lw));
        }
        run.ligne(&ligne)?;
        b.sante().map_err(Erreur::Validation)?;
        eprintln!(
            "l = {lw} → {} : ΔS₂ = {:.4} ± {:.4}  C = {:.4} ± {:.4}  ({:.0} s)",
            lw + 1,
            b.delta_f,
            b.erreur,
            c,
            ec,
            b.secondes
        );
    }
    Ok(code::OK)
}

pub fn lancer_doublement(p: &ParamsDoublement, run: &mut Run) -> Resultat<i32> {
    let (s0, s1) = (run.spec("geometrie")?, run.spec("geometrie_alpha1")?);
    let d = portes::doublement_specs(&s0, &s1, p.beta, p.nalpha, p.nbeta, p.chaine, p.graine)
        .map_err(err)?;
    run.ligne(&sj(&d))?;
    d.sante().map_err(Erreur::Validation)?;
    Ok(if d.z.abs() < 4.0 {
        code::OK
    } else {
        code::VALIDATION
    })
}

/// Débit (secondes par pas) sur la géométrie interpolée l/2 → l/2 + 1. Relecture intrication-claude, objection 9 : la
/// géométrie passe par `specs` (capacités, débordement du volume) et `--pas` doit valoir au moins 1 (code 2).
pub fn debit(l: usize, t: usize, pas: usize, n_or: usize) -> Resultat<Value> {
    if pas == 0 {
        return Err(Erreur::Config(
            "debit : --pas = 0 ; au moins un pas chronométré requis".into(),
        ));
    }
    // relecture c19 (majeure 5) : un pas préparatoire + `pas` pas chronométrés < 2³² balayages ; `--or` ≤ 1000
    let pc = ParamsChaine {
        therm: 1,
        mes: pas,
        sep: 1,
        n_or,
        blocs: 2,
    };
    if portes::balayages_par_chaine(pc.therm, pc.mes, pc.sep).is_none() {
        return Err(Erreur::Config(format!(
            "debit : 1 + --pas {pas} : au plus 2³² − 1 balayages par chaîne (le compteur Philox de balayage ne reboucle pas)"
        )));
    }
    if n_or > portes::OR_MAX {
        return Err(Erreur::Config(format!(
            "--or = {n_or} : au plus {} surrelaxations par pas",
            portes::OR_MAX
        )));
    }
    let (s0, s1) = specs(l, t, l / 2, l / 2 + 1)?;
    garde_memoire(
        "debit",
        l,
        t,
        1,
        &ParamsChaine {
            mes: 0,
            blocs: 0,
            ..pc
        },
        1,
    )?;
    let flux = portes::plan_debit(l / 2).map_err(err)?;
    let mut res = Reseau::depuis_spec_flux(&s0, &s1, &flux, Mutant::Aucun).map_err(err)?;
    let geo = res.geo;
    res.beta = 2.3;
    res.fixer_alpha(0.5);
    res.init(true);
    res.pas(n_or);
    let t0 = std::time::Instant::now();
    for _ in 0..pas {
        res.pas(n_or);
    }
    let s = t0.elapsed().as_secs_f64() / pas as f64;
    if res.n_cap > 0 {
        return Err(Erreur::Validation(format!(
            "debit : plafond d'essais du bain thermique atteint {} fois (tirage dégénéré)",
            res.n_cap
        )));
    }
    Ok(
        json!({"l": l, "t": t, "liens": 4 * geo.v, "or": n_or, "secondes_par_pas": s, "fils": rayon::current_num_threads(),
        "liens_par_seconde": 4.0 * geo.v as f64 * (1.0 + n_or as f64) / s}),
    )
}

/// Exécute `intrication <sous-commande>` ; `m` = correspondances de la commande `intrication` elle-même.
pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom != NOM {
        return Err(Erreur::Config(format!(
            "module intrication : commande inconnue « {nom} »"
        )));
    }
    eprintln!("{AVERTISSEMENT_EXPERIMENTAL}");
    // UN seul contrôle des options (DOMAINE), avant l'admission mémoire et toute écriture
    valider(m)?;
    match m.subcommand() {
        Some(("portes", s)) => {
            let p = ParamsPortes {
                rapide: s.get_flag("rapide"),
            };
            // plusieurs géométries : pas de clé `geometrie` au niveau du run, un bloc par porte dans resultats.jsonl
            let run = Run::ouvrir(
                &sortie(s),
                "portes",
                &[],
                json!({"rapide": p.rapide, "geometries": "un bloc kvu-geometrie/1 par porte dans resultats.jsonl"}),
            )?;
            conduire(run, |run| lancer_portes(&p, run))
        }
        Some(("balayage", s)) => {
            let p = ParamsBalayage {
                l: u(s, "l"),
                t: u(s, "t"),
                l0: u(s, "l0"),
                l1: u(s, "l1"),
                beta: f(s, "beta"),
                nalpha: u(s, "nalpha"),
                graine: g(s),
                chaine: chaine(s),
            };
            let params = json!({"beta": p.beta, "nalpha": p.nalpha, "graine": p.graine, "chaine": chaine_json(&p.chaine)});
            garde_memoire(
                "balayage",
                p.l,
                p.t,
                portes::chaines_simultanees(p.nalpha),
                &p.chaine,
                p.nalpha,
            )?;
            let (s0, s1) = specs(p.l, p.t, p.l0, p.l1)?;
            let run = Run::ouvrir(
                &sortie(s),
                "balayage",
                &[("geometrie".into(), s0), ("geometrie_alpha1".into(), s1)],
                params,
            )?;
            conduire(run, |run| lancer_balayage(&p, run))
        }
        Some(("courbe", s)) => {
            // admission AVANT toute collecte proportionnelle aux entrées : nombre de largeurs compté sans allocation
            let n_largeurs = s.get_many::<usize>("largeurs").map_or(0, |v| v.count());
            let ch = chaine(s);
            garde_memoire(
                "courbe",
                u(s, "l"),
                u(s, "t"),
                portes::chaines_simultanees(u(s, "nalpha")),
                &ch,
                u(s, "nalpha").saturating_mul(n_largeurs),
            )?;
            let p = ParamsCourbe {
                l: u(s, "l"),
                t: u(s, "t"),
                beta: f(s, "beta"),
                largeurs: s
                    .get_many::<usize>("largeurs")
                    .expect("--largeurs")
                    .copied()
                    .collect(),
                nalpha: u(s, "nalpha"),
                graine: g(s),
                chaine: chaine(s),
            };
            let params = json!({"beta": p.beta, "nalpha": p.nalpha, "graine": p.graine, "chaine": chaine_json(&p.chaine),
                "aire_bord": "2L² (deux faces)", "C": "(l+½)³ ΔS₂ / (2L²)"});
            for &lw in p.largeurs.iter().filter(|&&lw| lw + 1 == p.l) {
                eprintln!(
                    "kvu intrication courbe : avertissement : {}",
                    avertissement_tranche_entiere(lw)
                );
            }
            // spécifications construites APRÈS l'admission (relecture c19b, majeure 11 : rien de proportionnel aux entrées avant)
            let mut geos = Vec::new();
            for &lw in &p.largeurs {
                let (s0, s1) = specs(p.l, p.t, lw, lw + 1)?;
                let (c0, c1) = cles_courbe(p.largeurs.len(), lw);
                geos.push((c0, s0));
                geos.push((c1, s1));
            }
            let run = Run::ouvrir(&sortie(s), "courbe", &geos, params)?;
            conduire(run, |run| lancer_courbe(&p, run))
        }
        Some(("doublement", s)) => {
            let p = ParamsDoublement {
                l: u(s, "l"),
                t: u(s, "t"),
                beta: f(s, "beta"),
                nalpha: u(s, "nalpha"),
                nbeta: u(s, "nbeta"),
                graine: g(s),
                chaine: chaine(s),
            };
            let params = json!({"beta": p.beta, "nalpha": p.nalpha, "nbeta": p.nbeta, "graine": p.graine, "chaine": chaine_json(&p.chaine)});
            garde_memoire(
                "doublement",
                p.l,
                p.t,
                portes::chaines_simultanees(p.nalpha.max(p.nbeta)),
                &p.chaine,
                p.nalpha.saturating_add(p.nbeta),
            )?;
            let (s0, s1) = specs(p.l, p.t, 0, p.l)?;
            let run = Run::ouvrir(
                &sortie(s),
                "doublement",
                &[("geometrie".into(), s0), ("geometrie_alpha1".into(), s1)],
                params,
            )?;
            conduire(run, |run| lancer_doublement(&p, run))
        }
        Some(("debit", s)) => {
            println!("{}", debit(u(s, "l"), u(s, "t"), u(s, "pas"), u(s, "or"))?);
            Ok(code::OK)
        }
        _ => Err(Erreur::Config(
            "intrication : sous-commande manquante (portes, balayage, courbe, doublement, debit)"
                .into(),
        )),
    }
}
