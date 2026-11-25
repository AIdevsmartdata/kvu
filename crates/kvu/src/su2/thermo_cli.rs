//! `kvu su2 thermo mesure | analyse` : pilote de la méthode intégrale sur boîtes rectangulaires (campagne « une boite »).
//!
//! ```text
//! su2 thermo mesure --cotes l0,l1,l2,l3 [--torsion 01|aucune] [--poids -1,1] --betas b1,b2,… --repliques R --therm N --nmeas M
//!                   [--skip 1] [--nor 2] [--graine G] [--depart alterne|froid|chaud] [--backend cuda|hip] [--chaines-max 512]
//!                   --sortie DIR [--predictions FICHIER --sha256 H]
//! su2 thermo analyse --entree DIR --mode torsion [--poids-tordu -1] | --mode a3 --entree-cube DIR2  [--quadrature parabole|trapeze] [--json F]
//! ```
//! `mesure` : un lot GPU par bras (poids de feuille) et par groupe de nœuds ; la graine de la chaîne (nœud k, réplique i) est G + k·R + i
//! pour TOUS les bras : mode APPARIÉ. Un groupe déjà écrit (`points/<bras>_<groupe>.csv`) est sauté à la reprise. Le GPU doit être
//! pris sous le verrou de la machine (`flock`) par l'appelant, comme `su2 boite`.

use super::thermo::{self, PointRep, Quadrature};
use super::torsion::Torsion;
use super::une_boite::{ParamBoite, UneBoite};
use super::ParamChaine;
use crate::commun::{Depart, Erreur, Graine, Precision, Resultat};
use crate::heritage::su2_hb::g17;
use clap::{Arg, ArgMatches, Command};
use std::path::{Path, PathBuf};

fn arg(nom: &'static str, aide: &'static str) -> Arg {
    Arg::new(nom)
        .long(nom)
        .help(aide)
        .num_args(1)
        .allow_hyphen_values(true)
}

pub fn thermo() -> Command {
    Command::new("thermo")
        .about("Méthode intégrale sur boîtes rectangulaires : `mesure` (GPU F64, S_tot et S_feuille par réplique) et `analyse` (ln R, a₃)")
        .subcommand_required(true)
        .subcommand(
            Command::new("mesure")
                .arg(arg("cotes", "l0,l1,l2,l3 (pairs ; impairs avec --impairs oui) ; avec --cotes-b : la PETITE boîte A de la paire"))
                .arg(arg("cotes-b", "seconde boîte B (volume = 2 × celui de A) : A et B sont mesurées par le MÊME moteur, la même précision, la même exécution, en tranches alternées (dossiers sortie/A et sortie/B)"))
                .arg(arg("decalage", "v0,v1,v2,v3 : COUTURE décalée (tore oblique) de la boîte A (ou de la boîte unique) : le tour de l'axe --decalage-axe ramène le champ décalé de v (entiers relatifs ; composante selon l'axe nulle ; mêmes conventions que `su2 gen --decalage`). Absent, nul ou d'une période entière : réseau ordinaire, sorties inchangées au bit. Sans torsion ni --paire-n"))
                .arg(arg("decalage-b", "couture décalée de la boîte B (avec --cotes-b), même convention que --decalage"))
                .arg(arg("decalage-axe", "axe recollé de la couture (0..3)").default_value("0"))
                .arg(arg("paire-n", "raccourci : A = (n, m·n, m·n, m·n), B = (2n, m·n, m·n, m·n) (a₃ = 2 ln Z(A) − ln Z(B)) ; exige --paire-m").conflicts_with_all(["cotes", "cotes-b"]))
                .arg(arg("paire-m", "aspect m de --paire-n (2 : B est le cube (2n)⁴)"))
                .arg(arg("torsion", "« aucune » ou plans (ex. 01)").default_value("aucune"))
                .arg(arg("torsion-fixe", "« aucune » ou plans (ex. 01) dont le coefficient reste −1 quel que soit --poids (pile « masque fixe » ; hors S_feuille) ; --torsion garde les plans à poids variable ; un plan dans les deux listes est refusé").default_value("aucune"))
                .arg(Arg::new("series").long("series").action(clap::ArgAction::SetTrue).help("écrit aussi les séries PAR MESURE de S_feuille et S_tot : series/bras<a>_g<ggg>_fe.bin et _tot.bin (f64 petit-boutiste, [mesure][chaîne du groupe], chaîne = k·R + i comme les lignes de points/*.csv)"))
                .arg(Arg::new("lien-integre").long("lien-integre").action(clap::ArgAction::SetTrue).help("OPTION (éteinte par défaut ; colonnes et séries actuelles inchangées au bit) : ajoute les estimateurs « à lien intégré » (multi-hit exact de Parisi–Petronzio–Rapuano, 1983) : colonnes y0_t, ys_t, zs_s, y0_tot, ys_tot des points/*.csv et séries series/*_{y0t,yst,zss,y0tot,ystot}.bin (Y0 : liens temporels seuls ; Ys : liens spatiaux seuls ; deux estimateurs de S_t) ; même espérance que l'estimateur brut. Sans torsion ni couture (S_feuille à lien intégré : non fait). Module GPU à part (+ su2_lien.cu)"))
                .arg(arg("poids", "poids de feuille des bras, séparés par des virgules (ex. -1,1)").default_value("1"))
                .arg(arg("betas", "nœuds β > 0, croissants (avec --beta-s : les couplages TEMPORELS β_t, ≥ 0, croissants)").required(true))
                .arg(arg("beta-s", "OPTION : couplage des plaquettes d'ESPACE, fixe (> 0) ; --betas devient la liste des couplages TEMPORELS (plaquettes 0i, direction 0 = premier côté). Poids exp[β_s·S_s + β_t·S_t], S = Σ ½ReTr U_p. Ajoute S_t et S_s aux sorties. Sans torsion (refusé avec --torsion, --torsion-fixe, --poids ≠ 1). Un lot par nœud : prendre --repliques grand (≥ 64) pour le débit"))
                .arg(arg("repliques", "répliques par nœud").required(true))
                .arg(arg("therm", "pas jetés").required(true))
                .arg(arg("nmeas", "mesures").required(true))
                .arg(arg("skip", "pas entre mesures").default_value("1"))
                .arg(arg("nor", "surrelaxations par pas").default_value("2"))
                .arg(arg("graine", "graine de base").default_value("1"))
                .arg(arg("depart", "alterne | froid | chaud").default_value("alterne"))
                .arg(arg("backend", "cuda | hip").default_value("cuda"))
                .arg(arg("precision", "f64 (référence) | f32 (rapide : noyaux float, sommes F64)").default_value("f64"))
                .arg(arg("chaines-max", "chaînes au plus par lot GPU").default_value("512"))
                .arg(arg("impairs", "oui | non : côtés IMPAIRS admis (coloriage du balayage à trois classes ; côtés pairs : sorties inchangées au bit)").default_value("non"))
                .arg(arg("sortie", "répertoire de sortie").required(true))
                .arg(arg("predictions", "fichier de prédictions écrit AVANT la mesure"))
                .arg(arg("sha256", "sha256 annoncé de ce fichier (obligatoire avec --predictions)")),
        )
        .subcommand(
            Command::new("analyse")
                .arg(arg("entree", "dossier de `mesure`").required(true))
                .arg(arg("mode", "torsion | a3").required(true))
                .arg(arg("poids-tordu", "poids du bras tordu (mode torsion)").default_value("-1"))
                .arg(arg("entree-cube", "dossier du cube (mode a3 : volume du cube = 2 × volume de la boîte)"))
                .arg(arg("quadrature", "parabole | trapeze").default_value("parabole"))
                .arg(arg("json", "fichier de sortie JSON")),
        )
}

fn texte<'a>(m: &'a ArgMatches, n: &str) -> Option<&'a str> {
    m.get_one::<String>(n).map(|s| s.as_str())
}
fn requis<'a>(m: &'a ArgMatches, n: &str) -> Resultat<&'a str> {
    texte(m, n).ok_or_else(|| Erreur::Config(format!("--{n} requis")))
}
fn liste<T: std::str::FromStr>(s: &str, n: &str) -> Resultat<Vec<T>> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<T>()
                .map_err(|_| Erreur::Config(format!("--{n} : « {t} » invalide")))
        })
        .collect()
}
fn un<T: std::str::FromStr>(m: &ArgMatches, n: &str) -> Resultat<T> {
    let t = requis(m, n)?;
    t.parse::<T>()
        .map_err(|_| Erreur::Config(format!("--{n} : « {t} » invalide")))
}

pub fn executer(m: &ArgMatches) -> Resultat<i32> {
    match m.subcommand() {
        Some(("mesure", g)) => mesure(g),
        Some(("analyse", g)) => analyse(g),
        _ => Err(Erreur::Config("thermo : mesure ou analyse".into())),
    }
}

const ENTETE: &str = "poids,beta,rep,graine,depart,n,tot,fe,h1_tot,h1_fe,h2_tot,h2_fe";
/// Avec --beta-s : mêmes 12 premières colonnes (beta = β_t, fe = S_t) puis beta_s, st (moyenne de S_t), ss (moyenne de S_s = tot − fe).
/// Suffixe ajouté par --lien-integre (moyennes par réplique des estimateurs à lien intégré).
const SUFFIXE_LI: &str = ",y0_t,ys_t,zs_s,y0_tot,ys_tot";
const ENTETE_ST: &str = "poids,beta,rep,graine,depart,n,tot,fe,h1_tot,h1_fe,h2_tot,h2_fe,beta_s,st,ss";

fn ligne(poids: f64, p: &PointRep) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{}",
        g17(poids),
        g17(p.beta),
        p.rep,
        p.graine,
        p.depart,
        p.n,
        g17(p.tout[0]),
        g17(p.tout[1]),
        g17(p.h1[0]),
        g17(p.h1[1]),
        g17(p.h2[0]),
        g17(p.h2[1])
    )
}

/// `--series` : `series/bras<a>_g<ggg>_{fe,tot}.bin`, f64 petit-boutiste, [mesure][chaîne du groupe] (chaîne = k·R + i, l'ordre des
/// lignes de `points/bras<a>_g<ggg>.csv`) ; écrit AVANT le csv (le csv fait foi pour la reprise).
fn ecrire_series(dossier: &Path, ia: usize, ig: usize, series: &[Vec<[f64; 2]>], st: bool, li: Option<&[Vec<[f64; 3]>]>) -> Resultat<()> {
    let d = dossier.join("series");
    std::fs::create_dir_all(&d).map_err(|e| Erreur::io("création de series/", e))?;
    let nm = series.first().map_or(0, |s| s.len());
    let noms: &[&str] = if st { &["tot", "fe", "st", "ss"] } else { &["tot", "fe"] };
    for nom in noms.iter() {
        let mut oct = Vec::with_capacity(8 * nm * series.len());
        for t in 0..nm {
            for s in series {
                let x = match *nom {
                    "tot" => s[t][0],
                    "fe" | "st" => s[t][1],
                    _ => s[t][0] - s[t][1],
                };
                oct.extend_from_slice(&x.to_le_bytes());
            }
        }
        let f = d.join(format!("bras{ia}_g{ig:03}_{nom}.bin"));
        let tmp = f.with_extension("tmp");
        std::fs::write(&tmp, oct).map_err(|e| Erreur::io("écriture des séries", e))?;
        std::fs::rename(&tmp, &f).map_err(|e| Erreur::io("renommage des séries", e))?;
    }
    if let Some(li) = li {
        for (k, nom) in ["y0t", "yst", "zss", "y0tot", "ystot"].iter().enumerate() {
            let mut oct = Vec::with_capacity(8 * nm * li.len());
            for t in 0..nm {
                for s in li {
                    let x = match k {
                        0..=2 => s[t][k],
                        3 => s[t][0] + s[t][2],
                        _ => s[t][1] + s[t][2],
                    };
                    oct.extend_from_slice(&x.to_le_bytes());
                }
            }
            let f = d.join(format!("bras{ia}_g{ig:03}_{nom}.bin"));
            let tmp = f.with_extension("tmp");
            std::fs::write(&tmp, oct).map_err(|e| Erreur::io("écriture des séries", e))?;
            std::fs::rename(&tmp, &f).map_err(|e| Erreur::io("renommage des séries", e))?;
        }
    }
    Ok(())
}

fn lire_points(dossier: &Path) -> Resultat<Vec<(f64, PointRep)>> {
    let mut sortie = vec![];
    let pts = dossier.join("points");
    let mut fichiers: Vec<PathBuf> = std::fs::read_dir(&pts)
        .map_err(|e| Erreur::io(format!("lecture de {}", pts.display()), e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "csv"))
        .collect();
    fichiers.sort();
    for f in fichiers {
        let t = std::fs::read_to_string(&f)
            .map_err(|e| Erreur::io(format!("lecture de {}", f.display()), e))?;
        let mut l = t.lines();
        let h = l.next();
        if !h.is_some_and(|x| x.starts_with(ENTETE)) {
            return Err(Erreur::Config(format!(
                "{} : en-tête inattendu",
                f.display()
            )));
        }
        for (i, ln) in l.enumerate() {
            let c: Vec<&str> = ln.split(',').collect();
            let f64_ = |k: usize| {
                c.get(k)
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|x| x.is_finite())
            };
            let (Some(poids), Some(beta), Some(tot), Some(fe), Some(a), Some(b), Some(cc), Some(d)) = (
                f64_(0),
                f64_(1),
                f64_(6),
                f64_(7),
                f64_(8),
                f64_(9),
                f64_(10),
                f64_(11),
            ) else {
                return Err(Erreur::Config(format!(
                    "{} ligne {} : champ illisible ou non fini",
                    f.display(),
                    i + 2
                )));
            };
            let rep = c
                .get(2)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| Erreur::Config(format!("{} ligne {} : rep", f.display(), i + 2)))?;
            let graine = c.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let n = c.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
            sortie.push((
                poids,
                PointRep {
                    beta,
                    rep,
                    graine,
                    depart: c.get(4).unwrap_or(&"").to_string(),
                    n,
                    tout: [tot, fe],
                    h1: [a, b],
                    h2: [cc, d],
                },
            ));
        }
    }
    Ok(sortie)
}

#[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
fn mesure(m: &ArgMatches) -> Resultat<i32> {
    let quatre = |v: Vec<usize>, n: &str| -> Resultat<[usize; 4]> {
        v.try_into()
            .map_err(|_| Erreur::Config(format!("--{n} : quatre côtés")))
    };
    let (cotes, cotes_b): ([usize; 4], Option<[usize; 4]>) = if texte(m, "paire-n").is_some() {
        let (n, mm): (usize, usize) = (un(m, "paire-n")?, un(m, "paire-m")?);
        // produits contrôlés : une entrée extrême est une erreur de configuration, pas un panic ni un débordement silencieux
        let trop_grand = || Erreur::Config(format!("--paire-n {n} --paire-m {mm} : dimensions hors de portée (débordement)"));
        let mn = mm.checked_mul(n).ok_or_else(trop_grand)?;
        let deux_n = n.checked_mul(2).ok_or_else(trop_grand)?;
        ([n, mn, mn, mn], Some([deux_n, mn, mn, mn]))
    } else {
        let a = quatre(liste(requis(m, "cotes")?, "cotes")?, "cotes")?;
        let b = texte(m, "cotes-b")
            .map(|t| liste::<usize>(t, "cotes-b").and_then(|v| quatre(v, "cotes-b")))
            .transpose()?;
        (a, b)
    };
    if let Some(b) = cotes_b {
        // B2-14 : volumes contrôlés (chaque côté représentable ne suffit pas : le produit peut déborder)
        let volume = |c: &[usize; 4]| c.iter().try_fold(1usize, |p, &x| p.checked_mul(x));
        let (va, vb) = match (volume(&cotes), volume(&b)) {
            (Some(va), Some(vb)) if va.checked_mul(2).is_some() => (va, vb),
            _ => return Err(Erreur::Config(format!("paire {cotes:?} / {b:?} : volume hors de portée (débordement)"))),
        };
        if vb != 2 * va {
            return Err(Erreur::Config(format!("paire {cotes:?} / {b:?} : le volume de B doit valoir 2 × celui de A (constantes e^(−6Vβ) de a₃)")));
        }
    }
    let axe_couture: usize = un(m, "decalage-axe")?;
    let decalage = |nom: &str| -> Resultat<Option<(usize, [i64; 4])>> {
        match texte(m, nom) {
            None => Ok(None),
            Some(t) => {
                let v: Vec<i64> = liste(t, nom)?;
                if v.len() != 4 || axe_couture > 3 {
                    return Err(Erreur::Config(format!("--{nom} : 4 entiers (v0,v1,v2,v3) et --decalage-axe 0..3 attendus")));
                }
                Ok(Some((axe_couture, [v[0], v[1], v[2], v[3]])))
            }
        }
    };
    let (couture_a, couture_b) = (decalage("decalage")?, decalage("decalage-b")?);
    if (couture_a.is_some() || couture_b.is_some()) && texte(m, "paire-n").is_some() {
        return Err(Erreur::Config("--decalage : avec --cotes (et --cotes-b), pas avec --paire-n".into()));
    }
    if couture_b.is_some() && cotes_b.is_none() {
        return Err(Erreur::Config("--decalage-b exige --cotes-b".into()));
    }
    let jobs: Vec<([usize; 4], PathBuf, Option<(usize, [i64; 4])>)> = match cotes_b {
        None => vec![(cotes, PathBuf::from(requis(m, "sortie")?), couture_a)],
        Some(b) => vec![
            (cotes, PathBuf::from(requis(m, "sortie")?).join("A"), couture_a),
            (b, PathBuf::from(requis(m, "sortie")?).join("B"), couture_b),
        ],
    };
    let torsion = Torsion::analyser(requis(m, "torsion")?)?;
    let fixe = Torsion::analyser(requis(m, "torsion-fixe")?)?;
    let series_on = m.get_flag("series");
    let li_on = m.get_flag("lien-integre");
    if li_on && (!torsion.est_triviale() || !fixe.est_triviale() || texte(m, "decalage").is_some() || texte(m, "decalage-b").is_some()) {
        return Err(Erreur::Config("--lien-integre est refusé avec --torsion, --torsion-fixe ou --decalage (S_feuille à lien intégré : non fait)".into()));
    }
    for &(a, b) in &Torsion::PLANS {
        if fixe.tordu(a, b) && torsion.tordu(a, b) {
            return Err(Erreur::Config(format!(
                "plan {a}{b} à la fois dans --torsion-fixe et --torsion : refusé"
            )));
        }
    }
    let poids: Vec<f64> = liste(requis(m, "poids")?, "poids")?;
    let betas: Vec<f64> = liste(requis(m, "betas")?, "betas")?;
    let beta_s: Option<f64> = texte(m, "beta-s")
        .map(|_| un::<f64>(m, "beta-s"))
        .transpose()?;
    if let Some(bs) = beta_s {
        if !(bs.is_finite() && bs > 0.0) {
            return Err(Erreur::Config("--beta-s : réel fini > 0 requis".into()));
        }
        if !torsion.est_triviale() || !fixe.est_triviale() || poids != [1.0] {
            return Err(Erreur::Config("--beta-s (couplages d'espace et de temps séparés) est refusé avec --torsion, --torsion-fixe ou --poids ≠ 1 : la feuille de torsion et la feuille temporelle se partagent le même masque ; mesurer la torsion en isotrope".into()));
        }
        if betas.iter().any(|&b| !(b.is_finite() && b >= 0.0))
            || betas.windows(2).any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
        {
            return Err(Erreur::Config("--betas avec --beta-s : couplages TEMPORELS ≥ 0 strictement croissants".into()));
        }
    } else if betas
        .iter()
        .any(|&b| b.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
        || betas
            .windows(2)
            .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
    {
        return Err(Erreur::Config(
            "--betas : nœuds > 0 strictement croissants (β = 0 est ajouté à l'analyse)".into(),
        ));
    }
    let (r, therm, nmeas, skip, nor): (usize, usize, usize, usize, usize) = (
        un(m, "repliques")?,
        un(m, "therm")?,
        un(m, "nmeas")?,
        un(m, "skip")?,
        un(m, "nor")?,
    );
    if r < 2 || nmeas < 2 || nmeas % 2 != 0 {
        return Err(Erreur::Config(
            "--repliques ≥ 2 et --nmeas pair ≥ 2 requis (moitiés, jackknife)".into(),
        ));
    }
    let g0: u64 = un(m, "graine")?;
    let max: usize = un(m, "chaines-max")?;
    let depart = requis(m, "depart")?;
    if !["alterne", "froid", "chaud"].contains(&depart) {
        return Err(Erreur::Config("--depart : alterne | froid | chaud".into()));
    }
    let pred = match (texte(m, "predictions"), texte(m, "sha256")) {
        (Some(f), Some(h)) => Some((
            f.to_string(),
            thermo::verifier_predictions(Path::new(f), h)?,
        )),
        (None, None) => None,
        _ => {
            return Err(Erreur::Config(
                "--predictions et --sha256 vont ensemble".into(),
            ))
        }
    };
    let impairs = match requis(m, "impairs")? {
        "oui" => true,
        "non" => false,
        autre => return Err(Erreur::Config(format!("--impairs « {autre} » : oui | non"))),
    };
    let precision = match requis(m, "precision")? {
        "f64" => Precision::F64,
        "f32" => Precision::F32,
        autre => {
            return Err(Erreur::Config(format!(
                "--precision « {autre} » inconnue (f64|f32)"
            )))
        }
    };
    for (c, d, cou) in &jobs {
        let pv = ParamBoite {
            cotes: *c,
            torsion,
            poids: 1.0,
            nor,
            impairs,
            chaines: vec![ParamChaine {
                beta: 1.0,
                graine: Graine(1),
                depart: Depart::Froid,
            }],
            precision,
            aniso: None,
            couture: *cou,
        };
        pv.valider()?;
        if cou.is_some() {
            if !torsion.est_triviale() || !fixe.est_triviale() {
                return Err(Erreur::Config("--decalage : sans --torsion ni --torsion-fixe".into()));
            }
            super::une_boite::tables_boite(&pv)?; // coloriage de la couture vérifié côté hôte, avant tout GPU
        }
        std::fs::create_dir_all(d.join("points"))
            .map_err(|e| Erreur::io("création de la sortie", e))?;
    }
    let session = format!(
        "{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
            ^ ((std::process::id() as u64) << 40)
    );
    let app = super::cli::ouvrir_backend(requis(m, "backend")?)?
        .ok_or_else(|| Erreur::Config("thermo mesure exige un GPU (--backend cuda|hip)".into()))?;
    // --beta-s : un lot par nœud (les poids de plans sont propres à la boîte, le β de chaîne ne peut pas les faire varier).
    let par_groupe = if beta_s.is_some() { 1 } else { (max / r).max(1) };
    let t0 = std::time::Instant::now();
    let mut empreinte = String::new();
    let mut plafonds = 0u64;
    for (ia, &w) in poids.iter().enumerate() {
        for (ig, groupe) in betas.chunks(par_groupe).enumerate() {
            for (cotes_j, dossier, couture_j) in &jobs {
                let fichier = dossier
                    .join("points")
                    .join(format!("bras{ia}_g{ig:03}.csv"));
                if fichier.exists() {
                    eprintln!("thermo : {} déjà écrit, sauté (reprise)", fichier.display());
                    continue;
                }
                let k0 = ig * par_groupe;
                let mut chaines = vec![];
                let mut meta = vec![];
                for (k, &b) in groupe.iter().enumerate() {
                    for i in 0..r {
                        let graine = g0 + ((k0 + k) * r + i) as u64;
                        let d = match depart {
                            "froid" => Depart::Froid,
                            "chaud" => Depart::Chaud,
                            _ => {
                                if i % 2 == 0 {
                                    Depart::Froid
                                } else {
                                    Depart::Chaud
                                }
                            }
                        };
                        chaines.push(ParamChaine {
                            beta: beta_s.unwrap_or(b),
                            graine: Graine(graine),
                            depart: d,
                        });
                        meta.push((
                            b,
                            i,
                            graine,
                            if d == Depart::Chaud { "chaud" } else { "froid" },
                        ));
                    }
                }
                let pb = ParamBoite {
                        cotes: *cotes_j,
                        torsion,
                        poids: w,
                        nor,
                        chaines,
                        precision,
                        impairs,
                        aniso: None,
                        couture: *couture_j,
                };
                let mut moteur = if li_on {
                    UneBoite::nouveau_li(&app, &pb, beta_s.map(|bs| (1.0, groupe[0] / bs)))?
                } else if let Some(bs) = beta_s {
                    // β de chaîne = β_s, poids d'espace 1, poids de temps β_t/β_s (= 1 exactement si β_t = β_s)
                    UneBoite::nouveau_st(&app, &pb, 1.0, groupe[0] / bs)?
                } else if fixe.est_triviale() {
                    UneBoite::nouveau(&app, &pb)?
                } else {
                    UneBoite::nouveau_fixe(&app, &pb, &fixe)?
                };
                empreinte = moteur.empreinte_source.clone();
                moteur.init()?;
                let (series, series_li) = moteur.serie_li(therm, nmeas, skip)?;
                plafonds += moteur.compteurs()?.iter().map(|c| c.0).sum::<u64>();
                if series_on {
                    ecrire_series(dossier, ia, ig, &series, beta_s.is_some(), series_li.as_deref())?;
                }
                let mut txt = format!(
                    "{}{}\n",
                    if beta_s.is_some() { ENTETE_ST } else { ENTETE },
                    if li_on { SUFFIXE_LI } else { "" }
                );
                for (ir, ((b, i, gr, dep), s)) in meta.iter().zip(&series).enumerate() {
                    if s.iter().any(|x| !x[0].is_finite() || !x[1].is_finite()) {
                        return Err(Erreur::Config(
                            "valeur non finie dans une série : run arrêté".into(),
                        ));
                    }
                    txt.push_str(&ligne(w, &thermo::resumer(*b, *i, *gr, dep, s)));
                    if let Some(bs) = beta_s {
                        let nn = s.len() as f64;
                        let st = s.iter().map(|x| x[1]).sum::<f64>() / nn;
                        let ss = s.iter().map(|x| x[0] - x[1]).sum::<f64>() / nn;
                        txt.push_str(&format!(",{},{},{}", g17(bs), g17(st), g17(ss)));
                    }
                    if let Some(sl) = &series_li {
                        let nn = sl[ir].len() as f64;
                        let m3: Vec<f64> = (0..3).map(|k| sl[ir].iter().map(|x| x[k]).sum::<f64>() / nn).collect();
                        if m3.iter().any(|x| !x.is_finite()) {
                            return Err(Erreur::Config("valeur non finie dans une série à lien intégré : run arrêté".into()));
                        }
                        txt.push_str(&format!(",{},{},{},{},{}", g17(m3[0]), g17(m3[1]), g17(m3[2]), g17(m3[0] + m3[2]), g17(m3[1] + m3[2])));
                    }
                    txt.push('\n');
                }
                let tmp = fichier.with_extension("tmp");
                std::fs::write(&tmp, txt).map_err(|e| Erreur::io("écriture du groupe", e))?;
                std::fs::rename(&tmp, &fichier)
                    .map_err(|e| Erreur::io("renommage du groupe", e))?;
                eprintln!(
                    "thermo : bras {ia} (poids {w}) groupe {ig} : {} chaînes, {:.1} s",
                    series.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }
    for (cotes_j, dossier, couture_j) in &jobs {
        let mut man = serde_json::json!({
            "schema": "kvu-thermo-mesure/1",
            "precision": requis(m, "precision")?, "session": session, "paire": cotes_b.is_some(), "cotes": cotes_j, "torsion": torsion.texte(), "poids": poids, "betas": betas, "repliques": r,
            "therm": therm, "nmeas": nmeas, "skip": skip, "nor": nor, "graine_base": g0, "depart": depart,
            "pile": "coin (l_mu - 1, l_nu - 1) de chaque plan tordu (docs/TORSION.md)",
            "appariement": "la graine de la chaîne (nœud k, réplique i) est G + k·R + i pour tous les bras",
            "noyaux": "su2_commun.h + su2_ref64.cu + su2_geo.cu + su2_feuille.cu", "sha256_source_assemblee": empreinte,
            "plafonds_du_bain": plafonds, "secondes": t0.elapsed().as_secs_f64(),
            "predictions": pred.as_ref().map(|(f, h)| serde_json::json!({"fichier": f, "sha256": h})),
            "argv": std::env::args().collect::<Vec<_>>(),
        });
        if let Some((axe, vec)) = couture_j {
            man["couture"] = serde_json::json!({"axe": axe, "decalage": vec, "sens": "voisin haut du site (l_axe-1, y) = (0, y + decalage) ; Z = Tr(S^decalage T^l_axe)"});
        }
        if !fixe.est_triviale() {
            man["torsion_fixe"] = serde_json::json!(fixe.texte());
        }
        if li_on {
            man["lien_integre"] = serde_json::json!({
                "estimateur": "multi-hit exact de Parisi-Petronzio-Rapuano (1983) : E[U_l | agrafes] = [I2(k)/I1(k)] V_l^dag, k_l = beta_chaine*|A_l|, A_l = somme PONDEREE (poids de plan) des six agrafes du lien, V_l = A_l/|A_l|",
                "formule_tous_liens": "(S_t : (Y0 + Ys)/2 ; NON ecrit seul : voir colonnes) S_E^li = 1/4 sum_l [I2/I1](k_l) * (1/2)Tr(V_l^dag B_l^E), B_l^E = somme NON ponderee des agrafes de l dont la plaquette est dans E ; meme esperance que S_E",
                "colonnes_csv": "y0_t, ys_t, zs_s, y0_tot, ys_tot : moyennes par replique sur les nmeas mesures. y0_t = Y0 = (1/2) sum_{liens temporels} [I2/I1](k) <A,B^t>/|A| (S_t par les SEULS liens temporels, exactement 0 a beta_t = 0) ; ys_t = Ys = meme somme sur les liens SPATIAUX (deux estimateurs sans biais distincts de S_t, direction du temps = 0, plans 0i) ; zs_s = (1/4) sum_{liens spatiaux} (S_s, plans spatiaux ; seul estimateur : aucun lien temporel n'est dans une plaquette spatiale) ; y0_tot = y0_t + zs_s ; ys_tot = ys_t + zs_s (sommes coherentes pour S_tot). L'estimateur tous-liens (Y0+Ys)/2 se reconstruit ; a combiner a l'analyse (combinaison optimale {brut, Y0, Ys}). Disponibles aussi sans --beta-s (isotrope)",
                "series": "series/bras<a>_g<ggg>_{y0t,yst,zss,y0tot,ystot}.bin : f64 petit-boutiste, meme ordre que _tot.bin ; mesurees sur les MEMES configurations que _tot.bin",
                "ratio_bessel": "fraction continue descendante, N = 12 + 1.5k + 6 sqrt(k) termes, erreur relative < 3e-16 pour k <= 300 (verifiee contre mpmath) ; k < 1e-8 : k/4 ; F64 : double ; F32 : fraction continue en float (N = 6 + 0.7k + 2.5 sqrt(k), erreur ~1e-7), agrafes en double (champ float promu), sommes F64",
                "non_fait": "S_feuille a lien integre (torsion) : refuse avec --torsion, --torsion-fixe, --decalage",
            });
        }
        if let Some(bs) = beta_s {
            man["couplages_separes"] = serde_json::json!({
                "beta_s": bs, "betas_t": betas, "direction_du_temps": 0,
                "poids": "exp[beta_s * S_s + beta_t * S_t] (beta > 0 favorise plaquette = 1) ; chaine : beta_chaine = beta_s, poids de plan espace = 1, temps = beta_t/beta_s",
                "S_t": "somme des demi-traces ReTr/2 U_p des SEULES plaquettes temporelles (plans 01, 02, 03 : direction 0 = premier cote), brute, non signee, sans torsion",
                "S_s": "somme des demi-traces des plaquettes spatiales (plans 12, 13, 23) = S_tot - S_t (par difference au niveau de la mesure)",
                "S_tot": "S_t + S_s (tous les plans)",
                "csv": "colonne beta = beta_t ; fe = S_t ; beta_s, st (moyenne de S_t), ss (moyenne de S_s) ajoutees",
                "derivee": "d ln Z / d beta_t = E[S_t] (beta_s fixe) ; d ln Z / d beta_s = E[S_s]",
                "lot": "un lot GPU par noeud (poids de plans propres a la boite)",
            });
        }
        if series_on {
            man["series"] = serde_json::json!({
                "fichiers": "series/bras<a>_g<ggg>_{fe,tot}.bin",
                "format": "f64 petit-boutiste, [mesure][chaine du groupe], chaine = k*R + i (ordre des lignes de points/bras<a>_g<ggg>.csv)",
                "nmeas": nmeas, "repliques": r, "betas_par_groupe": par_groupe,
                "st_ss": "avec --beta-s : fichiers _st.bin (= _fe.bin : S_t) et _ss.bin (S_s = tot - fe) en plus ; meme format",
                "fe": "S_feuille brute (somme de 1/2 ReTr U_p sur la pile à poids variable)", "tot": "S_tot brute (toutes plaquettes)",
            });
        }
        std::fs::write(
            dossier.join("manifest_thermo.json"),
            serde_json::to_string_pretty(&man).unwrap(),
        )
        .map_err(|e| Erreur::io("manifeste", e))?;
    }
    if plafonds != 0 {
        return Err(Erreur::Config(format!(
            "{plafonds} tirages du bain ont atteint le plafond : résultats NON valides"
        )));
    }
    Ok(0)
}

fn analyse(m: &ArgMatches) -> Resultat<i32> {
    let q = Quadrature::analyser(requis(m, "quadrature")?)?;
    for d in ["entree", "entree-cube"] {
        if let Some(Ok(t)) = texte(m, d).map(|d| std::fs::read_to_string(Path::new(d).join("manifest_thermo.json"))) {
            if t.contains("\"couplages_separes\"") {
                return Err(Erreur::Config(format!("--{d} : mesure à couplages d'espace et de temps séparés (--beta-s) : `analyse` est isotrope et refuse ; intégrer E[S_t] en β_t (colonnes st) avec le script du projet")));
            }
        }
    }
    let pts = lire_points(Path::new(requis(m, "entree")?))?;
    let bras = |pts: &[(f64, PointRep)], w: f64| -> Vec<PointRep> {
        pts.iter()
            .filter(|(p, _)| *p == w)
            .map(|(_, r)| r.clone())
            .collect()
    };
    let (res, quoi) = match requis(m, "mode")? {
        "torsion" => {
            let s: f64 = un(m, "poids-tordu")?;
            let (a, b) = (bras(&pts, s), bras(&pts, 1.0));
            if a.is_empty() || b.is_empty() {
                return Err(Erreur::Config(format!(
                    "mode torsion : bras de poids {s} et 1 attendus dans l'entrée"
                )));
            }
            (thermo::ln_r(&a, &b, s, q)?, "ln(Z_tordu/Z_périodique)")
        }
        "a3" => {
            let c = lire_points(Path::new(requis(m, "entree-cube")?))?;
            let (a, b) = (bras(&pts, 1.0), bras(&c, 1.0));
            let man = |d: &str| -> Resultat<Vec<usize>> {
                let t = std::fs::read_to_string(Path::new(d).join("manifest_thermo.json"))
                    .map_err(|e| Erreur::io("manifeste", e))?;
                let v: serde_json::Value = serde_json::from_str(&t)
                    .map_err(|e| Erreur::Config(format!("manifeste illisible : {e}")))?;
                Ok(v["cotes"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|y| y as usize))
                            .collect()
                    })
                    .unwrap_or_default())
            };
            let champ = |d: &str, k: &str| -> Resultat<String> {
                let t = std::fs::read_to_string(Path::new(d).join("manifest_thermo.json"))
                    .map_err(|e| Erreur::io("manifeste", e))?;
                let v: serde_json::Value = serde_json::from_str(&t)
                    .map_err(|e| Erreur::Config(format!("manifeste illisible : {e}")))?;
                Ok(v[k].to_string())
            };
            for k in [
                "session",
                "precision",
                "sha256_source_assemblee",
                "therm",
                "nmeas",
                "repliques",
                "betas",
                "nor",
                "torsion",
            ] {
                let (x, y) = (
                    champ(requis(m, "entree")?, k)?,
                    champ(requis(m, "entree-cube")?, k)?,
                );
                if x != y || (k == "session" && x == "null") {
                    return Err(Erreur::Config(format!(
                        "mode a3 : « {k} » diffère entre les deux dossiers ({x} / {y}) : les deux boîtes d'une paire doivent venir de la MÊME exécution (même moteur, même précision, même carte, tranches alternées) ; ne jamais différencier des boîtes de deux codes ou de deux exécutions"
                    )));
                }
            }
            let (cb, cc) = (man(requis(m, "entree")?)?, man(requis(m, "entree-cube")?)?);
            if cb.len() != 4
                || cc.len() != 4
                || cc.iter().product::<usize>() != 2 * cb.iter().product::<usize>()
            {
                return Err(Erreur::Config(format!("mode a3 : volume du cube {cc:?} ≠ 2 × volume de la boîte {cb:?} (les constantes e^(−6Vβ) ne se compenseraient pas)")));
            }
            (thermo::a3(&a, &b, q)?, "2 ln Z(boîte) − ln Z(cube)")
        }
        autre => {
            return Err(Erreur::Config(format!(
                "--mode « {autre} » inconnu (torsion|a3)"
            )))
        }
    };
    let mut lignes = vec![];
    for k in 0..res.betas.len() {
        lignes.push(serde_json::json!({"beta": res.betas[k], "valeur": res.valeur[k], "erreur_jackknife": res.erreur[k], "ecart_moities": res.ecart_moities[k]}));
        println!(
            "beta = {:.6}  {quoi} = {:+.6} ± {:.6}   (moitiés : {:+.2e})",
            res.betas[k], res.valeur[k], res.erreur[k], res.ecart_moities[k]
        );
    }
    if let Some(f) = texte(m, "json") {
        let o = serde_json::json!({"schema": "kvu-thermo-analyse/1", "grandeur": quoi, "quadrature": format!("{q:?}"), "nœuds": lignes,
            "note": "erreur = jackknife sur les répliques (bras appariés par graine) ; biais de quadrature NON inclus : comparer trapèze et parabole"});
        std::fs::write(f, serde_json::to_string_pretty(&o).unwrap())
            .map_err(|e| Erreur::io("écriture du json", e))?;
    }
    Ok(0)
}
