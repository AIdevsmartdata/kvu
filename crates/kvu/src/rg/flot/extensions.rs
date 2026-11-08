//! Options WP9 : statistiques conservées par configuration et jackknife par blocs.
use super::cellule::{
    beta_courbe, beta_sd_cellule, moyenne_observable, DonneesChaine, ParamFlot, Schema,
};
use super::jackknife::{jackknife_groupes, ResJackknife};
use super::plaq::Observable;
use super::sd::{ModeRapport, StatsSd};
use crate::commun::{Erreur, Resultat};
use crate::rg::bloc::StatsTheta;
use crate::rg::operateur::Operateur;
use serde_json::{json, Value};

pub struct OptionsFlot {
    pub backend: String,
    pub bins: Option<usize>,
    /// Tailles dans l'ordre demandé ; vide conserve l'API à une taille de la passe 1.
    pub bins_tailles: Vec<usize>,
    pub detail_configs: bool,
    /// Opérateur de blocage de facteur 3 (WP22) ; `3centre` (défaut) = chemin historique, octet pour octet.
    pub operateur: Operateur,
    /// `--hors-famille` : accepte un opérateur dont le rayon de lissage dépasse le bloc.
    pub hors_famille: bool,
}

impl Default for OptionsFlot {
    fn default() -> Self {
        Self {
            backend: "cpu".into(),
            bins: None,
            bins_tailles: Vec::new(),
            detail_configs: false,
            operateur: Operateur::TroisCentre,
            hors_famille: false,
        }
    }
}

impl OptionsFlot {
    pub fn verifier(&self, schema: Schema) -> Resultat<()> {
        if !matches!(self.backend.as_str(), "cpu" | "hip" | "cuda" | "auto") {
            return Err(Erreur::Config(
                "--backend attendu : cpu|hip|cuda|auto".into(),
            ));
        }
        if self.backend != "cpu" && schema != Schema::Balaban {
            return Err(Erreur::Config(
                "--backend GPU exige --schema balaban".into(),
            ));
        }
        if !self.operateur.est_trois_centre() {
            if schema != Schema::Balaban {
                return Err(Erreur::Config(format!(
                    "--operateur {} exige --schema balaban (facteur 3)",
                    self.operateur.spec()
                )));
            }
            if self.backend != "cpu" {
                return Err(Erreur::Config(format!(
                    "--operateur {} : seul le backend cpu est disponible",
                    self.operateur.spec()
                )));
            }
            self.operateur.verifier_famille(self.hors_famille)?;
        }
        let tailles = self.tailles_bins();
        if tailles.contains(&0)
            || tailles
                .iter()
                .enumerate()
                .any(|(i, k)| tailles[..i].contains(k))
        {
            return Err(Erreur::Config("--bins : entiers ≥ 1, sans doublon".into()));
        }
        if self.bins.is_some()
            && !self.bins_tailles.is_empty()
            && self.bins != tailles.first().copied()
        {
            return Err(Erreur::Config(
                "bins doit correspondre à la première taille".into(),
            ));
        }
        Ok(())
    }

    pub fn tailles_bins(&self) -> Vec<usize> {
        if self.bins_tailles.is_empty() {
            self.bins.into_iter().collect()
        } else {
            self.bins_tailles.clone()
        }
    }
}

pub fn analyser_bins(texte: &str) -> Resultat<Vec<usize>> {
    let tailles = texte
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            Erreur::Config("--bins : liste à virgules d'entiers ≥ 1, sans doublon".into())
        })?;
    let options = OptionsFlot {
        bins_tailles: tailles.clone(),
        ..Default::default()
    };
    options.verifier(Schema::Balaban)?;
    Ok(tailles)
}

pub(super) struct ConfigFlot {
    pub fichier: String,
    pub sd: StatsSd,
    pub theta: Option<StatsTheta>,
}

/// Blocs consécutifs, reste rattaché au dernier bloc ; aucune configuration perdue.
pub fn plages_bins(n: usize, taille: usize) -> Vec<std::ops::Range<usize>> {
    if n == 0 || taille == 0 {
        return Vec::new();
    }
    let g = (n / taille).max(1);
    (0..g)
        .map(|i| i * taille..if i + 1 == g { n } else { (i + 1) * taille })
        .collect()
}

fn json_jk(j: &Option<ResJackknife>, n: usize) -> Value {
    j.as_ref().map_or(Value::Null, |j| {
        json!({"erreur": j.erreur, "moyenne_replicats": j.moyenne_replicats,
        "replicats": j.replicats, "n_groupes": n})
    })
}

fn erreur_totale(j: &Option<ResJackknife>, sigma: Option<f64>) -> Option<f64> {
    j.as_ref()
        .map(|j| sigma.map_or(j.erreur, |s| j.erreur.hypot(s)))
}

fn ajouter_jk(b: &mut Value, j: &Option<ResJackknife>, n: usize) {
    let totale = erreur_totale(j, b["sigma_etalonnage"].as_f64());
    b["jackknife_bins"] = json_jk(j, n);
    b["erreur_jackknife_bins"] = json!(j.as_ref().map(|j| j.erreur));
    b["erreur_totale_bins"] = json!(totale);
    if let Some(a) = b.get_mut("a_j") {
        a["erreur_bins"] = json!(totale);
    }
}

fn ajouter_taille(b: &mut Value, j: &Option<ResJackknife>, taille: usize, n: usize) {
    let entree = json!({"taille": taille, "n_groupes": n,
        "erreur": j.as_ref().map(|j| j.erreur),
        "moyenne_replicats": j.as_ref().map(|j| j.moyenne_replicats),
        "replicats": j.as_ref().map(|j| &j.replicats)});
    let objet = b.as_object_mut().expect("bloc d'estimateur");
    objet
        .entry("jackknife_bins_tailles")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("tailles de jackknife")
        .push(entree);
}

pub(super) fn ajouter_options(
    cel: &mut Value,
    donnees: &[DonneesChaine],
    configs: &[Vec<ConfigFlot>],
    par: &ParamFlot,
    options: &OptionsFlot,
) {
    if options.detail_configs {
        let mut details = Vec::new();
        for (d, cs) in donnees.iter().zip(configs) {
            for (i, c) in cs.iter().enumerate() {
                let mut v = json!({"chaine": d.nom, "fichier": c.fichier, "p": d.p[i], "r": d.r[i],
                    "sd_somme_w": c.sd.somme_w, "sd_somme_k": c.sd.ks.iter().sum::<f64>()});
                if let Some(t) = &c.theta {
                    v["fraction_hors"] = json!(t.fraction_hors());
                    v["theta_max"] = json!(t.theta_max);
                    v["n_ambigues"] = json!(t.n_ambigues);
                    v["n_boucles"] = json!(t.n);
                }
                details.push(v);
            }
        }
        cel["configs"] = json!(details);
    }
    for (indice, taille) in options.tailles_bins().into_iter().enumerate() {
        ajouter_bins(cel, donnees, configs, par, taille, indice == 0);
    }
}

fn ajouter_bins(
    cel: &mut Value,
    donnees: &[DonneesChaine],
    configs: &[Vec<ConfigFlot>],
    par: &ParamFlot,
    taille: usize,
    premiere: bool,
) {
    let mut groupes = Vec::new();
    for (d, cs) in donnees.iter().zip(configs) {
        for plage in plages_bins(cs.len(), taille) {
            groupes.push(DonneesChaine {
                nom: d.nom.clone(),
                l_fin: d.l_fin,
                sd: StatsSd::somme(cs[plage.clone()].iter().map(|c| &c.sd)),
                p: d.p[plage.clone()].to_vec(),
                r: d.r[plage].to_vec(),
                theta: vec![],
            });
        }
    }
    let n = groupes.len();
    if premiere {
        cel["bins"] = json!({"taille": taille, "n_groupes": n, "tailles": []});
    }
    cel["bins"]["tailles"]
        .as_array_mut()
        .unwrap()
        .push(json!({"taille": taille, "n_groupes": n}));
    let est = &mut cel["estimateurs"];
    if est["sd"]["etat"] == "ok" {
        let j = jackknife_groupes(&groupes, |g| {
            beta_sd_cellule(g, par.rapport, ModeRapport::Exact)
        });
        if premiere {
            ajouter_jk(&mut est["sd"], &j, n);
        }
        ajouter_taille(&mut est["sd"], &j, taille, n);
    }
    if let Some(courbe) = par.courbes.iter().find(|c| c.l == donnees[0].l_fin) {
        let bp = |g: &[&DonneesChaine]| {
            beta_courbe(
                courbe,
                Observable::Plaquette,
                moyenne_observable(g, Observable::Plaquette),
            )
            .map(|x| x.0)
        };
        let br = |g: &[&DonneesChaine]| {
            beta_courbe(
                courbe,
                Observable::Rectangle,
                moyenne_observable(g, Observable::Rectangle),
            )
            .map(|x| x.0)
        };
        if est["plaq"]["etat"] == "ok" {
            let j = jackknife_groupes(&groupes, bp);
            if premiere {
                ajouter_jk(&mut est["plaq"], &j, n);
            }
            ajouter_taille(&mut est["plaq"], &j, taille, n);
        }
        if est["mcrg2"]["etat"] == "ok" {
            let j = jackknife_groupes(&groupes, |g| Some(0.5 * (bp(g)? + br(g)?)));
            let ecart = jackknife_groupes(&groupes, |g| Some(br(g)? - bp(g)?));
            if premiere {
                ajouter_jk(&mut est["mcrg2"], &j, n);
                est["mcrg2"]["ecart_schema"]["jackknife_bins"] = json_jk(&ecart, n);
            }
            ajouter_taille(&mut est["mcrg2"], &j, taille, n);
            ajouter_taille(&mut est["mcrg2"]["ecart_schema"], &ecart, taille, n);
            if !premiere {
                return;
            }
            for (cle, obs, j) in [
                (
                    "a_j_p",
                    Observable::Plaquette,
                    jackknife_groupes(&groupes, bp),
                ),
                (
                    "a_j_r",
                    Observable::Rectangle,
                    jackknife_groupes(&groupes, br),
                ),
            ] {
                if let Some(a) = est["mcrg2"].get_mut(cle) {
                    let refs = donnees.iter().collect::<Vec<_>>();
                    let sigma = beta_courbe(courbe, obs, moyenne_observable(&refs, obs))
                        .map(|(_, p, s)| s / p.abs());
                    a["erreur_bins"] = json!(erreur_totale(&j, sigma));
                }
            }
        }
    }
}
