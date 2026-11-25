//! Géométrie commune des réseaux, des conditions au bord et des configurations de kvu v0.2.

pub mod champ;
pub mod reseau;
pub mod tables;
pub mod torsion;

pub use champ::ChampGeo;
pub use reseau::{indice_plan, GeoA, Pile, PLANS};
pub use tables::{Coloriage, Plaq, PlaquettesExplicites, Tables, MAXCASE};
pub use torsion::Torsion;

use crate::commun::{Erreur, Resultat};
use crate::su2::boite::GeometrieBoite;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "kvu-geometrie/1";
pub const SCHEMA_TEXTE: &str = include_str!("kvu-geometrie-1.schema.json");
pub const NOM_FICHIER_GEOMETRIE: &str = "geometrie.json";
pub const DISPOSITION: &str = concat!(
    "s = ((x0*l1 + x1)*l2 + x2)*l3 + x3 ; li = mu*V + s ; ",
    "repliques empilees sur l'axe 0 (la plus lente) ; ",
    "NPY (4, l0', l1, l2, l3, 2, 2) c16 ou (4, l0', l1, l2, l3, 4) q64, l0' = repliques * l0"
);
pub const PILE: &str = "coin (l_mu - 1, l_nu - 1) de chaque plan tordu (convention kvu M4)";

/// Analyse le schéma embarqué, dont la syntaxe est un invariant du dépôt.
pub fn schema() -> Value {
    serde_json::from_str(SCHEMA_TEXTE)
        .expect("le schéma JSON embarqué kvu-geometrie/1 doit être syntaxiquement valide")
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeoSpec {
    /// Côtés d'une réplique ; les répliques sont empilées suivant l'axe 0 au stockage.
    pub cotes: [usize; 4],
    pub dir_temps: u8,
    pub bords: [Bord; 4],
    pub torsion: Torsion,
    pub beta_plans: Option<[f64; 6]>,
    pub masque: Option<SourceMasque>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Bord {
    Periodique,
    Fixe {
        valeurs: SourceBord,
    },
    Recolle {
        repliques: u8,
        region: RegionSpec,
        couture: Couture,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceBord {
    Froid,
    Chaud {
        graine: u64,
    },
    Copie {
        source: String,
    },
    Npy {
        chemin: String,
        sha256: Option<String>,
    },
    ParChaine,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegionSpec {
    /// Un lien spatial est dans A si ses deux extrémités vérifient x_axe < largeur.
    Tranche { axe: u8, largeur: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Couture {
    /// Poids ½ + ½ de la convention T14 §1.
    Symetrique,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceMasque {
    /// Boîte WP-DIR de longueur ℓ plongée dans un réseau périodique de côté ℓ + 2.
    Boite { ell: usize, valeurs: SourceBord },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chemin {
    C0,
    C1,
    C2,
}

impl GeoSpec {
    /// Préréglage non validé : appeler [`Self::valider`] avant utilisation.
    pub fn cubique(l: usize, dir_temps: u8) -> GeoSpec {
        Self::anisotrope([l; 4], dir_temps)
    }

    pub fn anisotrope(cotes: [usize; 4], dir_temps: u8) -> GeoSpec {
        GeoSpec {
            cotes,
            dir_temps,
            bords: std::array::from_fn(|_| Bord::Periodique),
            torsion: Torsion::triviale(),
            beta_plans: None,
            masque: None,
        }
    }

    pub fn hyperplan(l: usize, dir_temps: u8, valeurs: SourceBord) -> GeoSpec {
        let mut spec = Self::cubique(l, dir_temps);
        spec.bords[0] = Bord::Fixe { valeurs };
        spec
    }

    pub fn boite(ell: usize, dir_temps: u8, valeurs: SourceBord) -> GeoSpec {
        // Un argument hors limites reste représentable et sera refusé par valider().
        let mut spec = Self::cubique(ell.saturating_add(2), dir_temps);
        spec.masque = Some(SourceMasque::Boite { ell, valeurs });
        spec
    }

    pub fn repliques(l: usize, t: usize, repliques: u8, axe: u8, largeur: usize) -> GeoSpec {
        let mut spec = Self::anisotrope([t, l, l, l], 0);
        spec.bords[0] = Bord::Recolle {
            repliques,
            region: RegionSpec::Tranche { axe, largeur },
            couture: Couture::Symetrique,
        };
        spec
    }

    pub fn avec_torsion(mut self, t: Torsion) -> GeoSpec {
        self.torsion = t;
        self
    }

    pub fn avec_beta_plans(mut self, b: [f64; 6]) -> GeoSpec {
        self.beta_plans = Some(b);
        self
    }

    pub fn repliques_nb(&self) -> usize {
        self.bords
            .iter()
            .find_map(|bord| match bord {
                Bord::Recolle { repliques, .. } => Some(usize::from(*repliques)),
                _ => None,
            })
            .unwrap_or(1)
    }

    /// Exact après validation ; saturant pour une spécification hors limites non encore validée.
    pub fn cotes_stockees(&self) -> [usize; 4] {
        let mut cotes = self.cotes;
        cotes[0] = cotes[0].saturating_mul(self.repliques_nb());
        cotes
    }

    /// Volume stocké, exact après validation ; saturant avant validation.
    pub fn v(&self) -> usize {
        self.cotes_stockees()
            .iter()
            .fold(1usize, |v, &c| v.saturating_mul(c))
    }

    pub fn chemin(&self) -> Chemin {
        if self.bords.iter().any(|b| matches!(b, Bord::Recolle { .. })) {
            Chemin::C2
        } else if self
            .cotes
            .iter()
            .all(|&c| c == self.cotes[0] && c.is_multiple_of(2))
            && self.bords.iter().all(|b| matches!(b, Bord::Periodique))
            && self.torsion.est_triviale()
            && self.beta_plans.is_none()
            && self.masque.is_none()
        {
            Chemin::C0
        } else {
            Chemin::C1
        }
    }

    pub fn coloriage(&self) -> &'static str {
        if self.chemin() == Chemin::C2 {
            "explicite"
        } else if self.cotes_stockees().iter().all(|c| c.is_multiple_of(2)) {
            "parite"
        } else {
            "glouton"
        }
    }

    pub fn valider(&self) -> Resultat<()> {
        if self.cotes.iter().any(|&c| c < 2) {
            return Err(Erreur::Config(format!(
                "côtés {:?} : chaque côté doit être ≥ 2",
                self.cotes
            )));
        }
        if self.dir_temps > 3 {
            return Err(Erreur::Config(format!(
                "dir_temps = {} : direction attendue dans 0..=3",
                self.dir_temps
            )));
        }
        let mut recolles = 0;
        let mut fixes = 0;
        for (mu, bord) in self.bords.iter().enumerate() {
            match bord {
                Bord::Periodique => {}
                Bord::Fixe { valeurs } => {
                    fixes += 1;
                    valeurs.valider()?;
                }
                Bord::Recolle {
                    repliques, region, ..
                } => {
                    recolles += 1;
                    if mu != 0 || *repliques < 2 {
                        return Err(Erreur::Config(
                            "recollement : direction 0 seulement et au moins deux répliques requises".into(),
                        ));
                    }
                    let RegionSpec::Tranche { axe, largeur } = region;
                    if !(1..=3).contains(axe) || *largeur > self.cotes[usize::from(*axe)] {
                        return Err(Erreur::Config(
                            "région de recollement : axe dans 1..=3 et largeur ≤ côté de cet axe requis".into(),
                        ));
                    }
                }
            }
        }
        if recolles > 1 {
            return Err(Erreur::Config(
                "au plus un bord recollé est pris en charge".into(),
            ));
        }
        // O6 (relecture du 11/11) : deux hyperplans fixes se coupent, et leurs valeurs prescrites pourraient se contredire
        // sur les liens communs ; la v0.2 n'en accepte qu'un (R2-3 = un seul hyperplan, x0 = 0).
        if fixes > 1 {
            return Err(Erreur::Config(
                "au plus un bord fixe est pris en charge en v0.2 (valeurs non définies à l'intersection de deux hyperplans)"
                    .into(),
            ));
        }
        if recolles != 0
            && (fixes != 0
                || !self.torsion.est_triviale()
                || self.masque.is_some()
                || self.beta_plans.is_some())
        {
            return Err(Erreur::Config(
                "recollement combiné à une torsion, un bord fixe, un masque ou beta_plans non pris en charge en v0.2"
                    .into(),
            ));
        }
        let volume = self
            .cotes
            .iter()
            .try_fold(1usize, |v, &c| v.checked_mul(c))
            .and_then(|v| v.checked_mul(self.repliques_nb()));
        let liens = volume.and_then(|v| v.checked_mul(4));
        if !liens.is_some_and(|n| n <= u32::MAX as usize) {
            return Err(Erreur::Config(
                "volume stocké hors limites : 4·V doit tenir dans un compteur Philox sur 32 bits"
                    .into(),
            ));
        }
        if recolles != 0 {
            // O7 : capacités des encodages C2 (voir `tables` : entrées (li << 1) | dague, incidences (id << 2) | position,
            // sentinelle NUL = u32::MAX), vérifiées AVANT toute allocation.
            let v = volume.unwrap_or(usize::MAX);
            let couture = self.cotes[1..]
                .iter()
                .try_fold(3 * self.repliques_nb(), |a, &c| a.checked_mul(c));
            // Majorant de l'UNION des plaquettes des deux géométries de l'interpolation (relecture c04b, N1) : deux fois
            // les plaquettes brutes d'une géométrie (6V + 3·s·ℓ1ℓ2ℓ3).
            let plaquettes = v
                .checked_mul(6)
                .and_then(|n| couture.and_then(|c| n.checked_add(c)))
                .and_then(|n| n.checked_mul(2));
            if 4 * v > tables::LIMITE_LIENS_C2
                || !plaquettes.is_some_and(|n| n <= tables::LIMITE_PLAQUETTES_C2)
            {
                return Err(Erreur::Config(format!(
                    "recollement : capacité C2 dépassée (4·V = {} liens, au plus {} ; {:?} plaquettes, au plus {}) : \
                     les encodages (li << 1) | dague et (id << 2) | position tiennent sur 32 bits",
                    4 * v,
                    tables::LIMITE_LIENS_C2,
                    plaquettes,
                    tables::LIMITE_PLAQUETTES_C2
                )));
            }
        }
        if let Some(beta) = &self.beta_plans {
            if beta.iter().any(|b| !b.is_finite() || *b <= 0.0) {
                return Err(Erreur::Config(
                    "beta_plans : six réels finis strictement positifs requis".into(),
                ));
            }
        }
        if let Some(SourceMasque::Boite { ell, valeurs }) = &self.masque {
            let boite = GeometrieBoite::nouvelle(*ell)?;
            if self.cotes != [boite.lc; 4]
                || !self.bords.iter().all(|b| matches!(b, Bord::Periodique))
                || !self.torsion.est_triviale()
            {
                return Err(Erreur::Config(
                    "masque de boîte : côtés [ℓ+2; 4], bords périodiques et torsion triviale requis".into(),
                ));
            }
            valeurs.valider()?;
        }
        Ok(())
    }

    /// Bloc unique des manifestes et de geometrie.json, y compris les champs dérivés.
    pub fn bloc(&self) -> Value {
        let bords: Vec<Value> = self.bords.iter().map(Bord::json).collect();
        let plans: Vec<String> = Torsion::PLANS
            .iter()
            .filter(|&&(mu, nu)| self.torsion.tordu(mu, nu))
            .map(|&(mu, nu)| format!("{mu}{nu}"))
            .collect();
        let chemin = match self.chemin() {
            Chemin::C0 => "C0",
            Chemin::C1 => "C1",
            Chemin::C2 => "C2",
        };
        json!({
            "schema": SCHEMA,
            "cotes": self.cotes,
            "dir_temps": self.dir_temps,
            "bords": bords,
            "torsion": { "plans": plans, "facteur": -1, "pile": PILE },
            "beta_plans": self.beta_plans,
            "masque": self.masque.as_ref().map(SourceMasque::json),
            "chemin": chemin,
            "disposition": DISPOSITION,
            "coloriage": self.coloriage(),
        })
    }

    pub fn texte(&self) -> String {
        // Une Value se sérialise toujours ; aucun type utilisateur n'intervient ici.
        let mut texte =
            serde_json::to_string_pretty(&self.bloc()).expect("une valeur JSON doit se sérialiser");
        texte.push('\n');
        texte
    }

    pub fn depuis_json(v: &Value) -> Resultat<GeoSpec> {
        let erreurs = crate::regen::schema::valider(&schema(), v)
            .map_err(|e| Erreur::Config(format!("schéma de géométrie : {e}")))?;
        if !erreurs.is_empty() {
            return Err(Erreur::Config(format!(
                "géométrie non conforme au schéma :\n{}",
                erreurs.join("\n")
            )));
        }
        let mut cotes = [0; 4];
        for (c, valeur) in cotes
            .iter_mut()
            .zip(tableau_exact(&v["cotes"], 4, "cotes")?)
        {
            *c = entier_json(valeur, "cotes")?;
        }
        let mut bords = std::array::from_fn(|_| Bord::Periodique);
        for (bord, valeur) in bords
            .iter_mut()
            .zip(tableau_exact(&v["bords"], 4, "bords")?)
        {
            *bord = Bord::analyser(valeur)?;
        }
        let beta_plans = if v["beta_plans"].is_null() {
            None
        } else {
            let mut beta = [0.0; 6];
            for (b, valeur) in
                beta.iter_mut()
                    .zip(tableau_exact(&v["beta_plans"], 6, "beta_plans")?)
            {
                *b = valeur
                    .as_f64()
                    .ok_or_else(|| champ_invalide("beta_plans"))?;
            }
            Some(beta)
        };
        let spec = GeoSpec {
            cotes,
            dir_temps: octet_json(&v["dir_temps"], "dir_temps")?,
            bords,
            torsion: torsion_depuis_plans(&v["torsion"]["plans"])?,
            beta_plans,
            masque: if v["masque"].is_null() {
                None
            } else {
                Some(SourceMasque::analyser(&v["masque"])?)
            },
        };
        spec.valider()?;
        let derive = spec.bloc();
        for champ in ["chemin", "coloriage", "disposition"] {
            if v[champ] != derive[champ] {
                return Err(Erreur::Config(format!("champ dérivé incohérent : {champ}")));
            }
        }
        if v["torsion"]["pile"] != derive["torsion"]["pile"] {
            return Err(Erreur::Config(
                "champ dérivé incohérent : torsion.pile".into(),
            ));
        }
        Ok(spec)
    }

    pub fn inscrire(&self, man: &mut crate::stockage::Manifeste, cle: &str) -> Resultat<()> {
        man.champ(cle, &self.bloc().to_string())
    }
}

impl SourceBord {
    pub fn json(&self) -> Value {
        match self {
            Self::Froid => json!({ "type": "froid" }),
            Self::Chaud { graine } => json!({ "type": "chaud", "graine": graine }),
            Self::Copie { source } => json!({ "type": "copie", "source": source }),
            Self::Npy { chemin, sha256 } => {
                json!({ "type": "npy", "chemin": chemin, "sha256": sha256 })
            }
            Self::ParChaine => json!({ "type": "par_chaine" }),
        }
    }

    fn analyser(v: &Value) -> Resultat<SourceBord> {
        match texte_json(&v["type"], "valeurs.type")? {
            "froid" => {
                champs_exacts(v, &["type"], "valeurs froides")?;
                Ok(Self::Froid)
            }
            "chaud" => {
                champs_exacts(v, &["type", "graine"], "valeurs chaudes")?;
                let graine = v["graine"]
                    .as_u64()
                    .ok_or_else(|| champ_invalide("valeurs.graine"))?;
                Ok(Self::Chaud { graine })
            }
            "copie" => {
                champs_exacts(v, &["type", "source"], "valeurs copiées")?;
                Ok(Self::Copie {
                    source: texte_json(&v["source"], "valeurs.source")?.to_string(),
                })
            }
            "npy" => {
                champs_exacts(v, &["type", "chemin", "sha256"], "valeurs NPY")?;
                Ok(Self::Npy {
                    chemin: texte_json(&v["chemin"], "valeurs.chemin")?.to_string(),
                    sha256: if v["sha256"].is_null() {
                        None
                    } else {
                        Some(texte_json(&v["sha256"], "valeurs.sha256")?.to_string())
                    },
                })
            }
            "par_chaine" => {
                champs_exacts(v, &["type"], "valeurs par chaîne")?;
                Ok(Self::ParChaine)
            }
            _ => Err(champ_invalide("valeurs.type")),
        }
    }

    fn valider(&self) -> Resultat<()> {
        match self {
            Self::Copie { source } if source.is_empty() => {
                return Err(Erreur::Config("source de copie vide".into()));
            }
            Self::Npy { chemin, sha256 } => {
                if chemin.is_empty() {
                    return Err(Erreur::Config("chemin des valeurs NPY vide".into()));
                }
                if let Some(sha) = sha256 {
                    if sha.len() != 64
                        || !sha
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    {
                        return Err(Erreur::Config(
                            "sha256 NPY : 64 chiffres hexadécimaux minuscules requis".into(),
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Bord {
    pub fn json(&self) -> Value {
        match self {
            Self::Periodique => json!({ "type": "periodique" }),
            Self::Fixe { valeurs } => json!({ "type": "fixe", "valeurs": valeurs.json() }),
            Self::Recolle {
                repliques,
                region,
                couture: Couture::Symetrique,
            } => json!({
                "type": "recolle", "repliques": repliques, "region": region.json(), "couture": "symetrique"
            }),
        }
    }

    fn analyser(v: &Value) -> Resultat<Bord> {
        match texte_json(&v["type"], "bord.type")? {
            "periodique" => {
                champs_exacts(v, &["type"], "bord périodique")?;
                Ok(Self::Periodique)
            }
            "fixe" => {
                champs_exacts(v, &["type", "valeurs"], "bord fixe")?;
                Ok(Self::Fixe {
                    valeurs: SourceBord::analyser(&v["valeurs"])?,
                })
            }
            "recolle" => {
                champs_exacts(
                    v,
                    &["type", "repliques", "region", "couture"],
                    "bord recollé",
                )?;
                if texte_json(&v["couture"], "bord.couture")? != "symetrique" {
                    return Err(champ_invalide("bord.couture"));
                }
                Ok(Self::Recolle {
                    repliques: octet_json(&v["repliques"], "bord.repliques")?,
                    region: RegionSpec::analyser(&v["region"])?,
                    couture: Couture::Symetrique,
                })
            }
            _ => Err(champ_invalide("bord.type")),
        }
    }
}

impl RegionSpec {
    pub fn json(&self) -> Value {
        match self {
            Self::Tranche { axe, largeur } => {
                json!({ "type": "tranche", "axe": axe, "largeur": largeur })
            }
        }
    }

    fn analyser(v: &Value) -> Resultat<RegionSpec> {
        champs_exacts(v, &["type", "axe", "largeur"], "région")?;
        if texte_json(&v["type"], "region.type")? != "tranche" {
            return Err(champ_invalide("region.type"));
        }
        Ok(Self::Tranche {
            axe: octet_json(&v["axe"], "region.axe")?,
            largeur: entier_json(&v["largeur"], "region.largeur")?,
        })
    }
}

impl SourceMasque {
    fn json(&self) -> Value {
        match self {
            Self::Boite { ell, valeurs } => {
                json!({ "type": "boite", "ell": ell, "valeurs": valeurs.json() })
            }
        }
    }

    fn analyser(v: &Value) -> Resultat<SourceMasque> {
        champs_exacts(v, &["type", "ell", "valeurs"], "masque")?;
        if texte_json(&v["type"], "masque.type")? != "boite" {
            return Err(champ_invalide("masque.type"));
        }
        Ok(Self::Boite {
            ell: entier_json(&v["ell"], "masque.ell")?,
            valeurs: SourceBord::analyser(&v["valeurs"])?,
        })
    }
}

fn champ_invalide(nom: &str) -> Erreur {
    Erreur::Config(format!(
        "géométrie : champ « {nom} » absent, mal typé ou hors limites"
    ))
}

fn champs_exacts(v: &Value, attendus: &[&str], contexte: &str) -> Resultat<()> {
    let objet = v.as_object().ok_or_else(|| champ_invalide(contexte))?;
    for nom in attendus {
        if !objet.contains_key(*nom) {
            return Err(Erreur::Config(format!(
                "{contexte} : champ « {nom} » obligatoire"
            )));
        }
    }
    for nom in objet.keys() {
        if !attendus.contains(&nom.as_str()) {
            return Err(Erreur::Config(format!(
                "{contexte} : champ « {nom} » propre à un autre type"
            )));
        }
    }
    Ok(())
}

fn tableau_exact<'a>(v: &'a Value, n: usize, nom: &str) -> Resultat<&'a [Value]> {
    let valeurs = v.as_array().ok_or_else(|| champ_invalide(nom))?;
    if valeurs.len() != n {
        return Err(Erreur::Config(format!(
            "{nom} : exactement {n} éléments requis, reçu {}",
            valeurs.len()
        )));
    }
    Ok(valeurs)
}

fn texte_json<'a>(v: &'a Value, nom: &str) -> Resultat<&'a str> {
    v.as_str().ok_or_else(|| champ_invalide(nom))
}

fn entier_json(v: &Value, nom: &str) -> Resultat<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| champ_invalide(nom))
}

fn octet_json(v: &Value, nom: &str) -> Resultat<u8> {
    v.as_u64()
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| champ_invalide(nom))
}

fn torsion_depuis_plans(v: &Value) -> Resultat<Torsion> {
    let plans = v
        .as_array()
        .ok_or_else(|| champ_invalide("torsion.plans"))?;
    let textes: Vec<&str> = plans
        .iter()
        .map(|p| texte_json(p, "torsion.plans"))
        .collect::<Resultat<_>>()?;
    let texte = if textes.is_empty() {
        "aucune".to_string()
    } else {
        textes.join(",")
    };
    Torsion::analyser(&texte)
}

/// Écrit geometrie.json par fichier temporaire synchronisé puis renommage atomique.
pub fn ecrire_geometrie(dossier: &Path, spec: &GeoSpec) -> Resultat<PathBuf> {
    spec.valider()?;
    let chemin = dossier.join(NOM_FICHIER_GEOMETRIE);
    let part = dossier.join("geometrie.json.part");
    let mut fichier = std::fs::File::create(&part)
        .map_err(|e| Erreur::io(format!("création {}", part.display()), e))?;
    let resultat = fichier
        .write_all(spec.texte().as_bytes())
        .and_then(|()| fichier.sync_all());
    drop(fichier);
    if let Err(e) = resultat {
        let _ = std::fs::remove_file(&part);
        return Err(Erreur::io(
            format!("écriture et synchronisation {}", part.display()),
            e,
        ));
    }
    if let Err(e) = std::fs::rename(&part, &chemin) {
        let _ = std::fs::remove_file(&part);
        return Err(Erreur::io(
            format!("renommage {} vers {}", part.display(), chemin.display()),
            e,
        ));
    }
    Ok(chemin)
}

pub fn lire_geometrie(chemin: &Path) -> Resultat<GeoSpec> {
    let v = lire_json(chemin)?;
    GeoSpec::depuis_json(&v).map_err(|e| match e {
        Erreur::Config(m) => Erreur::Config(format!("{} : {m}", chemin.display())),
        autre => autre,
    })
}

/// Mutants (contrôles négatifs) de la recherche de géométrie voisine, pour les portes G-REFUS : JAMAIS en production.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutantVoisinage {
    #[default]
    Aucun,
    /// Le bloc canonique `geometrie` des `manifest.json` voisins est ignoré (défaut O1 de la relecture du 11/11).
    IgnoreBlocManifeste,
    /// Les dossiers sont pris dans le chemin TEL QU'ÉCRIT, ni rendu absolu ni résolu (défaut O2 de la relecture du 11/11).
    CheminTelQuEcrit,
}

/// Ce que les métadonnées voisines d'une configuration disent de sa géométrie (voir [`examiner_voisinage`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GeometrieVoisine {
    /// Géométrie déclarée et le fichier qui la porte (`geometrie.json`, sinon bloc canonique d'un `manifest.json`) ;
    /// None : aucune déclaration canonique, et aucune déclaration historique non C0.
    pub spec: Option<(GeoSpec, PathBuf)>,
    /// Dossiers examinés (absolus et résolus), dans l'ordre.
    pub dossiers: Vec<PathBuf>,
    /// Fichiers de métadonnées lus.
    pub fichiers_lus: Vec<PathBuf>,
}

/// Géométrie déclarée à côté de la configuration `cfg` (voir [`examiner_voisinage`]) et le fichier qui la porte.
pub fn geometrie_voisine(cfg: &Path) -> Resultat<Option<(GeoSpec, PathBuf)>> {
    Ok(examiner_voisinage(cfg)?.spec)
}

/// Examine TOUTES les métadonnées voisines de la configuration `cfg` (fichier existant) et rend sa géométrie déclarée.
///
/// Dossiers : celui du fichier, rendu CANONIQUE, et son parent canonique (lots `c<k>/`), pour le chemin rendu ABSOLU puis
/// pour le chemin RÉSOLU (liens symboliques) : la réponse ne dépend pas de la façon d'écrire le chemin. Dans chacun : `geometrie.json`, le bloc
/// `geometrie` de `manifest.json` s'il est canonique (il porte un champ `schema` ; un bloc sans `schema` est une
/// description historique libre et n'est pas interprété), et les déclarations historiques : `torsion.json`, champs `torsion`,
/// `bord`, `masque`, `mode` de `manifest.json` et de `config_run.json` (et de leur objet `config_run`).
///
/// Règles (refus = `Erreur::Config`, code 2) : un fichier de métadonnées illisible comme JSON ou une déclaration mal formée
/// est refusé ; la géométrie retenue est le premier `geometrie.json`, sinon le premier bloc canonique, et toute autre
/// déclaration canonique doit lui être ÉGALE ; une torsion historique doit égaler sa torsion ; un bord ou un mode « bord »
/// (R2-3 : liens spatiaux de x0 = 0 gelés) exige un bord `Fixe` sur la direction 0, un masque ou un mode « boite » exige
/// un masque ; sans géométrie canonique, toute déclaration
/// historique non C0 est refusée. Erreurs d'entrée-sortie réelles : code 4.
pub fn examiner_voisinage(cfg: &Path) -> Resultat<GeometrieVoisine> {
    examiner_voisinage_avec(cfg, MutantVoisinage::Aucun)
}

#[doc(hidden)]
pub fn examiner_voisinage_avec(cfg: &Path, mutant: MutantVoisinage) -> Resultat<GeometrieVoisine> {
    let dossiers = dossiers_meta(cfg, mutant)?;
    let mut lus = Vec::new();
    let mut fichiers_geo: Vec<(GeoSpec, PathBuf)> = Vec::new();
    let mut blocs: Vec<(GeoSpec, PathBuf)> = Vec::new();
    let mut torsions: Vec<(Torsion, PathBuf)> = Vec::new();
    // (fichier, true = bord fixe, false = masque)
    let mut non_periodiques: Vec<(PathBuf, bool)> = Vec::new();
    for dossier in &dossiers {
        let g = dossier.join(NOM_FICHIER_GEOMETRIE);
        if existe(&g)? {
            fichiers_geo.push((lire_geometrie(&g)?, g.clone()));
            lus.push(g);
        }
        let t = dossier.join(crate::su2::torsion::NOM_FICHIER_TORSION);
        if existe(&t)? {
            let v = lire_json(&t)?;
            match torsion_declaree(&v, false).map_err(|e| prefixer(e, &t))? {
                Some(tor) => torsions.push((tor, t.clone())),
                None => return Err(mal_formee(&t, "torsion.json vide")),
            }
            lus.push(t);
        }
        for nom in ["manifest.json", "config_run.json"] {
            let p = dossier.join(nom);
            if !existe(&p)? {
                continue;
            }
            let v = lire_json(&p)?;
            lus.push(p.clone());
            if !v.is_object() {
                return Err(mal_formee(&p, "objet JSON attendu"));
            }
            if nom == "manifest.json" && mutant != MutantVoisinage::IgnoreBlocManifeste {
                if let Some(bloc) = v.get("geometrie").filter(|b| b.get("schema").is_some()) {
                    let spec = GeoSpec::depuis_json(bloc).map_err(|e| prefixer(e, &p))?;
                    blocs.push((spec, p.clone()));
                }
            }
            let cr = v.get("config_run").cloned().unwrap_or(Value::Null);
            for champ in [v.get("torsion"), cr.get("torsion")].into_iter().flatten() {
                if let Some(tor) = torsion_declaree(champ, true).map_err(|e| prefixer(e, &p))? {
                    torsions.push((tor, p.clone()));
                }
            }
            let mode = v.get("mode").and_then(Value::as_str).unwrap_or("");
            if [v.get("bord"), cr.get("bord")]
                .into_iter()
                .flatten()
                .any(bord_declare)
                || mode.contains("bord")
            {
                non_periodiques.push((p.clone(), true));
            }
            if [v.get("masque"), cr.get("masque")]
                .into_iter()
                .flatten()
                .any(bord_declare)
                || mode.contains("boite")
            {
                non_periodiques.push((p.clone(), false));
            }
        }
    }
    let spec = fichiers_geo.first().or(blocs.first()).cloned();
    match &spec {
        Some((s, source)) => {
            for (autre, chemin) in fichiers_geo.iter().chain(&blocs) {
                if autre != s {
                    return Err(contradiction(
                        chemin,
                        source,
                        "géométries canoniques différentes",
                    ));
                }
            }
            for (tor, chemin) in &torsions {
                if *tor != s.torsion {
                    return Err(contradiction(chemin, source, "torsion"));
                }
            }
            for (chemin, bord) in &non_periodiques {
                // Les formats historiques n'ont qu'un bord : R2-3, liens spatiaux de x0 = 0 gelés (su2/bord.rs) ⇒ bord
                // `Fixe` sur la direction 0 exactement ; boîte WP-DIR ⇒ masque.
                let accorde = if *bord {
                    matches!(s.bords[0], Bord::Fixe { .. })
                } else {
                    s.masque.is_some()
                };
                if !accorde {
                    return Err(contradiction(
                        chemin,
                        source,
                        if *bord {
                            "bord fixe déclaré"
                        } else {
                            "masque ou boîte déclarés"
                        },
                    ));
                }
            }
        }
        None => {
            if let Some((_, chemin)) = torsions.iter().find(|(tor, _)| !tor.est_triviale()) {
                return Err(sans_geometrie(chemin, "torsion non triviale"));
            }
            if let Some((chemin, _)) = non_periodiques.first() {
                return Err(sans_geometrie(
                    chemin,
                    "bord, masque ou mode non périodique déclaré",
                ));
            }
        }
    }
    Ok(GeometrieVoisine {
        spec,
        dossiers,
        fichiers_lus: lus,
    })
}

fn existe(chemin: &Path) -> Resultat<bool> {
    chemin
        .try_exists()
        .map_err(|e| Erreur::io(format!("recherche {}", chemin.display()), e))
}

fn prefixer(e: Erreur, chemin: &Path) -> Erreur {
    match e {
        Erreur::Config(m) => Erreur::Config(format!("{} : {m}", chemin.display())),
        autre => autre,
    }
}

fn mal_formee(chemin: &Path, raison: &str) -> Erreur {
    Erreur::Config(format!(
        "{} : déclaration de géométrie mal formée ({raison})",
        chemin.display()
    ))
}

fn contradiction(chemin: &Path, source: &Path, quoi: &str) -> Erreur {
    Erreur::Config(format!(
        "{} contredit la géométrie de {} ({quoi})",
        chemin.display(),
        source.display()
    ))
}

fn sans_geometrie(chemin: &Path, raison: &str) -> Erreur {
    Erreur::Config(format!(
        "configuration non C0 sans geometrie.json : {} : {raison}",
        chemin.display()
    ))
}

/// Dossiers examinés : dossier du fichier rendu canonique PUIS son parent, pour le chemin rendu absolu puis pour le chemin
/// résolu (dédoublonnés). Règle du scanner corrigé (`integ/c01b-scanner-corr`, 6cab9d0), à une différence près (relecture
/// c04b, O2) : on remonte depuis le dossier canonique, pas depuis le parent lexical (`c0/../champ.npy` ⇒ `lot/` et son
/// parent, jamais `lot/c0`).
fn dossiers_meta(cfg: &Path, mutant: MutantVoisinage) -> Resultat<Vec<PathBuf>> {
    let io =
        |quoi: &str, p: &Path, e: std::io::Error| Erreur::io(format!("{quoi} {}", p.display()), e);
    if mutant == MutantVoisinage::CheminTelQuEcrit {
        let mut v = Vec::new();
        if let Some(d) = cfg.parent() {
            let d = if d.as_os_str().is_empty() {
                Path::new(".")
            } else {
                d
            };
            v.push(d.to_path_buf());
            if let Some(p) = d.parent().filter(|p| !p.as_os_str().is_empty()) {
                v.push(p.to_path_buf());
            }
        }
        return Ok(v);
    }
    let absolu = std::path::absolute(cfg).map_err(|e| io("chemin absolu de", cfg, e))?;
    let resolu = std::fs::canonicalize(cfg).map_err(|e| io("résolution de", cfg, e))?;
    let mut v: Vec<PathBuf> = Vec::new();
    for chemin in [absolu, resolu] {
        let Some(d) = chemin.parent() else { continue };
        // Le dossier est rendu canonique AVANT de remonter : `c0/../champ.npy` a pour parent `lot/`, jamais `lot/c0`.
        let canon = std::fs::canonicalize(d).map_err(|e| io("résolution de", d, e))?;
        let parent = canon.parent().map(Path::to_path_buf);
        for dossier in std::iter::once(canon).chain(parent) {
            if !v.contains(&dossier) {
                v.push(dossier);
            }
        }
    }
    Ok(v)
}

/// Déclaration historique de torsion, lue STRICTEMENT (O4) : `Ok(None)` seulement pour `null` dans un champ de manifeste
/// (`null_absent`) ; formes admises : texte (« aucune », « 01,23 »), liste de paires, liste de 6 booléens (sérialisation de
/// `Torsion`), objet `{"plans": …}` avec au plus `pile` (texte) et `facteur` (−1) en plus. Tout le reste : refus, code 2.
fn torsion_declaree(v: &Value, null_absent: bool) -> Resultat<Option<Torsion>> {
    let refus = |raison: &str| {
        Erreur::Config(format!(
            "déclaration de torsion mal formée ({raison}) : attendu {{\"plans\": [\"01\", …]}}, une liste de paires ou « aucune »"
        ))
    };
    match v {
        Value::Null if null_absent => Ok(None),
        Value::String(t) => Torsion::analyser(t).map(Some),
        Value::Array(a) => plans_declares(a).map(Some),
        Value::Object(o) => {
            if let Some(cle) = o
                .keys()
                .find(|k| !["plans", "pile", "facteur"].contains(&k.as_str()))
            {
                return Err(refus(&format!("champ inconnu « {cle} »")));
            }
            if o.get("pile").is_some_and(|p| !p.is_string()) {
                return Err(refus("« pile » doit être un texte"));
            }
            if o.get("facteur").is_some_and(|f| f.as_i64() != Some(-1)) {
                return Err(refus("« facteur » doit valoir −1"));
            }
            match o.get("plans") {
                Some(Value::Array(a)) => plans_declares(a).map(Some),
                Some(Value::String(t)) => Torsion::analyser(t).map(Some),
                _ => Err(refus("champ « plans » absent ou mal typé")),
            }
        }
        _ => Err(refus("valeur absente ou mal typée")),
    }
}

/// Liste de plans : paires (« 01 ») ou 6 booléens dans l'ordre de `Torsion::PLANS`.
fn plans_declares(a: &[Value]) -> Resultat<Torsion> {
    if a.len() == 6 && a.iter().all(Value::is_boolean) {
        let plans: Vec<String> = Torsion::PLANS
            .iter()
            .zip(a)
            .filter(|(_, b)| b.as_bool() == Some(true))
            .map(|(&(mu, nu), _)| format!("{mu}{nu}"))
            .collect();
        return Torsion::analyser(&if plans.is_empty() {
            "aucune".to_string()
        } else {
            plans.join(",")
        });
    }
    let textes: Vec<&str> = a
        .iter()
        .map(|p| p.as_str().ok_or_else(|| champ_invalide("torsion.plans")))
        .collect::<Resultat<_>>()?;
    Torsion::analyser(&if textes.is_empty() {
        "aucune".to_string()
    } else {
        textes.join(",")
    })
}

fn bord_declare(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => false,
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

/// Lecture d'un fichier de métadonnées : E/S réelle ⇒ code 4 ; JSON invalide ⇒ déclaration illisible, code 2 (O8).
fn lire_json(chemin: &Path) -> Resultat<Value> {
    let texte = std::fs::read_to_string(chemin)
        .map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    serde_json::from_str(&texte).map_err(|e| {
        Erreur::Config(format!(
            "{} : JSON invalide ({e}) : métadonnée de géométrie illisible",
            chemin.display()
        ))
    })
}
