//! Courbes d'étalonnage ⟨P⟩, ⟨R⟩(β ; L) de Wilson au volume grossier : chaînes CPU courtes, mesures EN LIGNE (aucune configuration écrite).
//!
//! Protocole GELÉ au journal (ligne « WP8 GEL des protocoles », point 5) : L = 4 (et L = 6 en supplément hors gate pour le rapport
//! RG-SCHEMA), grille de 14 β, 3 chaînes par point, départ chaud, 2 000 pas de thermalisation, 20 000 mesures (une par pas ; pas = 1 bain
//! thermique + 4 OR), moteur `CpuSu2` (Philox, graine = clé), graine = base + 100·i + c (c = 1, 2, 3 pour l'étalonnage, 9 pour la
//! chaîne INDÉPENDANTE de ré-estimation de RG-PLAQ), erreur de chaîne √(2 τ var/N) (Sokal c = 6), moyenne des chaînes pondérée 1/σ².
//! Observables : plaquette 1×1 (P) et rectangle 1×2 (R), moyennes sur sites et orientations de ½ReTr.

use super::mcrg2::plaquette_et_rectangle;
use super::plaq::{combiner_chaines, Courbe, PointEtalon};
use super::tau::{stat_serie, StatSerie};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::manifest::{hex, Sha256};
use crate::heritage::su2_hb::CpuSu2;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub const SCHEMA_ETALONNAGE: &str = "kvu-rg-etalonnage/1";
/// Grille de β GELÉE (brief WP8, point 3).
pub const GRILLE_GELEE: [f64; 14] = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0];
/// Grille étendue du protocole WP9.
pub const GRILLE_WP9: [f64; 31] = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0,
    6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0,
];
pub const THERM_GELE: usize = 2000;
pub const MESURES_GELEES: usize = 20000;
pub const NOR_GELE: usize = 4;
pub const N_CHAINES_GELE: usize = 3;
pub const GRAINE_BASE_L4: u64 = 26_102_608_000;
pub const GRAINE_BASE_L6: u64 = 26_102_610_000;
/// Indice de chaîne réservé à la chaîne indépendante de ré-estimation.
pub const C_INDEPENDANTE: u64 = 9;
pub const C_MILIEUX: u64 = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct ParamEtalonnage {
    pub l: usize,
    pub betas: Vec<f64>,
    /// Chaînes par point (étalonnage) ; ignoré si `independante`.
    pub n_chaines: usize,
    pub therm: usize,
    pub mesures: usize,
    pub graine_base: u64,
    /// Une seule chaîne par point, d'indice c = 9 (RG-PLAQ).
    pub independante: bool,
    /// Mutant de contrôle : simule à facteur·β, analyse aux β nominaux. 1.0 en production.
    pub facteur_beta: f64,
}

impl ParamEtalonnage {
    /// Protocole gelé pour L = 4 (graines 26102608xxx) ou L = 6 (supplément, 26102609xxx).
    pub fn gele(l: usize) -> Result<Self, String> {
        let graine_base = match l {
            4 => GRAINE_BASE_L4,
            6 => GRAINE_BASE_L6,
            _ => return Err(format!("pas de protocole gelé pour L = {l} (4 ou 6)")),
        };
        Ok(Self {
            l,
            betas: GRILLE_GELEE.to_vec(),
            n_chaines: N_CHAINES_GELE,
            therm: THERM_GELE,
            mesures: MESURES_GELEES,
            graine_base,
            independante: false,
            facteur_beta: 1.0,
        })
    }

    /// Protocole WP9, une courbe pour chacun des volumes grossiers 4 et 8.
    pub fn wp9(l: usize) -> Result<Self, String> {
        let (graine_base, mesures) = match l {
            4 => (26_102_620_000, 20_000),
            8 => (26_102_624_000, 10_000),
            _ => return Err(format!("pas de protocole wp9 pour L = {l} (4 ou 8)")),
        };
        Ok(Self {
            l, betas: GRILLE_WP9.to_vec(), n_chaines: N_CHAINES_GELE,
            therm: THERM_GELE, mesures, graine_base, independante: false, facteur_beta: 1.0,
        })
    }

    /// Les bases réservées identifient WP9 sans changer la structure publique de WP8.
    pub fn est_wp9(&self) -> bool {
        matches!((self.l, self.graine_base), (4, 26_102_620_000) | (8, 26_102_624_000))
    }

    pub fn graine(&self, i: usize, c: u64) -> u64 {
        self.graine_base + 100 * i as u64 + c
    }

    /// Validation WP9 aux milieux (une chaîne c = 8 par milieu). Pas de champ dédié dans la structure (les constructions littérales des
    /// tests existants de WP8 doivent rester valides) : un protocole « milieux » se reconnaît à sa forme exacte — base wp9, une chaîne,
    /// ni indépendante ni mutant, et les 30 milieux exacts de `GRILLE_WP9`.
    pub fn est_milieux(&self) -> bool {
        self.est_wp9()
            && !self.independante
            && self.facteur_beta == 1.0
            && self.n_chaines == 1
            && self.betas.len() == GRILLE_WP9.len() - 1
            && GRILLE_WP9.windows(2).zip(&self.betas).all(|(w, &b)| b == (w[0] + w[1]) / 2.0)
    }

    pub fn activer_milieux(&mut self) -> Result<(), String> {
        if !self.est_wp9() || self.independante || self.facteur_beta != 1.0 || self.est_milieux() || self.betas != GRILLE_WP9.to_vec() {
            return Err("--milieux exige wp9, sans --independante ni --facteur-beta".into());
        }
        self.betas = self.betas.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        self.n_chaines = 1;
        Ok(())
    }
}

/// Une chaîne mesurée (statistiques seulement ; la série brute n'est pas conservée, son empreinte l'est).
#[derive(Clone, Debug, PartialEq)]
pub struct ChaineMesuree {
    pub indice_beta: usize,
    pub beta: f64,
    pub beta_simule: f64,
    pub graine: u64,
    pub c: u64,
    pub p: StatSerie,
    pub r: StatSerie,
    pub empreinte: String,
}

/// Une chaîne : (série P, série R). Départ chaud, `therm` pas jetés puis `mesures` pas mesurés.
pub fn lancer_chaine(l: usize, beta: f64, graine: u64, therm: usize, mesures: usize) -> Resultat<(Vec<f64>, Vec<f64>)> {
    let mut moteur = CpuSu2::new(l, graine).map_err(Erreur::Config)?;
    moteur.init(true);
    for _ in 0..therm {
        moteur.step(beta, NOR_GELE);
    }
    let mut champ = ChampHote { l, q: vec![0.0; 16 * l.pow(4)] };
    let (mut p, mut r) = (Vec::with_capacity(mesures), Vec::with_capacity(mesures));
    for _ in 0..mesures {
        moteur.step(beta, NOR_GELE);
        for (dst, src) in champ.q.chunks_exact_mut(4).zip(&moteur.u) {
            dst.copy_from_slice(src);
        }
        let o = plaquette_et_rectangle(&champ)?;
        p.push(o.plaquette);
        r.push(o.rectangle);
    }
    Ok((p, r))
}

fn empreinte(p: &[f64], r: &[f64]) -> String {
    let mut h = Sha256::new();
    for x in p.iter().chain(r) {
        h.update(&x.to_bits().to_le_bytes());
    }
    hex(&h.finalize())
}

/// Lance toutes les chaînes du protocole (en parallèle sur `fils` fils, une chaîne par fil, moteur à 1 fil) ; sortie triée par (i, c),
/// indépendante du nombre de fils.
pub fn etalonner(par: &ParamEtalonnage, fils: usize) -> Resultat<Vec<ChaineMesuree>> {
    if par.betas.is_empty() || par.mesures < 100 || fils == 0 {
        return Err(Erreur::Config("protocole d'étalonnage invalide".into()));
    }
    let cs: Vec<u64> = if par.est_milieux() { vec![C_MILIEUX] } else if par.independante { vec![C_INDEPENDANTE] } else { (1..=par.n_chaines as u64).collect() };
    let travaux: Vec<(usize, u64)> = (0..par.betas.len()).flat_map(|i| cs.iter().map(move |&c| (i, c))).collect();
    let suivant = AtomicUsize::new(0);
    let sorties: Mutex<Vec<Option<Resultat<ChaineMesuree>>>> = Mutex::new((0..travaux.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..fils.min(travaux.len()) {
            s.spawn(|| loop {
                let k = suivant.fetch_add(1, Ordering::SeqCst);
                if k >= travaux.len() {
                    break;
                }
                let (i, c) = travaux[k];
                let res = (|| -> Resultat<ChaineMesuree> {
                    let beta = par.betas[i];
                    let beta_simule = beta * par.facteur_beta;
                    let graine = par.graine(i, c);
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .map_err(|e| Erreur::Config(format!("pool rayon : {e}")))?;
                    let (p, r) = pool.install(|| lancer_chaine(par.l, beta_simule, graine, par.therm, par.mesures))?;
                    Ok(ChaineMesuree {
                        indice_beta: i,
                        beta,
                        beta_simule,
                        graine,
                        c,
                        p: stat_serie(&p),
                        r: stat_serie(&r),
                        empreinte: empreinte(&p, &r),
                    })
                })();
                if let Ok(mut g) = sorties.lock() {
                    g[k] = Some(res);
                }
            });
        }
    });
    let g = sorties.into_inner().map_err(|_| Erreur::Config("verrou empoisonné".into()))?;
    g.into_iter().map(|o| o.unwrap_or_else(|| Err(Erreur::Config("chaîne non exécutée".into())))).collect()
}

/// Courbe d'étalonnage : par β, moyenne des chaînes pondérée 1/σ² (P et R séparément).
pub fn courbe_depuis(l: usize, chaines: &[ChaineMesuree]) -> Resultat<Courbe> {
    let mut indices: Vec<usize> = chaines.iter().map(|c| c.indice_beta).collect();
    indices.sort_unstable();
    indices.dedup();
    let mut points = Vec::new();
    for i in indices {
        let du_point: Vec<&ChaineMesuree> = chaines.iter().filter(|c| c.indice_beta == i).collect();
        let cp = combiner_chaines(&du_point.iter().map(|c| (c.p.moy, c.p.err)).collect::<Vec<_>>());
        let cr = combiner_chaines(&du_point.iter().map(|c| (c.r.moy, c.r.err)).collect::<Vec<_>>());
        let (Some((p, sp)), Some((r, sr))) = (cp, cr) else {
            return Err(Erreur::Validation(format!("point β = {} : erreurs de chaîne inutilisables", du_point[0].beta)));
        };
        points.push(PointEtalon { beta: du_point[0].beta, p, sigma_p: sp, r, sigma_r: sr, n_chaines: du_point.len() });
    }
    Courbe::nouvelle(l, points).map_err(Erreur::Validation)
}

fn stat_json(s: &StatSerie) -> Value {
    json!({"moy": s.moy, "err": s.err, "tau": s.tau, "fenetre": s.fenetre, "dtau": s.dtau, "n": s.n, "sd": s.sd})
}

fn stat_lue(v: &Value) -> Option<StatSerie> {
    Some(StatSerie {
        moy: v["moy"].as_f64()?,
        err: v["err"].as_f64()?,
        tau: v["tau"].as_f64()?,
        fenetre: v["fenetre"].as_u64()? as usize,
        dtau: v["dtau"].as_f64()?,
        n: v["n"].as_u64()? as usize,
        sd: v["sd"].as_f64()?,
    })
}

/// Document JSON d'un étalonnage (protocole, chaînes, courbe) : aucune durée, aucun nombre de fils (déterministe).
pub fn en_json(par: &ParamEtalonnage, chaines: &[ChaineMesuree]) -> Resultat<Value> {
    let courbe = courbe_depuis(par.l, chaines)?;
    let mut doc = json!({
        "format": SCHEMA_ETALONNAGE,
        "l": par.l,
        "betas": par.betas,
        "n_chaines": if par.independante { 1 } else { par.n_chaines },
        "therm": par.therm,
        "mesures": par.mesures,
        "nor": NOR_GELE,
        "graine_base": par.graine_base,
        "independante": par.independante,
        "facteur_beta": par.facteur_beta,
        "moteur": "CpuSu2 (Philox4x32-10, graine = clé, départ chaud)",
        "chaines": chaines.iter().map(|c| json!({
            "indice_beta": c.indice_beta, "beta": c.beta, "beta_simule": c.beta_simule, "graine": c.graine, "c": c.c,
            "p": stat_json(&c.p), "r": stat_json(&c.r), "empreinte_series": c.empreinte,
        })).collect::<Vec<_>>(),
        "courbe": courbe.points.iter().map(|p| json!({
            "beta": p.beta, "p": p.p, "sigma_p": p.sigma_p, "r": p.r, "sigma_r": p.sigma_r, "n_chaines": p.n_chaines,
        })).collect::<Vec<_>>(),
    });
    if par.est_wp9() {
        doc["protocole"] = json!("wp9");
    }
    if par.est_milieux() {
        doc["milieux"] = json!(true);
    }
    Ok(doc)
}

/// Étalonnage relu depuis son JSON : (protocole, chaînes). La courbe se recalcule à partir des chaînes.
pub fn depuis_json(v: &Value) -> Resultat<(ParamEtalonnage, Vec<ChaineMesuree>)> {
    let mal = || Erreur::Format("étalonnage : JSON incomplet ou mal formé".into());
    if v["format"].as_str() != Some(SCHEMA_ETALONNAGE) {
        return Err(Erreur::Format(format!("étalonnage : format attendu {SCHEMA_ETALONNAGE}")));
    }
    let par = ParamEtalonnage {
        l: v["l"].as_u64().ok_or_else(mal)? as usize,
        betas: v["betas"].as_array().ok_or_else(mal)?.iter().map(|x| x.as_f64()).collect::<Option<Vec<_>>>().ok_or_else(mal)?,
        n_chaines: v["n_chaines"].as_u64().ok_or_else(mal)? as usize,
        therm: v["therm"].as_u64().ok_or_else(mal)? as usize,
        mesures: v["mesures"].as_u64().ok_or_else(mal)? as usize,
        graine_base: v["graine_base"].as_u64().ok_or_else(mal)?,
        independante: v["independante"].as_bool().ok_or_else(mal)?,
        facteur_beta: v["facteur_beta"].as_f64().ok_or_else(mal)?,
    };
    let mut chaines = Vec::new();
    for c in v["chaines"].as_array().ok_or_else(mal)? {
        chaines.push(ChaineMesuree {
            indice_beta: c["indice_beta"].as_u64().ok_or_else(mal)? as usize,
            beta: c["beta"].as_f64().ok_or_else(mal)?,
            beta_simule: c["beta_simule"].as_f64().ok_or_else(mal)?,
            graine: c["graine"].as_u64().ok_or_else(mal)?,
            c: c["c"].as_u64().ok_or_else(mal)?,
            p: stat_lue(&c["p"]).ok_or_else(mal)?,
            r: stat_lue(&c["r"]).ok_or_else(mal)?,
            empreinte: c["empreinte_series"].as_str().ok_or_else(mal)?.to_string(),
        });
    }
    Ok((par, chaines))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocole_gele() {
        let p = ParamEtalonnage::gele(4).unwrap();
        assert_eq!(p.betas.len(), 14);
        assert_eq!(p.graine(0, 1), 26_102_608_001);
        assert_eq!(p.graine(13, 9), 26_102_609_309);
        assert_eq!(ParamEtalonnage::gele(6).unwrap().graine(2, 3), 26_102_610_203);
        assert!(ParamEtalonnage::gele(5).is_err());
    }

    #[test]
    fn chaine_courte_deterministe_et_plaquette_du_moteur() {
        let (p1, r1) = lancer_chaine(4, 2.3, 7, 20, 30).unwrap();
        let (p2, r2) = lancer_chaine(4, 2.3, 7, 20, 30).unwrap();
        assert_eq!((p1.clone(), r1.clone()), (p2, r2));
        assert!(p1.iter().all(|&x| x > 0.3 && x < 1.0) && r1.iter().all(|&x| x > 0.0 && x < 1.0));
        // la plaquette mesurée = celle du moteur (arbre de réduction du GPU) à 1e-12
        let mut m = CpuSu2::new(4, 7).unwrap();
        m.init(true);
        for _ in 0..21 {
            m.step(2.3, NOR_GELE);
        }
        let mut champ = ChampHote { l: 4, q: vec![0.0; 16 * 256] };
        for (d, s) in champ.q.chunks_exact_mut(4).zip(&m.u) {
            d.copy_from_slice(s);
        }
        let o = plaquette_et_rectangle(&champ).unwrap();
        assert!((o.plaquette - m.measure()[0]).abs() < 1e-12, "{} {}", o.plaquette, m.measure()[0]);
        assert_eq!(o.plaquette, p1[0]);
    }

    #[test]
    fn courbe_et_json_aller_retour_et_fils() {
        let par = ParamEtalonnage { l: 4, betas: vec![2.0, 3.0, 4.0], n_chaines: 2, therm: 50, mesures: 400, graine_base: 5000, independante: false, facteur_beta: 1.0 };
        let a = etalonner(&par, 1).unwrap();
        let b = etalonner(&par, 4).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 6);
        let j = en_json(&par, &a).unwrap();
        let (par2, ch2) = depuis_json(&j).unwrap();
        assert_eq!(par2, par);
        assert_eq!(ch2, a);
        let c = courbe_depuis(4, &a).unwrap();
        assert_eq!(c.points.len(), 3);
        assert!(c.points[0].p < c.points[1].p && c.points[1].p < c.points[2].p);
    }
}
