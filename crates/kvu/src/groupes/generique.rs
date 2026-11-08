//! Moteur CPU monofil des groupes matriciels, protocole GATES v4.4.

use super::Cmat;
use crate::commun::{Depart, Erreur, Graine, Resultat};
pub use crate::heritage::su2_hb::Geo;
use crate::heritage::su2_hb::{k_final, k_reduce, philox4x32_10, u01};
use std::f64::consts::TAU;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat<const D: usize> {
    pub re: [[f64; D]; D],
    pub im: [[f64; D]; D],
}

impl<const D: usize> Mat<D> {
    #[inline]
    pub fn zero() -> Self {
        Self {
            re: [[0.; D]; D],
            im: [[0.; D]; D],
        }
    }
    #[inline]
    pub fn identite() -> Self {
        let mut m = Self::zero();
        for i in 0..D {
            m.re[i][i] = 1.;
        }
        m
    }
    #[inline]
    pub fn mul(&self, b: &Self) -> Self {
        let mut m = Self::zero();
        for i in 0..D {
            for j in 0..D {
                for k in 0..D {
                    m.re[i][j] += self.re[i][k] * b.re[k][j] - self.im[i][k] * b.im[k][j];
                    m.im[i][j] += self.re[i][k] * b.im[k][j] + self.im[i][k] * b.re[k][j];
                }
            }
        }
        m
    }
    #[inline]
    pub fn dag(&self) -> Self {
        let mut m = Self::zero();
        for i in 0..D {
            for j in 0..D {
                m.re[i][j] = self.re[j][i];
                m.im[i][j] = -self.im[j][i];
            }
        }
        m
    }
    #[inline]
    pub fn add(&self, b: &Self) -> Self {
        let mut m = *self;
        for i in 0..D {
            for j in 0..D {
                m.re[i][j] += b.re[i][j];
                m.im[i][j] += b.im[i][j];
            }
        }
        m
    }
    #[inline]
    pub fn sub(&self, b: &Self) -> Self {
        self.add(&b.echelle(-1.))
    }
    #[inline]
    pub fn echelle(&self, c: f64) -> Self {
        let mut m = *self;
        for i in 0..D {
            for j in 0..D {
                m.re[i][j] *= c;
                m.im[i][j] *= c;
            }
        }
        m
    }
    #[inline]
    pub fn re_tr(&self) -> f64 {
        (0..D).map(|i| self.re[i][i]).sum()
    }
    #[inline]
    pub fn re_tr_prod(&self, b: &Self) -> f64 {
        let mut tr = 0.;
        for i in 0..D {
            for k in 0..D {
                tr += self.re[i][k] * b.re[k][i] - self.im[i][k] * b.im[k][i];
            }
        }
        tr
    }
    #[inline]
    pub fn depuis_cmat(m: &Cmat) -> Self {
        assert_eq!(m.n, D, "Dimension matricielle incompatible");
        assert_eq!(m.re.len(), D * D);
        assert_eq!(m.im.len(), D * D);
        let mut a = Self::zero();
        for i in 0..D {
            for j in 0..D {
                a.re[i][j] = m.re[i * D + j];
                a.im[i][j] = m.im[i * D + j];
            }
        }
        a
    }
    /// La conversion vers le stockage dynamique Cmat alloue ses deux tableaux.
    #[inline]
    pub fn vers_cmat(&self) -> Cmat {
        let mut m = Cmat::zeros(D);
        for i in 0..D {
            for j in 0..D {
                m.re[i * D + j] = self.re[i][j];
                m.im[i * D + j] = self.im[i][j];
            }
        }
        m
    }
}

pub const TAG_METRO: u32 = 0x4D45_5452;
pub const TAG_INIT_GEN: u32 = 0x494E_4954;

pub trait Noyau<const D: usize>: Sync {
    fn nom(&self) -> &'static str;
    fn d_adj(&self) -> usize;
    fn exp_i(&self, c: &[f64]) -> Mat<D>;
    fn reunitariser(&self, u: &mut Mat<D>);
    fn defaut(&self, u: &Mat<D>) -> f64;
    fn haar_philox(&self, i: u32, slot: u32, k0: u32, k1: u32, etiquette: u32) -> Mat<D>;
    fn a_surrelaxation(&self) -> bool {
        false
    }
    fn surrelaxer(&self, _u: &mut Mat<D>, _a: &Mat<D>) {}
}

pub struct Reseau<const D: usize> {
    pub geo: Geo,
    pub liens: Vec<Mat<D>>,
    pub beta: f64,
    pub eps: f64,
    pub graine: u64,
    pub pas_faits: u64,
    pub suivre_defaut: bool,
    pub defaut_avant_reproj: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StatsPas {
    pub acceptees: u64,
    pub propositions: u64,
}

fn verifier_l(l: usize) -> Resultat<()> {
    if l < 2
        || !l.is_multiple_of(2)
        || l.checked_pow(4)
            .and_then(|v| v.checked_mul(4))
            .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(Erreur::Config(
            "L doit être pair, ≥ 2, avec 4 L⁴ ≤ u32::MAX (compteurs Philox)".into(),
        ));
    }
    Ok(())
}

impl<const D: usize> Reseau<D> {
    /// Constructeur direct ; les entrées externes passent par `essayer_nouveau`.
    pub fn nouveau(noyau: &dyn Noyau<D>, l: usize, beta: f64, graine: u64, depart: Depart) -> Self {
        match Self::essayer_nouveau(noyau, l, beta, graine, depart) {
            Ok(reseau) => reseau,
            Err(erreur) => std::panic::panic_any(erreur),
        }
    }

    /// Variante contrôlée : L invalide et β non fini rendent une `Erreur::Config`.
    pub fn essayer_nouveau(
        noyau: &dyn Noyau<D>,
        l: usize,
        beta: f64,
        graine: u64,
        depart: Depart,
    ) -> Resultat<Self> {
        verifier_l(l)?;
        if D == 0 || !beta.is_finite() {
            return Err(Erreur::Config("Dimension nulle ou β non fini".into()));
        }
        let geo = Geo::new(l);
        let (k0, k1) = Graine(graine).cle();
        let liens = (0..4 * geo.v)
            .map(|li| match depart {
                Depart::Froid => Mat::identite(),
                Depart::Chaud => noyau.haar_philox(li as u32, u32::MAX, k0, k1, TAG_INIT_GEN),
            })
            .collect();
        Ok(Self {
            geo,
            liens,
            beta,
            eps: 0.3,
            graine,
            pas_faits: 0,
            suivre_defaut: false,
            defaut_avant_reproj: 0.,
        })
    }

    pub fn agrafe(&self, mu: usize, s: usize, x: &[usize; 4]) -> Mat<D> {
        let g = &self.geo;
        let ld = |nu, site| self.liens[nu * g.v + site];
        let smu = g.nb_up(s, mu, x[mu]);
        let mut a = Mat::zero();
        for (nu, &xnu) in x.iter().enumerate() {
            if nu == mu {
                continue;
            }
            let snu = g.nb_up(s, nu, xnu);
            let sdn = g.nb_dn(s, nu, xnu);
            let smu_dn = g.nb_dn(smu, nu, xnu);
            let t = ld(nu, smu).mul(&ld(mu, snu).dag()).mul(&ld(nu, s).dag());
            let b = ld(nu, smu_dn)
                .dag()
                .mul(&ld(mu, sdn).dag())
                .mul(&ld(nu, sdn));
            a = a.add(&t.add(&b));
        }
        a
    }

    pub fn pas(&mut self, noyau: &dyn Noyau<D>) -> StatsPas {
        // Une sous-algèbre de u(D) a au plus D² générateurs réels.
        let d = noyau.d_adj();
        assert!(d > 0 && d <= D * D, "Dimension d'algèbre incompatible");
        assert!(
            self.pas_faits <= u32::MAX as u64,
            "Compteur Philox des pas épuisé"
        );
        let mut tampon = [[0.; D]; D];
        let c = tampon.as_flattened_mut();
        let (k0, k1) = Graine(self.graine).cle();
        let pas = self.pas_faits as u32;
        let paires = d.div_ceil(2) as u32;
        let mut stats = StatsPas::default();
        self.defaut_avant_reproj = 0.;
        for mu in 0..4 {
            for parite in 0..2 {
                for t in 0..self.geo.v / 2 {
                    let (s, x) = self.geo.site_of(t, parite);
                    let li = mu * self.geo.v + s;
                    let a = self.agrafe(mu, s, &x);
                    let mut u = self.liens[li];
                    let mut trace = u.re_tr_prod(&a);
                    for p in 0..10 {
                        for j in 0..paires {
                            let w = philox4x32_10(
                                [pas, li as u32, (paires + 1) * p + j, TAG_METRO],
                                k0,
                                k1,
                            );
                            let r = (-2. * u01(w[0], w[1]).ln()).sqrt();
                            let (sin, cos) = (TAU * u01(w[2], w[3])).sin_cos();
                            c[2 * j as usize] = self.eps * (r * cos);
                            if 2 * j as usize + 1 < d {
                                c[2 * j as usize + 1] = self.eps * (r * sin);
                            }
                        }
                        let w = philox4x32_10(
                            [pas, li as u32, (paires + 1) * p + paires, TAG_METRO],
                            k0,
                            k1,
                        );
                        let alea = u01(w[0], w[1]);
                        let proposition = noyau.exp_i(&c[..d]).mul(&u);
                        let nouvelle_trace = proposition.re_tr_prod(&a);
                        if alea < ((self.beta / D as f64) * (nouvelle_trace - trace)).exp() {
                            u = proposition;
                            trace = nouvelle_trace;
                            stats.acceptees += 1;
                        }
                        stats.propositions += 1;
                    }
                    self.reprojeter(noyau, &mut u);
                    self.liens[li] = u;
                }
            }
        }
        if noyau.a_surrelaxation() {
            for mu in 0..4 {
                for parite in 0..2 {
                    for t in 0..self.geo.v / 2 {
                        let (s, x) = self.geo.site_of(t, parite);
                        let li = mu * self.geo.v + s;
                        let a = self.agrafe(mu, s, &x);
                        let mut u = self.liens[li];
                        noyau.surrelaxer(&mut u, &a);
                        self.reprojeter(noyau, &mut u);
                        self.liens[li] = u;
                    }
                }
            }
        }
        self.pas_faits += 1;
        stats
    }

    fn reprojeter(&mut self, noyau: &dyn Noyau<D>, u: &mut Mat<D>) {
        if self.suivre_defaut {
            self.defaut_avant_reproj = self.defaut_avant_reproj.max(noyau.defaut(u));
        }
        noyau.reunitariser(u);
    }

    pub fn thermaliser(&mut self, noyau: &dyn Noyau<D>, n: u64) {
        self.thermaliser_avec(noyau, n, 0, 0, |_, _| Ok(()))
            .expect("Le rappel sans E/S ne peut pas échouer");
    }

    /// Le rappel reçoit aussi les acceptations du bloc incomplet : la reprise
    /// pendant la thermalisation conserve exactement l'adaptation de ε.
    fn thermaliser_avec(
        &mut self,
        noyau: &dyn Noyau<D>,
        n: u64,
        deja: u64,
        mut accepte: u64,
        mut sauvegarder: impl FnMut(&Self, u64) -> Resultat<()>,
    ) -> Resultat<()> {
        let debut = Instant::now();
        for k in 1..=n {
            accepte += self.pas(noyau).acceptees;
            if (deja + k).is_multiple_of(50) {
                let propose = 10 * 4 * self.geo.v as u64 * 50;
                let taux = accepte as f64 / propose as f64;
                if taux > 0.6 {
                    self.eps *= 1.1;
                } else if taux < 0.4 {
                    self.eps *= 0.9;
                }
                accepte = 0;
            }
            sauvegarder(self, accepte)?;
            if self.pas_faits.is_multiple_of(1000) {
                progression(noyau.nom(), self.pas_faits, k, debut);
            }
        }
        Ok(())
    }

    pub fn plaquette(&self) -> f64 {
        let g = &self.geo;
        let ld = |nu, site| self.liens[nu * g.v + site];
        let sites: Vec<f64> = (0..g.v)
            .map(|s| {
                let x = g.coords(s);
                let mut somme = 0.;
                for mu in 0..4 {
                    let smu = g.nb_up(s, mu, x[mu]);
                    for (nu, &xnu) in x.iter().enumerate().skip(mu + 1) {
                        let snu = g.nb_up(s, nu, xnu);
                        let p = ld(mu, s)
                            .mul(&ld(nu, smu))
                            .mul(&ld(mu, snu).dag().mul(&ld(nu, s).dag()));
                        somme += p.re_tr() / D as f64;
                    }
                }
                somme
            })
            .collect();
        k_final(&k_reduce(&sites), 1. / (6. * g.v as f64))
    }

    pub fn defaut_max(&self, noyau: &dyn Noyau<D>) -> f64 {
        self.liens
            .iter()
            .map(|u| noyau.defaut(u))
            .fold(0., f64::max)
    }
}

fn progression(nom: &str, total: u64, faits: u64, debut: Instant) {
    eprintln!(
        "{nom} : {total} pas, {:.3} pas/s",
        faits as f64 / debut.elapsed().as_secs_f64()
    );
}

#[derive(Clone, Debug)]
pub struct SpecChaine {
    pub id: String,
    pub l: usize,
    pub beta: f64,
    pub graine: u64,
    pub depart: Depart,
    pub therm: u64,
    pub mesures: u64,
}
#[derive(Clone, Debug)]
pub struct Serie {
    pub spec: SpecChaine,
    pub eps: f64,
    pub valeurs: Vec<f64>,
}

pub fn chemin_serie(dossier: &Path, id: &str) -> PathBuf {
    dossier.join(format!("{id}.serie"))
}

fn nom_depart(d: Depart) -> &'static str {
    match d {
        Depart::Chaud => "chaud",
        Depart::Froid => "froid",
    }
}

fn entete(spec: &SpecChaine, groupe: &str, eps: f64) -> String {
    format!("# id = {}\n# groupe = {}\n# L = {}\n# beta = {:.17e}\n# graine = {}\n# depart = {}\n# therm = {}\n# mesures = {}\n# eps = {:.17e}\n", spec.id, groupe, spec.l, spec.beta, spec.graine, nom_depart(spec.depart), spec.therm, spec.mesures, eps)
}

fn format_invalide(message: impl Into<String>) -> Erreur {
    Erreur::Format(message.into())
}

fn analyser_entete(texte: &str) -> Resultat<(SpecChaine, String, f64)> {
    let mut champs = std::collections::BTreeMap::new();
    for ligne in texte
        .lines()
        .filter(|l| l.starts_with("# ") && !l.starts_with("# fin "))
    {
        let (cle, valeur) = ligne[2..]
            .split_once(" = ")
            .ok_or_else(|| format_invalide("En-tête de série invalide"))?;
        if champs.insert(cle, valeur).is_some() {
            return Err(format_invalide("Clé dupliquée"));
        }
    }
    let champ = |cle| {
        champs
            .get(cle)
            .copied()
            .ok_or_else(|| format_invalide(format!("Champ absent : {cle}")))
    };
    let entier = |cle| -> Resultat<u64> {
        champ(cle)?
            .parse()
            .map_err(|_| format_invalide(format!("Entier invalide : {cle}")))
    };
    let reel = |cle| -> Resultat<f64> {
        let x: f64 = champ(cle)?
            .parse()
            .map_err(|_| format_invalide(format!("Réel invalide : {cle}")))?;
        if !x.is_finite() {
            return Err(format_invalide("Réel non fini"));
        }
        Ok(x)
    };
    let l = usize::try_from(entier("L")?).map_err(|_| format_invalide("L trop grand"))?;
    verifier_l(l)?;
    let depart = match champ("depart")? {
        "chaud" => Depart::Chaud,
        "froid" => Depart::Froid,
        _ => return Err(format_invalide("Départ inconnu")),
    };
    let spec = SpecChaine {
        id: champ("id")?.into(),
        l,
        beta: reel("beta")?,
        graine: entier("graine")?,
        depart,
        therm: entier("therm")?,
        mesures: entier("mesures")?,
    };
    let eps = reel("eps")?;
    if eps <= 0. {
        return Err(format_invalide("ε doit être positif"));
    }
    Ok((spec, champ("groupe")?.into(), eps))
}

fn lire_serie_complete(chemin: &Path) -> Resultat<(Serie, String)> {
    let texte =
        fs::read_to_string(chemin).map_err(|e| Erreur::io(chemin.display().to_string(), e))?;
    let (spec, groupe, eps) = analyser_entete(&texte)?;
    let mut valeurs = Vec::new();
    let mut fin = None;
    for ligne in texte.lines() {
        if let Some(n) = ligne.strip_prefix("# fin n = ") {
            if fin.is_some() {
                return Err(format_invalide("Fin dupliquée"));
            }
            fin = Some(
                n.parse::<usize>()
                    .map_err(|_| format_invalide("Fin invalide"))?,
            );
        } else if !ligne.starts_with('#') && !ligne.trim().is_empty() {
            if fin.is_some() {
                return Err(format_invalide("Valeur après la fin"));
            }
            let x: f64 = ligne
                .parse()
                .map_err(|_| format_invalide("Valeur de série invalide"))?;
            if !x.is_finite() {
                return Err(format_invalide("Valeur de série non finie"));
            }
            valeurs.push(x);
        }
    }
    if fin != Some(valeurs.len()) || spec.mesures != valeurs.len() as u64 {
        return Err(format_invalide("Série tronquée ou longueur incohérente"));
    }
    Ok((Serie { spec, eps, valeurs }, groupe))
}

pub fn lire_serie(chemin: &Path) -> Resultat<Serie> {
    Ok(lire_serie_complete(chemin)?.0)
}

fn compatibles(a: &SpecChaine, b: &SpecChaine) -> bool {
    a.id == b.id
        && a.l == b.l
        && a.beta.to_bits() == b.beta.to_bits()
        && a.graine == b.graine
        && a.depart == b.depart
        && a.therm == b.therm
}

fn atomique(chemin: &Path, contenu: &[u8]) -> Resultat<()> {
    let temporaire = chemin.with_extension(format!("tmp-{}", std::process::id()));
    let resultat = (|| {
        let mut fichier = fs::File::create(&temporaire)
            .map_err(|e| Erreur::io(temporaire.display().to_string(), e))?;
        fichier
            .write_all(contenu)
            .and_then(|()| fichier.sync_all())
            .map_err(|e| Erreur::io(temporaire.display().to_string(), e))?;
        fs::rename(&temporaire, chemin).map_err(|e| Erreur::io(chemin.display().to_string(), e))
    })();
    if resultat.is_err() {
        let _ = fs::remove_file(&temporaire);
    }
    resultat
}

// FNV-1a, sur l'en-tête ET tout le champ, avec vérification avant décodage.
fn somme_controle(octets: &[u8]) -> u64 {
    octets.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    })
}
fn pousser_u64(o: &mut Vec<u8>, n: u64) {
    o.extend_from_slice(&n.to_le_bytes());
}
const MAGIE: &[u8; 8] = b"KVUGEN01";

fn ecrire_reprise<const D: usize>(
    chemin: &Path,
    noyau: &dyn Noyau<D>,
    r: &Reseau<D>,
    serie: &Serie,
    accepte_bloc: u64,
) -> Resultat<()> {
    let tete = entete(&serie.spec, noyau.nom(), r.eps);
    let mut o = Vec::new();
    o.extend_from_slice(MAGIE);
    pousser_u64(&mut o, D as u64);
    pousser_u64(&mut o, tete.len() as u64);
    o.extend_from_slice(tete.as_bytes());
    pousser_u64(&mut o, r.pas_faits);
    pousser_u64(&mut o, r.eps.to_bits());
    pousser_u64(&mut o, serie.valeurs.len() as u64);
    pousser_u64(&mut o, accepte_bloc);
    for &v in &serie.valeurs {
        pousser_u64(&mut o, v.to_bits());
    }
    for u in &r.liens {
        for i in 0..D {
            for j in 0..D {
                pousser_u64(&mut o, u.re[i][j].to_bits());
                pousser_u64(&mut o, u.im[i][j].to_bits());
            }
        }
    }
    let somme = somme_controle(&o);
    pousser_u64(&mut o, somme);
    atomique(chemin, &o)
}

struct Lecteur<'a> {
    reste: &'a [u8],
}
impl<'a> Lecteur<'a> {
    fn prendre(&mut self, n: usize) -> Resultat<&'a [u8]> {
        if n > self.reste.len() {
            return Err(format_invalide("Reprise tronquée"));
        }
        let (a, b) = self.reste.split_at(n);
        self.reste = b;
        Ok(a)
    }
    fn entier(&mut self) -> Resultat<u64> {
        let mut b = [0; 8];
        b.copy_from_slice(self.prendre(8)?);
        Ok(u64::from_le_bytes(b))
    }
    fn reel(&mut self) -> Resultat<f64> {
        let x = f64::from_bits(self.entier()?);
        if !x.is_finite() {
            return Err(format_invalide("Reprise : réel non fini"));
        }
        Ok(x)
    }
}

fn lire_reprise<const D: usize>(
    chemin: &Path,
    noyau: &dyn Noyau<D>,
    voulu: &SpecChaine,
) -> Resultat<(Reseau<D>, Serie, u64)> {
    let o = fs::read(chemin).map_err(|e| Erreur::io(chemin.display().to_string(), e))?;
    if o.len() < 16 {
        return Err(format_invalide("Reprise trop courte"));
    }
    let (donnees, somme) = o.split_at(o.len() - 8);
    let mut lecteur_somme = Lecteur { reste: somme };
    if somme_controle(donnees) != lecteur_somme.entier()? {
        return Err(format_invalide("Somme de contrôle de reprise incorrecte"));
    }
    let mut l = Lecteur { reste: donnees };
    if l.prendre(8)? != MAGIE || l.entier()? != D as u64 {
        return Err(format_invalide("Magie ou dimension de reprise incorrecte"));
    }
    let taille = usize::try_from(l.entier()?).map_err(|_| format_invalide("En-tête trop long"))?;
    let texte = std::str::from_utf8(l.prendre(taille)?)
        .map_err(|_| format_invalide("En-tête non UTF-8"))?;
    let (spec, groupe, eps) = analyser_entete(texte)?;
    if !compatibles(&spec, voulu) || groupe != noyau.nom() {
        return Err(Erreur::Config("Paramètres de reprise incompatibles".into()));
    }
    let pas = l.entier()?;
    let eps_binaire = l.reel()?;
    let n = l.entier()?;
    let accepte_bloc = l.entier()?;
    let en_thermalisation = pas < spec.therm;
    let compteurs_valides = if en_thermalisation {
        n == 0 && accepte_bloc <= 10 * 4 * Geo::new(spec.l).v as u64 * (pas % 50)
    } else {
        spec.therm.checked_add(n) == Some(pas) && accepte_bloc == 0
    };
    if eps.to_bits() != eps_binaire.to_bits()
        || n > spec.mesures
        || !compteurs_valides
        || pas > u32::MAX as u64 + 1
    {
        return Err(format_invalide("Compteurs ou ε de reprise incohérents"));
    }
    let geo = Geo::new(spec.l);
    let nombre = usize::try_from(n).map_err(|_| format_invalide("Série trop longue"))?;
    let attendu = nombre.checked_mul(8).and_then(|v| {
        (4 * geo.v)
            .checked_mul(D)?
            .checked_mul(D)?
            .checked_mul(16)?
            .checked_add(v)
    });
    if attendu != Some(l.reste.len()) {
        return Err(format_invalide("Taille de reprise incohérente"));
    }
    let mut valeurs = Vec::with_capacity(nombre);
    for _ in 0..n {
        valeurs.push(l.reel()?);
    }
    let mut liens = vec![Mat::zero(); 4 * geo.v];
    for u in &mut liens {
        for i in 0..D {
            for j in 0..D {
                u.re[i][j] = l.reel()?;
                u.im[i][j] = l.reel()?;
            }
        }
    }
    let r = Reseau {
        geo,
        liens,
        beta: spec.beta,
        eps,
        graine: spec.graine,
        pas_faits: pas,
        suivre_defaut: false,
        defaut_avant_reproj: 0.,
    };
    Ok((r, Serie { spec, eps, valeurs }, accepte_bloc))
}

pub fn executer_chaine<const D: usize>(
    noyau: &dyn Noyau<D>,
    spec: &SpecChaine,
    dossier: &Path,
    pas_reprise: u64,
) -> Resultat<Serie> {
    verifier_l(spec.l)?;
    if spec.id.is_empty()
        || !spec
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_- .".contains(c))
        || spec.id.contains(' ')
        || spec.id == "."
        || spec.id == ".."
    {
        return Err(Erreur::Config("Identifiant de chaîne invalide".into()));
    }
    if !spec.beta.is_finite()
        || spec
            .therm
            .checked_add(spec.mesures)
            .is_none_or(|n| n > u32::MAX as u64 + 1)
    {
        return Err(Erreur::Config("β ou nombre de pas invalide".into()));
    }
    fs::create_dir_all(dossier).map_err(|e| Erreur::io(dossier.display().to_string(), e))?;
    let chemin = chemin_serie(dossier, &spec.id);
    let reprise = dossier.join(format!("{}.reprise", spec.id));
    let ancienne = if chemin
        .try_exists()
        .map_err(|e| Erreur::io(chemin.display().to_string(), e))?
    {
        let (serie, groupe) = lire_serie_complete(&chemin)?;
        if !compatibles(&serie.spec, spec) || groupe != noyau.nom() {
            return Err(Erreur::Config("Paramètres de série incompatibles".into()));
        }
        Some(serie)
    } else {
        None
    };
    let reprise_existe = reprise
        .try_exists()
        .map_err(|e| Erreur::io(reprise.display().to_string(), e))?;
    let sauvegarde = if reprise_existe {
        Some(lire_reprise(&reprise, noyau, spec)?)
    } else {
        None
    };
    if let Some(ref serie) = ancienne {
        if let Some((_, ref partielle, _)) = sauvegarde {
            let n = serie.valeurs.len().min(partielle.valeurs.len());
            if serie.eps.to_bits() != partielle.eps.to_bits()
                || serie.valeurs[..n] != partielle.valeurs[..n]
            {
                return Err(format_invalide("Série et reprise contradictoires"));
            }
        }
        if serie.valeurs.len() as u64 >= spec.mesures {
            return Ok(serie.clone());
        }
        if sauvegarde
            .as_ref()
            .is_none_or(|(_, s, _)| s.valeurs.len() < serie.valeurs.len())
        {
            return Err(format_invalide(
                "Point de reprise absent ou antérieur à la série à prolonger",
            ));
        }
    }
    let (mut r, mut serie, accepte_bloc) = if let Some(etat) = sauvegarde {
        etat
    } else {
        let r = Reseau::essayer_nouveau(noyau, spec.l, spec.beta, spec.graine, spec.depart)?;
        let serie = Serie {
            spec: spec.clone(),
            eps: r.eps,
            valeurs: Vec::new(),
        };
        (r, serie, 0)
    };
    if r.pas_faits < spec.therm {
        let deja = r.pas_faits;
        r.thermaliser_avec(
            noyau,
            spec.therm - deja,
            deja,
            accepte_bloc,
            |etat, accepte| {
                if pas_reprise != 0 && etat.pas_faits.is_multiple_of(pas_reprise) {
                    let mut partielle = serie.clone();
                    partielle.eps = etat.eps;
                    // Le dernier bloc, même incomplet, ne sert plus après thermalisation.
                    let accepte = if etat.pas_faits == spec.therm {
                        0
                    } else {
                        accepte
                    };
                    ecrire_reprise(&reprise, noyau, etat, &partielle, accepte)?;
                }
                Ok(())
            },
        )?;
        serie.eps = r.eps;
    }
    if serie.valeurs.is_empty() {
        let _mesure_zero = r.plaquette();
    }

    // Une reprise plus longue répond également aux demandes plus courtes sans tronquer le disque.
    serie.spec.mesures = spec.mesures.max(serie.valeurs.len() as u64);
    let debut = Instant::now();
    let depart_pas = r.pas_faits;
    while (serie.valeurs.len() as u64) < spec.mesures {
        r.pas(noyau);
        serie.valeurs.push(r.plaquette());
        if pas_reprise != 0 && r.pas_faits.is_multiple_of(pas_reprise) {
            ecrire_reprise(&reprise, noyau, &r, &serie, 0)?;
        }
        if r.pas_faits.is_multiple_of(1000) {
            progression(&spec.id, r.pas_faits, r.pas_faits - depart_pas, debut);
        }
    }
    ecrire_reprise(&reprise, noyau, &r, &serie, 0)?;
    let mut texte = entete(&serie.spec, noyau.nom(), serie.eps);
    for x in &serie.valeurs {
        texte.push_str(&format!("{x:.17e}\n"));
    }
    texte.push_str(&format!("# fin n = {}\n", serie.valeurs.len()));
    atomique(&chemin, texte.as_bytes())?;
    Ok(serie)
}

#[allow(clippy::too_many_arguments)]
pub fn simuler_reseau<const D: usize>(
    noyau: &dyn Noyau<D>,
    l: usize,
    beta: f64,
    graine: u64,
    depart: Depart,
    therm: u64,
    pas_apres: u64,
    suivre_defaut_dernier: bool,
) -> Reseau<D> {
    let mut r = Reseau::nouveau(noyau, l, beta, graine, depart);
    r.suivre_defaut = suivre_defaut_dernier && pas_apres == 0;
    r.thermaliser(noyau, therm);
    for k in 0..pas_apres {
        r.suivre_defaut = suivre_defaut_dernier && k + 1 == pas_apres;
        r.pas(noyau);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groupes::sp4::Sp4;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn dossier() -> PathBuf {
        static COMPTE: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "kvu-wp16-{}-{}",
            std::process::id(),
            COMPTE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
    fn spec() -> SpecChaine {
        SpecChaine {
            id: "essai".into(),
            l: 2,
            beta: 8.,
            graine: 26102616004,
            depart: Depart::Chaud,
            therm: 51,
            mesures: 5,
        }
    }

    #[test]
    fn matrices_fixes() {
        let a = Cmat::depuis_entrelace(2, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = a.dag();
        let m = Mat::<2>::depuis_cmat(&a);
        let n = Mat::<2>::depuis_cmat(&b);
        assert_eq!(m.vers_cmat(), a);
        assert_eq!(m.mul(&n).vers_cmat(), a.mul(&b));
        assert_eq!(m.dag().vers_cmat(), b);
        assert_eq!(m.add(&n).vers_cmat(), a.add(&b));
        assert_eq!(m.sub(&n).vers_cmat(), a.sub(&b));
        assert_eq!(m.echelle(2.).vers_cmat(), a.echelle(2.));
        assert_eq!(m.re_tr(), a.trace().0);
        assert_eq!(m.re_tr_prod(&n), a.mul(&b).trace().0);
    }

    #[test]
    fn reseau_froid_chaud_et_pas() {
        let g = Sp4::nouveau();
        assert!(Reseau::essayer_nouveau(&g, 1, 8., 1, Depart::Froid).is_err());
        assert!(Reseau::essayer_nouveau(&g, 3, 8., 1, Depart::Froid).is_err());
        assert_eq!(
            Reseau::essayer_nouveau(&g, 4, 8., 1, Depart::Froid)
                .unwrap()
                .plaquette(),
            1.
        );
        let r = Reseau::essayer_nouveau(&g, 4, 8., 1, Depart::Chaud).unwrap();
        assert!(r.defaut_max(&g) < 1e-14);
        let u = g.haar_philox(17, u32::MAX, 1, 0, TAG_INIT_GEN);
        assert_eq!(r.liens[17], u);
        let mut r = Reseau::essayer_nouveau(&g, 2, 8., 1, Depart::Chaud).unwrap();
        r.suivre_defaut = true;
        for k in 1..=50 {
            let stats = r.pas(&g);
            assert_eq!(stats.propositions, 640);
            assert!(stats.acceptees > 0 && stats.acceptees < 640);
            assert!(r.plaquette().is_finite());
            assert!(r.defaut_max(&g) < 1e-14);
            assert_eq!(r.pas_faits, k);
        }
        assert!(r.defaut_avant_reproj > 0. && r.defaut_avant_reproj < 1e-13);
    }

    #[test]
    fn agrafe_parcours_independant() {
        let g = Sp4::nouveau();
        let r = Reseau::essayer_nouveau(&g, 2, 8., 123, Depart::Chaud).unwrap();
        // Énumère toutes les plaquettes orientées positivement et distribue leur trace
        // aux quatre liens du contour, indépendamment de la formule d'agrafe.
        let mut sommes = vec![0.; r.liens.len()];
        for s in 0..r.geo.v {
            let x = r.geo.coords(s);
            for mu in 0..4 {
                for nu in mu + 1..4 {
                    let mut y = x;
                    y[mu] = (y[mu] + 1) % r.geo.l;
                    let smu = ((y[0] * r.geo.l + y[1]) * r.geo.l + y[2]) * r.geo.l + y[3];
                    let mut z = x;
                    z[nu] = (z[nu] + 1) % r.geo.l;
                    let snu = ((z[0] * r.geo.l + z[1]) * r.geo.l + z[2]) * r.geo.l + z[3];
                    let indices = [
                        mu * r.geo.v + s,
                        nu * r.geo.v + smu,
                        mu * r.geo.v + snu,
                        nu * r.geo.v + s,
                    ];
                    let p = r.liens[indices[0]]
                        .mul(&r.liens[indices[1]])
                        .mul(&r.liens[indices[2]].dag())
                        .mul(&r.liens[indices[3]].dag());
                    for li in indices {
                        sommes[li] += p.re_tr();
                    }
                }
            }
        }
        for (li, &somme) in sommes.iter().enumerate() {
            let s = li % r.geo.v;
            let mu = li / r.geo.v;
            assert!(
                (r.liens[li].re_tr_prod(&r.agrafe(mu, s, &r.geo.coords(s))) - somme).abs() < 1e-12
            );
        }
    }

    #[test]
    fn adaptation_blocs_complets() {
        let g = Sp4::nouveau();
        let mut r = Reseau::essayer_nouveau(&g, 2, 0., 1, Depart::Chaud).unwrap();
        r.thermaliser(&g, 49);
        assert_eq!(r.eps, 0.3);
        let mut r = Reseau::essayer_nouveau(&g, 2, 0., 1, Depart::Chaud).unwrap();
        r.thermaliser(&g, 101);
        assert_eq!(r.eps, 0.3 * 1.1 * 1.1);
    }

    #[test]
    fn series_determinisme_reprise_et_extension() {
        let g = Sp4::nouveau();
        let a = dossier();
        let b = dossier();
        let c = dossier();
        let mut spec = spec();
        let direct = executer_chaine(&g, &spec, &a, 0).unwrap();
        assert_eq!(
            direct.valeurs,
            executer_chaine(&g, &spec, &b, 2).unwrap().valeurs
        );
        assert_eq!(
            fs::read(chemin_serie(&a, &spec.id)).unwrap(),
            fs::read(chemin_serie(&b, &spec.id)).unwrap()
        );
        spec.mesures = 2;
        executer_chaine(&g, &spec, &c, 1).unwrap();
        spec.mesures = 5;
        let prolongee = executer_chaine(&g, &spec, &c, 1).unwrap();
        assert_eq!(direct.valeurs, prolongee.valeurs);
        assert_eq!(direct.eps.to_bits(), prolongee.eps.to_bits());
        assert_eq!(
            fs::read(c.join("essai.reprise")).unwrap(),
            fs::read(a.join("essai.reprise")).unwrap()
        );
        // Un arrêt simulé après la reprise finale, avant l'écriture de la série.
        fs::remove_file(chemin_serie(&c, &spec.id)).unwrap();
        assert_eq!(
            direct.valeurs,
            executer_chaine(&g, &spec, &c, 1).unwrap().valeurs
        );
        assert_eq!(
            lire_serie(&chemin_serie(&c, &spec.id)).unwrap().valeurs,
            direct.valeurs
        );
        spec.mesures = 3;
        assert_eq!(
            executer_chaine(&g, &spec, &c, 0).unwrap().valeurs,
            direct.valeurs
        );
        spec.beta = 9.;
        assert!(executer_chaine(&g, &spec, &c, 0).is_err());
        for p in [a, b, c] {
            fs::remove_dir_all(p).unwrap();
        }
    }

    #[test]
    fn reprise_thermalisation_bloc_incomplet() {
        let g = Sp4::nouveau();
        let a = dossier();
        let mut spec = spec();
        spec.therm = 101;
        let direct = executer_chaine(&g, &spec, &a, 0).unwrap();
        for coupure in [27, 50, 78, 101, 103] {
            let b = dossier();
            let mut r =
                Reseau::essayer_nouveau(&g, spec.l, spec.beta, spec.graine, spec.depart).unwrap();
            let mut serie = Serie {
                spec: spec.clone(),
                eps: r.eps,
                valeurs: Vec::new(),
            };
            let therm = spec.therm.min(coupure);
            let mut accepte = 0;
            r.thermaliser_avec(&g, therm, 0, 0, |_, acc| {
                accepte = acc;
                Ok(())
            })
            .unwrap();
            serie.eps = r.eps;
            if coupure >= spec.therm {
                accepte = 0;
            }
            for _ in spec.therm..coupure {
                r.pas(&g);
                serie.valeurs.push(r.plaquette());
            }
            ecrire_reprise(&b.join("essai.reprise"), &g, &r, &serie, accepte).unwrap();
            let reprise = executer_chaine(&g, &spec, &b, 7).unwrap();
            assert_eq!(direct.valeurs, reprise.valeurs);
            assert_eq!(direct.eps.to_bits(), reprise.eps.to_bits());
            assert_eq!(
                fs::read(a.join("essai.reprise")).unwrap(),
                fs::read(b.join("essai.reprise")).unwrap()
            );
            fs::remove_dir_all(b).unwrap();
        }
        fs::remove_dir_all(a).unwrap();
    }

    #[test]
    fn refus_corruption_et_erreurs_utilisateur() {
        let g = Sp4::nouveau();
        let p = dossier();
        let spec = spec();
        executer_chaine(&g, &spec, &p, 2).unwrap();
        let chemin = p.join("essai.reprise");
        let mut octets = fs::read(&chemin).unwrap();
        octets[32] ^= 1;
        fs::write(&chemin, octets).unwrap();
        assert!(executer_chaine(&g, &spec, &p, 0).is_err());
        let mut s = spec.clone();
        s.id = "../interdit".into();
        assert!(executer_chaine(&g, &s, &p, 0).is_err());
        s = spec.clone();
        s.beta = f64::NAN;
        assert!(executer_chaine(&g, &s, &p, 0).is_err());
        fs::remove_dir_all(p).unwrap();
    }
}
