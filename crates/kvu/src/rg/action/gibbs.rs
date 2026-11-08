//! Échantillonneur de Gibbs de l'action tronquée S_β (instrument de contrôle, pas un
//! générateur de production). Bain thermique séquentiel exact (von Mises–Fisher) et
//! sur-relaxation.
//!
//! Les sorties sont déterministes (aucun parallélisme). L'étiquette Philox est
//! `ETIQUETTE_GIBBS` ("GIBS").

use super::boucles::{Famille, TableFormes};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, qnorm, qscale, sample_x0};
use crate::rg::lissage::Voisinage;
use crate::su2::cpu::haar_q;

/// Étiquette Philox du Gibbs.
pub const ETIQUETTE_GIBBS: u32 = 0x4749_4253; // "GIBS"

/// Paramètres du Gibbs.
#[derive(Clone, Debug)]
pub struct ParamGibbs {
    pub l: usize,
    pub famille: Famille,
    pub beta: Vec<f64>,
    pub graine: u64,
    pub depart_chaud: bool,
    pub sur_relaxations: usize,
}

/// Échantillonneur de Gibbs séquentiel.
pub struct Gibbs {
    champ: ChampHote,
    table: TableFormes,
    vois: Voisinage,
    famille: Famille,
    beta: Vec<f64>,
    k0: u32,
    k1: u32,
    balayage: u32,
    sur_relaxations: usize,
    n_plafonds: u64,
}

impl Gibbs {
    pub fn nouveau(p: ParamGibbs) -> Resultat<Gibbs> {
        if p.l < 4 || !p.l.is_multiple_of(2) {
            return Err(Erreur::Config(format!("L = {} : L pair ≥ 4 requis", p.l)));
        }
        if p.beta.len() != p.famille.n() || p.beta.iter().any(|b| !b.is_finite()) {
            return Err(Erreur::Config(
                "β de Gibbs invalides (longueur ou non-finie)".into(),
            ));
        }
        let table = TableFormes::construire()?;
        let vois = Voisinage::nouveau(p.l);
        let mut champ = ChampHote::froid(p.l)?;
        let (k0, k1) = ((p.graine & 0xFFFF_FFFF) as u32, (p.graine >> 32) as u32);
        if p.depart_chaud {
            for li in 0..4 * champ.v() {
                let q = haar_q(li as u32, 0xFFFF_FFFF, ETIQUETTE_GIBBS, k0, k1);
                let o = 4 * li;
                champ.q[o..o + 4].copy_from_slice(&q);
            }
        }
        Ok(Gibbs {
            champ,
            table,
            vois,
            famille: p.famille,
            beta: p.beta,
            k0,
            k1,
            balayage: 0,
            sur_relaxations: p.sur_relaxations,
            n_plafonds: 0,
        })
    }

    /// Calcule l'agrafe totale A = Σ_γ β_γ A_ℓ^γ pour le lien (mu, s) dans `a`.
    /// Utilise les formes directement sans allocation intermédiaire.
    fn calculer_a(&self, mu: usize, s: usize, a: &mut [f64; 4]) {
        a.fill(0.0);
        for (idx_ty, &ty) in self.famille.types().iter().enumerate() {
            let b = self.beta[idx_ty];
            let formes = self.table.formes(mu, ty);
            let x_plus = self.vois.haut(mu, s);
            let mut somme_agrafe = [0.0; 4];
            for forme in formes {
                let prod =
                    super::boucles::produit_chemin(&self.champ, &self.vois, x_plus, &forme[1..]);
                somme_agrafe = crate::heritage::su2_hb::qadd(somme_agrafe, prod);
            }
            for i in 0..4 {
                a[i] += b * somme_agrafe[i];
            }
        }
    }

    /// Un balayage complet : bain thermique séquentiel puis `sur_relaxations` sur-relaxations.
    pub fn balayage(&mut self) {
        let v = self.champ.v();
        let (k0, k1) = (self.k0, self.k1);
        let balayage_courant = self.balayage;

        // Bain thermique séquentiel (ordre μ puis s)
        for mu in 0..4 {
            for s in 0..v {
                let li = mu * v + s;
                let mut a = [0.0; 4];
                self.calculer_a(mu, s, &mut a);
                let k = qnorm(a);
                let mut tirage = 0u32;
                if k < 1e-3 {
                    // Rejet exact depuis Haar : exp(Re(UA) − |A|) ≤ 1.
                    // Pas de division par k ni de soustraction exp(−2k) ≈ 1.
                    let mut accepte = false;
                    for essai in 0..1_000_000u32 {
                        let r1 = crate::heritage::su2_hb::philox4x32_10(
                            [li as u32, balayage_courant, 3 * essai, ETIQUETTE_GIBBS],
                            k0,
                            k1,
                        );
                        let r2 = crate::heritage::su2_hb::philox4x32_10(
                            [li as u32, balayage_courant, 3 * essai + 1, ETIQUETTE_GIBBS],
                            k0,
                            k1,
                        );
                        let r3 = crate::heritage::su2_hb::philox4x32_10(
                            [li as u32, balayage_courant, 3 * essai + 2, ETIQUETTE_GIBBS],
                            k0,
                            k1,
                        );
                        let u01 = crate::heritage::su2_hb::u01;
                        let ra = (-2.0 * u01(r1[0], r1[1]).ln()).sqrt();
                        let rb = (-2.0 * u01(r2[0], r2[1]).ln()).sqrt();
                        let pa = std::f64::consts::TAU * u01(r1[2], r1[3]);
                        let pb = std::f64::consts::TAU * u01(r2[2], r2[3]);
                        let brut = [ra * pa.cos(), ra * pa.sin(), rb * pb.cos(), rb * pb.sin()];
                        let q = qscale(brut, 1.0 / qnorm(brut));
                        if u01(r3[0], r3[1]).ln() <= qmul(q, a)[0] - k {
                            let o = 4 * li;
                            self.champ.q[o..o + 4].copy_from_slice(&q);
                            accepte = true;
                            break;
                        }
                    }
                    if !accepte {
                        self.n_plafonds += 1;
                    }
                } else {
                    let (x0, _, ok) =
                        sample_x0(k, li as u32, balayage_courant, &mut tirage, k0, k1);
                    if !ok {
                        self.n_plafonds += 1;
                        continue; // lien inchangé
                    }
                    // tirage directionnel, exactement comme CpuSu2::hb_link
                    let r = crate::heritage::su2_hb::philox4x32_10(
                        [
                            li as u32,
                            balayage_courant,
                            tirage,
                            crate::commun::ETIQUETTE_HB,
                        ],
                        k0,
                        k1,
                    );
                    let ct = 2.0 * crate::heritage::su2_hb::u01(r[0], r[1]) - 1.0;
                    let phi = 2.0 * std::f64::consts::PI * crate::heritage::su2_hb::u01(r[2], r[3]);
                    let st = (1.0 - ct * ct).max(0.0).sqrt();
                    let rr = (1.0 - x0 * x0).max(0.0).sqrt();
                    let w = [x0, rr * st * phi.cos(), rr * st * phi.sin(), rr * ct];
                    let vv = qscale(a, 1.0 / k);
                    let nouveau = qmul(w, qdag(vv));
                    let norme = qnorm(nouveau);
                    let o = 4 * li;
                    for (qi, ni) in self.champ.q[o..o + 4].iter_mut().zip(nouveau) {
                        *qi = ni / norme;
                    }
                }
            }
        }
        self.balayage = self.balayage.wrapping_add(1);

        // Sur-relaxations séquentielles
        for _ in 0..self.sur_relaxations {
            for mu in 0..4 {
                for s in 0..v {
                    let li = mu * v + s;
                    let mut a = [0.0; 4];
                    self.calculer_a(mu, s, &mut a);
                    let k = qnorm(a);
                    if k <= 1e-300 {
                        continue;
                    }
                    let vd = qdag(qscale(a, 1.0 / k));
                    let u = self.champ.lien(li);
                    let nouveau = qmul(qmul(vd, qdag(u)), vd);
                    let norme = qnorm(nouveau);
                    let o = 4 * li;
                    for (qi, ni) in self.champ.q[o..o + 4].iter_mut().zip(nouveau) {
                        *qi = ni / norme;
                    }
                }
            }
        }
    }

    pub fn champ(&self) -> ChampHote {
        self.champ.clone()
    }

    pub fn plafonds(&self) -> u64 {
        self.n_plafonds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rg::swendsen::plaquette_moyenne;

    #[test]
    fn petit_beta_positif_decorrele_un_depart_froid() {
        let mut gibbs = Gibbs::nouveau(ParamGibbs {
            l: 4,
            famille: Famille::analyser("P").unwrap(),
            beta: vec![1e-20],
            graine: 2611020117,
            depart_chaud: false,
            sur_relaxations: 0,
        })
        .unwrap();
        gibbs.balayage();
        let champ = gibbs.champ();
        let n = 4 * champ.v();
        let moyenne = (0..n).map(|li| champ.lien(li)[0]).sum::<f64>() / n as f64;
        assert!(moyenne.abs() < 4.0 * 0.5 / (n as f64).sqrt());
        assert!((0..n).filter(|&li| champ.lien(li)[0] < 0.9).count() > n / 2);
        assert_eq!(gibbs.plafonds(), 0);
    }

    #[test]
    fn beta_nul_donne_haar_et_plafonds_zero_a_beta_2() {
        // β = 0 : chaque balayage doit produire des liens de Haar.
        let famille = Famille::analyser("P").unwrap();
        let params = ParamGibbs {
            l: 4,
            famille,
            beta: vec![0.0],
            graine: 26110201,
            depart_chaud: true,
            sur_relaxations: 0,
        };
        let mut gibbs = Gibbs::nouveau(params).unwrap();
        let mut somme = 0.0;
        let mut n_mesures = 0usize;
        for _ in 0..50 {
            gibbs.balayage();
            somme += plaquette_moyenne(&gibbs.champ());
            n_mesures += 1;
        }
        let moyenne = somme / n_mesures as f64;
        // Sous Haar Var(½Tr U) = 1/4 : σ = 1/(2 sqrt(6 L⁴ × 50)) ≈ 0,001804.
        let sigma = 0.5 / (6.0 * 4.0_f64.powi(4) * n_mesures as f64).sqrt();
        assert!(
            moyenne.abs() < 4.0 * sigma,
            "moyenne de plaquette {moyenne:.5} incompatible avec 0 (|moyenne| < 4σ = {:.5})",
            4.0 * sigma
        );

        // β = 2.0, famille {P} : vérifier qu'aucun plafond n'est atteint.
        let famille_p = Famille::analyser("P").unwrap();
        let params2 = ParamGibbs {
            l: 4,
            famille: famille_p,
            beta: vec![2.0],
            graine: 987654,
            depart_chaud: true,
            sur_relaxations: 1,
        };
        let mut gibbs2 = Gibbs::nouveau(params2).unwrap();
        for _ in 0..10 {
            gibbs2.balayage();
        }
        assert_eq!(gibbs2.plafonds(), 0, "plafonds non nuls à β=2");
    }
}
