//! [EXPÉRIMENTAL] Blocage de Teper (superliens de longueur 2^b) : VERSION PROCESSEUR, référence F64 des noyaux de carte
//! `k_bloc`, `k_ape_l`, `k_boucles_l`, `k_lignes_l` (`noyaux/su2_spectro.cu`). Branche `spectro/blocage-1120`.
//!
//! Définitions (Lucini–Teper–Wenger 2004, hep-lat/0404008, éq. « eqn_oldblock », reprise ici pour SU(2), axes spatiaux seulement) :
//! - champ de base B⁰ = APE^{n₀}(U) (`ape_bloc` itérations du lissage APE ordinaire, `super::ape_iteration`) ;
//! - itération de blocage, de la longueur ℓ = 2^b à 2ℓ (superlien de x à x + 2ℓ î, pour chaque site x et chaque axe spatial i) :
//!   B^{b+1}_i(x) = Proj[ B^b_i(x) B^b_i(x+ℓî) + p Σ_{j≠i} ( B^b_j(x) B^b_i(x+ℓĵ) B^b_i(x+ℓĵ+ℓî) B^b_j(x+2ℓî)†
//!                                                        + B^b_j(x−ℓĵ)† B^b_i(x−ℓĵ) B^b_i(x−ℓĵ+ℓî) B^b_j(x−ℓĵ+2ℓî) ) ],
//!   Proj(V) = V/|V| (SU(2) : une somme de matrices SU(2) est un multiple réel d'une matrice SU(2)) ; liens de temps intacts ;
//! - lissage des superliens (optionnel, `lissage_bloc` itérations après chaque blocage, stratégie « blocage puis lissage » de LTW) :
//!   APE dont les agrafes sont faites de superliens de même longueur ℓ : V = (1−α) B_i(x) + (α/4) Σ_{j≠i} [B_j(x) B_i(x+ℓĵ) B_j(x+ℓî)†
//!   + B_j(x−ℓĵ)† B_i(x−ℓĵ) B_j(x−ℓĵ+ℓî)], puis V/|V| ; le superlissé sert d'entrée au blocage suivant ;
//! - opérateur d'un niveau de blocage b : les MÊMES formes et images que `neutre`, chaque pas valant un superlien (ℓ = 2^b sites) ;
//!   S_f(t) = (1/(N_im V₃)) Σ_x Σ_im ½Tr W_im^{(ℓ)}(x,t) ; impulsion : phase e^{2πik(x_d + ℓ c_d)/ℓ_d} au centre (agrandi ℓ fois) ;
//! - torelon d'un niveau b (ℓ divise la longueur d'enroulement L_w) : P^{(ℓ)}(x⊥,t) = (1/ℓ) Σ_{x₀=0}^{ℓ−1} ½Tr ∏_{k<L_w/ℓ} B_w(x₀ŵ + kℓŵ + x⊥),
//!   moyenne sur les ℓ points de départ (invariance par translation le long de ŵ).
//! Tout est en F64, sans parallélisme flottant (sommes séquentielles) : cette version sert de référence aux tests de carte.
#![allow(clippy::doc_lazy_continuation)] // style seulement : mise en forme des listes de la documentation
#![allow(clippy::doc_overindented_list_items)] // style seulement : mise en forme des listes de la documentation

use super::neutre::{centre_chemin, correlateurs, correlateurs_orbite, correlateurs_ps, images, tranches_ps, tranches_vec, MesureNeutre, Orbite, ParamNeutre, FORMES};
use super::{ape_iteration, correlateur, phi_moyen, Axes, Mutant};
use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{qadd, qdag, qmul, qnorm, qscale, Q};
use crate::su2::tube::reseau::GeoA;

/// Paramètres du blocage.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamBloc {
    /// niveaux de blocage MESURÉS (b ≥ 1, croissants) ; le superlien du niveau b a la longueur 2^b
    pub niveaux: Vec<usize>,
    /// itérations APE ordinaires du champ avant le premier blocage
    pub ape_bloc: usize,
    /// itérations du lissage des superliens après chaque blocage
    pub lissage_bloc: usize,
    /// poids des agrafes allongées du blocage (p_a de LTW04)
    pub p_bloc: f64,
}

impl ParamBloc {
    pub fn vide() -> ParamBloc {
        ParamBloc { niveaux: Vec::new(), ape_bloc: 0, lissage_bloc: 0, p_bloc: 1.0 }
    }
    pub fn actif(&self) -> bool {
        !self.niveaux.is_empty()
    }
    /// Étiquette d'un niveau dans le manifeste (`niveaux_ape`) : « B<b> » ou « B<b>s<lissage> ».
    pub fn etiquette(&self, b: usize) -> String {
        if self.lissage_bloc == 0 {
            format!("B{b}")
        } else {
            format!("B{b}s{}", self.lissage_bloc)
        }
    }
    /// Contrôles : niveaux croissants ≥ 1, superlien plus court que le plus petit côté spatial (pas d'opérateur qui s'enroule),
    /// `p_bloc` fini ≥ 0.
    pub fn controler(&self, cotes: &[usize; 4], temps: usize) -> Resultat<()> {
        if !self.actif() {
            return Ok(());
        }
        if self.niveaux[0] < 1 || self.niveaux.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Erreur::Config("--blocage : niveaux croissants ≥ 1 attendus".into()));
        }
        if !(self.p_bloc.is_finite() && self.p_bloc >= 0.0) {
            return Err(Erreur::Config("--p-bloc : réel fini ≥ 0 attendu".into()));
        }
        let bmax = *self.niveaux.last().unwrap();
        if bmax > 12 {
            return Err(Erreur::Config("--blocage : niveau ≤ 12".into()));
        }
        let lmin = (0..4).filter(|&d| d != temps).map(|d| cotes[d]).min().unwrap();
        if (1usize << bmax) >= lmin {
            return Err(Erreur::Config(format!(
                "--blocage : superlien de longueur {} ≥ plus petit côté spatial {} (opérateurs enroulés interdits)",
                1usize << bmax,
                lmin
            )));
        }
        Ok(())
    }
}

/// Site s décalé de l (signé) le long de l'axe d (périodique).
#[inline]
pub fn decaler(g: &GeoA, s: usize, d: usize, l: i64) -> usize {
    let x = (s / g.st[d]) % g.d[d];
    let n = (x as i64 + l).rem_euclid(g.d[d] as i64) as usize;
    s - x * g.st[d] + n * g.st[d]
}

fn spatiaux(temps: usize) -> Vec<usize> {
    (0..4).filter(|&d| d != temps).collect()
}

/// Une itération de blocage ℓ → 2ℓ (voir l'en-tête). `faux` = mutant (agrafe basse sans le dagger du premier superlien).
pub fn bloc_iteration(g: &GeoA, u: &[Q], temps: usize, l: usize, p: f64, faux: bool) -> Vec<Q> {
    let mut out = u.to_vec();
    let sp = spatiaux(temps);
    let li = l as i64;
    let ld = |d: usize, s: usize| u[d * g.v + s];
    for &mu in &sp {
        for s in 0..g.v {
            let s_mu = decaler(g, s, mu, li);
            let s_2mu = decaler(g, s, mu, 2 * li);
            let direct = qmul(ld(mu, s), ld(mu, s_mu));
            let mut a: Q = [0.0; 4];
            for &nu in &sp {
                if nu == mu {
                    continue;
                }
                let s_nu = decaler(g, s, nu, li);
                let haut = qmul(qmul(qmul(ld(nu, s), ld(mu, s_nu)), ld(mu, decaler(g, s_nu, mu, li))), qdag(ld(nu, s_2mu)));
                let s_dn = decaler(g, s, nu, -li);
                let prem = if faux { ld(nu, s_dn) } else { qdag(ld(nu, s_dn)) };
                let bas = qmul(qmul(qmul(prem, ld(mu, s_dn)), ld(mu, decaler(g, s_dn, mu, li))), ld(nu, decaler(g, s_dn, mu, 2 * li)));
                a = qadd(a, qadd(haut, bas));
            }
            let v = qadd(direct, qscale(a, p));
            let n = qnorm(v);
            out[mu * g.v + s] = if n > 1e-30 { qscale(v, 1.0 / n) } else { direct }; // même seuil que k_bloc
        }
    }
    out
}

/// Une itération de lissage APE des superliens de longueur ℓ (agrafes de superliens).
pub fn ape_iteration_l(g: &GeoA, u: &[Q], temps: usize, alpha: f64, l: usize) -> Vec<Q> {
    let mut out = u.to_vec();
    let sp = spatiaux(temps);
    let li = l as i64;
    let ld = |d: usize, s: usize| u[d * g.v + s];
    for &mu in &sp {
        for s in 0..g.v {
            let s_mu = decaler(g, s, mu, li);
            let mut a: Q = [0.0; 4];
            for &nu in &sp {
                if nu == mu {
                    continue;
                }
                let s_nu = decaler(g, s, nu, li);
                let s_dn = decaler(g, s, nu, -li);
                let haut = qmul(qmul(ld(nu, s), ld(mu, s_nu)), qdag(ld(nu, s_mu)));
                let bas = qmul(qmul(qdag(ld(nu, s_dn)), ld(mu, s_dn)), ld(nu, decaler(g, s_dn, mu, li)));
                a = qadd(a, qadd(haut, bas));
            }
            let v = qadd(qscale(ld(mu, s), 1.0 - alpha), qscale(a, alpha / 4.0));
            let n = qnorm(v);
            out[mu * g.v + s] = if n > 1e-30 { qscale(v, 1.0 / n) } else { ld(mu, s) }; // même seuil que k_ape_l
        }
    }
    out
}

/// Champs blocés des niveaux demandés : [(b, champ du niveau b)], dans l'ordre de `pb.niveaux`.
pub fn champs_bloques(g: &GeoA, u: &[Q], temps: usize, alpha: f64, pb: &ParamBloc) -> Vec<(usize, Vec<Q>)> {
    let mut out = Vec::new();
    if !pb.actif() {
        return out;
    }
    let mut cur = u.to_vec();
    for _ in 0..pb.ape_bloc {
        cur = ape_iteration(g, &cur, temps, alpha);
    }
    let bmax = *pb.niveaux.last().unwrap();
    for b in 1..=bmax {
        cur = bloc_iteration(g, &cur, temps, 1usize << (b - 1), pb.p_bloc, false);
        for _ in 0..pb.lissage_bloc {
            cur = ape_iteration_l(g, &cur, temps, alpha, 1usize << b);
        }
        if pb.niveaux.contains(&b) {
            out.push((b, cur.clone()));
        }
    }
    out
}

/// ½Tr du chemin `pas` (pas de ±1..3 sur les axes spatiaux, chaque pas = un superlien de longueur ℓ) depuis le site s.
pub fn trace_chemin_l(g: &GeoA, u: &[Q], ax: &[usize; 3], pas: &[i8], s0: usize, l: usize) -> f64 {
    let mut p: Q = [1.0, 0.0, 0.0, 0.0];
    let mut s = s0;
    let li = l as i64;
    for &st in pas {
        let a = ax[st.unsigned_abs() as usize - 1];
        if st > 0 {
            p = qmul(p, u[a * g.v + s]);
            s = decaler(g, s, a, li);
        } else {
            s = decaler(g, s, a, -li);
            p = qmul(p, qdag(u[a * g.v + s]));
        }
    }
    debug_assert_eq!(s, s0, "chemin non fermé");
    p[0]
}

/// Comme `neutre::tranches_ps`, pour des superliens de longueur ℓ ; la phase d'une image est calculée directement
/// (θ = 2πk(x_d + ℓ c_d)/ℓ_d), pas par la factorisation de `tranches_ps` : implémentation indépendante. Normalisation 1/(N_im V₃).
#[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn tranches_ps_l(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>], ks: &[i64], l: usize) -> (Vec<f64>, Vec<Vec<[[f64; 2]; 3]>>) {
    let sp = spatiaux(temps);
    let ax = [sp[0], sp[1], sp[2]];
    let tt = g.d[temps];
    let v3 = g.v / tt;
    let n = (imgs.len() * v3) as f64;
    let centres: Vec<[f64; 3]> = imgs.iter().map(|im| centre_chemin(im)).collect();
    let pi2 = 2.0 * std::f64::consts::PI;
    let mut s_t = vec![0.0f64; tt];
    let mut p_t = vec![vec![[[0.0f64; 2]; 3]; tt]; ks.len()];
    for s in 0..g.v {
        let t = (s / g.st[temps]) % tt;
        for (kk, im) in imgs.iter().enumerate() {
            let tr = trace_chemin_l(g, u, &ax, im, s, l);
            s_t[t] += tr;
            for (ik, &k) in ks.iter().enumerate() {
                for d in 0..3 {
                    let xd = ((s / g.st[ax[d]]) % g.d[ax[d]]) as f64;
                    let th = pi2 * (k as f64) * (xd + l as f64 * centres[kk][d]) / g.d[ax[d]] as f64;
                    p_t[ik][t][d][0] += tr * th.cos();
                    p_t[ik][t][d][1] += tr * th.sin();
                }
            }
        }
    }
    for x in s_t.iter_mut() {
        *x /= n;
    }
    for a in p_t.iter_mut() {
        for b in a.iter_mut() {
            for c in b.iter_mut() {
                c[0] /= n;
                c[1] /= n;
            }
        }
    }
    (s_t, p_t)
}

/// Comme `neutre::tranches_vec`, pour des superliens de longueur ℓ : S(t; k) = Σ_x tr e^{i k·(x + ℓ c)} / (N_im V₃) pour des vecteurs
/// k quelconques ; phase calculée directement (angle total), comme `tranches_ps_l` (implémentation indépendante de `tranches_vec`).
pub fn tranches_vec_l(g: &GeoA, u: &[Q], temps: usize, imgs: &[Vec<i8>], vs: &[[i64; 3]], l: usize) -> (Vec<f64>, Vec<Vec<[f64; 2]>>) {
    let sp = spatiaux(temps);
    let ax = [sp[0], sp[1], sp[2]];
    let tt = g.d[temps];
    let v3 = g.v / tt;
    let n = (imgs.len() * v3) as f64;
    let centres: Vec<[f64; 3]> = imgs.iter().map(|im| centre_chemin(im)).collect();
    let pi2 = 2.0 * std::f64::consts::PI;
    let mut s_t = vec![0.0f64; tt];
    let mut p_t = vec![vec![[0.0f64; 2]; vs.len()]; tt];
    for s in 0..g.v {
        let t = (s / g.st[temps]) % tt;
        for (kk, im) in imgs.iter().enumerate() {
            let tr = trace_chemin_l(g, u, &ax, im, s, l);
            s_t[t] += tr;
            for (iv, k) in vs.iter().enumerate() {
                let mut th = 0.0f64;
                for d in 0..3 {
                    let xd = ((s / g.st[ax[d]]) % g.d[ax[d]]) as f64;
                    th += pi2 * (k[d] as f64) * (xd + l as f64 * centres[kk][d]) / g.d[ax[d]] as f64;
                }
                p_t[t][iv][0] += tr * th.cos();
                p_t[t][iv][1] += tr * th.sin();
            }
        }
    }
    for x in s_t.iter_mut() {
        *x /= n;
    }
    for a in p_t.iter_mut() {
        for c in a.iter_mut() {
            c[0] /= n;
            c[1] /= n;
        }
    }
    (s_t, p_t)
}

/// Mesure neutre (même sortie que `neutre::mesurer_ps`) : niveaux APE de `p` PUIS niveaux de blocage de `pb` (dans cet ordre,
/// op = i_niveau · N_formes + i_forme sur la liste concaténée). Mutant toujours « aucun ».
pub fn mesurer_avec_blocs(g: &GeoA, u: &[Q], p: &ParamNeutre, ks: &[i64], pb: &ParamBloc) -> Resultat<MesureNeutre> {
    mesurer_avec_blocs_orb(g, u, p, ks, &[], pb)
}

/// Comme `mesurer_avec_blocs`, avec les corrélateurs moyennés sur les orbites cubiques `orbs` (impulsions hors axe) à TOUS les
/// niveaux (APE puis blocage). Sans orbite : identique à `mesurer_avec_blocs` au bit près.
pub fn mesurer_avec_blocs_orb(g: &GeoA, u: &[Q], p: &ParamNeutre, ks: &[i64], orbs: &[Orbite], pb: &ParamBloc) -> Resultat<MesureNeutre> {
    for o in orbs {
        super::neutre::controler_vecteur(g, p.temps, o.representant)?;
    }
    let vs: Vec<[i64; 3]> = orbs.iter().flat_map(|o| o.vecteurs.iter().copied()).collect();
    let mut sv: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
    if p.niveaux.is_empty() || p.niveaux.windows(2).any(|w| w[0] >= w[1]) || p.formes.is_empty() || p.tau_max >= g.d[p.temps] {
        return Err(Erreur::Config("niveaux croissants, formes non vides, tau_max < ℓ_temps".into()));
    }
    pb.controler(&g.d, p.temps)?;
    let imgs: Vec<Vec<Vec<i8>>> = p.formes.iter().map(|&f| images(FORMES[f].1)).collect();
    let mut s: Vec<Vec<f64>> = Vec::new();
    let mut sps: Vec<Vec<Vec<[[f64; 2]; 3]>>> = vec![Vec::new(); ks.len()];
    let mut courant = u.to_vec();
    let mut fait = 0usize;
    for &n in &p.niveaux {
        for _ in fait..n {
            courant = ape_iteration(g, &courant, p.temps, p.alpha);
        }
        fait = n;
        for im in &imgs {
            let (a, b) = tranches_ps(g, &courant, p.temps, im, ks);
            s.push(a);
            for (ik, v) in b.into_iter().enumerate() {
                sps[ik].push(v);
            }
            if !vs.is_empty() {
                sv.push(tranches_vec(g, &courant, p.temps, im, &vs).1);
            }
        }
    }
    for (b, champ) in champs_bloques(g, u, p.temps, p.alpha, pb) {
        for im in &imgs {
            let (a, bb) = tranches_ps_l(g, &champ, p.temps, im, ks, 1usize << b);
            s.push(a);
            for (ik, v) in bb.into_iter().enumerate() {
                sps[ik].push(v);
            }
            if !vs.is_empty() {
                sv.push(tranches_vec_l(g, &champ, p.temps, im, &vs, 1usize << b).1);
            }
        }
    }
    let corr = correlateurs(&s, p.tau_max);
    let tt = g.d[p.temps] as f64;
    let moy = s.iter().map(|x| x.iter().sum::<f64>() / tt).collect();
    let corr_ps = correlateurs_ps(&sps, p.tau_max);
    let corr_p = corr_ps.first().cloned().unwrap_or_default();
    let mut corr_orb: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut deb = 0usize;
    for o in orbs {
        let fin = deb + o.vecteurs.len();
        let sp: Vec<Vec<Vec<[f64; 2]>>> = sv.iter().map(|op| op.iter().map(|tv| tv[deb..fin].to_vec()).collect()).collect();
        corr_orb.push(correlateurs_orbite(&sp, p.tau_max));
        deb = fin;
    }
    Ok(MesureNeutre { s, corr, moy, corr_p, corr_ps, corr_orb })
}

/// φ(t; p) d'un torelon de superliens de longueur ℓ (ℓ divise L_w), moyenne sur les ℓ points de départ : [moment][t] = (Re, Im).
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn phi_l(g: &GeoA, u: &[Q], ax: &Axes, moments: &[[i32; 2]], l: usize) -> Resultat<Vec<Vec<[f64; 2]>>> {
    let lw = g.d[ax.w];
    if l == 0 || !lw.is_multiple_of(l) {
        return Err(Erreur::Config(format!("torelon bloqué : ℓ = {l} ne divise pas L_w = {lw}")));
    }
    let (ta, tb, tt) = (g.d[ax.tr[0]], g.d[ax.tr[1]], g.d[ax.temps]);
    let nperp = (ta * tb) as f64;
    let mut out = vec![vec![[0.0f64; 2]; tt]; moments.len()];
    for t in 0..tt {
        for a in 0..ta {
            for b in 0..tb {
                let mut x = [0usize; 4];
                x[ax.temps] = t;
                x[ax.tr[0]] = a;
                x[ax.tr[1]] = b;
                let mut tot = 0.0;
                for x0 in 0..l {
                    x[ax.w] = x0;
                    let mut s = g.site(&x);
                    let mut p: Q = [1.0, 0.0, 0.0, 0.0];
                    for _ in 0..lw / l {
                        p = qmul(p, u[ax.w * g.v + s]);
                        s = decaler(g, s, ax.w, l as i64);
                    }
                    tot += p[0];
                }
                let lig = tot / l as f64;
                for (m, &[na, nb]) in moments.iter().enumerate() {
                    let th = 2.0 * std::f64::consts::PI * (na as f64 * a as f64 / ta as f64 + nb as f64 * b as f64 / tb as f64);
                    out[m][t][0] += lig * th.cos() / nperp;
                    out[m][t][1] += lig * th.sin() / nperp;
                }
            }
        }
    }
    Ok(out)
}

/// Torelons (référence processeur de `gpu-torelon` avec blocage) : niveaux APE puis niveaux de blocage ;
/// rend ([moment][paire][τ], [moment][niveau] = ⟨φ⟩_t).
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn torelons_avec_blocs(
    g: &GeoA,
    u: &[Q],
    ax: &Axes,
    niveaux: &[usize],
    alpha: f64,
    moments: &[[i32; 2]],
    tau_max: usize,
    pb: &ParamBloc,
) -> Resultat<(Vec<Vec<Vec<f64>>>, Vec<Vec<[f64; 2]>>)> {
    let mut par_niveau: Vec<Vec<Vec<[f64; 2]>>> = Vec::new(); // [niveau][moment][t]
    let mut courant = u.to_vec();
    let mut fait = 0usize;
    for &n in niveaux {
        for _ in fait..n {
            courant = ape_iteration(g, &courant, ax.temps, alpha);
        }
        fait = n;
        par_niveau.push(phi_l(g, &courant, ax, moments, 1)?);
    }
    for (b, champ) in champs_bloques(g, u, ax.temps, alpha, pb) {
        par_niveau.push(phi_l(g, &champ, ax, moments, 1usize << b)?);
    }
    let mut corr = Vec::new();
    let mut pm = Vec::new();
    for m in 0..moments.len() {
        let phis: Vec<Vec<[f64; 2]>> = par_niveau.iter().map(|lv| lv[m].clone()).collect();
        corr.push(correlateur(&phis, tau_max, Mutant::Aucun));
        pm.push(phi_moyen(&phis));
    }
    Ok((corr, pm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::su2::spectro::transformer_jauge;

    fn champ_aleatoire(g: &GeoA, graine: u64) -> Vec<Q> {
        // champ SU(2) pseudo-aléatoire (générateur congruentiel : test seulement)
        let mut x = graine.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut r = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        (0..4 * g.v)
            .map(|_| {
                let q = [r() + 1.5, r(), r(), r()];
                qscale(q, 1.0 / qnorm(q))
            })
            .collect()
    }

    fn pn(niveaux: Vec<usize>, formes: Vec<usize>) -> ParamNeutre {
        ParamNeutre { temps: 3, niveaux, alpha: 0.5, formes, tau_max: 3, mutant: super::super::neutre::MutantN::Aucun }
    }

    #[test]
    fn bloc_froid_et_jauge() {
        let g = GeoA::analyser("6,6,6,8").unwrap();
        let pb = ParamBloc { niveaux: vec![1, 2], ape_bloc: 1, lissage_bloc: 1, p_bloc: 0.7 };
        let p = pn(vec![0, 2], vec![0, 1, 2, 3]);
        let froid = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
        let m = mesurer_avec_blocs(&g, &froid, &p, &[1], &pb).unwrap();
        assert!(m.moy.iter().all(|x| (x - 1.0).abs() < 1e-14));
        assert!(m.corr_ps[0].iter().flatten().all(|x| x.abs() < 1e-13));
        let u = champ_aleatoire(&g, 5);
        let m0 = mesurer_avec_blocs(&g, &u, &p, &[1], &pb).unwrap();
        let mut uj = u.clone();
        transformer_jauge(&g, &mut uj, 17);
        let m1 = mesurer_avec_blocs(&g, &uj, &p, &[1], &pb).unwrap();
        let e = m0.corr.iter().flatten().zip(m1.corr.iter().flatten()).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(e < 1e-13, "jauge {e}");
    }

    /// Réponse connue : avec p = 0 et sans lissage, B¹_i(x) = U_i(x) U_i(x+î) ; la plaquette de superliens du niveau 1
    /// est la boucle 2×2 (S22) des liens ordinaires, et le torelon du niveau 1 est la ligne de Polyakov ordinaire.
    #[test]
    fn bloc_p0_egale_boucle_2x2_et_polyakov() {
        let g = GeoA::analyser("8,6,6,8").unwrap();
        let u = champ_aleatoire(&g, 9);
        let pb = ParamBloc { niveaux: vec![1], ape_bloc: 0, lissage_bloc: 0, p_bloc: 0.0 };
        let p_bl = pn(vec![0], vec![0]); // P11 au niveau APE 0, puis P11 de superliens (niveau 1)
        let m = mesurer_avec_blocs(&g, &u, &p_bl, &[], &pb).unwrap();
        let s22 = mesurer_avec_blocs(&g, &u, &pn(vec![0], vec![2]), &[], &ParamBloc::vide()).unwrap();
        let e = m.s[1].iter().zip(&s22.s[0]).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(e < 1e-13, "P11 bloc 1 contre S22 : {e}");
        let ax = Axes::nouveaux(0, 3).unwrap();
        let b1 = champs_bloques(&g, &u, 3, 0.5, &pb).remove(0).1;
        let a = phi_l(&g, &b1, &ax, &[[0, 0], [1, 0]], 2).unwrap();
        let b = phi_l(&g, &u, &ax, &[[0, 0], [1, 0]], 1).unwrap();
        let e2 = a.iter().flatten().zip(b.iter().flatten()).fold(0.0f64, |m, (x, y)| m.max((x[0] - y[0]).abs()).max((x[1] - y[1]).abs()));
        assert!(e2 < 1e-13, "torelon bloc 1 contre Polyakov : {e2}");
        // et la référence torelon d'origine (`super::phi`) donne la même chose au niveau 0
        let c = super::super::phi(&g, &u, &ax, &[[0, 0], [1, 0]], Mutant::Aucun);
        let e3 = c.iter().flatten().zip(b.iter().flatten()).fold(0.0f64, |m, (x, y)| m.max((x[0] - y[0]).abs()).max((x[1] - y[1]).abs()));
        assert!(e3 < 1e-13, "phi_l(1) contre phi : {e3}");
    }

    #[test]
    fn bloc_controles() {
        let pb = ParamBloc { niveaux: vec![1, 3], ape_bloc: 0, lissage_bloc: 0, p_bloc: 1.0 };
        assert!(pb.controler(&[16, 16, 16, 16], 3).is_ok());
        assert!(pb.controler(&[8, 16, 16, 16], 3).is_err()); // 2^3 = 8 ≥ 8
        assert!(ParamBloc { niveaux: vec![2, 1], ..pb.clone() }.controler(&[16; 4], 3).is_err());
        assert!(ParamBloc { niveaux: vec![0], ..pb.clone() }.controler(&[16; 4], 3).is_err());
        assert_eq!(pb.etiquette(2), "B2");
        assert_eq!(ParamBloc { lissage_bloc: 3, ..pb }.etiquette(2), "B2s3");
    }
}
