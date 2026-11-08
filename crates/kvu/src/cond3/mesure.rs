//! Mesure et accumulation (SPEC_T3 §2.6, §3.1, §4.1, §7.1 M6) : pour chaque réplique, les SOMMES globales FP64
//! Σ O_p, Σ σ̄_ē, Σ O_p σ̄_ē par bloc de temps et par paire (jamais de moyenne de covariances par morceaux : piège « moyennes par
//! morceaux »), accumulées par lots de `lot` itérations (produit matriciel par blocs, réduction dans un ordre FIXÉ : le résultat ne
//! dépend ni du nombre de fils ni du découpage en tâches).
//!
//! Estimateur : R̂_p(ē) = −(Σ Oσ/n − (Σ O/n)(Σ σ/n)) (b ⩾ 2 : E[X(O)] = 0, SPEC §2.2) et, avec deux répliques indépendantes A, B,
//! Ŝ = R̂^A · R̂^B, sans biais pour |R|² (SPEC §2.6).

use super::geom::Geom;
use super::modele::{avancer, Canal, Modele, Ordonnancement, StatsEchantillonneur};
use super::paires::Paires;
use crate::commun::{Erreur, Resultat};
use rayon::prelude::*;
use std::time::Instant;

/// Paramètres d'exécution d'UNE réplique.
#[derive(Clone, Debug)]
pub struct ConfigMesure {
    /// Itérations écartées (décorrélation des répliques).
    pub n_dec: usize,
    /// Itérations mesurées.
    pub n_mes: usize,
    pub n_or: usize,
    /// Nombre de blocs de temps (2 : moitiés tôt/tard ; 20 : jackknife de temps des oracles). `n_mes` doit être multiple de `nblk`.
    pub nblk: usize,
    /// Taille des lots d'accumulation.
    pub lot: usize,
    /// Scores des trois relèvements séparés (3·nc composantes) au lieu du moyen.
    pub separes: bool,
    pub ord: Ordonnancement,
    /// Contrôles (contrainte, unitarité, Ward) tous les `controle_chaque` balayages.
    pub controle_chaque: usize,
    /// Plaquettes dont on garde la série temporelle (positions dans `Paires::faces`).
    pub series_faces: Vec<usize>,
    /// Sondes dont on garde les composantes de score en série temporelle (positions dans `Paires::sondes`).
    pub series_sondes: Vec<usize>,
    /// Numéro du premier balayage (compteur Philox).
    pub sweep0: u32,
    /// Moyennes de plaquette par classe t aux contrôles (INFO).
    pub classes_t: bool,
    /// Canaux d'oracle évalués à chaque itération mesurée.
    pub canaux: Vec<Canal>,
}

impl Default for ConfigMesure {
    fn default() -> Self {
        ConfigMesure {
            n_dec: 200,
            n_mes: 10_000,
            n_or: 1,
            nblk: 2,
            lot: 64,
            separes: false,
            ord: Ordonnancement::Simultane,
            controle_chaque: 50,
            series_faces: Vec::new(),
            series_sondes: Vec::new(),
            sweep0: 0,
            classes_t: false,
            canaux: Vec::new(),
        }
    }
}

/// Contrôles de trajectoire à une itération donnée.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controle {
    pub iteration: usize,
    pub contrainte: f64,
    pub unitarite: f64,
    pub ward: f64,
}

/// Sortie d'une réplique : sommes globales par bloc de temps et diagnostics.
#[derive(Clone, Debug)]
pub struct SortieReplique {
    pub nc: usize,
    pub nblk: usize,
    pub nfaces: usize,
    pub nsondes: usize,
    pub npaires: usize,
    /// Itérations mesurées par bloc.
    pub n_blk: Vec<u64>,
    /// Σ O : [blk][face].
    pub so: Vec<f64>,
    /// Σ σ : [blk][sonde][composante].
    pub ss: Vec<f64>,
    /// Σ O σ : [blk][paire][composante].
    pub sos: Vec<f64>,
    /// Paires à terme E[X(O)] non nul (indices q) et Σ X̄(O_p) : [blk][k][composante].
    pub px: Vec<u32>,
    pub sx: Vec<f64>,
    /// Séries temporelles des plaquettes `series_pis` : [k][t] (t sur les itérations mesurées).
    pub serie_o: Vec<f64>,
    /// Séries des composantes de score des sondes `series_cis` : [sonde·nc + composante][t], en f32.
    pub serie_s: Vec<f32>,
    pub n_serie_o: usize,
    pub n_serie_s: usize,
    /// Séries des canaux d'oracle : [valeur][t] ; `n_canal` valeurs.
    pub serie_canaux: Vec<f64>,
    pub n_canal: usize,
    /// Moyenne de O sur toutes les plaquettes de chaque classe t = 0..=d−2, tous les `controle_chaque` balayages mesurés.
    pub moy_classes_t: Vec<Vec<f64>>,
    pub controles: Vec<Controle>,
    /// Empreintes sha256 du champ tous les `JALON` itérations (rejeu au bit d'un préfixe, Règle 15 FIAB).
    pub jalons: Vec<(usize, String)>,
    pub stats: StatsEchantillonneur,
    pub empreinte_finale: String,
    pub secondes: f64,
    pub n_mes: usize,
}

/// Période (en itérations) des empreintes de jalon du champ.
pub const JALON: usize = 200;

/// Tampon de lot : O [face][t], σ [sonde·nc + comp][t].
struct Lot {
    lot: usize,
    nb: usize,
    obuf: Vec<f64>,
    sbuf: Vec<f64>,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = [0.0f64; 4];
    let mut i = 0;
    while i + 4 <= n {
        s[0] += a[i] * b[i];
        s[1] += a[i + 1] * b[i + 1];
        s[2] += a[i + 2] * b[i + 2];
        s[3] += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut r = (s[0] + s[1]) + (s[2] + s[3]);
    while i < n {
        r += a[i] * b[i];
        i += 1;
    }
    r
}

/// Paires traitées par tâche rayon (le découpage ne change pas le résultat : chaque produit est calculé en entier par une tâche).
const PAIRES_PAR_TACHE: usize = 4096;

fn vider_lot(lot: &mut Lot, p: &Paires, nc: usize, sos_blk: &mut [f64]) {
    let nb = lot.nb;
    if nb == 0 {
        return;
    }
    let l = lot.lot;
    let (obuf, sbuf) = (&lot.obuf, &lot.sbuf);
    sos_blk.par_chunks_mut(PAIRES_PAR_TACHE * nc).enumerate().for_each(|(ic, out)| {
        let q0 = ic * PAIRES_PAR_TACHE;
        let nq = out.len() / nc;
        let mut s = p.debut.partition_point(|&d| d <= q0 as u64) - 1;
        for j in 0..nq {
            let q = q0 + j;
            while q as u64 >= p.debut[s + 1] {
                s += 1;
            }
            let f = p.face_loc[q] as usize;
            let orow = &obuf[f * l..f * l + nb];
            for c in 0..nc {
                let srow = &sbuf[(s * nc + c) * l..(s * nc + c) * l + nb];
                out[j * nc + c] += dot(orow, srow);
            }
        }
    });
    lot.nb = 0;
}

/// Exécute UNE réplique sur le modèle `m` (déjà au départ exact : champ de la configuration, V figé) :
/// `n_dec` itérations écartées puis `n_mes` mesurées. Le numéro de balayage Philox est `sweep0 + t`.
/// Le numéro de balayage Philox est un u32 : `sweep0 + n_dec + n_mes` ne doit pas le dépasser (sinon des tirages se répéteraient).
pub fn verifier_compteur_philox(cfg: &ConfigMesure) -> Resultat<()> {
    if cfg.sweep0 as u64 + (cfg.n_dec + cfg.n_mes) as u64 > u32::MAX as u64 {
        return Err(Erreur::Config(format!("compteur Philox : sweep0 + n_dec + n_mes = {} dépasse u32", cfg.sweep0 as u64 + (cfg.n_dec + cfg.n_mes) as u64)));
    }
    Ok(())
}

pub fn executer_replique<M: Modele + ?Sized>(m: &mut M, p: &Paires, cfg: &ConfigMesure) -> Resultat<SortieReplique> {
    if cfg.n_mes == 0 || cfg.nblk == 0 || !cfg.n_mes.is_multiple_of(cfg.nblk) {
        return Err(Erreur::Config(format!("n_mes = {} doit être un multiple non nul de nblk = {}", cfg.n_mes, cfg.nblk)));
    }
    if cfg.lot == 0 {
        return Err(Erreur::Config("lot = 0".into()));
    }
    verifier_compteur_philox(cfg)?;
    let t0 = Instant::now();
    let g: std::sync::Arc<Geom> = m.geom().clone();
    let nc = if cfg.separes { 3 * m.ncomp() } else { m.ncomp() };
    let (nf, ns, np) = (p.faces.len(), p.sondes.len(), p.npaires());
    let nblk = cfg.nblk;
    let lon_blk = cfg.n_mes / nblk;
    let mut so = vec![0.0f64; nblk * nf];
    let mut ss = vec![0.0f64; nblk * ns * nc];
    let mut sos = vec![0.0f64; nblk * np * nc];
    let paires_x: Vec<(u32, u32)> = p.px.iter().map(|&q| p.paire(q as usize)).collect();
    let npx = paires_x.len();
    let mut sx = vec![0.0f64; nblk * npx * nc];
    let mut xvec = vec![0.0f64; npx * nc];
    let mut n_blk = vec![0u64; nblk];
    let mut lot = Lot { lot: cfg.lot, nb: 0, obuf: vec![0.0; nf * cfg.lot], sbuf: vec![0.0; ns * nc * cfg.lot] };
    let mut ovec = vec![0.0f64; nf];
    let mut svec = vec![0.0f64; ns * nc];
    let n_so = cfg.series_faces.len();
    let n_ss = cfg.series_sondes.len() * nc;
    if cfg.series_faces.iter().any(|&f| f >= nf) || cfg.series_sondes.iter().any(|&s| s >= ns) {
        return Err(Erreur::Config("série demandée hors des listes de faces ou de sondes".into()));
    }
    let mut serie_o = vec![0.0f64; n_so * cfg.n_mes];
    let mut serie_s = vec![0.0f32; n_ss * cfg.n_mes];
    let mut controles = Vec::new();
    let mut jalons: Vec<(usize, String)> = Vec::new();
    let largeurs: Vec<usize> = cfg.canaux.iter().map(|c| c.largeur(g.d)).collect();
    let n_canal: usize = largeurs.iter().sum();
    let mut serie_canaux = vec![0.0f64; n_canal * cfg.n_mes];
    let mut tmp_canal = vec![0.0f64; largeurs.iter().cloned().max().unwrap_or(0).max(1)];
    let ntmax = if g.d >= 2 { g.d - 1 } else { 1 };
    let mut moy_classes_t: Vec<Vec<f64>> = Vec::new();
    let classes: Vec<u8> = if cfg.classes_t { (0..g.nplaq()).map(|pi| g.classe_t(pi) as u8).collect() } else { Vec::new() };
    let toutes: Vec<u32> = if cfg.classes_t { (0..g.nplaq() as u32).collect() } else { Vec::new() };
    let mut touts_o = vec![0.0f64; toutes.len()];
    let total = cfg.n_dec + cfg.n_mes;
    for t in 0..total {
        let sweep = cfg.sweep0.wrapping_add(t as u32);
        avancer(m, cfg.ord, sweep, cfg.n_or);
        if (t + 1).is_multiple_of(JALON) {
            jalons.push((t + 1, m.empreinte()));
        }
        if cfg.controle_chaque > 0 && (t + 1).is_multiple_of(cfg.controle_chaque) {
            controles.push(Controle {
                iteration: t + 1,
                contrainte: m.residu_contrainte(),
                unitarite: m.defaut_unitarite(),
                ward: m.residu_ward(),
            });
        }
        if t < cfg.n_dec {
            continue;
        }
        let tm = t - cfg.n_dec;
        let blk = tm / lon_blk;
        m.observables(&p.faces, &mut ovec);
        m.scores(&p.sondes, cfg.separes, &mut svec);
        let tb = lot.nb;
        // tampons transposés
        lot.obuf.par_chunks_mut(cfg.lot).zip(ovec.par_iter()).for_each(|(row, &o)| row[tb] = o);
        {
            let l = cfg.lot;
            lot.sbuf.par_chunks_mut(l).enumerate().for_each(|(k, row)| row[tb] = svec[k]);
        }
        lot.nb += 1;
        for (f, o) in ovec.iter().enumerate() {
            so[blk * nf + f] += o;
        }
        for (k, s) in svec.iter().enumerate() {
            ss[blk * ns * nc + k] += s;
        }
        n_blk[blk] += 1;
        if npx > 0 {
            m.terme_x(&paires_x, cfg.separes, &mut xvec);
            for (k, x) in xvec.iter().enumerate() {
                sx[blk * npx * nc + k] += x;
            }
        }
        // séries temporelles (extraites des vecteurs déjà calculés)
        for (k, &f) in cfg.series_faces.iter().enumerate() {
            serie_o[k * cfg.n_mes + tm] = ovec[f];
        }
        for (k, &sd) in cfg.series_sondes.iter().enumerate() {
            for c in 0..nc {
                serie_s[(k * nc + c) * cfg.n_mes + tm] = svec[sd * nc + c] as f32;
            }
        }
        {
            let mut off = 0;
            for (c, &w) in cfg.canaux.iter().zip(&largeurs) {
                if !m.canal(c, &mut tmp_canal[..w]) {
                    return Err(Erreur::Config(format!("canal {c:?} non géré par le groupe {}", m.groupe())));
                }
                for k in 0..w {
                    serie_canaux[(off + k) * cfg.n_mes + tm] = tmp_canal[k];
                }
                off += w;
            }
        }
        if cfg.classes_t && cfg.controle_chaque > 0 && (t + 1).is_multiple_of(cfg.controle_chaque) {
            m.observables(&toutes, &mut touts_o);
            let mut somme = vec![0.0f64; ntmax];
            let mut n = vec![0usize; ntmax];
            for (pi, o) in touts_o.iter().enumerate() {
                let c = classes[pi] as usize;
                if c < ntmax {
                    somme[c] += o;
                    n[c] += 1;
                }
            }
            moy_classes_t.push(somme.iter().zip(&n).map(|(s, &k)| s / k.max(1) as f64).collect());
        }
        let fin_de_bloc = (tm + 1).is_multiple_of(lon_blk);
        if lot.nb == cfg.lot || fin_de_bloc {
            vider_lot(&mut lot, p, nc, &mut sos[blk * np * nc..(blk + 1) * np * nc]);
        }
    }
    Ok(SortieReplique {
        nc,
        nblk,
        nfaces: nf,
        nsondes: ns,
        npaires: np,
        n_blk,
        so,
        ss,
        sos,
        px: p.px.clone(),
        sx,
        serie_o,
        serie_s,
        n_serie_o: n_so,
        n_serie_s: n_ss,
        serie_canaux,
        n_canal,
        moy_classes_t,
        controles,
        jalons,
        stats: m.stats(),
        empreinte_finale: m.empreinte(),
        secondes: t0.elapsed().as_secs_f64(),
        n_mes: cfg.n_mes,
    })
}

// ------------------------------------------------------------------------------------------------------------
// Estimateurs
// ------------------------------------------------------------------------------------------------------------

impl SortieReplique {
    /// R̂ de toutes les paires pour l'ensemble de blocs `blocs` : [paire][composante] (−covariance, SPEC §2.2), à partir des SOMMES
    /// globales des blocs choisis.
    pub fn r_chapeau(&self, p: &Paires, blocs: &[usize]) -> Vec<f64> {
        let (nc, nf, ns, np) = (self.nc, self.nfaces, self.nsondes, self.npaires);
        let n: f64 = blocs.iter().map(|&b| self.n_blk[b] as f64).sum();
        let mut mo = vec![0.0f64; nf];
        let mut ms = vec![0.0f64; ns * nc];
        let mut tot = vec![0.0f64; np * nc];
        for &b in blocs {
            for f in 0..nf {
                mo[f] += self.so[b * nf + f];
            }
            for k in 0..ns * nc {
                ms[k] += self.ss[b * ns * nc + k];
            }
            for k in 0..np * nc {
                tot[k] += self.sos[b * np * nc + k];
            }
        }
        for x in mo.iter_mut() {
            *x /= n;
        }
        for x in ms.iter_mut() {
            *x /= n;
        }
        let mut r = vec![0.0f64; np * nc];
        for s in 0..ns {
            for q in p.debut[s] as usize..p.debut[s + 1] as usize {
                let f = p.face_loc[q] as usize;
                for c in 0..nc {
                    r[q * nc + c] = -(tot[q * nc + c] / n - mo[f] * ms[s * nc + c]);
                }
            }
        }
        // terme E[X(O)] des paires dont la plaquette contient un lien de la ligne
        let npx = self.px.len();
        for (k, &q) in self.px.iter().enumerate() {
            for c in 0..nc {
                let mut t = 0.0;
                for &b in blocs {
                    t += self.sx[b * npx * nc + k * nc + c];
                }
                r[q as usize * nc + c] += t / n;
            }
        }
        r
    }

    /// Moyenne de O sur chaque bloc de temps et sur l'ensemble : F̂_p par face.
    pub fn f_chapeau(&self, blocs: &[usize]) -> Vec<f64> {
        let n: f64 = blocs.iter().map(|&b| self.n_blk[b] as f64).sum();
        let mut mo = vec![0.0f64; self.nfaces];
        for &b in blocs {
            for f in 0..self.nfaces {
                mo[f] += self.so[b * self.nfaces + f];
            }
        }
        mo.iter().map(|x| x / n).collect()
    }
}

/// Ŝ = R̂^A · R̂^B (produit scalaire sur les composantes) de chaque paire, pour les blocs `blocs_a` de A et `blocs_b` de B.
pub fn s_chapeau(a: &SortieReplique, b: &SortieReplique, p: &Paires, blocs_a: &[usize], blocs_b: &[usize]) -> Vec<f64> {
    let ra = a.r_chapeau(p, blocs_a);
    let rb = b.r_chapeau(p, blocs_b);
    let nc = a.nc;
    (0..a.npaires).map(|q| (0..nc).map(|c| ra[q * nc + c] * rb[q * nc + c]).sum()).collect()
}

/// Somme et effectif par tranche b (0..=bmax) d'une quantité par paire.
pub fn somme_par_tranche(p: &Paires, v: &[f64]) -> (Vec<f64>, Vec<u64>) {
    let nb = p.bmax as usize + 1;
    let mut somme = vec![0.0f64; nb];
    let mut n = vec![0u64; nb];
    for q in 0..p.npaires() {
        let b = p.tranche[q] as usize;
        somme[b] += v[q];
        n[b] += 1;
    }
    (somme, n)
}

// ------------------------------------------------------------------------------------------------------------
// Temps d'autocorrélation (Sokal, fenêtre automatique)
// ------------------------------------------------------------------------------------------------------------

/// τ_int de Sokal (fenêtre automatique de facteur `c`, 6 par défaut) d'une série ; 0,5 si la série est constante. L'autocovariance est l'estimateur
/// « biaisé » en 1/N (convention des séries temporelles) : l'écart à 1/(N − w) est O(w/N), soit < 1 % aux fenêtres utilisées (w ≲ 50, N = 10⁴).
pub fn tau_int(x: &[f64], c: f64) -> f64 {
    let n = x.len();
    if n < 4 {
        return 0.5;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = x.iter().map(|v| v - m).collect();
    let c0 = dot(&y, &y) / n as f64;
    if c0 <= 0.0 || !c0.is_finite() {
        return 0.5;
    }
    let mut tau = 0.5;
    let wmax = (n / 2).max(1);
    for w in 1..wmax {
        let cw = dot(&y[..n - w], &y[w..]) / n as f64;
        tau += cw / c0;
        if (w as f64) >= c * tau {
            break;
        }
    }
    tau.max(0.5)
}

pub fn tau_int_f32(x: &[f32], c: f64) -> f64 {
    let v: Vec<f64> = x.iter().map(|&a| a as f64).collect();
    tau_int(&v, c)
}

impl SortieReplique {
    /// τ_int (c = 6) de chaque série conservée : plaquettes puis composantes de score.
    pub fn taus(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_mes;
        let to: Vec<f64> = (0..self.n_serie_o).into_par_iter().map(|k| tau_int(&self.serie_o[k * n..(k + 1) * n], 6.0)).collect();
        let ts: Vec<f64> =
            (0..self.n_serie_s).into_par_iter().map(|k| tau_int_f32(&self.serie_s[k * n..(k + 1) * n], 6.0)).collect();
        (to, ts)
    }
}

/// Sommes par chunk de temps pour la corrélation croisée A/B à décalage 0 : par série et par chunk,
/// (Σa, Σb, Σab, Σa², Σb²) ; `nchunk` chunks de longueur n_mes / nchunk. Sortie [chunk][série][5].
pub fn sommes_croisees(a: &SortieReplique, b: &SortieReplique, nchunk: usize) -> Vec<f64> {
    let n = a.n_mes;
    assert!(nchunk > 0 && n.is_multiple_of(nchunk), "sommes croisées : n_mes = {n} doit être un multiple non nul de nchunk = {nchunk}");
    let len = n / nchunk;
    let nso = a.n_serie_o;
    let nss = a.n_serie_s;
    let nser = nso + nss;
    let mut out = vec![0.0f64; nchunk * nser * 5];
    for ch in 0..nchunk {
        for k in 0..nser {
            let (mut sa, mut sb, mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for t in ch * len..(ch + 1) * len {
                let (x, y) = if k < nso {
                    (a.serie_o[k * n + t], b.serie_o[k * n + t])
                } else {
                    let kk = k - nso;
                    (a.serie_s[kk * n + t] as f64, b.serie_s[kk * n + t] as f64)
                };
                sa += x;
                sb += y;
                sab += x * y;
                saa += x * x;
                sbb += y * y;
            }
            let o = (ch * nser + k) * 5;
            out[o..o + 5].copy_from_slice(&[sa, sb, sab, saa, sbb]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cond3::modele::Ordonnancement;
    use crate::cond3::paires::Paires;
    use crate::cond3::reduction::reduire;
    use crate::cond3::su2::{champ_haar_test, FibreSu2};

    #[test]
    fn tau_int_d_un_ar1_connu() {
        // AR(1) x_t = φ x_{t−1} + ε : τ_int = ½ (1 + φ)/(1 − φ)
        let phi = 0.6f64;
        let mut x = vec![0.0f64; 200_000];
        let mut s = 12345u64;
        let mut gauss = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        for t in 1..x.len() {
            x[t] = phi * x[t - 1] + gauss();
        }
        let att = 0.5 * (1.0 + phi) / (1.0 - phi);
        let tau = tau_int(&x, 6.0);
        assert!((tau - att).abs() < 0.15 * att, "τ_int = {tau}, attendu {att}");
        assert_eq!(tau_int(&[1.0; 100], 6.0), 0.5);
    }

    #[test]
    fn reduction_de_production_egale_la_somme_naive_par_paire() {
        // deux répliques CPU courtes (d = 4, L = 6, β = 2,5) : Σ Ŝ par tranche de `reduire` = somme naive de `s_chapeau`
        let g = std::sync::Arc::new(Geom::nouvelle(4, 6).unwrap());
        let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 2, 4);
        let u = champ_haar_test(g.nliens(), 77);
        let cfg = ConfigMesure { n_dec: 4, n_mes: 12, nblk: 2, lot: 4, controle_chaque: 0, ..ConfigMesure::default() };
        let mut sorties = Vec::new();
        for r in 0..2u64 {
            let mut f = FibreSu2::nouveau(g.clone(), u.clone(), 2.5, 100 + r);
            sorties.push(executer_replique(&mut f, &p, &cfg).unwrap());
        }
        let red = reduire(&sorties[0], &sorties[1], &p, 1, None, None);
        for (somme, blocs_a, blocs_b) in [(&red.s_full, vec![0usize, 1], vec![0usize, 1]), (&red.s_early, vec![0], vec![0]), (&red.s_late, vec![1], vec![1])] {
            let s = s_chapeau(&sorties[0], &sorties[1], &p, &blocs_a, &blocs_b);
            let (naif, n) = somme_par_tranche(&p, &s);
            for b in 2..=4usize {
                assert_eq!(red.n_paires[b], n[b]);
                assert!((somme[b] - naif[b]).abs() <= 1e-12 * naif[b].abs().max(1e-30), "tranche {b} : {} contre {}", somme[b], naif[b]);
            }
        }
        // un seul balayage de départ commun ne rend pas les répliques identiques (flux distincts)
        assert_ne!(sorties[0].empreinte_finale, sorties[1].empreinte_finale);
        let _ = Ordonnancement::Simultane;
    }

    #[test]
    fn compteur_philox_refuse_le_depassement_de_u32() {
        let ok = ConfigMesure { sweep0: 0, n_dec: 3, n_mes: 4, ..ConfigMesure::default() };
        assert!(verifier_compteur_philox(&ok).is_ok());
        let limite = ConfigMesure { sweep0: u32::MAX - 7, n_dec: 3, n_mes: 4, ..ConfigMesure::default() };
        assert!(verifier_compteur_philox(&limite).is_ok(), "sweep0 + n_dec + n_mes = u32::MAX est encore représentable");
        let trop = ConfigMesure { sweep0: u32::MAX - 6, n_dec: 3, n_mes: 4, ..ConfigMesure::default() };
        assert!(verifier_compteur_philox(&trop).is_err());
    }

    #[test]
    fn lots_de_tailles_differentes_donnent_les_memes_sommes() {
        let g = std::sync::Arc::new(Geom::nouvelle(4, 6).unwrap());
        let p = Paires::nouvelles(&g, g.faces(), g.sondes_paires(), 2, 3);
        let u = champ_haar_test(g.nliens(), 78);
        let run = |lot: usize| {
            let mut f = FibreSu2::nouveau(g.clone(), u.clone(), 2.5, 7);
            executer_replique(&mut f, &p, &ConfigMesure { n_dec: 2, n_mes: 12, nblk: 2, lot, controle_chaque: 0, ..ConfigMesure::default() }).unwrap()
        };
        let (a, b) = (run(3), run(12));
        assert_eq!(a.empreinte_finale, b.empreinte_finale);
        for (x, y) in a.sos.iter().zip(&b.sos) {
            assert!((x - y).abs() <= 1e-12 * x.abs().max(1e-12));
        }
        assert_eq!(a.so, b.so);
        assert_eq!(a.ss, b.ss);
    }
}
