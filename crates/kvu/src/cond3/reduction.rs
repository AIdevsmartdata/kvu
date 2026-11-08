//! Réduction de production (SPEC_T3 §2.6-§2.7, §7.1 M7) : à partir des sommes globales de deux répliques A et B, forme Ŝ = R̂^A · R̂^B de CHAQUE paire
//! (jamais de moyenne de covariances par morceaux), et n'en garde que des sommes par tranche b : Ŝ complet, Ŝ_tôt (premières moitiés A₁·B₁),
//! Ŝ_tard (A₂·B₂), |R̂^A|² et |R̂^B|² (diagnostic du biais d'une seule réplique), et l'histogramme par classe de charge de tube (compte, ΣŜ, Σf).
//! Les accumulateurs complets (≈ 1 Go par moitié et par réplique à L = 24) ne sont donc jamais écrits sur disque.

use super::mesure::SortieReplique;
use super::paires::Paires;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Reduction {
    pub bmax: usize,
    pub nbin: usize,
    pub n_paires: Vec<u64>,
    pub s_full: Vec<f64>,
    pub s_early: Vec<f64>,
    pub s_late: Vec<f64>,
    pub a2: Vec<f64>,
    pub b2: Vec<f64>,
    /// Histogrammes [b·nbin + bin] : effectif, Σ Ŝ (complet), Σ f.
    pub hist_n: Vec<u64>,
    pub hist_s: Vec<f64>,
    pub hist_f: Vec<f64>,
}

struct Ensemble {
    n: f64,
    mo: Vec<f64>,
    ms: Vec<f64>,
}

fn ensemble(s: &SortieReplique, blocs: &[usize]) -> Ensemble {
    let (nc, nf, ns) = (s.nc, s.nfaces, s.nsondes);
    let n: f64 = blocs.iter().map(|&b| s.n_blk[b] as f64).sum();
    let mut mo = vec![0.0f64; nf];
    let mut ms = vec![0.0f64; ns * nc];
    for &b in blocs {
        for f in 0..nf {
            mo[f] += s.so[b * nf + f];
        }
        for k in 0..ns * nc {
            ms[k] += s.ss[b * ns * nc + k];
        }
    }
    for x in mo.iter_mut() {
        *x /= n;
    }
    for x in ms.iter_mut() {
        *x /= n;
    }
    Ensemble { n, mo, ms }
}

#[derive(Clone)]
struct Acc {
    n: Vec<u64>,
    sf: Vec<f64>,
    se: Vec<f64>,
    sl: Vec<f64>,
    a2: Vec<f64>,
    b2: Vec<f64>,
    hn: Vec<u64>,
    hs: Vec<f64>,
    hf: Vec<f64>,
}

/// `bins` : classe de charge de tube de chaque paire (u16 < nbin) et `fval` leur charge f, ou rien.
/// Préconditions (garanties par `run::valider_parametres` sur le chemin de production ; sinon panique) : deux moitiés de temps, fenêtre b ⩾ 2, au plus 9 composantes.
pub fn reduire(a: &SortieReplique, b: &SortieReplique, p: &Paires, nbin: usize, bins: Option<&[u16]>, fval: Option<&[f32]>) -> Reduction {
    assert_eq!(a.nblk, 2, "réduction de production : deux moitiés de temps");
    assert_eq!(a.nc, b.nc);
    assert!(p.px.is_empty(), "réduction de production : fenêtre b ⩾ 2 (aucun terme E[X(O)])");
    assert!(a.nc <= 9, "réduction de production : au plus 9 composantes de score (SU(2) séparé)");
    let nc = a.nc;
    let np = a.npaires;
    let nb = p.bmax as usize + 1;
    let ens: Vec<Ensemble> = [a, b].iter().flat_map(|s| [ensemble(s, &[0, 1]), ensemble(s, &[0]), ensemble(s, &[1])]).collect();
    // ens : A_full, A_early, A_late, B_full, B_early, B_late
    let ns = p.sondes.len();
    let avec_hist = bins.is_some();
    let init = move || Acc {
        n: vec![0; nb],
        sf: vec![0.0; nb],
        se: vec![0.0; nb],
        sl: vec![0.0; nb],
        a2: vec![0.0; nb],
        b2: vec![0.0; nb],
        hn: if avec_hist { vec![0; nb * nbin] } else { Vec::new() },
        hs: if avec_hist { vec![0.0; nb * nbin] } else { Vec::new() },
        hf: if avec_hist { vec![0.0; nb * nbin] } else { Vec::new() },
    };
    // Déterminisme : sondes découpées en tranches de PAS fixe, chaque tranche sommée en séquence, tranches fusionnées dans l'ordre.
    const PAS: usize = 16;
    let nch = ns.div_ceil(PAS);
    let partiels: Vec<Acc> = (0..nch)
        .into_par_iter()
        .map(|ch| {
            let mut ac = init();
            for s in ch * PAS..((ch + 1) * PAS).min(ns) {
              for q in p.debut[s] as usize..p.debut[s + 1] as usize {
                let f = p.face_loc[q] as usize;
                let bt = p.tranche[q] as usize;
                // R̂ des trois ensembles de blocs pour A et B
                let mut r = [[0.0f64; 9]; 6];
                for (k, e) in ens.iter().enumerate() {
                    let (rep, bl): (&SortieReplique, [usize; 2]) = match k {
                        0 => (a, [0, 1]),
                        1 => (a, [0, 0]),
                        2 => (a, [1, 1]),
                        3 => (b, [0, 1]),
                        4 => (b, [0, 0]),
                        _ => (b, [1, 1]),
                    };
                    for c in 0..nc {
                        let tot = match k % 3 {
                            0 => rep.sos[q * nc + c] + rep.sos[np * nc + q * nc + c],
                            1 => rep.sos[q * nc + c],
                            _ => rep.sos[np * nc + q * nc + c],
                        };
                        let _ = bl;
                        r[k][c] = -(tot / e.n - e.mo[f] * e.ms[s * nc + c]);
                    }
                }
                let dotp = |x: &[f64; 9], y: &[f64; 9]| -> f64 { (0..nc).map(|c| x[c] * y[c]).sum() };
                let s_full = dotp(&r[0], &r[3]);
                ac.n[bt] += 1;
                ac.sf[bt] += s_full;
                ac.se[bt] += dotp(&r[1], &r[4]);
                ac.sl[bt] += dotp(&r[2], &r[5]);
                ac.a2[bt] += dotp(&r[0], &r[0]);
                ac.b2[bt] += dotp(&r[3], &r[3]);
                if let Some(bn) = bins {
                    let k = bt * nbin + bn[q] as usize;
                    ac.hn[k] += 1;
                    ac.hs[k] += s_full;
                    if let Some(fv) = fval {
                        ac.hf[k] += fv[q] as f64;
                    }
                }
              }
            }
            ac
        })
        .collect();
    let mut acc = init();
    for y in partiels {
        for i in 0..nb {
            acc.n[i] += y.n[i];
            acc.sf[i] += y.sf[i];
            acc.se[i] += y.se[i];
            acc.sl[i] += y.sl[i];
            acc.a2[i] += y.a2[i];
            acc.b2[i] += y.b2[i];
        }
        for i in 0..acc.hn.len() {
            acc.hn[i] += y.hn[i];
            acc.hs[i] += y.hs[i];
            acc.hf[i] += y.hf[i];
        }
    }
    Reduction {
        bmax: p.bmax as usize,
        nbin,
        n_paires: acc.n,
        s_full: acc.sf,
        s_early: acc.se,
        s_late: acc.sl,
        a2: acc.a2,
        b2: acc.b2,
        hist_n: acc.hn,
        hist_s: acc.hs,
        hist_f: acc.hf,
    }
}
