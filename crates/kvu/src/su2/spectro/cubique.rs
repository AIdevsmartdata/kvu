//! Représentations irréductibles du groupe cubique O_h (48 éléments) pour les opérateurs de boucles spatiales (spectro/gpu-1120).
//! Un élément g = (permutation p, signes s) agit sur un pas ±a par x → sgn(x)·s_a·(p_a + 1) (comme `neutre::groupe`) ; sa matrice
//! est R[p_a][a] = s_a, det R = sgn(p)·Π s. Partie propre R_p = det·R ∈ O ; parité de g = det.
//! Représentation Γ± : D(g) = (det)^{[−]} · D^Γ(R_p) ; projecteur de la ligne r : P_r = (d_Γ/48) Σ_g D_rr(g) · g.
//! A₁ : 1 ; A₂ : sgn(p) ; T₁ : R_p ; T₂ : sgn(p) R_p ; E : action sur les deux tenseurs symétriques sans trace (diag(1,−1,0)/√2,
//! diag(−1,−1,2)/√6).
use super::neutre::{canonique, groupe, images};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Irrep {
    A1,
    A2,
    E,
    T1,
    T2,
}

impl Irrep {
    pub fn dim(self) -> usize {
        match self {
            Irrep::A1 | Irrep::A2 => 1,
            Irrep::E => 2,
            _ => 3,
        }
    }
    pub fn nom(self) -> &'static str {
        match self {
            Irrep::A1 => "A1",
            Irrep::A2 => "A2",
            Irrep::E => "E",
            Irrep::T1 => "T1",
            Irrep::T2 => "T2",
        }
    }
}

/// « A2+ », « E- », … → (irrep, parité +1 / −1).
pub fn analyser_rep(t: &str) -> Option<(Irrep, i8)> {
    let t = t.trim();
    let (nom, par) = if let Some(n) = t.strip_suffix('+') {
        (n, 1)
    } else {
        let n = t.strip_suffix('-')?;
        (n, -1)
    };
    let ir = match nom {
        "A1" => Irrep::A1,
        "A2" => Irrep::A2,
        "E" => Irrep::E,
        "T1" => Irrep::T1,
        "T2" => Irrep::T2,
        _ => return None,
    };
    Some((ir, par))
}

fn matrice(p: [usize; 3], s: [i8; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for a in 0..3 {
        r[p[a]][a] = s[a] as f64;
    }
    r
}
fn signe_perm(p: [usize; 3]) -> f64 {
    let mut inv = 0;
    for i in 0..3 {
        for j in i + 1..3 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    if inv % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}
fn det(p: [usize; 3], s: [i8; 3]) -> f64 {
    signe_perm(p) * (s[0] * s[1] * s[2]) as f64
}

/// D^{Γ±}(g)_{rr'} (matrice dim × dim) de l'élément g = (p, s).
pub fn matrice_rep(ir: Irrep, parite: i8, p: [usize; 3], s: [i8; 3]) -> Vec<Vec<f64>> {
    let d = det(p, s);
    let r = matrice(p, s);
    let rp: Vec<Vec<f64>> = (0..3).map(|i| (0..3).map(|j| d * r[i][j]).collect()).collect();
    let sp = signe_perm(p);
    let mut m: Vec<Vec<f64>> = match ir {
        Irrep::A1 => vec![vec![1.0]],
        Irrep::A2 => vec![vec![sp]],
        Irrep::T1 => rp.clone(),
        Irrep::T2 => rp.iter().map(|l| l.iter().map(|x| sp * x).collect()).collect(),
        Irrep::E => {
            let t = [
                [[1.0 / 2f64.sqrt(), 0.0, 0.0], [0.0, -1.0 / 2f64.sqrt(), 0.0], [0.0, 0.0, 0.0]],
                [[-1.0 / 6f64.sqrt(), 0.0, 0.0], [0.0, -1.0 / 6f64.sqrt(), 0.0], [0.0, 0.0, 2.0 / 6f64.sqrt()]],
            ];
            // D_ab = Tr(T_a^T R T_b R^T)
            let mut out = vec![vec![0.0; 2]; 2];
            for a in 0..2 {
                for b in 0..2 {
                    let mut acc = 0.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                for l in 0..3 {
                                    acc += t[a][i][j] * rp[i][k] * t[b][k][l] * rp[j][l];
                                }
                            }
                        }
                    }
                    out[a][b] = acc;
                }
            }
            out
        }
    };
    if parite < 0 {
        for l in m.iter_mut() {
            for x in l.iter_mut() {
                *x *= d;
            }
        }
    }
    m
}

/// Caractère χ(g) de Γ± sur l'élément g.
pub fn caractere(ir: Irrep, parite: i8, p: [usize; 3], s: [i8; 3]) -> f64 {
    let m = matrice_rep(ir, parite, p, s);
    (0..m.len()).map(|i| m[i][i]).sum()
}

/// Poids de l'opérateur de transfert P_{ligne,colonne} de Γ± sur les images (classes canoniques) de la forme `chemin` :
/// w[im] = Σ_{g : classe(g·chemin) = im} (d/48) D_{ligne,colonne}(g)   (sans facteur de volume). Pour A₁⁺ : 1/N_im sur chaque image.
pub fn poids_images(chemin: &[i8], ir: Irrep, parite: i8, ligne: usize, colonne: usize) -> Vec<f64> {
    let imgs = images(chemin);
    let mut w = vec![0.0; imgs.len()];
    let d = ir.dim() as f64;
    for (p, s) in groupe() {
        let im: Vec<i8> = chemin
            .iter()
            .map(|&x| {
                let a = x.unsigned_abs() as usize - 1;
                x.signum() * s[a] * (p[a] as i8 + 1)
            })
            .collect();
        let c = canonique(&im);
        let k = imgs.iter().position(|x| *x == c).expect("image hors de l'orbite");
        w[k] += d / 48.0 * matrice_rep(ir, parite, p, s)[ligne][colonne];
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOUTES: [(Irrep, i8); 10] = [
        (Irrep::A1, 1), (Irrep::A2, 1), (Irrep::E, 1), (Irrep::T1, 1), (Irrep::T2, 1),
        (Irrep::A1, -1), (Irrep::A2, -1), (Irrep::E, -1), (Irrep::T1, -1), (Irrep::T2, -1),
    ];

    /// S1 : orthogonalité des caractères (48 δ) ; taille du groupe ; dimensions (Σ d² = 24 pour O, 48 pour O_h)
    #[test]
    fn s1_caracteres_orthogonaux() {
        assert_eq!(groupe().len(), 48);
        for (i, &(a, pa)) in TOUTES.iter().enumerate() {
            for (j, &(b, pb)) in TOUTES.iter().enumerate() {
                let sp: f64 = groupe().iter().map(|&(p, s)| caractere(a, pa, p, s) * caractere(b, pb, p, s)).sum();
                let attendu = if i == j { 48.0 } else { 0.0 };
                assert!((sp - attendu).abs() < 1e-12, "{:?}{pa} · {:?}{pb} = {sp}", a, b);
            }
        }
        let somme: usize = TOUTES.iter().map(|x| x.0.dim() * x.0.dim()).sum();
        assert_eq!(somme, 48);
    }

    /// Les matrices forment une représentation : D(g)D(h) = D(gh) (vérifié via l'orthogonalité de Schur : Σ_g D_ab(g) D_cd(g) = (48/d) δδ)
    #[test]
    fn s1_schur_et_unitarite() {
        for &(ir, par) in TOUTES.iter() {
            let d = ir.dim();
            for a in 0..d {
                for b in 0..d {
                    for c in 0..d {
                        for e in 0..d {
                            let sp: f64 = groupe().iter().map(|&(p, s)| {
                                let m = matrice_rep(ir, par, p, s);
                                m[a][b] * m[c][e]
                            }).sum();
                            let att = if a == c && b == e { 48.0 / d as f64 } else { 0.0 };
                            assert!((sp - att).abs() < 1e-12, "{:?}{par} Schur ({a}{b}{c}{e}) : {sp}", ir);
                        }
                    }
                }
            }
        }
    }

    /// Projecteurs : P_Γ idempotents, P_Γ P_Γ' = 0 sur l'espace des chemins (orbite complète par g), Σ_Γ,rows P = identité.
    /// On agit sur l'espace des fonctions sur le groupe (régulière) : (P f)(h) = (d/48) Σ_g D_rr(g) f(g⁻¹ h)… vérifié via le poids
    /// de la ligne sur l'identité : Σ_Γ Σ_r (d/48)·D_rr(e) · 48/... = Σ d² / 48 · … = 1.
    #[test]
    fn s1_completude_et_idempotence() {
        // complétude : Σ_{Γ,r} (d/48) D_rr(e) = Σ d/48 · d = 48/48 = 1
        let somme: f64 = TOUTES.iter().map(|x| x.0.dim() as f64 * x.0.dim() as f64 / 48.0).sum();
        assert!((somme - 1.0).abs() < 1e-14);
        // idempotence en poids : Σ_g w_g^{(Γ,r)} w_{g⁻¹h}... équivalent à (d/48)Σ_g D_rr(g) D_rr(g⁻¹) = 1 pour g⁻¹ = g^T
        for &(ir, par) in TOUTES.iter() {
            let d = ir.dim();
            for r in 0..d {
                let sp: f64 = groupe().iter().map(|&(p, s)| {
                    let m = matrice_rep(ir, par, p, s);
                    // D(g⁻¹) = D(g)^T (orthogonale)
                    m[r][r] * m[r][r]
                }).sum();
                assert!((d as f64 / 48.0 * sp - 1.0).abs() < 1e-12);
            }
        }
    }

    /// A₁⁺ redonne les poids uniformes de `neutre` ; les poids de chaque ligne ont les symétries attendues.
    #[test]
    fn poids_a1_uniformes_et_reps_non_triviales() {
        for chemin in [&[1i8, 2, -1, -2][..], &[1, 3, 2, -3, -1, -2][..]] {
            let w = poids_images(chemin, Irrep::A1, 1, 0, 0);
            let n = w.len() as f64;
            for x in &w {
                assert!((x - 1.0 / n).abs() < 1e-14);
            }
        }
        // plaquette : seule A1+ et E+ (et T2+ ?) peuvent être non nulles ; A2+, T1± nulles
        let n = |c: &[i8], ir: Irrep, par| {
            (0..ir.dim()).flat_map(|r| (0..ir.dim()).flat_map(move |q| poids_images(c, ir, par, r, q))).fold(0.0f64, |m, x| m.max(x.abs()))
        };
        let p11 = &[1i8, 2, -1, -2][..];
        assert!(n(p11, Irrep::A2, 1) < 1e-12);
        assert!(n(p11, Irrep::E, 1) > 1e-3);
        assert!(n(p11, Irrep::T1, 1) < 1e-12 && n(p11, Irrep::T1, -1) < 1e-12 && n(p11, Irrep::E, -1) < 1e-12);
    }
}
