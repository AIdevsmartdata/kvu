//! Rapports de Bessel de l'équation de Schwinger–Dyson à un lien : I₂(x)/I₁(x) (SU(2)) et I₁(x)/I₀(x) (U(1), MUTANT).
//!
//! Évaluation exacte par la fraction continue de la récurrence (I_{ν−1} = I_{ν+1} + (2ν/x) I_ν) :
//! ρ_ν = I_{ν+1}/I_ν vérifie ρ_{ν−1} = 1/(2ν/x + ρ_ν) ; on part de ρ_N = 0 à profondeur N assez grande (récurrence
//! descendante, stable) et on remonte. La profondeur croît avec x ; la précision est contrôlée par le test contre
//! `mpmath` (60 chiffres) dans `tests/rg_flot_bessel.rs` (≤ 1e-14 relatif).

use super::sd::RapportBessel;

/// Profondeur de la fraction continue pour l'argument x ≥ 0.
fn profondeur(x: f64) -> usize {
    // L'erreur de troncature décroît comme exp(−2 N / √(x² + N²) · …) : N = 40 + 2x + √x·6 est très large (testé jusqu'à x = 1e4).
    40 + (2.0 * x) as usize + (6.0 * x.sqrt()) as usize
}

/// ρ_ν(x) = I_{ν+1}(x)/I_ν(x) pour ν entier ≥ 0, x > 0.
fn rho(nu: usize, x: f64) -> f64 {
    let n_max = profondeur(x) + nu;
    let mut r = 0.0_f64;
    // r = ρ_k pour k = n_max, n_max−1, …, nu
    let mut k = n_max;
    while k > nu {
        // ρ_{k−1} = 1/(2k/x + ρ_k)
        r = 1.0 / (2.0 * k as f64 / x + r);
        k -= 1;
    }
    r
}

/// I₂(x)/I₁(x) (x ≥ 0 ; 0 en 0).
pub fn i2_sur_i1(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        rho(1, x)
    }
}

/// I₁(x)/I₀(x) (x ≥ 0 ; 0 en 0).
pub fn i1_sur_i0(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        rho(0, x)
    }
}

/// Rapport choisi.
pub fn rapport(r: RapportBessel, x: f64) -> f64 {
    match r {
        RapportBessel::I2SurI1 => i2_sur_i1(x),
        RapportBessel::I1SurI0 => i1_sur_i0(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valeurs_connues() {
        // I2(1)/I1(1) = 0.1357476697670383.../0.5651591039924851... (mpmath) ; limite x → 0 : x/4 ; limite x → ∞ : 1 − 3/(2x)
        assert!((i2_sur_i1(1.0) - 0.240_2).abs() < 1e-3);
        assert!((i2_sur_i1(1e-6) - 2.5e-7).abs() < 1e-18);
        assert!((i2_sur_i1(1e4) - (1.0 - 1.5e-4)).abs() < 1e-7);
        assert_eq!(i2_sur_i1(0.0), 0.0);
        assert!((i1_sur_i0(1e-6) - 5e-7).abs() < 1e-18);
    }
}
