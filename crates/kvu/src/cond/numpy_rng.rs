//! Générateur compatible numpy : `np.random.default_rng(seed)` (SeedSequence → PCG64),
//! `random()` (53 bits), `standard_normal()` (ziggurat de numpy, tables BSD-3 de numpy citées dans
//! [`super::numpy_tables`]) et le tirage vectorisé `_hb_numpy` de cb.py rejoué À L'IDENTIQUE (mêmes appels,
//! même ordre). Propriétaire : T5.
//!
//! # Ce qui est exact à l'octet, et ce qui ne l'est pas
//! - **Exact à l'octet** : SeedSequence, PCG64 (XSL-RR 128/64), `random()`, `standard_normal()` (les
//!   transcendantes de la queue sont `log1p` et `exp` scalaires de la libm, comme numpy), le NOMBRE de
//!   tirages consommés par `hb_numpy` et donc l'état final du générateur (preuve EQC-1b), ainsi que la
//!   direction `v` (normales divisées par la norme `((v0² + v1²) + v2²)`).
//! - **À l'ULP près** : `x0` (donc `r`). cb.py calcule `np.log1p` et `np.expm1` sur des TABLEAUX ; numpy
//!   les envoie alors vers SVML quand le CPU a AVX-512 (selon la machine), dont
//!   les résultats diffèrent de ceux de la libm scalaire dans ~8 % des cas, d'au plus quelques ULP. cb.py
//!   n'est donc pas lui-même reproductible à l'octet d'une machine à l'autre ; ici `x0` est calculé avec
//!   la libm (`ln_1p`, `exp_m1`), et l'écart à cb.py est de quelques ULP (voir tests/cond_numpy.rs).
//!   Une décision d'acceptation ne bascule qu'avec une probabilité de l'ordre de 1e-16 par tirage.

use super::numpy_tables::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};
use crate::commun::{Erreur, Resultat};
use rayon::prelude::*;

/// En dessous de cette taille de lot, le calcul des transcendantes reste séquentiel (le coût de la répartition dominerait).
const SEUIL_PARALLELE: usize = 4096;

/// Multiplicateur du LCG 128 bits de PCG (`PCG_DEFAULT_MULTIPLIER_128`).
const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

// Constantes de SeedSequence (numpy/random/bit_generator.pyx).
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const TAILLE_POOL: usize = 4;

/// Sérialisation décimale des entiers 128 bits (comme `bit_generator.state` de numpy).
mod u128_texte {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &u128, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        let t = String::deserialize(d)?;
        t.parse::<u128>().map_err(serde::de::Error::custom)
    }
}

/// État PCG64 (128 bits d'état, 128 bits d'incrément), sérialisable (reprise à l'octet).
/// Identique au dictionnaire `rng.bit_generator.state['state']` de numpy (`state`, `inc`).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GenerateurNumpy {
    #[serde(with = "u128_texte")]
    pub etat: u128,
    #[serde(with = "u128_texte")]
    pub inc: u128,
}

#[inline]
fn hashmix(valeur: u32, constante: &mut u32) -> u32 {
    let mut v = valeur ^ *constante;
    *constante = constante.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*constante);
    v ^ (v >> XSHIFT)
}

#[inline]
fn melange(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// Entropie d'un entier numpy : mots de 32 bits, poids faible d'abord (0 ⇒ un mot nul).
fn mots_entropie(graine: u64) -> Vec<u32> {
    let mut mots = Vec::with_capacity(2);
    let mut g = graine;
    while g > 0 {
        mots.push((g & 0xFFFF_FFFF) as u32);
        g >>= 32;
    }
    if mots.is_empty() {
        mots.push(0);
    }
    mots
}

/// Le pool de 4 mots de `SeedSequence(graine)` (sans clé d'engendrement).
pub fn pool_seed_sequence(graine: u64) -> [u32; TAILLE_POOL] {
    let entropie = mots_entropie(graine);
    let mut pool = [0u32; TAILLE_POOL];
    let mut hc = INIT_A;
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropie.get(i).copied().unwrap_or(0), &mut hc);
    }
    for src in 0..TAILLE_POOL {
        for dst in 0..TAILLE_POOL {
            if src != dst {
                let h = hashmix(pool[src], &mut hc);
                pool[dst] = melange(pool[dst], h);
            }
        }
    }
    pool
}

/// `SeedSequence(graine).generate_state(n_mots, np.uint64)`.
pub fn generate_state_u64(graine: u64, n_mots: usize) -> Vec<u64> {
    let pool = pool_seed_sequence(graine);
    let n32 = 2 * n_mots;
    let mut hc = INIT_B;
    let mut etat32 = Vec::with_capacity(n32);
    for i in 0..n32 {
        let mut d = pool[i % TAILLE_POOL] ^ hc;
        hc = hc.wrapping_mul(MULT_B);
        d = d.wrapping_mul(hc);
        d ^= d >> XSHIFT;
        etat32.push(d);
    }
    // vue uint64 little-endian : mot de poids faible d'abord
    (0..n_mots).map(|i| (etat32[2 * i] as u64) | ((etat32[2 * i + 1] as u64) << 32)).collect()
}

/// Maximum qui propage les NaN (`np.maximum`), contrairement à `f64::max`.
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// Saute `delta` pas du LCG de PCG : `état ← mult^delta · état + inc · (mult^delta − 1)/(mult − 1)` (mod 2¹²⁸),
/// en O(log delta) (algorithme `pcg_advance_lcg_128` de PCG).
fn avancer_lcg(etat: u128, mut delta: u128, mult: u128, plus: u128) -> u128 {
    let (mut acc_mult, mut acc_plus) = (1u128, 0u128);
    let (mut cur_mult, mut cur_plus) = (mult, plus);
    while delta > 0 {
        if delta & 1 == 1 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    acc_mult.wrapping_mul(etat).wrapping_add(acc_plus)
}

/// Sortie XSL-RR 128/64 d'un état déjà avancé.
#[inline]
fn sortie_xsl_rr(s: u128) -> u64 {
    (((s >> 64) as u64) ^ (s as u64)).rotate_right((s >> 122) as u32)
}

/// Taille d'un bloc de tirages générés en parallèle par saut de PCG.
const BLOC_FLUX: usize = 1 << 15;

/// Source de tirages d'un balayage de ziggurat par blocs : un tampon de sorties u64 précalculées (en parallèle par
/// sauts), consommé séquentiellement ; il se prolonge à la demande (la consommation du ziggurat est variable).
struct FluxU64 {
    depart: GenerateurNumpy,
    tampon: Vec<u64>,
    pos: usize,
}

impl FluxU64 {
    fn nouveau(depart: &GenerateurNumpy, prevu: usize) -> FluxU64 {
        FluxU64 { depart: depart.clone(), tampon: depart.flux_u64(prevu), pos: 0 }
    }

    /// Prolonge le tampon d'un bloc de sorties supplémentaires.
    fn prolonger(&mut self) {
        let mut g = self.depart.clone();
        g.avancer(self.tampon.len() as u128);
        let suite = g.flux_u64(BLOC_FLUX);
        self.tampon.extend_from_slice(&suite);
    }

    #[inline]
    fn suivant(&mut self) -> u64 {
        if self.pos == self.tampon.len() {
            self.prolonger();
        }
        let r = self.tampon[self.pos];
        self.pos += 1;
        r
    }
}

/// `random()` à partir d'un tirage u64.
#[inline]
fn double_de(r: u64) -> f64 {
    (r >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// Ziggurat de numpy (`random_standard_normal`) sur une source de u64 : mêmes décisions, même consommation.
#[inline]
fn normale_depuis(mut suivant: impl FnMut() -> u64) -> f64 {
    loop {
        let mut r = suivant();
        let idx = (r & 0xff) as usize;
        r >>= 8;
        let signe = r & 0x1;
        let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
        let mut x = rabs as f64 * WI_DOUBLE[idx];
        if signe & 0x1 == 1 {
            x = -x;
        }
        if rabs < KI_DOUBLE[idx] {
            return x; // 99,3 % des cas
        }
        if idx == 0 {
            loop {
                // « 1 − U » pour éviter log(0), cf. GH 13361 de numpy
                let xx = -ZIGGURAT_NOR_INV_R * (-double_de(suivant())).ln_1p();
                let yy = -(-double_de(suivant())).ln_1p();
                if yy + yy > xx * xx {
                    return if (rabs >> 8) & 0x1 == 1 { -(ZIGGURAT_NOR_R + xx) } else { ZIGGURAT_NOR_R + xx };
                }
            }
        } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * double_de(suivant()) + FI_DOUBLE[idx] < (-0.5 * x * x).exp() {
            return x;
        }
    }
}

impl GenerateurNumpy {
    /// Saute `delta` tirages (`rng.bit_generator.advance(delta)` de numpy) sans les produire.
    pub fn avancer(&mut self, delta: u128) {
        self.etat = avancer_lcg(self.etat, delta, PCG_MULT, self.inc);
    }

    /// Les `n` prochaines sorties u64, SANS modifier le générateur, produites en parallèle par blocs (chaque bloc
    /// saute à sa position) : identiques à `n` appels de `prochain_u64`.
    pub fn flux_u64(&self, n: usize) -> Vec<u64> {
        let mut out = vec![0u64; n];
        let (etat, inc) = (self.etat, self.inc);
        out.par_chunks_mut(BLOC_FLUX).enumerate().for_each(|(k, bloc)| {
            let mut e = avancer_lcg(etat, (k * BLOC_FLUX) as u128, PCG_MULT, inc);
            for o in bloc.iter_mut() {
                e = e.wrapping_mul(PCG_MULT).wrapping_add(inc);
                *o = sortie_xsl_rr(e);
            }
        });
        out
    }

    /// = np.random.default_rng(graine) (SeedSequence(graine).generate_state(4, uint64) → PCG64).
    pub fn depuis_graine(graine: u64) -> GenerateurNumpy {
        let v = generate_state_u64(graine, 4);
        let initstate = ((v[0] as u128) << 64) | (v[1] as u128);
        let initseq = ((v[2] as u128) << 64) | (v[3] as u128);
        // pcg_setseq_128_srandom_r
        let mut g = GenerateurNumpy { etat: 0, inc: (initseq << 1) | 1 };
        g.pas();
        g.etat = g.etat.wrapping_add(initstate);
        g.pas();
        g
    }

    /// Reconstruit depuis l'état numpy (`state`, `inc`).
    pub fn depuis_etat(etat: u128, inc: u128) -> GenerateurNumpy {
        GenerateurNumpy { etat, inc }
    }

    #[inline]
    fn pas(&mut self) {
        self.etat = self.etat.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// Prochain entier 64 bits (`pcg64_random_r` : pas du LCG, puis sortie XSL-RR de l'état NEUF).
    #[inline]
    pub fn prochain_u64(&mut self) -> u64 {
        self.pas();
        let s = self.etat;
        let xorshift = ((s >> 64) as u64) ^ (s as u64);
        xorshift.rotate_right((s >> 122) as u32)
    }

    /// = rng.random() : (next_uint64 >> 11) · 2⁻⁵³.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.prochain_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// = rng.random(n). Au-delà de `SEUIL_PARALLELE`, les tirages sont produits en parallèle par sauts de PCG, en une
    /// seule passe (génération et conversion en flottant) : même résultat, même état final.
    pub fn random_n(&mut self, n: usize) -> Vec<f64> {
        if n < SEUIL_PARALLELE {
            return (0..n).map(|_| self.random()).collect();
        }
        let mut out = vec![0.0f64; n];
        let (etat, inc) = (self.etat, self.inc);
        out.par_chunks_mut(BLOC_FLUX).enumerate().for_each(|(k, bloc)| {
            let mut e = avancer_lcg(etat, (k * BLOC_FLUX) as u128, PCG_MULT, inc);
            for o in bloc.iter_mut() {
                e = e.wrapping_mul(PCG_MULT).wrapping_add(inc);
                *o = double_de(sortie_xsl_rr(e));
            }
        });
        self.avancer(n as u128);
        out
    }

    /// = rng.standard_normal() (ziggurat 256 tranches de numpy, `random_standard_normal`).
    pub fn standard_normal(&mut self) -> f64 {
        normale_depuis(|| self.prochain_u64())
    }

    /// = rng.normal(size=n) (loc = 0, scale = 1 : `0.0 + 1.0 * z`). Pour les grands `n`, les sorties u64 sont
    /// précalculées en parallèle ; seule la lecture du ziggurat (consommation variable) reste séquentielle, avec une
    /// boucle serrée pour le cas courant (99 %) et le ziggurat complet pour les autres.
    pub fn normal_n(&mut self, n: usize) -> Vec<f64> {
        if n < SEUIL_PARALLELE {
            return (0..n).map(|_| 0.0 + 1.0 * self.standard_normal()).collect();
        }
        let mut flux = FluxU64::nouveau(self, n + n / 16 + 64);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            // marge de 16 tirages : le cas rapide en consomme 1, un cas lent en consomme quelques-uns
            if flux.pos + 16 > flux.tampon.len() {
                flux.prolonger();
            }
            let r = flux.tampon[flux.pos];
            let idx = (r & 0xff) as usize;
            let rabs = ((r >> 8) >> 1) & 0x000f_ffff_ffff_ffff;
            if rabs < KI_DOUBLE[idx] {
                // cas rapide de random_standard_normal : x = ± rabs · wi[idx]
                let x = rabs as f64 * WI_DOUBLE[idx];
                out.push(0.0 + 1.0 * if (r >> 8) & 1 == 1 { -x } else { x });
                flux.pos += 1;
            } else {
                // cas lent (coin de tranche ou queue) : ziggurat complet, qui relit le même tirage
                out.push(0.0 + 1.0 * normale_depuis(|| flux.suivant()));
            }
        }
        self.avancer(flux.pos as u128);
        out
    }

    /// = cb._hb_numpy(alpha, rng) : (x0, r, v) pour chaque alpha, même boucle de rejet vectorisée
    /// (u pour tous les « todo », puis l'acceptation pour tous, puis les rejetés), puis
    /// rng.normal(size=(n,3)) normalisé.
    ///
    /// Les α doivent être finis et > 0 (la référence ne termine pas sinon : α = 0 donne x = NaN).
    #[allow(clippy::type_complexity)] // signature gelée : (x0, r, v)
    pub fn hb_numpy(&mut self, alpha: &[f64]) -> Resultat<(Vec<f64>, Vec<f64>, Vec<[f64; 3]>)> {
        if let Some(a) = alpha.iter().find(|a| !(a.is_finite() && **a > 0.0)) {
            return Err(Erreur::Config(format!(
                "hb_numpy : alpha = {a} (fini et > 0 requis ; la référence ne termine pas sinon)"
            )));
        }
        let n = alpha.len();
        // -np.expm1(-2 * a) ne dépend que de α : calculé une fois (valeurs identiques à celles que numpy recalcule
        // à chaque passe de la boucle de rejet)
        let calcul_em = |a: &f64| -((-2.0 * a).exp_m1());
        let em: Vec<f64> = if n >= SEUIL_PARALLELE {
            alpha.par_iter().map(calcul_em).collect()
        } else {
            alpha.iter().map(calcul_em).collect()
        };
        let mut x0 = vec![0.0f64; n];
        let mut todo: Vec<usize> = (0..n).collect();
        while !todo.is_empty() {
            let m = todo.len();
            // u = rng.random(len(todo)) ; x = 1.0 + np.log1p(-u * (-np.expm1(-2 * a))) / a
            // (les tirages sont séquentiels ou produits par sauts ; le calcul des transcendantes est parallèle)
            // (u puis w sont deux tirages consécutifs de m valeurs : un seul appel de 2m valeurs, même flux)
            let uw = self.random_n(2 * m);
            let (u, w) = uw.split_at(m);
            let calcul_x = |(&i, &ui): (&usize, &f64)| 1.0 + ((-ui) * em[i]).ln_1p() / alpha[i];
            let x: Vec<f64> = if m >= SEUIL_PARALLELE {
                todo.par_iter().zip(u.par_iter()).map(calcul_x).collect()
            } else {
                todo.iter().zip(u.iter()).map(calcul_x).collect()
            };
            // ok = rng.random(len(todo)) < np.sqrt(np.maximum(1 - x * x, 0))
            let accepte = |(&wj, &xj): (&f64, &f64)| wj < np_maximum(1.0 - xj * xj, 0.0).sqrt();
            let ok: Vec<bool> = if m >= SEUIL_PARALLELE {
                w.par_iter().zip(x.par_iter()).map(accepte).collect()
            } else {
                w.iter().zip(x.iter()).map(accepte).collect()
            };
            // x0[todo[ok]] = x[ok] ; todo = todo[~ok]   (filtrage ORDONNÉ, parallèle pour les grands lots)
            for j in 0..m {
                if ok[j] {
                    x0[todo[j]] = x[j];
                }
            }
            todo = if m >= SEUIL_PARALLELE {
                (0..m).into_par_iter().filter(|&j| !ok[j]).map(|j| todo[j]).collect()
            } else {
                (0..m).filter(|&j| !ok[j]).map(|j| todo[j]).collect()
            };
        }
        // r = np.sqrt(np.maximum(1 - x0 ** 2, 0)) ; v = rng.normal(size=(n, 3)) ; v /= norm
        let r: Vec<f64> = x0.iter().map(|&x| np_maximum(1.0 - x * x, 0.0).sqrt()).collect();
        let normales = self.normal_n(3 * n);
        let normaliser = |t: &[f64]| {
            let norme = ((t[0] * t[0] + t[1] * t[1]) + t[2] * t[2]).sqrt();
            [t[0] / norme, t[1] / norme, t[2] / norme]
        };
        let v: Vec<[f64; 3]> = if n >= SEUIL_PARALLELE {
            normales.par_chunks_exact(3).map(normaliser).collect()
        } else {
            normales.chunks_exact(3).map(normaliser).collect()
        };
        Ok((x0, r, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Valeurs de référence prises sur numpy 2.5.3 :
    //   np.random.default_rng(0).random()          = 0.6369616873214543
    //   np.random.default_rng(0).standard_normal() = 0.1257302210933933
    //   np.random.default_rng(1).random()          = 0.5118216247002567
    //   SeedSequence(0).generate_state(4, uint64)  = [15793235383387715774, 12390638538380655177,
    //                                                 2361836109651742017, 3188717715514472916]
    #[test]
    fn premiers_tirages_numpy() {
        assert_eq!(GenerateurNumpy::depuis_graine(0).random(), 0.6369616873214543);
        assert_eq!(GenerateurNumpy::depuis_graine(1).random(), 0.5118216247002567);
        assert_eq!(GenerateurNumpy::depuis_graine(0).standard_normal(), 0.1257302210933933);
        assert_eq!(
            generate_state_u64(0, 4),
            vec![15793235383387715774, 12390638538380655177, 2361836109651742017, 3188717715514472916]
        );
    }

    #[test]
    fn mots_d_entropie() {
        assert_eq!(mots_entropie(0), vec![0]);
        assert_eq!(mots_entropie(2709), vec![2709]);
        assert_eq!(mots_entropie((1u64 << 63) - 1), vec![0xFFFF_FFFF, 0x7FFF_FFFF]);
        assert_eq!(mots_entropie(1u64 << 32), vec![0, 1]);
    }

    #[test]
    fn serialisation_decimale() {
        let g = GenerateurNumpy::depuis_graine(2709);
        let j = serde_json::to_string(&g).unwrap();
        assert!(j.contains(&g.etat.to_string()));
        let h: GenerateurNumpy = serde_json::from_str(&j).unwrap();
        assert_eq!(g, h);
    }

    #[test]
    fn saut_de_pcg_equivaut_aux_pas() {
        for graine in [0u64, 1, 2709] {
            for delta in [0u128, 1, 2, 3, 100, 12_345, 1 << 20] {
                let mut a = GenerateurNumpy::depuis_graine(graine);
                let mut b = a.clone();
                for _ in 0..delta {
                    a.prochain_u64();
                }
                b.avancer(delta);
                assert_eq!(a, b, "graine {graine}, delta {delta}");
            }
        }
    }

    #[test]
    fn flux_parallele_identique_aux_appels_sequentiels() {
        let g = GenerateurNumpy::depuis_graine(5);
        let n = 3 * BLOC_FLUX + 17; // plusieurs blocs et un reste
        let flux = g.flux_u64(n);
        let mut s = g.clone();
        let seq: Vec<u64> = (0..n).map(|_| s.prochain_u64()).collect();
        assert_eq!(flux, seq);
        assert_eq!(g, GenerateurNumpy::depuis_graine(5), "flux_u64 ne modifie pas le générateur");
    }

    #[test]
    fn random_n_parallele_identique_au_sequentiel_et_meme_etat() {
        for n in [SEUIL_PARALLELE - 1, SEUIL_PARALLELE, 100_000] {
            let mut a = GenerateurNumpy::depuis_graine(11);
            let mut b = a.clone();
            let x = a.random_n(n);
            let y: Vec<f64> = (0..n).map(|_| b.random()).collect();
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a, b, "état final, n = {n}");
        }
    }

    #[test]
    fn normal_n_parallele_identique_au_sequentiel_et_meme_etat() {
        // 300 000 normales : la queue (idx = 0) et les coins de tranche (≈ 1 %) sont exercés
        for n in [SEUIL_PARALLELE - 1, SEUIL_PARALLELE, 300_000] {
            let mut a = GenerateurNumpy::depuis_graine(13);
            let mut b = a.clone();
            let x = a.normal_n(n);
            let y: Vec<f64> = (0..n).map(|_| b.standard_normal()).collect();
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a, b, "état final, n = {n}");
        }
    }

    #[test]
    fn tampon_de_ziggurat_se_prolonge_a_la_demande() {
        // tampon volontairement trop court : la consommation dépasse sa taille, le flux se prolonge sans erreur
        let g = GenerateurNumpy::depuis_graine(17);
        let mut flux = FluxU64::nouveau(&g, 10);
        let x: Vec<f64> = (0..100_000).map(|_| normale_depuis(|| flux.suivant())).collect();
        let mut s = g.clone();
        let y: Vec<f64> = (0..100_000).map(|_| s.standard_normal()).collect();
        assert_eq!(
            x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut apres = g.clone();
        apres.avancer(flux.pos as u128);
        assert_eq!(apres, s);
    }

    #[test]
    fn hb_numpy_refuse_alpha_nul() {
        let mut g = GenerateurNumpy::depuis_graine(0);
        assert!(g.hb_numpy(&[1.0, 0.0]).is_err());
        assert!(g.hb_numpy(&[f64::NAN]).is_err());
        assert!(g.hb_numpy(&[-1.0]).is_err());
    }

    #[test]
    fn hb_numpy_borne_et_direction_unitaire() {
        let mut g = GenerateurNumpy::depuis_graine(5);
        let alpha: Vec<f64> = (0..500).map(|i| 0.01 + 0.4 * i as f64).collect();
        let (x0, r, v) = g.hb_numpy(&alpha).unwrap();
        for i in 0..alpha.len() {
            assert!((-1.0..=1.0).contains(&x0[i]));
            assert!((x0[i] * x0[i] + r[i] * r[i] - 1.0).abs() < 1e-14);
            let n2: f64 = v[i].iter().map(|c| c * c).sum();
            assert!((n2 - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn hb_numpy_deterministe_et_depend_de_la_graine() {
        let alpha = vec![0.5, 2.0, 30.0, 200.0];
        let a = GenerateurNumpy::depuis_graine(3).hb_numpy(&alpha).unwrap();
        let b = GenerateurNumpy::depuis_graine(3).hb_numpy(&alpha).unwrap();
        let c = GenerateurNumpy::depuis_graine(4).hb_numpy(&alpha).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
