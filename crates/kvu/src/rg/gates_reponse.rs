//! Gates de l'instrument WP22-B : protocoles et seuils de l'avenant gelé v5.WP22 §B.
use super::bloc::{coordonnees, site};
use super::gates::{jauge_haar, transformer_jauge};
use super::lissage::Variante;
use super::operateur::Operateur;
use super::reponse::{self, ecart, liens_bloc, norme, Mesure, PAS_DEFAUT, U};
use super::support;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::groupes::gates::{LigneGate, Verdict};
use crate::stockage::npy::lire_champ;
use std::collections::BTreeMap;
use std::path::Path;

pub const SEUIL_LIN: f64 = 1e-5;
pub const SEUIL_LIN_MUT: f64 = 1e-3;
pub const SEUIL_SYM: f64 = 1e-9;
pub const SEUIL_JAUGE: f64 = 1e-5;
pub const PLANCHER_JAUGE: f64 = 1e-5;
pub const FRACTION_NON_NULLE: f64 = 0.9;
pub const RICH_MIN: f64 = 3.0;
pub const RICH_MAX: f64 = 5.0;
pub const RICH_FRACTION: f64 = 0.95;
pub const RICH_PUISSANCE: usize = 20;
pub const RICH_ARRONDI: f64 = 1e3;
pub const ECHANTILLON: usize = 16;
pub const GRAINE_JAUGE: u64 = 26_102_822_001;
pub const IDS: [&str; 7] = [
    "RG-QL-REP-ZERO",
    "RG-QL-REP-ZERO-MUT",
    "RG-QL-REP-LIN",
    "RG-QL-REP-LIN-MUT",
    "RG-QL-REP-SYM",
    "RG-QL-REP-JAUGE",
    "RG-QL-REP-RICH",
];
pub const ZERO_OPS: [&str; 3] = ["dec", "ape:0.5:2", "agr:1"];
pub const LIN_OPS: [&str; 4] = ["dec", "ape:0.5:2", "ape:0.5:6", "flot:0.05:0.01"];
pub const RICH_OPS: [&str; 2] = ["ape:0.5:6", "flot:0.5:0.01"];
const L: usize = 9;

fn froid(l: usize) -> ChampHote {
    let mut q = vec![0.0; 16 * l.pow(4)];
    for lien in q.chunks_exact_mut(4) {
        lien[0] = 1.0;
    }
    ChampHote { l, q }
}
fn op(s: &str) -> Resultat<Operateur> {
    Operateur::analyser(s)
}
fn mesurer(c: &ChampHote, op: &Operateur, liens: &[usize], fils: usize) -> Resultat<Vec<Mesure>> {
    reponse::table(
        c,
        op,
        Variante::Normale,
        liens,
        &[PAS_DEFAUT, PAS_DEFAUT / 2.0],
        fils,
    )
}
fn exiger(ok: bool, detail: String) -> Resultat<String> {
    if ok {
        Ok(detail)
    } else {
        Err(Erreur::Validation(detail))
    }
}
fn joindre(resultats: impl Iterator<Item = Resultat<String>>) -> Resultat<String> {
    let mut details = Vec::new();
    let mut ok = true;
    for r in resultats {
        match r {
            Ok(d) => details.push(d),
            Err(e) => {
                ok = false;
                details.push(stable(&e));
            }
        }
    }
    exiger(ok, details.join(" ; "))
}
fn stable(e: &Erreur) -> String {
    match e {
        Erreur::Validation(s) | Erreur::Config(s) => s.clone(),
        _ => "erreur interne de lecture ou de calcul".into(),
    }
}

/// Support déclaré calculé par l'API de support, sans déduction depuis la réponse.
fn zero(
    c: &ChampHote,
    operateur: &Operateur,
    variante: Variante,
    liens: &[usize],
    fils: usize,
    mutant: bool,
) -> Resultat<String> {
    let t = reponse::table(
        c,
        operateur,
        variante,
        liens,
        &[PAS_DEFAUT, PAS_DEFAUT / 2.0],
        fils,
    )?;
    let lc = c.l / 3;
    let vc = lc.pow(4);
    let n = 4 * vc;
    let supports: Vec<_> = (0..n)
        .map(|j| support::support(operateur, c.l, coordonnees(j % vc, lc), j / vc))
        .collect();
    let (mut hors, mut violations, mut dedans, mut non_nulles) = (0usize, 0usize, 0usize, 0usize);
    for (&li, m) in liens.iter().zip(t) {
        for (j, s) in supports.iter().enumerate() {
            if s.binary_search(&li).is_ok() {
                dedans += 1;
                non_nulles += usize::from(norme(&m.matrices[n + j]) > 0.0);
            } else {
                hors += 1;
                violations += usize::from(
                    !m.nulles_exactes[j]
                        || m.matrices[n + j]
                            .iter()
                            .flatten()
                            .any(|x| x.to_bits() != 0.0f64.to_bits()),
                );
            }
        }
    }
    let ok = if mutant {
        violations > 0
    } else {
        hors > 0
            && dedans > 0
            && violations == 0
            && non_nulles as f64 >= FRACTION_NON_NULLE * dedans as f64
    };
    exiger(ok, format!("{} : extérieur {hors}, violations {violations}, intérieur non nul {non_nulles}/{dedans}{}",operateur.spec(),if mutant {", mutant détecté si violations > 0"} else {""}))
}

fn deplacer(mut x: [usize; 4], mu: usize, d: isize, l: usize) -> [usize; 4] {
    x[mu] = (x[mu] as isize + d).rem_euclid(l as isize) as usize;
    x
}
/// Forme différentielle indépendante : laplacien de θμ moins le gradient de la divergence de θ.
fn a_lineaire(theta: &[f64], l: usize) -> Vec<f64> {
    let v = l.pow(4);
    let mut sortie = vec![0.0; 4 * v];
    for mu in 0..4 {
        for s in 0..v {
            let x = coordonnees(s, l);
            let lire = |d, y| theta[d * v + site(y, l)];
            let mut somme = 0.0;
            for nu in 0..4 {
                let xp = deplacer(x, nu, 1, l);
                let xm = deplacer(x, nu, -1, l);
                let xmu = deplacer(x, mu, 1, l);
                let xmum = deplacer(xmu, nu, -1, l);
                somme += lire(mu, xp) - 2.0 * lire(mu, x) + lire(mu, xm);
                somme -= lire(nu, xmu) - lire(nu, xmum) - lire(nu, x) + lire(nu, xm);
            }
            sortie[mu * v + s] = somme;
        }
    }
    sortie
}
fn reference_lineaire(l: usize, li: usize, operateur: &Operateur) -> Resultat<Vec<f64>> {
    let mut theta = vec![0.0; 4 * l.pow(4)];
    theta[li] = 1.0;
    match *operateur {
        Operateur::Dec => (),
        Operateur::Ape { alpha, n } => {
            for _ in 0..n {
                let a = a_lineaire(&theta, l);
                for (t, dt) in theta.iter_mut().zip(a) {
                    *t += alpha / 6.0 * dt;
                }
            }
        }
        Operateur::Flot { eps, n, .. } => {
            for _ in 0..n {
                let a = a_lineaire(&theta, l);
                let a2 = a_lineaire(&a, l);
                let a3 = a_lineaire(&a2, l);
                for i in 0..theta.len() {
                    theta[i] +=
                        eps * a[i] + eps * eps / 2.0 * a2[i] + eps * eps * eps / 6.0 * a3[i];
                }
            }
        }
        _ => {
            return Err(Erreur::Config(
                "référence linéaire : opérateur non prévu".into(),
            ))
        }
    }
    let lc = l / 3;
    let vc = lc.pow(4);
    Ok((0..4 * vc)
        .map(|j| {
            support::chemin_droit(l, coordonnees(j % vc, lc), j / vc)
                .iter()
                .map(|&i| theta[i])
                .sum()
        })
        .collect())
}
fn erreur_lineaire(
    c: &ChampHote,
    operateur: &Operateur,
    reference: &Operateur,
    liens: &[usize],
    t: &[Mesure],
) -> Resultat<f64> {
    let n = 4 * (c.l / 3).pow(4);
    let mut max = 0.0f64;
    for (&li, m) in liens.iter().zip(t) {
        let kappas = reference_lineaire(c.l, li, reference)?;
        for (j, &kappa) in kappas.iter().enumerate() {
            for b in 0..3 {
                for a in 0..3 {
                    let attendu = if a == b { kappa } else { 0.0 };
                    max = max.max((m.matrices[n + j][b][a] - attendu).abs());
                }
            }
        }
    }
    if !max.is_finite() {
        return Err(Erreur::Validation(format!(
            "{} : erreur linéaire non finie",
            operateur.spec()
        )));
    }
    Ok(max)
}

type Cle = (usize, usize, [usize; 4]);
fn cle(l: usize, li: usize, j: usize) -> Cle {
    let v = l.pow(4);
    let lc = l / 3;
    let vc = lc.pow(4);
    let x = coordonnees(li % v, l);
    let a = coordonnees(j % vc, lc).map(|z| 3 * z);
    (
        j / vc,
        li / v,
        std::array::from_fn(|d| (x[d] + l - a[d]) % l),
    )
}
/// Action affine sur une base de lien ; un axe renversé décale la base du lien direct d'une unité.
fn image(cle: Cle, permutation: [usize; 4], signes: [isize; 4], l: usize) -> Cle {
    let (mu, nu, x) = cle;
    let mut y = [0usize; 4];
    for d in 0..4 {
        let e = permutation[d];
        let translation = if d == mu && signes[d] < 0 { 3 } else { 0 };
        y[e] = (signes[d] * x[d] as isize + translation).rem_euclid(l as isize) as usize;
    }
    let np = permutation[nu];
    if signes[nu] < 0 {
        y[np] = (y[np] + l - 1) % l;
    }
    (mu, np, y)
}
fn symetrie(l: usize, liens: &[usize], t: &[Mesure]) -> Resultat<String> {
    let n = 4 * (l / 3).pow(4);
    let mut valeurs = BTreeMap::new();
    let mut doublons = 0;
    for (&li, m) in liens.iter().zip(t) {
        for j in 0..n {
            if valeurs
                .insert(cle(l, li, j), norme(&m.matrices[n + j]))
                .is_some()
            {
                doublons += 1;
            }
        }
    }
    let attendu = 16 * l.pow(4);
    if valeurs.len() != attendu || doublons != 0 {
        return Err(Erreur::Validation(format!(
            "couverture incomplète : {} clés, attendu {attendu}, doublons {doublons}",
            valeurs.len()
        )));
    }
    let mut max = 0.0f64;
    let mut comparaisons = 0usize;
    for (&k, &valeur) in &valeurs {
        let trans: Vec<_> = (0..4).filter(|&d| d != k.0).collect();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let reste = 3 - i - j;
                let mut p = [0, 1, 2, 3];
                p[trans[0]] = trans[i];
                p[trans[1]] = trans[j];
                p[trans[2]] = trans[reste];
                for masque in 0..16 {
                    let signes =
                        std::array::from_fn(|d| if masque & (1 << d) == 0 { 1 } else { -1 });
                    let im = image(k, p, signes, l);
                    let autre = valeurs
                        .get(&im)
                        .ok_or_else(|| Erreur::Validation(format!("clé image absente : {im:?}")))?;
                    max = max.max((valeur - autre).abs());
                    comparaisons += 1;
                }
            }
        }
    }
    exiger(
        max <= SEUIL_SYM,
        format!(
            "{} clés, {comparaisons} comparaisons, max |Δ‖J‖| {max:.12e}, seuil {SEUIL_SYM:.0e}",
            valeurs.len()
        ),
    )
}
fn jauge(c: &ChampHote, liens: &[usize], fils: usize) -> Resultat<String> {
    let operateur = op("ape:0.5:6")?;
    let t = mesurer(c, &operateur, liens, fils)?;
    let g = jauge_haar(c.l, GRAINE_JAUGE);
    let cg = transformer_jauge(c, &g);
    let tg = mesurer(&cg, &operateur, liens, fils)?;
    let n = 4 * (c.l / 3).pow(4);
    let mut ratio = 0.0f64;
    let mut max = 0.0f64;
    for (a, b) in t.iter().zip(tg) {
        for j in 0..n {
            let na = norme(&a.matrices[n + j]);
            let nb = norme(&b.matrices[n + j]);
            let d = (na - nb).abs();
            max = max.max(d);
            ratio = ratio.max(d / (SEUIL_JAUGE * na.max(PLANCHER_JAUGE)));
        }
    }
    exiger(
        ratio <= 1.0,
        format!(
            "max |Δ‖J‖| {max:.12e}, max écart/seuil {ratio:.12e}, {} paires",
            liens.len() * n
        ),
    )
}
fn richardson(
    c: &ChampHote,
    operateur: &Operateur,
    liens: &[usize],
    fils: usize,
) -> Resultat<String> {
    let h = PAS_DEFAUT;
    let t = reponse::table(
        c,
        operateur,
        Variante::Normale,
        liens,
        &[h, h / 2.0, h / 4.0],
        fils,
    )?;
    let n = 4 * (c.l / 3).pow(4);
    let seuil = RICH_ARRONDI * 8.0 * U / (h / 4.0);
    let (mut eligibles, mut bons) = (0usize, 0usize);
    let (mut min, mut max) = (f64::INFINITY, 0.0f64);
    for m in t {
        for j in 0..n {
            let d = ecart(&m.matrices[n + j], &m.matrices[2 * n + j]);
            if d >= seuil {
                let rapport = ecart(&m.matrices[j], &m.matrices[n + j]) / d;
                eligibles += 1;
                bons += usize::from((RICH_MIN..=RICH_MAX).contains(&rapport));
                min = min.min(rapport);
                max = max.max(rapport);
            }
        }
    }
    let puissance = eligibles >= RICH_PUISSANCE;
    exiger(
        puissance && bons as f64 >= RICH_FRACTION * eligibles as f64,
        format!(
            "{} : éligibles {eligibles}, rapports [3,5] {bons}, min {min:.12e}, max {max:.12e}{}",
            operateur.spec(),
            if puissance {
                ""
            } else {
                " ; puissance insuffisante"
            }
        ),
    )
}

fn ligne(id: &str, r: Resultat<String>) -> LigneGate {
    match r {
        Ok(d) => LigneGate::nouvelle(Verdict::Pass, id, d),
        Err(e) => LigneGate::nouvelle(Verdict::Fail, id, stable(&e)),
    }
}
/// La précondition A est contrôlée par le pilote avant cet appel (avenant gelé §B).
pub fn batterie(fixtures: &Path, fils: usize) -> Resultat<Vec<LigneGate>> {
    if fils == 0 {
        return Err(Erreur::Config("--fils doit être ≥ 1".into()));
    }
    let c = lire_champ(&fixtures.join("cfg_s7_q64.npy"))?;
    if c.l != L {
        return Err(Erreur::Config("fixture s7 : L = 9 requis".into()));
    }
    let liens = liens_bloc(L, 324)?;
    let petits = liens_bloc(L, ECHANTILLON)?;
    let mut lignes = Vec::new();
    lignes.push(ligne(
        IDS[0],
        joindre(
            ZERO_OPS
                .iter()
                .map(|s| zero(&c, &op(s)?, Variante::Normale, &liens, fils, false)),
        ),
    ));
    lignes.push(ligne(
        IDS[1],
        zero(
            &c,
            &op("ape:0.5:2")?,
            Variante::PasSupplementaire,
            &liens,
            fils,
            true,
        ),
    ));
    let froid = froid(L);
    let mut lineaires = BTreeMap::new();
    let mut resultats = Vec::new();
    for s in LIN_OPS {
        let o = op(s)?;
        let r = mesurer(&froid, &o, &liens, fils).and_then(|t| {
            let max = erreur_lineaire(&froid, &o, &o, &liens, &t)?;
            if s == "ape:0.5:6" {
                lineaires.insert(s, t);
            }
            exiger(
                max <= SEUIL_LIN,
                format!("{s} : max composante {max:.12e}, seuil {SEUIL_LIN:.0e}"),
            )
        });
        resultats.push(r);
    }
    lignes.push(ligne(IDS[2], joindre(resultats.into_iter())));
    if let Some(t) = lineaires.get("ape:0.5:6") {
        let o = op("ape:0.5:6")?;
        let r=erreur_lineaire(&froid,&o,&op("ape:0.5:5")?,&liens,t).and_then(|max|exiger(max>=SEUIL_LIN_MUT,format!("APE6 contre APE5 : max composante {max:.12e}, seuil mutant {SEUIL_LIN_MUT:.0e}")));
        lignes.push(ligne(IDS[3], r));
        lignes.push(ligne(IDS[4], symetrie(L, &liens, t)));
    } else {
        lignes.push(ligne(
            IDS[3],
            Err(Erreur::Validation("table APE6 indisponible".into())),
        ));
        lignes.push(ligne(
            IDS[4],
            Err(Erreur::Validation("table APE6 indisponible".into())),
        ));
    }
    lignes.push(ligne(IDS[5], jauge(&c, &petits, fils)));
    lignes.push(ligne(
        IDS[6],
        joindre(
            RICH_OPS
                .iter()
                .map(|s| richardson(&c, &op(s)?, &petits, fils)),
        ),
    ));
    Ok(lignes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_lineaire_annule_gradient_et_champ_constant() {
        let l = 3usize;
        let v = l.pow(4);
        let constant = vec![1.0; 4 * v];
        assert!(a_lineaire(&constant, l).iter().all(|&x| x == 0.0));
        let f: Vec<_> = (0..v).map(|s| ((s * 7) % 19) as f64).collect();
        let gradient: Vec<_> = (0..4 * v)
            .map(|li| {
                let x = coordonnees(li % v, l);
                f[site(deplacer(x, li / v, 1, l), l)] - f[li % v]
            })
            .collect();
        assert!(a_lineaire(&gradient, l).iter().all(|&x| x == 0.0));
    }
    #[test]
    fn sous_cas_lineaires_et_mutant_sur_un_lien() {
        let c = froid(9);
        let liens = [0];
        for s in LIN_OPS {
            let o = op(s).unwrap();
            let t = mesurer(&c, &o, &liens, 1).unwrap();
            let max = erreur_lineaire(&c, &o, &o, &liens, &t).unwrap();
            assert!(max <= SEUIL_LIN, "{s}: {max}");
            if s == "ape:0.5:6" {
                let mutant =
                    erreur_lineaire(&c, &o, &op("ape:0.5:5").unwrap(), &liens, &t).unwrap();
                assert!(mutant >= SEUIL_LIN_MUT, "mutant : {mutant}");
            }
        }
    }
    #[test]
    fn sous_cas_support_et_mutant_sur_un_lien() {
        let c = froid(9);
        for s in ZERO_OPS {
            zero(&c, &op(s).unwrap(), Variante::Normale, &[0], 1, false).unwrap();
        }
        zero(
            &c,
            &op("ape:0.5:2").unwrap(),
            Variante::PasSupplementaire,
            &[0],
            1,
            true,
        )
        .unwrap();
    }
    #[test]
    fn stabilisateur_inverse_et_couverture_des_cles() {
        let liens = liens_bloc(9, 324).unwrap();
        let mut cles = std::collections::BTreeSet::new();
        for li in liens {
            for j in 0..324 {
                assert!(cles.insert(cle(9, li, j)));
            }
        }
        assert_eq!(cles.len(), 16 * 9usize.pow(4));
        for mu in 0..4 {
            let mut signes = [1; 4];
            signes[mu] = -1;
            let k = (mu, mu, [2, 4, 8, 1]);
            assert_eq!(
                image(image(k, [0, 1, 2, 3], signes, 9), [0, 1, 2, 3], signes, 9),
                k
            );
        }
    }
    #[test]
    fn sous_cas_jauge_et_richardson_sur_un_lien() {
        let c = lire_champ(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/rg/fixtures/cfg_s7_q64.npy"),
        )
        .unwrap();
        jauge(&c, &[0], 1).unwrap();
        let h = PAS_DEFAUT;
        for s in RICH_OPS {
            let t = reponse::table(
                &c,
                &op(s).unwrap(),
                Variante::Normale,
                &[0],
                &[h, h / 2.0, h / 4.0],
                1,
            )
            .unwrap();
            let m = &t[0];
            let n = 324;
            let (mut eligibles, mut bons) = (0, 0);
            for j in 0..n {
                let den = ecart(&m.matrices[n + j], &m.matrices[2 * n + j]);
                if den >= RICH_ARRONDI * 8.0 * U / (h / 4.0) {
                    eligibles += 1;
                    let rapport = ecart(&m.matrices[j], &m.matrices[n + j]) / den;
                    bons += usize::from((RICH_MIN..=RICH_MAX).contains(&rapport));
                }
            }
            // Un seul lien ne garantit pas la puissance du lot canonique de seize liens.
            if eligibles < RICH_PUISSANCE {
                let erreur = richardson(&c, &op(s).unwrap(), &[0], 1).unwrap_err();
                assert!(stable(&erreur).contains("puissance insuffisante"));
            }
            if s == "ape:0.5:6" {
                assert!(eligibles > 0);
            }
            assert!(
                bons as f64 >= RICH_FRACTION * eligibles as f64,
                "{s} : {bons}/{eligibles}"
            );
        }
    }
    #[test]
    #[ignore = "banc de coût ciblé : dix liens, aucune nouvelle configuration"]
    fn cout_dix_liens() {
        let c = lire_champ(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("validation/rg/fixtures/cfg_s7_q64.npy")
                .as_path(),
        )
        .unwrap();
        let liens = liens_bloc(9, 10).unwrap();
        for s in [
            "dec",
            "ape:0.5:2",
            "ape:0.5:6",
            "agr:1",
            "flot:0.05:0.01",
            "flot:0.5:0.01",
        ] {
            let debut = std::time::Instant::now();
            let t = mesurer(&c, &op(s).unwrap(), &liens, 4).unwrap();
            assert_eq!(t.len(), 10);
            println!(
                "COUT {s} liens=10 perturbations=120 fils=4 secondes={:.6}",
                debut.elapsed().as_secs_f64()
            );
        }
    }
}
