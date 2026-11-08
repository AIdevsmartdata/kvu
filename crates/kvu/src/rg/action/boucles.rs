//! Boucles fondamentales de l'action effective : plaquette (P), rectangle 1×2 (R), chaise (C),
//! torsade (T), et leurs agrafes.
//!
//! L'énumération est faite par force brute (§3.2 de la spécification) : toutes les suites de
//! codes de pas de longueur 4 et 6, commençant par (μ, +1), fermées, sans rebroussement
//! cyclique. Les comptes par direction sont vérifiés (P: 6, R: 18, C: 72, T: 24).

use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qadd, qdag, qmul, Q};
use crate::rg::lissage::Voisinage;

/// Un pas de chemin : direction 0..3 et sens ±1.
pub type Pas = (u8, i8);

/// Les quatre types de boucles fondamentales, triés dans l'ordre canonique.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeBoucle {
    P,
    R,
    C,
    T,
}

impl TypeBoucle {
    pub const TOUS: [TypeBoucle; 4] = [TypeBoucle::P, TypeBoucle::R, TypeBoucle::C, TypeBoucle::T];

    pub fn nom(self) -> &'static str {
        match self {
            TypeBoucle::P => "P",
            TypeBoucle::R => "R",
            TypeBoucle::C => "C",
            TypeBoucle::T => "T",
        }
    }

    pub fn analyser(t: &str) -> Resultat<TypeBoucle> {
        match t {
            "P" => Ok(TypeBoucle::P),
            "R" => Ok(TypeBoucle::R),
            "C" => Ok(TypeBoucle::C),
            "T" => Ok(TypeBoucle::T),
            autre => Err(Erreur::Config(format!(
                "type de boucle « {autre} » inconnu (P|R|C|T)"
            ))),
        }
    }

    pub fn longueur(self) -> usize {
        match self {
            TypeBoucle::P => 4,
            TypeBoucle::R | TypeBoucle::C | TypeBoucle::T => 6,
        }
    }

    /// Nombre de boucles de ce type contenant un lien donné (par direction d'énumération).
    pub fn par_lien(self) -> usize {
        match self {
            TypeBoucle::P => 6,
            TypeBoucle::R => 18,
            TypeBoucle::C => 72,
            TypeBoucle::T => 24,
        }
    }

    /// Nombre de boucles de ce type par site (par_lien × 4 / longueur).
    pub fn par_site(self) -> usize {
        self.par_lien() * 4 / self.longueur()
    }

    /// Poids dans la norme R4 : par_lien × (longueur − 1).
    pub fn poids_r4(self) -> f64 {
        (self.par_lien() * (self.longueur() - 1)) as f64
    }
}

/// Famille de boucles : non vide, sans doublon, triée (P < R < C < T) et contenant P.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Famille(Vec<TypeBoucle>);

impl Famille {
    pub fn analyser(t: &str) -> Resultat<Famille> {
        let mut types = Vec::new();
        for m in t.split(',').map(str::trim).filter(|m| !m.is_empty()) {
            let ty = TypeBoucle::analyser(m)?;
            if types.contains(&ty) {
                return Err(Erreur::Config(format!(
                    "type {m} en double dans la famille"
                )));
            }
            types.push(ty);
        }
        if types.is_empty() {
            return Err(Erreur::Config("famille de boucles vide".into()));
        }
        if !types.contains(&TypeBoucle::P) {
            return Err(Erreur::Config(
                "la famille doit contenir la plaquette P".into(),
            ));
        }
        types.sort();
        Ok(Famille(types))
    }

    pub fn types(&self) -> &[TypeBoucle] {
        &self.0
    }

    pub fn n(&self) -> usize {
        self.0.len()
    }

    pub fn noms(&self) -> Vec<&'static str> {
        self.0.iter().map(|t| t.nom()).collect()
    }

    pub fn index(&self, t: TypeBoucle) -> Option<usize> {
        self.0.iter().position(|&x| x == t)
    }
}

/// Table des formes pour chaque direction et chaque type, dans l'ordre d'énumération.
pub struct TableFormes {
    /// formes[mu][type_idx] : liste des chemins (suites de pas) partant de la base du lien.
    formes: Vec<Vec<Vec<Vec<Pas>>>>,
}

fn pas_vers_code(pas: Pas) -> u8 {
    2 * pas.0 + if pas.1 == 1 { 0 } else { 1 }
}

fn code_vers_pas(code: u8) -> Pas {
    (code / 2, if code.is_multiple_of(2) { 1 } else { -1 })
}

/// Déplacement cumulé d'une suite de pas à partir de l'origine, sans modulo.
fn deplacement_total_exact(suite: &[Pas]) -> [isize; 4] {
    let mut de = [0isize; 4];
    for &(d, s) in suite {
        de[d as usize] += s as isize;
    }
    de
}

/// Vérifie qu'aucun pas n'est suivi (cycliquement) de son inverse.
fn sans_rebroussement_cyclique(suite: &[Pas]) -> bool {
    suite.iter().enumerate().all(|(i, &pas)| {
        let suivant = suite[(i + 1) % suite.len()];
        pas.0 != suivant.0 || pas.1 != -suivant.1
    })
}

/// Simule le chemin exactement (sans modulo) et retourne l'ensemble des sites visités.
fn sites_visites(suite: &[Pas]) -> Vec<[isize; 4]> {
    let mut pos = [0isize; 4];
    let mut sites = vec![pos];
    for &(d, s) in suite {
        pos[d as usize] += s as isize;
        sites.push(pos);
    }
    sites
}

/// Teste si les sites visités contiennent les 4 coins d'un carré unité dans un plan de deux
/// directions parmi `dirs` (les directions distinctes de la boucle).
fn contient_carre_unite(sites: &[[isize; 4]], dirs: &[usize]) -> bool {
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let (d1, d2) = (dirs[i], dirs[j]);
            for &base in sites {
                let mut coin1 = base;
                let mut coin2 = base;
                let mut coin3 = base;
                let mut coin4 = base;
                coin1[d1] += 1;
                coin2[d2] += 1;
                coin3[d1] += 1;
                coin3[d2] += 1;
                coin4[d1] = coin1[d1];
                coin4[d2] = coin2[d2];
                if sites.contains(&coin1)
                    && sites.contains(&coin2)
                    && sites.contains(&coin3)
                    && sites.contains(&coin4)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Énumère toutes les formes valides pour une direction μ, classées par type.
/// Formes d'une direction, par type (P, R, C, T).
type FormesParType = (Vec<Vec<Pas>>, Vec<Vec<Pas>>, Vec<Vec<Pas>>, Vec<Vec<Pas>>);

fn enumerer_direction(mu: usize) -> FormesParType {
    let mut p = Vec::new();
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut t = Vec::new();

    for longueur in [4usize, 6] {
        let nb_suites = 8usize.pow(longueur as u32);
        for code_entier in 0..nb_suites {
            let mut suite = Vec::with_capacity(longueur);
            let mut n = code_entier;
            for _ in 0..longueur {
                suite.push(code_vers_pas((n % 8) as u8));
                n /= 8;
            }
            suite.reverse(); // l'ordre des codes est lexicographique.

            if pas_vers_code(suite[0]) != 2 * mu as u8 {
                continue; // premier pas = (mu, +1)
            }
            if !sans_rebroussement_cyclique(&suite) {
                continue;
            }
            if deplacement_total_exact(&suite) != [0, 0, 0, 0] {
                continue; // fermée exactement sur le tore universel
            }
            if longueur == 4 {
                p.push(suite);
            } else {
                let mut dirs: Vec<usize> = suite.iter().map(|&(d, _)| d as usize).collect();
                dirs.sort_unstable();
                dirs.dedup();
                match dirs.len() {
                    2 => r.push(suite),
                    3 => {
                        let sites = sites_visites(&suite);
                        if contient_carre_unite(&sites, &dirs) {
                            c.push(suite);
                        } else {
                            t.push(suite);
                        }
                    }
                    _ => {} // impossible pour une suite fermée sans rebroussement de longueur ≤ 6
                }
            }
        }
    }
    (p, r, c, t)
}

impl TableFormes {
    /// Construit la table par énumération et vérifie les comptes par direction.
    pub fn construire() -> Resultat<TableFormes> {
        let mut formes = vec![vec![Vec::new(); 4]; 4]; // [mu][type_idx]
        for (mu, fm) in formes.iter_mut().enumerate() {
            let (p, r, c, t) = enumerer_direction(mu);
            if p.len() != TypeBoucle::P.par_lien()
                || r.len() != TypeBoucle::R.par_lien()
                || c.len() != TypeBoucle::C.par_lien()
                || t.len() != TypeBoucle::T.par_lien()
            {
                return Err(Erreur::Config(format!(
                    "énumération des boucles mu={mu} : comptes ({},{},{},{}) au lieu de ({},{},{},{})",
                    p.len(),
                    r.len(),
                    c.len(),
                    t.len(),
                    TypeBoucle::P.par_lien(),
                    TypeBoucle::R.par_lien(),
                    TypeBoucle::C.par_lien(),
                    TypeBoucle::T.par_lien()
                )));
            }
            fm[0] = p;
            fm[1] = r;
            fm[2] = c;
            fm[3] = t;
        }
        Ok(TableFormes { formes })
    }

    pub fn formes(&self, mu: usize, t: TypeBoucle) -> &[Vec<Pas>] {
        let idx = match t {
            TypeBoucle::P => 0,
            TypeBoucle::R => 1,
            TypeBoucle::C => 2,
            TypeBoucle::T => 3,
        };
        &self.formes[mu][idx]
    }
}

/// Variante des agrafes pour le contrôle négatif RG-ACTION-MUT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarianteAgrafes {
    Normale,
    ChaiseInversee,
}

/// Agrafes A_ℓ^γ pour chaque type de `famille`, lien (mu, s), avec la variante éventuelle.
pub(crate) fn agrafes_lien_variante(
    champ: &ChampHote,
    vois: &Voisinage,
    table: &TableFormes,
    famille: &Famille,
    mu: usize,
    s: usize,
    variante: VarianteAgrafes,
) -> Vec<Q> {
    let x_plus = vois.haut(mu, s);
    famille
        .types()
        .iter()
        .map(|&ty| {
            let mut somme = [0.0; 4];
            for forme in table.formes(mu, ty) {
                let prod = produit_chemin(champ, vois, x_plus, &forme[1..]);
                somme = qadd(somme, prod);
            }
            if matches!(variante, VarianteAgrafes::ChaiseInversee) && ty == TypeBoucle::C {
                qdag(somme)
            } else {
                somme
            }
        })
        .collect()
}

/// Agrafes normales.
pub(crate) fn agrafes_lien(
    champ: &ChampHote,
    vois: &Voisinage,
    table: &TableFormes,
    famille: &Famille,
    mu: usize,
    s: usize,
) -> Vec<Q> {
    agrafes_lien_variante(champ, vois, table, famille, mu, s, VarianteAgrafes::Normale)
}

/// ½ Tr du produit d'un chemin (fermé ou non) depuis le site `s`.
///
/// Convention §3.1 : pas (d, +1) : multiplier à droite par U_d(pos) puis pos ← pos + d̂ ;
/// pas (d, −1) : pos ← pos − d̂ puis multiplier à droite par U_d(pos)†.
pub(crate) fn produit_chemin(champ: &ChampHote, vois: &Voisinage, s: usize, chemin: &[Pas]) -> Q {
    let v = champ.v();
    let mut pos = s;
    let mut acc = [1.0, 0.0, 0.0, 0.0];
    for &(d, sig) in chemin {
        if sig == 1 {
            let li = d as usize * v + pos;
            let u = champ.lien(li);
            acc = qmul(acc, u);
            pos = vois.haut(d as usize, pos);
        } else {
            pos = vois.bas(d as usize, pos);
            let li = d as usize * v + pos;
            let u = champ.lien(li);
            acc = qmul(acc, qdag(u));
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb::CpuSu2;
    use std::collections::HashSet;

    fn champ_haar(l: usize, graine: u64) -> ChampHote {
        let mut gen = CpuSu2::new(l, graine).unwrap();
        gen.init(true);
        ChampHote { l, q: gen.flat() }
    }

    #[test]
    fn comptes_par_site() {
        assert_eq!(TypeBoucle::P.par_site(), 6);
        assert_eq!(TypeBoucle::R.par_site(), 12);
        assert_eq!(TypeBoucle::C.par_site(), 48);
        assert_eq!(TypeBoucle::T.par_site(), 16);
    }

    #[test]
    fn famille_analyse() {
        let f = Famille::analyser("P,R,C,T").unwrap();
        assert_eq!(f.n(), 4);
        assert!(Famille::analyser("P,R,P").is_err());
        assert!(Famille::analyser("R,C,T").is_err());
        assert!(Famille::analyser("").is_err());
        assert!(Famille::analyser("P,X").is_err());
    }

    #[test]
    fn table_construite_comptes() {
        let table = TableFormes::construire().unwrap();
        for mu in 0..4 {
            assert_eq!(table.formes(mu, TypeBoucle::P).len(), 6);
            assert_eq!(table.formes(mu, TypeBoucle::R).len(), 18);
            assert_eq!(table.formes(mu, TypeBoucle::C).len(), 72);
            assert_eq!(table.formes(mu, TypeBoucle::T).len(), 24);
        }
    }

    #[test]
    fn exemples_specification() {
        let table = TableFormes::construire().unwrap();
        let exemples = [
            (vec![(0u8, 1i8), (1, 1), (0, -1), (1, -1)], TypeBoucle::P),
            (
                vec![(0, 1), (0, 1), (1, 1), (0, -1), (0, -1), (1, -1)],
                TypeBoucle::R,
            ),
            (
                vec![(0, 1), (1, 1), (0, -1), (2, 1), (1, -1), (2, -1)],
                TypeBoucle::C,
            ),
            (
                vec![(0, 1), (1, 1), (2, 1), (0, -1), (1, -1), (2, -1)],
                TypeBoucle::T,
            ),
        ];
        for (suite, type_attendu) in &exemples {
            // trouvée dans le type attendu (direction 0)
            assert!(
                table.formes(0, *type_attendu).contains(suite),
                "exemple de type {} absent de sa liste",
                type_attendu.nom()
            );
            // absente des autres types
            for autre in TypeBoucle::TOUS {
                if *type_attendu != autre {
                    assert!(
                        !table.formes(0, autre).contains(suite),
                        "exemple de type {} trouvé à tort dans le type {}",
                        type_attendu.nom(),
                        autre.nom()
                    );
                }
            }
        }
    }

    /// Énumération INDÉPENDANTE des boucles distinctes d'un champ.
    ///
    /// Retourne pour chaque type la somme des ½ Tr U_C sur toutes les boucles distinctes.
    /// N'utilise pas `TableFormes` : exploration exhaustive des chemins fermés depuis
    /// chaque site et chaque premier pas, dédupliqués par l'ensemble trié des liens.
    /// La fermeture exige un déplacement total NUL dans ℤ⁴ (pas seulement le retour au
    /// site de départ sur le tore).
    fn enumerer_boucles_independantes(champ: &ChampHote) -> [f64; 4] {
        let l = champ.l;
        let v = champ.v();
        let vois = Voisinage::nouveau(l);
        let mut sommes = [0.0; 4];
        let mut vues: HashSet<Vec<usize>> = HashSet::new();

        // Explorer les chemins depuis chaque site et chaque premier pas (mu, +1).
        for s in 0..v {
            for mu in 0..4 {
                let premier = (mu as u8, 1i8);
                let depart = s;
                let pos_initiale = vois.haut(mu, s);
                let deplacement_initial = {
                    let mut de = [0isize; 4];
                    de[mu] = 1;
                    de
                };
                // Pile : (chemin, position, dernier_pas, déplacement cumulé)
                let mut pile: Vec<(Vec<Pas>, usize, Pas, [isize; 4])> =
                    vec![(vec![premier], pos_initiale, premier, deplacement_initial)];
                while let Some((chemin, pos, dernier, deplacement)) = pile.pop() {
                    let longueur = chemin.len();
                    // si chemin fermé (retour au site ET déplacement total nul) et longueur autorisée
                    if pos == depart
                        && deplacement == [0, 0, 0, 0]
                        && (longueur == 4 || longueur == 6)
                    {
                        // sans rebroussement cyclique : le dernier pas ne doit pas être l'inverse du premier
                        if dernier.0 != premier.0 || dernier.1 != -premier.1 {
                            // construire la liste des liens traversés (triée)
                            let mut liens = Vec::with_capacity(longueur);
                            let mut curseur = depart;
                            for &(d, sig) in &chemin {
                                if sig == 1 {
                                    liens.push(d as usize * v + curseur);
                                    curseur = vois.haut(d as usize, curseur);
                                } else {
                                    curseur = vois.bas(d as usize, curseur);
                                    liens.push(d as usize * v + curseur);
                                }
                            }
                            liens.sort_unstable();
                            if vues.insert(liens) {
                                let type_idx = super::super::gates::classe_independante(&chemin);
                                sommes[type_idx] += super::super::gates::trace_chemin_independante(
                                    champ, depart, &chemin,
                                );
                            }
                        }
                    }
                    // étendre le chemin si longueur max non atteinte
                    if longueur < 6 {
                        let prochain_max = 6;
                        for d in 0..4u8 {
                            for sig in [1i8, -1] {
                                // éviter le rebroussement non cyclique (pas inverse du précédent)
                                if d == dernier.0 && sig == -dernier.1 {
                                    continue;
                                }
                                let nouveau_pas = (d, sig);
                                let mut nouveau_chemin = chemin.clone();
                                nouveau_chemin.push(nouveau_pas);
                                let nouvelle_pos = if sig == 1 {
                                    vois.haut(d as usize, pos)
                                } else {
                                    vois.bas(d as usize, pos)
                                };
                                let mut nouveau_deplacement = deplacement;
                                nouveau_deplacement[d as usize] += sig as isize;
                                if nouveau_chemin.len() <= prochain_max {
                                    pile.push((
                                        nouveau_chemin,
                                        nouvelle_pos,
                                        nouveau_pas,
                                        nouveau_deplacement,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        sommes
    }

    #[test]
    fn identite_controle_contre_enumeration_independante() {
        let champ = champ_haar(4, 26110201);
        let table = TableFormes::construire().unwrap();
        let vois = Voisinage::nouveau(champ.l);
        let famille = Famille::analyser("P,R,C,T").unwrap();
        let mut gauche = [0.0; 4];
        for mu in 0..4 {
            for s in 0..champ.v() {
                let agrafes = agrafes_lien(&champ, &vois, &table, &famille, mu, s);
                let u = champ.lien(mu * champ.v() + s);
                for (i, ag) in agrafes.iter().enumerate() {
                    gauche[i] += qmul(u, *ag)[0];
                }
            }
        }
        let droite = enumerer_boucles_independantes(&champ);
        for (i, ty) in TypeBoucle::TOUS.iter().enumerate() {
            let ecart = (gauche[i] - ty.longueur() as f64 * droite[i]).abs();
            let denom = ty.longueur() as f64 * droite[i].abs().max(1e-300);
            let rel = ecart / denom;
            assert!(rel <= 1e-12, "type {} : écart relatif {rel:.2e}", ty.nom());
        }

        // Le mutant ChaiseInversee casse l'identité pour C.
        let mut gauche_c_mutant = 0.0;
        for mu in 0..4 {
            for s in 0..champ.v() {
                let agrafes = agrafes_lien_variante(
                    &champ,
                    &vois,
                    &table,
                    &famille,
                    mu,
                    s,
                    VarianteAgrafes::ChaiseInversee,
                );
                let idx_c = famille.index(TypeBoucle::C).unwrap();
                let u = champ.lien(mu * champ.v() + s);
                gauche_c_mutant += qmul(u, agrafes[idx_c])[0];
            }
        }
        let ecart_mutant = (gauche_c_mutant - TypeBoucle::C.longueur() as f64 * droite[2]).abs();
        assert!(
            ecart_mutant > 1e-8,
            "le mutant ChaiseInversee devrait violer l'identité (écart = {ecart_mutant:.3e})"
        );
    }
}
