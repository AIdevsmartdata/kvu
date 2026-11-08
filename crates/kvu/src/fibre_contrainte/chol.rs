//! Cholesky creux par blocs 3×3, sans permutation, et références denses indépendantes.

use super::jacobien::M3;
use crate::commun::{Erreur, Resultat};
use rayon::prelude::*;

#[derive(Debug)]
pub struct ChDefaut {
    pub bloc: usize,
    pub pivot: f64,
}

/// Triangle inférieur en lignes compactes ; le dernier bloc de chaque ligne est diagonal.
#[derive(Debug)]
pub struct CholBlocs {
    debuts: Vec<usize>,
    colonnes: Vec<usize>,
    blocs: Vec<M3>,
    permutation: Vec<u32>,
}

fn verifier_pivot(pivot: f64, bloc: usize) -> Resultat<()> {
    if pivot <= 0.0 || !pivot.is_finite() {
        return Err(Erreur::Validation(format!(
            "G non définie positive : bloc {bloc}, pivot {pivot}"
        )));
    }
    Ok(())
}

/// Retire a bᵀ, dans l'ordre croissant des trois indices contractés.
#[inline]
fn soustraire_produit(bloc: &mut M3, a: &M3, b: &M3) {
    for i in 0..3 {
        for j in 0..3 {
            bloc[3 * i + j] -=
                a[3 * i] * b[3 * j] + a[3 * i + 1] * b[3 * j + 1] + a[3 * i + 2] * b[3 * j + 2];
        }
    }
}

fn diagonale(mut bloc: M3, indice: usize) -> Resultat<M3> {
    for j in 0..3 {
        let mut pivot = bloc[3 * j + j];
        for k in 0..j {
            pivot -= bloc[3 * j + k] * bloc[3 * j + k];
        }
        verifier_pivot(pivot, indice)?;
        bloc[3 * j + j] = pivot.sqrt();
        for i in j + 1..3 {
            let mut valeur = bloc[3 * i + j];
            for k in 0..j {
                valeur -= bloc[3 * i + k] * bloc[3 * j + k];
            }
            bloc[3 * i + j] = valeur / bloc[3 * j + j];
        }
        for k in j + 1..3 {
            bloc[3 * j + k] = 0.0;
        }
    }
    Ok(bloc)
}

impl CholBlocs {
    /// Remplissage symbolique par élimination du graphe, puis numérique « left-looking ».
    /// Les blocs sont éliminés dans leur ordre naturel (permutation identité).
    pub fn factoriser(g: &[Vec<(u32, M3)>]) -> Resultat<Self> {
        let n = g.len();
        if n > u32::MAX as usize {
            return Err(Erreur::Validation("trop de blocs de Cholesky".into()));
        }
        // Graphe supérieur en ensembles de bits : l'élimination d'un sommet complète
        // la clique de ses voisins ultérieurs, sans allocation dans ces boucles.
        let mots = n.div_ceil(64);
        let mut graphe = vec![0_u64; n * mots];
        for (i, ligne) in g.iter().enumerate() {
            let mut precedente = None;
            for &(colonne, _) in ligne {
                let j = colonne as usize;
                if j >= n || precedente.is_some_and(|p| p >= j) {
                    return Err(Erreur::Validation(format!(
                        "colonnes invalides ou non triées : bloc {i}, colonne {j}"
                    )));
                }
                precedente = Some(j);
                if j != i {
                    let (a, b) = (i.min(j), i.max(j));
                    graphe[a * mots + b / 64] |= 1_u64 << (b % 64);
                }
            }
        }
        for k in 0..n {
            for i in k + 1..n {
                if graphe[k * mots + i / 64] & (1_u64 << (i % 64)) == 0 {
                    continue;
                }
                let (avant, apres) = graphe.split_at_mut(i * mots);
                let source = &avant[k * mots..(k + 1) * mots];
                let cible = &mut apres[..mots];
                for m in i / 64..mots {
                    // Le décalage par 64 est évité au bord d'un mot.
                    let masque = if m == i / 64 {
                        u64::MAX.checked_shl((i % 64 + 1) as u32).unwrap_or(0)
                    } else {
                        u64::MAX
                    };
                    cible[m] |= source[m] & masque;
                }
            }
        }
        let mut debuts = Vec::with_capacity(n + 1);
        let mut colonnes = Vec::new();
        debuts.push(0);
        for i in 0..n {
            for j in 0..i {
                if graphe[j * mots + i / 64] & (1_u64 << (i % 64)) != 0 {
                    colonnes.push(j);
                }
            }
            colonnes.push(i);
            debuts.push(colonnes.len());
        }
        drop(graphe);
        let mut blocs = vec![[0.0; 9]; colonnes.len()];
        for i in 0..n {
            let mut entree = 0;
            for position in debuts[i]..debuts[i + 1] {
                let j = colonnes[position];
                while entree < g[i].len() && (g[i][entree].0 as usize) < j {
                    entree += 1;
                }
                let mut bloc = if entree < g[i].len() && g[i][entree].0 as usize == j {
                    g[i][entree].1
                } else {
                    [0.0; 9]
                };
                // Intersection des préfixes de lignes : seules les contributions
                // structurellement présentes sont visitées, en ordre croissant.
                let mut gauche = debuts[i];
                let mut droite = debuts[j];
                let fin = if i == j { position } else { debuts[j + 1] - 1 };
                while gauche < position && droite < fin {
                    let a = colonnes[gauche];
                    let b = colonnes[droite];
                    match a.cmp(&b) {
                        std::cmp::Ordering::Less => gauche += 1,
                        std::cmp::Ordering::Greater => droite += 1,
                        std::cmp::Ordering::Equal => {
                            soustraire_produit(&mut bloc, &blocs[gauche], &blocs[droite]);
                            gauche += 1;
                            droite += 1;
                        }
                    }
                }
                if i == j {
                    bloc = diagonale(bloc, i)?;
                } else {
                    let d = &blocs[debuts[j + 1] - 1];
                    // Résolution à droite de L_ij L_jjᵀ = bloc.
                    for r in 0..3 {
                        for c in 0..3 {
                            let mut valeur = bloc[3 * r + c];
                            for k in 0..c {
                                valeur -= bloc[3 * r + k] * d[3 * c + k];
                            }
                            bloc[3 * r + c] = valeur / d[3 * c + c];
                        }
                    }
                }
                blocs[position] = bloc;
            }
        }
        Ok(Self {
            debuts,
            colonnes,
            blocs,
            permutation: (0..n as u32).collect(),
        })
    }

    pub fn n_blocs(&self) -> usize {
        self.permutation.len()
    }

    pub fn permutation(&self) -> &[u32] {
        &self.permutation
    }

    /// Nombre de blocs du triangle inférieur, diagonale et remplissage compris.
    pub fn nb_blocs_non_nuls(&self) -> usize {
        self.blocs.len()
    }

    pub fn log_det(&self) -> f64 {
        let mut somme = 0.0;
        for i in 0..self.n_blocs() {
            let d = &self.blocs[self.debuts[i + 1] - 1];
            for j in 0..3 {
                somme += d[3 * j + j].ln();
            }
        }
        2.0 * somme
    }

    /// Résout G x = b en place ; `b` doit contenir exactement 3n composantes.
    pub fn resoudre(&self, b: &mut [f64]) {
        assert_eq!(b.len(), 3 * self.n_blocs(), "longueur du second membre");
        for i in 0..self.n_blocs() {
            let fin = self.debuts[i + 1] - 1;
            for p in self.debuts[i]..fin {
                let j = self.colonnes[p];
                let a = &self.blocs[p];
                for r in 0..3 {
                    b[3 * i + r] -= a[3 * r] * b[3 * j]
                        + a[3 * r + 1] * b[3 * j + 1]
                        + a[3 * r + 2] * b[3 * j + 2];
                }
            }
            let d = &self.blocs[fin];
            for r in 0..3 {
                for c in 0..r {
                    b[3 * i + r] -= d[3 * r + c] * b[3 * i + c];
                }
                b[3 * i + r] /= d[3 * r + r];
            }
        }
        // La passe arrière disperse dans les lignes précédentes : aucun index
        // supplémentaire ni tampon n'est nécessaire.
        for i in (0..self.n_blocs()).rev() {
            let fin = self.debuts[i + 1] - 1;
            let d = &self.blocs[fin];
            for r in (0..3).rev() {
                for c in r + 1..3 {
                    b[3 * i + r] -= d[3 * c + r] * b[3 * i + c];
                }
                b[3 * i + r] /= d[3 * r + r];
            }
            for p in self.debuts[i]..fin {
                let j = self.colonnes[p];
                let a = &self.blocs[p];
                for c in 0..3 {
                    b[3 * j + c] -=
                        a[c] * b[3 * i] + a[3 + c] * b[3 * i + 1] + a[6 + c] * b[3 * i + 2];
                }
            }
        }
    }

    /// Facteur L dense, en ordre ligne (référence de comparaison).
    pub fn facteur_dense(&self) -> Vec<f64> {
        let n = 3 * self.n_blocs();
        let mut dense = vec![0.0; n * n];
        for i in 0..self.n_blocs() {
            for p in self.debuts[i]..self.debuts[i + 1] {
                let j = self.colonnes[p];
                for r in 0..3 {
                    dense[(3 * i + r) * n + 3 * j..(3 * i + r) * n + 3 * j + 3]
                        .copy_from_slice(&self.blocs[p][3 * r..3 * r + 3]);
                }
            }
        }
        dense
    }

    /// Colonnes indépendantes, calculées dans `pool`, identiques à l'octet entre pools.
    /// La moyenne des deux triangles retire leur seule dissymétrie d'arrondi.
    pub fn inverse_dense(&self, pool: &rayon::ThreadPool) -> Vec<f64> {
        let n = 3 * self.n_blocs();
        if n == 0 {
            return Vec::new();
        }
        let mut inverse = vec![0.0; n * n];
        pool.install(|| {
            inverse
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(j, colonne)| {
                    colonne[j] = 1.0;
                    self.resoudre(colonne);
                });
        });
        // Après symétrisation, les représentations par lignes et colonnes coïncident.
        for i in 0..n {
            for j in i + 1..n {
                let valeur = 0.5 * inverse[i * n + j] + 0.5 * inverse[j * n + i];
                inverse[i * n + j] = valeur;
                inverse[j * n + i] = valeur;
            }
        }
        inverse
    }
}

/// Cholesky dense classique, colonne par colonne, triangle inférieur en ordre ligne.
pub fn cholesky_dense(a: &[f64], n: usize) -> Resultat<Vec<f64>> {
    if n.checked_mul(n) != Some(a.len()) {
        return Err(Erreur::Validation(
            "dimensions du Cholesky dense invalides".into(),
        ));
    }
    let mut l = vec![0.0; a.len()];
    for j in 0..n {
        let mut pivot = a[j * n + j];
        for k in 0..j {
            pivot -= l[j * n + k] * l[j * n + k];
        }
        verifier_pivot(pivot, j / 3)?;
        l[j * n + j] = pivot.sqrt();
        for i in j + 1..n {
            let mut valeur = a[i * n + j];
            for k in 0..j {
                valeur -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = valeur / l[j * n + j];
        }
    }
    Ok(l)
}

/// Logarithme du déterminant par Gauss à pivot partiel, indépendant de Cholesky.
/// Les matrices singulières, non finies ou de déterminant négatif sont refusées.
pub fn logdet_lu(a: &[f64], n: usize) -> Resultat<f64> {
    if n.checked_mul(n) != Some(a.len()) || a.iter().any(|x| !x.is_finite()) {
        return Err(Erreur::Validation("matrice LU invalide".into()));
    }
    let mut lu = a.to_vec();
    let mut signe = 1;
    let mut logarithme = 0.0;
    for k in 0..n {
        let mut ligne = k;
        for i in k + 1..n {
            if lu[i * n + k].abs() > lu[ligne * n + k].abs() {
                ligne = i;
            }
        }
        let pivot = lu[ligne * n + k];
        if pivot == 0.0 || !pivot.is_finite() {
            return Err(Erreur::Validation(format!(
                "pivot LU invalide : colonne {k}, pivot {pivot}"
            )));
        }
        if ligne != k {
            for j in 0..n {
                lu.swap(k * n + j, ligne * n + j);
            }
            signe = -signe;
        }
        if pivot < 0.0 {
            signe = -signe;
        }
        logarithme += pivot.abs().ln();
        for i in k + 1..n {
            let facteur = lu[i * n + k] / pivot;
            lu[i * n + k] = 0.0;
            for j in k + 1..n {
                lu[i * n + j] -= facteur * lu[k * n + j];
            }
        }
    }
    if signe < 0 || !logarithme.is_finite() {
        return Err(Erreur::Validation(
            "déterminant LU négatif ou non fini".into(),
        ));
    }
    Ok(logarithme)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Aleatoire(u64);

    impl Aleatoire {
        fn reel(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            2.0 * ((self.0 >> 11) as f64 / ((1_u64 << 53) as f64)) - 1.0
        }
    }

    fn pool(fils: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(fils)
            .build()
            .unwrap()
    }

    fn par_blocs(dense: &[f64], n: usize) -> Vec<Vec<(u32, M3)>> {
        (0..n / 3)
            .map(|i| {
                (0..n / 3)
                    .filter_map(|j| {
                        let bloc =
                            std::array::from_fn(|k| dense[(3 * i + k / 3) * n + 3 * j + k % 3]);
                        bloc.iter().any(|&x| x != 0.0).then_some((j as u32, bloc))
                    })
                    .collect()
            })
            .collect()
    }

    fn assembler(g: &[Vec<(u32, M3)>]) -> Vec<f64> {
        let n = 3 * g.len();
        let mut dense = vec![0.0; n * n];
        for (i, ligne) in g.iter().enumerate() {
            for &(j, bloc) in ligne {
                for r in 0..3 {
                    for c in 0..3 {
                        dense[(3 * i + r) * n + 3 * j as usize + c] = bloc[3 * r + c];
                    }
                }
            }
        }
        dense
    }

    fn bande(n: usize) -> Vec<Vec<(u32, M3)>> {
        let mut g = vec![Vec::new(); n];
        let mut aleatoire = Aleatoire(0xabc7_3985);
        for i in 0..n {
            let mut diagonale = [0.0; 9];
            for r in 0..3 {
                diagonale[3 * r + r] = 8.0 + aleatoire.reel();
            }
            // Bloc diagonal non diagonal pour éprouver le Cholesky 3×3.
            diagonale[1] = 0.1;
            diagonale[3] = 0.1;
            diagonale[5] = -0.2;
            diagonale[7] = -0.2;
            g[i].push((i as u32, diagonale));
            for distance in [1, 3] {
                let j = i + distance;
                if j < n {
                    let bloc = std::array::from_fn(|_| 0.2 * aleatoire.reel());
                    let transposee = std::array::from_fn(|k| bloc[3 * (k % 3) + k / 3]);
                    g[i].push((j as u32, bloc));
                    g[j].push((i as u32, transposee));
                }
            }
        }
        for ligne in &mut g {
            ligne.sort_unstable_by_key(|&(j, _)| j);
        }
        g
    }

    fn comparer_facteurs(g: &[Vec<(u32, M3)>]) -> CholBlocs {
        let dense = assembler(g);
        let n = 3 * g.len();
        let debut = std::time::Instant::now();
        let creux = CholBlocs::factoriser(g).unwrap();
        eprintln!(
            "Cholesky {} blocs, {} blocs du facteur : {:?}",
            g.len(),
            creux.nb_blocs_non_nuls(),
            debut.elapsed()
        );
        let reference = cholesky_dense(&dense, n).unwrap();
        let facteur = creux.facteur_dense();
        let ecart = facteur
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norme = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(ecart / norme <= 1e-12, "FIB-CHOL : {:e}", ecart / norme);
        let determinant = logdet_lu(&dense, n).unwrap();
        let relatif = (creux.log_det() - determinant).abs() / determinant.abs();
        assert!(relatif <= 1e-10, "log det : {relatif:e}");
        assert_eq!(creux.n_blocs(), g.len());
        assert_eq!(
            creux.permutation(),
            &(0..g.len() as u32).collect::<Vec<_>>()
        );
        creux
    }

    #[test]
    fn spd_aleatoire_dense_cent_quatre_vingt_douze() {
        let n = 192;
        let mut aleatoire = Aleatoire(0x52e7_3a17_0b19);
        let a: Vec<_> = (0..n * n).map(|_| aleatoire.reel()).collect();
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut valeur = if i == j { n as f64 } else { 0.0 };
                for k in 0..n {
                    valeur += a[i * n + k] * a[j * n + k];
                }
                g[i * n + j] = valeur;
                g[j * n + i] = valeur;
            }
        }
        let blocs = par_blocs(&g, n);
        let facteur = comparer_facteurs(&blocs);
        assert_eq!(facteur.nb_blocs_non_nuls(), 64 * 65 / 2);
        verifier_resolution(&g, &facteur);
    }

    fn verifier_resolution(g: &[f64], facteur: &CholBlocs) {
        let n = 3 * facteur.n_blocs();
        let mut aleatoire = Aleatoire(0x7827_361a);
        let b: Vec<_> = (0..n).map(|_| aleatoire.reel()).collect();
        let mut x = b.clone();
        facteur.resoudre(&mut x);
        let residu = (0..n)
            .map(|i| {
                (g[i * n..(i + 1) * n]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    - b[i])
                    .abs()
            })
            .fold(0.0, f64::max);
        let norme = b.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(
            residu <= 1e-10 * norme,
            "résidu : {residu:e}, norme : {norme:e}"
        );
    }

    #[test]
    fn bande_quarante_blocs_et_remplissage_symbolique() {
        let g = bande(40);
        let facteur = comparer_facteurs(&g);
        let initiaux = g
            .iter()
            .enumerate()
            .map(|(i, ligne)| ligne.iter().filter(|(j, _)| *j as usize <= i).count())
            .sum::<usize>();
        assert!(
            facteur.nb_blocs_non_nuls() > initiaux,
            "le motif doit créer du remplissage"
        );
        assert!(facteur.nb_blocs_non_nuls() < 40 * 41 / 2);
        verifier_resolution(&assembler(&g), &facteur);
    }

    #[test]
    fn inverse_residu_et_determinisme_un_trois_huit_fils() {
        let g = bande(40);
        let facteur = CholBlocs::factoriser(&g).unwrap();
        let inverse = facteur.inverse_dense(&pool(1));
        for fils in [3, 8] {
            let autre = facteur.inverse_dense(&pool(fils));
            assert!(inverse
                .iter()
                .zip(autre)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let dense = assembler(&g);
        let n = 3 * g.len();
        let mut residu: f64 = 0.0;
        for i in 0..n {
            let mut somme = 0.0;
            for j in 0..n {
                assert_eq!(inverse[i * n + j].to_bits(), inverse[j * n + i].to_bits());
                let produit = (0..n)
                    .map(|k| dense[i * n + k] * inverse[k * n + j])
                    .sum::<f64>();
                somme += (produit - f64::from(i == j)).abs();
            }
            residu = residu.max(somme);
        }
        assert!(
            residu <= 1e-9,
            "norme infinie du résidu de l'inverse : {residu:e}"
        );
    }

    #[test]
    fn diagonale_connue_et_pivots_invalides() {
        let g = vec![
            vec![(0, [4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0])],
            vec![(1, [25.0, 0.0, 0.0, 0.0, 36.0, 0.0, 0.0, 0.0, 49.0])],
        ];
        let facteur = CholBlocs::factoriser(&g).unwrap();
        let attendu = [4.0_f64, 9.0, 16.0, 25.0, 36.0, 49.0]
            .iter()
            .map(|x| x.ln())
            .sum::<f64>();
        assert!((facteur.log_det() - attendu).abs() < 1e-14);
        assert_eq!(facteur.nb_blocs_non_nuls(), 2);
        for pivot in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut invalide = g.clone();
            invalide[1][0].1[4] = pivot;
            let erreur = CholBlocs::factoriser(&invalide).unwrap_err();
            assert!(matches!(&erreur, Erreur::Validation(message)
                if message.starts_with("G non définie positive : bloc 1, pivot ")));
            assert!(matches!(
                cholesky_dense(&assembler(&invalide), 6),
                Err(Erreur::Validation(_))
            ));
        }
        let indefinie = vec![
            vec![
                (0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                (1, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
            ],
            vec![
                (0, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
                (1, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ],
        ];
        assert!(CholBlocs::factoriser(&indefinie).is_err());
    }

    #[test]
    fn lu_avec_pivot_partiel_et_validation_des_dimensions() {
        let a = [0.0, 2.0, -3.0, 4.0];
        assert!((logdet_lu(&a, 2).unwrap() - 6.0_f64.ln()).abs() < 1e-15);
        for a in [[1.0, 0.0, 0.0, -1.0], [0.0; 4], [f64::NAN; 4]] {
            assert!(logdet_lu(&a, 2).is_err());
        }
        assert!(logdet_lu(&[], usize::MAX).is_err());
        assert!(cholesky_dense(&[], usize::MAX).is_err());
        assert!(CholBlocs::factoriser(&[vec![(1, [0.0; 9])]]).is_err());
        assert!(CholBlocs::factoriser(&[vec![(0, [0.0; 9]), (0, [0.0; 9])]]).is_err());
        let vide = CholBlocs::factoriser(&[]).unwrap();
        assert_eq!(vide.log_det(), 0.0);
        assert!(vide.facteur_dense().is_empty());
        assert!(vide.inverse_dense(&pool(1)).is_empty());
        vide.resoudre(&mut []);
    }

    #[test]
    fn bande_mille_vingt_quatre_blocs() {
        let g = bande(1024);
        let debut = std::time::Instant::now();
        let facteur = CholBlocs::factoriser(&g).unwrap();
        eprintln!("Cholesky bande 1024 blocs : {:?}", debut.elapsed());
        assert!(facteur.nb_blocs_non_nuls() < 4 * 1024);
        let n = 3 * g.len();
        let mut b = vec![1.0; n];
        facteur.resoudre(&mut b);
        for ligne in &g {
            for r in 0..3 {
                let produit = ligne
                    .iter()
                    .map(|&(j, bloc)| {
                        (0..3)
                            .map(|c| bloc[3 * r + c] * b[3 * j as usize + c])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                assert!((produit - 1.0).abs() < 1e-10);
            }
        }
    }
}
