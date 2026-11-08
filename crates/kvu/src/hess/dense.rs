//! Algèbre dense F64 ; toutes les réductions respectent un découpage fixe.
use rayon::prelude::*;

const TRONCON: usize = 4096;
/// Cosinus maximal toléré entre deux colonnes à la convergence de la SVD de Jacobi (1e-15 est sous le bruit d'arrondi des produits scalaires longs).
const ORTHOGONALITE: f64 = 2e-14;
/// Une colonne de norme inférieure à ce rapport à la plus grande est nulle (bien sous tout seuil de rang raisonnable).
const NULLE_REL: f64 = 1e-13;
type BaseImage = Result<(Vec<Vec<f64>>, Vec<f64>, usize), String>;
type Spectre = Result<(Vec<f64>, Option<Vec<f64>>), String>;

/// Produit scalaire, avec sommation séquentielle par tronçons de 4096.
pub fn pdscal(pool: Option<&rayon::ThreadPool>, x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(
        x.len(),
        y.len(),
        "Tailles incompatibles du produit scalaire"
    );
    let calcul = |(a, b): (&[f64], &[f64])| {
        let mut somme = 0.0;
        for (&u, &v) in a.iter().zip(b) {
            somme += u * v;
        }
        somme
    };
    // Une seule tâche n'a pas besoin d'allocation ni d'entrée dans le pool.
    if x.len() <= TRONCON {
        return calcul((x, y));
    }
    let sommes: Vec<f64> = match pool {
        Some(pool) => pool.install(|| {
            x.par_chunks(TRONCON)
                .zip(y.par_chunks(TRONCON))
                .map(calcul)
                .collect()
        }),
        None => x
            .chunks(TRONCON)
            .zip(y.chunks(TRONCON))
            .map(calcul)
            .collect(),
    };
    let mut somme = 0.0;
    for partielle in sommes {
        somme += partielle;
    }
    somme
}

/// Norme euclidienne déterministe.
pub fn norme2(pool: Option<&rayon::ThreadPool>, x: &[f64]) -> f64 {
    pdscal(pool, x, x).sqrt()
}

/// Factorisation inférieure en place ; le premier pivot non positif est signalé.
pub fn cholesky(a: &mut [f64], n: usize) -> Result<(), usize> {
    assert_eq!(a.len(), n * n, "Taille incorrecte pour Cholesky");
    for i in 0..n {
        for j in 0..=i {
            let somme = pdscal(None, &a[i * n..i * n + j], &a[j * n..j * n + j]);
            let pivot = a[i * n + j] - somme;
            if i == j {
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(i);
                }
                a[i * n + i] = pivot.sqrt();
            } else {
                a[i * n + j] = pivot / a[j * n + j];
            }
        }
        a[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

/// Résout L y = b en place.
pub fn resoudre_l(l: &[f64], n: usize, b: &mut [f64]) {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    for i in 0..n {
        b[i] = (b[i] - pdscal(None, &l[i * n..i * n + i], &b[..i])) / l[i * n + i];
    }
}

/// Résout Lᵀ y = b en place.
pub fn resoudre_lt(l: &[f64], n: usize, b: &mut [f64]) {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    for i in (0..n).rev() {
        let mut somme = 0.0;
        for debut in (i + 1..n).step_by(TRONCON) {
            let mut partielle = 0.0;
            for j in debut..n.min(debut + TRONCON) {
                partielle += l[j * n + i] * b[j];
            }
            somme += partielle;
        }
        b[i] = (b[i] - somme) / l[i * n + i];
    }
}

/// Réduction symétrique de Householder, puis QL implicite avec décalage.
/// Les vecteurs propres sont les colonnes de la matrice rendue, en ligne-majeure.
pub fn eigh_sym(mut a: Vec<f64>, n: usize, vecteurs: bool) -> Spectre {
    if n.checked_mul(n) != Some(a.len()) || a.iter().any(|v| !v.is_finite()) {
        return Err("Matrice symétrique : taille incorrecte ou coefficient non fini".into());
    }
    if n == 0 {
        return Ok((Vec::new(), vecteurs.then(Vec::new)));
    }
    let mut d = a[(n - 1) * n..].to_vec();
    let mut e = vec![0.0; n];
    for i in (1..n).rev() {
        let mut echelle = 0.0;
        for troncon in d[..i].chunks(TRONCON) {
            let mut partielle = 0.0;
            for valeur in troncon {
                partielle += valeur.abs();
            }
            echelle += partielle;
        }
        let mut h = 0.0;
        if echelle == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = a[(i - 1) * n + j];
                a[i * n + j] = 0.0;
                a[j * n + i] = 0.0;
            }
        } else {
            for valeur in &mut d[..i] {
                *valeur /= echelle;
            }
            h = pdscal(None, &d[..i], &d[..i]);
            let f = d[i - 1];
            let g = if f > 0.0 { -h.sqrt() } else { h.sqrt() };
            e[i] = echelle * g;
            h -= f * g;
            d[i - 1] = f - g;
            e[..i].fill(0.0);
            // Chaque composante du produit symétrique est sommée par
            // tronçons fixes, en parcourant le triangle inférieur par lignes.
            let mut partielles = vec![0.0; i];
            for debut in (0..i).step_by(TRONCON) {
                let fin = i.min(debut + TRONCON);
                partielles.fill(0.0);
                for k in debut..i {
                    let borne = fin.min(k + 1);
                    partielles[k] =
                        pdscal(None, &a[k * n + debut..k * n + borne], &d[debut..borne]);
                }
                for k in debut..fin {
                    for j in 0..k {
                        partielles[j] += a[k * n + j] * d[k];
                    }
                }
                for j in 0..i {
                    e[j] += partielles[j];
                }
            }
            for valeur in &mut e[..i] {
                *valeur /= h;
            }
            let hh = pdscal(None, &e[..i], &d[..i]) / (2.0 * h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for k in 0..i {
                for j in 0..=k {
                    a[k * n + j] -= d[k] * e[j] + e[k] * d[j];
                }
            }
            for j in 0..i {
                a[j * n + i] = d[j];
                d[j] = a[(i - 1) * n + j];
                a[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulation par lignes contiguës pour éviter les parcours en colonnes
    // coûteux lorsque n dépasse les caches du processeur.
    for i in 0..n - 1 {
        a[(n - 1) * n + i] = a[i * n + i];
        a[i * n + i] = 1.0;
        let h = d[i + 1];
        if vecteurs && h != 0.0 {
            for k in 0..=i {
                d[k] = a[k * n + i + 1] / h;
            }
            let mut g = vec![0.0; i + 1];
            let mut partielles = vec![0.0; i + 1];
            for debut in (0..=i).step_by(TRONCON) {
                partielles.fill(0.0);
                for k in debut..(i + 1).min(debut + TRONCON) {
                    let facteur = a[k * n + i + 1];
                    for j in 0..=i {
                        partielles[j] += facteur * a[k * n + j];
                    }
                }
                for j in 0..=i {
                    g[j] += partielles[j];
                }
            }
            for k in 0..=i {
                for j in 0..=i {
                    a[k * n + j] -= g[j] * d[k];
                }
            }
        }
        for k in 0..=i {
            a[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = a[(n - 1) * n + j];
        a[(n - 1) * n + j] = 0.0;
    }
    a[n * n - 1] = 1.0;
    if vecteurs {
        // Pendant QL, les rotations agissent sur deux colonnes adjacentes.
        // Les ranger temporairement par colonnes rend ces écritures contiguës
        // et permet la vectorisation sans changer l'ordre des opérations.
        let mut colonnes = vec![0.0; n * n];
        for debut_i in (0..n).step_by(32) {
            for debut_j in (0..n).step_by(32) {
                for i in debut_i..n.min(debut_i + 32) {
                    for j in debut_j..n.min(debut_j + 32) {
                        colonnes[j * n + i] = a[i * n + j];
                    }
                }
            }
        }
        a = colonnes;
    }
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut decalage = 0.0;
    let mut borne: f64 = 0.0;
    for l in 0..n {
        borne = borne.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n && e[m].abs() > f64::EPSILON * borne {
            m += 1;
        }
        if m > l {
            let mut iterations = 0;
            loop {
                iterations += 1;
                if iterations > 100 {
                    return Err("QL implicite : absence de convergence".into());
                }
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let suivant = d[l + 1];
                let h = g - d[l];
                for valeur in &mut d[l + 2..] {
                    *valeur -= h;
                }
                decalage += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    let h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    if vecteurs {
                        let (avant, apres) = a.split_at_mut((i + 1) * n);
                        let gauche = &mut avant[i * n..];
                        let droite = &mut apres[..n];
                        for (u, v) in gauche.iter_mut().zip(droite) {
                            let h = *v;
                            *v = s * *u + c * h;
                            *u = c * *u - s * h;
                        }
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / suivant;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= f64::EPSILON * borne {
                    break;
                }
            }
        }
        d[l] += decalage;
        e[l] = 0.0;
    }
    if d.iter().any(|v| !v.is_finite()) {
        return Err("Valeur propre non finie".into());
    }
    // Tri stable : les espaces propres multiples conservent un ordre déterministe.
    let mut ordre: Vec<usize> = (0..n).collect();
    ordre.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let valeurs = ordre.iter().map(|&j| d[j]).collect();
    let v = if vecteurs {
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = a[ordre[j] * n + i];
            }
        }
        Some(v)
    } else {
        None
    };
    Ok((valeurs, v))
}

/// Résout A x = λ B x par congruence de Cholesky ; B doit être définie positive.
pub fn eigh_generalise(a: Vec<f64>, mut b: Vec<f64>, n: usize, vecteurs: bool) -> Spectre {
    if n.checked_mul(n) != Some(a.len()) || a.len() != b.len() {
        return Err("Problème généralisé : dimensions incompatibles".into());
    }
    cholesky(&mut b, n).map_err(|i| format!("B non définie positive : pivot {i}"))?;
    let mut c = a;
    let mut colonne = vec![0.0; n];
    for j in 0..n {
        for i in 0..n {
            colonne[i] = c[i * n + j];
        }
        resoudre_l(&b, n, &mut colonne);
        for i in 0..n {
            c[i * n + j] = colonne[i];
        }
    }
    for ligne in c.chunks_mut(n.max(1)) {
        resoudre_l(&b, n, ligne);
    }
    for i in 0..n {
        for j in 0..i {
            let moyenne = 0.5 * (c[i * n + j] + c[j * n + i]);
            c[i * n + j] = moyenne;
            c[j * n + i] = moyenne;
        }
    }
    let (valeurs, mut v) = eigh_sym(c, n, vecteurs)?;
    if let Some(v) = &mut v {
        for j in 0..n {
            for i in 0..n {
                colonne[i] = v[i * n + j];
            }
            resoudre_lt(&b, n, &mut colonne);
            for i in 0..n {
                v[i * n + j] = colonne[i];
            }
        }
    }
    Ok((valeurs, v))
}

/// Rotation de deux colonnes possédées : aucune écriture partagée entre paires.
/// `seuil2` : carré de la norme sous laquelle une colonne est considérée nulle (non tournée, non comptée dans la convergence).
fn tourner(mut paire: Paire, seuil2: f64) -> Paire {
    let a = pdscal(None, &paire.gauche, &paire.gauche);
    let b = pdscal(None, &paire.droite, &paire.droite);
    let g = pdscal(None, &paire.gauche, &paire.droite);
    if a > seuil2 && b > seuil2 && (g / a.sqrt() / b.sqrt()).abs() > ORTHOGONALITE {
        let z = (b - a) / (2.0 * g);
        let t = if z.is_infinite() {
            g / (b - a)
        } else {
            1.0_f64.copysign(z) / (z.abs() + z.hypot(1.0))
        };
        let c = 1.0 / t.hypot(1.0);
        let s = c * t;
        for (u, v) in paire.gauche.iter_mut().zip(&mut paire.droite) {
            let ancien = *u;
            *u = c * ancien - s * *v;
            *v = s * ancien + c * *v;
        }
        for (u, v) in paire.vg.iter_mut().zip(&mut paire.vd) {
            let ancien = *u;
            *u = c * ancien - s * *v;
            *v = s * ancien + c * *v;
        }
    }
    paire
}

struct Paire {
    i: usize,
    j: usize,
    gauche: Vec<f64>,
    droite: Vec<f64>,
    vg: Vec<f64>,
    vd: Vec<f64>,
}

/// SVD à un côté (Hestenes), tournoi fixe de paires disjointes.
/// Les valeurs et les colonnes restent dans le même ordre (sans tri).
pub fn svd_jacobi_colonnes(
    pool: Option<&rayon::ThreadPool>,
    cols: &mut [Vec<f64>],
    avec_v: bool,
) -> Spectre {
    let n = cols.len();
    let m = cols.first().map_or(0, Vec::len);
    if cols
        .iter()
        .any(|c| c.len() != m || c.iter().any(|v| !v.is_finite()))
    {
        return Err("SVD : colonnes incompatibles ou non finies".into());
    }
    let mut v = vec![Vec::new(); n];
    if avec_v {
        for (j, colonne) in v.iter_mut().enumerate() {
            *colonne = vec![0.0; n];
            colonne[j] = 1.0;
        }
    }
    let participants = n + n % 2;
    let mut tournoi: Vec<usize> = (0..participants).collect();
    for balayage in 0..=60 {
        let normes: Vec<f64> = cols.iter().map(|c| norme2(None, c)).collect();
        let nmax = normes.iter().copied().fold(0.0, f64::max);
        let seuil = NULLE_REL * nmax;
        let mut correlation: f64 = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if normes[i] > seuil && normes[j] > seuil {
                    correlation = correlation
                        .max((pdscal(None, &cols[i], &cols[j]) / normes[i] / normes[j]).abs());
                }
            }
        }
        if correlation <= ORTHOGONALITE {
            let matrice_v = avec_v.then(|| {
                let mut matrice = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        matrice[i * n + j] = v[j][i];
                    }
                }
                matrice
            });
            return Ok((normes, matrice_v));
        }
        if balayage == 60 {
            break;
        }
        for _ in 0..participants.saturating_sub(1) {
            let mut paires = Vec::with_capacity(participants / 2);
            for k in 0..participants / 2 {
                let i = tournoi[k];
                let j = tournoi[participants - 1 - k];
                if i < n && j < n {
                    paires.push(Paire {
                        i,
                        j,
                        gauche: std::mem::take(&mut cols[i]),
                        droite: std::mem::take(&mut cols[j]),
                        vg: std::mem::take(&mut v[i]),
                        vd: std::mem::take(&mut v[j]),
                    });
                }
            }
            let resultats: Vec<_> = match pool {
                Some(pool) => pool.install(|| paires.into_par_iter().map(|p| tourner(p, seuil * seuil)).collect()),
                None => paires.into_iter().map(|p| tourner(p, seuil * seuil)).collect(),
            };
            for paire in resultats {
                cols[paire.i] = paire.gauche;
                cols[paire.j] = paire.droite;
                v[paire.i] = paire.vg;
                v[paire.j] = paire.vd;
            }
            if participants > 2 {
                tournoi[1..].rotate_right(1);
            }
        }
    }
    Err("SVD de Jacobi : absence de convergence après 60 balayages".into())
}

/// Base de l'image par sélection des valeurs singulières strictement supérieures au seuil.
pub fn base_de_l_image(
    pool: Option<&rayon::ThreadPool>,
    mut cols: Vec<Vec<f64>>,
    seuil_rel: f64,
) -> BaseImage {
    if !seuil_rel.is_finite() || seuil_rel < 0.0 {
        return Err("Seuil relatif invalide".into());
    }
    let (sigma, _) = svd_jacobi_colonnes(pool, &mut cols, false)?;
    let mut ordre: Vec<usize> = (0..sigma.len()).collect();
    ordre.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    let maximum = ordre.first().map_or(0.0, |&j| sigma[j]);
    let mut base = Vec::new();
    let mut valeurs = Vec::with_capacity(sigma.len());
    for j in ordre {
        valeurs.push(sigma[j]);
        if sigma[j] > seuil_rel * maximum {
            let mut colonne = std::mem::take(&mut cols[j]);
            for valeur in &mut colonne {
                *valeur /= sigma[j];
            }
            base.push(colonne);
        }
    }
    let ecartes = sigma.len() - base.len();
    Ok((base, valeurs, ecartes))
}
