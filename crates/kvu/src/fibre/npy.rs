//! NPY v1.0 minimal pour la mesure de fibre (lecture des champs et des fixtures, écriture de F, D, blk).
//!
//! Local à T6 : `stockage::npy` (T8) fournira la version complète avec garde SSD ; les sorties de la
//! mesure sont minuscules (D = 23 Ko, blk ≤ quelques Mo), l'écriture est atomique (.part puis rename).

use crate::commun::{ChampHote, Erreur, Resultat};
use std::path::Path;

/// Contenu d'un tableau NPY (ordre C).
#[derive(Clone, Debug, PartialEq)]
pub enum Donnees {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U8(Vec<u8>),
    /// complex128 : réels et imaginaires entrelacés.
    C128(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Npy {
    pub forme: Vec<usize>,
    pub donnees: Donnees,
}

fn fmt_err(chemin: &Path, m: impl std::fmt::Display) -> Erreur {
    Erreur::Format(format!("{} : {m}", chemin.display()))
}

/// Analyse un contenu NPY v1 (en-tête puis octets).
pub fn analyser(octets: &[u8], origine: &Path) -> Resultat<Npy> {
    if octets.len() < 10 || &octets[..6] != b"\x93NUMPY" || octets[6] != 1 {
        return Err(fmt_err(origine, "NPY v1 attendu"));
    }
    let hl = u16::from_le_bytes([octets[8], octets[9]]) as usize;
    if octets.len() < 10 + hl {
        return Err(fmt_err(origine, "en-tête tronqué"));
    }
    let hd = String::from_utf8_lossy(&octets[10..10 + hl]).to_string();
    if !hd.contains("'fortran_order': False") {
        return Err(fmt_err(origine, "ordre Fortran refusé"));
    }
    let d0 = hd.find("'descr': '").ok_or_else(|| fmt_err(origine, "descr absent"))? + 10;
    let d1 = d0 + hd[d0..].find('\'').ok_or_else(|| fmt_err(origine, "descr"))?;
    let descr = &hd[d0..d1];
    let i0 = hd.find("'shape': (").ok_or_else(|| fmt_err(origine, "forme absente"))? + 10;
    let i1 = i0 + hd[i0..].find(')').ok_or_else(|| fmt_err(origine, "forme"))?;
    let forme: Vec<usize> = hd[i0..i1]
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|_| fmt_err(origine, "forme illisible"))
        })
        .collect::<Resultat<_>>()?;
    let n: usize = forme.iter().product();
    let data = &octets[10 + hl..];
    let (taille, _) = match descr {
        "<f8" | "<i8" => (8, ()),
        "<f4" | "<u4" => (4, ()),
        "|u1" => (1, ()),
        "<c16" => (16, ()),
        autre => return Err(fmt_err(origine, format!("dtype {autre} non géré"))),
    };
    if data.len() != n * taille {
        return Err(fmt_err(
            origine,
            format!("{} octets au lieu de {}", data.len(), n * taille),
        ));
    }
    let donnees = match descr {
        "<f8" => Donnees::F64(
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        "<c16" => Donnees::C128(
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        "<i8" => Donnees::I64(
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        "<f4" => Donnees::F32(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        "<u4" => Donnees::U32(
            data.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        _ => Donnees::U8(data.to_vec()),
    };
    Ok(Npy { forme, donnees })
}

pub fn lire(chemin: &Path) -> Resultat<Npy> {
    let b = std::fs::read(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    analyser(&b, chemin)
}

/// Lit un tableau f64 de forme libre.
pub fn lire_f64(chemin: &Path) -> Resultat<(Vec<usize>, Vec<f64>)> {
    match lire(chemin)? {
        Npy {
            forme,
            donnees: Donnees::F64(d),
        } => Ok((forme, d)),
        _ => Err(fmt_err(chemin, "float64 attendu")),
    }
}

fn en_tete(descr: &str, forme: &[usize]) -> Vec<u8> {
    let shape = match forme.len() {
        1 => format!("({},)", forme[0]),
        _ => format!(
            "({})",
            forme.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // longueur totale (10 + h + '\n') multiple de 64
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

fn ecrire_octets(chemin: &Path, descr: &str, forme: &[usize], corps: &[u8]) -> Resultat<()> {
    let mut contenu = en_tete(descr, forme);
    contenu.extend_from_slice(corps);
    let part = chemin.with_extension("npy.part");
    std::fs::write(&part, &contenu).map_err(|e| Erreur::io(format!("écriture {}", part.display()), e))?;
    std::fs::rename(&part, chemin).map_err(|e| Erreur::io(format!("renommage {}", chemin.display()), e))
}

pub fn ecrire_f64(chemin: &Path, forme: &[usize], d: &[f64]) -> Resultat<()> {
    assert_eq!(forme.iter().product::<usize>(), d.len());
    let corps: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    ecrire_octets(chemin, "<f8", forme, &corps)
}

pub fn ecrire_f32(chemin: &Path, forme: &[usize], d: &[f32]) -> Resultat<()> {
    assert_eq!(forme.iter().product::<usize>(), d.len());
    let corps: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    ecrire_octets(chemin, "<f4", forme, &corps)
}

/// Lit une configuration : complex128 (4,L,L,L,L,2,2) (contrôle SU(2) à 1e-12, via le lecteur du moteur
/// historique) ou quaternions float64 (4,L,L,L,L,4).
pub fn lire_champ(chemin: &Path) -> Resultat<ChampHote> {
    let b = std::fs::read(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    let npy = analyser(&b, chemin)?;
    match npy.donnees {
        Donnees::C128(_) => {
            let (l, q) = crate::heritage::su2_hb::read_npy_quat(chemin).map_err(|m| fmt_err(chemin, m))?;
            Ok(ChampHote { l, q })
        }
        Donnees::F64(q) => {
            let f = &npy.forme;
            if f.len() != 6 || f[0] != 4 || f[5] != 4 || f[1..5].iter().any(|&x| x != f[1]) {
                return Err(fmt_err(chemin, format!("forme {f:?} : (4, L, L, L, L, 4) attendu")));
            }
            let l = f[1];
            let c = ChampHote { l, q };
            let defaut = c.defaut_unitarite();
            if defaut.is_nan() || defaut > 1e-12 {
                return Err(fmt_err(chemin, "quaternions non unitaires à 1e-12"));
            }
            Ok(c)
        }
        _ => Err(fmt_err(
            chemin,
            "complex128 (4,L,L,L,L,2,2) ou float64 (4,L,L,L,L,4) attendu",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(nom: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_fibre_npy_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(nom)
    }

    #[test]
    fn aller_retour_f64_et_f32() {
        let p = tmp("a.npy");
        ecrire_f64(&p, &[2, 3], &[1.0, -2.5, 3.0, 4.0, 5.5, 6.0]).unwrap();
        let (f, d) = lire_f64(&p).unwrap();
        assert_eq!(f, vec![2, 3]);
        assert_eq!(d, vec![1.0, -2.5, 3.0, 4.0, 5.5, 6.0]);
        let q = tmp("b.npy");
        ecrire_f32(&q, &[4], &[1.5, 2.5, 3.5, 4.5]).unwrap();
        assert_eq!(lire(&q).unwrap().donnees, Donnees::F32(vec![1.5, 2.5, 3.5, 4.5]));
        assert!(!p.with_extension("npy.part").exists());
    }

    #[test]
    fn en_tete_aligne_64() {
        for forme in [vec![3usize], vec![2, 2], vec![4, 4, 4, 4, 4, 4]] {
            let h = en_tete("<f8", &forme);
            assert_eq!(h.len() % 64, 0);
        }
    }

    #[test]
    fn refus_hors_format() {
        let p = tmp("c.npy");
        std::fs::write(&p, b"pas un npy").unwrap();
        assert!(lire(&p).is_err());
    }

    #[test]
    fn champ_q64_aller_retour() {
        let c = ChampHote::froid(4).unwrap();
        let p = tmp("q.npy");
        ecrire_f64(&p, &[4, 4, 4, 4, 4, 4], &c.q).unwrap();
        assert_eq!(lire_champ(&p).unwrap(), c);
        let mauvais = tmp("m.npy");
        ecrire_f64(&mauvais, &[4, 4, 4, 4, 4, 4], &vec![2.0; c.q.len()]).unwrap();
        assert!(lire_champ(&mauvais).is_err());
    }
}
