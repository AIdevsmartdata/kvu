//! Lecteur NPY minimal (v1.0, C-order) pour les fixtures et les tests de `cond`, et écriture atomique des
//! fichiers de reprise (octets bruts). Les sorties de production (F, D, blk) et la lecture des configurations
//! passent par `stockage::npy` (T8) ; ce module ne sert qu'à lire les fixtures de cb.py (`int64`, `float64`)
//! sans dépendre du format des champs. Écriture atomique : `.part`, `fsync`, renommage, `fsync` du répertoire.

use crate::commun::{Erreur, Resultat};
use std::io::Write;
use std::path::Path;

/// Contenu typé d'un tableau NPY.
#[derive(Clone, Debug, PartialEq)]
pub enum Donnees {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

/// Tableau NPY : forme et contenu (C-order).
#[derive(Clone, Debug, PartialEq)]
pub struct Tableau {
    pub forme: Vec<usize>,
    pub donnees: Donnees,
}

impl Tableau {
    pub fn f64(self) -> Resultat<Vec<f64>> {
        match self.donnees {
            Donnees::F64(v) => Ok(v),
            _ => Err(Erreur::Format("NPY : float64 attendu".into())),
        }
    }
    pub fn f32(self) -> Resultat<Vec<f32>> {
        match self.donnees {
            Donnees::F32(v) => Ok(v),
            _ => Err(Erreur::Format("NPY : float32 attendu".into())),
        }
    }
    pub fn i64(self) -> Resultat<Vec<i64>> {
        match self.donnees {
            Donnees::I64(v) => Ok(v),
            _ => Err(Erreur::Format("NPY : int64 attendu".into())),
        }
    }
    pub fn u32(self) -> Resultat<Vec<u32>> {
        match self.donnees {
            Donnees::U32(v) => Ok(v),
            _ => Err(Erreur::Format("NPY : uint32 attendu".into())),
        }
    }
    /// Entiers (int64 ou uint32 ou uint64) convertis en `u32` (indices de sites).
    pub fn indices_u32(self) -> Resultat<Vec<u32>> {
        let trop_grand = || Erreur::Format("NPY : indice hors de u32".into());
        match self.donnees {
            Donnees::I64(v) => v.into_iter().map(|x| u32::try_from(x).map_err(|_| trop_grand())).collect(),
            Donnees::U64(v) => v.into_iter().map(|x| u32::try_from(x).map_err(|_| trop_grand())).collect(),
            Donnees::U32(v) => Ok(v),
            _ => Err(Erreur::Format("NPY : entiers attendus".into())),
        }
    }
}

fn en_tete(descr: &str, forme: &[usize]) -> Vec<u8> {
    let f = match forme.len() {
        0 => "()".to_string(),
        1 => format!("({},)", forme[0]),
        _ => format!("({})", forme.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut hd = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {f}, }}");
    let tot = 10 + hd.len() + 1;
    hd.push_str(&" ".repeat((64 - tot % 64) % 64));
    hd.push('\n');
    let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    out.extend_from_slice(&(hd.len() as u16).to_le_bytes());
    out.extend_from_slice(hd.as_bytes());
    out
}

/// Écrit un fichier de façon atomique (`.part`, fsync, renommage, fsync du répertoire).
pub fn ecrire_atomique(chemin: &Path, morceaux: &[&[u8]]) -> Resultat<()> {
    let mut tmp = chemin.as_os_str().to_owned();
    tmp.push(".part");
    let tmp = std::path::PathBuf::from(tmp);
    let ctx = |op: &str| format!("{op} {}", chemin.display());
    {
        let f = std::fs::File::create(&tmp).map_err(|e| Erreur::io(ctx("création"), e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
        for m in morceaux {
            w.write_all(m).map_err(|e| Erreur::io(ctx("écriture"), e))?;
        }
        w.flush().map_err(|e| Erreur::io(ctx("écriture"), e))?;
        w.into_inner()
            .map_err(|e| Erreur::io(ctx("écriture"), e.into_error()))?
            .sync_all()
            .map_err(|e| Erreur::io(ctx("fsync"), e))?;
    }
    std::fs::rename(&tmp, chemin).map_err(|e| Erreur::io(ctx("renommage vers"), e))?;
    if let Some(dir) = chemin.parent() {
        let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

fn octets_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn verifier_forme(forme: &[usize], n: usize) -> Resultat<()> {
    if forme.iter().product::<usize>() != n {
        return Err(Erreur::Format(format!("NPY : forme {forme:?} incompatible avec {n} éléments")));
    }
    Ok(())
}

pub fn ecrire_f64(chemin: &Path, forme: &[usize], donnees: &[f64]) -> Resultat<()> {
    verifier_forme(forme, donnees.len())?;
    ecrire_atomique(chemin, &[&en_tete("<f8", forme), &octets_f64(donnees)])
}

pub fn ecrire_f32(chemin: &Path, forme: &[usize], donnees: &[f32]) -> Resultat<()> {
    verifier_forme(forme, donnees.len())?;
    let o: Vec<u8> = donnees.iter().flat_map(|x| x.to_le_bytes()).collect();
    ecrire_atomique(chemin, &[&en_tete("<f4", forme), &o])
}

pub fn ecrire_i64(chemin: &Path, forme: &[usize], donnees: &[i64]) -> Resultat<()> {
    verifier_forme(forme, donnees.len())?;
    let o: Vec<u8> = donnees.iter().flat_map(|x| x.to_le_bytes()).collect();
    ecrire_atomique(chemin, &[&en_tete("<i8", forme), &o])
}

/// Lit un NPY (descr `<f8`, `<f4`, `<i8`, `<u4`, `<u8`), C-order, version 1 ou 2.
pub fn lire(chemin: &Path) -> Resultat<Tableau> {
    let b = std::fs::read(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    analyser(&b).map_err(|m| Erreur::Format(format!("{} : {m}", chemin.display())))
}

fn analyser(b: &[u8]) -> Result<Tableau, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("signature NPY absente".into());
    }
    let (hl, debut) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => return Err(format!("version NPY {v} non gérée")),
    };
    let hd = std::str::from_utf8(b.get(debut..debut + hl).ok_or("en-tête tronqué")?).map_err(|e| e.to_string())?;
    if !hd.contains("'fortran_order': False") {
        return Err("fortran_order non gérée".into());
    }
    let champ = |cle: &str| -> Result<&str, String> {
        let i = hd.find(cle).ok_or(format!("{cle} absent"))? + cle.len();
        Ok(&hd[i..])
    };
    let d = champ("'descr': '")?;
    let descr = &d[..d.find('\'').ok_or("descr")?];
    let s = champ("'shape': (")?;
    let forme: Vec<usize> = s[..s.find(')').ok_or("shape")?]
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let n: usize = forme.iter().product();
    let corps = &b[debut + hl..];
    let lire_n = |taille: usize| -> Result<&[u8], String> {
        if corps.len() != n * taille {
            Err(format!("taille {} octets au lieu de {} (forme {forme:?})", corps.len(), n * taille))
        } else {
            Ok(corps)
        }
    };
    let donnees = match descr {
        "<f8" => Donnees::F64(lire_n(8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
        "<f4" => Donnees::F32(lire_n(4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        "<i8" => Donnees::I64(lire_n(8)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
        "<u4" => Donnees::U32(lire_n(4)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()),
        "<u8" => Donnees::U64(lire_n(8)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()),
        autre => return Err(format!("descr « {autre} » non géré")),
    };
    Ok(Tableau { forme, donnees })
}

/// Lit un champ SU(2) de quaternions : NPY `float64 (4, L, L, L, L, 4)` (format q64) OU `complex128
/// (4, L, L, L, L, 2, 2)` (format c16, via `heritage::su2_hb::read_npy_quat`). Renvoie (L, q plat).
pub fn lire_champ_quaternions(chemin: &Path) -> Resultat<(usize, Vec<f64>)> {
    let b = std::fs::read(chemin).map_err(|e| Erreur::io(format!("lecture {}", chemin.display()), e))?;
    if b.len() > 128 && b[..6] == *b"\x93NUMPY" && b[10..128.min(b.len())].windows(5).any(|w| w == b"'<c16") {
        return crate::heritage::su2_hb::read_npy_quat(chemin).map_err(Erreur::Format);
    }
    let t = analyser(&b).map_err(|m| Erreur::Format(format!("{} : {m}", chemin.display())))?;
    if t.forme.len() != 6 || t.forme[0] != 4 || t.forme[5] != 4 || t.forme[1..5].iter().any(|&d| d != t.forme[1]) {
        return Err(Erreur::Format(format!(
            "{} : forme {:?} refusée (q64 : (4,L,L,L,L,4))",
            chemin.display(),
            t.forme
        )));
    }
    let l = t.forme[1];
    Ok((l, t.f64()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(nom: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_cond_npy_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(nom)
    }

    #[test]
    fn aller_retour_f64_f32_i64() {
        let p = tmp("a.npy");
        ecrire_f64(&p, &[2, 3], &[1.0, -2.5, 3.0, 4.0, f64::MIN_POSITIVE, 6.0]).unwrap();
        let t = lire(&p).unwrap();
        assert_eq!(t.forme, vec![2, 3]);
        assert_eq!(t.f64().unwrap()[4], f64::MIN_POSITIVE);
        ecrire_f32(&p, &[3], &[1.5, 2.5, 3.5]).unwrap();
        assert_eq!(lire(&p).unwrap().f32().unwrap(), vec![1.5, 2.5, 3.5]);
        ecrire_i64(&p, &[2], &[-1, 7]).unwrap();
        assert_eq!(lire(&p).unwrap().i64().unwrap(), vec![-1, 7]);
        assert!(!p.with_extension("npy.part").exists());
    }

    #[test]
    fn en_tete_aligne_et_numpy_compatible() {
        let h = en_tete("<f8", &[5]);
        assert_eq!(h.len() % 64, 0);
        assert!(std::str::from_utf8(&h[10..])
            .unwrap()
            .starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (5,), }"));
    }

    #[test]
    fn forme_incoherente_refusee() {
        assert!(ecrire_f64(&tmp("b.npy"), &[2, 2], &[1.0]).is_err());
    }

    #[test]
    fn fichier_tronque_refuse() {
        let p = tmp("c.npy");
        ecrire_f64(&p, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut b = std::fs::read(&p).unwrap();
        b.truncate(b.len() - 8);
        std::fs::write(&p, b).unwrap();
        assert!(lire(&p).is_err());
    }
}
