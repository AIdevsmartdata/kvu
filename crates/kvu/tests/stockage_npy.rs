//! Tests du format NPY de la tâche T8 :
//! - c16 IDENTIQUE à l'octet à `su2_hb::write_npy_quat` et aux fichiers écrits par numpy (fixtures) ;
//! - q64 (4,L,L,L,L,4) ; tableaux f64, f32, u32 ; écriture atomique ;
//! - aller-retour c16 ↔ q64 EXACT (vérifié), lecture de `.npy.zst`, refus hors SU(2) à 1e-12 ;
//! - lecture EN PLACE des dumps de su2_hip de production (gpu_port/ref_amd/p2/L6).
//!
//! Chaque test imprime `PASS <nom>`.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::{ChampHote, Erreur};
use kvu::heritage::su2_hb::{self, quat_to_mat};
use kvu::stockage::compression;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::npy::{self, FormatChamp};
use std::fs;
use std::path::Path;
use support::*;

fn libre() -> GardeSsd {
    GardeSsd::nouvelle(0.0)
}

fn octets(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap()
}

// ------------------------------------------------------------------------------------------
// c16 : à l'octet
// ------------------------------------------------------------------------------------------

#[test]
fn c16_identique_a_write_npy_quat_au_bit() {
    let sc = Scratch::nouveau("npy_c16");
    for (l, graine) in [(2usize, 1u64), (4, 2), (6, 3)] {
        let c = champ_aleatoire(l, graine);
        npy::ecrire_champ(&sc.p("mien.npy"), &c, FormatChamp::C16, &libre()).unwrap();
        su2_hb::write_npy_quat(&sc.p("heritage.npy"), l, &c.q).unwrap();
        assert_eq!(octets(&sc.p("mien.npy")), octets(&sc.p("heritage.npy")), "L = {l}");
        // aucun fichier temporaire résiduel
        assert!(!sc.p("mien.npy.part").exists());
    }
    println!("PASS c16 identique à write_npy_quat au bit (L = 2, 4, 6)");
}

#[test]
fn c16_et_tableaux_identiques_aux_fichiers_de_numpy() {
    let sc = Scratch::nouveau("npy_numpy");
    // champ de Hurwitz L = 2
    let c = champ_hurwitz(2);
    npy::ecrire_champ(&sc.p("c16.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    npy::ecrire_champ(&sc.p("q64.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    assert_eq!(
        octets(&sc.p("c16.npy")),
        octets(&donnees("c16_L2.npy")),
        "c16 contre numpy"
    );
    assert_eq!(
        octets(&sc.p("q64.npy")),
        octets(&donnees("q64_L2.npy")),
        "q64 contre numpy"
    );
    // lecture des fixtures de numpy
    let lu = npy::lire_champ(&donnees("c16_L2.npy")).unwrap();
    assert_eq!(lu, c);
    assert_eq!(npy::lire_champ(&donnees("q64_L2.npy")).unwrap(), c);

    // tableaux
    npy::ecrire_f64(
        &sc.p("a.npy"),
        &[3, 2],
        &[0.5, -1.25, 3.0, 4e-10, -0.0, 1e300],
        &libre(),
    )
    .unwrap();
    assert_eq!(octets(&sc.p("a.npy")), octets(&donnees("f64_3x2.npy")));
    npy::ecrire_f32(&sc.p("b.npy"), &[5], &[1.5, -2.25, 0.1, 3e38, -0.0], &libre()).unwrap();
    assert_eq!(octets(&sc.p("b.npy")), octets(&donnees("f32_5.npy")));
    let u: Vec<u32> = (0..8u32).map(|i| i.wrapping_mul(123_456_789) % 4_294_967_291).collect();
    npy::ecrire_u32(&sc.p("c.npy"), &[2, 2, 2], &u, &libre()).unwrap();
    assert_eq!(octets(&sc.p("c.npy")), octets(&donnees("u32_2x2x2.npy")));
    npy::ecrire_f64(&sc.p("d.npy"), &[], &[2.5], &libre()).unwrap();
    assert_eq!(octets(&sc.p("d.npy")), octets(&donnees("f64_scalaire.npy")));

    // lecture des tableaux de numpy
    let (forme, v) = npy::lire_f64(&donnees("f64_3x2.npy")).unwrap();
    assert_eq!(forme, [3, 2]);
    assert_eq!(v[3], 4e-10);
    assert!(v[4].is_sign_negative() && v[4] == 0.0);
    let (forme, v) = npy::lire_f32(&donnees("f32_5.npy")).unwrap();
    assert_eq!((forme, v[2]), (vec![5], 0.1f32));
    assert_eq!(npy::lire_u32(&donnees("u32_2x2x2.npy")).unwrap().1, u);
    let (forme, v) = npy::lire_f64(&donnees("f64_scalaire.npy")).unwrap();
    assert_eq!((forme.len(), v), (0, vec![2.5]));
    // un f32 se lit en f64, sans perte
    let (_, v) = npy::lire_f64(&donnees("f32_5.npy")).unwrap();
    assert_eq!(v[2], f64::from(0.1f32));
    println!("PASS c16, q64 et tableaux identiques à l'octet aux fichiers écrits par numpy");
}

#[test]
fn numpy_relit_nos_fichiers() {
    if !numpy_disponible() {
        println!("SKIP numpy_relit_nos_fichiers (python3 + numpy absents : ne vaut pas vert)");
        return;
    }
    let sc = Scratch::nouveau("npy_pyload");
    let c = champ_aleatoire(4, 11);
    npy::ecrire_champ(&sc.p("c.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    npy::ecrire_champ(&sc.p("q.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    npy::ecrire_f64(&sc.p("f.npy"), &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.5], &libre()).unwrap();
    let code = format!(
        "import numpy as np\n\
         c=np.load(r'{c}'); q=np.load(r'{q}'); f=np.load(r'{f}')\n\
         assert c.shape==(4,4,4,4,4,2,2) and c.dtype==np.complex128, (c.shape,c.dtype)\n\
         assert q.shape==(4,4,4,4,4,4) and q.dtype==np.float64\n\
         assert f.shape==(2,3) and f[1,2]==6.5\n\
         # c16 reconstruit depuis q64 : [[a0+i a3, a2+i a1], [-a2+i a1, a0-i a3]]\n\
         a0,a1,a2,a3=q[...,0],q[...,1],q[...,2],q[...,3]\n\
         assert np.array_equal(c[...,0,0], a0+1j*a3) and np.array_equal(c[...,0,1], a2+1j*a1)\n\
         assert np.array_equal(c[...,1,0], -a2+1j*a1) and np.array_equal(c[...,1,1], a0-1j*a3)\n\
         u=c@np.conj(np.swapaxes(c,-1,-2)); assert abs(u-np.eye(2)).max()<1e-12\n\
         print('numpy ok')",
        c = sc.p("c.npy").display(),
        q = sc.p("q.npy").display(),
        f = sc.p("f.npy").display()
    );
    let o = std::process::Command::new("python3")
        .args(["-c", &code])
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "numpy refuse nos fichiers : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    println!("PASS numpy relit nos fichiers (c16 unitaire, q64 cohérent avec c16)");
}

// ------------------------------------------------------------------------------------------
// Aller-retour exact
// ------------------------------------------------------------------------------------------

#[test]
fn aller_retour_q64_c16_exact() {
    let sc = Scratch::nouveau("npy_rt");
    let c = champ_aleatoire(4, 7);
    npy::ecrire_champ(&sc.p("c16.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    npy::ecrire_champ(&sc.p("q64.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    assert_eq!(
        fs::metadata(sc.p("c16.npy")).unwrap().len(),
        npy::octets_champ(4, FormatChamp::C16)
    );
    assert_eq!(
        fs::metadata(sc.p("q64.npy")).unwrap().len(),
        npy::octets_champ(4, FormatChamp::Q64)
    );
    // moitié de la taille (en-têtes mis à part)
    let (g, p) = (
        fs::metadata(sc.p("c16.npy")).unwrap().len(),
        fs::metadata(sc.p("q64.npy")).unwrap().len(),
    );
    assert_eq!(g - 128, 2 * (p - 128));

    // lecture : les deux formats donnent exactement le même champ
    assert_eq!(npy::lire_champ(&sc.p("c16.npy")).unwrap(), c);
    assert_eq!(npy::lire_champ(&sc.p("q64.npy")).unwrap(), c);

    // conversion en flux c16 -> q64 (vérifiée) : identique à l'écriture directe
    let r = npy::convertir_champ(
        &sc.p("c16.npy"),
        &sc.p("conv_q64.npy"),
        FormatChamp::Q64,
        &libre(),
        true,
    )
    .unwrap();
    assert_eq!(octets(&sc.p("conv_q64.npy")), octets(&sc.p("q64.npy")));
    assert_eq!(r.octets, p);
    assert_eq!(r.sha256.as_deref().unwrap(), sha256sum(&sc.p("q64.npy")));
    // et q64 -> c16 : retour à l'octet
    npy::convertir_champ(
        &sc.p("conv_q64.npy"),
        &sc.p("conv_c16.npy"),
        FormatChamp::C16,
        &libre(),
        true,
    )
    .unwrap();
    assert_eq!(octets(&sc.p("conv_c16.npy")), octets(&sc.p("c16.npy")));
    // convertir vers le même format est refusé
    assert!(npy::convertir_champ(&sc.p("c16.npy"), &sc.p("x.npy"), FormatChamp::C16, &libre(), true).is_err());
    assert!(!sc.p("x.npy").exists());
    // en place : le remplacement est un renommage, l'original reste lisible jusqu'au bout
    npy::convertir_champ(&sc.p("c16.npy"), &sc.p("c16.npy"), FormatChamp::Q64, &libre(), true).unwrap();
    assert_eq!(octets(&sc.p("c16.npy")), octets(&sc.p("q64.npy")));
    assert!(!sc.p("c16.npy.part").exists());
    println!(
        "PASS aller-retour q64 <-> c16 exact à l'octet (L = 4, {} octets -> {})",
        g, p
    );
}

#[test]
fn conversion_inexacte_refusee_original_intact() {
    // Un c16 « presque SU(2) » : U11 diffère de conj(U00) de 1 ulp. Il passe la tolérance 1e-12 mais le
    // q64 réencodé ne redonne PAS ses octets : la conversion doit être refusée et rien ne doit changer.
    let sc = Scratch::nouveau("npy_inexact");
    let c = champ_aleatoire(2, 5);
    npy::ecrire_champ(&sc.p("src.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    let mut b = octets(&sc.p("src.npy"));
    let entete = su2_hb::npy_header(2).len();
    // lien 0 : re11 = octets [entete+48..entete+56]
    let o = entete + 48;
    let v = f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    b[o..o + 8].copy_from_slice(&f64::from_bits(v.to_bits() ^ 1).to_le_bytes());
    fs::write(sc.p("src.npy"), &b).unwrap();
    assert!(
        npy::lire_champ(&sc.p("src.npy")).is_ok(),
        "défaut de 1 ulp : toléré à la lecture"
    );
    let avant = sha256sum(&sc.p("src.npy"));
    let e = npy::convertir_champ(&sc.p("src.npy"), &sc.p("src.npy"), FormatChamp::Q64, &libre(), true).unwrap_err();
    assert!(matches!(e, Erreur::Validation(_)), "{e}");
    assert_eq!(sha256sum(&sc.p("src.npy")), avant, "l'original a été modifié");
    assert!(!sc.p("src.npy.part").exists(), "le .part doit être supprimé");
    println!("PASS conversion inexacte refusée, original intact, aucun .part");
}

// ------------------------------------------------------------------------------------------
// zstd
// ------------------------------------------------------------------------------------------

#[test]
fn zstd_lecture_et_compression() {
    assert!(compression::disponible(), "zstd doit être installé");
    let sc = Scratch::nouveau("npy_zstd");
    let c = champ_aleatoire(4, 9);
    npy::ecrire_champ(&sc.p("q.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    let sha_avant = sha256sum(&sc.p("q.npy"));
    let z = compression::compresser(&sc.p("q.npy")).unwrap();
    assert_eq!(z, sc.p("q.npy.zst"));
    assert!(!sc.p("q.npy").exists(), "l'original est supprimé après zstd -t");
    assert!(!sc.p("q.npy.zst.part").exists());
    compression::tester(&z).unwrap();
    // lecture directe du .npy.zst
    assert_eq!(npy::lire_champ(&z).unwrap(), c);
    assert_eq!(npy::inspecter_champ(&z).unwrap(), (FormatChamp::Q64, 4));
    // décompression en mémoire = original
    let brut = compression::decompresser_en_memoire(&z).unwrap();
    assert_eq!(kvu::heritage::manifest::sha256_hex(&brut), sha_avant);
    // un c16 compressé se convertit en q64 en flux
    npy::ecrire_champ(&sc.p("c.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    let zc = compression::compresser(&sc.p("c.npy")).unwrap();
    npy::convertir_champ(&zc, &sc.p("c_q64.npy"), FormatChamp::Q64, &libre(), true).unwrap();
    assert_eq!(npy::lire_champ(&sc.p("c_q64.npy")).unwrap(), c);
    // refus d'écraser un .zst existant
    npy::ecrire_champ(&sc.p("q.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    assert!(compression::compresser(&sc.p("q.npy")).is_err());
    assert!(sc.p("q.npy").exists(), "l'original survit à un refus");
    println!("PASS zstd : lecture à la volée, zstd -t avant de supprimer l'original, refus d'écraser");
}

#[test]
fn zstd_corrompu_ou_tronque_detecte() {
    let sc = Scratch::nouveau("npy_zstd_ko");
    let c = champ_aleatoire(4, 10);
    npy::ecrire_champ(&sc.p("q.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
    let z = compression::compresser(&sc.p("q.npy")).unwrap();
    let b = octets(&z);
    // tronqué
    fs::write(sc.p("trop_court.npy.zst"), &b[..b.len() / 2]).unwrap();
    assert!(npy::lire_champ(&sc.p("trop_court.npy.zst")).is_err());
    assert!(compression::tester(&sc.p("trop_court.npy.zst")).is_err());
    // un octet retourné au milieu
    let mut mauvais = b.clone();
    let m = mauvais.len() / 2;
    mauvais[m] ^= 0x40;
    fs::write(sc.p("corrompu.npy.zst"), &mauvais).unwrap();
    assert!(npy::lire_champ(&sc.p("corrompu.npy.zst")).is_err());
    println!("PASS zstd tronqué et corrompu détectés");
}

// ------------------------------------------------------------------------------------------
// Refus hors SU(2), fichiers mal formés
// ------------------------------------------------------------------------------------------

#[test]
fn hors_su2_refuse_a_1e12() {
    let sc = Scratch::nouveau("npy_hors");
    let c = champ_aleatoire(2, 3);
    // écriture : un lien de norme 1 + 1e-9 est refusé, et rien n'est créé
    let mut mauvais = c.clone();
    for v in mauvais.q[8..12].iter_mut() {
        *v *= 1.0 + 1e-9;
    }
    let e = npy::ecrire_champ(&sc.p("m.npy"), &mauvais, FormatChamp::C16, &libre()).unwrap_err();
    assert!(matches!(e, Erreur::Validation(_)), "{e}");
    assert!(!sc.p("m.npy").exists() && !sc.p("m.npy.part").exists());
    // NaN refusé aussi
    let mut nan = c.clone();
    nan.q[5] = f64::NAN;
    assert!(npy::ecrire_champ(&sc.p("n.npy"), &nan, FormatChamp::Q64, &libre()).is_err());
    // défaut 1e-13 : accepté
    let mut peu = c.clone();
    for v in peu.q[0..4].iter_mut() {
        *v *= 1.0 + 1e-13;
    }
    npy::ecrire_champ(&sc.p("p.npy"), &peu, FormatChamp::Q64, &libre()).unwrap();
    // renormalisation en double précision (cas d'un champ calculé en F32)
    let mut f32ish = c.clone();
    for (i, v) in f32ish.q.iter_mut().enumerate() {
        *v *= 1.0 + 6e-8 * ((i % 7) as f64 - 3.0) / 3.0;
    }
    assert!(npy::ecrire_champ(&sc.p("f.npy"), &f32ish, FormatChamp::C16, &libre()).is_err());
    npy::renormaliser(&mut f32ish).unwrap();
    npy::ecrire_champ(&sc.p("f.npy"), &f32ish, FormatChamp::C16, &libre()).unwrap();

    // lecture : fichiers fabriqués à la main
    let l = 2usize;
    let n = 4 * l.pow(4);
    let mut q64 = npy::entete_champ(l, FormatChamp::Q64);
    for i in 0..n {
        let q: [f64; 4] = if i == 17 {
            [1.0, 1e-5, 0.0, 0.0]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };
        for v in q {
            q64.extend_from_slice(&v.to_le_bytes());
        }
    }
    fs::write(sc.p("hors.npy"), &q64).unwrap();
    let e = npy::lire_champ(&sc.p("hors.npy")).unwrap_err();
    assert!(e.to_string().contains("lien 17 hors SU(2)"), "{e}");
    // c16 dont U11 n'est pas conj(U00)
    let mut c16 = npy::entete_champ(l, FormatChamp::C16);
    for _ in 0..n {
        for v in quat_to_mat([1.0, 0.0, 0.0, 0.0]) {
            c16.extend_from_slice(&v.to_le_bytes());
        }
    }
    let o = c16.len() - 64 + 6 * 8; // re11 du dernier lien
    c16[o..o + 8].copy_from_slice(&0.9f64.to_le_bytes());
    fs::write(sc.p("hors16.npy"), &c16).unwrap();
    assert!(npy::lire_champ(&sc.p("hors16.npy")).is_err());
    println!("PASS hors SU(2) refusé à 1e-12 (lecture et écriture)");
}

#[test]
fn fichiers_mal_formes() {
    let sc = Scratch::nouveau("npy_mal");
    let c = champ_aleatoire(2, 4);
    npy::ecrire_champ(&sc.p("ok.npy"), &c, FormatChamp::C16, &libre()).unwrap();
    let b = octets(&sc.p("ok.npy"));
    // tronqué
    fs::write(sc.p("tronque.npy"), &b[..b.len() - 10]).unwrap();
    assert!(npy::lire_champ(&sc.p("tronque.npy")).is_err());
    // octets en trop
    let mut trop = b.clone();
    trop.push(0);
    fs::write(sc.p("trop.npy"), &trop).unwrap();
    assert!(npy::lire_champ(&sc.p("trop.npy")).is_err());
    // pas du NPY
    fs::write(sc.p("txt.npy"), b"bonjour").unwrap();
    assert!(npy::lire_champ(&sc.p("txt.npy")).is_err());
    // mauvaise forme : un tableau f64 n'est pas un champ
    npy::ecrire_f64(&sc.p("t.npy"), &[3, 3], &[0.0; 9], &libre()).unwrap();
    assert!(npy::lire_champ(&sc.p("t.npy")).is_err());
    // fortran_order refusé
    let mut f = b.clone();
    let i = f.windows(5).position(|w| w == b"False").unwrap();
    f.splice(i..i + 5, *b"True ");
    fs::write(sc.p("fortran.npy"), &f).unwrap();
    assert!(npy::lire_champ(&sc.p("fortran.npy")).is_err());
    // fichier absent
    assert!(matches!(npy::lire_champ(&sc.p("absent.npy")), Err(Erreur::Io { .. })));
    // tableaux : mauvais type et mauvaise forme
    assert!(npy::lire_u32(&sc.p("t.npy")).is_err());
    assert!(npy::lire_f32(&sc.p("t.npy")).is_err());
    assert!(npy::ecrire_f64(&sc.p("x.npy"), &[2, 2], &[1.0, 2.0, 3.0], &libre()).is_err());
    assert!(!sc.p("x.npy").exists());
    println!("PASS fichiers mal formés refusés (tronqué, trop long, non NPY, fortran, type, forme)");
}

#[test]
fn ecriture_atomique_sans_residu() {
    let sc = Scratch::nouveau("npy_atom");
    let c = champ_aleatoire(2, 8);
    // répertoire de destination absent : erreur d'E/S, rien de créé
    let e = npy::ecrire_champ(&sc.p("absent/x.npy"), &c, FormatChamp::C16, &libre()).unwrap_err();
    assert!(matches!(e, Erreur::Io { .. }));
    // réécriture d'un fichier existant : remplacement atomique
    npy::ecrire_f64(&sc.p("a.npy"), &[2], &[1.0, 2.0], &libre()).unwrap();
    npy::ecrire_f64(&sc.p("a.npy"), &[3], &[1.0, 2.0, 3.0], &libre()).unwrap();
    assert_eq!(npy::lire_f64(&sc.p("a.npy")).unwrap().1, [1.0, 2.0, 3.0]);
    let noms: Vec<String> = fs::read_dir(&sc.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(noms, ["a.npy"], "résidus : {noms:?}");
    println!("PASS écriture atomique sans résidu");
}

// ------------------------------------------------------------------------------------------
// Dumps de production lus EN PLACE
// ------------------------------------------------------------------------------------------

#[test]
fn dumps_de_production_lus_en_place() {
    let dir = Path::new("/chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/ref_amd/p2/L6");
    if !dir.is_dir() {
        println!(
            "SKIP dumps_de_production_lus_en_place ({} absent sur cette machine : ne vaut pas vert)",
            dir.display()
        );
        return;
    }
    let sc = Scratch::nouveau("npy_prod");
    let mut fichiers: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "npy"))
        .collect();
    fichiers.sort();
    assert!(fichiers.len() >= 20, "attendu 20 dumps, vu {}", fichiers.len());
    let mut identiques = 0;
    for f in &fichiers {
        let c = npy::lire_champ(f).unwrap_or_else(|e| panic!("{} : {e}", f.display()));
        assert_eq!(c.l, 6);
        assert!(c.defaut_unitarite() < 1e-12);
        // même lecture que l'héritage
        let (l, q) = su2_hb::read_npy_quat(f).unwrap();
        assert_eq!((l, &q), (6, &c.q));
        // notre écriture c16 redonne le fichier de su2_hip d'origine ?
        npy::ecrire_champ(&sc.p("r.npy"), &c, FormatChamp::C16, &libre()).unwrap();
        if octets(&sc.p("r.npy")) == octets(f) {
            identiques += 1;
        }
        // et l'aller-retour q64 est exact
        npy::ecrire_champ(&sc.p("r_q.npy"), &c, FormatChamp::Q64, &libre()).unwrap();
        assert_eq!(npy::lire_champ(&sc.p("r_q.npy")).unwrap(), c);
    }
    assert_eq!(
        identiques,
        fichiers.len(),
        "notre c16 doit redonner à l'octet les dumps de su2_hip"
    );
    println!(
        "PASS dumps de production (ref_amd/p2/L6) : {} lus en place, réécrits c16 identiques à l'octet",
        fichiers.len()
    );
}

#[test]
fn champ_hote_froid_c16_q64() {
    let sc = Scratch::nouveau("npy_froid");
    let c = ChampHote::froid(4).unwrap();
    for f in [FormatChamp::C16, FormatChamp::Q64] {
        npy::ecrire_champ(&sc.p("f.npy"), &c, f, &libre()).unwrap();
        assert_eq!(npy::lire_champ(&sc.p("f.npy")).unwrap(), c);
    }
    println!("PASS champ froid c16 et q64");
}

// ------------------------------------------------------------------------------------------
// Mesures (rapportées, non gatées) : cargo test --release -p kvu --test stockage_npy -- --ignored --nocapture mesures_debit
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "mesure"]
fn mesures_debit() {
    use std::time::Instant;
    let sc = Scratch::nouveau("npy_mesure");
    let l = 24usize;
    let c = champ_aleatoire(l, 77);
    let mo = |o: u64| o as f64 / 1e6;
    let debit = |o: u64, t: f64| o as f64 / 1e6 / t;
    println!(
        "MESURE L = {l} : c16 = {:.1} Mo, q64 = {:.1} Mo",
        mo(npy::octets_champ(l, FormatChamp::C16)),
        mo(npy::octets_champ(l, FormatChamp::Q64))
    );
    for f in [FormatChamp::C16, FormatChamp::Q64] {
        let p = sc.p(&format!("c.{}.npy", f.nom()));
        let t = Instant::now();
        let r = npy::ecrire_champ_hache(&p, &c, f, &libre()).unwrap();
        let te = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let lu = npy::lire_champ(&p).unwrap();
        let tl = t.elapsed().as_secs_f64();
        assert_eq!(lu, c);
        let t = Instant::now();
        let h = kvu::heritage::manifest::sha256_fichier(&p).unwrap();
        let th = t.elapsed().as_secs_f64();
        assert_eq!(Some(h), r.sha256);
        println!(
            "MESURE {} : écriture+sha256 {:.0} Mo/s, lecture+validation {:.0} Mo/s, sha256 seul {:.0} Mo/s",
            f.nom(),
            debit(r.octets, te),
            debit(r.octets, tl),
            debit(r.octets, th)
        );
    }
    let t = Instant::now();
    let r = npy::convertir_champ(&sc.p("c.c16.npy"), &sc.p("conv.npy"), FormatChamp::Q64, &libre(), true).unwrap();
    println!(
        "MESURE conversion c16 -> q64 vérifiée (lecture + écriture + relecture + sha256) : {:.2} s, {:.0} Mo/s de c16",
        t.elapsed().as_secs_f64(),
        debit(2 * r.octets, t.elapsed().as_secs_f64())
    );
    for f in ["c.c16.npy", "c.q64.npy"] {
        let p = sc.p(f);
        let avant = fs::metadata(&p).unwrap().len();
        let t = Instant::now();
        let z = compression::compresser(&p).unwrap();
        let apres = fs::metadata(&z).unwrap().len();
        println!(
            "MESURE zstd -3 -T4 sur {f} : {:.2} s, {:.1} Mo -> {:.1} Mo (ratio {:.3})",
            t.elapsed().as_secs_f64(),
            mo(avant),
            mo(apres),
            apres as f64 / avant as f64
        );
        let t = Instant::now();
        npy::lire_champ(&z).unwrap();
        println!("MESURE lecture directe de {f}.zst : {:.2} s", t.elapsed().as_secs_f64());
    }
}
