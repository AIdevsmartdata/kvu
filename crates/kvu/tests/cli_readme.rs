//! Cohérence du GUIDE.md avec la ligne de commande réelle : chaque commande `kevinotron …` / `kvu …` écrite dans un
//! bloc de code existe, et chacune de ses options `--xxx` figure dans l'aide de la sous-commande. Les lignes qui portent
//! `--dry-run` sont EXÉCUTÉES (ce sont des plans, sans effet).

use std::path::Path;
use std::process::Command;

fn kvu(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).output().unwrap();
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

/// Lignes de commande des blocs ``` du README (continuations `\` recollées).
fn commandes_du_readme() -> Vec<String> {
    let t = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("GUIDE.md")).unwrap();
    let (mut en_bloc, mut out, mut courante) = (false, Vec::new(), String::new());
    for l in t.lines() {
        if l.trim_start().starts_with("```") {
            en_bloc = !en_bloc;
            continue;
        }
        if !en_bloc {
            continue;
        }
        let l = l.split(" #").next().unwrap().trim_end();
        let (corps, suite) = match l.strip_suffix('\\') {
            Some(c) => (c.trim_end(), true),
            None => (l, false),
        };
        if !courante.is_empty()
            || corps.trim_start().starts_with("kevinotron ")
            || corps.trim_start().starts_with("kvu ")
        {
            courante.push(' ');
            courante.push_str(corps.trim());
        }
        if !suite && !courante.is_empty() {
            out.push(courante.trim().to_string());
            courante.clear();
        }
    }
    out
}

fn options_de(ligne: &str) -> Vec<String> {
    ligne
        .split(|c: char| c.is_whitespace() || "[]|(),".contains(c))
        .filter(|t| {
            t.starts_with("--") && t.len() > 2 && t.chars().skip(2).all(|c| c.is_ascii_alphanumeric() || c == '-')
        })
        .map(String::from)
        .collect()
}

#[test]
fn les_commandes_du_readme_existent_avec_leurs_options() {
    let cmds = commandes_du_readme();
    assert!(
        cmds.len() >= 12,
        "trop peu de commandes extraites du README : {}",
        cmds.len()
    );
    let mut verifiees = 0;
    for ligne in &cmds {
        // `kevinotron ; kevinotron` : on ne traite que la première commande de la ligne
        let premiere = ligne.split(';').next().unwrap();
        let mots: Vec<&str> = premiere.split_whitespace().collect();
        let sous = match mots.get(1) {
            Some(s) if !s.starts_with('-') && !s.starts_with('<') => *s,
            _ => continue, // « kevinotron --list-modes », « kevinotron <sous-commande> … »
        };
        let (rc, aide) = {
            // sous-sous-commande éventuelle (su2 gen, store du, cond chaine…)
            let deux = mots
                .get(2)
                .filter(|m| m.chars().all(|c| c.is_ascii_lowercase() || c == '-'));
            match deux {
                Some(d) => {
                    let (rc, a) = kvu(&[sous, d, "--help"]);
                    if rc == 0 {
                        (rc, a)
                    } else {
                        kvu(&[sous, "--help"])
                    }
                }
                None => kvu(&[sous, "--help"]),
            }
        };
        assert_eq!(rc, 0, "« {premiere} » : sous-commande {sous} inconnue\n{aide}");
        // aide de la sous-commande ET de ses sous-sous-commandes, pour les options
        let mut toute_aide = aide.clone();
        for extra in [
            "gen", "test-hb", "du", "gc", "add", "run", "list", "wilson", "flot", "polyakov", "chaine", "mesurer",
            "su2", "cond", "fibre", "obs",
        ] {
            if mots.contains(&extra) {
                toute_aide.push_str(&kvu(&[sous, extra, "--help"]).1);
            }
        }
        for o in options_de(premiere) {
            assert!(
                toute_aide.contains(&o),
                "« {premiere} » : l'option {o} n'existe pas dans l'aide de {sous}\n{toute_aide}"
            );
        }
        // les plans sont exécutés pour de bon
        if premiere.contains("--dry-run") {
            let args: Vec<&str> = mots[1..]
                .iter()
                .copied()
                .filter(|m| !m.starts_with('[') && !m.ends_with(']'))
                .collect();
            let (rc, s) = kvu(&args);
            assert_eq!(rc, 0, "« {premiere} » (--dry-run) : {s}");
        }
        verifiees += 1;
    }
    assert!(verifiees >= 10, "seulement {verifiees} commandes vérifiées");
    println!("PASS readme_coherent ({verifiees} commandes)");
}

#[test]
fn le_readme_mentionne_toutes_les_sous_commandes_et_les_dix_codes() {
    let t = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("GUIDE.md")).unwrap();
    for s in kvu::cli::SOUS_COMMANDES {
        assert!(t.contains(&format!("`{s}")), "sous-commande {s} absente du README");
    }
    for code in 0..=9 {
        assert!(
            t.contains(&format!("| {code} |")),
            "code de sortie {code} absent du README"
        );
    }
}
