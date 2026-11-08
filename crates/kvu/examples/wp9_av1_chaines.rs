//! WP9-AV1 : chaînes d'étalonnage P et R (L = 4 ou 8) à des β et des graines choisis, avec le MÊME moteur et le même protocole que
//! `kvu rg etalonner --protocole wp9` (`etalonner::etalonner`, départ chaud, pas = 1 bain thermique + 4 OR, erreur de Sokal c = 6).
//! Sert à l'instrument AV1 (nœuds de la grille raffinée, chaînes aux milieux, rejeu de chaînes anciennes) ; ce n'est pas le
//! binaire de production et il n'écrit que le fichier de sortie demandé, refusé s'il existe.
//!
//! Usage : `wp9_av1_chaines --L 4|8 --betas b1,b2,… --n-chaines N [--independante] --graine-base G --therm T --mesures M
//!          --fils F --role TEXTE --sortie FICHIER.json`
//! Graine d'une chaîne : `G + 100·i + c` (i = rang du β dans la liste ; c = 1…N, ou c = 9 avec `--independante`), comme `rg etalonner`.
//! Aucune valeur par défaut pour le protocole (T, M, G) : une dérive silencieuse du protocole est impossible.

use kvu::rg::flot::etalonner::{en_json, etalonner, ParamEtalonnage};
use std::collections::HashMap;
use std::io::Write;

fn principal() -> Result<(), String> {
    let mut args: HashMap<String, String> = HashMap::new();
    let mut independante = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--independante" => independante = true,
            "--L" | "--betas" | "--n-chaines" | "--graine-base" | "--therm" | "--mesures" | "--fils" | "--role" | "--sortie" => {
                let v = it.next().ok_or_else(|| format!("{a} exige une valeur"))?;
                if args.insert(a.clone(), v).is_some() {
                    return Err(format!("{a} donné deux fois"));
                }
            }
            _ => return Err(format!("option inconnue : {a}")),
        }
    }
    let prendre = |k: &str| args.get(k).cloned().ok_or_else(|| format!("{k} est obligatoire"));
    let l: usize = prendre("--L")?.parse().map_err(|e| format!("--L : {e}"))?;
    if l != 4 && l != 8 {
        return Err("--L : 4 ou 8 (volumes grossiers de WP9)".into());
    }
    let betas = prendre("--betas")?
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| format!("--betas « {s} » : {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    if betas.is_empty() || betas.iter().any(|b| !b.is_finite() || *b <= 0.0) {
        return Err("--betas : liste de réels > 0".into());
    }
    if betas.windows(2).any(|w| w[0] >= w[1]) {
        return Err("--betas : strictement croissants".into());
    }
    let n_chaines: usize = prendre("--n-chaines")?.parse().map_err(|e| format!("--n-chaines : {e}"))?;
    let graine_base: u64 = prendre("--graine-base")?.parse().map_err(|e| format!("--graine-base : {e}"))?;
    let therm: usize = prendre("--therm")?.parse().map_err(|e| format!("--therm : {e}"))?;
    let mesures: usize = prendre("--mesures")?.parse().map_err(|e| format!("--mesures : {e}"))?;
    let fils: usize = prendre("--fils")?.parse().map_err(|e| format!("--fils : {e}"))?;
    let role = prendre("--role")?;
    let sortie = prendre("--sortie")?;
    if n_chaines == 0 || n_chaines > 8 {
        return Err("--n-chaines : 1 à 8".into());
    }
    let par = ParamEtalonnage { l, betas, n_chaines, therm, mesures, graine_base, independante, facteur_beta: 1.0 };
    let chaines = etalonner(&par, fils).map_err(|e| format!("étalonnage : {e:?}"))?;
    let mut doc = en_json(&par, &chaines).map_err(|e| format!("JSON : {e:?}"))?;
    doc["role_av1"] = serde_json::json!(role);
    let texte = serde_json::to_string_pretty(&doc).map_err(|e| format!("sérialisation : {e}"))?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&sortie)
        .map_err(|e| format!("{sortie} : {e} (une sortie existante est refusée)"))?;
    f.write_all(texte.as_bytes()).map_err(|e| format!("{sortie} : {e}"))?;
    f.write_all(b"\n").map_err(|e| format!("{sortie} : {e}"))?;
    Ok(())
}

fn main() {
    if let Err(e) = principal() {
        eprintln!("wp9_av1_chaines : {e}");
        std::process::exit(2);
    }
}
