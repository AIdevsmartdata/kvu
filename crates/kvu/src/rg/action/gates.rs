//! Gates de contrôle du module action : RG-ACTION-ENUM, RG-ACTION-SD1, RG-ACTION-FERME,
//! RG-ACTION-WILSON-b<β>, RG-ACTION-MUT.
//!
//! Une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate ; le verdict global est `tout_vert`.
//! Les données de RG-ACTION-FERME sont partagées avec RG-ACTION-MUT (cache interne).

use super::boucles::{
    agrafes_lien, agrafes_lien_variante, Famille, TableFormes, TypeBoucle, VarianteAgrafes,
};
use super::cli::decouvrir_chaines;
use super::estimation::{estimer, Estimateur, StatsGroupe, Valeur};
use super::gibbs::{Gibbs, ParamGibbs};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::groupes::gates::{LigneGate, Verdict};
use crate::heritage::su2_hb::qmul;
use crate::rg::flot::sd::{beta_sd, stats_sd, ModeRapport, RapportBessel};
use crate::rg::lissage::Voisinage;
use crate::stockage::npy::lire_champ;
use std::path::Path;
use std::sync::Mutex;

/// Chemin en cours d'exploration : (pas, site courant, dernier pas, déplacement cumulé dans ℤ⁴).
type CheminEnCours = (Vec<(u8, i8)>, usize, (u8, i8), [isize; 4]);

/// Contexte des gates.
pub struct Contexte {
    pub grandl: Option<String>,
    pub seul: Vec<String>,
    pub sans_mutants: bool,
    pub json: bool,
    pub fils: usize,
}

/// Données partagées entre RG-ACTION-FERME et RG-ACTION-MUT.
#[derive(Clone)]
struct DonneesFerme {
    configs: Vec<ChampHote>,
    stats_complet: Vec<StatsGroupe>,
    plafonds: u64,
}

static DONNEES_FERME: Mutex<Option<DonneesFerme>> = Mutex::new(None);

/// Génère et met en cache les configurations et statistiques de RG-ACTION-FERME (T6).
fn donnees_ferme(fils: usize) -> Resultat<DonneesFerme> {
    let mut garde = DONNEES_FERME
        .lock()
        .map_err(|_| Erreur::Config("verrou du cache des données FERME empoisonné".into()))?;
    if let Some(d) = garde.as_ref() {
        return Ok(d.clone());
    }

    let famille = Famille::analyser("P,R,C,T")?;
    let beta_vrai = [1.5, -0.10, 0.05, 0.06];
    let params = ParamGibbs {
        l: 4,
        famille: famille.clone(),
        beta: beta_vrai.to_vec(),
        graine: 2611020103,
        depart_chaud: true,
        sur_relaxations: 1,
    };
    let mut gibbs = Gibbs::nouveau(params)?;
    for _ in 0..300 {
        gibbs.balayage();
    }
    let mut configs = Vec::with_capacity(800);
    for i in 0..800 {
        gibbs.balayage();
        configs.push(gibbs.champ());
        if i < 799 {
            for _ in 0..2 {
                gibbs.balayage();
            }
        }
    }
    let plafonds = gibbs.plafonds();

    let table = TableFormes::construire()?;
    let mut stats_complet = Vec::with_capacity(configs.len());
    for cfg in &configs {
        stats_complet.push(super::estimation::stats_config(
            cfg, &table, &famille, fils,
        )?);
    }
    let donnees = DonneesFerme {
        configs,
        stats_complet,
        plafonds,
    };
    *garde = Some(donnees.clone());
    Ok(donnees)
}

/// Batterie de gates.
pub fn batterie(ctx: &Contexte) -> Resultat<Vec<LigneGate>> {
    let ids = [
        "RG-ACTION-ENUM",
        "RG-ACTION-SD1",
        "RG-ACTION-FERME",
        "RG-ACTION-WILSON-b2.3",
        "RG-ACTION-WILSON-b2.5",
        "RG-ACTION-WILSON-b2.8",
        "RG-ACTION-WILSON-b4",
        "RG-ACTION-MUT",
    ];
    if ctx.seul.iter().any(|p| {
        p.is_empty()
            || !ids
                .iter()
                .any(|id| (!ctx.sans_mutants || *id != "RG-ACTION-MUT") && id.starts_with(p))
    }) {
        return Err(Erreur::Config(
            "--seul : identifiant inconnu ou sélection vide".into(),
        ));
    }
    let actif = |id: &str| ctx.seul.is_empty() || ctx.seul.iter().any(|p| id.starts_with(p));
    let rendre = |id: &str, r: Resultat<LigneGate>| {
        r.unwrap_or_else(|e| LigneGate::nouvelle(Verdict::Fail, id, format!("ECHEC_OUTIL : {e}")))
    };
    let mut lignes = Vec::new();
    if actif(ids[0]) {
        lignes.push(rendre(ids[0], gate_enum()));
    }
    if actif(ids[1]) {
        lignes.push(rendre(ids[1], gate_sd1()));
    }
    if actif(ids[2]) {
        lignes.push(rendre(ids[2], gate_ferme(ctx)));
    }
    for beta in [2.3, 2.5, 2.8, 4.0] {
        let id = format!("RG-ACTION-WILSON-b{beta}");
        if actif(&id) {
            lignes.push(rendre(&id, gate_wilson(ctx, beta)));
        }
    }
    if !ctx.sans_mutants && actif("RG-ACTION-MUT") {
        lignes.push(rendre("RG-ACTION-MUT", gate_mut(ctx)));
    }
    if lignes.is_empty() {
        return Err(Erreur::Config("--seul : sélection vide".into()));
    }
    Ok(lignes)
}

/// Oracle : classification par la séparation cyclique des pas opposés (aucune géométrie
/// de la classification de production). À six pas et trois axes, une chaise a un
/// couple de pas opposés séparés par deux positions ; une torsade les sépare tous de trois.
pub(crate) fn classe_independante(chemin: &[(u8, i8)]) -> usize {
    if chemin.len() == 4 {
        return 0;
    }
    let mut axes = [false; 4];
    for &(d, _) in chemin {
        axes[d as usize] = true;
    }
    if axes.iter().filter(|&&x| x).count() == 2 {
        return 1;
    }
    if (0..6).any(|i| {
        let (a, b) = (chemin[i], chemin[(i + 2) % 6]);
        a.0 == b.0 && a.1 == -b.1
    }) {
        2
    } else {
        3
    }
}

/// Produit explicite de matrices complexes 2×2 : indépendant de produit_chemin,
/// qmul, qdag et du déplacement de production.
pub(crate) fn trace_chemin_independante(
    champ: &ChampHote,
    depart: usize,
    chemin: &[(u8, i8)],
) -> f64 {
    let l = champ.l;
    let mut x = [0usize; 4];
    let mut site = depart;
    for coord in x.iter_mut().rev() {
        *coord = site % l;
        site /= l;
    }
    let mut acc = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]];
    for &(d, sens) in chemin {
        let d = d as usize;
        if sens < 0 {
            x[d] = (x[d] + l - 1) % l;
        }
        let site = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
        let o = 4 * (d * l.pow(4) + site);
        let q = &champ.q[o..o + 4];
        let u = [[(q[0], q[3]), (q[2], q[1])], [(-q[2], q[1]), (q[0], -q[3])]];
        let mut nouveau = [[(0.0, 0.0); 2]; 2];
        for (i, ligne) in nouveau.iter_mut().enumerate() {
            for (j, z) in ligne.iter_mut().enumerate() {
                for (k, &(ar, ai)) in acc[i].iter().enumerate() {
                    let (br, bi) = if sens > 0 {
                        u[k][j]
                    } else {
                        (u[j][k].0, -u[j][k].1)
                    };
                    z.0 += ar * br - ai * bi;
                    z.1 += ar * bi + ai * br;
                }
            }
        }
        acc = nouveau;
        if sens > 0 {
            x[d] = (x[d] + 1) % l;
        }
    }
    (acc[0][0].0 + acc[1][1].0) * 0.5
}

/// Énumération indépendante des boucles distinctes : somme des ½Tr par type.
/// Suit le déplacement cumulé exact (sans modulo) pour décider de la fermeture.
fn enumerer_independante(champ: &ChampHote) -> Resultat<[f64; 4]> {
    let l = champ.l;
    let v = champ.v();
    let vois = Voisinage::nouveau(l);
    let mut sommes = [0.0; 4];
    let mut vues: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    for depart in 0..v {
        for mu in 0..4u8 {
            let premier = (mu, 1i8);
            let pos_initiale = vois.haut(mu as usize, depart);
            let mut pile: Vec<CheminEnCours> = vec![(vec![premier], pos_initiale, premier, {
                let mut de = [0isize; 4];
                de[mu as usize] = 1;
                de
            })];
            while let Some((chemin, pos, dernier, deplacement)) = pile.pop() {
                let longueur = chemin.len();
                let ferme = pos == depart && deplacement == [0, 0, 0, 0];
                if ferme && (longueur == 4 || longueur == 6) {
                    // sans rebroussement cyclique : dernier pas pas inverse du premier
                    let rebroussement = dernier.0 == premier.0 && dernier.1 == -premier.1;
                    if !rebroussement {
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
                            let type_idx = classe_independante(&chemin);
                            sommes[type_idx] += trace_chemin_independante(champ, depart, &chemin);
                        }
                    }
                }
                if longueur < 6 {
                    for d in 0..4u8 {
                        for sig in [1i8, -1] {
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
    Ok(sommes)
}

/// Génère un champ de Haar L=4, graine donnée.
fn champ_haar_l4(graine: u64) -> Resultat<ChampHote> {
    let l = 4;
    let (k0, k1) = ((graine & 0xFFFF_FFFF) as u32, (graine >> 32) as u32);
    let mut champ = ChampHote::froid(l)?;
    for li in 0..4 * champ.v() {
        let q = crate::su2::cpu::haar_q(
            li as u32,
            0xFFFF_FFFF,
            crate::commun::ETIQUETTE_INIT,
            k0,
            k1,
        );
        let o = 4 * li;
        champ.q[o..o + 4].copy_from_slice(&q);
    }
    Ok(champ)
}

/// RG-ACTION-ENUM
fn gate_enum() -> Resultat<LigneGate> {
    let table = TableFormes::construire()?;
    for mu in 0..4 {
        if table.formes(mu, TypeBoucle::P).len() != 6
            || table.formes(mu, TypeBoucle::R).len() != 18
            || table.formes(mu, TypeBoucle::C).len() != 72
            || table.formes(mu, TypeBoucle::T).len() != 24
        {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-ACTION-ENUM",
                "comptes incorrects",
            ));
        }
    }

    let champ = champ_haar_l4(26110201)?;
    let vois = Voisinage::nouveau(champ.l);
    let famille = Famille::analyser("P,R,C,T")?;
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
    let droite = enumerer_independante(&champ)?;
    let mut max_rel = 0.0f64;
    for (i, ty) in TypeBoucle::TOUS.iter().enumerate() {
        let denom = (ty.longueur() as f64 * droite[i].abs()).max(1e-300);
        let rel = (gauche[i] - ty.longueur() as f64 * droite[i]).abs() / denom;
        max_rel = max_rel.max(rel);
        if rel > 1e-12 {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                "RG-ACTION-ENUM",
                format!("type {} : écart relatif {rel:.2e}", ty.nom()),
            ));
        }
    }
    Ok(LigneGate::nouvelle(
        Verdict::Pass,
        "RG-ACTION-ENUM",
        format!("écart relatif max {max_rel:.2e}"),
    ))
}

/// RG-ACTION-SD1
fn gate_sd1() -> Resultat<LigneGate> {
    let fixtures = [
        format!(
            "{}/tests/donnees/rg/cfg_su2_L8_b2.40_s24001.npy",
            env!("CARGO_MANIFEST_DIR")
        ),
        format!(
            "{}/validation/rg/fixtures/cfg_s7_q64.npy",
            env!("CARGO_MANIFEST_DIR")
        ),
        format!(
            "{}/validation/rg/fixtures/cfg_s11_q64.npy",
            env!("CARGO_MANIFEST_DIR")
        ),
        format!(
            "{}/validation/rg/fixtures/cfg_s13_q64.npy",
            env!("CARGO_MANIFEST_DIR")
        ),
    ];
    let famille = Famille::analyser("P")?;
    let table = TableFormes::construire()?;
    let mut max_rel = 0.0f64;
    for fichier in fixtures.iter() {
        let champ = lire_champ(Path::new(fichier))?;
        let stats = super::estimation::stats_config(&champ, &table, &famille, 1)?;
        let groupes = [&stats];
        let sol = super::estimation::resoudre_pl(&groupes, 1, &[0.0], 1)?;
        let stats_sd = stats_sd(&champ)?;
        let b_exact = beta_sd(&stats_sd, RapportBessel::I2SurI1, ModeRapport::Exact)
            .ok_or_else(|| Erreur::Validation("beta_sd échoue sur la fixture".into()))?;
        let rel = (sol.beta[0] - b_exact).abs() / b_exact.abs().max(1e-300);
        max_rel = max_rel.max(rel);
    }
    if max_rel <= 1e-10 {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-ACTION-SD1",
            format!("écart relatif max {max_rel:.2e}"),
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-ACTION-SD1",
            format!("écart relatif max {max_rel:.2e}"),
        ))
    }
}

/// Estime PL avec un rapport de Bessel donné, valeur centrale + jackknife vectoriel.
fn estimer_pl_rapport(
    groupes: &[StatsGroupe],
    n: usize,
    depart: &[f64],
    rapport: RapportBessel,
    fils: usize,
) -> Resultat<Vec<Valeur>> {
    let refs: Vec<&StatsGroupe> = groupes.iter().collect();
    let verifier_convergence =
        |sol: &super::estimation::SolutionPl, groupes: &[&StatsGroupe]| -> Resultat<()> {
            let liens = groupes.iter().map(|g| g.n_liens).sum::<usize>().max(1);
            if !sol.norme_gradient.is_finite() || sol.norme_gradient > 1e-10 * liens as f64 {
                return Err(Erreur::Validation(
                    "estimation mutante non convergée (résidu du gradient)".into(),
                ));
            }
            Ok(())
        };
    let sol_central = super::estimation::resoudre_pl_avec(&refs, n, depart, rapport, fils)?;
    verifier_convergence(&sol_central, &refs)?;
    let beta_central = sol_central.beta;
    let mut replicats = Vec::with_capacity(groupes.len());
    for exclu in 0..groupes.len() {
        let sous: Vec<&StatsGroupe> = refs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != exclu)
            .map(|(_, g)| *g)
            .collect();
        let sol = super::estimation::resoudre_pl_avec(&sous, n, &beta_central, rapport, fils)?;
        verifier_convergence(&sol, &sous)?;
        replicats.push(sol.beta);
    }
    Ok(super::estimation::jackknife_vectoriel(
        &beta_central,
        &replicats,
    ))
}

/// RG-ACTION-FERME
fn gate_ferme(ctx: &Contexte) -> Resultat<LigneGate> {
    let donnees = donnees_ferme(ctx.fils)?;
    let famille = Famille::analyser("P,R,C,T")?;
    let beta_vrai = [1.5, -0.10, 0.05, 0.06];

    // Former blocs:8
    let stats_complet = &donnees.stats_complet;
    let mut groupes = Vec::with_capacity(8);
    let total = stats_complet.len();
    let mut debut = 0;
    for g in 0..8 {
        let taille = total / 8 + if g < total % 8 { 1 } else { 0 };
        let mut groupe = StatsGroupe::default();
        for s in &stats_complet[debut..debut + taille] {
            groupe.fusionner(s);
        }
        groupes.push(groupe);
        debut += taille;
    }

    let res_pl = estimer(&groupes, &famille, Estimateur::Pl, ctx.fils)?;
    let res_sd = estimer(&groupes, &famille, Estimateur::Sd, ctx.fils)?;

    let mut ok = true;
    let mut details = format!(
        "PL: β_P={:.6}±{:.6}",
        res_pl.couplages[0].valeur, res_pl.couplages[0].erreur
    );
    for (i, res) in [&res_pl, &res_sd].iter().enumerate() {
        for (j, b) in beta_vrai.iter().enumerate() {
            let ecart = (res.couplages[j].valeur - b).abs();
            if ecart >= 4.0 * res.couplages[j].erreur {
                ok = false;
                details.push_str(&format!(
                    " | {} γ={} écart {:.3e} ≥ 4σ",
                    if i == 0 { "PL" } else { "SD" },
                    j,
                    ecart
                ));
            }
        }
    }
    // puissance
    if res_pl.couplages[1].erreur >= 0.025
        || res_pl.couplages[2].erreur >= 0.0125
        || res_pl.couplages[3].erreur >= 0.015
        || res_sd.couplages[1].erreur >= 0.025
        || res_sd.couplages[2].erreur >= 0.0125
        || res_sd.couplages[3].erreur >= 0.015
    {
        ok = false;
        details.push_str(" | erreur de puissance trop grande");
    }

    // Troncature famille {P}
    let table = TableFormes::construire()?;
    let famille_p = Famille::analyser("P")?;
    let mut stats_p = Vec::with_capacity(donnees.configs.len());
    for cfg in &donnees.configs {
        stats_p.push(super::estimation::stats_config(
            cfg, &table, &famille_p, ctx.fils,
        )?);
    }
    let mut groupes_p = Vec::with_capacity(8);
    let mut debut = 0;
    for g in 0..8 {
        let taille = total / 8 + if g < total % 8 { 1 } else { 0 };
        let mut groupe = StatsGroupe::default();
        for s in &stats_p[debut..debut + taille] {
            groupe.fusionner(s);
        }
        groupes_p.push(groupe);
        debut += taille;
    }
    let res_p = estimer(&groupes_p, &famille_p, Estimateur::Pl, ctx.fils)?;
    if (res_p.couplages[0].valeur - 1.5).abs() <= 5.0 * res_p.couplages[0].erreur {
        ok = false;
        details.push_str(" | troncature {P} non détectée");
    }

    // Troncature famille {P,R}
    let famille_pr = Famille::analyser("P,R")?;
    let mut stats_pr = Vec::with_capacity(donnees.configs.len());
    for cfg in &donnees.configs {
        stats_pr.push(super::estimation::stats_config(
            cfg,
            &table,
            &famille_pr,
            ctx.fils,
        )?);
    }
    let mut groupes_pr = Vec::with_capacity(8);
    let mut debut = 0;
    for g in 0..8 {
        let taille = total / 8 + if g < total % 8 { 1 } else { 0 };
        let mut groupe = StatsGroupe::default();
        for s in &stats_pr[debut..debut + taille] {
            groupe.fusionner(s);
        }
        groupes_pr.push(groupe);
        debut += taille;
    }
    let res_pr = estimer(&groupes_pr, &famille_pr, Estimateur::Pl, ctx.fils)?;
    let depasse_p = (res_pr.couplages[0].valeur - 1.5).abs() / res_pr.couplages[0].erreur;
    let depasse_r = (res_pr.couplages[1].valeur + 0.10).abs() / res_pr.couplages[1].erreur;
    if depasse_p <= 5.0 && depasse_r <= 5.0 {
        ok = false;
        details.push_str(" | troncature {P,R} non détectée");
    }

    if donnees.plafonds != 0 {
        ok = false;
        details.push_str(&format!(" | plafonds={}", donnees.plafonds));
    } else {
        details.push_str(" | plafonds=0");
    }

    if ok {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-ACTION-FERME",
            details,
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-ACTION-FERME",
            details,
        ))
    }
}

/// RG-ACTION-WILSON-b<β>
fn gate_wilson(ctx: &Contexte, beta: f64) -> Resultat<LigneGate> {
    let id = format!("RG-ACTION-WILSON-b{beta}");
    let Some(grandl) = &ctx.grandl else {
        return Ok(LigneGate::nouvelle(Verdict::Skip, &id, "--grandl absent"));
    };
    let racine = Path::new(grandl);
    if !racine
        .try_exists()
        .map_err(|e| Erreur::io("racine Wilson", e))?
    {
        return Ok(LigneGate::nouvelle(
            Verdict::Skip,
            &id,
            "données Wilson absentes",
        ));
    }
    let motif = format!("L12_b{beta}_c*");
    let chaines = match decouvrir_chaines(racine, Some(&motif)) {
        Ok(c) => c,
        Err(e) => {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                &id,
                format!("données Wilson invalides : {e}"),
            ))
        }
    };

    let total = chaines.iter().map(|c| c.configs.len()).sum::<usize>();
    let assez = chaines.len() >= 2 && total >= 10;
    let famille = Famille::analyser("P,R,C,T")?;
    let table = TableFormes::construire()?;
    let tailles: Vec<usize> = (0..10)
        .map(|g| total / 10 + usize::from(g < total % 10))
        .collect();
    let mut groupes = vec![StatsGroupe::default(); 10];
    let (mut groupe_idx, mut dans_groupe) = (0, 0);
    for chaine in &chaines {
        if chaine.l != 12 || (chaine.beta - beta).abs() > 1e-9 {
            return Ok(LigneGate::nouvelle(
                Verdict::Fail,
                &id,
                "β ou L du manifeste Wilson incohérent",
            ));
        }
        for chemin in &chaine.configs {
            let champ = lire_champ(chemin)?;
            if champ.l != chaine.l {
                return Ok(LigneGate::nouvelle(
                    Verdict::Fail,
                    &id,
                    "côté réel différent du manifeste Wilson",
                ));
            }
            // Même si l'effectif est insuffisant, toutes les données présentes sont
            // validées : une donnée invalide ne doit pas être masquée par SKIP.
            if assez {
                super::estimation::accumuler_config(
                    &champ,
                    &table,
                    &famille,
                    ctx.fils,
                    true,
                    &mut groupes[groupe_idx],
                )?;
                dans_groupe += 1;
                if dans_groupe == tailles[groupe_idx] {
                    groupe_idx += 1;
                    dans_groupe = 0;
                }
            }
        }
    }
    if !assez {
        return Ok(LigneGate::nouvelle(
            Verdict::Skip,
            &id,
            "moins de 2 chaînes ou de 10 configurations",
        ));
    }

    let res_pl = estimer(&groupes, &famille, Estimateur::Pl, ctx.fils)?;
    let res_sd = estimer(&groupes, &famille, Estimateur::Sd, ctx.fils)?;

    let mut ok = true;
    let mut detail = format!(
        "PL β_P={:.6}±{:.6}",
        res_pl.couplages[0].valeur, res_pl.couplages[0].erreur
    );
    for (nom, res) in [("PL", &res_pl), ("SD", &res_sd)] {
        let bp = res.couplages[0].valeur;
        let err_p = res.couplages[0].erreur;
        if (bp / beta - 1.0).abs() >= 0.01 {
            ok = false;
            detail.push_str(&format!(
                " | {nom} β_P/β−1 = {:.3e} ≥ 0,01",
                (bp / beta - 1.0).abs()
            ));
        }
        if err_p / beta >= 0.006 {
            ok = false;
            detail.push_str(&format!(" | {nom} σ(β_P)/β = {:.3e} ≥ 0,006", err_p / beta));
        }
        for (j, gamma) in [(1, "R"), (2, "C"), (3, "T")] {
            let z = (res.couplages[j].valeur / res.couplages[j].erreur).abs();
            if z >= 3.5 {
                ok = false;
                detail.push_str(&format!(" | {nom} z_{gamma} = {z:.2e} ≥ 3,5"));
            }
        }
    }

    if ok {
        Ok(LigneGate::nouvelle(Verdict::Pass, &id, detail))
    } else {
        Ok(LigneGate::nouvelle(Verdict::Fail, &id, detail))
    }
}

/// RG-ACTION-MUT
fn gate_mut(ctx: &Contexte) -> Resultat<LigneGate> {
    // Mutant (a) : ChaiseInversee casse l'identité pour C
    let table = TableFormes::construire()?;
    let famille = Famille::analyser("P,R,C,T")?;
    let champ = champ_haar_l4(26110201)?;
    let vois = Voisinage::nouveau(champ.l);
    let mut gauche_c = 0.0;
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
            let idx_c = famille
                .index(TypeBoucle::C)
                .ok_or_else(|| Erreur::Config("C absent de la famille du mutant".into()))?;
            let u = champ.lien(mu * champ.v() + s);
            gauche_c += qmul(u, agrafes[idx_c])[0];
        }
    }
    let droite = enumerer_independante(&champ)?[2];
    let identite_cassee = (gauche_c - TypeBoucle::C.longueur() as f64 * droite).abs() > 1e-8;

    // Mutant (b) : PL avec I1/I0 sur les données FERME
    let DonneesFerme { stats_complet, .. } = donnees_ferme(ctx.fils)?;
    let famille = Famille::analyser("P,R,C,T")?;
    let n = famille.n();
    let beta_vrai = [1.5, -0.10, 0.05, 0.06];
    // Former blocs:8
    let mut groupes = Vec::with_capacity(8);
    let total = stats_complet.len();
    let mut debut = 0;
    for g in 0..8 {
        let taille = total / 8 + if g < total % 8 { 1 } else { 0 };
        let mut groupe = StatsGroupe::default();
        for s in &stats_complet[debut..debut + taille] {
            groupe.fusionner(s);
        }
        groupes.push(groupe);
        debut += taille;
    }
    let valeurs_mutantes =
        match estimer_pl_rapport(&groupes, n, &beta_vrai, RapportBessel::I1SurI0, ctx.fils) {
            Ok(v) => v,
            Err(e) => {
                return Ok(LigneGate::nouvelle(
                    Verdict::Fail,
                    "RG-ACTION-MUT",
                    format!("mutant I1/I0 non convergé : {e}"),
                ))
            }
        };
    let mutant_rapport_rouge = (0..n).any(|i| {
        let ecart = (valeurs_mutantes[i].valeur - beta_vrai[i]).abs();
        ecart > 4.0 * valeurs_mutantes[i].erreur
    });

    let mutant_rouge = identite_cassee && mutant_rapport_rouge;
    if mutant_rouge {
        Ok(LigneGate::nouvelle(
            Verdict::Pass,
            "RG-ACTION-MUT",
            "mutants rouges ; estimation I1/I0 convergée (centrale et réplicats)",
        ))
    } else {
        Ok(LigneGate::nouvelle(
            Verdict::Fail,
            "RG-ACTION-MUT",
            "ECHEC_OUTIL",
        ))
    }
}
