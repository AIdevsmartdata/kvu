//! Bord gelé R2-3 (port de su2_bord.cu, sha 0a01a43e). Propriétaire : T3.
//!
//! - Sur le GPU F64, la pose du bord et la jauge passent par les noyaux DEVICE de su2_bord.cu (méthode fournie
//!   `NoyauxSu2::poser_bord_device`) : bit-identité avec su2_bord, car les `log/cos/sin` du device ne sont pas
//!   ceux de la libm. Sur le CPU et en F32, la même opération est faite sur l'hôte avec les mêmes formules
//!   (`cpu::haar_q`, `staple`-free : pose et jauge ne lisent que le bord).
//! - `empreinte_bord` = FNV-1a 64 des octets f64 des liens mu = 1..3, s ∈ [0, L³) (`fnv_bord` de su2_bord).
//! - `executer_bord` reproduit la séquence `run_bord` de su2_bord : pré-thermalisation périodique (clé = graine de
//!   la chaîne), pose du bord (cold | hot | eq : second champ à `bc_graine`, pré-thermalisé puis copié), jauge
//!   éventuelle (contrôle nul C2a), gel, thermalisation par blocs (`therm_blocs.csv`), production (`blocs.csv`).
//!   Reprise : points de reprise alignés sur un bloc, état de la pose dans `bord_etat.json`.

use super::reprise;
use super::run::{self, creer, ecrire_ligne, io, ouvrir_ajout, tronquer_lignes, CtxRun, Fin};
use super::{cpu, Gel, NoyauxSu2};
use crate::commun::{ChampHote, Erreur, Graine, Resultat, ETIQUETTE_BORD, ETIQUETTE_JAUGE};
use crate::heritage::manifest;
use crate::heritage::su2_hb::{g17, qdag, qmul, qnorm, qscale, Geo};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Condition au bord (liens spatiaux de x0 = 0).
#[derive(Clone, Debug, PartialEq)]
pub enum Bord {
    /// Identité.
    Froid,
    /// Haar i.i.d., compteur (li, 0xFFFFFFFE, 0|1, ETIQUETTE_BORD), clé = graine du bord.
    Chaud(Graine),
    /// Tranche x0 = 0 d'un champ source (eq : pré-thermalisé à la même (L, β)).
    Copie(ChampHote),
}

/// Capacité (en pas) de l'historique de tranches : le lot de pas mesurés avant rapatriement (FLB de su2_bord).
pub const CAPACITE_TRANCHES: usize = 200;
/// Valeur de départ du FNV-1a 64 de su2_bord (`fnv_bord`). ATTENTION : c'est la constante telle qu'elle est écrite
/// dans su2_bord.cu, 1469598103934665603, à laquelle il manque le dernier chiffre « 7 » de la base d'offset
/// standard (14695981039346656037 = 0xcbf29ce484222325). On la reprend TELLE QUELLE : les empreintes des
/// manifestes de production (fixtures_bord, binaire 5af3ea49) en dépendent.
pub const FNV_DEPART: u64 = 1469598103934665603;
/// Base d'offset standard du FNV-1a 64 (pour le test de l'algorithme).
pub const FNV_BASE_STANDARD: u64 = 0xcbf29ce484222325;
const FNV_PREMIER: u64 = 1099511628211;
const SLOT_BORD: u32 = 0xFFFF_FFFE;
const SLOT_JAUGE: u32 = 0xFFFF_FFFD;

/// FNV-1a 64 sur des octets, à partir de `h`.
pub fn fnv1a_octets(mut h: u64, octets: &[u8]) -> u64 {
    for &b in octets {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PREMIER);
    }
    h
}

/// FNV-1a 64 sur les octets little-endian des f64 (disposition mémoire de `struct Q {double x,y,z,w}`).
pub fn fnv1a_f64(mut h: u64, v: &[f64]) -> u64 {
    for x in v {
        h = fnv1a_octets(h, &x.to_le_bytes());
    }
    h
}

/// Empreinte des liens gelés d'un champ hôte : mu = 1..3, s ∈ [0, L³).
pub fn empreinte_champ_bord(h: &ChampHote) -> u64 {
    let v = h.v();
    let l3 = v / h.l;
    let mut e = FNV_DEPART;
    for mu in 1..4 {
        e = fnv1a_f64(e, &h.q[4 * mu * v..4 * (mu * v + l3)]);
    }
    e
}

/// Empreinte FNV-1a 64 des octets FP64 des liens gelés de la chaîne `c` (contrôle C1).
pub fn empreinte_bord(n: &dyn NoyauxSu2, c: usize) -> Resultat<u64> {
    if let Some(r) = n.empreinte_bord_device(c) {
        return r;
    }
    Ok(empreinte_champ_bord(&n.telecharger(c)?))
}

/// Pose du bord sur un champ hôte (formules de k_set_bord : li = mu·V + s, mu = 1..3, s ∈ [0, L³)).
pub fn poser_hote(h: &mut ChampHote, bord: &Bord) -> Resultat<()> {
    let v = h.v();
    let l3 = v / h.l;
    if let Bord::Copie(src) = bord {
        if src.l != h.l {
            return Err(Erreur::Config(format!("bord copié de L = {} dans un champ de L = {}", src.l, h.l)));
        }
    }
    for i in 0..3 * l3 {
        let mu = 1 + i / l3;
        let s = i % l3;
        let li = mu * v + s;
        let q = match bord {
            Bord::Froid => [1.0, 0.0, 0.0, 0.0],
            Bord::Chaud(g) => {
                let (k0, k1) = g.cle();
                cpu::haar_q(li as u32, SLOT_BORD, ETIQUETTE_BORD, k0, k1)
            }
            Bord::Copie(src) => src.lien(li),
        };
        h.q[4 * li..4 * li + 4].copy_from_slice(&q);
    }
    Ok(())
}

/// Transformation de jauge de l'hyperplan x0 = 0 sur un champ hôte (k_gen_jauge puis k_app_jauge) :
/// U_mu(x) → g(x) U_mu(x) g(x+mu)⁺, g Haar sur x0 = 0 et 1 ailleurs ; chaque lien est renormalisé.
pub fn jauge_hote(h: &mut ChampHote, graine: Graine) {
    let g = Geo::new(h.l);
    let (v, l3) = (g.v, g.v / g.l);
    let (k0, k1) = graine.cle();
    let gauge: Vec<[f64; 4]> = (0..l3).map(|i| cpu::haar_q(i as u32, SLOT_JAUGE, ETIQUETTE_JAUGE, k0, k1)).collect();
    for i in 0..4 * v {
        let (mu, s) = (i / v, i % v);
        let x = g.coords(s);
        let mut u = h.lien(i);
        if x[0] == 0 {
            u = qmul(gauge[s], u);
        }
        let sm = g.nb_up(s, mu, x[mu]);
        let xm0 = if mu == 0 { (x[0] + 1) % g.l } else { x[0] };
        if xm0 == 0 {
            u = qmul(u, qdag(gauge[sm % l3]));
        }
        let r = qscale(u, 1.0 / qnorm(u));
        h.q[4 * i..4 * i + 4].copy_from_slice(&r);
    }
}

/// Pose le bord sur la chaîne `c`, puis, si `jauge` est donnée, applique la transformation de
/// jauge Haar de l'hyperplan (compteur (i, 0xFFFFFFFD, 0|1, ETIQUETTE_JAUGE)) ; rend
/// (empreinte FNV-1a 64 des liens gelés AVANT la jauge, écart max des tranches avant/après la jauge ; −1 sans jauge).
pub fn poser(n: &mut dyn NoyauxSu2, c: usize, bord: &Bord, jauge: Option<Graine>) -> Resultat<(u64, f64)> {
    if let Some(r) = n.poser_bord_device(c, bord, jauge) {
        return r;
    }
    // version hôte (CPU, F32) : mêmes formules
    let mut h = n.telecharger(c)?;
    poser_hote(&mut h, bord)?;
    n.televerser(c, &h)?;
    let fnv = empreinte_champ_bord(&h);
    let mut ecart = -1.0;
    if let Some(g) = jauge {
        let (l2, nch) = (2 * h.l, n.config().chaines.len());
        n.mesurer_tranches(0, CAPACITE_TRANCHES)?;
        let t0 = n.historique_tranches()?;
        jauge_hote(&mut h, g);
        n.televerser(c, &h)?;
        n.mesurer_tranches(1, CAPACITE_TRANCHES)?;
        let t1 = n.historique_tranches()?;
        let cap = t0.len() / (nch * l2);
        let o = c * cap * l2;
        ecart = 0.0;
        for q in 0..l2 {
            ecart = f64::max(ecart, (t1[o + l2 + q] - t0[o + q]).abs());
        }
    }
    Ok((fnv, ecart))
}

// ------------------------------------------------------------------------------------------
// Séquence run_bord
// ------------------------------------------------------------------------------------------

/// Informations de la pose pour une chaîne (écrites dans bord_etat.json et dans le manifeste).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InfoBord {
    pub fnv_avant_jauge: String,
    pub fnv_gel_debut: String,
    pub jauge_ecart_max: f64,
    pub p_pre: f64,
    pub p_bc_src: f64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
struct EtatBord {
    chaines: Vec<InfoBord>,
}

const NOM_ETAT_BORD: &str = "bord_etat.json";

fn hex16(x: u64) -> String {
    format!("{x:016x}")
}

fn ecrire_etat_bord(dir: &Path, e: &EtatBord) -> Resultat<()> {
    let p = dir.join(NOM_ETAT_BORD);
    let j = serde_json::to_vec_pretty(e).map_err(|x| Erreur::Format(x.to_string()))?;
    manifest::ecrire_atomique(&p, &j).map_err(|x| io("écriture", &p, x))
}

fn lire_etat_bord(dir: &Path) -> Resultat<EtatBord> {
    let p = dir.join(NOM_ETAT_BORD);
    let t = fs::read_to_string(&p).map_err(|x| io("lecture", &p, x))?;
    serde_json::from_str(&t).map_err(|x| Erreur::Format(format!("{} illisible : {x}", p.display())))
}

/// Accumulateur des moyennes par blocs (S_t, T_t) de toutes les chaînes.
struct Accum {
    /// pas mesurés depuis le dernier rapatriement
    j: usize,
    /// pas dans le bloc courant
    cnt: u64,
    /// numéro du prochain bloc
    nb: u64,
    /// 2L sommes par chaîne
    acc: Vec<f64>,
}

struct Fichiers {
    w: Vec<BufWriter<fs::File>>,
    chemins: Vec<PathBuf>,
}

fn entete_blocs(l: usize) -> String {
    let mut s = String::from("bloc");
    for t in 0..l {
        s.push_str(&format!(",S{t}"));
    }
    for t in 0..l {
        s.push_str(&format!(",T{t}"));
    }
    s
}

fn ouvrir_fichiers(dirs: &[PathBuf], nom: &str, l: usize, lignes_reprise: Option<usize>) -> Resultat<Fichiers> {
    let mut w = Vec::new();
    let mut chemins = Vec::new();
    for d in dirs {
        let p = d.join(nom);
        match lignes_reprise {
            None => {
                let mut f = creer(&p)?;
                ecrire_ligne(&mut f, &p, &entete_blocs(l))?;
                w.push(f);
            }
            Some(n) => {
                tronquer_lignes(&p, 1 + n)?;
                w.push(ouvrir_ajout(&p)?);
            }
        }
        chemins.push(p);
    }
    Ok(Fichiers { w, chemins })
}

impl Accum {
    fn nouveau(nch: usize, l: usize, nb0: u64) -> Accum {
        Accum { j: 0, cnt: 0, nb: nb0, acc: vec![0.0; nch * 2 * l] }
    }

    /// Rapatrie les `j` pas mesurés, les accumule dans l'ordre (somme séquentielle, comme su2_bord) et écrit
    /// chaque bloc complet.
    fn vider(&mut self, ctx: &CtxRun, f: &mut Fichiers, bloc: u64) -> Resultat<()> {
        if self.j == 0 {
            return Ok(());
        }
        let (nch, l2) = (ctx.nch(), 2 * ctx.cfg.lot.l);
        let h = ctx.n.historique_tranches()?;
        let cap = h.len() / (nch * l2);
        for r in 0..self.j {
            for c in 0..nch {
                for q in 0..l2 {
                    self.acc[c * l2 + q] += h[(c * cap + r) * l2 + q];
                }
            }
            self.cnt += 1;
            if self.cnt == bloc {
                for c in 0..nch {
                    // contrôle de trajectoire (voir run::sentinelle) : un bloc non fini arrête le run sans COMPLETE
                    let bloc_c = &self.acc[c * l2..(c + 1) * l2];
                    if let Some(q) = bloc_c.iter().position(|x| !x.is_finite()) {
                        return Err(Erreur::Validation(format!(
                            "contrôle de trajectoire : chaîne {c}, bloc {}, tranche {q} non finie ({}) : faute GPU \
                             probable, aucune COMPLETE ; relancer avec les mêmes graines",
                            self.nb, bloc_c[q]
                        )));
                    }
                    let mut ligne = format!("{}", self.nb);
                    for q in 0..l2 {
                        ligne.push(',');
                        ligne.push_str(&g17(self.acc[c * l2 + q] / bloc as f64));
                    }
                    ecrire_ligne(&mut f.w[c], &f.chemins[c], &ligne)?;
                }
                self.nb += 1;
                self.acc.iter_mut().for_each(|x| *x = 0.0);
                self.cnt = 0;
            }
        }
        self.j = 0;
        for (w, p) in f.w.iter_mut().zip(&f.chemins) {
            w.flush().map_err(|e| io("écriture", p, e))?;
        }
        Ok(())
    }
}

/// Plan des phases en pas globaux.
struct Plan {
    pre: u64,
    fin_a: u64,
    fin_b: u64,
    fin_c: u64,
    bloc: u64,
}

impl Plan {
    /// Un point de reprise exige un accumulateur vide : frontière de bloc dans les phases mesurées.
    fn aligne(&self, pas: u64) -> bool {
        if pas <= self.pre {
            true
        } else if pas <= self.fin_a {
            (pas - self.pre).is_multiple_of(self.bloc)
        } else if pas <= self.fin_b {
            true
        } else {
            (pas - self.fin_b).is_multiple_of(self.bloc)
        }
    }
    fn blocs_therm(&self, pas: u64) -> usize {
        (pas.saturating_sub(self.pre).min(self.fin_a - self.pre) / self.bloc) as usize
    }
    fn blocs_prod(&self, pas: u64) -> usize {
        (pas.saturating_sub(self.fin_b) / self.bloc) as usize
    }
}

/// Séquence de su2_bord (`run_bord`), pour un lot de chaînes. Voir l'en-tête du module.
pub(super) fn executer_bord(ctx: &mut CtxRun) -> Resultat<Fin> {
    let cfg = ctx.cfg;
    let cb = cfg.bord.clone().ok_or_else(|| Erreur::Config("executer_bord sans configuration de bord".into()))?;
    let (nch, l) = (ctx.nch(), cfg.lot.l);
    let plan = Plan {
        pre: cb.pre,
        fin_a: cb.pre + cfg.therm - cfg.therm % cb.bloc,
        fin_b: cb.pre + cfg.therm,
        fin_c: cb.pre + cfg.therm + cfg.nmeas,
        bloc: cb.bloc,
    };
    ctx.n.allouer_historique(3)?;
    // --- départ ou reprise ---
    let mut pas = 0u64;
    let mut etat: Option<EtatBord> = None;
    if ctx.reprendre {
        let e = reprise::lire(&cfg.outdir, ctx.n.as_mut())?;
        pas = e.pas_faits;
        ctx.pas_au_depart = pas;
        ctx.compteurs_avant = e.compteurs;
        if pas >= plan.pre {
            let eb = lire_etat_bord(&cfg.outdir)?;
            if eb.chaines.len() != nch {
                return Err(Erreur::Format("bord_etat.json : nombre de chaînes différent du lot".into()));
            }
            etat = Some(eb);
        }
    } else {
        ctx.n.init()?;
    }
    let reprise_apres_pose = etat.is_some();
    let mut dernier = pas;
    // --- (1) pré-thermalisation périodique (clé = graine de la chaîne) ---
    while pas < plan.pre {
        ctx.pas()?;
        pas += 1;
        ctx.progres(pas, plan.fin_c);
        if pas < plan.pre && (ctx.arret_apres.is_some_and(|a| pas >= a) || due(ctx, pas, dernier)) {
            ctx.point_de_reprise(pas, 0)?;
            dernier = pas;
            if ctx.arret_apres.is_some_and(|a| pas >= a) {
                return Ok(Fin::Interrompu);
            }
        }
    }
    // --- (2) pose du bord, (3) jauge, (4) gel ---
    if etat.is_none() {
        etat = Some(poser_tout(ctx, &cb)?);
        ecrire_etat_bord(&cfg.outdir, etat.as_ref().ok_or_else(|| Erreur::Gpu("état du bord".into()))?)?;
    }
    ctx.n.fixer_gel(Gel::BordX0Spatial)?;
    let etat = etat.ok_or_else(|| Erreur::Gpu("état du bord absent".into()))?;
    // CSV : création après la pose (comme su2_bord) ou troncature à la position du point de reprise
    let mut ftherm = ouvrir_fichiers(&ctx.dirs, "therm_blocs.csv", l, reprise_apres_pose.then(|| plan.blocs_therm(pas)))?;
    let mut fprod = ouvrir_fichiers(&ctx.dirs, "blocs.csv", l, reprise_apres_pose.then(|| plan.blocs_prod(pas)))?;
    // un point de reprise dû exactement à la fin de la pré-thermalisation (post-pose)
    if pas == plan.pre && !reprise_apres_pose {
        if let Some(f) = point_bord(ctx, &plan, pas, &mut dernier)? {
            return Ok(f);
        }
    }
    // --- (5) thermalisation par blocs (therm_blocs.csv) ---
    let mut ac = Accum::nouveau(nch, l, plan.blocs_therm(pas) as u64);
    while pas < plan.fin_a {
        ctx.pas()?;
        ctx.n.mesurer_tranches(ac.j, CAPACITE_TRANCHES)?;
        ac.j += 1;
        pas += 1;
        ctx.progres(pas, plan.fin_c);
        if ac.j == CAPACITE_TRANCHES || pas == plan.fin_a {
            ac.vider(ctx, &mut ftherm, plan.bloc)?;
        }
        if veut_point(ctx, &plan, pas, dernier) {
            ac.vider(ctx, &mut ftherm, plan.bloc)?;
            if let Some(f) = point_bord(ctx, &plan, pas, &mut dernier)? {
                return Ok(f);
            }
        }
    }
    // --- reliquat de thermalisation, non mesuré ---
    while pas < plan.fin_b {
        ctx.pas()?;
        pas += 1;
        if let Some(f) = point_bord(ctx, &plan, pas, &mut dernier)? {
            return Ok(f);
        }
    }
    // --- (6) production par blocs (blocs.csv) ---
    let mut ac = Accum::nouveau(nch, l, plan.blocs_prod(pas) as u64);
    while pas < plan.fin_c {
        ctx.pas()?;
        ctx.n.mesurer_tranches(ac.j, CAPACITE_TRANCHES)?;
        ac.j += 1;
        pas += 1;
        ctx.progres(pas, plan.fin_c);
        if ac.j == CAPACITE_TRANCHES || pas == plan.fin_c {
            ac.vider(ctx, &mut fprod, plan.bloc)?;
        }
        if veut_point(ctx, &plan, pas, dernier) {
            ac.vider(ctx, &mut fprod, plan.bloc)?;
            if let Some(f) = point_bord(ctx, &plan, pas, &mut dernier)? {
                return Ok(f);
            }
        }
    }
    drop((ftherm, fprod));
    // --- fin : gel intact, P_fin, compteurs ---
    ctx.n.mesurer(2)?;
    let hist = ctx.n.historique()?;
    let (plafond, essais) = ctx.plafond_total()?;
    let mut infos = Vec::new();
    let mut intact = true;
    for c in 0..nch {
        let fin = empreinte_bord(ctx.n.as_ref(), c)?;
        let info = &etat.chaines[c];
        let ok = info.fnv_gel_debut == hex16(fin);
        intact &= ok;
        infos.push(serde_json::json!({
            "chaine": c,
            "bc": cb.bc, "bc_graine": cb.bc_graine, "jauge": cb.jauge,
            "fnv_bord_avant_jauge": info.fnv_avant_jauge,
            "fnv_bord_gel_debut": info.fnv_gel_debut,
            "fnv_bord_gel_fin": hex16(fin),
            "gel_intact": ok,
            "jauge_ecart_max": info.jauge_ecart_max,
            "P_pre": info.p_pre, "P_bc_src": info.p_bc_src,
            "P_fin": hist[(c * 3 + 2) * 5],
        }));
    }
    ctx.rapport = run::RapportRun {
        pas: plan.fin_c,
        mesures: cfg.nmeas / cb.bloc,
        dumps: 0,
        plafond,
        secondes: 0.0,
    };
    println!(
        "BORD L={} chaines={} bc={} jauge={} pre={} therm={} nmeas={} bloc={} gel_intact={} plafond={} essais_hb={}",
        l, nch, cb.bc, cb.jauge.is_some(), cb.pre, cfg.therm, cfg.nmeas, cb.bloc, intact, plafond, essais
    );
    ctx.man.champ("bord", serde_json::Value::Array(infos))?;
    ctx.man.champ("gel_intact", serde_json::json!(intact))?;
    ctx.man.champ("essais_hb", serde_json::json!(essais))?;
    ctx.man.champ(
        "geometrie",
        serde_json::json!(
            "L^4 periodique ; liens U_mu(x), mu=1..3, x0=0 geles ; S_t = moyenne 1/2ReTr des 3L^3 plaquettes spatiales de la tranche x0=t ; T_t = idem plaquettes (0,nu) basees en x0=t"
        ),
    )?;
    if plafond != 0 {
        return Err(Erreur::Validation(format!("plafond d'essais atteint {plafond} fois")));
    }
    if !intact {
        return Err(Erreur::Validation("liens gelés modifiés (empreinte FNV du bord changée)".into()));
    }
    run::enregistrer_sorties(ctx)?;
    Ok(Fin::Termine)
}

fn due(ctx: &CtxRun, pas: u64, dernier: u64) -> bool {
    ctx.cfg.reprise_every > 0 && pas - dernier >= ctx.cfg.reprise_every
}

/// Vrai si un point de reprise (période ou arrêt demandé) est dû à ce pas, sur une frontière alignée.
fn veut_point(ctx: &CtxRun, plan: &Plan, pas: u64, dernier: u64) -> bool {
    plan.aligne(pas) && (ctx.arret_apres.is_some_and(|a| pas >= a) || due(ctx, pas, dernier))
}

/// Écrit le point de reprise si `veut_point` ; Some(fin) pour s'arrêter. L'accumulateur doit être vidé avant.
fn point_bord(ctx: &mut CtxRun, plan: &Plan, pas: u64, dernier: &mut u64) -> Resultat<Option<Fin>> {
    if !veut_point(ctx, plan, pas, *dernier) {
        return Ok(None);
    }
    let arret = ctx.arret_apres.is_some_and(|a| pas >= a);
    let blocs = (plan.blocs_therm(pas) + plan.blocs_prod(pas)) as u64;
    ctx.point_de_reprise(pas, blocs)?;
    *dernier = pas;
    if arret {
        ctx.rapport.pas = pas;
    }
    Ok(if arret { Some(Fin::Interrompu) } else { None })
}

/// Pose du bord sur toutes les chaînes (+ jauge) : séquence (2)-(4) de run_bord.
fn poser_tout(ctx: &mut CtxRun, cb: &run::ConfigBord) -> Resultat<EtatBord> {
    let (nch, cfg) = (ctx.nch(), ctx.cfg);
    // P_pre : plaquette de la chaîne périodique pré-thermalisée
    ctx.n.mesurer(0)?;
    let hist = ctx.n.historique()?;
    let p_pre: Vec<f64> = (0..nch).map(|c| hist[(c * 3) * 5]).collect();
    // source « eq » : second champ à la graine du bord, pré-thermalisé à la même (L, β)
    let mut sources: Vec<Option<ChampHote>> = vec![None; nch];
    let mut p_src = vec![-1.0; nch];
    if cb.bc == "eq" {
        let mut lot = cfg.lot.clone();
        for c in &mut lot.chaines {
            c.graine = Graine(cb.bc_graine);
            c.depart = crate::commun::Depart::Chaud;
        }
        let mut eq = super::construire(ctx.app, lot)?;
        eq.init()?;
        eq.allouer_historique(1)?;
        for _ in 0..cb.pre {
            eq.sweep_hb()?;
            for _ in 0..cfg.lot.nor {
                eq.sweep_or()?;
            }
        }
        eq.mesurer(0)?;
        let h = eq.historique()?;
        for c in 0..nch {
            p_src[c] = h[c * 5];
            sources[c] = Some(eq.telecharger(c)?);
        }
    }
    let mut infos = Vec::new();
    for c in 0..nch {
        let bord = match cb.bc.as_str() {
            "cold" => Bord::Froid,
            "hot" => Bord::Chaud(Graine(cb.bc_graine)),
            _ => Bord::Copie(sources[c].take().ok_or_else(|| Erreur::Config("source eq absente".into()))?),
        };
        let (fnv_av, ecart) = poser(ctx.n.as_mut(), c, &bord, cb.jauge.map(Graine))?;
        // gel : l'empreinte est prise APRÈS la jauge (h_bord0 de su2_bord)
        let fnv0 = empreinte_bord(ctx.n.as_ref(), c)?;
        infos.push(InfoBord {
            fnv_avant_jauge: hex16(fnv_av),
            fnv_gel_debut: hex16(fnv0),
            jauge_ecart_max: ecart,
            p_pre: p_pre[c],
            p_bc_src: p_src[c],
        });
    }
    Ok(EtatBord { chaines: infos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::{Depart, Precision};
    use crate::su2::cpu::Cpu64;
    use crate::su2::{ConfigLot, ParamChaine};

    fn lot(l: usize, nch: usize) -> ConfigLot {
        ConfigLot {
            l,
            chaines: (0..nch)
                .map(|i| ParamChaine { beta: 2.3, graine: Graine(40 + i as u64), depart: Depart::Chaud })
                .collect(),
            nor: 2,
            precision: Precision::F64,
            options: crate::gpu::OptionsCompil::reference_k1(),
            mutant_metro: false,
        }
    }

    #[test]
    fn fnv_vecteurs_de_reference() {
        // l'algorithme : vecteurs publics du FNV-1a 64 (base standard)
        assert_eq!(fnv1a_octets(FNV_BASE_STANDARD, b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a_octets(FNV_BASE_STANDARD, b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a_octets(FNV_BASE_STANDARD, b"foobar"), 0x85944171f73967e8);
        // la constante de su2_bord est bien celle du fichier (typo comprise)
        assert_eq!(FNV_DEPART.to_string(), "1469598103934665603");
        assert_ne!(FNV_DEPART, FNV_BASE_STANDARD);
    }

    #[test]
    fn empreinte_du_bord_froid_egale_celle_de_la_production() {
        // fixtures_bord/t_cold/manifest.json (su2_bord, L = 8) : fnv_bord_avant_jauge = 6ff2d00186910383
        let mut h = ChampHote::froid(8).unwrap();
        poser_hote(&mut h, &Bord::Froid).unwrap();
        assert_eq!(hex16(empreinte_champ_bord(&h)), "6ff2d00186910383");
    }

    #[test]
    fn empreinte_ne_voit_que_le_bord() {
        let mut a = ChampHote::froid(4).unwrap();
        let e0 = empreinte_champ_bord(&a);
        // un lien temporel (mu = 0) ou un lien intérieur ne changent pas l'empreinte
        a.q[0] = 0.0;
        a.q[4 * 256 + 4 * 100] = 0.5; // mu = 1, s = 100 ≥ L³ = 64
        assert_eq!(empreinte_champ_bord(&a), e0);
        // un lien du bord la change
        a.q[4 * 256 + 4 * 10 + 1] = 0.25;
        assert_ne!(empreinte_champ_bord(&a), e0);
    }

    #[test]
    fn pose_froide_chaude_et_copie() {
        let mut a = ChampHote::froid(4).unwrap();
        let v = 256;
        let l3 = 64;
        poser_hote(&mut a, &Bord::Chaud(Graine(5))).unwrap();
        for mu in 1..4 {
            for s in 0..l3 {
                let q = a.lien(mu * v + s);
                assert!((qnorm(q) - 1.0).abs() < 1e-14);
            }
        }
        // les autres liens sont intacts
        assert_eq!(a.lien(0), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(a.lien(v + l3), [1.0, 0.0, 0.0, 0.0]);
        let chaud = a.clone();
        // déterminisme et dépendance à la graine
        let mut b = ChampHote::froid(4).unwrap();
        poser_hote(&mut b, &Bord::Chaud(Graine(5))).unwrap();
        assert_eq!(a, b);
        poser_hote(&mut b, &Bord::Chaud(Graine(6))).unwrap();
        assert_ne!(a, b);
        // froid : retour à l'identité ; copie : tranche du champ source
        poser_hote(&mut a, &Bord::Froid).unwrap();
        assert_eq!(a, ChampHote::froid(4).unwrap());
        poser_hote(&mut a, &Bord::Copie(chaud.clone())).unwrap();
        assert_eq!(a, chaud);
        assert!(poser_hote(&mut a, &Bord::Copie(ChampHote::froid(6).unwrap())).is_err());
    }

    #[test]
    fn la_jauge_conserve_l_unitarite_et_les_tranches() {
        // C2a : la jauge de l'hyperplan laisse les S_t, T_t invariants (≤ 1e-14)
        let mut n = Cpu64::nouveau(lot(4, 1)).unwrap();
        n.init().unwrap();
        for _ in 0..3 {
            n.sweep_hb().unwrap();
            n.sweep_or().unwrap();
        }
        let (fnv, ecart) = poser(&mut n, 0, &Bord::Chaud(Graine(9)), Some(Graine(77))).unwrap();
        assert!((0.0..=1e-14).contains(&ecart), "écart de jauge {ecart}");
        let h = n.telecharger(0).unwrap();
        assert!(h.defaut_unitarite() < 1e-14);
        // la jauge change bien le bord (FNV pris AVANT la jauge, différent de celui d'après)
        assert_ne!(fnv, empreinte_champ_bord(&h));
        // sans jauge : −1
        let (_, e2) = poser(&mut n, 0, &Bord::Froid, None).unwrap();
        assert_eq!(e2, -1.0);
    }

    #[test]
    fn jauge_nulle_si_pas_de_bord_actif() {
        // jauge sur un champ froid : g(x) 1 g(x+mu)⁺ n'est pas l'identité, mais S_t = T_t = 1 reste exact
        let mut h = ChampHote::froid(4).unwrap();
        jauge_hote(&mut h, Graine(3));
        assert!(h.defaut_unitarite() < 1e-14);
        let g = Geo::new(4);
        let u: Vec<[f64; 4]> = (0..4 * 256).map(|i| h.lien(i)).collect();
        let t = cpu::tranches_hote(&g, &u);
        assert!(t.iter().all(|x| (x - 1.0).abs() < 1e-13), "{t:?}");
    }

    #[test]
    fn plan_des_phases() {
        let p = Plan { pre: 10, fin_a: 10 + 20, fin_b: 10 + 23, fin_c: 10 + 23 + 40, bloc: 5 };
        assert!(p.aligne(3) && p.aligne(10));
        assert!(!p.aligne(12) && p.aligne(15) && p.aligne(30));
        assert!(p.aligne(31) && p.aligne(33)); // reliquat : toujours aligné
        assert!(!p.aligne(34) && p.aligne(38));
        assert_eq!(p.blocs_therm(10), 0);
        assert_eq!(p.blocs_therm(25), 3);
        assert_eq!(p.blocs_therm(33), 4);
        assert_eq!(p.blocs_prod(33), 0);
        assert_eq!(p.blocs_prod(43), 2);
    }

    #[test]
    fn entete_des_blocs() {
        assert_eq!(entete_blocs(2), "bloc,S0,S1,T0,T1");
    }
}
