//! Bloc S1 FP64 : un fil par lien, produits et somme dans l'ordre de la référence CPU.

use super::bloc::{SortieBloc, StatsTheta, N_CLASSES};
use super::geometrie::{Geometrie, Table, SITES_BLOC};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::{
    assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, Pod, Tampon,
};
use crate::noyaux::{RG_BLOC, SU2_COMMUN};
use crate::su2::{ChampGpu, TamponChamp};

const BLOC: u32 = 128;
const MAXN: usize = 24;

/// Tables compactées, construites sans appareil ni accès au runtime GPU.
pub struct TablesGpu {
    pub steps: Vec<u32>,
    pub ch_ir: Vec<i32>,
    pub ch_pas: Vec<i32>,
    pub poids: Vec<f64>,
    pub ir_ref: i32,
    pub nb_boucles: usize,
}

pub fn tables(table: &Table) -> TablesGpu {
    assert_eq!(table.debut.len(), SITES_BLOC + 1);
    assert_eq!(table.debut[0], 0);
    assert_eq!(table.debut[SITES_BLOC], table.chemins.len());
    assert!(table.chemins.len() <= i32::MAX as usize);
    for d in table.debut.windows(2) {
        assert!(
            d[0] < d[1] && d[1] - d[0] <= MAXN,
            "1 à 24 chemins par site requis"
        );
    }
    assert!(table.ref_site.iter().all(|&x| x < 3));
    let mut steps = Vec::new();
    let mut ch_pas = Vec::with_capacity(table.chemins.len() + 1);
    for chemin in &table.chemins {
        assert!(steps.len() <= i32::MAX as usize);
        ch_pas.push(steps.len() as i32);
        for pas in chemin {
            assert!(pas.mu < 4 && (pas.signe == -1 || pas.signe == 1));
            let mut mot = (pas.mu as u32) << 16;
            if pas.signe > 0 {
                mot |= 1 << 18;
            }
            for (m, &dep) in pas.dep.iter().enumerate() {
                let chiffre = i32::from(dep) + 8;
                assert!(
                    (0..=15).contains(&chiffre),
                    "déplacement hors du compactage sur 4 bits"
                );
                mot |= (chiffre as u32) << (4 * m);
            }
            steps.push(mot);
        }
    }
    assert!(steps.len() <= i32::MAX as usize);
    ch_pas.push(steps.len() as i32);
    TablesGpu {
        steps,
        ch_ir: table.debut.iter().map(|&x| x as i32).collect(),
        ch_pas,
        poids: (0..SITES_BLOC).map(|ir| table.poids(ir)).collect(),
        ir_ref: Table::ir_de(table.ref_site) as i32,
        nb_boucles: table.nb_boucles(),
    }
}

fn taille_champ(l: usize) -> Resultat<usize> {
    if l < 3 || !l.is_multiple_of(3) {
        return Err(Erreur::Config(format!(
            "L = {l} : multiple de 3 ≥ 3 requis"
        )));
    }
    let taille = l
        .checked_pow(4)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| Erreur::Config("taille du champ trop grande".into()))?;
    if l > i32::MAX as usize || taille > isize::MAX as usize / size_of::<f64>() {
        return Err(Erreur::Config(
            "dimensions du champ trop grandes pour le noyau RG".into(),
        ));
    }
    let nb_liens = 4 * (l / 3).pow(4);
    if nb_liens.div_ceil(BLOC as usize) > u32::MAX as usize {
        return Err(Erreur::Config("grille du noyau RG trop grande".into()));
    }
    Ok(taille)
}

fn valider_composantes(q: &[f64]) -> Resultat<()> {
    if let Some(i) = q.iter().position(|x| !x.is_finite()) {
        return Err(Erreur::Config(format!(
            "composante non finie au lien {}, composante {}",
            i / 4,
            i % 4
        )));
    }
    Ok(())
}

/// Validation pure, à appeler également avant de construire un pilote pour une entrée non fiable.
pub fn valider_entree(champ: &ChampHote) -> Resultat<()> {
    if champ.q.len() != taille_champ(champ.l)? {
        return Err(Erreur::Config(
            "dimensions ou longueur du champ invalides".into(),
        ));
    }
    valider_composantes(&champ.q)
}

fn verifier_memoire(app: &Appareil, octets: usize) -> Resultat<()> {
    let requis = (octets as u64)
        .checked_mul(2)
        .ok_or_else(|| Erreur::Config("taille des tampons RG trop grande".into()))?;
    let libre = app.memoire_libre_octets()?;
    if libre < requis {
        return Err(Erreur::Config(format!(
            "mémoire GPU insuffisante pour RG : {libre} octets libres, {requis} requis (deux fois les tampons)"
        )));
    }
    Ok(())
}

fn octets_sortie(nb_liens: usize, avec_stats: bool) -> Resultat<usize> {
    let par_lien = 4 * size_of::<f64>()
        + if avec_stats {
            (N_CLASSES + 2) * size_of::<u32>() + size_of::<f64>()
        } else {
            0
        };
    nb_liens
        .checked_mul(par_lien)
        .ok_or_else(|| Erreur::Config("taille des tampons RG trop grande".into()))
}

fn televerser<T: Pod>(app: &Appareil, valeurs: &[T]) -> Resultat<Tampon<T>> {
    let mut tampon = app.alloc(valeurs.len())?;
    app.ecrire(&mut tampon, 0, valeurs)?;
    Ok(tampon)
}

fn fusionner_stats(hist: &[u32], cnt: &[u32], tmax: &[f64], nb_boucles: usize) -> StatsTheta {
    let mut stats = StatsTheta::vide();
    for (li, &theta) in tmax.iter().enumerate() {
        stats.n += nb_boucles as u64;
        for (c, h) in stats.histogramme.iter_mut().enumerate() {
            *h += u64::from(hist[li * N_CLASSES + c]);
        }
        stats.n_hors += u64::from(cnt[2 * li]);
        stats.n_ambigues += u64::from(cnt[2 * li + 1]);
        stats.theta_max = stats.theta_max.max(theta);
    }
    stats
}

/// Appareil, module gardé en vie et tables réutilisables entre plusieurs blocages.
pub struct BlocGpu {
    app: Appareil,
    module: Module,
    fonction: Fonction,
    steps: Tampon<u32>,
    ch_ir: Tampon<i32>,
    ch_pas: Tampon<i32>,
    poids: Tampon<f64>,
    ir_ref: i32,
    nb_boucles: usize,
}

impl BlocGpu {
    pub fn nouveau(app: &Appareil, geometrie: Geometrie) -> Resultat<BlocGpu> {
        let t = tables(&Table::nouvelle(geometrie));
        let octets = (t.steps.len() + t.ch_ir.len() + t.ch_pas.len()) * size_of::<u32>()
            + t.poids.len() * size_of::<f64>();
        let src = assembler(app.backend(), "rg_bloc", &[&SU2_COMMUN, &RG_BLOC], &[]);
        let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
        verifier_memoire(app, octets)?;
        Ok(BlocGpu {
            app: app.clone(),
            fonction: app.fonction(&module, "k_rg_bloc")?,
            module,
            steps: televerser(app, &t.steps)?,
            ch_ir: televerser(app, &t.ch_ir)?,
            ch_pas: televerser(app, &t.ch_pas)?,
            poids: televerser(app, &t.poids)?,
            ir_ref: t.ir_ref,
            nb_boucles: t.nb_boucles,
        })
    }

    pub fn provenance(&self) -> crate::gpu::ProvenanceCompil {
        self.module.provenance().clone()
    }

    pub fn bloquer_hote(&self, champ: &ChampHote, avec_stats: bool) -> Resultat<SortieBloc> {
        valider_entree(champ)?;
        let nb_liens = 4 * (champ.l / 3).pow(4);
        let sortie_octets = octets_sortie(nb_liens, avec_stats)?;
        let octets = champ
            .q
            .len()
            .checked_mul(size_of::<f64>())
            .and_then(|n| n.checked_add(sortie_octets))
            .ok_or_else(|| Erreur::Config("taille des tampons RG trop grande".into()))?;
        verifier_memoire(&self.app, octets)?;
        let entree = televerser(&self.app, &champ.q)?;
        self.bloquer(entree.arg(), champ.l, avec_stats)
    }

    pub fn bloquer_resident(
        &self,
        champ: &ChampGpu,
        chaine: usize,
        avec_stats: bool,
    ) -> Resultat<SortieBloc> {
        let taille = taille_champ(champ.l)?;
        let tampon = match &champ.tampon {
            TamponChamp::F64(t) => t,
            TamponChamp::F32(_) => {
                return Err(Erreur::Config(
                    "bloc RG : champ résident F64 requis, F32 refusé".into(),
                ))
            }
        };
        if chaine >= champ.nchaines || champ.nchaines.checked_mul(taille) != Some(tampon.len()) {
            return Err(Erreur::Config(
                "chaîne ou longueur du champ résident invalide".into(),
            ));
        }
        // chaîne * 16 * V, en éléments f64. Pas de relecture du champ pour la finitude : un champ résident vient du
        // générateur (sentinelle de `su2::run`), et relire 16·L⁴ réels à chaque mesure coûterait autant que le noyau.
        let decalage = chaine * taille;
        self.bloquer(tampon.arg_decale(decalage), champ.l, avec_stats)
    }

    fn bloquer(&self, entree: Arg, l: usize, avec_stats: bool) -> Resultat<SortieBloc> {
        let lc = l / 3;
        let nb_liens = 4 * lc.pow(4);
        verifier_memoire(&self.app, octets_sortie(nb_liens, avec_stats)?)?;
        let sortie = self.app.alloc::<f64>(4 * nb_liens)?;
        let mesures = if avec_stats {
            Some((
                self.app.alloc::<u32>(nb_liens * N_CLASSES)?,
                self.app.alloc::<u32>(2 * nb_liens)?,
                self.app.alloc::<f64>(nb_liens)?,
            ))
        } else {
            None
        };
        let (hist, cnt, tmax) = match &mesures {
            Some((h, c, t)) => (h.arg(), c.arg(), t.arg()),
            None => (Arg::Nul, Arg::Nul, Arg::Nul),
        };
        self.app.lancer(
            &self.fonction,
            &Lancement::lineaire(nb_liens, BLOC),
            &[
                entree,
                Arg::I32(l as i32),
                self.steps.arg(),
                self.ch_ir.arg(),
                self.ch_pas.arg(),
                self.poids.arg(),
                Arg::I32(self.ir_ref),
                sortie.arg(),
                hist,
                cnt,
                tmax,
            ],
        )?;
        let mut q = vec![0.0; 4 * nb_liens];
        self.app.lire(&sortie, 0, &mut q)?;
        let stats = if let Some((h, c, t)) = mesures {
            let mut hist = vec![0; nb_liens * N_CLASSES];
            let mut cnt = vec![0; 2 * nb_liens];
            let mut tmax = vec![0.0; nb_liens];
            self.app.lire(&h, 0, &mut hist)?;
            self.app.lire(&c, 0, &mut cnt)?;
            self.app.lire(&t, 0, &mut tmax)?;
            fusionner_stats(&hist, &cnt, &tmax, self.nb_boucles)
        } else {
            StatsTheta::vide()
        };
        Ok(SortieBloc {
            grossier: ChampHote { l: lc, q },
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_entiere_par_lien() {
        let mut hist = vec![0; 2 * N_CLASSES];
        hist[0] = 3;
        hist[N_CLASSES + 63] = 3;
        let stats = fusionner_stats(&hist, &[1, 0, 2, 1], &[0.5, 3.0], 3);
        assert_eq!(stats.n, 6);
        assert_eq!(stats.histogramme[0], 3);
        assert_eq!(stats.histogramme[63], 3);
        assert_eq!(stats.n_hors, 3);
        assert_eq!(stats.n_ambigues, 1);
        assert_eq!(stats.theta_max, 3.0);
    }

    #[test]
    fn compte_memoire_et_dimensions() {
        assert_eq!(octets_sortie(4, false).unwrap(), 128);
        assert_eq!(octets_sortie(4, true).unwrap(), 1216);
        assert!(octets_sortie(usize::MAX, true).is_err());
        assert!(taille_champ(usize::MAX).is_err());
        assert_eq!(taille_champ(3).unwrap(), 16 * 81);
    }
}
