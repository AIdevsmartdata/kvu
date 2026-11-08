// SPDX-License-Identifier: Apache-2.0
//! Santé GPU du processus : registre des fautes d'exécution, contrôle par lots, injection de pannes pour les essais.
//!
//! **Pourquoi.** Un noyau qui fait une faute (accès mémoire illégal, lancement raté) ne le dit pas quand on le lance : le
//! lancement est asynchrone, et l'erreur ne remonte qu'à la synchronisation ou à la copie suivante, sous la forme d'un code
//! collant du runtime (« an illegal memory access was encountered (700) »). Ce code ne doit jamais se perdre avant la sortie :
//! - un destructeur (`hipFree`, `hipModuleUnload`…) ne peut pas remonter le sien, et une commande peut n'avoir aucune
//!   synchronisation entre son dernier noyau et sa sortie ;
//! - le point d'entrée (`cli::executer_argv`) interroge donc ce registre à la sortie et refuse de rendre 0 (code 3) si une
//!   faute y est consignée, même si la commande n'en a rien vu ; le message est écrit sur stderr.
//!
//! **Qui consigne.** `Appareil::{lancer, synchroniser, lire, ecrire, copier, zero}` consignent tout code d'erreur du runtime qu'ils
//! reçoivent (`Erreur::Gpu`) : l'appelant le reçoit aussi, le registre garde la trace même s'il l'avale. La destruction d'un appareil
//! fait une dernière synchronisation ([`controle_destruction`]) : comme personne ne reçoit son erreur, elle n'est consignée que si
//! son texte décrit une faute d'exécution ([`est_faute_execution`]) ; un bruit de démontage (contexte déjà détruit, appareil
//! invalide) ne transforme pas un run sain en échec. Ni l'allocation, ni la compilation, ni le chargement d'un module ne consignent :
//! leurs échecs sont parfois tolérés par l'appelant (cache de code objet refusé puis recompilé, mémoire libre illisible), ce ne sont
//! pas des fautes d'exécution.
//!
//! **Contrôle par lots.** [`LANCEMENTS_MAX_SANS_CONTROLE`] lancements de suite sans aucune synchronisation en forcent une
//! (`Appareil::lancer`) : une faute est relevée au plus tard après ce nombre de lancements, quel que soit l'appelant.
//!
//! **Jamais de libération sous un noyau en vol.** Un tampon libéré (`hipFree`) ou un module déchargé (`hipModuleUnload`) pendant qu'un noyau
//! lancé de façon asynchrone s'exécute encore fait, sur un vrai GPU, une faute de page (données : client TCP ; instructions : client SQC) que
//! le processus peut ne jamais voir, puisque tout est démonté après l'écriture des résultats. L'ordre des `Drop` de Rust ne garantit pas que
//! les noyaux ont fini (une erreur qui court-circuite la fin d'une commande libère ses tampons sans avoir synchronisé). Chaque tampon
//! et chaque module de kvu attend donc l'appareil (`hipDeviceSynchronize`, code de retour vérifié et consigné) AVANT d'être libéré ou déchargé,
//! quel que soit l'ordre de destruction : [`controle_destruction`] est ce contrôle.
//!
//! **Injection de pannes** (essais seulement ; elle ne simule JAMAIS un succès, elle ne peut que faire échouer). La variable
//! `KVU_INJECTION_FAUTE_GPU=<point>[:<rang>]` fait échouer le `<rang>`-ième appel (défaut 1) du point : `lancement`,
//! `synchronisation`, `copie` (`lire`, `ecrire` ou `copier`), `liberation` (l'attente de l'appareil qui précède la libération d'un
//! tampon, le déchargement d'un module ou la destruction de l'appareil), ou `sortie` (faute relevée par le contrôle de sortie du
//! processus, y compris pour une commande sans GPU). Le message injecté porte le texte du runtime (« an illegal memory access was
//! encountered (700) ») pour que les détecteurs des lanceurs se déclenchent comme sur une vraie faute. Une valeur invalide fait
//! échouer la commande en usage (rc 2) : une faute de frappe ne désactive jamais l'injection en silence.

use crate::commun::{Erreur, Resultat};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Variable d'environnement de l'injection de pannes (essais seulement).
pub const ENV_INJECTION: &str = "KVU_INJECTION_FAUTE_GPU";

/// Nombre maximal de lancements de noyaux sans synchronisation : au-delà, `Appareil::lancer` en force une.
pub const LANCEMENTS_MAX_SANS_CONTROLE: u64 = 4096;

/// Nombre de lignes de faute conservées pour le message de sortie (le total est compté à part).
const FAUTES_CONSERVEES: usize = 16;

const TEXTE_ACCES_ILLEGAL: &str = "an illegal memory access was encountered (700)";
const TEXTE_LANCEMENT_RATE: &str = "unspecified launch failure (719)";

/// Point du runtime où une panne peut être injectée.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// `Appareil::lancer` (lancement d'un noyau).
    Lancement,
    /// `Appareil::synchroniser` (y compris les synchronisations forcées par le contrôle par lots).
    Synchronisation,
    /// `Appareil::{lire, ecrire, copier}`.
    Copie,
    /// Attente de l'appareil qui précède la libération d'un tampon, le déchargement d'un module ou la destruction de l'appareil
    /// ([`controle_destruction`]).
    Liberation,
    /// Contrôle de sortie du processus (`cli::executer_argv`).
    Sortie,
}

impl Point {
    /// Noms acceptés, pour les messages d'erreur.
    pub const NOMS: &'static str = "lancement|synchronisation|copie|liberation|sortie";

    pub fn nom(self) -> &'static str {
        match self {
            Point::Lancement => "lancement",
            Point::Synchronisation => "synchronisation",
            Point::Copie => "copie",
            Point::Liberation => "liberation",
            Point::Sortie => "sortie",
        }
    }

    fn depuis_nom(nom: &str) -> Option<Point> {
        [Point::Lancement, Point::Synchronisation, Point::Copie, Point::Liberation, Point::Sortie]
            .into_iter()
            .find(|p| p.nom() == nom.to_ascii_lowercase())
    }

    /// Indice du compteur d'appels (la sortie n'a pas de compteur : elle est évaluée une fois).
    fn indice(self) -> Option<usize> {
        match self {
            Point::Lancement => Some(0),
            Point::Synchronisation => Some(1),
            Point::Copie => Some(2),
            Point::Liberation => Some(3),
            Point::Sortie => None,
        }
    }
}

/// Une panne à injecter : le `rang`-ième appel (à partir de 1) du `point`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Injection {
    pub point: Point,
    pub rang: u64,
}

impl Injection {
    /// Analyse `point[:rang]` (valeur de [`ENV_INJECTION`]).
    pub fn analyser(texte: &str) -> Result<Injection, String> {
        let t = texte.trim();
        let (nom, rang) = match t.split_once(':') {
            Some((n, r)) => (n.trim(), Some(r.trim())),
            None => (t, None),
        };
        let point = Point::depuis_nom(nom).ok_or_else(|| format!("point « {nom} » inconnu ({})", Point::NOMS))?;
        let rang = match rang {
            None => 1,
            Some(r) => r
                .parse::<u64>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("rang « {r} » invalide (entier ≥ 1)"))?,
        };
        if point == Point::Sortie && rang != 1 {
            return Err("le point « sortie » n'a qu'un rang : 1".to_string());
        }
        Ok(Injection { point, rang })
    }

    /// Forme canonique, celle qui est lue par [`Injection::analyser`].
    pub fn texte(&self) -> String {
        if self.point == Point::Sortie {
            "sortie".to_string()
        } else {
            format!("{}:{}", self.point.nom(), self.rang)
        }
    }

    /// Message de la faute injectée : le texte du runtime, précédé de la marque de l'injection.
    fn message(&self, operation: &str) -> String {
        let texte = if self.point == Point::Lancement { TEXTE_LANCEMENT_RATE } else { TEXTE_ACCES_ILLEGAL };
        format!("[injection {ENV_INJECTION}={}] {operation} : {texte}", self.texte())
    }
}

/// Registre des fautes d'exécution d'un processus, avec ses compteurs d'appels et son injection éventuelle.
///
/// Le processus en a UN ([`globale`]) ; les essais en créent à volonté ([`Sante::nouvelle`]) sans toucher au registre global.
pub struct Sante {
    injection: Result<Option<Injection>, String>,
    /// Appels vus par point d'injection (lancement, synchronisation, copie, libération).
    appels: [AtomicU64; 4],
    /// Lancements depuis la dernière synchronisation réussie.
    sans_controle: AtomicU64,
    sortie_injectee: AtomicBool,
    fautes: Mutex<Vec<String>>,
    total: AtomicU64,
}

impl Sante {
    /// Registre vide, avec l'injection donnée (ou aucune).
    pub fn nouvelle(injection: Option<Injection>) -> Sante {
        Sante::construire(Ok(injection))
    }

    /// Registre dont l'injection vient du texte de [`ENV_INJECTION`] (absent ou vide : aucune). Un texte invalide est conservé :
    /// [`Sante::verifier_configuration`] le refuse, et `avant` aussi.
    pub fn depuis_texte(texte: Option<&str>) -> Sante {
        let injection = match texte.map(str::trim).filter(|t| !t.is_empty()) {
            None => Ok(None),
            Some(t) => Injection::analyser(t).map(Some),
        };
        Sante::construire(injection)
    }

    fn construire(injection: Result<Option<Injection>, String>) -> Sante {
        Sante {
            injection,
            appels: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            sans_controle: AtomicU64::new(0),
            sortie_injectee: AtomicBool::new(false),
            fautes: Mutex::new(Vec::new()),
            total: AtomicU64::new(0),
        }
    }

    /// Refuse une injection mal écrite (rc 2) : à appeler avant d'exécuter la commande.
    pub fn verifier_configuration(&self) -> Resultat<()> {
        match &self.injection {
            Err(m) => Err(Erreur::Config(format!("{ENV_INJECTION} : {m}"))),
            Ok(_) => Ok(()),
        }
    }

    /// Injection active, s'il y en a une (valide).
    pub fn injection(&self) -> Option<Injection> {
        self.injection.as_ref().ok().copied().flatten()
    }

    /// Appelée AVANT l'opération `operation` du `point` : compte l'appel et, si l'injection le demande, fait échouer celui-ci
    /// (`Erreur::Gpu`, consignée comme toute faute). Le point `Sortie` n'est jamais évalué ici.
    pub fn avant(&self, point: Point, operation: &str) -> Resultat<()> {
        self.verifier_configuration()?;
        let Some(i) = point.indice() else { return Ok(()) };
        let rang = self.appels[i].fetch_add(1, Ordering::Relaxed) + 1;
        match self.injection() {
            Some(inj) if inj.point == point && inj.rang == rang => {
                let e = Erreur::Gpu(inj.message(operation));
                self.consigner(operation, &e);
                Err(e)
            }
            _ => Ok(()),
        }
    }

    /// Consigne `e` s'il vient du runtime GPU (`Erreur::Gpu`) ; toute autre erreur est ignorée ici.
    pub fn consigner(&self, origine: &str, e: &Erreur) {
        if let Erreur::Gpu(m) = e {
            self.consigner_texte(origine, m);
        }
    }

    /// Consigne une faute (texte libre) : une ligne par texte distinct, au plus [`FAUTES_CONSERVEES`] lignes.
    pub fn consigner_texte(&self, origine: &str, detail: &str) {
        self.total.fetch_add(1, Ordering::Relaxed);
        let ligne = format!("{origine} : {detail}");
        let mut fautes = self.fautes.lock().unwrap_or_else(|p| p.into_inner());
        if fautes.len() < FAUTES_CONSERVEES && !fautes.contains(&ligne) {
            fautes.push(ligne);
        }
    }

    /// Fautes consignées (lignes distinctes, dans l'ordre).
    pub fn fautes(&self) -> Vec<String> {
        self.fautes.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Nombre total de consignations (les lignes identiques sont comptées, pas répétées).
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Un lancement de plus a eu lieu. Vrai quand [`LANCEMENTS_MAX_SANS_CONTROLE`] lancements se sont suivis sans
    /// synchronisation réussie : l'appelant doit alors synchroniser.
    pub fn lancement_compte(&self) -> bool {
        self.sans_controle.fetch_add(1, Ordering::Relaxed) + 1 >= LANCEMENTS_MAX_SANS_CONTROLE
    }

    /// Une synchronisation a réussi : le compte des lancements non contrôlés repart de zéro.
    pub fn controle_fait(&self) {
        self.sans_controle.store(0, Ordering::Relaxed);
    }

    /// Bilan du contrôle de sortie : les fautes consignées, plus la faute du point `sortie` si elle est demandée (une seule
    /// fois). `None` si le processus est sain.
    pub fn bilan_sortie(&self) -> Option<Vec<String>> {
        let mut lignes = self.fautes();
        if let Some(inj) = self.injection() {
            if inj.point == Point::Sortie && !self.sortie_injectee.swap(true, Ordering::AcqRel) {
                lignes.push(inj.message("contrôle de sortie"));
            }
        }
        (!lignes.is_empty()).then_some(lignes)
    }
}

/// Le registre du processus, construit à la première demande depuis [`ENV_INJECTION`].
pub fn globale() -> &'static Sante {
    static GLOBALE: OnceLock<Sante> = OnceLock::new();
    GLOBALE.get_or_init(|| Sante::depuis_texte(std::env::var(ENV_INJECTION).ok().as_deref()))
}

/// Attente de l'appareil AVANT de libérer un tampon, de décharger un module ou de détruire l'appareil : `synchroniser` est appelé une fois et
/// son erreur est consignée dans le registre du processus si elle décrit une faute d'exécution ([`est_faute_execution`]). Ne remonte rien :
/// c'est un destructeur, et une panique de la synchronisation (cudarc en émet quand sa bibliothèque a disparu) est contenue, jamais
/// propagée dans un `drop`. C'est aussi le point d'injection `liberation`.
pub(crate) fn controle_destruction(origine: &str, synchroniser: impl FnOnce() -> Resultat<()>) {
    controle_destruction_sur(globale(), origine, synchroniser);
}

/// Cœur de [`controle_destruction`] sur un registre donné (essais).
pub(crate) fn controle_destruction_sur(sante: &Sante, origine: &str, synchroniser: impl FnOnce() -> Resultat<()>) {
    // Injection : la faute injectée est consignée par `avant` ; l'attente de l'appareil a lieu quand même (la libération qui suit en dépend).
    let _ = sante.avant(Point::Liberation, origine);
    if let Ok(Err(Erreur::Gpu(m))) = catch_unwind(AssertUnwindSafe(synchroniser)) {
        if est_faute_execution(&m) {
            sante.consigner_texte(origine, &m);
        }
    }
}

/// Le texte d'erreur décrit-il une faute d'exécution (noyau fautif, mémoire illégale) plutôt qu'un appareil absent ou une
/// ressource refusée ? Sert à choisir le message de sortie et à décider si l'erreur de l'attente qui précède une libération ou une
/// destruction ([`controle_destruction`], dont personne ne reçoit l'erreur) est consignée : c'est le seul cas où ce classement
/// influence le code de sortie (une erreur rendue à la commande donne le code 3 quel que soit son texte).
pub fn est_faute_execution(texte: &str) -> bool {
    let t = texte.to_lowercase();
    [
        "illegal memory access",
        "illegal_address",
        "memory_fault",
        "memory fault",
        "launch failure",
        "launch_failed",
        "launch timed out",
        "launch_timeout",
        "illegal instruction",
        "illegal_instruction",
        "misaligned address",
        "invalid program counter",
        "invalid_pc",
        "invalid address space",
        "invalid_address_space",
        "hardware_stack_error",
        "uncorrectable",
        "device-side assert",
    ]
    .iter()
    .any(|m| t.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyse_des_injections() {
        let ok = |t: &str| Injection::analyser(t).unwrap();
        assert_eq!(ok("lancement"), Injection { point: Point::Lancement, rang: 1 });
        assert_eq!(ok(" Synchronisation : 3 "), Injection { point: Point::Synchronisation, rang: 3 });
        assert_eq!(ok("copie:12"), Injection { point: Point::Copie, rang: 12 });
        assert_eq!(ok("LIBERATION:4"), Injection { point: Point::Liberation, rang: 4 });
        assert_eq!(ok("sortie"), Injection { point: Point::Sortie, rang: 1 });
        assert_eq!(ok("sortie:1").texte(), "sortie");
        for mauvais in ["", "bidule", "lancement:0", "lancement:x", "lancement:-1", "lancement:1:2", "sortie:2", ":3"] {
            assert!(Injection::analyser(mauvais).is_err(), "« {mauvais} » aurait dû être refusé");
        }
        // forme canonique : relue à l'identique
        for t in ["lancement:7", "synchronisation:1", "copie:2", "liberation:3", "sortie"] {
            assert_eq!(Injection::analyser(t).unwrap().texte(), t);
        }
    }

    #[test]
    fn texte_absent_ou_vide_ne_donne_aucune_injection_et_texte_invalide_est_refuse() {
        for t in [None, Some(""), Some("   ")] {
            let s = Sante::depuis_texte(t);
            assert!(s.verifier_configuration().is_ok());
            assert!(s.injection().is_none());
        }
        let s = Sante::depuis_texte(Some("synchronisation:zéro"));
        let e = s.verifier_configuration().unwrap_err();
        assert!(matches!(e, Erreur::Config(_)), "{e}");
        assert!(e.to_string().contains(ENV_INJECTION) && e.to_string().contains("rang"), "{e}");
        // une injection invalide fait aussi échouer l'opération : jamais ignorée en silence
        assert!(matches!(s.avant(Point::Lancement, "lancer"), Err(Erreur::Config(_))));
    }

    #[test]
    fn l_injection_tombe_sur_le_bon_appel_et_une_seule_fois() {
        let s = Sante::nouvelle(Some(Injection { point: Point::Synchronisation, rang: 3 }));
        // les autres points ne comptent pas pour celui-là
        for _ in 0..5 {
            s.avant(Point::Lancement, "lancer").unwrap();
            s.avant(Point::Copie, "lire").unwrap();
        }
        s.avant(Point::Synchronisation, "synchroniser").unwrap();
        s.avant(Point::Synchronisation, "synchroniser").unwrap();
        let e = s.avant(Point::Synchronisation, "synchroniser").unwrap_err();
        assert!(matches!(e, Erreur::Gpu(_)), "{e}");
        let t = e.to_string();
        assert!(t.contains("[injection KVU_INJECTION_FAUTE_GPU=synchronisation:3]"), "{t}");
        assert!(t.contains("an illegal memory access was encountered (700)"), "{t}");
        assert_eq!(e.code_sortie(), crate::commun::code::GPU_INDISPONIBLE);
        // le quatrième appel et les suivants passent : une panne, pas un état collant
        s.avant(Point::Synchronisation, "synchroniser").unwrap();
        // la faute injectée est consignée comme une vraie
        assert_eq!(s.fautes().len(), 1);
        assert_eq!(s.total(), 1);
    }

    #[test]
    fn le_lancement_injecte_porte_le_texte_d_un_echec_de_lancement() {
        let s = Sante::nouvelle(Some(Injection { point: Point::Lancement, rang: 1 }));
        let t = s.avant(Point::Lancement, "lancer").unwrap_err().to_string();
        assert!(t.contains("unspecified launch failure (719)"), "{t}");
        assert!(est_faute_execution(&t));
    }

    #[test]
    fn sans_injection_rien_n_est_consigne() {
        let s = Sante::nouvelle(None);
        for p in [Point::Lancement, Point::Synchronisation, Point::Copie, Point::Liberation, Point::Sortie] {
            for _ in 0..100 {
                s.avant(p, "op").unwrap();
            }
        }
        assert!(s.fautes().is_empty());
        assert!(s.bilan_sortie().is_none());
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn seules_les_erreurs_du_runtime_sont_consignees() {
        let s = Sante::nouvelle(None);
        s.consigner("lancer", &Erreur::Config("dimension nulle".into()));
        s.consigner("lire", &Erreur::Ressources("mémoire".into()));
        s.consigner("alloc", &Erreur::Format("x".into()));
        assert!(s.fautes().is_empty());
        s.consigner("synchroniser", &Erreur::Gpu("hipStreamSynchronize: an illegal memory access was encountered (700)".into()));
        assert_eq!(
            s.fautes(),
            vec!["synchroniser : hipStreamSynchronize: an illegal memory access was encountered (700)".to_string()]
        );
    }

    #[test]
    fn les_lignes_identiques_ne_sont_pas_repetees_et_le_nombre_de_lignes_est_borne() {
        let s = Sante::nouvelle(None);
        for _ in 0..10 {
            s.consigner_texte("fin d'appareil", "même faute");
        }
        assert_eq!(s.fautes().len(), 1);
        assert_eq!(s.total(), 10);
        for i in 0..(2 * FAUTES_CONSERVEES) {
            s.consigner_texte("op", &format!("faute {i}"));
        }
        assert_eq!(s.fautes().len(), FAUTES_CONSERVEES);
        assert_eq!(s.total(), 10 + 2 * FAUTES_CONSERVEES as u64);
    }

    #[test]
    fn le_bilan_de_sortie_rend_les_fautes_consignees() {
        let s = Sante::nouvelle(None);
        assert!(s.bilan_sortie().is_none());
        s.consigner_texte("fin d'appareil HIP", "hipStreamSynchronize: an illegal memory access was encountered (700)");
        let b = s.bilan_sortie().unwrap();
        assert_eq!(b.len(), 1);
        assert!(b[0].starts_with("fin d'appareil HIP : hipStreamSynchronize"), "{b:?}");
        // le bilan se relit à l'identique (il ne vide rien)
        assert_eq!(s.bilan_sortie().unwrap(), b);
    }

    #[test]
    fn la_sortie_injectee_est_rendue_une_seule_fois() {
        let s = Sante::nouvelle(Some(Injection { point: Point::Sortie, rang: 1 }));
        // `avant` ne touche pas au point de sortie, quel que soit le nombre d'appels
        for _ in 0..10 {
            s.avant(Point::Lancement, "lancer").unwrap();
        }
        let b = s.bilan_sortie().unwrap();
        assert_eq!(b.len(), 1);
        assert!(b[0].contains("[injection KVU_INJECTION_FAUTE_GPU=sortie] contrôle de sortie"), "{b:?}");
        assert!(est_faute_execution(&b[0]));
        assert!(s.bilan_sortie().is_none(), "la faute du point sortie ne se répète pas");
    }

    #[test]
    fn le_controle_par_lots_force_une_synchronisation_tous_les_n_lancements() {
        let s = Sante::nouvelle(None);
        for i in 1..LANCEMENTS_MAX_SANS_CONTROLE {
            assert!(!s.lancement_compte(), "contrôle forcé trop tôt au lancement {i}");
        }
        assert!(s.lancement_compte(), "le lancement n° {LANCEMENTS_MAX_SANS_CONTROLE} doit déclencher la synchronisation");
        // sans synchronisation réussie, le contrôle reste dû
        assert!(s.lancement_compte());
        s.controle_fait();
        assert!(!s.lancement_compte());
        for _ in 1..(LANCEMENTS_MAX_SANS_CONTROLE - 1) {
            assert!(!s.lancement_compte());
        }
        assert!(s.lancement_compte());
    }

    #[test]
    fn le_controle_de_destruction_ne_consigne_qu_une_faute_d_execution() {
        let s = Sante::nouvelle(None);
        controle_destruction_sur(&s, "fin d'appareil HIP", || Ok(()));
        assert!(s.fautes().is_empty(), "un appareil sain ne laisse rien");
        controle_destruction_sur(&s, "fin d'appareil HIP", || {
            Err(Erreur::Gpu("hipStreamSynchronize: an illegal memory access was encountered (700)".into()))
        });
        controle_destruction_sur(&s, "fin d'appareil CUDA", || {
            Err(Erreur::Gpu("DriverError(CUDA_ERROR_LAUNCH_FAILED, \"unspecified launch failure\")".into()))
        });
        assert_eq!(s.fautes().len(), 2);
        // le bruit d'un démontage (contexte déjà détruit, appareil invalide) ne fait pas échouer un run sain
        controle_destruction_sur(&s, "fin d'appareil HIP", || {
            Err(Erreur::Gpu("hipSetDevice : invalid device ordinal (101)".into()))
        });
        controle_destruction_sur(&s, "fin d'appareil CUDA", || {
            Err(Erreur::Gpu("DriverError(CUDA_ERROR_INVALID_CONTEXT, \"invalid device context\")".into()))
        });
        // une erreur qui n'est pas du runtime n'est pas une faute GPU
        controle_destruction_sur(&s, "fin d'appareil HIP", || Err(Erreur::Config("autre".into())));
        assert_eq!(s.fautes().len(), 2);
        // une panique de la synchronisation (bibliothèque absente) est contenue : ni propagée dans le `drop`, ni prise pour une faute
        controle_destruction_sur(&s, "fin d'appareil CUDA", || panic!("bibliothèque absente"));
        assert_eq!(s.fautes().len(), 2);
    }

    #[test]
    fn l_injection_de_liberation_consigne_la_faute_de_la_n_ieme_attente_sans_la_remonter() {
        let s = Sante::nouvelle(Some(Injection { point: Point::Liberation, rang: 2 }));
        // l'attente de l'appareil a lieu à chaque libération, faute injectée ou non (la libération qui suit en dépend)
        let attentes = std::cell::Cell::new(0);
        let attendre = || -> Resultat<()> {
            attentes.set(attentes.get() + 1);
            Ok(())
        };
        controle_destruction_sur(&s, "avant hipFree", attendre);
        assert!(s.fautes().is_empty(), "la première libération est saine");
        controle_destruction_sur(&s, "avant hipModuleUnload", attendre);
        assert_eq!(attentes.get(), 2, "l'attente a eu lieu aussi pour la libération injectée");
        let f = s.fautes();
        assert_eq!(f.len(), 1, "{f:?}");
        assert!(f[0].starts_with("avant hipModuleUnload : [injection KVU_INJECTION_FAUTE_GPU=liberation:2]"), "{f:?}");
        assert!(est_faute_execution(&f[0]));
        // une panne, pas un état collant
        controle_destruction_sur(&s, "fin d'appareil HIP", attendre);
        assert_eq!(s.fautes().len(), 1);
        assert_eq!(s.bilan_sortie().unwrap().len(), 1, "le contrôle de sortie la verra");
    }

    #[test]
    fn classement_des_textes_d_erreur() {
        for oui in [
            "hipStreamSynchronize: an illegal memory access was encountered (700)",
            "hipMemcpyDtoHAsync : an illegal memory access was encountered (700)",
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")",
            "DriverError(CUDA_ERROR_LAUNCH_FAILED, \"unspecified launch failure\")",
            "Queue error - HSA_STATUS_ERROR_MEMORY_FAULT",
            "hipModuleLaunchKernel: unspecified launch failure (719)",
            "DriverError(CUDA_ERROR_INVALID_PC, \"an invalid program counter was encountered\")",
        ] {
            assert!(est_faute_execution(oui), "{oui}");
        }
        for non in [
            "backend cuda non compilé : reconstruire kvu avec --features cuda",
            "HIP indisponible : dlopen failed for all candidates",
            "hipMalloc: out of memory (2)",
            "architecture HIP illisible sur le périphérique",
        ] {
            assert!(!est_faute_execution(non), "{non}");
        }
    }

    #[test]
    fn la_variable_d_environnement_a_le_nom_documente() {
        assert_eq!(ENV_INJECTION, "KVU_INJECTION_FAUTE_GPU");
    }
}
