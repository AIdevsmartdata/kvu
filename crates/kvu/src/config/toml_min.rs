//! Sous-ensemble strict de TOML pour les profils de machines de `kvu.toml`.

#![forbid(unsafe_code)]

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Valeur {
    Chaine(String),
    Entier(i64),
    Flottant(f64),
    Booleen(bool),
    Tableau(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entree {
    pub valeur: Valeur,
    pub ligne: usize,
    pub colonne: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SousTable {
    pub nom: String,
    pub ligne: usize,
    pub colonne: usize,
    pub entrees: Vec<(String, Entree)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub tables: Vec<SousTable>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErreurToml {
    pub ligne: usize,
    pub colonne: usize,
    pub message: String,
}

impl fmt::Display for ErreurToml {
    fn fmt(&self, sortie: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            sortie,
            "ligne {}, colonne {} : {}",
            self.ligne, self.colonne, self.message
        )
    }
}

impl std::error::Error for ErreurToml {}

impl Table {
    pub fn table(&self, nom: &str) -> Option<&SousTable> {
        self.tables.iter().find(|table| table.nom == nom)
    }
}

impl SousTable {
    pub fn get(&self, cle: &str) -> Option<&Entree> {
        self.entrees
            .iter()
            .find(|(nom, _)| nom == cle)
            .map(|(_, entree)| entree)
    }
}

#[derive(Clone, Copy)]
struct Position {
    ligne: usize,
    colonne: usize,
}

impl Position {
    fn erreur(self, message: &str) -> ErreurToml {
        ErreurToml {
            ligne: self.ligne,
            colonne: self.colonne,
            message: message.to_owned(),
        }
    }
}

struct Lecteur<'a> {
    texte: &'a str,
    indice: usize,
    position: Position,
}

impl<'a> Lecteur<'a> {
    fn reste(&self) -> &'a str {
        &self.texte[self.indice..]
    }

    fn voir(&self) -> Option<char> {
        self.reste().chars().next()
    }

    fn fin_ligne(&self) -> bool {
        self.reste().starts_with('\n') || self.reste().starts_with("\r\n")
    }

    // L'indice compte les octets, la colonne compte les caractères Unicode.
    // CRLF est consommé en une seule fin de ligne.
    fn avancer(&mut self) -> Option<char> {
        let caractere = self.voir()?;
        if self.reste().starts_with("\r\n") {
            self.indice += 2;
            self.position.ligne += 1;
            self.position.colonne = 1;
            return Some('\n');
        }
        self.indice += caractere.len_utf8();
        if caractere == '\n' {
            self.position.ligne += 1;
            self.position.colonne = 1;
        } else {
            self.position.colonne += 1;
        }
        Some(caractere)
    }

    fn espaces(&mut self) {
        while matches!(self.voir(), Some(' ' | '\t')) {
            self.avancer();
        }
    }

    fn commentaire(&mut self) -> Result<(), ErreurToml> {
        if self.voir() == Some('#') {
            while let Some(caractere) = self.voir() {
                if self.fin_ligne() {
                    break;
                }
                if controle_interdit(caractere) {
                    return Err(self
                        .position
                        .erreur("caractère de contrôle dans un commentaire"));
                }
                self.avancer();
            }
        }
        Ok(())
    }

    fn terminer_ligne(&mut self) -> Result<(), ErreurToml> {
        self.espaces();
        self.commentaire()?;
        if self.fin_ligne() {
            self.avancer();
            Ok(())
        } else if self.voir().is_none() {
            Ok(())
        } else {
            Err(self
                .position
                .erreur("caractère en trop après une valeur ou un en-tête"))
        }
    }

    fn cle_nue(&mut self) -> String {
        let debut = self.indice;
        while self.voir().is_some_and(caractere_cle) {
            self.avancer();
        }
        self.texte[debut..self.indice].to_owned()
    }

    fn entete(&mut self) -> Result<String, ErreurToml> {
        let ouverture = self.position;
        self.avancer();
        if self.voir() == Some('[') {
            return Err(ouverture.erreur("tableaux de tables [[...]] non pris en charge"));
        }
        self.espaces();
        let mut segments = Vec::new();
        loop {
            if self.voir().is_none() || self.fin_ligne() || self.voir() == Some('#') {
                return Err(ouverture.erreur("en-tête non fermé"));
            }
            let debut = self.position;
            let segment = self.cle_nue();
            if segment.is_empty() {
                let message = match self.voir() {
                    Some(']') if segments.is_empty() => "en-tête vide",
                    Some('.' | ']') => "segment vide dans l'en-tête",
                    Some('\'' | '"') => "segments entre guillemets non pris en charge",
                    _ => "caractère interdit dans l'en-tête",
                };
                return Err(debut.erreur(message));
            }
            segments.push(segment);
            self.espaces();
            match self.voir() {
                Some(']') => {
                    self.avancer();
                    return Ok(segments.join("."));
                }
                Some('.') => {
                    self.avancer();
                    self.espaces();
                }
                None | Some('#') => return Err(ouverture.erreur("en-tête non fermé")),
                _ if self.fin_ligne() => return Err(ouverture.erreur("en-tête non fermé")),
                _ => return Err(self.position.erreur("caractère interdit dans l'en-tête")),
            }
        }
    }

    fn paire(&mut self) -> Result<(String, Entree), ErreurToml> {
        let debut = self.position;
        if matches!(self.voir(), Some('\'' | '"')) {
            return Err(debut.erreur("clés entre guillemets non prises en charge"));
        }
        let cle = self.cle_nue();
        if cle.is_empty() {
            return Err(debut.erreur("clé nue non vide attendue"));
        }
        self.espaces();
        if self.voir() == Some('.') {
            return Err(self.position.erreur("clés pointées non prises en charge"));
        }
        if self.voir() != Some('=') {
            return Err(self.position.erreur("signe = attendu après la clé"));
        }
        self.avancer();
        self.espaces();
        let valeur = self.valeur()?;
        Ok((
            cle,
            Entree {
                valeur,
                ligne: debut.ligne,
                colonne: debut.colonne,
            },
        ))
    }

    fn valeur(&mut self) -> Result<Valeur, ErreurToml> {
        let debut = self.position;
        match self.voir() {
            None | Some('#') => Err(debut.erreur("valeur manquante")),
            _ if self.fin_ligne() => Err(debut.erreur("valeur manquante")),
            Some('\'' | '"') => self.chaine().map(Valeur::Chaine),
            Some('[') => self.tableau().map(Valeur::Tableau),
            Some('{') => Err(debut.erreur("tables en ligne non prises en charge")),
            _ => {
                let indice = self.indice;
                while self
                    .voir()
                    .is_some_and(|c| !matches!(c, ' ' | '\t' | '\r' | '\n' | '#'))
                {
                    self.avancer();
                }
                let mot = &self.texte[indice..self.indice];
                match mot {
                    "true" => Ok(Valeur::Booleen(true)),
                    "false" => Ok(Valeur::Booleen(false)),
                    _ => nombre(mot, debut),
                }
            }
        }
    }

    fn chaine(&mut self) -> Result<String, ErreurToml> {
        let ouverture = self.position;
        let guillemet = match self.avancer() {
            Some(c @ ('\'' | '"')) => c,
            _ => return Err(ouverture.erreur("chaîne attendue")),
        };
        let double = if guillemet == '"' { "\"\"" } else { "''" };
        if self.reste().starts_with(double) {
            return Err(ouverture.erreur("chaînes multilignes non prises en charge"));
        }
        let mut resultat = String::new();
        loop {
            let position = self.position;
            if self.fin_ligne() {
                return Err(ouverture.erreur("chaîne non fermée"));
            }
            match self.voir() {
                None => return Err(ouverture.erreur("chaîne non fermée")),
                Some(c) if c == guillemet => {
                    self.avancer();
                    return Ok(resultat);
                }
                Some(c) if controle_interdit(c) => {
                    return Err(position.erreur("caractère de contrôle brut dans une chaîne"));
                }
                Some('\\') if guillemet == '"' => {
                    self.avancer();
                    if self.voir().is_none() || self.fin_ligne() {
                        return Err(ouverture.erreur("chaîne non fermée"));
                    }
                    let echappe = match self.avancer() {
                        Some('"') => '"',
                        Some('\\') => '\\',
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('r') => '\r',
                        Some('b') => '\u{0008}',
                        Some('f') => '\u{000c}',
                        Some('u') => self.unicode(ouverture, position)?,
                        _ => return Err(position.erreur("échappement inconnu")),
                    };
                    resultat.push(echappe);
                }
                Some(c) => {
                    self.avancer();
                    resultat.push(c);
                }
            }
        }
    }

    fn unicode(&mut self, ouverture: Position, echappement: Position) -> Result<char, ErreurToml> {
        let mut code = 0;
        for _ in 0..4 {
            if self.voir().is_none() || self.fin_ligne() {
                return Err(ouverture.erreur("chaîne non fermée"));
            }
            let chiffre = self.voir().and_then(|c| c.to_digit(16)).ok_or_else(|| {
                self.position
                    .erreur("quatre chiffres hexadécimaux attendus après \\u")
            })?;
            code = code * 16 + chiffre;
            self.avancer();
        }
        char::from_u32(code)
            .ok_or_else(|| echappement.erreur("échappement Unicode invalide (demi-substitut)"))
    }

    fn espaces_tableau(&mut self) -> Result<(), ErreurToml> {
        loop {
            self.espaces();
            self.commentaire()?;
            if self.fin_ligne() {
                self.avancer();
            } else {
                return Ok(());
            }
        }
    }

    fn tableau(&mut self) -> Result<Vec<String>, ErreurToml> {
        let ouverture = self.position;
        self.avancer();
        let mut elements = Vec::new();
        loop {
            self.espaces_tableau()?;
            match self.voir() {
                None => return Err(ouverture.erreur("tableau non fermé")),
                Some(']') => {
                    self.avancer();
                    return Ok(elements);
                }
                Some('\'' | '"') => elements.push(self.chaine()?),
                _ => return Err(self.position.erreur("tableau de chaînes attendu")),
            }
            self.espaces_tableau()?;
            match self.voir() {
                Some(',') => {
                    self.avancer();
                }
                Some(']') => {
                    self.avancer();
                    return Ok(elements);
                }
                None => return Err(ouverture.erreur("tableau non fermé")),
                _ => return Err(self.position.erreur("virgule ou ] attendu dans le tableau")),
            }
        }
    }
}

fn caractere_cle(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-')
}

fn controle_interdit(c: char) -> bool {
    (c < ' ' && c != '\t') || c == '\u{007f}'
}

// Chaque souligné doit avoir un chiffre immédiatement de chaque côté.
fn chiffres_valides(texte: &str) -> bool {
    let octets = texte.as_bytes();
    !octets.is_empty()
        && octets.iter().enumerate().all(|(indice, c)| {
            c.is_ascii_digit()
                || (*c == b'_'
                    && indice > 0
                    && octets[indice - 1].is_ascii_digit()
                    && octets.get(indice + 1).is_some_and(u8::is_ascii_digit))
        })
}

fn sans_signe(texte: &str) -> &str {
    match texte.strip_prefix('+').or_else(|| texte.strip_prefix('-')) {
        Some(corps) => corps,
        None => texte,
    }
}

fn nombre(mot: &str, position: Position) -> Result<Valeur, ErreurToml> {
    let corps = sans_signe(mot);
    if matches!(corps, "inf" | "nan") {
        return Err(position.erreur("inf et nan non pris en charge"));
    }
    if ["0x", "0o", "0b"]
        .iter()
        .any(|base| corps.starts_with(base))
    {
        return Err(position.erreur("entiers hexadécimaux, octaux et binaires non pris en charge"));
    }
    let (mantisse, exposant) = match corps.find(['e', 'E']) {
        Some(indice) => (&corps[..indice], Some(&corps[indice + 1..])),
        None => (corps, None),
    };
    let (partie_entiere, fraction) = match mantisse.split_once('.') {
        Some((entiere, fraction)) => (entiere, Some(fraction)),
        None => (mantisse, None),
    };
    if !chiffres_valides(partie_entiere)
        || fraction.is_some_and(|chiffres| !chiffres_valides(chiffres))
        || exposant.is_some_and(|chiffres| !chiffres_valides(sans_signe(chiffres)))
    {
        return Err(position.erreur("valeur inconnue ou nombre décimal invalide"));
    }
    if partie_entiere.starts_with('0') && partie_entiere.len() > 1 {
        return Err(position.erreur("zéros initiaux interdits"));
    }
    let nettoye = mot.replace('_', "");
    if fraction.is_some() || exposant.is_some() {
        let valeur = nettoye
            .parse::<f64>()
            .map_err(|_| position.erreur("flottant invalide"))?;
        if !valeur.is_finite() {
            return Err(position.erreur("flottant hors limites finies"));
        }
        Ok(Valeur::Flottant(valeur))
    } else {
        nettoye
            .parse::<i64>()
            .map(Valeur::Entier)
            .map_err(|_| position.erreur("entier hors limites i64"))
    }
}

pub fn analyser(texte: &str) -> Result<Table, ErreurToml> {
    let mut lecteur = Lecteur {
        texte,
        indice: 0,
        position: Position {
            ligne: 1,
            colonne: 1,
        },
    };
    let mut resultat = Table {
        tables: vec![SousTable {
            nom: String::new(),
            ligne: 1,
            colonne: 1,
            entrees: Vec::new(),
        }],
    };
    let mut courante = 0;
    loop {
        lecteur.espaces();
        lecteur.commentaire()?;
        if lecteur.fin_ligne() {
            lecteur.avancer();
            continue;
        }
        if lecteur.voir().is_none() {
            return Ok(resultat);
        }
        let debut = lecteur.position;
        if lecteur.voir() == Some('[') {
            let nom = lecteur.entete()?;
            if resultat.table(&nom).is_some() {
                return Err(debut.erreur("en-tête de table dupliqué"));
            }
            resultat.tables.push(SousTable {
                nom,
                ligne: debut.ligne,
                colonne: debut.colonne,
                entrees: Vec::new(),
            });
            courante = resultat.tables.len() - 1;
        } else {
            let (cle, entree) = lecteur.paire()?;
            let table = &mut resultat.tables[courante];
            if table.get(&cle).is_some() {
                return Err(debut.erreur("clé dupliquée dans la table"));
            }
            table.entrees.push((cle, entree));
        }
        lecteur.terminer_ligne()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valeur(texte: &str) -> Valeur {
        analyser(texte)
            .unwrap()
            .table("")
            .unwrap()
            .get("x")
            .unwrap()
            .valeur
            .clone()
    }

    // Chaque cas produit un test distinct et vérifie les deux coordonnées.
    macro_rules! rejet {
        ($nom:ident, $texte:expr, $ligne:expr, $colonne:expr, $message:expr) => {
            #[test]
            fn $nom() {
                let erreur = analyser($texte).unwrap_err();
                assert_eq!(
                    (erreur.ligne, erreur.colonne),
                    ($ligne, $colonne),
                    "{erreur}"
                );
                assert!(erreur.message.contains($message), "{erreur}");
            }
        };
    }

    #[test]
    fn fichier_vide() {
        let resultat = analyser("").unwrap();
        assert_eq!(resultat.tables.len(), 1);
        let racine = resultat.table("").unwrap();
        assert!(racine.entrees.is_empty());
        assert_eq!((racine.ligne, racine.colonne), (1, 1));
    }

    #[test]
    fn racine_vide_et_tables_ordonnees() {
        let resultat = analyser("[z]\n[a]\n").unwrap();
        assert!(resultat.tables[0].entrees.is_empty());
        assert_eq!(
            resultat
                .tables
                .iter()
                .map(|t| t.nom.as_str())
                .collect::<Vec<_>>(),
            ["", "z", "a"]
        );
        assert!(resultat.table("absente").is_none());
        assert!(resultat.table("a").unwrap().get("absente").is_none());
    }

    #[test]
    fn commentaires_et_espaces() {
        let resultat =
            analyser("  # début\n\t\n x \t= 1 # valeur\n [ a . b ] # table\n\tcle = 'ok' # fin")
                .unwrap();
        assert_eq!(resultat.tables.len(), 2);
        assert_eq!(resultat.table("").unwrap().get("x").unwrap().colonne, 2);
        let table = resultat.table("a.b").unwrap();
        assert_eq!((table.ligne, table.colonne), (4, 2));
        assert_eq!(
            table.get("cle").unwrap().valeur,
            Valeur::Chaine("ok".into())
        );
    }

    #[test]
    fn fins_de_ligne_crlf() {
        let resultat = analyser("# é\r\nx = 1\r\n[a]\r\n y = true\r\n").unwrap();
        let entree = resultat.table("a").unwrap().get("y").unwrap();
        assert_eq!((entree.ligne, entree.colonne), (4, 2));
    }

    #[test]
    fn ordre_des_entrees() {
        let resultat = analyser("z = 1\na = 2\nm = 3").unwrap();
        assert_eq!(
            resultat.tables[0]
                .entrees
                .iter()
                .map(|(c, _)| c.as_str())
                .collect::<Vec<_>>(),
            ["z", "a", "m"]
        );
    }

    #[test]
    fn cles_nues_permises() {
        let resultat = analyser("9_a-Z = 1\n[9_a-Z . _]\nx = 2").unwrap();
        assert!(resultat.table("").unwrap().get("9_a-Z").is_some());
        assert!(resultat.table("9_a-Z._").is_some());
    }

    #[test]
    fn meme_cle_dans_plusieurs_tables() {
        assert!(analyser("x=1\n[a]\nx=2\n[b]\nx=3").is_ok());
    }

    #[test]
    fn chaines_et_tous_les_echappements() {
        assert_eq!(
            valeur(r#"x = "é\"\\\n\t\r\b\f\u00E9""#),
            Valeur::Chaine("é\"\\\n\t\r\u{0008}\u{000c}é".into())
        );
    }

    #[test]
    fn chaine_litterale_sans_echappement() {
        assert_eq!(
            valeur(r#"x = 'C:\temp\é # "'"#),
            Valeur::Chaine("C:\\temp\\é # \"".into())
        );
    }

    #[test]
    fn chaines_vides_et_tabulation_brute() {
        assert_eq!(valeur("x = \"\""), Valeur::Chaine(String::new()));
        assert_eq!(valeur("x = ''"), Valeur::Chaine(String::new()));
        assert_eq!(valeur("x = \"a\tb\""), Valeur::Chaine("a\tb".into()));
    }

    #[test]
    fn bornes_entieres_et_signes() {
        for (texte, attendu) in [
            ("0", 0),
            ("-0", 0),
            ("+42", 42),
            ("1_000", 1000),
            ("9223372036854775807", i64::MAX),
            ("-9223372036854775808", i64::MIN),
        ] {
            assert_eq!(valeur(&format!("x = {texte}")), Valeur::Entier(attendu));
        }
    }

    #[test]
    fn flottants_decimaux_et_exposants() {
        for (texte, attendu) in [
            ("1.5", 1.5),
            ("-0.25", -0.25),
            ("1e3", 1000.0),
            ("2.5E-3", 0.0025),
            ("1_000.5", 1000.5),
            ("+0.0", 0.0),
            ("1.2_5e+0_2", 125.0),
            ("0e00", 0.0),
        ] {
            assert_eq!(valeur(&format!("x = {texte}")), Valeur::Flottant(attendu));
        }
    }

    #[test]
    fn booleens() {
        assert_eq!(valeur("x=true"), Valeur::Booleen(true));
        assert_eq!(valeur("x=false"), Valeur::Booleen(false));
    }

    #[test]
    fn tableaux_vides_et_simples() {
        assert_eq!(valeur("x = []"), Valeur::Tableau(vec![]));
        assert_eq!(
            valeur("x = [\"a\", 'b',]"),
            Valeur::Tableau(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn tableau_multiligne_commente() {
        let resultat = analyser(
            "x = [ # début\r\n # commentaire\r\n \"é#\",\r\n 'b', # suite\r\n] # fin\r\n y = 2",
        )
        .unwrap();
        let racine = resultat.table("").unwrap();
        assert_eq!(
            racine.get("x").unwrap().valeur,
            Valeur::Tableau(vec!["é#".into(), "b".into()])
        );
        let entree = racine.get("y").unwrap();
        assert_eq!((entree.ligne, entree.colonne), (6, 2));
    }

    #[test]
    fn colonne_de_cle_apres_chaine_accentuee() {
        let resultat = analyser("x = \"é漢🙂\"\n   y = 'été'\n").unwrap();
        let entree = resultat.table("").unwrap().get("y").unwrap();
        assert_eq!((entree.ligne, entree.colonne), (2, 4));
    }

    #[test]
    fn affichage_erreur() {
        let erreur = analyser("x = abc").unwrap_err();
        assert_eq!(
            erreur.to_string(),
            format!("ligne 1, colonne 5 : {}", erreur.message)
        );
        let _: &dyn std::error::Error = &erreur;
    }

    #[test]
    fn exemple_realiste_kvu() {
        let texte = r#"# Configuration des machines
version = 1
actif = true
nom = "Atelier de Kévin"

[profils.acemagic]
hote = "acemagic"
systeme = 'Linux'
coeurs = 16
memoire_go = 32
charge_max = 0.85
gpu = false
roles = ["calcul", "pilotage",]
repertoire = '/srv/projet'

[profils.gamer]
hote = "gamer"
systeme = "Linux"
coeurs = 24
memoire_go = 64
charge_max = 9.5e-1
gpu = true
roles = [
  "inférence", # modèle local
  'rendu',
]
repertoire = '/srv/modeles'
"#;
        let resultat = analyser(texte).unwrap();
        assert_eq!(resultat.tables.len(), 3);
        let acemagic = resultat.table("profils.acemagic").unwrap();
        let gamer = resultat.table("profils.gamer").unwrap();
        assert_eq!((acemagic.ligne, gamer.ligne), (6, 16));
        assert_eq!(
            acemagic.get("memoire_go").unwrap().valeur,
            Valeur::Entier(32)
        );
        assert_eq!(
            gamer.get("charge_max").unwrap().valeur,
            Valeur::Flottant(0.95)
        );
        assert_eq!(
            gamer.get("roles").unwrap().valeur,
            Valeur::Tableau(vec!["inférence".into(), "rendu".into()])
        );
    }

    rejet!(doublon_racine, "x=1\n  x=2", 2, 3, "clé dupliquée");
    rejet!(doublon_table, "[a]\nx=1\n x=2", 3, 2, "clé dupliquée");
    rejet!(doublon_entete, "[a.b]\n [ a . b ]", 2, 2, "table dupliqué");
    rejet!(chaine_non_fermee, "x = \"abc\n", 1, 5, "chaîne non fermée");
    rejet!(litterale_non_fermee, "x = 'abc", 1, 5, "chaîne non fermée");
    rejet!(
        chaine_crlf_non_fermee,
        "# début\r\nx = \"é\r\n",
        2,
        5,
        "chaîne non fermée"
    );
    rejet!(
        tableau_non_ferme,
        "x = [\n 'a', # fin\n",
        1,
        5,
        "tableau non fermé"
    );
    rejet!(
        tableau_non_ferme_sans_virgule,
        "x = ['a'",
        1,
        5,
        "tableau non fermé"
    );
    rejet!(entete_non_ferme, "  [a", 1, 3, "en-tête non fermé");
    rejet!(entete_non_ferme_ligne, "[a\nx=1", 1, 1, "en-tête non fermé");
    rejet!(entete_vide, "[]", 1, 2, "en-tête vide");
    rejet!(segment_vide, "[a..b]", 1, 4, "segment vide");
    rejet!(segment_final_vide, "[a.]", 1, 4, "segment vide");
    rejet!(type_inattendu, "x = abc", 1, 5, "valeur inconnue");
    rejet!(
        tableau_mixte,
        "x = [\"a\", 2]",
        1,
        11,
        "tableau de chaînes attendu"
    );
    rejet!(
        tableau_imbrique,
        "x = [[\"a\"]]",
        1,
        6,
        "tableau de chaînes attendu"
    );
    rejet!(
        tableau_mixte_multiligne,
        "x=[\n 'é',\n  true\n]",
        3,
        3,
        "tableau de chaînes attendu"
    );
    rejet!(entier_trop_grand, "x = 9223372036854775808", 1, 5, "i64");
    rejet!(entier_trop_petit, "x = -9223372036854775809", 1, 5, "i64");
    rejet!(zero_initial, "x = 01", 1, 5, "zéros initiaux");
    rejet!(zero_initial_signe, "x = -0_1", 1, 5, "zéros initiaux");
    rejet!(zero_initial_flottant, "x = 01.5", 1, 5, "zéros initiaux");
    rejet!(cle_pointee, "a.b = 1", 1, 2, "clés pointées");
    rejet!(tableau_de_tables, " [[a]]", 1, 2, "tableaux de tables");
    rejet!(table_en_ligne, "x = { a = 1 }", 1, 5, "tables en ligne");
    rejet!(
        chaine_multiligne,
        "x = \"\"\"abc\"\"\"",
        1,
        5,
        "multilignes"
    );
    rejet!(litterale_multiligne, "x = '''abc'''", 1, 5, "multilignes");
    rejet!(
        echappement_inconnu,
        r#"x = "a\q""#,
        1,
        7,
        "échappement inconnu"
    );
    rejet!(
        echappement_unicode_long,
        r#"x = "\U00000041""#,
        1,
        6,
        "échappement inconnu"
    );
    rejet!(
        unicode_invalide,
        r#"x = "\uD800""#,
        1,
        6,
        "Unicode invalide"
    );
    rejet!(
        unicode_non_hexadecimal,
        r#"x = "\u00G0""#,
        1,
        10,
        "hexadécimaux"
    );
    rejet!(controle_brut, "x = \"é\u{0001}\"", 1, 7, "contrôle brut");
    rejet!(controle_litteral, "x = 'a\u{007f}'", 1, 7, "contrôle brut");
    rejet!(caractere_en_trop, "x = 1 abc", 1, 7, "caractère en trop");
    rejet!(
        colonne_unicode_apres_valeur,
        "x = \"é漢🙂\" y=2",
        1,
        11,
        "caractère en trop"
    );
    rejet!(
        colonne_unicode_dans_tableau,
        "x = [\"é\", 1]",
        1,
        11,
        "tableau de chaînes attendu"
    );
    rejet!(valeur_manquante, "x =", 1, 4, "valeur manquante");
    rejet!(
        valeur_manquante_commentaire,
        "x = # rien",
        1,
        5,
        "valeur manquante"
    );
    rejet!(egal_manquant, "cle  1", 1, 6, "signe = attendu");
    rejet!(cle_vide, " = 1", 1, 2, "clé nue non vide");
    rejet!(
        cle_entre_guillemets,
        "\"cle\" = 1",
        1,
        1,
        "clés entre guillemets"
    );
    rejet!(
        entete_entre_guillemets,
        "[\"a\"]",
        1,
        2,
        "segments entre guillemets"
    );
    rejet!(
        entete_caractere_interdit,
        "[a/b]",
        1,
        3,
        "caractère interdit"
    );
    rejet!(cle_unicode_interdite, "é = 1", 1, 1, "clé nue non vide");
    rejet!(nan_interdit, "x = nan", 1, 5, "nan");
    rejet!(inf_interdit, "x = +inf", 1, 5, "inf");
    rejet!(inf_negatif_interdit, "x = -inf", 1, 5, "inf");
    rejet!(nan_signe_interdit, "x = -nan", 1, 5, "nan");
    rejet!(date_interdite, "x = 2026-10-25", 1, 5, "valeur inconnue");
    rejet!(heure_interdite, "x = 12:30:00", 1, 5, "valeur inconnue");
    rejet!(hexadecimal_interdit, "x = 0x10", 1, 5, "hexadécimaux");
    rejet!(octal_interdit, "x = 0o10", 1, 5, "octaux");
    rejet!(binaire_interdit, "x = 0b10", 1, 5, "binaires");
    rejet!(fraction_sans_entier, "x = .5", 1, 5, "valeur inconnue");
    rejet!(fraction_sans_chiffre, "x = 5.", 1, 5, "valeur inconnue");
    rejet!(souligne_double, "x = 1__0", 1, 5, "valeur inconnue");
    rejet!(souligne_final, "x = 1_", 1, 5, "valeur inconnue");
    rejet!(souligne_initial, "x = _1", 1, 5, "valeur inconnue");
    rejet!(souligne_pres_point, "x = 1_.5", 1, 5, "valeur inconnue");
    rejet!(exposant_vide, "x = 1e+", 1, 5, "valeur inconnue");
    rejet!(exposants_multiples, "x = 1e2e3", 1, 5, "valeur inconnue");
    rejet!(flottant_infini, "x = 1e9999", 1, 5, "limites finies");
    rejet!(booleen_majuscule, "x = True", 1, 5, "valeur inconnue");
    rejet!(virgule_manquante, "x = ['a' 'b']", 1, 10, "virgule ou ]");
    rejet!(
        virgule_double,
        "x = ['a',,]",
        1,
        10,
        "tableau de chaînes attendu"
    );
    rejet!(texte_apres_entete, "[a] x=1", 1, 5, "caractère en trop");
    rejet!(cr_isole, "x=1\ry=2", 1, 4, "caractère en trop");
    rejet!(controle_commentaire, "# é\u{0000}", 1, 4, "contrôle");
    rejet!(octet_nul_final, "x=1\n\0", 2, 1, "clé nue non vide");
}
