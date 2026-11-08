//! Binaire léger `kvu` : la même ligne de commande que `kevinotron <sous-commande>`, sans le
//! moteur historique (build rapide, bancs, tests). Propriétaire : T9.

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    std::process::exit(kvu::cli::principal(argv));
}
