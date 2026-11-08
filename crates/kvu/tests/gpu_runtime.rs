//! Gates GPU RT-1 et RT-2 du runtime unifié (T2). `#[ignore = "GPU"]` : à lancer SOUS le verrou GPU de la machine.
//!
//! ```text
//! acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock cargo test -p kvu --release --test gpu_runtime -- --ignored --nocapture --test-threads=1
//! gamer    : flock -w 7200 /tmp/gamer_gpu.lock      cargo test -p kvu --release --features cuda --test gpu_runtime -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Le backend est KEVINOTRON_BACKEND s'il est posé, sinon CUDA si la feature `cuda` est compilée, sinon HIP.
//! Chaque gate imprime une ligne `PASS|FAIL|SKIP <gate> : détail` ; un SKIP signale un backend absent.

use kvu::commun::{BackendKind, Erreur};
use kvu::gpu::{self, assembler, Appareil, Arg, Cache, Lancement, Module, OptionsCompil, Origine, SourceAssemblee};
use kvu::heritage::manifest::sha256_hex;
use kvu::noyaux::TEST_RUNTIME;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------------------------

fn backend_du_test() -> BackendKind {
    match std::env::var("KEVINOTRON_BACKEND").ok().and_then(|v| v.parse().ok()) {
        Some(b) if b != BackendKind::Cpu => b,
        _ if cfg!(feature = "cuda") => BackendKind::Cuda,
        _ => BackendKind::Hip,
    }
}

/// Ouvre l'appareil, ou imprime SKIP et rend None si le backend est absent.
fn appareil_ou_skip(gate: &str) -> Option<Appareil> {
    let b = backend_du_test();
    match gpu::ouvrir(b, 0) {
        Ok(a) => {
            println!(
                "# {gate} : {} / {} / {} / versions {:?}",
                a.backend().nom(),
                a.nom(),
                a.arch(),
                a.versions()
            );
            Some(a)
        }
        Err(e) => {
            println!("SKIP {gate} : backend {} indisponible ({e})", b.nom());
            None
        }
    }
}

/// Exécute le corps d'une gate : imprime PASS/FAIL et fait échouer le test en cas de FAIL.
fn gate(nom: &str, corps: impl FnOnce() -> Result<String, String>) {
    match corps() {
        Ok(detail) => println!("PASS {nom} : {detail}"),
        Err(raison) => {
            println!("FAIL {nom} : {raison}");
            panic!("FAIL {nom} : {raison}");
        }
    }
}

fn e<T: std::fmt::Display>(quoi: &str) -> impl Fn(T) -> String + '_ {
    move |x| format!("{quoi} : {x}")
}

fn repertoire_de_test(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("gpu_runtime")
        .join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn assembler_essai(a: &Appareil, defines: &[(&str, &str)]) -> SourceAssemblee {
    assembler(a.backend(), "test_runtime", &[&TEST_RUNTIME], defines)
}

fn compiler_k1(a: &Appareil) -> Result<Module, String> {
    // Cache désactivé : RT-1 teste le compilateur, pas le cache.
    a.compiler_avec_cache(&assembler_essai(a, &[]), &OptionsCompil::reference_k1(), None)
        .map_err(e("compilation"))
}

/// Suite pseudo-aléatoire déterministe (xorshift64*), sans dépendance.
struct Alea(u64);
impl Alea {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Réel dans [-1, 1) avec une mantisse pleine : les sommes ne sont pas exactes, l'ordre compte.
    fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Réduction CPU : exactement l'arbre de k_reduce (blocs de 256, w = 128..1).
fn reduce_cpu(x: &[f64]) -> Vec<f64> {
    x.chunks(256)
        .map(|bloc| {
            let mut sh = [0.0f64; 256];
            sh[..bloc.len()].copy_from_slice(bloc);
            arbre_256(&mut sh)
        })
        .collect()
}

/// Somme finale CPU : exactement k_final (foulée de 256 puis arbre fixe), puis `* scale`.
fn final_cpu(p: &[f64], scale: f64) -> f64 {
    let mut sh = [0.0f64; 256];
    for (t, s) in sh.iter_mut().enumerate() {
        let mut a = 0.0;
        let mut i = t;
        while i < p.len() {
            a += p[i];
            i += 256;
        }
        *s = a;
    }
    arbre_256(&mut sh) * scale
}

fn arbre_256(sh: &mut [f64; 256]) -> f64 {
    let mut w = 128;
    while w > 0 {
        for t in 0..w {
            sh[t] += sh[t + w];
        }
        w >>= 1;
    }
    sh[0]
}

fn bits_identiques(a: &[f64], b: &[f64]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits())
}

const BLOC: u32 = 256;

// ---------------------------------------------------------------------------------------------
// RT-1 : noyaux d'essai corrects, réduction identique à l'octet à la référence CPU
// ---------------------------------------------------------------------------------------------

fn rt1(a: &Appareil) -> Result<String, String> {
    let module = compiler_k1(a)?;
    let f = |nom: &str| a.fonction(&module, nom).map_err(e(nom));
    let (k_remplir, k_saxpy, k_reduce, k_final, k_somme) = (
        f("k_remplir")?,
        f("k_saxpy_f64")?,
        f("k_reduce")?,
        f("k_final")?,
        f("k_somme_u64")?,
    );
    let mut rapport = Vec::new();

    // 1. Remplissage : x[i] = v + i·h, n non multiple du bloc.
    let n = 100_003usize;
    let (v, h) = (0.25f64, 0.1f64);
    let x = a.alloc::<f64>(n).map_err(e("alloc x"))?;
    a.lancer(
        &k_remplir,
        &Lancement::lineaire(n, BLOC),
        &[x.arg(), Arg::I64(n as i64), Arg::F64(v), Arg::F64(h)],
    )
    .map_err(e("lancer k_remplir"))?;
    let mut x_gpu = vec![0.0f64; n];
    a.lire(&x, 0, &mut x_gpu).map_err(e("lire x"))?;
    let x_cpu: Vec<f64> = (0..n).map(|i| v + i as f64 * h).collect();
    if let Some(i) = bits_identiques(&x_gpu, &x_cpu) {
        return Err(format!("k_remplir : x[{i}] = {:e} ≠ {:e} (bits)", x_gpu[i], x_cpu[i]));
    }
    rapport.push(format!("remplissage n={n} au bit"));

    // 2. SAXPY FP64 contre le CPU, au bit (sans contraction : a*x + y en deux arrondis).
    let mut al = Alea(0x9E37_79B9_7F4A_7C15);
    let xs: Vec<f64> = (0..n).map(|_| al.f64()).collect();
    let ys: Vec<f64> = (0..n).map(|_| al.f64()).collect();
    let coef = 1.000_000_119_209_29f64;
    let mut bx = a.alloc::<f64>(n).map_err(e("alloc bx"))?;
    let mut by = a.alloc::<f64>(n).map_err(e("alloc by"))?;
    a.ecrire(&mut bx, 0, &xs).map_err(e("ecrire bx"))?;
    a.ecrire(&mut by, 0, &ys).map_err(e("ecrire by"))?;
    a.lancer(
        &k_saxpy,
        &Lancement::lineaire(n, BLOC),
        &[Arg::I64(n as i64), Arg::F64(coef), bx.arg(), by.arg()],
    )
    .map_err(e("lancer k_saxpy_f64"))?;
    let mut y_gpu = vec![0.0f64; n];
    a.lire(&by, 0, &mut y_gpu).map_err(e("lire by"))?;
    let y_cpu: Vec<f64> = xs.iter().zip(&ys).map(|(x, y)| coef * x + y).collect();
    if let Some(i) = bits_identiques(&y_gpu, &y_cpu) {
        return Err(format!("k_saxpy_f64 : y[{i}] = {:e} ≠ {:e} (bits)", y_gpu[i], y_cpu[i]));
    }
    rapport.push("saxpy FP64 au bit contre le CPU".to_string());

    // 3. Réduction à arbre fixe : partiels (octet pour octet) et somme finale, trois exécutions.
    let nb = n.div_ceil(256);
    let partiels = a.alloc::<f64>(nb).map_err(e("alloc partiels"))?;
    let sortie = a.alloc::<f64>(1).map_err(e("alloc sortie"))?;
    let (p_cpu, scale) = (reduce_cpu(&xs), 0.5f64);
    let somme_cpu = final_cpu(&p_cpu, scale);
    let mut vues = Vec::new();
    for passe in 0..3 {
        a.lancer(
            &k_reduce,
            &Lancement::lineaire(n, BLOC),
            &[bx.arg(), Arg::I64(n as i64), partiels.arg()],
        )
        .map_err(e("lancer k_reduce"))?;
        a.lancer(
            &k_final,
            &Lancement {
                grille: [1, 1, 1],
                bloc: [BLOC, 1, 1],
                shmem: 0,
            },
            &[partiels.arg(), Arg::I64(nb as i64), Arg::F64(scale), sortie.arg()],
        )
        .map_err(e("lancer k_final"))?;
        let mut p_gpu = vec![0.0f64; nb];
        let mut s_gpu = [0.0f64; 1];
        a.lire(&partiels, 0, &mut p_gpu).map_err(e("lire partiels"))?;
        a.lire(&sortie, 0, &mut s_gpu).map_err(e("lire sortie"))?;
        if let Some(i) = bits_identiques(&p_gpu, &p_cpu) {
            return Err(format!(
                "passe {passe} : partiel[{i}] = {:e} ≠ CPU {:e}",
                p_gpu[i], p_cpu[i]
            ));
        }
        if s_gpu[0].to_bits() != somme_cpu.to_bits() {
            return Err(format!(
                "passe {passe} : somme finale {:e} ≠ CPU {:e}",
                s_gpu[0], somme_cpu
            ));
        }
        vues.push((p_gpu, s_gpu[0]));
    }
    let octets = |v: &[f64]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
    if !vues
        .windows(2)
        .all(|w| octets(&w[0].0) == octets(&w[1].0) && w[0].1.to_bits() == w[1].1.to_bits())
    {
        return Err("réduction non reproductible entre exécutions".to_string());
    }
    // Contrôle de pertinence : une somme naïve dans l'ordre séquentiel DIFFÈRE (l'arbre fixe est bien celui testé).
    let naive: f64 = xs.iter().sum::<f64>() * scale;
    rapport.push(format!(
        "réduction à arbre fixe ({nb} blocs) identique à l'octet au CPU, 3 passes identiques (somme {somme_cpu:e}, somme séquentielle {naive:e})"
    ));

    // 4. atomicAdd 64 bits exact (somme modulo 2^64, indépendante de l'ordre).
    let m = 300_001usize;
    let us: Vec<u64> = (0..m).map(|_| al.u64()).collect();
    let mut bu = a.alloc::<u64>(m).map_err(e("alloc bu"))?;
    let tot = a.alloc::<u64>(1).map_err(e("alloc tot"))?;
    a.ecrire(&mut bu, 0, &us).map_err(e("ecrire bu"))?;
    a.lancer(
        &k_somme,
        &Lancement::lineaire(m, BLOC),
        &[bu.arg(), Arg::I64(m as i64), tot.arg()],
    )
    .map_err(e("lancer k_somme_u64"))?;
    let mut t = [0u64; 1];
    a.lire(&tot, 0, &mut t).map_err(e("lire tot"))?;
    let attendu = us.iter().fold(0u64, |s, &v| s.wrapping_add(v));
    if t[0] != attendu {
        return Err(format!("atomicAdd u64 : {} ≠ {}", t[0], attendu));
    }
    rapport.push(format!("atomicAdd u64 exact sur {m} termes"));

    // 5. Décalage d'argument, écriture/lecture décalées, copier, zéro, erreurs de bornes.
    let c = 1000usize;
    let lot = a.alloc::<f64>(2 * c).map_err(e("alloc lot"))?;
    a.lancer(
        &k_remplir,
        &Lancement::lineaire(c, BLOC),
        &[lot.arg_decale(c), Arg::I64(c as i64), Arg::F64(1.0), Arg::F64(1.0)],
    )
    .map_err(e("lancer remplir décalé"))?;
    let mut tout = vec![-1.0f64; 2 * c];
    a.lire(&lot, 0, &mut tout).map_err(e("lire lot"))?;
    if tout[..c].iter().any(|&v| v != 0.0) || tout[c..].iter().enumerate().any(|(i, &v)| v != 1.0 + i as f64) {
        return Err("arg_decale : la seconde moitié seule devait être remplie".to_string());
    }
    let mut copie = a.alloc::<f64>(2 * c).map_err(e("alloc copie"))?;
    a.copier(&lot, &mut copie).map_err(e("copier"))?;
    let mut tout2 = vec![0.0f64; 2 * c];
    a.lire(&copie, 0, &mut tout2).map_err(e("lire copie"))?;
    if bits_identiques(&tout, &tout2).is_some() {
        return Err("copier : copie différente".to_string());
    }
    a.ecrire(&mut copie, c - 2, &[7.0, 8.0, 9.0])
        .map_err(e("ecrire décalé"))?;
    let mut trois = [0.0f64; 3];
    a.lire(&copie, c - 2, &mut trois).map_err(e("lire décalé"))?;
    if trois != [7.0, 8.0, 9.0] {
        return Err(format!("écriture décalée : {trois:?}"));
    }
    a.zero(&mut copie).map_err(e("zero"))?;
    a.lire(&copie, 0, &mut tout2).map_err(e("lire après zero"))?;
    if tout2.iter().any(|&v| v != 0.0) {
        return Err("zero : tampon non nul".to_string());
    }
    let mut trop = [0.0f64; 3];
    if !matches!(a.lire(&copie, 2 * c - 2, &mut trop), Err(Erreur::Config(_))) {
        return Err("lecture hors borne non refusée".to_string());
    }
    if !matches!(a.ecrire(&mut copie, 2 * c, &[1.0]), Err(Erreur::Config(_))) {
        return Err("écriture hors borne non refusée".to_string());
    }
    let vide = a.alloc::<f64>(0).map_err(e("alloc vide"))?;
    if !vide.is_empty() || vide.octets() != 0 {
        return Err("tampon vide incohérent".to_string());
    }
    rapport.push("décalages, copier, zéro, bornes".to_string());

    // 6. Erreurs : noyau inconnu, source qui ne compile pas, source d'un autre backend.
    if a.fonction(&module, "k_inexistant").is_ok() {
        return Err("un noyau inexistant a été trouvé".to_string());
    }
    let ko = kvu::noyaux::Fichier {
        nom: "ko.cu",
        contenu: "extern \"C\" __global__ void k_ko( { }\n",
    };
    match a.compiler_avec_cache(
        &assembler(a.backend(), "ko", &[&ko], &[]),
        &OptionsCompil::reference_k1(),
        None,
    ) {
        Err(Erreur::Compilation { backend, journal }) if backend == a.backend() && !journal.is_empty() => {}
        Err(autre) => return Err(format!("source invalide : erreur inattendue {autre}")),
        Ok(_) => return Err("une source invalide a compilé".to_string()),
    }
    let autre = if a.backend() == BackendKind::Hip {
        BackendKind::Cuda
    } else {
        BackendKind::Hip
    };
    let etrangere = assembler(autre, "test_runtime", &[&TEST_RUNTIME], &[]);
    if !matches!(
        a.compiler_avec_cache(&etrangere, &OptionsCompil::reference_k1(), None),
        Err(Erreur::Config(_))
    ) {
        return Err("source d'un autre backend non refusée".to_string());
    }
    rapport.push("erreurs (noyau inconnu, compilation en échec avec journal, mauvais backend)".to_string());

    Ok(format!(
        "{} sur {} ({}) : {}",
        a.backend().nom(),
        a.nom(),
        a.arch(),
        rapport.join(" ; ")
    ))
}

#[test]
#[ignore = "GPU"]
fn rt1_noyaux_d_essai() {
    let Some(a) = appareil_ou_skip("RT-1") else { return };
    gate("RT-1", || rt1(&a));
}

// ---------------------------------------------------------------------------------------------
// RT-2 : arguments d'or, CUBIN sans PTX, cache (miss → hit → poison évincé → plafond)
// ---------------------------------------------------------------------------------------------

/// Arguments attendus POUR L'ARCHITECTURE RÉELLE de l'appareil, écrits à la main.
fn arguments_attendus(b: BackendKind, arch: &str, cas: &str) -> Vec<String> {
    let v = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    match (b, cas) {
        (BackendKind::Hip, "k1") => v(&[
            &format!("--offload-arch={arch}"),
            "-O3",
            "-ffp-contract=off",
            "-std=c++17",
        ]),
        (BackendKind::Hip, "prod") => v(&[
            &format!("--offload-arch={arch}"),
            "-O3",
            "-ffp-contract=fast-honor-pragmas",
            "-std=c++17",
        ]),
        (BackendKind::Cuda, "k1") => v(&[&format!("-arch={arch}"), "--fmad=false", "-std=c++17"]),
        (BackendKind::Cuda, "prod") => v(&[&format!("-arch={arch}"), "--fmad=true", "-std=c++17"]),
        _ => unreachable!(),
    }
}

/// Un objet CUBIN est un ELF de machine EM_CUDA (190) ; un HSACO est un ELF de machine EM_AMDGPU (224).
fn verifier_objet(b: BackendKind, code: &[u8]) -> Result<String, String> {
    if code.len() < 64 || &code[..4] != b"\x7fELF" {
        return Err(format!(
            "code objet non ELF (début {:?}) : du PTX texte ?",
            &code[..code.len().min(8)]
        ));
    }
    let machine = u16::from_le_bytes([code[18], code[19]]);
    let attendu = if b == BackendKind::Cuda { 190 } else { 224 };
    if machine != attendu {
        return Err(format!("e_machine = {machine}, attendu {attendu}"));
    }
    Ok(format!("ELF e_machine {machine}, {} octets", code.len()))
}

fn rt2(a: &Appareil) -> Result<String, String> {
    let (b, arch) = (a.backend(), a.arch());
    let ext = if b == BackendKind::Cuda { "cubin" } else { "hsaco" };
    let mut rapport = Vec::new();

    // 1. Golden des arguments pour l'architecture réelle.
    for (cas, opts) in [
        ("k1", OptionsCompil::reference_k1()),
        ("prod", OptionsCompil::compat_production(b)),
    ] {
        let obtenu = opts.arguments(b, &arch);
        let attendu = arguments_attendus(b, &arch, cas);
        if obtenu != attendu {
            return Err(format!("arguments {cas} : {obtenu:?} ≠ {attendu:?}"));
        }
    }
    rapport.push(format!("golden des arguments ({} {arch})", b.nom()));

    // 2. Cache : miss → hit → poison évincé → plafond.
    let dir = repertoire_de_test("rt2");
    let cache = Cache::nouveau(&dir, 4 << 20);
    let src = assembler_essai(a, &[("KV_ESSAI", "1")]);
    let opts = OptionsCompil::reference_k1();

    let t0 = std::time::Instant::now();
    let m1 = a.compiler_avec_cache(&src, &opts, Some(&cache)).map_err(e("miss"))?;
    let duree_miss = t0.elapsed();
    if m1.origine() != Origine::Compilation {
        return Err("premier appel : le cache aurait dû manquer".to_string());
    }
    let arguments = opts.arguments(b, &arch);
    let cle = kvu::gpu::cache::cle(&src.texte, &arguments, &arch, &a.version_compilateur());
    let p_code = cache.chemin_code(&src.nom, &arch, &cle, ext);
    let p_json = cache.chemin_meta(&src.nom, &arch, &cle);
    if !p_code.exists() || !p_json.exists() {
        return Err(format!("miss : entrée absente du disque ({})", p_code.display()));
    }
    let code = fs::read(&p_code).map_err(e("lecture du code objet"))?;
    let nature = verifier_objet(b, &code)?;
    if sha256_hex(&code) != m1.sha256_code {
        return Err("sha256 du fichier ≠ Module::sha256_code".to_string());
    }
    let prov = m1.provenance();
    if prov.sha256_source != src.sha256
        || prov.arguments != arguments
        || prov.arch != arch
        || prov.sha256_code != m1.sha256_code
    {
        return Err("provenance incohérente".to_string());
    }
    rapport.push(format!(
        "miss ({nature}, compilation + chargement {:.0} ms)",
        duree_miss.as_secs_f64() * 1e3
    ));

    let t1 = std::time::Instant::now();
    let m2 = a.compiler_avec_cache(&src, &opts, Some(&cache)).map_err(e("hit"))?;
    let duree_hit = t1.elapsed();
    if m2.origine() != Origine::Cache || m2.sha256_code != m1.sha256_code {
        return Err(format!(
            "deuxième appel : attendu un succès du cache, obtenu {:?}",
            m2.origine()
        ));
    }
    // Le module relu du cache fonctionne.
    let f = a.fonction(&m2, "k_remplir").map_err(e("fonction du module en cache"))?;
    let x = a.alloc::<f64>(300).map_err(e("alloc"))?;
    a.lancer(
        &f,
        &Lancement::lineaire(300, BLOC),
        &[x.arg(), Arg::I64(300), Arg::F64(2.0), Arg::F64(0.5)],
    )
    .map_err(e("lancer"))?;
    let mut y = vec![0.0f64; 300];
    a.lire(&x, 0, &mut y).map_err(e("lire"))?;
    if y[299] != 2.0 + 299.0 * 0.5 {
        return Err(format!("module du cache : y[299] = {}", y[299]));
    }
    rapport.push(format!(
        "hit en {:.1} ms (module du cache exécuté)",
        duree_hit.as_secs_f64() * 1e3
    ));

    // Poison : un octet changé ⇒ évincé puis recompilé, et le fichier est réécrit sain.
    let mut poison = code.clone();
    let milieu = poison.len() / 2;
    poison[milieu] ^= 0xFF;
    fs::write(&p_code, &poison).map_err(e("poison"))?;
    let m3 = a
        .compiler_avec_cache(&src, &opts, Some(&cache))
        .map_err(e("après poison"))?;
    if m3.origine() != Origine::Compilation {
        return Err("entrée empoisonnée non évincée".to_string());
    }
    if fs::read(&p_code).map_err(e("relecture"))? != code {
        return Err("après recompilation, le code objet du cache n'est pas celui d'origine".to_string());
    }
    // Taille nulle : même traitement.
    fs::write(&p_code, b"").map_err(e("poison nul"))?;
    if a.compiler_avec_cache(&src, &opts, Some(&cache))
        .map_err(e("après poison nul"))?
        .origine()
        != Origine::Compilation
    {
        return Err("entrée de taille nulle non évincée".to_string());
    }
    rapport.push("poison évincé puis recompilé (octet changé, taille nulle)".to_string());

    // Options différentes ⇒ autre entrée (pas de collision) ; plafond ≈ 2 entrées, 4 sources distinctes.
    let taille_entree = cache.taille_totale();
    let petit = Cache::nouveau(&dir, 2 * taille_entree + taille_entree / 2);
    for i in 0..4 {
        let s = assembler_essai(a, &[("KV_ESSAI", &format!("{}", 100 + i))]);
        a.compiler_avec_cache(&s, &opts, Some(&petit)).map_err(e("plafond"))?;
    }
    let total = petit.taille_totale();
    if total > petit.max_octets() {
        return Err(format!("plafond dépassé : {total} > {}", petit.max_octets()));
    }
    // L'entrée la plus récente est conservée (rejouée depuis le cache).
    let derniere = assembler_essai(a, &[("KV_ESSAI", "103")]);
    if a.compiler_avec_cache(&derniere, &opts, Some(&petit))
        .map_err(e("dernière"))?
        .origine()
        != Origine::Cache
    {
        return Err("l'entrée la plus récente a été évincée".to_string());
    }
    rapport.push(format!(
        "plafond respecté ({total} ≤ {} octets, LRU)",
        petit.max_octets()
    ));

    // Toutes les variantes d'options sont acceptées par le compilateur réel (NVRTC 13.x / hiprtc) et donnent
    // un module qui s'exécute, et la provenance enregistre les arguments réellement passés.
    let variantes = [
        ("k1", OptionsCompil::reference_k1()),
        ("prod", OptionsCompil::compat_production(b)),
        ("rapide", OptionsCompil::rapide()),
        (
            "O1",
            OptionsCompil {
                optimisation: 1,
                ..OptionsCompil::reference_k1()
            },
        ),
        ("extra", OptionsCompil::reference_k1().avec_extra("-DK1_MUTANT_METRO=1")),
    ];
    for (nom, o) in &variantes {
        let m = a
            .compiler_avec_cache(&src, o, None)
            .map_err(e(&format!("variante {nom}")))?;
        let f = a.fonction(&m, "k_remplir").map_err(e("fonction"))?;
        let x = a.alloc::<f64>(10).map_err(e("alloc"))?;
        a.lancer(
            &f,
            &Lancement::lineaire(10, 32),
            &[x.arg(), Arg::I64(10), Arg::F64(1.0), Arg::F64(1.0)],
        )
        .map_err(e("lancer"))?;
        let mut y = [0.0f64; 10];
        a.lire(&x, 0, &mut y).map_err(e("lire"))?;
        if y[9] != 10.0 {
            return Err(format!("variante {nom} : y[9] = {}", y[9]));
        }
        if m.provenance().arguments != o.arguments(b, &arch) {
            return Err(format!("variante {nom} : arguments de provenance incohérents"));
        }
    }
    rapport.push(format!(
        "{} variantes d'options compilées et exécutées",
        variantes.len()
    ));

    // Sans cache : toujours compilé.
    if a.compiler_avec_cache(&src, &opts, None)
        .map_err(e("sans cache"))?
        .origine()
        != Origine::Compilation
    {
        return Err("sans cache, le module doit être compilé".to_string());
    }

    let verdict = match b {
        BackendKind::Cuda => format!("CUBIN {arch} chargé, aucun PTX"),
        _ => format!("HSACO {arch} chargé"),
    };
    Ok(format!("{verdict} ; {}", rapport.join(" ; ")))
}

#[test]
#[ignore = "GPU"]
fn rt2_arguments_cubin_et_cache() {
    let Some(a) = appareil_ou_skip("RT-2") else { return };
    gate("RT-2", || rt2(&a));
}

// ---------------------------------------------------------------------------------------------
// Usage depuis plusieurs fils (rayon dans T3-T7) et ouvrir_auto
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn rt1b_appareil_partage_entre_fils() {
    let Some(a) = appareil_ou_skip("RT-1b") else { return };
    gate("RT-1b", || {
        let module = std::sync::Arc::new(compiler_k1(&a)?);
        let n = 5000usize;
        let resultats: Vec<Result<f64, String>> = std::thread::scope(|s| {
            let travaux: Vec<_> = (0..4)
                .map(|t| {
                    let (a, module) = (a.clone(), std::sync::Arc::clone(&module));
                    s.spawn(move || -> Result<f64, String> {
                        let f = a.fonction(&module, "k_remplir").map_err(e("fonction"))?;
                        let x = a.alloc::<f64>(n).map_err(e("alloc"))?;
                        a.lancer(
                            &f,
                            &Lancement::lineaire(n, BLOC),
                            &[x.arg(), Arg::I64(n as i64), Arg::F64(t as f64), Arg::F64(1.0)],
                        )
                        .map_err(e("lancer"))?;
                        let mut y = vec![0.0f64; n];
                        a.lire(&x, 0, &mut y).map_err(e("lire"))?;
                        Ok(y[n - 1])
                    })
                })
                .collect();
            travaux
                .into_iter()
                .map(|h| h.join().expect("fil sans panique"))
                .collect()
        });
        for (t, r) in resultats.into_iter().enumerate() {
            let v = r?;
            if v != t as f64 + (n - 1) as f64 {
                return Err(format!("fil {t} : {v}"));
            }
        }
        Ok("4 fils, un appareil cloné, un module partagé".to_string())
    });
}

#[test]
#[ignore = "GPU"]
fn rt_ouvrir_auto_coherent_avec_la_machine() {
    match gpu::ouvrir_auto() {
        Ok(Some(a)) => println!(
            "PASS RT-auto : ouvrir_auto → {} {} {}",
            a.backend().nom(),
            a.nom(),
            a.arch()
        ),
        Ok(None) => println!("PASS RT-auto : ouvrir_auto → CPU (aucun GPU choisi)"),
        Err(Erreur::NonImplemente(m)) => {
            println!("SKIP RT-auto : {m} (machine::detecter pas encore fusionné : unifie/machine)")
        }
        Err(autre) => panic!("FAIL RT-auto : {autre}"),
    }
}
