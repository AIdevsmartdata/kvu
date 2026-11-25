use kvu::commun::Resultat;
use kvu::su2::spectro::bloc::ParamBloc;
use kvu::su2::spectro::flux::{self, SecteurCentre, SecteurFlux};
use kvu::su2::spectro::gpu_torelon::TorelonGpu;
use kvu::su2::tube::reseau::GeoA;
use serde_json::json;

#[test]
fn api_flux_v04() {
    let g = GeoA::nouvelle([4, 4, 4, 6]).unwrap();
    let champ = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
    let f = SecteurFlux { d: [0, 1] };
    let bloc = ParamBloc::vide();
    assert_eq!(SecteurFlux::analyser("1,1,0", &g.d, 3).unwrap().d, f.d);
    assert!(SecteurFlux::analyser("1,1,1", &g.d, 3).is_err());
    f.controler_niveaux(&g.d, &bloc).unwrap();

    let niveaux = flux::champs_niveaux(&g, &champ, 3, 0.5, &[0], &bloc);
    assert_eq!(niveaux, vec![(champ.clone(), 1)]);
    assert_eq!(flux::sommes_cpu(&g, &champ, 3, &f, 1), vec![[64.0; 6]; 6]);
    let sommes = flux::sommes_niveaux_cpu(&g, &champ, 3, &f, 0.5, &[0], &bloc);
    let ops = flux::assembler(&sommes, 1, 6);
    assert_eq!(ops, [64.0, 64.0, 64.0, 64.0, 4096.0, 64.0, 64.0].repeat(6));
    let base = flux::base_corr(&ops, 1, 6);
    assert_eq!(base[0], vec![[128.0, 0.0]; 6]);
    assert_eq!(flux::corr_flux(&ops, 1, 6, 1)[0], [16384.0, 16384.0]);
    assert_eq!(flux::moyennes(&ops, 1, 6), ops[..7]);
    let manifeste = flux::manifeste(&f, &g.d, 3, &json!([0]), 1);
    assert_eq!(manifeste["directions_flux"], json!([0, 1]));
    assert_eq!(manifeste["ordre_operateurs_bruts"], json!(["0:a+", "0:a-", "0:b+", "0:b-", "0:c=T1*T2", "0:T1", "0:T2"]));
    assert_eq!(ops, flux::mesurer_cpu(&g, &champ, 3, &SecteurCentre::Deux(f), 0.5, &[0], &bloc).unwrap());

    let _mesurer_flux: fn(&mut TorelonGpu, &SecteurFlux) -> Resultat<Vec<Vec<f64>>> = TorelonGpu::mesurer_flux;
}
