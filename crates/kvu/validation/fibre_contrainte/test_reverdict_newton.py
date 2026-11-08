#!/usr/bin/env python3
"""Tests de non-régression de reverdict_newton.py. Le témoin est l'essai à blanc N = 100 000 (hors verdict, fixtures_newton/) dont on aligne les
champs de protocole sur ceux d'une exécution de verdict : il doit donner COUT_SANS_BIAIS ; chaque mutation doit donner le statut prévu.
Exécution : /home/utilisateur/.venvs/certif/bin/python -I -B test_reverdict_newton.py -v (dans ce dossier)."""
import copy, importlib.util, json, math, os, unittest

ICI = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rv", os.path.join(ICI, "reverdict_newton.py"))
rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)

def temoin():
    """Essai à blanc N = 100 000 (hors verdict) dont on aligne TOUS les champs de protocole et de cohérence sur ceux d'une exécution de verdict
    (n de cellule, n et bloc des résumés recalculés depuis les comptes d'échecs, invariance à 200 000 départs)."""
    with open(os.path.join(ICI, "fixtures_newton", "essai_N100000_hors_verdict.json")) as f:
        d = json.load(f)
    d["hors_verdict"] = False
    N = 2_000_000
    for c in d["cellules"]:
        iv = c["invariance"]
        ok = round(iv["aller_ok"] / iv["n"] * 200_000)
        iv.update(n=200_000, aller_ok=ok, aller_echec=200_000 - ok)
        c["n"] = N
        nf = round(c["taux_echec"] * N)
        for est, n in (("production", rv.n_apres_thermalisation(N)), ("reessai", rv.n_apres_thermalisation(N)), ("sans_echecs", rv.n_apres_thermalisation(N - nf))):
            for k, r in enumerate(c[est]):
                r.update(n=n, bloc=n // 40, obs=k)
                r["domaine_campagne"] = r["statut"] == "OK" and r["bloc"] >= 40 * r["tau_int"]
    return d

def temoin_run2():
    """Témoin du run 2 : mêmes champs, graine_base 26110170000, graines des cellules recalculées, invariance à 10 000 000 départs par cellule."""
    d = temoin()
    d["graine_base"] = 26_110_170_000
    for ic, c in enumerate(d["cellules"]):
        g = 26_110_170_000 + 10 * ic
        c["graines"] = [g + 1, g + 2, g + 3, g + 4]
        iv = c["invariance"]
        ok = iv["aller_ok"] * 50
        iv.update(n=10_000_000, aller_ok=ok, aller_echec=10_000_000 - ok)
    return d

SU2 = {"resume": "1000 trajectoires : acceptées 939, refus Metropolis 32, échecs de calcul 29 {\"Newton\": 29}, par fenêtre de 100 [0, 0, 0, 0, 0, 1, 0, 2, 26, 0] ; rejoués avec max_newton = 500 : 25 sur 29 aboutissent (17 acceptées)", "echecs": 29, "acceptees": 939, "attendus_echecs": 29, "refus_metropolis": 32, "resolus_cap500": 25, "resolus_cap500_acceptes": 17,
       "identique_n2bis": True, "fenetres": [0, 0, 0, 0, 0, 1, 0, 2, 26, 0]}
SU2_AUTRE = dict(SU2, echecs=30, acceptees=938, identique_n2bis=False, fenetres=[0, 0, 0, 0, 0, 1, 0, 2, 27, 0],
                 resume="1000 trajectoires : acceptées 938, refus Metropolis 32, échecs de calcul 30 {\"Newton\": 30}, par fenêtre de 100 [0, 0, 0, 0, 0, 1, 0, 2, 27, 0] ; rejoués avec max_newton = 500 : 25 sur 30 aboutissent (17 acceptées)")   # cohérent mais ≠ n2bis

class Cas(unittest.TestCase):
    def statut(self, d, s=SU2):
        L, res, st = rv.calculer(d, s)
        return st
    def test_temoin(self):
        self.assertEqual(self.statut(temoin()), "COUT_SANS_BIAIS")
    def test_essai_hors_verdict_refuse(self):
        with open(os.path.join(ICI, "fixtures_newton", "essai_N100000_hors_verdict.json")) as f:
            self.assertEqual(self.statut(json.load(f)), "HORS_PROTOCOLE")
    def test_ecarts_de_protocole(self):
        def grille(d):
            for c in d["cellules"]: c["profil"]["x"] = [-math.pi + k * 2 * math.pi / 400 for k in range(400)]
        muts = [grille,
                lambda d: d["cellules"][3].__setitem__("cellule", 9),
                lambda d: d["cellules"][2].__setitem__("graines", [1, 2, 3, 4]),
                lambda d: d["cellules"][1].__setitem__("tol_newton", 1e-10),
                lambda d: d["cellules"][1].__setitem__("tol_rev", 1e-8),
                lambda d: d["cellules"][0]["profil"]["echecs"].__setitem__(5, 20_001),
                lambda d: d["cellules"][0]["profil"]["echecs"].__setitem__(5, 3.5),
                lambda d: d["cellules"][4].__setitem__("n", True),
                lambda d: d["cellules"][4].__setitem__("max_newton", 51),
                lambda d: d.__setitem__("hors_verdict", True),
                lambda d: d["cellules"].pop(),
                lambda d: d["cellules"][2].pop("profil"),
                lambda d: d["cellules"][2]["production"][0].__setitem__("erreur", -1.0),
                lambda d: d["cellules"][2]["production"][0].__setitem__("domaine_campagne", 1),
                lambda d: d["cellules"][6]["invariance"].__setitem__("n", 10_000)]
        for i, m in enumerate(muts):
            d = temoin(); m(d)
            self.assertEqual(self.statut(d), "HORS_PROTOCOLE", f"mutation {i}")
    def test_rejeu_su2_obligatoire_et_recalcule(self):
        self.assertEqual(self.statut(temoin(), None), "INDETERMINE")
        self.assertEqual(self.statut(temoin(), SU2_AUTRE), "INDETERMINE")                       # cohérent mais pas la reprise n2bis
        self.assertEqual(self.statut(temoin(), dict(SU2, echecs=28)), "HORS_PROTOCOLE")         # incohérent (somme des trajectoires)
        self.assertEqual(self.statut(temoin(), dict(SU2, echecs=True)), "HORS_PROTOCOLE")
        self.assertEqual(self.statut(temoin(), {"resume": "x", "identique_n2bis": True}), "HORS_PROTOCOLE")
    def test_mutant_invalide_rend_indetermine(self):
        d = temoin()
        for c in d["cellules"]:
            if c["taux_echec"] >= 0.04:
                c["reessai"][0]["statut"] = "NON-STATIONNAIRE"; c["reessai"][0]["domaine_campagne"] = False; break
        self.assertEqual(self.statut(d), "INDETERMINE")
    def test_biais_de_production_detecte(self):
        d = temoin()
        d["cellules"][1]["production"][0]["moyenne"] += 0.5
        self.assertEqual(self.statut(d), "BIAIS_DETECTE")
    def test_invariance_violee(self):
        d = temoin(); d["cellules"][1]["invariance"]["retour_echec"] = 1
        self.assertEqual(self.statut(d), "INDETERMINE")
    def test_derivee_complete_de_la_prediction(self):
        # différences finies sur b(f) = -Cov(O,f)/(1-f̄) : la dérivée analytique complète doit s'y accorder
        import numpy as np
        x = np.array([-3.0, -1.0, 0.5, 2.0]); w = np.array([0.1, 0.4, 0.3, 0.2]); O = np.cos(x); f = np.array([0.1, 0.3, 0.2, 0.4])
        def b(f):
            ef = (w * f).sum(); eo = (w * O).sum(); return -(w * (O - eo) * (f - ef)).sum() / (1 - ef)
        ef = (w * f).sum(); eo = (w * O).sum(); bb = b(f)
        ana = -w * ((O - eo) - bb) / (1 - ef)
        num = np.array([(b(f + 1e-7 * np.eye(4)[k]) - b(f - 1e-7 * np.eye(4)[k])) / 2e-7 for k in range(4)])
        self.assertTrue(np.allclose(ana, num, atol=1e-7))

    # ------------------------------------------------------------------ durcissement : chaque champ enregistré par Rust, et le rejeu SU(2)
    def chemins(self, o, p=()):
        out = [(p, o)]
        if isinstance(o, dict):
            for k, v in o.items(): out += self.chemins(v, p + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if i < 2 or i == len(o) - 1: out += self.chemins(v, p + (i,))
        return out

    def mutations(self, v, p):
        muts = [("supprimer", lambda o, k: o.pop(k)), ("none", lambda o, k: o.__setitem__(k, None)), ("bool", lambda o, k: o.__setitem__(k, True)),
                ("str", lambda o, k: o.__setitem__(k, "x")), ("liste", lambda o, k: o.__setitem__(k, [1]))]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            muts += [("nan", lambda o, k: o.__setitem__(k, float("nan"))), ("neg", lambda o, k: o.__setitem__(k, -abs(v) - 1)),
                     ("zero", lambda o, k: o.__setitem__(k, 0)), ("un", lambda o, k: o.__setitem__(k, 1)),
                     ("type", lambda o, k: o.__setitem__(k, float(v) if isinstance(v, int) else int(v) if float(v).is_integer() else 7))]
            if p[-1] not in ("moyenne", "erreur", "tau_int"):   # ces trois-là sont des DONNÉES (leur modification est un autre résultat, pas un défaut de schéma)
                muts += [("derive", lambda o, k: o.__setitem__(k, v * (1 + 1e-6) if v else 1e-6)), ("plus_un", lambda o, k: o.__setitem__(k, v + 1))]
        if isinstance(v, dict): muts += [("cle_en_trop", lambda o, k: o[k].__setitem__("inattendu", 1))]
        return muts

    DONNEES = {"moyenne", "erreur", "tau_int", "pire_ecart", "echecs", "taux_acceptation", "resolus_cap500", "resolus_cap500_acceptes", "retour_echec", "retour_sans_retour_au_depart"}
    def est_donnee(self, p, nom):
        """Une valeur numérique modifiée DANS son domaine est un autre résultat, pas un défaut de schéma (seule une ré-exécution la détecte)."""
        nom_champ = [x for x in p if isinstance(x, str)][-1]
        return nom_champ in self.DONNEES and nom in ("zero", "un", "plus_un", "derive", "type")

    def appliquer(self, d, p, f):
        o = d
        for k in p[:-1]: o = o[k]
        f(o, p[-1])

    def test_mutation_de_chaque_champ_de_la_courbe(self):
        ref = temoin(); n = 0; mauvais = []
        for p, v in self.chemins(ref):
            if not p: continue
            for nom, f in self.mutations(v, p):
                d = copy.deepcopy(ref); self.appliquer(d, p, f)
                if d == ref: continue
                n += 1
                st = self.statut(d)
                if self.est_donnee(p, nom):
                    if st not in ("COUT_SANS_BIAIS", "INDETERMINE", "BIAIS_DETECTE", "HORS_PROTOCOLE"): mauvais.append((p, nom, st))
                elif st != "HORS_PROTOCOLE": mauvais.append((p, nom, st))   # un défaut de schéma doit être classé HORS_PROTOCOLE
        self.assertGreater(n, 1500)
        self.assertEqual(mauvais, [], f"{len(mauvais)} mutations mal classées sur {n} : {mauvais[:5]}")

    def test_mutation_de_chaque_champ_du_rejeu_su2(self):
        ref = dict(SU2)
        self.assertEqual(self.statut(temoin(), ref), "COUT_SANS_BIAIS")
        n = 0; mauvais = []
        for p, v in self.chemins(ref):
            if not p: continue
            for nom, f in self.mutations(v, p):
                d = copy.deepcopy(ref); self.appliquer(d, p, f)
                if d == ref: continue
                n += 1
                st = self.statut(temoin(), d)
                if self.est_donnee(p, nom) and st in ("INDETERMINE", "COUT_SANS_BIAIS"): continue
                if st != "HORS_PROTOCOLE": mauvais.append((p, nom, st))
        self.assertGreater(n, 60)
        self.assertEqual(mauvais, [], f"{len(mauvais)} mutations mal classées sur {n} : {mauvais[:5]}")

    @unittest.skipUnless(os.path.isdir(os.path.join(ICI, "resultats_newton_2026-11")),
                         "données de campagne interne retirées de l'instantané public (docs/PUBLICATION.md)")
    def test_donnees_reelles_gelees(self):
        """Les données de verdict gelées (run 1) redonnent le verdict gelé INDETERMINE, ligne pour ligne."""
        rep = os.path.join(ICI, "resultats_newton_2026-11")
        with open(os.path.join(rep, "courbe.json")) as f: c = json.load(f)
        with open(os.path.join(rep, "su2.json")) as f: s = json.load(f)
        L, res, st = rv.calculer(c, s)
        with open(os.path.join(rep, "verdicts.txt")) as f: gele = f.read().splitlines()
        self.assertEqual(st, "INDETERMINE")
        self.assertEqual(L, gele)

    @unittest.skipUnless(os.path.isdir(os.path.join(ICI, "resultats_newton_2026-11")),
                         "données de campagne interne retirées de l'instantané public (docs/PUBLICATION.md)")
    def test_donnees_reelles_run2_gelees(self):
        """Les données du run 2 (graines 26110170000) redonnent le verdict calculé INDETERMINE, ligne pour ligne (le verdict du run 1 reste gelé et distinct)."""
        rep = os.path.join(ICI, "resultats_newton_run2_2026-11")
        with open(os.path.join(rep, "courbe_run2.json")) as f: c = rv.charger_json(f)
        with open(os.path.join(ICI, "resultats_newton_2026-11", "su2.json")) as f: s = rv.charger_json(f)
        L, res, st = rv.calculer(c, s)
        with open(os.path.join(rep, "verdicts.txt")) as f: gele = f.read().splitlines()
        self.assertEqual(st, "INDETERMINE")
        self.assertEqual(L, gele)
        self.assertIn("FAIL NEWTON-INVARIANCE-TAUX par cellule : 5 PASS, 3 FAIL, 0 SKIP sur 8", "\n".join(L))

    # ------------------------------------------------------------------ run 2 : graines neuves, gate d'invariance à taux borné
    def test_run2_temoin(self):
        self.assertEqual(self.statut(temoin_run2()), "COUT_SANS_BIAIS")
    # référence INDÉPENDANTE de scipy : borne supérieure de Clopper–Pearson par bissection sur la fonction de répartition binomiale (lgamma)
    @staticmethod
    def cp_upper(k, s, conf):
        def cdf(p):
            tot = 0.0
            for i in range(k + 1):
                lg = math.lgamma(s + 1) - math.lgamma(i + 1) - math.lgamma(s - i + 1) + i * math.log(p) + (s - i) * math.log1p(-p)
                tot += math.exp(lg)
            return tot
        lo, hi = 0.0, 1.0 - 1e-12
        for _ in range(200):
            mid = (lo + hi) / 2
            if cdf(mid) > 1 - conf: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    @staticmethod
    def tau_ub(tau, n):
        from statistics import NormalDist
        return tau * (1 + NormalDist().inv_cdf(1 - 0.05 / 16) * math.sqrt(2 * (2 * (6 * tau + 6) + 1) / n))
    @staticmethod
    def r_pi():
        import numpy as np
        x = np.linspace(-math.pi, math.pi, 400001)[:-1]
        w = np.exp(-0.5 * np.cos(x) - 0.72 * np.sin(x) ** 2)
        return float(w.max() / w.mean())
    def eps_max(self, c):
        """ε_max = min_k σ_stat,k/(32 τ_ub,k)/R_π (σ_stat = erreur jackknife de la production de la cellule), recalculé indépendamment."""
        return min(c["production"][k]["erreur"] / (32 * self.tau_ub(c["production"][k]["tau_int"], c["production"][k]["n"])) for k in (0, 1)) / self.r_pi()
    def test_rapport_de_densites(self):
        # départs uniformes vs π : R = max π/U = 2,129 (quadrature indépendante) ; le script retrouve la même valeur
        self.assertAlmostEqual(rv.rapport_pi(), self.r_pi(), places=3)
        self.assertGreater(self.r_pi(), 2.0); self.assertLess(self.r_pi(), 2.3)
    def kmax_cp(self, s, e, alpha=0.05 / 8):
        """Plus grand nombre d'exceptions dont la borne de Clopper–Pearson (indépendante, cp_upper) reste ≤ e : recherche dichotomique (la borne croît avec k)."""
        lo, hi = 0, 20000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cp_upper(mid, s, 1 - alpha) <= e: lo = mid
            else: hi = mid - 1
        return lo
    def test_run2_seuil_prend_le_minimum_des_observables(self):
        # l'observable la plus précise (σ_stat petit) fixe le seuil : le script refuse k_min + 1 exceptions, que la seule observable 0 aurait acceptées
        x = temoin_run2(); c = x["cellules"][3]; s = c["invariance"]["aller_ok"]
        c["production"][1]["erreur"] = c["production"][0]["erreur"] / 10
        e_min = self.eps_max(c)
        e_0 = c["production"][0]["erreur"] / (32 * self.tau_ub(c["production"][0]["tau_int"], c["production"][0]["n"])) / self.r_pi()
        self.assertLess(e_min, e_0 / 5)
        k_min, k_0 = self.kmax_cp(s, e_min), self.kmax_cp(s, e_0)
        self.assertGreater(k_min, 2); self.assertGreater(k_0, k_min + 5)
        for k, attendu in ((k_min, "COUT_SANS_BIAIS"), (k_min + 1, "INDETERMINE")):
            c["invariance"]["retour_echec"] = k
            self.assertEqual(self.statut(x), attendu, k)       # appel du script (calculer) : pas de comparaison de deux expressions du même helper
    def test_cp_de_reference(self):
        # k = 0 : forme close 1 − α^(1/s)
        self.assertAlmostEqual(self.cp_upper(0, 1_000_000, 0.99375), 1 - 0.00625 ** (1 / 1_000_000), places=12)
        from scipy.stats import beta as loi_beta
        self.assertAlmostEqual(self.cp_upper(3, 6_000_000, 0.99375) / loi_beta.ppf(0.99375, 4, 6_000_000 - 3), 1.0, places=6)
    def test_run2_seuil_par_cellule(self):
        d = temoin_run2()
        # pour chaque cellule : k exceptions passent tant que la borne de Clopper–Pearson (indépendante) reste ≤ ε_max(cellule) ; le suivant échoue
        for ic in (1, 5, 7):
            c = d["cellules"][ic]; s = c["invariance"]["aller_ok"]; e = self.eps_max(c)
            kmax = max(k for k in range(0, 400) if self.cp_upper(k, s, 1 - 0.05 / 8) <= e)
            self.assertGreater(kmax, 2, (ic, kmax))
            for k, attendu in ((kmax, "COUT_SANS_BIAIS"), (kmax + 1, "INDETERMINE")):
                x = temoin_run2(); x["cellules"][ic]["invariance"]["retour_echec"] = k
                self.assertEqual(self.statut(x), attendu, (ic, k))
    def test_run2_aucune_dilution_entre_cellules(self):
        # une cellule dépasse sa borne d'UNE exception de plus que le maximum admis ; le taux cumulé sur les 8 cellules serait resté très sous le seuil de cette cellule :
        # la dilution est refusée (gate par cellule), même si les 7 autres sont parfaites
        x = temoin_run2(); c = x["cellules"][7]; s = c["invariance"]["aller_ok"]; e = self.eps_max(c)
        kmax = max(k for k in range(0, 600) if self.cp_upper(k, s, 1 - 0.05 / 8) <= e)
        c["invariance"]["retour_echec"] = kmax + 1
        tot = sum(cc["invariance"]["aller_ok"] for cc in x["cellules"])
        self.assertLess((kmax + 1) / tot, e / 4)                 # un taux cumulé aurait été largement conforme
        self.assertGreater(self.cp_upper(kmax + 1, s, 1 - 0.05 / 8), e)
        self.assertEqual(self.statut(x), "INDETERMINE")
    def test_run2_le_seuil_suit_tau(self):
        # τ plus grand ⇒ ε_max plus petit : les mêmes k exceptions passent à τ = 1 et échouent à τ = 100 (production valide : bloc ≥ 40 τ)
        for tau, attendu in ((1.0, "COUT_SANS_BIAIS"), (100.0, "INDETERMINE")):
            x = temoin_run2(); c = x["cellules"][3]
            for k in (0, 1): c["production"][k]["tau_int"] = tau
            c["invariance"]["retour_echec"] = 20
            self.assertEqual(self.statut(x), attendu, tau)
    def test_tau_majore_borne_de_confiance(self):
        # la borne du SCRIPT (rv.tau_majore) majore τ̂ de ~3-4 % pour n = 1,6·10⁶ (incertitude statistique de Sokal), jamais en dessous ; plus n est petit, plus la marge croît
        for tau in (1.0, 5.0, 12.0, 50.0):
            ub = rv.tau_majore(tau, 1_600_000)
            self.assertGreater(ub, tau); self.assertLess(ub / tau - 1, 0.1 if tau <= 12 else 0.2)
            self.assertAlmostEqual(ub, self.tau_ub(tau, 1_600_000), places=12)          # recalcul indépendant de la formule enregistrée
        self.assertAlmostEqual(rv.tau_majore(12.0, 1_600_000), 12.45966683, places=7)    # constante numérique indépendante (z = Φ⁻¹(1 − 0,05/16) = 2,7343688)
        self.assertGreater(rv.tau_majore(12.0, 16_000) / 12.0, rv.tau_majore(12.0, 1_600_000) / 12.0)
        x = temoin_run2(); c = x["cellules"][3]
        plat = min(c["production"][k]["erreur"] / (32 * c["production"][k]["tau_int"]) for k in (0, 1)) / self.r_pi()
        self.assertLess(self.eps_max(c), plat)   # seuil plus sévère que la formule au τ̂ ponctuel
    def test_run2_puissance_insuffisante(self):
        x = temoin_run2(); iv = x["cellules"][2]["invariance"]
        iv.update(aller_ok=20_000, aller_echec=10_000_000 - 20_000)        # trop peu d'allers réussis pour certifier ε_max même à zéro exception
        self.assertEqual(self.statut(x), "INDETERMINE")
        L, res, st = rv.calculer(x, SU2)
        self.assertTrue(any(l.startswith("SKIP NEWTON-INVARIANCE-TAUX") for l in L), [l for l in L if "INVARIANCE" in l][-2:])
    def test_run2_protocole(self):
        d = temoin_run2(); d["graine_base"] = 26_110_180_000
        self.assertEqual(self.statut(d), "HORS_PROTOCOLE")
        d = temoin_run2(); d["cellules"][2]["invariance"].update(n=200_000, aller_ok=100_000, aller_echec=100_000)   # n du run 1 sous la graine_base du run 2
        self.assertEqual(self.statut(d), "HORS_PROTOCOLE")
        d = temoin_run2(); d["cellules"][1]["graines"] = [26_110_160_011, 26_110_160_012, 26_110_160_013, 26_110_160_014]   # graines du run 1 sous la base du run 2
        self.assertEqual(self.statut(d), "HORS_PROTOCOLE")
        d = temoin_run2(); d["graine_base"] = True
        self.assertEqual(self.statut(d), "HORS_PROTOCOLE")
        d = temoin(); d["graine_base"] = 26_110_160_000      # run 1 explicite : mêmes règles que le run 1 (zéro exception)
        self.assertEqual(self.statut(d), "COUT_SANS_BIAIS")
        d["cellules"][4]["invariance"]["retour_echec"] = 1
        self.assertEqual(self.statut(d), "INDETERMINE")
    def test_statuts_non_ok_sous_leur_forme_reelle(self):
        """Les quatre statuts non OK tels que Rust les sérialise (NaN → null) : le verdict aboutit (INDETERMINE), sans exception ni faux HORS_PROTOCOLE."""
        formes = {
            "PUISSANCE": dict(statut="PUISSANCE", erreur=None, bloc=0, domaine_campagne=False),
            "TAU-NON-ESTIMABLE-initial": dict(statut="TAU-NON-ESTIMABLE", erreur=None, tau_int=None, bloc=0, domaine_campagne=False),
            "TAU-NON-ESTIMABLE-tardif": dict(statut="TAU-NON-ESTIMABLE", bloc=1600000 // 40, domaine_campagne=False),
            "NON-STATIONNAIRE": dict(statut="NON-STATIONNAIRE", bloc=1600000 // 40, domaine_campagne=False),
            "PUISSANCE-moyenne-nulle": dict(statut="PUISSANCE", moyenne=None, erreur=None, bloc=0, domaine_campagne=False),
        }
        for nom, forme in formes.items():
            for est in ("production", "reessai", "sans_echecs"):
                d = temoin()
                c = next(c for c in d["cellules"] if c["taux_echec"] >= 0.04)
                r = c[est][0]
                bloc = forme["bloc"]
                if est == "sans_echecs" and bloc: bloc = r["n"] // 40
                r.update(forme); r["bloc"] = bloc
                L, res, st = rv.calculer(d, SU2)
                self.assertEqual(st, "INDETERMINE", f"{nom} / {est} : {st} {[l for l in L if l.startswith('FAIL')]}")

    @unittest.skipUnless(os.path.isdir(os.path.join(ICI, "resultats_newton_2026-11")),
                         "données de campagne interne retirées de l'instantané public (docs/PUBLICATION.md)")
    def test_niveau_texte_du_json(self):
        """Clés dupliquées, NaN/Infinity, débordement : refusés à la lecture (main) ; l'exécution complète donne HORS_PROTOCOLE, jamais un statut positif."""
        import subprocess, sys, tempfile, shutil
        rep = os.path.join(ICI, "resultats_newton_2026-11")
        with open(os.path.join(rep, "courbe.json")) as f: base = f.read()
        with open(os.path.join(rep, "su2.json")) as f: su2 = f.read()
        cas = [('"schema"', '"schema": "x", "schema"'), ('"dt": 0.5', '"dt": NaN, "dt": 0.5'), ('"statut": "OK"', '"statut": "x", "statut": "OK"'),
               ('"aller_ok"', '"aller_ok": 1, "aller_ok"'), ('"essais": 20000', '"essais": Infinity, "essais": 20000'), ('"n": 2000000', '"n": 1e999')]
        for avant, apres in cas:
            t = tempfile.mkdtemp()
            try:
                self.assertIn(avant, base)
                with open(os.path.join(t, "c.json"), "w") as f: f.write(base.replace(avant, apres, 1))
                r = subprocess.run([sys.executable, "-I", "-B", os.path.join(ICI, "reverdict_newton.py"), "--courbe", os.path.join(t, "c.json"), "--su2", os.path.join(rep, "su2.json"), "--sortie", os.path.join(t, "s")], capture_output=True, text=True)
                self.assertIn("STATUT NEWTON-REVERDICT HORS_PROTOCOLE", r.stdout, (avant, apres, r.stderr[-300:]))
            finally:
                shutil.rmtree(t)
        t = tempfile.mkdtemp()
        try:
            with open(os.path.join(t, "s.json"), "w") as f: f.write(su2.replace('"echecs"', '"echecs": 0, "echecs"', 1))
            r = subprocess.run([sys.executable, "-I", "-B", os.path.join(ICI, "reverdict_newton.py"), "--courbe", os.path.join(rep, "courbe.json"), "--su2", os.path.join(t, "s.json"), "--sortie", os.path.join(t, "s")], capture_output=True, text=True)
            self.assertNotIn("COUT_SANS_BIAIS", r.stdout)
        finally:
            shutil.rmtree(t)

if __name__ == "__main__":
    unittest.main()
