import importlib.util, csv, json, sys
spec = importlib.util.spec_from_file_location("ap", "/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/analyse_plaquette.py")
ap = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(ap)
d = "/chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production/L12_b2.3_c1"
rows = [r for r in csv.DictReader(open(d + "/measurements.csv")) if r["phase"] == "sample"]
P = [float(r["plaquette"]) for r in rows]
t, W, dt = ap.tau_int(P)
s = ap.stat(P)
json.dump({"chaine": "L12_b2.3_c1", "n": len(P), "tau": t, "W": W, "dtau": dt, "stat": s}, open("tau/ref_tau.json", "w"), indent=1)
print(json.dumps({"tau": t, "W": W, "dtau": dt, "stat": s}))
