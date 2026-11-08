"""Reference Python independante pour M3 (flux par face de bloc) : stats de boucles.py (desert_debut, copie a l'octet)
et flux coherent (implementation independante en quaternions, numpy)."""
import json, math, sys, hashlib, os
import numpy as np
ICI=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import boucles as B   # copie a l'octet de desert_debut_2026-11/scripts/boucles.py

PLANS=B.PLANS
def qmul(a,b):
    a0,a1,a2,a3=a[...,0],a[...,1],a[...,2],a[...,3]
    b0,b1,b2,b3=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+b0*a1-(a2*b3-a3*b2), a0*b2+b0*a2-(a3*b1-a1*b3), a0*b3+b0*a3-(a1*b2-a2*b1)],axis=-1)
def qdag(a):
    r=a.copy(); r[...,1:]*=-1; return r
def lire_q(chemin):
    U=np.load(chemin)
    if U.dtype==np.complex128:
        return np.stack([U[...,0,0].real,U[...,0,1].imag,U[...,0,1].real,U[...,0,0].imag],axis=-1)
    return np.asarray(U,dtype=np.float64)
def dec(X,mu,n):
    return np.roll(X,-n,axis=mu)
def ligne(U,mu,a):
    """S_mu^{(a)}(x) = U_mu(x) U_mu(x+mu) ... (a facteurs ; identite si a = 0)."""
    L=U.shape[1]
    out=np.zeros(U.shape[1:]); out[...,0]=1.0
    for j in range(a):
        out=qmul(out,dec(U[mu],mu,j))
    return out
def flux_coherent(q,n):
    """Pour chaque plan (ordre PLANS) et chaque site : Phi_coh, somme des |phi|, (6, L^4)."""
    phis=[]; sabs=[]
    for (mu,nu) in PLANS:
        Up=qmul(qmul(q[mu],dec(q[nu],mu,1)),qmul(qdag(dec(q[mu],nu,1)),qdag(q[nu])))  # plaquette en y
        tot=np.zeros(q.shape[1:-1]+(3,)); sa=np.zeros(q.shape[1:-1])
        for a in range(n):
            Sa=ligne(q,mu,a)
            for b in range(n):
                Sb=dec(ligne(q,nu,b),mu,a)       # S_nu^{(b)}(x + a mu)
                T=qmul(Sa,Sb)
                P=dec(dec(Up,mu,a),nu,b)         # U_p(x + a mu + b nu)
                Ut=qmul(qmul(T,P),qdag(T))
                v=Ut[...,1:]; nv=np.sqrt((v*v).sum(-1)); th=2.0*np.arctan2(nv,Ut[...,0])
                fac=np.where(nv>0,th/np.where(nv>0,nv,1.0),2.0)
                tot+=fac[...,None]*v; sa+=th
        phis.append(np.sqrt((tot*tot).sum(-1)).reshape(-1)); sabs.append(sa.reshape(-1))
    return np.stack(phis),np.stack(sabs)
COH_MIN=("0.5","0.8","0.95")
SEUILS_COH={"Q1_2":(math.pi,True),"Q3_4":(1.5*math.pi,True),"Q31_32":(2*math.pi*31/32,False),"Q1":(2*math.pi,False)}
def stats_coh(phi,sabs):
    pf=phi.reshape(-1); sf=sabs.reshape(-1)
    c=np.where(sf>0,pf/np.where(sf>0,sf,1.0),1.0)
    comptes={k:int(np.count_nonzero(pf>s)) if strict else int(np.count_nonzero(pf>=s)) for k,(s,strict) in SEUILS_COH.items()}
    joint={}
    for k,(s,strict) in SEUILS_COH.items():
        au_dessus=(pf>s) if strict else (pf>=s)
        joint[k]={cm:int(np.count_nonzero(au_dessus&(c>=float(cm)))) for cm in COH_MIN}
    return {"N":int(pf.size),"comptes":comptes,"comptes_coherence":joint,"s_phi":float(pf.sum()),"s_phi2":float((pf*pf).sum()),"s_coh":float(c.sum()),"phi_max":float(pf.max())}
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
if __name__=='__main__':
    sortie=sys.argv[1]; fixtures=sys.argv[2:]
    doc={"format":"kvu-rg-flux-reference-py/1","numpy":np.__version__,"python":sys.version.split()[0],
         "boucles_py_sha256":sha(os.path.join(ICI,'boucles.py')),"script_sha256":sha(os.path.abspath(__file__)),"fixtures":[]}
    for spec in fixtures:
        chemin,ns,coh=spec.split(':')   # chemin:1,2,3:2,3
        ns=[int(t) for t in ns.split(',') if t]; cohs=[int(t) for t in coh.split(',') if t]
        a,b,L=B.lire_cfg(chemin)
        ent={"fichier":os.path.basename(chemin),"sha256":sha(chemin),"L":L,"brut":{},"coherent":{}}
        for n in ns:
            s=B.stats_x(B.x_boucles(a,b,n)); h=s.pop("hist_alpha")
            s["hist_alpha_nonvides"]={str(i):int(v) for i,v in enumerate(h) if v}
            ent["brut"][f"n{n}"]=s
        q=lire_q(chemin)
        for n in cohs:
            phi,sa=flux_coherent(q,n); ent["coherent"][f"n{n}"]=stats_coh(phi,sa)
        doc["fixtures"].append(ent)
    json.dump(doc,open(sortie,'w'),indent=1,sort_keys=True)
    print("ecrit",sortie)
