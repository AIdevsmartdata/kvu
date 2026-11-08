import numpy as np

L=3
sites=[(x,y,z,t) for x in range(L) for y in range(L) for z in range(L) for t in range(L)]
sid={x:i for i,x in enumerate(sites)}
links=[]
for x in sites:
    for mu in range(4):
        links.append((x,mu))
lid={x:i for i,x in enumerate(links)}
B=np.zeros((len(sites),len(links)))
for l,(x,mu) in enumerate(links):
    y=list(x); y[mu]=(y[mu]+1)%L; y=tuple(y)
    B[sid[x],l]=1
    B[sid[y],l]-=1
plaq=[]
for x in sites:
    for mu in range(4):
        for nu in range(mu+1,4):
            plaq.append((x,mu,nu))
C=np.zeros((len(plaq),len(links)))
def link(x,mu):
    return lid[(tuple(x),mu)]
for p,(x,mu,nu) in enumerate(plaq):
    xmu=list(x);xmu[mu]+=1;xmu[mu]%=L
    xnu=list(x);xnu[nu]+=1;xnu[nu]%=L
    C[p,link(x,mu)]=1
    C[p,link(xmu,nu)]=1
    C[p,link(xnu,mu)]-=1
    C[p,link(x,nu)]-=1

def rank(a,tol=1e-8):
    return int(np.sum(np.linalg.svd(a,compute_uv=False)>tol))
rfull=rank(C)
print('L =',L,', links =',len(links),', plaquettes =',len(plaq))
print('rank(C) =',rfull,'; predicted 3 L^4 - 3 =',3*L**4-3)
# One block centre at the origin. Gauge fields in the conditioned fibre
# satisfy B0 A=0. Choose four generic full-rank coarse constraints obeying Q B0=0.
B0=B[1:,:]
U,s,Vh=np.linalg.svd(B0)
nullB0=Vh[len(s):].T
rng=np.random.default_rng(20261025)
Q=nullB0@rng.normal(size=(nullB0.shape[1],4))
Q=Q.T
print('rank(Q) =',rank(Q),'; max |Q B0^T| =',np.max(np.abs(Q@B0.T)))
# ker Q basis and rank of C restricted to it.
Uq,sq,Vhq=np.linalg.svd(Q)
kerQ=Vhq[4:].T
rcond=rank(C@kerQ)
print('conditioned rank =',rcond,'; predicted 3(L^4-1) =',3*(L**4-1))
print('conditional limit =',3*rcond/(2*len(plaq)),'; 20/27 =',20/27)
print('unconditioned limit =',3*rfull/(2*len(plaq)),'; 3/4-3/(4L^4) =',3/4-3/(4*L**4))
