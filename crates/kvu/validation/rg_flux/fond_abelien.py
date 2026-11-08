"""Fond abelien uniforme F_{01} = Phi (jauge de Landau), Phi L^2 = 4 pi m (periodique exact), format q64."""
import numpy as np, sys, math
def fond(L, m, plan=(0,1), quanta=4.0):
    Phi=quanta*math.pi*m/(L*L)
    a,b=plan
    q=np.zeros((4,L,L,L,L,4)); q[...,0]=1.0
    idx=np.indices((L,L,L,L))
    xa=idx[a]; xb=idx[b]
    # U_b(x) = exp(i Phi x_a sigma3/2)
    th=Phi*xa
    q[b,...,0]=np.cos(th/2); q[b,...,3]=np.sin(th/2)
    # U_a(x) a x_a = L-1 : exp(-i Phi L x_b sigma3/2)
    th2=-Phi*L*xb
    sel=(xa==L-1)
    q[a][sel,0]=np.cos(th2[sel]/2); q[a][sel,3]=np.sin(th2[sel]/2)
    return q,Phi
if __name__=='__main__':
    L=int(sys.argv[1]); m=int(sys.argv[2]); out=sys.argv[3]
    quanta=float(sys.argv[4]) if len(sys.argv)>4 else 4.0
    q,Phi=fond(L,m,quanta=quanta)
    np.save(out,q)
    print(f"L={L} m={m} Phi={Phi!r} 9Phi={9*Phi!r}")
