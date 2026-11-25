#!/usr/bin/env python3
"""MBAR/Ferrenberg-Swendsen biné, réduction pondérée des équations de repond_original.py.
Réutilise les fréquences complètes ; aucune décimation des trajectoires.
Références: Kovács–Tomboulis (2000), (6)-(11); Shirts–Chodera (2008), (11), annexe C.
"""
import numpy as np

def lse(a,axis=-1):
 m=np.max(a,axis=axis,keepdims=True)
 return (m+np.log(np.exp(a-m).sum(axis=axis,keepdims=True))).squeeze(axis)
class Histogramme:
 def __init__(self,lambdas,centres,counts):
  # counts[k,rep,bin]. Each replica is retained for the bootstrap.
  self.lambdas=np.asarray(lambdas,float);self.centres=np.asarray(centres,float)
  self.counts=np.asarray(counts,float)
  if self.counts.ndim!=3 or self.counts.shape[0]!=len(self.lambdas) or self.counts.shape[2]!=len(self.centres):raise ValueError('formes incompatibles')
  self.N=self.counts.sum(axis=(1,2));self.H=self.counts.sum(axis=(0,1));self.F=np.zeros(len(self.lambdas));self.ok=False
  if (self.N<=0).any():raise ValueError('état vide')
 def solve(self,counts=None,start=None,tol=1e-10,maxit=200):
  C=self.counts if counts is None else counts;N=C.sum(axis=(1,2));HH=C.sum(axis=(0,1));keep=HH>0;H=HH[keep];x=self.centres[keep]
  S0=float(x@H/H.sum());x=x-S0;b=self.lambdas;K=len(b);F=np.zeros(K)if start is None else np.array(start,copy=True);F-=F[0]
  A0=x[:,None]*b[None,:]+np.log(N)[None,:];ok=False
  for it in range(maxit):
   A=A0-F[None,:];D=lse(A,axis=1);p=np.exp(A-D[:,None]);sumP=H@p;grad=N-sumP
   hess=np.diag(sumP)-p.T@(H[:,None]*p)
   step=np.r_[0.,-np.linalg.solve(hess[1:,1:]+np.eye(K-1)*1e-14*N.max(),grad[1:])]
   value=H@D+N@F;t=1.
   while t>1e-7:
    Fn=F+t*step;v=H@lse(A0-Fn[None,:],axis=1)+N@Fn
    if v<=value+1e-12*abs(value):break
    t*=.5
   F=Fn
   if np.max(abs(t*step))<tol and np.max(abs(grad))/N.max()<1e-9:ok=True;break
  # Shift back to physical ln Z, gauge ln Z(lambda_0)=0.
  result=F+(b-b[0])*S0
  if counts is None:self.F=F;self.physical=result;self.ok=ok;self.it=it+1
  return result,ok
 def overlap(self):
  H=self.H;keep=H>0;x=self.centres[keep];HH=H[keep]
  # Physical gauge, avoids depending on the stored centered gauge.
  A=x[:,None]*self.lambdas[None,:]-self.physical[None,:]+np.log(self.N)[None,:]
  p=np.exp(A-lse(A,axis=1)[:,None]);O=(p.T@(HH[:,None]*p))/self.N[:,None]
  eigen=np.linalg.eigvals(O).real;return O,np.sort(eigen)[::-1]
 def bootstrap(self,nboot=100,seed=212345):
  rng=np.random.default_rng(seed);out=[];K,R,_=self.counts.shape
  for _ in range(nboot):
   C=np.stack([self.counts[k,rng.integers(R,size=R)]for k in range(K)])
   # F stored in centered gauge with original S0: do not reuse after bootstrap S0 changes.
   f,ok=self.solve(C)
   if not ok:raise RuntimeError('bootstrap non convergé')
   out.append(f[-1]-f[0])
  return np.asarray(out)
