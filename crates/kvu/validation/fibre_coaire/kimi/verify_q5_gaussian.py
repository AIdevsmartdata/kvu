import numpy as np
K=np.array([[2.0,.3,.1],[.3,1.5,.2],[.1,.2,1.0]])
Q=np.array([[1.0,-.4,.2],[.1,.7,-.3]])
phi=np.array([.8,-.2])
beta=2.7
Ki=np.linalg.inv(K); H=Q@Ki@Q.T
hard_mean=Ki@Q.T@np.linalg.solve(H,phi)
hard_cov=(Ki-Ki@Q.T@np.linalg.solve(H,Q@Ki))/beta
print('hard constraint mean residual:',np.max(np.abs(Q@hard_mean-phi)))
print('hard covariance tangent residual:',np.max(np.abs(Q@hard_cov@Q.T)))
for eps in [.05,.4,2.0]:
    A=beta*K+Q.T@Q/eps**2
    m=np.linalg.solve(A,Q.T@phi/eps**2)
    cov=np.linalg.inv(A)
    mf=Ki@Q.T@np.linalg.solve(H+beta*eps**2*np.eye(2),phi)
    covf=(Ki-Ki@Q.T@np.linalg.solve(H+beta*eps**2*np.eye(2),Q@Ki))/beta
    print('epsilon =',eps,'mean error =',np.max(np.abs(m-mf)),'covariance error =',np.max(np.abs(cov-covf)))
    print('soft constraint bias Qm-phi =',Q@m-phi)
