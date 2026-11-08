import mpmath as mp
import numpy as np

mp.mp.dps = 60
I2 = mp.eye(2)
sig = [mp.matrix([[0, 1], [1, 0]]),
       mp.matrix([[0, -1j], [1j, 0]]),
       mp.matrix([[1, 0], [0, -1]])]

def mexp(a):
    return mp.expm(a)

def dot(a):
    return sum((a[i] * sig[i] for i in range(3)), mp.zeros(2))

def su2(v):
    th = mp.sqrt(sum(x*x for x in v))
    if th < mp.mpf('1e-40'):
        return I2 + 1j*dot(v)
    return mp.cos(th)*I2 + 1j*mp.sin(th)/th*dot(v)

def logv(W):
    c = mp.re((W[0,0]+W[1,1])/2)
    c = max(mp.mpf(-1), min(mp.mpf(1), c))
    th = mp.acos(c)
    if th < mp.mpf('1e-30'):
        z = [mp.im(W[0,1]), mp.re(W[0,1]), mp.im(W[0,0])]
        return mp.matrix(z)
    return th/mp.sin(th)*mp.matrix([mp.im(W[0,1]), mp.re(W[0,1]), mp.im(W[0,0])])

def crossmat(v):
    return mp.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def Eg(v):
    th = mp.sqrt(sum(x*x for x in v))
    X = crossmat(v)
    if th < mp.mpf('1e-20'):
        return mp.eye(3) - X + mp.mpf(2)/3*X*X
    a = mp.sin(th)**2/th**2
    b = (th-mp.sin(th)*mp.cos(th))/th**3
    return mp.eye(3) - a*X + b*X*X

def Lg(v):
    th = mp.sqrt(sum(x*x for x in v))
    X = crossmat(v)
    if th < mp.mpf('1e-20'):
        return mp.eye(3) + X + X*X/3
    b = 1/th**2 - 1/(th*mp.tan(th))
    return mp.eye(3) + X + b*X*X

def Rof(v):
    th = mp.sqrt(sum(x*x for x in v))
    if th < mp.mpf('1e-30'):
        return mp.eye(3) - 2*crossmat(v)
    n = crossmat(v/th)
    return mp.eye(3) - mp.sin(2*th)*n + (1-mp.cos(2*th))*n*n

def err(A, B):
    return max(abs(A[i,j]-B[i,j]) for i in range(A.rows) for j in range(A.cols))

# Q2, with h^2 differences and mp arithmetic.
w = mp.matrix([mp.mpf('0.43'), mp.mpf('-0.71'), mp.mpf('0.36')])
W = su2(w)
h = mp.mpf('1e-20')
Lg_num = mp.zeros(3); Ld_num = mp.zeros(3); Eg_num = mp.zeros(3)
for j in range(3):
    e = mp.zeros(3,1); e[j] = h
    Lg_num[:,j] = (logv(su2(e)*W)-logv(su2(-e)*W))/(2*h)
    Ld_num[:,j] = (logv(W*su2(e))-logv(W*su2(-e)))/(2*h)
    Wp = su2(w+e); Wm = su2(w-e)
    # Central differences of the left-trivialized output perturbation.
    vp = logv(Wp*W**-1); vm = logv(Wm*W**-1)
    Eg_num[:,j] = (vp-vm)/(2*h)
print('Q2 max error L_g =', mp.nstr(err(Lg_num, Lg(w)), 8))
print('Q2 max error L_d =', mp.nstr(err(Ld_num, Lg(-w)), 8))
print('Q2 max error E_g =', mp.nstr(err(Eg_num, Eg(w)), 8))

# Q3 example: two fine links, reference path (1, -2, 1),
# family paths (-2, 2, 2), (1, 1, -2), (2, -1, 2).
U = [su2(mp.matrix([mp.mpf('0.31'), mp.mpf('-0.22'), mp.mpf('0.17')])),
     su2(mp.matrix([mp.mpf('-0.27'), mp.mpf('0.41'), mp.mpf('0.29')]))]
paths = [[(0,1),(1,-1),(0,1)], [(1,-1),(1,1),(1,1)],
         [(0,1),(0,1),(1,-1)], [(1,1),(0,-1),(1,1)]]
weights = [mp.mpf('0.21'), mp.mpf('0.34'), mp.mpf('0.27')]
weights.append(1-sum(weights))

def path_u(path, links):
    out = I2
    for ell,s in path:
        out *= links[ell] if s == 1 else links[ell]**-1
    return out

def Ts(path):
    blocks = [mp.zeros(3) for _ in range(2)]
    A = I2
    for ell,s in path:
        if s == 1:
            blocks[ell] += Rof(logv(A))
            A *= U[ell]
        else:
            A *= U[ell]**-1
            blocks[ell] -= Rof(logv(A))
    return blocks

def coarse(links):
    u0 = path_u(paths[0], links)
    ws = [logv(u0**-1*path_u(p,links)) for p in paths[1:]]
    wb = sum((weights[i]*ws[i] for i in range(3)), mp.zeros(3,1))
    return u0*su2(wb)

u0 = path_u(paths[0],U)
r0 = Rof(logv(u0))
wb = sum((weights[i]*logv(u0**-1*path_u(paths[i+1],U)) for i in range(3)), mp.zeros(3,1))
eb = Eg(wb)
Bs = []
for p in paths[1:]:
    wg = logv(u0**-1*path_u(p,U))
    Bs.append(r0*eb*Lg(wg)*r0.T)
B0 = mp.eye(3)-sum((weights[i]*Bs[i] for i in range(3)),mp.zeros(3))
T0 = Ts(paths[0])
Jblocks = [B0*T0[j] for j in range(2)]
for i,p in enumerate(paths[1:]):
    T = Ts(p)
    for j in range(2):
        Jblocks[j] += weights[i]*Bs[i]*T[j]

def Jnum_for_link(ell):
    out = mp.zeros(3)
    for j in range(3):
        e = mp.zeros(3,1); e[j] = h
        Up = list(U); Um = list(U)
        Up[ell] = su2(e)*Up[ell]; Um[ell] = su2(-e)*Um[ell]
        Cp = coarse(Up); Cm = coarse(Um)
        C0 = coarse(U)
        # First-order comparison around the common unperturbed coarse value.
        vp = logv(Cp*C0**-1); vm = logv(Cm*C0**-1)
        out[:,j] = (vp-vm)/(2*h)
    return out

for ell in range(2):
    print('Q3 max error J block link',ell+1,'=',mp.nstr(err(Jnum_for_link(ell),Jblocks[ell]),8))
print('weights =', [mp.nstr(x,5) for x in weights])
