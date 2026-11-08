import mpmath as mp
mp.mp.dps = 70

def bess(n,b):
    return mp.besseli(n,b)

def correct(b):
    return bess(2,b)/bess(1,b)

def omitted_half(b):
    i0,i1,i2=[bess(k,b) for k in (0,1,2)]
    return (2*i1-i0-i2)/(2*(i0-i1))

def D(theta,t):
    s=mp.sin(t*theta)
    return 1-2*s*mp.cos((1-t)*theta)/mp.sin(theta)+2*s*s/mp.sin(theta)**2

print('Q4(d), <cos(theta)> = I2/I1 (independant de t et V)')
for t in [mp.mpf('.5'),mp.mpf('.3')]:
    for b in [mp.mpf(2),mp.mpf(8)]:
        print('t =',t,'beta =',b,':',mp.nstr(correct(b),25))
print('Q4(e), t=1/2, Hausdorff sans co-aire')
for b in [mp.mpf(2),mp.mpf(8)]:
    print('beta =',b,':',mp.nstr(omitted_half(b),25))
print('Q4(e), ecart relatif (oublie-correct)/correct pour <1-cos>')
for b in [mp.mpf(2),mp.mpf(8),mp.mpf(1000)]:
    good=1-correct(b); bad=1-omitted_half(b)
    print('beta =',b,':',mp.nstr((bad-good)/good,25),'; correct =',mp.nstr(good,20),'; oublie =',mp.nstr(bad,20))
print('Q5(d), <cos(delta)> = I1/I0')
for b in [mp.mpf(2),mp.mpf(8)]:
    print('beta =',b,':',mp.nstr(bess(1,b)/bess(0,b),25))

# Independent quadrature checks, including the Haar conditional law for t=.3.
for b in [mp.mpf(2),mp.mpf(8)]:
    dg=mp.quad(lambda x: mp.exp(b*mp.cos(x))*mp.sin(x)**2,[0,mp.pi])
    ng=mp.quad(lambda x: mp.cos(x)*mp.exp(b*mp.cos(x))*mp.sin(x)**2,[0,mp.pi])
    db=mp.quad(lambda x: mp.exp(b*mp.cos(x))*(1-mp.cos(x)),[0,mp.pi])
    nb=mp.quad(lambda x: mp.cos(x)*mp.exp(b*mp.cos(x))*(1-mp.cos(x)),[0,mp.pi])
    print('quadrature errors beta',b,mp.nstr(ng/dg-correct(b),5),mp.nstr(nb/db-omitted_half(b),5))

# Check coarea/Hausdorff ratio: (d sigma/d omega)/sqrt(det JJt) = det E(omega).
# This is the statement that the conditional Haar density in W is one.
print('Q4(b,c), coarea identity checks')
for t in [mp.mpf('.3'),mp.mpf('.5'),mp.mpf('1.0')]:
    for th in [mp.mpf('.2'),mp.mpf('1.4'),mp.mpf('2.8')]:
        a=1-2*t+2*t*t
        d=D(th,t)
        rho_over_rootG=mp.sqrt(a)*mp.sin(th)**2*d/th**2/(mp.sqrt(a)*d)
        detE=mp.sin(th)**2/th**2
        print('t =',t,'theta =',th,'error =',mp.nstr(rho_over_rootG-detE,5))

print('Q4(a), det(JJ^T) checks at t=1/2')
for th in [mp.mpf('.0001'),mp.mpf('.4'),mp.mpf('1.7'),mp.mpf('3.0')]:
    c=mp.cos(th/2); q2=1/c**2
    lam=1-2*mp.mpf('.5')*1+2*mp.mpf('.5')**2*q2
    direct=mp.mpf('.5')*lam**2
    closed=1/(2*(1+mp.cos(th))**2)
    print('theta =',th,'error =',mp.nstr(direct-closed,5))
