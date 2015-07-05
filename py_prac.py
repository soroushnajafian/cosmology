import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
'''
def ss(x):
    return x**x+x

a = np.linspace(0,10,10)
b = map(ss,a)
print(a,b)
'''

import de_hh
import scalar_hh
'''
LCDM_nf = de_hh.LCDM_nf(0.3, 0, 0.00004)
print(map(LCDM_nf.hh, np.linspace(0,1,10)))


DE_Card = de_hh.DE_Card(0.3, 0.0004, 1)
print(map(DE_Card.hh, np.linspace(0,1,10)))

DDG = de_hh.DDG(0.3, 0.2)
print(map(DDG.hh, np.linspace(0,1,10)))
'''
'''
Qui = scalar_hh.Quintessence(0.01, 0.01, 2, 1, 0.0)
a, Z, x, y, z = Qui.solu(0.0)
#print(Z)

print(len(a))
w = (x**2-y**2)/(x**2+y**2)
Oo = x**2+y**2
plt.figure(1)
plt.plot(Z,Oo)
#plt.xlim([0, 10])
plt.show()
'''
LCDM = de_hh.LCDM(0.3, 0.001, 0.0005, 0.7)
z = 10**(np.linspace(-3,1,10))
weff = map(LCDM.weff, z)
q = map(LCDM.q, z)
qq = 1.0/np.array(q)
H = map(LCDM.E, z)
D_L = map(LCDM.D_L, z)
mu = map(LCDM.mu, z)
zz, D, D1, f= LCDM.D_solu(2)
plt.figure(1)
plt.plot(1/(1+zz),D)
f = -(1+zz)/D*D1
plt.figure(2)
plt.plot(zz,f)
#plt.xscale('log')

'''
Topo = de_hh.Topo_defc_2D(0.25, 0.01)
weff1 = map(Topo.weff,z)
plt.figure(1)
plt.plot(z,weff1,'r')
plt.xscale('log')
'''

plt.show()
