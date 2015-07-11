import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
from de_plot import DE_vis

####################################################
#  Define your model by specifying the parameters

LCDM = de_hh.LCDM(0.3, 0.0001, 0.0005, 0.7)
TOPO = de_hh.Topo_defc_2D(0.3, 0.0001, 0.7)
Phan = de_hh.Phan_DE(0.3, 0.0001, 0.7)
XCDM = de_hh.XCDM(0.3, 0.0001, -0.8, 0.7)
CG = de_hh.CG(0.3, 0.0001, 2, 0.7)
GCG = de_hh.GCG(0.3, 0.0001, 0.8, 0.0, 0.7)
W_Linear = de_hh.W_Linear(0.3, 0.0001, -0.9, 0.2, 0.7)
W_CPL = de_hh.W_CPL(0.3, 0.0001, -0.9, 0.1, 0.7)
DE_Casimir = de_hh.DE_Casimir(0.3, 0.0001, 0.02, 0.7)
DE_Card = de_hh.DE_Card(0.3, 0.0005, 0.8, 0.7)
DGP = de_hh.DGP(0.3, 0.002, 0.7)
DDG = de_hh.DDG(0.3, 0.4, 0.7)
RS = de_hh.RS(0.3, 0.0001, 0.05, 0.7)
RSL = de_hh.RSL(0.3, 0.0001, 0.05, 0.2, 0.7)
S_Brane1 = de_hh.S_Brane1(0.3, 0.0001, 0.02, 1, 0.7)
S_Brane2 = de_hh.S_Brane2(0.3, 0.0001, 0.02, 0.1, 0.7)
q_Linear = de_hh.q_Linear(-0.4, 0.3, 0.7)
q_CPL = de_hh.q_CPL(-0.4, 0.3, 0.7)

Quintessence = scalar_hh.Quintessence(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Phantom = scalar_hh.Phantom(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Tachyon = scalar_hh.Tachyon(0.8, 0.0001, 1, 2, 0.0, 0.7)
DiGhCondensate = scalar_hh.DiGhCondensate(0.00085, 0.00085, 1, 0, 0.1, 0.7)

H0 = 70
'''
################################################

plt.figure(1)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.xlim([0,3])
plt.ylim([0,400])

z = np.linspace(0, 3, 100)

H = H0*np.array(map(LCDM.E, z))
H_L, = plt.plot(z, H, 'b', label='$\Lambda$ CDM')

H = H0*np.array(map(XCDM.E, z))
H_X, = plt.plot(z, H, 'r', label='XCDM')

H = H0*np.array(map(GCG.E, z))
H_G, = plt.plot(z, H, 'g', label='GCG')

H = H0*np.array(map(W_CPL.E, z))
H_W, = plt.plot(z, H, 'k', label='CPL')

H = H0*np.array(map(q_CPL.E, z))
H_q, = plt.plot(z, H, 'c', label='CPL deceleration')

plt.legend(handles=[H_L, H_X, H_G, H_W, H_q], loc=2)

##############################
plt.figure(2)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.xlim([0,3])
plt.ylim([0,400])

H = H0*np.array(map(DE_Card.E, z))
H_C, = plt.plot(z, H, 'b', label='Cardissian expansion')


H = H0*np.array(map(DGP.E, z))
H_D, = plt.plot(z, H, 'r', label='DGP')


H = H0*np.array(map(RSL.E, z))
H_R, = plt.plot(z, H, 'g', label='RS braneworld')


H = H0*np.array(map(S_Brane1.E, z))
H_S, = plt.plot(z, H, 'k', label='Shtanov braneworld')

plt.legend(handles=[H_C, H_D, H_R, H_S], loc=2)

##################################
plt.figure(3)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.xlim([0,3])
plt.ylim([0,400])

H = H0*np.array(map(Quintessence.E, z))
H_Q, = plt.plot(z, H, 'b', label='Quintessence')


H = H0*np.array(map(Phantom.E, z))
H_P, = plt.plot(z, H, 'r', label='Phantom')


H = H0*np.array(map(Tachyon.E, z))
H_T, = plt.plot(z, H, 'g', label='Tachyon')


H = H0*np.array(map(DiGhCondensate.E, z))
H_D, = plt.plot(z, H, 'k', label='Dilatonic Ghost Condensate')

plt.legend(handles=[H_Q, H_P, H_T, H_D], loc=2)

'''

###################################

plt.figure(4)
plt.xlabel('z')
plt.ylabel('w(z)')
plt.xlim([0,3])
plt.ylim([-1.5,0])


w = np.array(map(LCDM.w_de, z))
w_L, = plt.plot(z, w, 'b', label='$\Lambda$ CDM')

w = np.array(map(XCDM.w_de, z))
w_X, = plt.plot(z, w, 'r', label='XCDM')

w = np.array(map(GCG.w_de, z))
w_G, = plt.plot(z, w, 'g', label='GCG')

w = np.array(map(W_CPL.w_de, z))
w_W, = plt.plot(z, w, 'k', label='CPL')

#w = np.array(map(q_CPL.w_de, z))
#w_q, = plt.plot(z, w, 'c', label='CPL deceleration')

plt.legend(handles=[w_L, w_X, w_G, w_W], loc=2)

#####################################
plt.figure(5)
plt.xlabel('z')
plt.ylabel('w(z)')
plt.xlim([0,3])
plt.ylim([-1.5,0])


w = np.array(map(.w_de, z))
w_L, = plt.plot(z, w, 'b', label='$\Lambda$ CDM')

w = np.array(map(XCDM.w_de, z))
w_X, = plt.plot(z, w, 'r', label='XCDM')

w = np.array(map(GCG.w_de, z))
w_G, = plt.plot(z, w, 'g', label='GCG')

w = np.array(map(W_CPL.w_de, z))
w_W, = plt.plot(z, w, 'k', label='CPL')

#w = np.array(map(q_CPL.w_de, z))
#w_q, = plt.plot(z, w, 'c', label='CPL deceleration')

plt.legend(handles=[w_L, w_X, w_G, w_W], loc=2)

############################################################












plt.show()

