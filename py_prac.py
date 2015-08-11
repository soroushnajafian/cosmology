import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
from de_plot import DE_vis

####################################################
#  Define your model by specifying the parameters

LCDM = de_hh.LCDM(0.3, 0.0, 0.0005, 0.7)
#TOPO = de_hh.Topo_defc_2D(0.3, 0.0001, 0.7)
#Phan = de_hh.Phan_DE(0.3, 0.0001, 0.7)
XCDM = de_hh.XCDM(0.3, 0.0, -0.9, 0.7)
#CG = de_hh.CG(0.3, 0.0001, 2, 0.7)
GCG = de_hh.GCG(0.3, 0.0, 0.98, 0.93, 0.7)
#W_Linear = de_hh.W_Linear(0.3, 0.0001, -0.9, 0.2, 0.7)
W_CPL = de_hh.W_CPL(0.3, 0.0001, -1.0, 0.5, 0.7)
#DE_Casimir = de_hh.DE_Casimir(0.3, 0.0001, 0.02, 0.7)
DE_Card = de_hh.DE_Card(0.3, 0.0005, 0.38, 0.7)
DGP = de_hh.DGP(0.3, 0.14, 0.7)
#DDG = de_hh.DDG(0.3, 0.4, 0.7)
#RS = de_hh.RS(0.3, 0.0001, 0.05, 0.7)
RSL = de_hh.RSL(0.3, 0.0001, 0.05, 0.2, 0.7)
S_Brane1 = de_hh.S_Brane1(0.3, 0.0, 0.2, 1, 0.7)
#S_Brane2 = de_hh.S_Brane2(0.3, 0.0001, 0.02, 0.1, 0.7)
#q_Linear = de_hh.q_Linear(-0.4, 0.3, 0.7)
q_CPL = de_hh.q_CPL(-0.6, 1.5, 0.7)

Quintessence = scalar_hh.Quintessence(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Phantom = scalar_hh.Phantom(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Tachyon = scalar_hh.Tachyon(0.8, 0.0001, 1, 2, 0.0, 0.7)
DiGhCondensate = scalar_hh.DiGhCondensate(0.00085, 0.00085, 1, 0, 0.1, 0.7)

H0 = 70

z = np.linspace(0, 3, 100)

###################################################################################
'''
plt.figure(1)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.xlim([0,3])
plt.ylim([0,400])


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



#################################################################################

plt.figure(4)
plt.xlabel('z')
plt.ylabel('$w_{de}$(z)')
plt.xlim([0,3])
plt.ylim([-1.5,01])


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
plt.ylabel('$w_{de}$(z)')
plt.xlim([0,3])
plt.ylim([-1.5,01])


w = np.array(map(DE_Card.w_de, z))
w_C, = plt.plot(z, w, 'b', label='Cardissian expansion')

w = np.array(map(DGP.w_de, z))
w_D, = plt.plot(z, w, 'r', label='DGP')

w = np.array(map(RSL.w_de, z))
w_R, = plt.plot(z, w, 'g', label='RS Braneworld')

w = np.array(map(S_Brane1.w_de, z))
w_S, = plt.plot(z, w, 'k', label='Shtanov braneworld')

#w = np.array(map(q_CPL.w_de, z))
#w_q, = plt.plot(z, w, 'c', label='CPL deceleration')

plt.legend(handles=[w_C, w_D, w_R, w_S], loc=4)

########################################
plt.figure(6)
plt.xlabel('z')
plt.ylabel('$w_{de}$(z)')
plt.xlim([0,3])
plt.ylim([-1.5,01])


w = np.array(map(Quintessence.w_de, z))
w_Q, = plt.plot(z, w, 'b', label='Quintessence')

w = np.array(map(Phantom.w_de, z))
w_P, = plt.plot(z, w, 'r', label='Phantom')

w = np.array(map(Tachyon.w_de, z))
w_T, = plt.plot(z, w, 'g', label='Tachyon')

w = np.array(map(DiGhCondensate.w_de, z))
w_D, = plt.plot(z, w, 'k', label='Dilatonic Ghost Condensate')

#w = np.array(map(q_CPL.w_de, z))
#w_q, = plt.plot(z, w, 'c', label='CPL deceleration')

plt.legend(handles=[w_Q, w_P, w_T, w_D], loc=2)


###################################################################################
plt.figure(7)
plt.xlabel('z')
plt.ylabel('q(z)')
plt.xlim([0,3])
#plt.ylim([-1.5,0])

q = np.array(map(LCDM.q, z))
q_L, = plt.plot(z, q, 'b', label='$\Lambda$ CDM')

q = np.array(map(XCDM.q, z))
q_X, = plt.plot(z, q, 'r', label='XCDM')

q = np.array(map(GCG.q, z))
q_G, = plt.plot(z, q, 'g', label='GCG')

q = np.array(map(W_CPL.q, z))
q_W, = plt.plot(z, q, 'k', label='CPL')

q = np.array(map(q_CPL.q, z))
q_q, = plt.plot(z, q, 'c', label='CPL deceleration')

plt.legend(handles=[q_L, q_X, q_G, q_W, q_q], loc=4)

#####################################
plt.figure(8)
plt.xlabel('z')
plt.ylabel('q(z)')
plt.xlim([0,3])

q = np.array(map(DE_Card.q, z))
q_C, = plt.plot(z, q, 'b', label='Cardissian expansion')

q = np.array(map(DGP.q, z))
q_D, = plt.plot(z, q, 'r', label='DGP')

q = np.array(map(RSL.q, z))
q_R, = plt.plot(z, q, 'g', label='RS braneworld')

q = np.array(map(S_Brane1.q, z))
q_S, = plt.plot(z, q, 'k', label='Shtanov braneworld')


plt.legend(handles=[q_C, q_D, q_R, q_S], loc=4)

####################################
plt.figure(9)
plt.xlabel('z')
plt.ylabel('q(z)')
plt.xlim([0,3])

q = np.array(map(Quintessence.q, z))
q_Q, = plt.plot(z, q, 'b', label='Quintessence')

q = np.array(map(Phantom.q, z))
q_P, = plt.plot(z, q, 'r', label='Phantom')

q = np.array(map(Tachyon.q, z))
q_T, = plt.plot(z, q, 'g', label='Tachyon')

q = np.array(map(DiGhCondensate.q, z))
q_D, = plt.plot(z, q, 'k', label='Dilatonic Ghost Condensate')

plt.legend(handles=[q_Q, q_P, q_T, q_D], loc=2)


####################################################################################
plt.figure(10)
plt.xlabel('z')
plt.ylabel('$D_{L}$(z)')
plt.xlim([0,3])

D_L = np.array(map(LCDM.D_L, z))
D_L_L, = plt.plot(z, D_L, 'b', label='$\Lambda$ CDM')

D_L = np.array(map(XCDM.D_L, z))
D_L_X, = plt.plot(z, D_L, 'r', label='XCDM')

D_L = np.array(map(GCG.D_L, z))
D_L_G, = plt.plot(z, D_L, 'g', label='GCG')

D_L = np.array(map(W_CPL.D_L, z))
D_L_W, = plt.plot(z, D_L, 'k', label='CPL')

D_L = np.array(map(q_CPL.D_L, z))
D_L_q, = plt.plot(z, D_L, 'c', label='CPL deceleration')

plt.legend(handles=[D_L_L, D_L_X, D_L_G, D_L_W, D_L_q], loc=2)

####################################
plt.figure(11)
plt.xlabel('z')
plt.ylabel('$D_{L}$(z)')
plt.xlim([0,3])

D_L = np.array(map(DE_Card.D_L, z))
D_L_C, = plt.plot(z, D_L, 'b', label='Cardissian expansion')

D_L = np.array(map(DGP.D_L, z))
D_L_D, = plt.plot(z, D_L, 'r', label='DGP')

D_L = np.array(map(RSL.D_L, z))
D_L_R, = plt.plot(z, D_L, 'g', label='RS braneworld')

D_L = np.array(map(S_Brane1.D_L, z))
D_L_S, = plt.plot(z, D_L, 'k', label='Shtanov braneworld')


plt.legend(handles=[D_L_C, D_L_D, D_L_R, D_L_S], loc=2)

#################################
plt.figure(12)
plt.xlabel('z')
plt.ylabel('$D_{L}$(z)')
plt.xlim([0,3])

D_L = np.array(map(Quintessence.D_L, z))
D_L_Q, = plt.plot(z, D_L, 'b', label='Quintessence')

D_L = np.array(map(Phantom.D_L, z))
D_L_P, = plt.plot(z, D_L, 'r', label='Phantom')

D_L = np.array(map(Tachyon.D_L, z))
D_L_T, = plt.plot(z, D_L, 'g', label='Tachyon')

D_L = np.array(map(DiGhCondensate.D_L, z))
D_L_D, = plt.plot(z, D_L, 'k', label='Dilatonic Ghost Condensate')


plt.legend(handles=[D_L_Q, D_L_P, D_L_T, D_L_D], loc=2)


######################################################################################
plt.figure(13)
plt.xlabel('z')
plt.ylabel('D(z)')
plt.xlim([0,3])

zz = 3.0

z, D = LCDM.D_solu(zz)[:2]
D_L, = plt.plot(z, D, 'b', label='$\Lambda$ CDM')

z, D = XCDM.D_solu(zz)[:2]
D_X, = plt.plot(z, D, 'r', label='XCDM')

z, D = GCG.D_solu(zz)[:2]
D_G, = plt.plot(z, D, 'g', label='GCG')

z, D = W_CPL.D_solu(zz)[:2]
D_W, = plt.plot(z, D, 'k', label='CPL')

plt.legend(handles=[D_L, D_X, D_G, D_W], loc=1)

##################################################
plt.figure(14)
plt.xlabel('z')
plt.ylabel('D(z)')
plt.xlim([0,3])

zz = 3.0

z, D = DE_Card.D_solu(zz)[:2]
D_C, = plt.plot(z, D, 'b', label='Cardissian expansion')

z, D = DGP.D_solu(zz)[:2]
D_D, = plt.plot(z, D, 'r', label='DGP')

z, D = RSL.D_solu(zz)[:2]
D_R, = plt.plot(z, D, 'g', label='RSL')

z, D = S_Brane1.D_solu(zz)[:2]
D_S, = plt.plot(z, D, 'k', label='Shtanov braneworld')

plt.legend(handles=[D_C, D_D, D_R, D_S], loc=1)

##################################################
plt.figure(15)
plt.xlabel('z')
plt.ylabel('D(z)')
plt.xlim([0,3])

zz = 3.0

z, D = Quintessence.D_solu(zz)[:2]
D_Q, = plt.plot(z, D, 'b', label='Quintessence')

z, D = Phantom.D_solu(zz)[:2]
D_P, = plt.plot(z, D, 'r', label='Phantom')

z, D = Tachyon.D_solu(zz)[:2]
D_T, = plt.plot(z, D, 'g', label='Tachyon')

z, D = DiGhCondensate.D_solu(zz)[:2]
D_D, = plt.plot(z, D, 'k', label='Dilatonic Ghost Condensate')

plt.legend(handles=[D_Q, D_P, D_T, D_D], loc=1)

######################################################################################
plt.figure(16)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.xlim([0,3])

zz = 3.0

z, D, f = LCDM.D_solu(zz)[:3]
f_L, = plt.plot(z, f, 'b', label='$\Lambda$ CDM')

z, D, f = XCDM.D_solu(zz)[:3]
f_X, = plt.plot(z, f, 'r', label='XCDM')

z, D, f = GCG.D_solu(zz)[:3]
f_G, = plt.plot(z, f, 'g', label='GCG')

z, D, f = W_CPL.D_solu(zz)[:3]
f_W, = plt.plot(z, f, 'k', label='CPL')

plt.legend(handles=[f_L, f_X, f_G, f_W], loc=4)

##################################################
plt.figure(17)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.xlim([0,3])

zz = 3.0

z, D, f = DE_Card.D_solu(zz)[:3]
f_C, = plt.plot(z, f, 'b', label='Cardissian expansion')

z, D, f = DGP.D_solu(zz)[:3]
f_D, = plt.plot(z, f, 'r', label='DGP')

z, D, f = RSL.D_solu(zz)[:3]
f_R, = plt.plot(z, f, 'g', label='RSL')

z, D, f = S_Brane1.D_solu(zz)[:3]
f_S, = plt.plot(z, f, 'k', label='Shtanov braneworld')

plt.legend(handles=[f_C, f_D, f_R, f_S], loc=4)

##################################################
plt.figure(18)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.xlim([0,3])

zz = 3.0

z, D, f = Quintessence.D_solu(zz)[:3]
f_Q, = plt.plot(z, f, 'b', label='Quintessence')

z, D, f = Phantom.D_solu(zz)[:3]
f_P, = plt.plot(z, f, 'r', label='Phantom')

z, D, f = Tachyon.D_solu(zz)[:3]
f_T, = plt.plot(z, f, 'g', label='Tachyon')

z, D, f = DiGhCondensate.D_solu(zz)[:3]
f_D, = plt.plot(z, f, 'k', label='Dilatonic Ghost Condensate')

plt.legend(handles=[f_Q, f_P, f_T, f_D], loc=4)

############################################################

'''

EDE = de_hh.EDE(0.3, 0.1, -1.0, 0.7)
H = H0*np.array(map(EDE.E, z))
plt.figure(1)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.xlim([0,3])
plt.ylim([0,400])
plt.plot(z, H, 'b', label='$\Lambda$ CDM')






plt.show()

