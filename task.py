from __future__ import division
import math

try:
    import numpy as np
except ImportError:
    print('Numpy is required, please make sure it is installed properly')

try:
    import scipy as sp
except ImportError:
    print('Scipy is required, please make sure it is installed properly')

from scipy import *
import de_hh
from mock_data import *
import scipy.optimize as op
import emcee
import time

class task_GoF:

    def __init__(self, modelname, dataname, N, Func, init_pos, bnds, Mock=False, method='L-BFGS-B', fitmodel=None, fitmodelrd=None, fitmodelsig8=None):

        self.modelname = modelname
        self.dataname = dataname
        self.N = N
        self.Func = Func
        self.init_pos = init_pos
        self.bnds = bnds
        self.fitmodel = fitmodel
        self.fitmodelrd = fitmodelrd
        self.fitmodelsig8 = fitmodelsig8
        
        self.dim = len(init_pos)
        self.pos = np.empty([0, self.dim])
        

        if Mock == False:
            self.chi2 = []
            for i in range(self.N):
                self.aa = self.init_pos+0.01*np.random.random(1)*self.init_pos   #!!!!!#
                self.result = op.minimize(self.Func, self.aa, method=method, bounds=self.bnds)
                self.pos=np.vstack((self.pos, self.result["x"]))
                #self.chi2.append(self.result["fun"])
                self.chi2.append(np.asscalar(self.Func(self.result["x"])))
                #print(i, self.result["fun"], self.result["x"])
                print(i, self.Func(self.result["x"]), self.result["x"])
            self.chi2=np.array(self.chi2)
            self.ind = np.where(self.chi2==min(self.chi2))
            self.bestfit=self.pos[np.where(self.chi2==min(self.chi2))][0]
            print(np.hstack((min(self.chi2), self.bestfit)))
            np.savetxt('output/bestfit/best_'+modelname+'_'+dataname+'.dat',  np.hstack((min(self.chi2), self.bestfit)), fmt='%.6f')

        elif Mock == True:
            self.chi2 = []
            for i in range(self.N):
                self.aa = self.init_pos
                data_generate(self.modelname, self.fitmodel, self.fitmodelrd, self.fitmodelsig8)
                self.result = op.minimize(self.Func, self.aa, method=method, bounds=self.bnds)
                self.chi2.append((self.result["fun"]))
                print('Run:', i, self.result["fun"], self.result["x"])
            np.savetxt('output/chi2_test/chi2_test_'+modelname+'_'+dataname+'.dat', self.chi2)


class task_MCMC:
    def __init__(self, modelname, dataname, position, nwalkers, Nstep, Nburnin, logf):
       
        ndim = len(position[0])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logf)
            
        sampler.run_mcmc(position, Nstep)
        samples = sampler.chain[:, Nburnin:, :].reshape((-1, ndim))
        chi2p = sampler.lnprobability[:, Nburnin:].reshape(-1)

        np.savetxt("./output/MCMC/"+modelname+"_"+dataname+".dat", samples, fmt='%.6e')
        np.savetxt("./output/MCMC/"+modelname+"_"+dataname+"_chi2.dat", -chi2p, fmt='%.6e')








