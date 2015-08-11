from __future__ import division

try:
    import numpy as np
except ImportError:
    print('Numpy is required, please make sure it is installed properly')

try:
    import scipy as sp
except ImportError:
    print('Scipy is required, please make sure it is installed properly')

import numpy as np
import scipy as sp
import math
import de_hh


class MCMC:
    def __init__(self, cov):
        
        self.cov = cov  # the covariance matrix of the proposal: assumed it to be gaussian

    def chi2(self, th, obs, icov):
        #  the inverse of the covariance matrix of the data need to be specified before using the function
        '''
        Calculate the chi2 from the data and thereotical values
        '''
        self.th = th
        self.obsc = obs
        self.icovc = icov
        self.diff = self.th - self.obsc
        return np.dot(self.diff,np.dot(self.icovc,self.diff))/2.0

    ## need to define the prior distribution of the parameters
    
    def lnprob(self, modelname, param, z, obs, icov):
        '''
        Calculate the logarithm of the posterior probability distribution. This include the likelihood function and prior of the parameters
        '''
        self.modelname = str(modelname)
        self.param = param
        self.z = z
        self.obs = obs
        self.icovp = icov
        
        self.mcmcmodel = self.model(self.modelname, self.param)
        
        self.Onu = self.param[-2]
        self.Oba = self.param[-1]
        self.mcmc_rd = self.mcmcmodel.rd(self.Onu, self.Oba)
        
        self.lp = self.lnprior(self.modelname, self.param)
            #if not np.isfinite(self.lp):
            #return -np.inf
        self.lp2 = self.lnprior2(self.param)
        
        self.BAO = 1  # turn on BAO or not
        if self.BAO == 1:
        
            self.zh = self.z[0:4]
            self.zm = self.z[4:8]
            self.zv = self.z[8:11]
        
            self.Dh = self.obs[0:4]
            self.Dm = self.obs[4:8]
            self.Dv = self.obs[8:11]
            
            self.sig = 1.0/np.diag(self.icovp)
            self.sigh = self.sig[0:4]
            self.sigm = self.sig[4:8]
            self.sigv = self.sig[8:11]
            self.icovh = np.diag(1.0/self.sigh)
            self.icovm = np.diag(1.0/self.sigm)
            self.icovv = np.diag(1.0/self.sigv)
            
            self.Dh_th = self.func_DH(self.modelname, self.zh, self.param)/self.mcmc_rd
            self.Dm_th = self.func_DM(self.modelname, self.zm, self.param)/self.mcmc_rd
            self.Dv_th = self.func_DV(self.modelname, self.zv, self.param)/self.mcmc_rd
            
            self.chi2h = self.chi2(self.Dh_th, self.Dh, self.icovh)
            self.chi2m = self.chi2(self.Dm_th, self.Dm ,self.icovm)
            self.chi2v = self.chi2(self.Dv_th, self.Dv, self.icovv)
            
            return -(self.chi2h + self.chi2m + self.chi2v)+self.lp+self.lp2
        
        else:
            return -self.chi2(self.func(self.modelname, self.z, self.param), self.obs, self.icov) +self.lp +self.lp2

    def sample(self, modelname, p0, Nstep, z, obs, icov):
        '''
        The sampling process. The method is M_H sampling. The parameters are:
            p0: initial position in the parameter space
            Nstep: the number of steps in one chain.
            z: the argument (redshift z) as the input of the thereotical function
            obs: observational values (data)
            icov: the inverse of the covariance matrix from data, which is same as in function: chi2(th, obs, icov)
        '''
        self.modelname = str(modelname)
        self.p = np.array(p0)
        self.Nstep = Nstep
        self.zz = z
        self.obss = obs
        self.icovvv = icov
        
        self.oldlnprob = self.lnprob(self.modelname, self.p, self.zz, self.obss, self.icovvv)     #  the initial value of the posterior pobability function
        
        self.acc = 0    #  to count the number of acceptance
        
        self.chain = np.hstack((np.array([int(self.acc), int(0), -self.oldlnprob]), [j for j in self.p]))
        #  the first entry of the chain, the format is: acceptance, trial, chi2, parameters (dimension varies in different model)
        
        #######  The M_H sampling process    #########
        
        for i in range(self.Nstep):
            self.q = np.random.multivariate_normal(self.p, self.cov)   # the distribution of the proposal which is a multivariate normal distribution. The covariance matrix as input can be tunned to reach certain performance.
            
            self.newlnprob = self.lnprob(self.modelname, self.q, self.zz, self.obss, self.icovvv)   #  the natural logarithm of the posterior probability function in the new parameter position.
            
            self.dif = self.newlnprob - self.oldlnprob
            
            if np.isinf(self.newlnprob) and np.isinf(self.oldlnprob):
                continue
            
            if self.dif < 0.0 :
                
                self.aran = np.random.random(1)
                self.dif_prob = np.exp(self.dif)
                
                if self.aran <= self.dif_prob:
                    
                    self.p = self.q
                    self.oldlnprob = self.newlnprob
                    self.acc += 1
                    self.chain_i = np.hstack((np.array([int(self.acc), int(i)+1, -self.oldlnprob]), [j for j in self.p]))
                    self.chain = np.vstack((self.chain, self.chain_i))
                    print(self.acc, i, -self.oldlnprob, [j for j in self.p])
                    
                else:
                    continue

            elif self.dif >= 0.0 :
                
                self.p = self.q
                self.oldlnprob = self.newlnprob
                self.acc += 1
                self.chain_i = np.hstack((np.array([int(self.acc), int(i)+1, -self.oldlnprob]), [j for j in self.p]))
                self.chain = np.vstack((self.chain, self.chain_i))
                print(self.acc, i, -self.oldlnprob, [j for j in self.p])
                    
        return self.chain

    def Nsample(self, modelname, NWalker, p00, Nstep, z, obs, icov):
        '''
        Almost the same as the function sample, but it can realise a number of chains which is controlled by NWalker.
        '''
        self.modelname = str(modelname)
        self.NWalker = NWalker
        self.p00 = np.array(p00)
        self.Nstep = Nstep
        self.zN = z
        self.obsN = obs
        self.icovN = icov
            
        self.Nchain = []
        
        if self.NWalker != len(self.p00):
            print("Error: The number of initial conditions and the number of walkers are not consistent.")
            exit()
        
        for ii in range(self.NWalker):
            self.p0 = self.p00[ii]
            self.Nchain.append(self.sample(self.modelname, self.p0, self.Nstep, self.zN, self.obsN, self.icovN))
        
        return self.Nchain

    '''
    The following function is an example of calling a particular quantity to calculate (the Hubble parameter H(z)). The other quantities or observables can be written in the same way.
    '''
    def func(self, modelname, z, param):
        '''
        The parameters are input of the particular cosmological models. This function transfers parameters to the functioin that will be compared with observation. For example, LCDM(Om, Ok, Or, h).H(z) ---> LCDM.H(Om, Ok, Or,h;z)
        '''
        self.modelname = str(modelname)
        self.zf = z
        self.param = param
        self.mcmcmodel = self.model(self.modelname, self.param)

        self.H = 100.0*self.param[3]*np.array(map(self.mcmcmodel.E, self.zf))
        return self.H
    
    '''
    The next three functions are the BAO quantities: DH, DM, DV.
    '''
    def func_DH(self, modelname, z, param):
        
        self.modelname = str(modelname)
        self.zf = z
        self.param = param
        self.mcmcmodel = self.model(self.modelname, self.param)
        
        return np.array(map(self.mcmcmodel.D_Hz, self.zf))
    
    def func_DM(self, modelname, z, param):
        
        self.modelname = str(modelname)
        self.zf = z
        self.param = param
        self.mcmcmodel = self.model(self.modelname, self.param)
    
        return np.array(map(self.mcmcmodel.D_M, self.zf))
    
    def func_DV(self, modelname, z, param):
        
        self.modelname = str(modelname)
        self.zf = z
        self.param = param
        self.mcmcmodel = self.model(self.modelname, self.param)
        
        return np.array(map(self.mcmcmodel.D_V, self.zf))
    
    '''
    The next part is related to cosmological model: choose the model and specify the prior of the parameters.
    '''
    def model(self, modelname, param):
        '''
        Choose the model by specifying "modelname" which is the same as the class name in de_hh
        '''
        
        self.modelname = str(modelname)
        self.param = param
        
        if self.modelname == 'LCDM':
            return de_hh.LCDM(self.param[0], self.param[1], self.param[2])
        
        elif self.modelname == 'Flat_LCDM':
            return de_hh.Flat_LCDM(self.param[0], self.param[1])
        
        elif self.modelname == 'Topo_defc_2':
            return de_hh.Topo_defc_2(self.param[0], self.param[1], self.param[2])
        
        elif self.modelname == 'Phan_DE':
            return de_hh.Phan_DE(self.param[0], self.param[1], self.param[2])
        
        elif self.modelname == 'XCDM':
            return de_hh.XCDM(self.param[0], self.param[1], self.param[2], self.param[3])
        
        elif self.modelname == 'CG':
            return de_hh.CG(self.param[0], self.param[1], self.param[2], self.param[3])
        
        elif self.modelname == 'GCG':
            return de_hh.GCG(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])
        
        elif self.modelname == 'W_Linear':
            return de_hh.W_Linear(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])
        
        elif self.modelname == 'W_CPL':
            return de_hh.W_CPL(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])
        
        elif self.modelname == 'DE_Casimir':
            return de_hh.DE_Casimir(self.param[0], self.param[1], self.param[2], self.param[3])
        
        elif self.modelname == 'DE_Card':
            # this model is a little special, since it includes radiation. need to check
            return de_hh.DE_Card(self.param[0], self.param[1], self.param[2])
        
        elif self.modelname == 'DGP':
            return de_hh.DGP(self.param[0], self.param[1], self.param[2])

        elif self.modelname == 'DDG':
            return de_hh.DDG(self.param[0], self.param[1], self.param[2], self.param[3])

        elif self.modelname == 'RS':
            return de_hh.RS(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])

        elif self.modelname == 'RSL':
            return de_hh.RSL(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])

        # the following two braneworld models should have additional priors, refer to de_hh.py
        elif self.modelname == 'S_Brane1':
            return de_hh.S_Brane1(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])

        elif self.modelname == 'S_Brane2':
            return de_hh.S_Brane2(self.param[0], self.param[1], self.param[2], self.param[3], self.param[4])

        elif self.modelname == 'q_Linear':
            return de_hh.q_Linear(self.param[0], self.param[1], self.param[2])

        elif self.modelname == 'q_CPL':
            return de_hh.q_CPL(self.param[0], self.param[1], self.param[2])

        elif self.modelname == 'EDE':
            return de_hh.EDE(self.param[0], self.param[1], self.param[2], self.param[3])




        else:
            print("I can't find your model")
            exit()


    def lnprior(self, modelname, param):
        '''
        Top-hat prior of the parameters, it strongly depends on cosmological model. If the model changes, the prior also need to be changed.
        '''
        self.modelname = str(modelname)
        self.param = param
        
        if self.modelname == 'LCDM':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 <= self.param[-2]*self.param[-3]**2 <= 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
    
        elif self.modelname == 'Flat_LCDM':
            if 0 < self.param[0] < 1.0 and 0 <= self.param[1] <= 1.0 and 0 <= self.param[-2]*self.param[-3]**2 <= 0.0107*0.6 and 0 <= self.param[-1] <= self.param[0]:
                return 0.0
            return -np.inf

        elif self.modelname == 'Topo_defc_2D':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
            
        elif self.modelname == 'Phan_DE':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'XCDM':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and -3.0 < self.param[2] < 0 and 0 < self.param[3] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'CG':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 2.0 and 0 < self.param[3] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'GCG':  #  astro-ph/0306319
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 2.0 and 0 < self.param[3] < 1.0 and 0 < self.param[4] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'W_Linear':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and -10 < self.param[2] < 0 and -10 < self.param[3] < 10 and 0 < self.param[4] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'W_CPL':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and -10 < self.param[2] < 0 and -10 < self.param[3] < 10 and 0 < self.param[4] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'DE_Casimir':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[3] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'DE_Card':
            if 0 < self.param[0] < 1.0 and 0 < self.param[1] < 1.0 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf

        elif self.modelname == 'DGP':
            if 0 < self.param[0] < 1.0 and 0 < self.param[1] < 1.0 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf

        #elif self.modelname == 'DDG':   #  Problem: need to check.

        elif self.modelname == 'RS':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[3] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname =='RSL':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[3] < 1.0 and 0 < self.param[4] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'S_Brane1':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[3] < 1.0 and 0 < self.param[4] < 1.0 and self.param[0]+self.param[1]+self.param[2]+2*self.param[3] >= 1.0and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'S_Brane2':
            if 0 < self.param[0] < 1.0 and -0.5 < self.param[1] < 0.5 and 0 < self.param[2] < 1.0 and 0 < self.param[3] < 1.0 and 0 < self.param[4] < 1.0 and self.param[0]+self.param[1]+self.param[2]+2*self.param[3] <= 1.0and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'q_Linear':
            if -10 < self.param[0] < 10 and -10 < self.param[1] < 10 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
        elif self.modelname == 'q_CPL':
            if -10 < self.param[0] < 10 and -10 < self.param[1] < 10 and 0 < self.param[2] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
    
        elif self.modelname == 'EDE':
            if 0 < self.param[0] < 1.0 and 0 < self.param[1] < 1.0 and -3.0 < self.param[2] < 1 and 0 < self.param[3] < 1.0 and 0 < self.param[-2]*self.param[-3]**2 < 0.0107*0.6 and 0 < self.param[-1] < self.param[0]:
                return 0.0
            return -np.inf
                
                
                
        else:
            print("I can't find your model")
            exit()






    def lnprior2(self, param):
        '''
        The prior of Oba from CMB
        '''
        self.param = param
        
        self.Oba0 = self.param[-1]
        self.h_pr = self.param[-3]   # Hubble constant
        return -(self.Oba0*self.h_pr**2-0.02202)**2/(2*0.00046**2)




