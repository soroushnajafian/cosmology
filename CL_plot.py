import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import math

class Con_Reg:
    '''
    Plot the triangular diagram: 2D confidence regions and one-dimensional PDF.
    The input parameters: N: the dimension of the parameter space
                          chi2: the chi square chain from the MCMC process
                          chain: the MCMC chains of the parameters. The number of the chains should be the same as N
                          cl: confidence levels that needs to be specified in the contour plot
                          IFdot: to turn on or off the display of the dots in the parameter space
                          label: the names of the axises that will appear in the 2D plot.
    '''

    def __init__(self, N, chi2, chain, cl, IFdot, label):
        
        self.N = N
        self.chi2 = chi2
        self.chain = chain
        self.cl = cl
        self.IFdot = IFdot
        self.label = label
        self.dim = len(self.chain)
        
        print('Dimension of the parameter space and number of chains are {} and {} respectively.'.format( self.N, self.dim))
        ### make sure the inputs are consistent.
        if self.N != self.dim:
            
            print("The input dimension is different from the chains, please modify the value")
            exit()

        else:
            '''
            The following loop calculates the one-dimensional PDF
            '''
            for i in range(self.N):
                
                self.xx = self.chain[i]
                self.PDF = np.histogram(self.xx, 100)
                self.x = self.PDF[1][0:len(self.PDF[1])-1]+(self.PDF[1][2]-self.PDF[1][1])*0.5
                self.y1 = self.PDF[0]
                self.y2 = [float(ii) for ii in self.PDF[0]]
                self.y3 = np.array(self.y2)
                self.y = self.y3*pow(float(max(self.y3)), -1)

                plt.subplot(self.N, self.N, 1+i*(self.N+1))
                plt.plot(self.x, self.y)
                '''
                Add the labels to the axises. The highest PDF has a y-axis named as "PDF"
                '''
                if i == 0:
                    plt.ylabel('PDF')
                if i == self.N-1:
                    plt.xlabel(self.label[i])

            '''
            The following loop calculates the 2D confidence regions
            '''
            del self.xx, self.x, self.y
            
            self.vv = self.cl + min(self.chi2)   # the confidence levels
            for j in range(self.N-1):
                
                self.x = self.chain[j]
                
                for k in range(j+1, self.N):
                    
                    self.y = self.chain[k]
                    
                    self.xi, self.yi = np.linspace(self.x.min(), self.x.max(), 50), np.linspace(self.y.min(), self.y.max(), 50)
                    self.xii, self.yii = self.xi, self.yi
                    self.xi, self.yi = np.meshgrid(self.xi, self.yi)
                    self.zi = scipy.interpolate.griddata((self.x, self.y), self.chi2, (self.xi, self.yi), method='linear')
                    
                    
                    ## more options should be added, e.g.: fill in the contours,
                    
                    plt.subplot(self.N, self.N, self.N*k+(j+1))
                    plt.contour(self.xi, self.yi, self.zi, self.vv)
                    if self.IFdot:
                        plt.scatter(self.x, self.y, s = 0.01)

                    '''
                    Add the labels to the axises.
                    '''
                    if j == 0:
                        plt.ylabel(self.label[k])

            
                    if k == self.N-1:
                        plt.xlabel(self.label[j])
    
        
        

        plt.show()
