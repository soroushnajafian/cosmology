#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double dota(double a, double phi, double dotphi, double kmp, double alpha)
{
    return (sqrt(4.0/9.0/(a) + (a)*(a)/12.0*(dotphi*dotphi + kmp*exp(-alpha*phi))));
}

double ddotphi(double a, double phi, double dotphi, double kmp, double alpha)
{
    return (-3.0*sqrt(4.0/9.0/a/a/a +
                      1.0/12.0*(dotphi*dotphi + kmp*exp(-alpha*phi)))*
            dotphi + kmp*alpha/2.0*exp(-alpha*phi));
}


int E1(double alpha, double OmegaM)
{
    const double t0 = 0.001;
    const double dt = 0.0001;
    
    double t1 = 1.0;
    double C = 1.0;
    double t = t0;
    double kmp = 12.0*C/alpha/alpha;
    
    double a_now, da_now, phi_now, dotphi_now;
    
    double a = pow(t0 , 2.0/3.0);
    double phi = 1.0/alpha*log(C*(t0*t0+t1*t1));
    double dotphi = 2.0*t0/alpha/(t0*t0+t1*t1);
    double OM = 4.0/9.0/a/a/a/(4.0/9.0/a/a/a + (dotphi*dotphi + kmp*exp(-alpha*phi))/12.0);
    
    double k11, k12, k13, k21, k22, k23, k31, k32, k33, k41, k42, k43;
    
    int i = 0;
    

    while(OM>OmegaM)
    {
        k11 = dota(a, phi, dotphi, kmp, alpha)*dt;
        k12 = dotphi*dt;
        k13 = ddotphi(a, phi, dotphi, kmp, alpha)*dt;
        
        k21 = dota(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        k22 = (dotphi + 0.5*k12)*dt;
        k23 = ddotphi(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        
        k31 = dota(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        k32 = (dotphi + 0.5*k22)*dt;
        k33 = ddotphi(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        
        k41 = dota(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        k42 = (dotphi + k32)*dt;
        k43 = ddotphi(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        
        a +=( k11/6.0 + k21/3.0 + k31/3.0 + k41/6.0);
        phi += (k12/6.0 + k22/3.0 + k32/3.0 + k42/6.0);
        dotphi += (k13/6.0 + k23/3.0 + k33/3.0 + k43/6.0);
        
        //t = t + dt;
        
        OM = 4.0/9.0/a/a/a/(4.0/9.0/a/a/a + (dotphi*dotphi + kmp*exp(-alpha*phi))/12.0);
        i++;
        
    }
   
    return i;

}

double* E2(double alpha, double OmegaM, int len)
{
    const double t0 = 0.001;
    const double dt = 0.0001;
    
    double t1 = 1.0;
    double C = 1.0;
    double t = t0;
    double kmp = 12.0*C/alpha/alpha;
    
    double a_now, da_now, phi_now, dotphi_now;
    
    double a = pow(t0 , 2.0/3.0);
    double phi = 1.0/alpha*log(C*(t0*t0+t1*t1));
    double dotphi = 2.0*t0/alpha/(t0*t0+t1*t1);
    double OM = 4.0/9.0/a/a/a/(4.0/9.0/a/a/a + (dotphi*dotphi + kmp*exp(-alpha*phi))/12.0);
    
    double k11, k12, k13, k21, k22, k23, k31, k32, k33, k41, k42, k43;
    
    int i = 0;
    
    
    while(OM>OmegaM)
    {
        k11 = dota(a, phi, dotphi, kmp, alpha)*dt;
        k12 = dotphi*dt;
        k13 = ddotphi(a, phi, dotphi, kmp, alpha)*dt;
        
        k21 = dota(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        k22 = (dotphi + 0.5*k12)*dt;
        k23 = ddotphi(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        
        k31 = dota(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        k32 = (dotphi + 0.5*k22)*dt;
        k33 = ddotphi(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        
        k41 = dota(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        k42 = (dotphi + k32)*dt;
        k43 = ddotphi(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        
        a +=( k11/6.0 + k21/3.0 + k31/3.0 + k41/6.0);
        phi += (k12/6.0 + k22/3.0 + k32/3.0 + k42/6.0);
        dotphi += (k13/6.0 + k23/3.0 + k33/3.0 + k43/6.0);
        
        //t = t + dt;
        
        OM = 4.0/9.0/a/a/a/(4.0/9.0/a/a/a + (dotphi*dotphi + kmp*exp(-alpha*phi))/12.0);
        i++;
        
    }
    //printf("%d\n", i);
    
    a_now = a;
    da_now = dota(a, phi, dotphi, kmp, alpha);
    phi_now = phi;
    dotphi_now = dotphi;
    
    double *zs;
    zs = (double *)malloc(sizeof(double)*2*len);
    
    int j = 0;
    while ((a > pow(t0 , 2.0/3.0)) && j<i)
    
    {
        k11 = dota(a, phi, dotphi, kmp, alpha)*dt;
        k12 = dotphi*dt;
        k13 = ddotphi(a, phi, dotphi, kmp, alpha)*dt;
        
        k21 = dota(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        k22 = (dotphi + 0.5*k12)*dt;
        k23 = ddotphi(a + 0.5*k11, phi + 0.5*k12, dotphi + 0.5*k13, kmp, alpha)*dt;
        
        k31 = dota(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        k32 = (dotphi + 0.5*k22)*dt;
        k33 = ddotphi(a + 0.5*k21, phi + 0.5*k22, dotphi + 0.5*k23, kmp, alpha)*dt;
        
        k41 = dota(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        k42 = (dotphi + k32)*dt;
        k43 = ddotphi(a + k31, phi + k32, dotphi + k33, kmp, alpha)*dt;
        
        a -= (k11/6.0 + k21/3.0 + k31/3.0 + k41/6.0);
        phi -= (k12/6.0 + k22/3.0 + k32/3.0 + k42/6.0);
        dotphi -= (k13/6.0 + k23/3.0 + k33/3.0 + k43/6.0);
        
        zs[2*j] = a_now/a-1.0;
        zs[2*j+1] = (dota(a, phi, dotphi, kmp, alpha)/a)/(da_now/a_now);
        //zs[0][j] = a_now/a-1.0;
        //zs[1][j] = (dota(a, phi, dotphi, kmp, alpha)/a)/(da_now/a_now);
        j++;
        
    }
    return zs;
    
}

void free_mem(double *a)
{
    free(a);
}















