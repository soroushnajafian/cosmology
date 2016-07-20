#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double Op_z(double z, double O_D, double Om0, double n, double Ok0, double Or0)
{
    double Gz = (Or0*(1.0+z)*(1.0+z)-Ok0)/(Om0*(1.0+z)+Or0*(1.0+z)*(1.0+z)+Ok0);
    return (  -O_D*(1.0-O_D)/(1.0+z)*(3.0+Gz-2.0*(1.0+z)/n*pow(O_D, 0.5))   );
}

int Odee(double z_f, double O_D0, double Om0, double n, double Ok0, double Or0, double **zs, double **ODs, double **Es)
{
    const double z0 = 0.0;
    double dz = 0.0001;
    double z = 0.0;
    double O_D = O_D0;
    double E = 1.0;
    
    double k1, k2, k3, k4;
    
    int i = floor((z_f-z0)/dz);
    int j = 0;
    
    double *zi;
    zi = (double *)malloc(sizeof(double)*i);
    double *ODi;
    ODi = (double *)malloc(sizeof(double)*i);
    double *Ei;
    Ei = (double *)malloc(sizeof(double)*i);
    
    while(z<z_f)
    {
        if (z>=1.0 && z<3.0)
        {
            dz = 0.001;
        }
        else if (z>=3.0 && z<10.0)
        {
            dz = 0.01;
        }
        else if (z>=10.0)
        {
            dz = 0.5;
        }
        
        k1 = Op_z(z, O_D, Om0, n, Ok0, Or0);
        k2 = Op_z(z+dz*0.5, O_D+0.5*dz*k1, Om0, n, Ok0, Or0);
        k3 = Op_z(z+dz*0.5, O_D+0.5*dz*k2, Om0, n, Ok0, Or0);
        k4 = Op_z(z+dz, O_D+dz*k3, Om0, n, Ok0, Or0);
    
        z +=dz;
        O_D += (k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)*dz;
        if (O_D == 1.0)
        {
            O_D = 1.0-1.0e-15;
        }
        E = pow((Om0*pow(1.0+z, 3.0) + Or0*pow(1.0+z, 4.0) + Ok0*pow(1.0+z, 2.0))/(1.0-O_D), 0.5);
        
        zi[j] = z;
        ODi[j] = O_D;
        Ei[j] = E;
        
        j++;
        
    }
    printf("%d\n", j);
    *zs = zi;
    *ODs = ODi;
    *Es = Ei;
    return j-1;
}
















