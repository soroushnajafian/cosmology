#include "HDE.c"
#include <stdio.h>
#include <math.h>

int main()
{   double z = 1090.0;
    double O_D = 0.7;
    double c = 0.6;
    double beta = 0.01;
    double gamma = 0.01;
    
    double *a1, *a2;
    
    int jc = Odee(z, O_D, c, beta, gamma, &a1, &a2);
    printf("%f  %f\n", a1[10], a1[11]);
    printf("%f  %f\n", a1[1000], a1[1001]);
    printf("%f  %f\n", a1[10000], a1[10001]);
    printf("%d\n", jc);
    
    //func(1.0, &a1);
    //printf("%f\n", a1[2]);
  
}
