#include <stdio.h>
#include <stdlib.h>

#include "cblas.h"

void partXY(double * u, double* v, int * rs, int* cs, double* output, int num, int r) {
    int p = 0;
    int k = 0;

    //for (i = 0; i < n; i++){
    //    for (j = 0; j < n; j++) {
    //        printf("a[%d]=%lli,b[%d]=%d,c[%d]=%d\n",k, a[k], k, b[k], k, c[k]);
    //        k++;
    //    }
    //}
    for (p = 0; p < num; p++) {
        double sub = 0.0;
        output[p] = cblas_ddot(r, u+rs[p]*r, 1, v+cs[p]*r, 1);
        //for (k = 0; k < r; k++) {
        //    sub = sub + u[rs[p] * r + k] * v[cs[p] * r + k];
        //}
        //output[p] = sub;
    }
    return ;
}
