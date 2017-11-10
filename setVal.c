#include <stdio.h>
#include <stdlib.h>

void setVal(double * input, double* output, int num) {
    int p = 0;


    //for (i = 0; i < n; i++){
    //    for (j = 0; j < n; j++) {
    //        printf("a[%d]=%lli,b[%d]=%d,c[%d]=%d\n",k, a[k], k, b[k], k, c[k]);
    //        k++;
    //    }
    //}
    for (p = 0; p < num; p++) {
        output[p] = input[p];
    }
    return ;
}
