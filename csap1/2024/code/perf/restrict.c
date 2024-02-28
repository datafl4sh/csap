#include <stdio.h>

void sum2(int * __restrict c, const int * __restrict a, const int * __restrict b)
{
    for (int i = 0; i < 4; i++)
        c[i] = a[i] + b[i];
}

int main(void) {

    int v[8];
    for (int i = 0; i < 8; i++)
        v[i] = i+1;

    int *a = &v[1];
    int *b = &v[0];
    int *c = &v[2]; 


    for (int i = 0; i < 4; i++)
        c[i] = a[i] + b[i];

    for (int i = 0; i < 4; i++)
        printf("%d ", c[i]);
    printf("\n");
}

