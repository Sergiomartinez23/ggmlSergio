#include <stdio.h>
#include "time.h"
#include "immintrin.h"

#define GGML_F32x16_REDUCE(res, x)                                      \
do {                                                                    \
    for (int i = 1; i < GGML_F32_ARR; i++) {                            \
        x[0] = _mm512_add_ps(x[0], x[i]);                               \
    }                                                                   \
    res = _mm512_reduce_add_ps(x[0]);                                   \
} while (0)

static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y);
static void ggml_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y);
void main (int argc, char ** argv) {
    int function;
    int size = 128; 
    int a[size][size];
    int b[size][size];
    int c[size][size];
    


  
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            a[i][j] = i;
        }   
    }
    
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            b[i][j] = j;
        }
    }

    
    ggml_vec_add_f32(size * size, (float *) c, (float *) a, (float *) b);
    
    
    ggml_vec_dot_f32(size * size, (float *) c, (float *) a, (float *) b);
}



static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { 

    const int np = (n & ~(64 - 1));
    __m512 ax[4];
    __m512 ay[4];
    __m512 sum[4];
    
    for (int i = 0; i < np; i += 64) {
        for (int j = 0; j < 4; j ++) {
            ax[j] = _mm512_loadu_ps(x + i + j * 16 );
            ay[j] = _mm512_loadu_ps(y + i + j * 16);
            sum[j] = _mm512_add_ps(ax[j], ay[j]);
            _mm512_storeu_ps(z + i + j * 16, sum[j]);
        }
    }
    //leftovers
    for (int i = np; i < n; i++) {
        z[i]  = x[i] + y[i];
    }

}

static void ggml_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y) {
    float sumf = 0.0f;
    const int np = (n & ~(64 - 1));
    __m512 sum[4] = { _mm512_setzero_ps() };

    __m512 ax[4];
    __m512 ay[4];

    for (int i = 0; i < np; i += 64) {
        for (int j = 0; j < 4; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j*16);
            ay[j] = _mm512_loadu_ps(y + i + j*16);
            sum[j] = _mm512_fmadd_ps( ax[j], ay[j], sum[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }

    *s = sumf;
}