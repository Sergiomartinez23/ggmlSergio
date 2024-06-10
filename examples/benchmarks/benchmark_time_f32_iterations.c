#include "../../include/ggml/ggml.h"
#include <stdio.h>
#include "time.h"
#include <stdlib.h>
#include "../../../papi/src/install/include/papi.h"


void perform_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      
void perform_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      
void perform_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                 
void perform_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                  
void perform_vec_dot(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);               
void perform_relu(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                                   
void perform_sqrt(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                          
void perform_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                
void perform_abs(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                              
#define iterations 10

void main (int argc, char ** argv) {
    //Initializations
    int function;
    int size = atoi(argv[1]); 
    struct ggml_init_params params = {
           .mem_size   = 100*4096*4096,
           .mem_buffer = NULL,
    };

    srand(clock());

    int retval;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    printf("PAPI library initialized");
    if (retval != PAPI_VER_CURRENT) {
        printf("Error initializing PAPI library %s\n", PAPI_strerror(retval));
    }
    printf("This the benchmark to get times of ggml op\n");
    printf("Each operation will be performed 10x times\n");
    printf("The size of the matrix is %d\n", size);
    printf("The matrix will store f32 numbers\n");
    if (ggml_cpu_has_avx512()) {
        printf("The CPU has AVX512 instructions, and will be used in this execution\n");
    } else {
        if (ggml_cpu_has_avx2()) {
            printf("The CPU has AVX2 instructions, and will be used in this execution\n");
        } else {
            printf("The CPU does not have SIMD instructions, the serial version will be used in this execution\n");
        }
    }

    printf("Initializing the matrix\n");
    printf("The matrix will be initialized with random values between 0 and 1\n");

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * integer = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    *(float *)(integer->data) = 2.0;
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    int f;
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            f = rand() % size;
            *(float*)((char *)a->data + i*a->nb[0] + j*a->nb[1])= (float)f/size;
        }   
    }
    
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            f = rand() % size;
            *(float*)((char *)b->data + i*b->nb[0] + j*b->nb[1])=(float)f/size;
        }
    }
    // Only for relu and abs, the values will be between -size /2 and size /2
    struct ggml_tensor * only_relu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            f = rand() % size;
            f = f - size/2;
            *(float*)((char *)only_relu->data + i*only_relu->nb[0] + j*only_relu->nb[1])= (float)f;
        }
    }
    // Only for f16 wont be using in this benchmark
    /** 
    struct ggml_tensor * a_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, size, size);
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            *(ggml_fp16_t*)((char *)a_f16->data + i*a_f16->nb[0] + j*a_f16->nb[1])=ggml_fp32_to_fp16((float)i);
        }
    }
    struct ggml_tensor * b_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, size, size);
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            *(ggml_fp16_t*)((char *)b_f16->data + i*b_f16->nb[0] + j*b_f16->nb[1])=ggml_fp32_to_fp16((float)j);
        }
    }
    */
    printf("Matrix initialized\n");
    printf("Starting operations\n");
    printf("\n--------------------------------\n");
    perform_add(ctx, a, b, size);
    /*
    printf("\n--------------------------------\n");
    perform_mul(ctx, a, b, size);
    printf("\n--------------------------------\n");
    perform_div(ctx, a, b, size);
    printf("\n--------------------------------\n");
    perform_sub(ctx, a, b, size);
    printf("\n--------------------------------\n");
    perform_vec_dot(ctx, a, b, size);
    printf("\n--------------------------------\n");
    perform_sqrt(ctx, a, size);
    printf("\n--------------------------------\n");
    perform_relu(ctx, only_relu, size);
    printf("\n--------------------------------\n");
    perform_out_prod(ctx, a, b, size);
    printf("\n--------------------------------\n");
    perform_abs(ctx, only_relu, size);
    printf("\n--------------------------------\n");
    */
    
}

void perform_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing add operation\n");
    int event_set = PAPI_NULL;
    int retval;
    int times[iterations];
    int medium_time = 0;
    long long data[4] = {0, 0, 0, 0};
    int medium_cycles = 0;
    int medium_instructions = 0;
    int medium_fp_512 = 0;
    int medium_fp = 0;
    int start, end;
    
    retval = PAPI_create_eventset(&event_set);
    if (retval != PAPI_OK) {
        printf("Error creating event set %s\n", PAPI_strerror(retval));
    }
    PAPI_add_event(event_set, PAPI_TOT_CYC);
    PAPI_add_event(event_set, PAPI_TOT_INS);
    PAPI_add_event(event_set, PAPI_SP_OPS);
    PAPI_add_event(event_set, PAPI_VEC_SP);

    struct ggml_tensor * suma = ggml_add(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, suma);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        if (i == 0) {
            PAPI_start(event_set);
        }
        start = PAPI_get_real_usec();
        PAPI_hl_region_begin("ggml_graph_compute_with_ctx");
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        PAPI_hl_region_end("ggml_graph_compute_with_ctx");
        end = PAPI_get_real_usec();
        times[i] = end - start;
        PAPI_read(event_set, data);
        medium_cycles += data[0];
        medium_instructions += data[1];
        medium_fp += data[2];
        medium_fp_512 += data[3];
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
        PAPI_reset(event_set);
    }
    PAPI_stop(event_set, data);
   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
    printf("Medium cycles: %d\n", medium_cycles/iterations);
    printf("Medium instructions: %d\n", medium_instructions/iterations);
    printf("Medium FP: %d\n", medium_fp/iterations);
    printf("Medium FP 512: %d\n", medium_fp_512/iterations);
}

void perform_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing mul operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;
    
    struct ggml_tensor * mul = ggml_mul(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, mul);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing div operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * div = ggml_div(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, div);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing sub operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * sub = ggml_sub(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, sub);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_vec_dot(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing vec_dot operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * vec_dot = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, vec_dot);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_sqrt(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing sqrt operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * sqrt = ggml_sqrt(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, sqrt);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_relu(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing relu operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * relu = ggml_relu(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, relu);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

void perform_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing out_prod operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * out_prod = ggml_out_prod(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, out_prod);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}


void perform_abs(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing abs operation\n");
    int event_set = PAPI_NULL;
    int times[iterations];
    int medium_time = 0;
    int start, end;

    struct ggml_tensor * abs = ggml_abs(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, abs);

    printf("Starting compute\n");
    for (int i = 0; i < iterations; i++) {
        printf ("Iteration %d\n", i);
        start = PAPI_get_real_usec();
        ggml_graph_compute_with_ctx(ctx, grafo, 1);
        end = PAPI_get_real_usec();
        times[i] = end - start;
        printf("Time: %d microseconds\n", times[i]);
        medium_time += times[i];
    }

   
    printf("Compute finished\n");
    printf("Medium time: %d microseconds\n", medium_time/iterations);
}

