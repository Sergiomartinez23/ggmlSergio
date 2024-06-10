#include "../../include/ggml/ggml.h"
#include <stdio.h>
#include "time.h"

void perform_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      //implemented and working
void perform_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      //implemented and working
void perform_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      //implemented and working
void perform_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                      //implemented and working   
void perform_vec_dot(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                  //implemented and working/not f16 check how to work with f16 data type 
void perform_relu(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                             //implemented and working                            
void perform_sqrt(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                             //implemented and working
void perform_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size);                 //implemented and working
void perform_abs(struct ggml_context * ctx, struct ggml_tensor * a, int size);                                              //implemented and working    


void main (int argc, char ** argv) {
    int function;
    int size = 128; 
    struct ggml_init_params params = {
           .mem_size   = 100*4096*4096,
           .mem_buffer = NULL,
    };
    

    printf("This the benchmark to check the correcteness of the library after aplying changes to avx512 SIMD instructions\n");
    printf("Results of the operation will be stored in a file so they can be compared and check if the results are correct\n");
    printf("The size of the matrix is %d\n", size);
    if (ggml_cpu_has_avx512()) {
        printf("The CPU has AVX512 instructions\n");
    } else {
        printf("The CPU doesn't have AVX512 instructions\n");
    }

    printf("Initializing the matrix\n");

    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * integer = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    *(float *)(integer->data) = 2.0;
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            *(float*)((char *)a->data + i*a->nb[0] + j*a->nb[1])= (float)i/size;
        }   
    }
    
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);
    
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            *(float*)((char *)b->data + i*b->nb[0] + j*b->nb[1])=(float)j/size;
        }
    }

    struct ggml_tensor * only_relu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, size);

    for (int j = 0; j < size; j++) {
        int f;
        for (int i = 0; i < size; i++) { 
            if (i % 2 == 0) {
                f = i;
            } else {
                f = -i;
            }
            *(float*)((char *)only_relu->data + i*only_relu->nb[0] + j*only_relu->nb[1])= (float)f;
        }
    }

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
    printf("Matrix initialized\n");
    printf("Introduce operation, to exit introduce -1\n");
    do
    {
        scanf("%d", &function);
        switch (function) {
        case 0:
            perform_add(ctx, a, b, size);
            break;
        case 1: 
            perform_mul(ctx, a, b, size);
            break;
        case 2:
            perform_div(ctx, a, b, size);
            break;
        case 3:
            perform_sub(ctx, a, b, size);
            break;
        case 4:
            perform_vec_dot(ctx, a, b, size);
            break;
        case 5:
            perform_sqrt(ctx, a, size);
            break;
        case 6:
            perform_relu(ctx, only_relu, size);
            break;
        case 7:
            perform_vec_dot(ctx, a_f16, b_f16, size);
            break;
        case 8: 
            perform_out_prod(ctx, a, b, size);
            break;
        case 9: 
            perform_abs(ctx, only_relu, size);
            break;
        default: 
            printf("The function %d is not implemented\n", function);
            break;
        }
    } while (function != -1);
    

}

void perform_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing add operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_add.txt";
    } else {
        filename = "serial_add.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * suma = ggml_add(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, suma);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)suma->data + i*suma->nb[0] + j*suma->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing mul operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_mul.txt";
    } else {
        filename = "serial_mul.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * mul = ggml_mul(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, mul);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)mul->data + i*mul->nb[0] + j*mul->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing div operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_div.txt";
    } else {
        filename = "serial_div.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * div = ggml_div(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, div);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)div->data + i*div->nb[0] + j*div->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing sub operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_sub.txt";
    } else {
        filename = "serial_sub.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * sub = ggml_sub(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, sub);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)sub->data + i*sub->nb[0] + j*sub->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_vec_dot(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing vec_dot operation\n");
    char * filename;
    if (a->type == GGML_TYPE_F16) {
        if (ggml_cpu_has_avx512()) {
            filename = "avx512_vec_dot_f16.txt";
        } else {
            filename = "serial_vec_dot_f16.txt";
        }
    } else {
        if (ggml_cpu_has_avx512()) {
            filename = "avx512_vec_dot.txt";
        } else {
            filename = "serial_vec_dot.txt";
        }
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * vec_dot = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, vec_dot);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);
    if (a->type == GGML_TYPE_F16) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) { 
                fprintf(file, "%f ", ggml_fp16_to_fp32(*(ggml_fp16_t*)((char *)vec_dot->data + i*vec_dot->nb[0] + j*vec_dot->nb[1])));
            }
            fprintf(file, "\n");
        }
    } else {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) { 
                fprintf(file, "%f ", *(float*)((char *)vec_dot->data + i*vec_dot->nb[0] + j*vec_dot->nb[1]));
            }
            fprintf(file, "\n");
        }
    }
    
    printf("Compute finished\n");
    fclose(file);
}

void perform_sqrt(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing sqrt operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_sqrt.txt";
    } else {
        filename = "serial_sqrt.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * sqrt = ggml_sqrt(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, sqrt);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)sqrt->data + i*sqrt->nb[0] + j*sqrt->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_relu(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing relu operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_relu.txt";
    } else {
        filename = "serial_relu.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * relu = ggml_relu(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, relu);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)relu->data + i*relu->nb[0] + j*relu->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

void perform_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int size) {
    printf("Performing out_prod operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_out_prod.txt";
    } else {
        filename = "serial_out_prod.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * out_prod = ggml_out_prod(ctx, a, b);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, out_prod);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)out_prod->data + i*out_prod->nb[0] + j*out_prod->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}


void perform_abs(struct ggml_context * ctx, struct ggml_tensor * a, int size) {
    printf("Performing abs operation\n");
    char * filename;
    if (ggml_cpu_has_avx512()) {
        filename = "avx512_abs.txt";
    } else {
        filename = "serial_abs.txt";
    }

    printf("The results will be stored in %s\n", filename);

    FILE *file = fopen(filename, "w");
    struct ggml_tensor * abs = ggml_abs(ctx, a);
    struct ggml_cgraph * grafo = ggml_new_graph(ctx);

    ggml_build_forward_expand(grafo, abs);

    printf("Starting compute\n");
    ggml_graph_compute_with_ctx(ctx, grafo, 1);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) { 
            fprintf(file, "%f ", *(float*)((char *)abs->data + i*abs->nb[0] + j*abs->nb[1]));
        }
        fprintf(file, "\n");
    }
    printf("Compute finished\n");
    fclose(file);
}

