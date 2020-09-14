/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char *dgemm_desc = "Simple blocked dgemm.";
// for sse instructions, the machine supports sse4, which means 4 doubles in a register
// so we will use _mm256 for operations
#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <string.h>

#define GEMM_STRIDE 144 // how large the block will be when partitioning matrix in GEMM
#define GEPP_STRIDE 12  // how large the block will be when partitioning matrix in GEPP
#define GEBP_WIDTH 4    // how large the block of C will be in GEBP
#define GEBP_HEIGHT 12  // how large the block of C will be in GEBP

int GEMM_BORDER = 0;  // how large is the edge case in GEMM
int GEPP_BORDER = 0;  // how large is the edge case in GEPP
int FIXED_WIDTH = 0;  // make lda a multiple of GEBP_WIDTH
int FIXED_HEIGHT = 0; // make lda a multiple of GEBP_HSIGHT

static double A_PACKED[GEMM_STRIDE * GEPP_STRIDE] __attribute__((aligned(32))); // store the packed A elements
static double B_PACKED[1092 * GEMM_STRIDE] __attribute__((aligned(32)));        // we do not know exactly how large this will be so set an upper limit
static double C_PACKED[GEBP_WIDTH * GEBP_HEIGHT] __attribute__((aligned(32)));  // the packed C

// #define min(a, b) (((a) < (b)) ? (a) : (b))

// // print a normal matrix
// void print_matrix(double *matrix, int width, int height, int lda)
// {
//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             printf("%f ", matrix[i + j * lda]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

// void print_B_packed(double *B, int width, int height, int pack_width, int pack_height)
// {
//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             int index_j = (j / GEBP_WIDTH) * (GEBP_WIDTH * pack_height) + j % GEBP_WIDTH;

//             int index = index_j + i * GEBP_WIDTH;
//             printf("%f ", B[index]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

// void print_A_packed(double *A, int width, int height, int pack_width, int pack_height)
// {
//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             int index_j = j * GEBP_HEIGHT;

//             int index = index_j + (i / GEBP_HEIGHT) * (GEBP_HEIGHT * pack_width) + i % GEBP_HEIGHT;
//             printf("%f ", A[index]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

// void print_matrix_sequential(double *matrix, int width, int height, int lda)
// {
//     for (int j = 0; j < width; j++)
//     {
//         for (int i = 0; i < height; i++)
//         {
//             printf("%f ", matrix[i + j * lda]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

static void PACK_B(double *B, int width, int height, int pack_width, int pack_height, int lda)
{
    // setting to 0 and basically brining the pages
    memset(B_PACKED, 0, pack_height * FIXED_WIDTH * 8);
    for (int j = 0; j < width; j++)
    {
        int index_j = (j / GEBP_WIDTH) * (GEBP_WIDTH * pack_height) + j % GEBP_WIDTH;
        for (int i = 0; i < height; i++)
        {
            int index = index_j + i * GEBP_WIDTH;
            B_PACKED[index] = B[j * lda + i];
        }
    }
}

static void PACK_A(double *A, int width, int height, int pack_width, int pack_height, int lda)
{
    memset(A_PACKED, 0, pack_height * pack_width * 8);
    for (int j = 0; j < width; j++)
    {
        int index_j = j * GEBP_HEIGHT;
        for (int i = 0; i < height; i++)
        {
            int index = index_j + (i / GEBP_HEIGHT) * (GEBP_HEIGHT * pack_width) + i % GEBP_HEIGHT;
            A_PACKED[index] = A[j * lda + i];
        }
    }
}

static void UNPACK_C(double *C, int width, int height, int lda)
{
    for (int j = 0; j < width; j++)
    {
        for (int i = 0; i < height; i++)
        {
            C[j * lda + i] += C_PACKED[j * GEBP_HEIGHT + i];
        }
    }
}

// the smallest computing units
// computes 12 by 4
// A is packed: GEBP_HEIGHT * GEMM_STRIDE
// B is packed: GEMM_STRIDE * GEBP_WIDTH
// C is packed: GEBP_HEIGHT * GEBP_WIDTH
void INNER_BLOCK_MULT(double *A_ordered, double *B_ordered, double *C_ordered)
{
    __m256d c00 = _mm256_load_pd(C_ordered);
    __m256d c10 = _mm256_load_pd(C_ordered + 4);
    __m256d c20 = _mm256_load_pd(C_ordered + 8);
    __m256d c01 = _mm256_load_pd(C_ordered + 12);
    __m256d c11 = _mm256_load_pd(C_ordered + 16);
    __m256d c21 = _mm256_load_pd(C_ordered + 20);
    __m256d c02 = _mm256_load_pd(C_ordered + 24);
    __m256d c12 = _mm256_load_pd(C_ordered + 28);
    __m256d c22 = _mm256_load_pd(C_ordered + 32);
    __m256d c03 = _mm256_load_pd(C_ordered + 36);
    __m256d c13 = _mm256_load_pd(C_ordered + 40);
    __m256d c23 = _mm256_load_pd(C_ordered + 44);
    __m256d a0;
    __m256d a1;
    __m256d a2;
    __m256d b;

    for (int k = 0; k < GEMM_STRIDE; k++)
    {
        a0 = _mm256_load_pd(A_ordered + k * 12);
        a1 = _mm256_load_pd(A_ordered + k * 12 + 4);
        a2 = _mm256_load_pd(A_ordered + k * 12 + 8);

        b = _mm256_broadcast_sd(B_ordered + k * 4);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c10 = _mm256_fmadd_pd(a1, b, c10);
        c20 = _mm256_fmadd_pd(a2, b, c20);

        b = _mm256_broadcast_sd(B_ordered + k * 4 + 1);
        c01 = _mm256_fmadd_pd(a0, b, c01);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        c21 = _mm256_fmadd_pd(a2, b, c21);

        b = _mm256_broadcast_sd(B_ordered + k * 4 + 2);
        c02 = _mm256_fmadd_pd(a0, b, c02);
        c12 = _mm256_fmadd_pd(a1, b, c12);
        c22 = _mm256_fmadd_pd(a2, b, c22);

        b = _mm256_broadcast_sd(B_ordered + k * 4 + 3);
        c03 = _mm256_fmadd_pd(a0, b, c03);
        c13 = _mm256_fmadd_pd(a1, b, c13);
        c23 = _mm256_fmadd_pd(a2, b, c23);
    }

    _mm256_store_pd(C_ordered, c00);
    _mm256_store_pd(C_ordered + 4, c10);
    _mm256_store_pd(C_ordered + 8, c20);
    _mm256_store_pd(C_ordered + 12, c01);
    _mm256_store_pd(C_ordered + 16, c11);
    _mm256_store_pd(C_ordered + 20, c21);
    _mm256_store_pd(C_ordered + 24, c02);
    _mm256_store_pd(C_ordered + 28, c12);
    _mm256_store_pd(C_ordered + 32, c22);
    _mm256_store_pd(C_ordered + 36, c03);
    _mm256_store_pd(C_ordered + 40, c13);
    _mm256_store_pd(C_ordered + 44, c23);
}

// the inner loop funciton call in GEBP
// A is packed: GEPP_STRIDE * GEMM_STRIDE, real: gepp_stide_size * gemm_stride_size
// B is packed: GEMM_STRIDE * GEBP_WIDTH real: gemm_stride_size * gebp_stride_size
// C is not packed, gepp_stride_size * gebp_stride_size
static inline void GEBP_INNER(double *A, double *B, double *C, int gepp_stride_size, int gebp_stride_size, int lda)
{
    int leftover = gepp_stride_size % GEBP_HEIGHT;
    for (int i = 0; i < gepp_stride_size - leftover; i += GEBP_HEIGHT)
    {
        // setting up C
        memset(C_PACKED, 0, GEBP_WIDTH * GEBP_HEIGHT * 8);
        INNER_BLOCK_MULT(A + i * GEMM_STRIDE, B, C_PACKED);
        UNPACK_C(C + i, gebp_stride_size, GEBP_HEIGHT, lda);
    }

    if (leftover != 0)
    {
        memset(C_PACKED, 0, GEBP_WIDTH * GEBP_HEIGHT * 8);
        INNER_BLOCK_MULT(A + (gepp_stride_size - leftover) * GEMM_STRIDE, B, C_PACKED);
        UNPACK_C(C + (gepp_stride_size - leftover), gebp_stride_size, leftover, lda);
    }
}

// compute GEBP
// A is packed: GEPP_STRIDE * GEMM_STRIDE, real: gepp_stide_size * gemm_stride_size
// B is packed: GEMM_STRIDE * FIXED_WIDTH, real: gemm_stride_size * lda
// C is not packed, gepp_stride_size * lda
static inline void GEBP(double *A, double *B, double *C, int gepp_stride_size, int lda)
{
    int leftover = lda % GEBP_WIDTH;
    for (int j = 0; j < lda - leftover; j += GEBP_WIDTH)
    {
        GEBP_INNER(A, B + j * GEMM_STRIDE, C + j * lda, gepp_stride_size, GEBP_WIDTH, lda);
    }

    if (leftover != 0)
    {
        GEBP_INNER(A, B + (lda - leftover) * GEMM_STRIDE, C + (lda - leftover) * lda, gepp_stride_size, leftover, lda);
    }
}

// compute GEPP using GEBP
// A is not packed, lda * gemm_stride_size
// B is packed: GEMM_STRIDE * FIXED_WIDTH, real: gemm_stride_size * lda
// C is not packed
static inline void GEPP(double *A, double *B, double *C, int gemm_stride_size, int lda)
{
    for (int i = 0; i < lda - GEPP_BORDER; i += GEPP_STRIDE)
    {
        PACK_A(A + i, gemm_stride_size, GEPP_STRIDE, GEMM_STRIDE, GEPP_STRIDE, lda);
        // printf("A packed:\n");
        // print_A_packed(A_PACKED, GEMM_STRIDE, GEPP_STRIDE, GEMM_STRIDE, GEPP_STRIDE);
        // printf("A packed sequential:\n");
        // print_matrix_sequential(A_PACKED, GEMM_STRIDE, GEPP_STRIDE, GEPP_STRIDE);
        // printf("A original:\n");
        // print_matrix(A + i, gemm_stride_size, GEPP_STRIDE, lda);
        GEBP(A_PACKED, B, C + i, GEPP_STRIDE, lda);
    }

    if (GEPP_BORDER != 0)
    {
        PACK_A(A + (lda - GEPP_BORDER), gemm_stride_size, GEPP_BORDER, GEMM_STRIDE, GEPP_STRIDE, lda);
        // printf("A packed edge:\n");
        // print_A_packed(A_PACKED, GEMM_STRIDE, GEPP_STRIDE, GEMM_STRIDE, GEPP_STRIDE);
        // printf("A packed sequential edge:\n");
        // print_matrix_sequential(A_PACKED, GEMM_STRIDE, GEPP_STRIDE, GEPP_STRIDE);
        // printf("A original edge:\n");
        // print_matrix(A + (lda - GEPP_BORDER), gemm_stride_size, GEPP_BORDER, lda);
        GEBP(A_PACKED, B, C + (lda - GEPP_BORDER), GEPP_BORDER, lda);
    }
}

// compute GEMM using GEPP
// A and B and C are not packed
static inline void GEMM(double *A, double *B, double *C, int lda)
{
    for (int i = 0; i < lda - GEMM_BORDER; i += GEMM_STRIDE)
    {
        PACK_B(B + i, lda, GEMM_STRIDE, FIXED_WIDTH, GEMM_STRIDE, lda);
        // printf("B packed:\n");
        // print_B_packed(B_PACKED, FIXED_WIDTH, GEMM_STRIDE, FIXED_WIDTH, GEMM_STRIDE);
        // printf("B packed sequential:\n");
        // print_matrix_sequential(B_PACKED, FIXED_WIDTH, GEMM_STRIDE, GEMM_STRIDE);
        // printf("B original:\n");
        // print_matrix(B + i, lda, GEMM_STRIDE, lda);
        GEPP(A + i * lda, B_PACKED, C, GEMM_STRIDE, lda);
    }

    if (GEMM_BORDER != 0)
    {
        PACK_B(B + (lda - GEMM_BORDER), lda, GEMM_BORDER, FIXED_WIDTH, GEMM_STRIDE, lda);
        // printf("B packed edge:\n");
        // print_B_packed(B_PACKED, FIXED_WIDTH, GEMM_STRIDE, FIXED_WIDTH, GEMM_STRIDE);
        // printf("B packed sequential edge:\n");
        // print_matrix_sequential(B_PACKED, FIXED_WIDTH, GEMM_STRIDE, GEMM_STRIDE);
        // printf("B original edge:\n");
        // print_matrix(B + (lda - GEMM_BORDER), lda, GEMM_BORDER, lda);
        GEPP(A + (lda * (lda - GEMM_BORDER)), B_PACKED, C, GEMM_BORDER, lda);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
    GEMM_BORDER = lda % GEMM_STRIDE; // get the border case
    GEPP_BORDER = lda % GEPP_STRIDE; // get the border case
    FIXED_WIDTH = ((lda + GEBP_WIDTH - 1) / GEBP_WIDTH) * GEBP_WIDTH;
    FIXED_HEIGHT = ((lda + GEBP_HEIGHT - 1) / GEBP_HEIGHT) * GEBP_HEIGHT;
    GEMM(A, B, C, lda);
}
