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

// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// Smaller case
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
#define ORDERED_BLOCK_SIZE 256                                              // this is the middle block size
#define ORDERED_BLOCK_SIZE_SQUARE (ORDERED_BLOCK_SIZE * ORDERED_BLOCK_SIZE) // how many elements inside each middle block
#define INNER_BLOCK_SIZE 4                                                  // we will do a inner block of 4 by 4, and will not change this
#define INNER_BLOCK_SIZE_SQUARE (INNER_BLOCK_SIZE * INNER_BLOCK_SIZE)       // the length of the inner block

#define BLOCK_HEIGHT 64                                                  // the height of block C when we perform GEPM, this will also be the height of A's block
#define BLOCK_WIDTH 128                                                  // the width of block A when we perform GEBP, will also be the height of block B
#define SMALLER_BLOCK_WIDTH 8                                           // this is the width of the block when performing GEBP
#define BLOCK_HEIGHT_DOUBLE (BLOCK_HEIGHT * 2)                           // redundent computation
#define BLOCK_HEIGHT_TRIPLE (BLOCK_HEIGHT * 3)                           // redundent computation
#define BLOCK_WIDTH_DOUBLE (BLOCK_WIDTH * 2)                             // redundent computation
#define BLOCK_WIDTH_TRIPLE (BLOCK_WIDTH * 3)                             // redundent computation
#define SMALLER_BLOCK_BYTE_SIZE (BLOCK_HEIGHT * SMALLER_BLOCK_WIDTH * 8) // redundent computation

int FIXED_WIDTH_SMALL = 0;

int edge_row = 0;
int edge_col = 0;
int smaller_edge_col = 0;
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================

// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// Larger case
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
#define GEMM_STRIDE 144 // how large the block will be when partitioning matrix in GEMM
#define GEPP_STRIDE 12  // how large the block will be when partitioning matrix in GEPP
#define GEBP_WIDTH 4    // how large the block of C will be in GEBP
#define GEBP_HEIGHT 12  // how large the block of C will be in GEBP

int GEMM_BORDER = 0;  // how large is the edge case in GEMM
int GEPP_BORDER = 0;  // how large is the edge case in GEPP
int FIXED_WIDTH = 0;  // make lda a multiple of GEBP_WIDTH
int FIXED_HEIGHT = 0; // make lda a multiple of GEBP_HSIGHT

// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================

// static double A_PACKED[GEMM_STRIDE * GEPP_STRIDE] __attribute__((aligned(32))); // store the packed A elements
static double B_PACKED[1092 * GEMM_STRIDE] __attribute__((aligned(32))); // we do not know exactly how large this will be so set an upper limit
// static double C_PACKED[GEBP_WIDTH * GEBP_HEIGHT] __attribute__((aligned(32)));  // the packed C

static double A_PACKED[8192] __attribute__((aligned(32))); // store the block of A that is packed
static double C_PACKED[1024] __attribute__((aligned(32))); // store the block of C that is packed

// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================

// pack a matrix
// user need to input what the available height and width is from the original matrix
// this is to avoid the edge cases
// then, user also need to specify what is the dimension of the packed matrix
// the lda is needed to know the gap for the original matrix
static inline void PACKING(double *packed_matrix, double *matrix, int width, int height, int pack_width, int pack_height, int lda)
{
    int height_gap = (pack_height - height) % SMALLER_BLOCK_WIDTH;
    // int width_gap = pack_width - width;
    for (int j = 0; j < width; j++)
    {
        memcpy(packed_matrix + j * pack_height, matrix + j * lda, height * 8); // copy the column
        memset(packed_matrix + j * pack_height + height, 0, height_gap * 8);   // set the borders to 0
    }
    // set the edges to 0
    memset(packed_matrix + width * pack_height, 0, ((pack_width - width) % SMALLER_BLOCK_WIDTH) * pack_height * 8);
}

// unpack a matrix
// this probably will just be used upon C
// we will need to add the value back when unpacking
__attribute__((optimize("unroll-loops"))) static inline void UNPACKING(double *packed_matrix, double *matrix, int width, int height, int pack_height, int lda)
{
    for (int j = 0; j < width; j++)
    {
        int j_lda = j * lda;
        int j_pack_height = j * pack_height;
        for (int i = 0; i < height; i++)
        {
            double val1 = matrix[i + j_lda];
            double val2 = packed_matrix[i + j_pack_height];
            matrix[i + j_lda] = val1 + val2;
        }
    }
}

static inline void GEBP_INNER_EXPANDED_SMALL(double *A, double *B, double *C, int width, int height)
{
    // // the goal is to have less access for A i think
    for (int k = 0; k < width; k += INNER_BLOCK_SIZE)
    {
        int k_BLOCK_HEIGHT = k * BLOCK_HEIGHT;
        for (int n = 0; n < SMALLER_BLOCK_WIDTH; n += INNER_BLOCK_SIZE)
        {
            int n_BLOCK_HEIGHT = n * BLOCK_HEIGHT;
            int n_BLOCK_WIDTH = n * BLOCK_WIDTH;

            // load elements in B, need k and n
            __m256d val0_0 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 0]);
            __m256d val0_1 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 1]);
            __m256d val0_2 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 2]);
            __m256d val0_3 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 3]);
            __m256d val1_0 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + BLOCK_WIDTH]);
            __m256d val1_1 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 1 + BLOCK_WIDTH]);
            __m256d val1_2 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 2 + BLOCK_WIDTH]);
            __m256d val1_3 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 3 + BLOCK_WIDTH]);
            __m256d val2_0 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + BLOCK_WIDTH_DOUBLE]);
            __m256d val2_1 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 1 + BLOCK_WIDTH_DOUBLE]);
            __m256d val2_2 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 2 + BLOCK_WIDTH_DOUBLE]);
            __m256d val2_3 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 3 + BLOCK_WIDTH_DOUBLE]);
            __m256d val3_0 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + BLOCK_WIDTH_TRIPLE]);
            __m256d val3_1 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 1 + BLOCK_WIDTH_TRIPLE]);
            __m256d val3_2 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 2 + BLOCK_WIDTH_TRIPLE]);
            __m256d val3_3 = _mm256_set1_pd(B[k + n_BLOCK_WIDTH + 3 + BLOCK_WIDTH_TRIPLE]);

            for (int m = 0; m < height; m += INNER_BLOCK_SIZE)
            {
                // load elements in A, need m and k
                __m256d col0 = _mm256_load_pd(A + m + k_BLOCK_HEIGHT);
                __m256d col1 = _mm256_load_pd(A + m + k_BLOCK_HEIGHT + BLOCK_HEIGHT);
                __m256d col2 = _mm256_load_pd(A + m + k_BLOCK_HEIGHT + BLOCK_HEIGHT_DOUBLE);
                __m256d col3 = _mm256_load_pd(A + m + k_BLOCK_HEIGHT + BLOCK_HEIGHT_TRIPLE);

                // load elements in C, nned m and n
                __m256d col_result0 = _mm256_load_pd(C + m + n_BLOCK_HEIGHT);
                __m256d col_result1 = _mm256_load_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT);
                __m256d col_result2 = _mm256_load_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT_DOUBLE);
                __m256d col_result3 = _mm256_load_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT_TRIPLE);

                // do computation
                col_result0 = _mm256_fmadd_pd(col3, val0_3, _mm256_fmadd_pd(col2, val0_2, _mm256_fmadd_pd(col1, val0_1, _mm256_fmadd_pd(col0, val0_0, col_result0))));
                col_result1 = _mm256_fmadd_pd(col3, val1_3, _mm256_fmadd_pd(col2, val1_2, _mm256_fmadd_pd(col1, val1_1, _mm256_fmadd_pd(col0, val1_0, col_result1))));
                col_result2 = _mm256_fmadd_pd(col3, val2_3, _mm256_fmadd_pd(col2, val2_2, _mm256_fmadd_pd(col1, val2_1, _mm256_fmadd_pd(col0, val2_0, col_result2))));
                col_result3 = _mm256_fmadd_pd(col3, val3_3, _mm256_fmadd_pd(col2, val3_2, _mm256_fmadd_pd(col1, val3_1, _mm256_fmadd_pd(col0, val3_0, col_result3))));

                // load back, need m and n
                _mm256_store_pd(C + m + n_BLOCK_HEIGHT, col_result0);
                _mm256_store_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT, col_result1);
                _mm256_store_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT_DOUBLE, col_result2);
                _mm256_store_pd(C + m + n_BLOCK_HEIGHT + BLOCK_HEIGHT_TRIPLE, col_result3);
            }
        }
    }
}

// compute GEBP
// here, A is not packed, B is packed, C is not packed
// we will need to know the height available for A to do packing
// which is also the height when unpacking C
// then also the width available for A to do packing
// The dimensions of the inputs are :
// A after packing: BLOCK_HEIGHT * BLOCK_WIDTH
// B: BLOCK_WIDTH * FIXED_WIDTH
// C: height * lda
static inline void GEBP_SMALL(double *A, double *B, double *C, int width, int height, int lda)
{
    // we first pack A into pack
    PACKING(&A_PACKED[0], A, width, height, BLOCK_WIDTH, BLOCK_HEIGHT, lda);
    // now A is packed, we go into the inner loop
    for (int i = 0; i < lda - smaller_edge_col; i += SMALLER_BLOCK_WIDTH)
    {
        // in this loop, this width should always be SMALLER_BLOCK_WIDTH
        // now pack C, which is setting the entire block to be 0
        // C_PACKED has dimension of BLOCK_HEIGHT * SMALLER_BLOCK_WIDTH
        memset(C_PACKED, 0, SMALLER_BLOCK_BYTE_SIZE);
        GEBP_INNER_EXPANDED_SMALL(A_PACKED, B + i * BLOCK_WIDTH, C_PACKED, width, height); // we need to move the pointer of B
        UNPACKING(C_PACKED, C + i * lda, SMALLER_BLOCK_WIDTH, height, BLOCK_HEIGHT, lda);
    }

    if (smaller_edge_col != 0)
    {
        // now do the edge case
        memset(C_PACKED, 0, SMALLER_BLOCK_BYTE_SIZE);
        GEBP_INNER_EXPANDED_SMALL(A_PACKED, B + (lda - smaller_edge_col) * BLOCK_WIDTH, C_PACKED, width, height);
        UNPACKING(C_PACKED, C + (lda - smaller_edge_col) * lda, smaller_edge_col, height, BLOCK_HEIGHT, lda);
    }
}

// compute GEPP using GEBP
// here, A, B, C are not packed
// the dimensions of the inputs are:
// A: lda * width
// B: BLOCK_WIDTH * FIXED_WIDTH
// C: lda * lda
static inline void GEPP_SMALL(double *A, double *B, double *C, int width, int lda)
{
    // printf("B_packed: \n");
    // print_matrix(B_PACKED, BLOCK_WIDTH, FIXED_WIDTH);
    for (int i = 0; i < lda - edge_row; i += BLOCK_HEIGHT)
    {
        GEBP_SMALL(A + i, B, C + i, width, BLOCK_HEIGHT, lda);
    }
    if (edge_row != 0)
    {
        GEBP_SMALL(A + (lda - edge_row), B, C + (lda - edge_row), width, edge_row, lda);
    }
}

// compute GEMM using GEPP
// here, A, B, C are not packed
// but B will be packed inside this function to pass into GEBP
__attribute__((optimize("unroll-loops"))) static inline void GEMM_SMALL(double *A, double *B, double *C, int lda)
{
    // we will need to divide A vertically, B horizontally
    edge_col = lda % BLOCK_WIDTH;
    edge_row = lda % BLOCK_HEIGHT;
    smaller_edge_col = lda % SMALLER_BLOCK_WIDTH;
    FIXED_WIDTH_SMALL = ((lda + SMALLER_BLOCK_WIDTH - 1) / SMALLER_BLOCK_WIDTH) * SMALLER_BLOCK_WIDTH;
    for (int i = 0; i < lda - edge_col; i += BLOCK_WIDTH)
    {
        // first do packing of B
        PACKING(B_PACKED, B + i, lda, BLOCK_WIDTH, FIXED_WIDTH_SMALL, BLOCK_WIDTH, lda);
        // now call GEPP on the entire C
        GEPP_SMALL(A + i * lda, B_PACKED, C, BLOCK_WIDTH, lda);
    }
    if (edge_col != 0)
    {
        // first do packing of B
        PACKING(B_PACKED, B + (lda - edge_col), lda, edge_col, FIXED_WIDTH_SMALL, BLOCK_WIDTH, lda);
        GEPP_SMALL(A + lda * (lda - edge_col), B_PACKED, C, edge_col, lda);
    }
}
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================
// ====================================================================================================================================================================================

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
static void INNER_BLOCK_MULT(double *A_ordered, double *B_ordered, double *C_ordered, int gemm_stride_size)
{
    __m256d c00 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd();
    __m256d c01 = _mm256_setzero_pd();
    __m256d c11 = _mm256_setzero_pd();
    __m256d c21 = _mm256_setzero_pd();
    __m256d c02 = _mm256_setzero_pd();
    __m256d c12 = _mm256_setzero_pd();
    __m256d c22 = _mm256_setzero_pd();
    __m256d c03 = _mm256_setzero_pd();
    __m256d c13 = _mm256_setzero_pd();
    __m256d c23 = _mm256_setzero_pd();
    // __m256d a0;
    // __m256d a1;
    // __m256d a2;
    // __m256d b;

    for (int k = 0; k < gemm_stride_size; k++)
    {
        __m256d a0 = _mm256_load_pd(A_ordered + k * 12);
        __m256d a1 = _mm256_load_pd(A_ordered + k * 12 + 4);
        __m256d a2 = _mm256_load_pd(A_ordered + k * 12 + 8);

        __m256d b = _mm256_broadcast_sd(B_ordered + k * 4);
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
static inline void GEBP_INNER(double *A, double *B, double *C, int gemm_stride_size, int gepp_stride_size, int gebp_stride_size, int lda)
{
    int leftover = gepp_stride_size % GEBP_HEIGHT;
    for (int i = 0; i < gepp_stride_size - leftover; i += GEBP_HEIGHT)
    {
        // setting up C
        // memset(C_PACKED, 0, GEBP_WIDTH * GEBP_HEIGHT * 8);
        INNER_BLOCK_MULT(A + i * GEMM_STRIDE, B, C_PACKED, gemm_stride_size);
        UNPACK_C(C + i, gebp_stride_size, GEBP_HEIGHT, lda);
    }

    if (leftover != 0)
    {
        // memset(C_PACKED, 0, GEBP_WIDTH * GEBP_HEIGHT * 8);
        INNER_BLOCK_MULT(A + (gepp_stride_size - leftover) * GEMM_STRIDE, B, C_PACKED, gemm_stride_size);
        UNPACK_C(C + (gepp_stride_size - leftover), gebp_stride_size, leftover, lda);
    }
}

// compute GEBP
// A is packed: GEPP_STRIDE * GEMM_STRIDE, real: gepp_stide_size * gemm_stride_size
// B is packed: GEMM_STRIDE * FIXED_WIDTH, real: gemm_stride_size * lda
// C is not packed, gepp_stride_size * lda
static inline void GEBP(double *A, double *B, double *C, int gemm_stride_size, int gepp_stride_size, int lda)
{
    int leftover = lda % GEBP_WIDTH;
    for (int j = 0; j < lda - leftover; j += GEBP_WIDTH)
    {
        GEBP_INNER(A, B + j * GEMM_STRIDE, C + j * lda, gemm_stride_size, gepp_stride_size, GEBP_WIDTH, lda);
    }

    if (leftover != 0)
    {
        GEBP_INNER(A, B + (lda - leftover) * GEMM_STRIDE, C + (lda - leftover) * lda, gemm_stride_size, gepp_stride_size, leftover, lda);
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
        GEBP(A_PACKED, B, C + i, gemm_stride_size, GEPP_STRIDE, lda);
    }

    if (GEPP_BORDER != 0)
    {
        PACK_A(A + (lda - GEPP_BORDER), gemm_stride_size, GEPP_BORDER, GEMM_STRIDE, GEPP_STRIDE, lda);
        GEBP(A_PACKED, B, C + (lda - GEPP_BORDER), gemm_stride_size, GEPP_BORDER, lda);
    }
}

// compute GEMM using GEPP
// A and B and C are not packed
static inline void GEMM(double *A, double *B, double *C, int lda)
{
    GEMM_BORDER = lda % GEMM_STRIDE; // get the border case
    GEPP_BORDER = lda % GEPP_STRIDE; // get the border case
    FIXED_WIDTH = ((lda + GEBP_WIDTH - 1) / GEBP_WIDTH) * GEBP_WIDTH;
    FIXED_HEIGHT = ((lda + GEBP_HEIGHT - 1) / GEBP_HEIGHT) * GEBP_HEIGHT;
    for (int i = 0; i < lda - GEMM_BORDER; i += GEMM_STRIDE)
    {
        PACK_B(B + i, lda, GEMM_STRIDE, FIXED_WIDTH, GEMM_STRIDE, lda);
        GEPP(A + i * lda, B_PACKED, C, GEMM_STRIDE, lda);
    }

    if (GEMM_BORDER != 0)
    {
        PACK_B(B + (lda - GEMM_BORDER), lda, GEMM_BORDER, FIXED_WIDTH, GEMM_STRIDE, lda);
        GEPP(A + (lda * (lda - GEMM_BORDER)), B_PACKED, C, GEMM_BORDER, lda);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
    if (lda < 97)
    {
        GEMM_SMALL(A, B, C, lda);
    }
    else
    {
        GEMM(A, B, C, lda);
    }
}
