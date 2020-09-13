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

#define ORDERED_BLOCK_SIZE 256                                              // this is the middle block size
#define ORDERED_BLOCK_SIZE_SQUARE (ORDERED_BLOCK_SIZE * ORDERED_BLOCK_SIZE) // how many elements inside each middle block
#define INNER_BLOCK_SIZE 4                                                  // we will do a inner block of 4 by 4, and will not change this
#define INNER_BLOCK_SIZE_SQUARE (INNER_BLOCK_SIZE * INNER_BLOCK_SIZE)       // the length of the inner block

#define BLOCK_HEIGHT 64                                                  // the height of block C when we perform GEPM, this will also be the height of A's block
#define BLOCK_WIDTH 128                                                  // the width of block A when we perform GEBP, will also be the height of block B
#define SMALLER_BLOCK_WIDTH 16                                           // this is the width of the block when performing GEBP
#define BLOCK_HEIGHT_DOUBLE (BLOCK_HEIGHT * 2)                           // redundent computation
#define BLOCK_HEIGHT_TRIPLE (BLOCK_HEIGHT * 3)                           // redundent computation
#define BLOCK_WIDTH_DOUBLE (BLOCK_WIDTH * 2)                             // redundent computation
#define BLOCK_WIDTH_TRIPLE (BLOCK_WIDTH * 3)                             // redundent computation
#define SMALLER_BLOCK_BYTE_SIZE (BLOCK_HEIGHT * SMALLER_BLOCK_WIDTH * 8) // redundent computation
// for sse instructions, the machine supports sse4, which means 4 doubles in a register
// so we will use _mm256 for operations
#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <string.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

int FIXED_WIDTH = 0;                                                                     // a multiple of SMALLER_BLOCK_WIDTH that is just larger than lda
static double A_PACKED[BLOCK_HEIGHT * BLOCK_WIDTH] __attribute__((aligned(32)));         // store the block of A that is packed
static double C_PACKED[BLOCK_HEIGHT * SMALLER_BLOCK_WIDTH] __attribute__((aligned(32))); // store the block of C that is packed
static double B_PACKED[1056 * BLOCK_WIDTH] __attribute__((aligned(32)));                 // store the block of C that is packed
static __m256d A_COLS[BLOCK_HEIGHT / 4][4];                                              // store the __m256d to avoid repetitive loading and storing
// static __m256d B_NUMS[SMALLER_BLOCK_WIDTH];                                              // same reason
int edge_row = 0;
int edge_col = 0;
int smaller_edge_col = 0;

// print a normal matrix
void print_matrix(double *matrix, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%f ", matrix[i + j * height]);
        }
        printf("\n");
    }
    printf("\n");
}

// the smallest computing units
// computes the result of a 4 by 4 matrix multiplication
// this version assumes that the matrix is not oriented in the way of register blocking
void inner_block_mult(double *A_ordered, double *B_ordered, double *C_ordered)
{
    // __m512d col4 = _mm512_load_pd(A_ordered);
    // load the four columns into register
    __m256d col0 = _mm256_load_pd(A_ordered);
    __m256d col1 = _mm256_load_pd(A_ordered + BLOCK_HEIGHT);
    __m256d col2 = _mm256_load_pd(A_ordered + BLOCK_HEIGHT_DOUBLE);
    __m256d col3 = _mm256_load_pd(A_ordered + BLOCK_HEIGHT_TRIPLE);

    // load the four values in the column

    __m256d val0_0 = _mm256_set1_pd(B_ordered[0]);
    __m256d val0_1 = _mm256_set1_pd(B_ordered[1]);
    __m256d val0_2 = _mm256_set1_pd(B_ordered[2]);
    __m256d val0_3 = _mm256_set1_pd(B_ordered[3]);
    // load whatever is already in C, since it will be added repeatedly
    __m256d col_result = _mm256_load_pd(C_ordered);
    // do the multiplication as well as the addition at the same time
    col_result = _mm256_fmadd_pd(col3, val0_3, _mm256_fmadd_pd(col2, val0_2, _mm256_fmadd_pd(col1, val0_1, _mm256_fmadd_pd(col0, val0_0, col_result))));
    // store the column result back in C
    _mm256_store_pd(C_ordered, col_result);

    __m256d val1_0 = _mm256_set1_pd(B_ordered[BLOCK_WIDTH]);
    __m256d val1_1 = _mm256_set1_pd(B_ordered[1 + BLOCK_WIDTH]);
    __m256d val1_2 = _mm256_set1_pd(B_ordered[2 + BLOCK_WIDTH]);
    __m256d val1_3 = _mm256_set1_pd(B_ordered[3 + BLOCK_WIDTH]);
    // load whatever is already in C, since it will be added repeatedly
    col_result = _mm256_load_pd(C_ordered + BLOCK_HEIGHT);
    // do the multiplication as well as the addition at the same time
    col_result = _mm256_fmadd_pd(col3, val1_3, _mm256_fmadd_pd(col2, val1_2, _mm256_fmadd_pd(col1, val1_1, _mm256_fmadd_pd(col0, val1_0, col_result))));
    // store the column result back in C
    _mm256_store_pd(C_ordered + BLOCK_HEIGHT, col_result);

    __m256d val2_0 = _mm256_set1_pd(B_ordered[BLOCK_WIDTH_DOUBLE]);
    __m256d val2_1 = _mm256_set1_pd(B_ordered[1 + BLOCK_WIDTH_DOUBLE]);
    __m256d val2_2 = _mm256_set1_pd(B_ordered[2 + BLOCK_WIDTH_DOUBLE]);
    __m256d val2_3 = _mm256_set1_pd(B_ordered[3 + BLOCK_WIDTH_DOUBLE]);
    // load whatever is already in C, since it will be added repeatedly
    col_result = _mm256_load_pd(C_ordered + BLOCK_HEIGHT_DOUBLE);
    // do the multiplication as well as the addition at the same time
    col_result = _mm256_fmadd_pd(col3, val2_3, _mm256_fmadd_pd(col2, val2_2, _mm256_fmadd_pd(col1, val2_1, _mm256_fmadd_pd(col0, val2_0, col_result))));
    // store the column result back in C
    _mm256_store_pd(C_ordered + BLOCK_HEIGHT_DOUBLE, col_result);

    __m256d val3_0 = _mm256_set1_pd(B_ordered[BLOCK_WIDTH_TRIPLE]);
    __m256d val3_1 = _mm256_set1_pd(B_ordered[1 + BLOCK_WIDTH_TRIPLE]);
    __m256d val3_2 = _mm256_set1_pd(B_ordered[2 + BLOCK_WIDTH_TRIPLE]);
    __m256d val3_3 = _mm256_set1_pd(B_ordered[3 + BLOCK_WIDTH_TRIPLE]);
    // load whatever is already in C, since it will be added repeatedly
    col_result = _mm256_load_pd(C_ordered + BLOCK_HEIGHT_TRIPLE);
    // do the multiplication as well as the addition at the same time
    col_result = _mm256_fmadd_pd(col3, val3_3, _mm256_fmadd_pd(col2, val3_2, _mm256_fmadd_pd(col1, val3_1, _mm256_fmadd_pd(col0, val3_0, col_result))));
    // store the column result back in C
    _mm256_store_pd(C_ordered + BLOCK_HEIGHT_TRIPLE, col_result);
}

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

// the inner loop funciton call in GEBP, compies a pannel = a small block times a pannel
// here, all three inputs should be packed, and the dimension for the three inputs are:
// A: BLOCK_HEIGHT * BLOCK_WIDTH
// B: BLOCK_WIDTH * SMALLER_BLOCK_WIDTH
// C: BLOCK_HEIGHT * SMALLER_BLOCK_WIDTH
// let us denote: M: BLOCK_HEIGHT, K: BLOCK_WIDTH, N: SMALLER_BLOCK_WIDTH
static inline void GEBP_INNER(double *A, double *B, double *C, int width, int height)
{
    // the goal is to have less access for A i think
    for (int k = 0; k < width; k += INNER_BLOCK_SIZE)
    {
        int k_BLOCK_HEIGHT = k * BLOCK_HEIGHT;
        for (int n = 0; n < SMALLER_BLOCK_WIDTH; n += INNER_BLOCK_SIZE)
        {
            int n_BLOCK_HEIGHT = n * BLOCK_HEIGHT;
            int n_BLOCK_WIDTH = n * BLOCK_WIDTH;
            for (int m = 0; m < height; m += INNER_BLOCK_SIZE)
            {
                inner_block_mult(A + m + k_BLOCK_HEIGHT, B + k + n_BLOCK_WIDTH, C + m + n_BLOCK_HEIGHT);
            }
        }
    }
}

static inline void GEBP_INNER_EXPANDED(double *A, double *B, double *C, int width, int height)
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
static inline void GEBP(double *A, double *B, double *C, int width, int height, int lda)
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
        GEBP_INNER_EXPANDED(A_PACKED, B + i * BLOCK_WIDTH, C_PACKED, width, height); // we need to move the pointer of B
        UNPACKING(C_PACKED, C + i * lda, SMALLER_BLOCK_WIDTH, height, BLOCK_HEIGHT, lda);
    }

    if (smaller_edge_col != 0)
    {
        // now do the edge case
        memset(C_PACKED, 0, SMALLER_BLOCK_BYTE_SIZE);
        GEBP_INNER_EXPANDED(A_PACKED, B + (lda - smaller_edge_col) * BLOCK_WIDTH, C_PACKED, width, height);
        UNPACKING(C_PACKED, C + (lda - smaller_edge_col) * lda, smaller_edge_col, height, BLOCK_HEIGHT, lda);
    }
}

// compute GEPP using GEBP
// here, A, B, C are not packed
// the dimensions of the inputs are:
// A: lda * width
// B: BLOCK_WIDTH * FIXED_WIDTH
// C: lda * lda
static inline void GEPP(double *A, double *B, double *C, int width, int lda)
{
    // printf("B_packed: \n");
    // print_matrix(B_PACKED, BLOCK_WIDTH, FIXED_WIDTH);
    for (int i = 0; i < lda - edge_row; i += BLOCK_HEIGHT)
    {
        GEBP(A + i, B, C + i, width, BLOCK_HEIGHT, lda);
    }
    if (edge_row != 0)
    {
        GEBP(A + (lda - edge_row), B, C + (lda - edge_row), width, edge_row, lda);
    }
}

// compute GEMM using GEPP
// here, A, B, C are not packed
// but B will be packed inside this function to pass into GEBP
__attribute__((optimize("unroll-loops"))) static inline void GEMM(double *A, double *B, double *C, int lda)
{
    // we will need to divide A vertically, B horizontally
    edge_col = lda % BLOCK_WIDTH;
    edge_row = lda % BLOCK_HEIGHT;
    smaller_edge_col = lda % SMALLER_BLOCK_WIDTH;
    FIXED_WIDTH = ((lda + SMALLER_BLOCK_WIDTH - 1) / SMALLER_BLOCK_WIDTH) * SMALLER_BLOCK_WIDTH;
    for (int i = 0; i < lda - edge_col; i += BLOCK_WIDTH)
    {
        // first do packing of B
        PACKING(B_PACKED, B + i, lda, BLOCK_WIDTH, FIXED_WIDTH, BLOCK_WIDTH, lda);
        // now call GEPP on the entire C
        GEPP(A + i * lda, B_PACKED, C, BLOCK_WIDTH, lda);
    }
    if (edge_col != 0)
    {
        // first do packing of B
        PACKING(B_PACKED, B + (lda - edge_col), lda, edge_col, FIXED_WIDTH, BLOCK_WIDTH, lda);
        GEPP(A + lda * (lda - edge_col), B_PACKED, C, edge_col, lda);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
    GEMM(A, B, C, lda);
}
