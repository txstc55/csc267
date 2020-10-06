#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>

using namespace std;

#define cutoff 0.01
#define density 0.0005
#define bin_size_multiplier 2
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

int num_threads;
double bin_size;
int num_bin_1d;

typedef std::vector<particle_t> BIN; // for not supporting c++11

void buildBins(vector<BIN> &bins, particle_t *particles, int n)
{
    double size = sqrt(n * density);
    bin_size = cutoff * bin_size_multiplier;
    num_bin_1d = size / bin_size + 1;

    bins.resize(num_bin_1d * num_bin_1d);

    for (int i = 0; i < n; i++)
    {
        int col = particles[i].x / bin_size;
        int row = particles[i].y / bin_size;
        bins[col * num_bin_1d + row].push_back(particles[i]);
    }
}

//
//  benchmarking program
//
int main(int argc, char **argv)
{
    int navg, nabsavg = 0;
    double dmin, absmin = 1.0, davg, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }

    // init the bin
    vector<BIN> bins;
    buildBins(bins, particles, n);

    // init the moved particles
    vector<BIN> moved_particles;
    moved_particles.resize(num_threads);
    // reserve some space
    for (int i = 0; i < num_threads; i++)
    {
        moved_particles[i].reserve(num_bin_1d / num_threads);
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
#pragma omp parallel private(dmin)
    {
        int thread_num = omp_get_thread_num();
        for (int step = 0; step < 1000; step++)
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;
#pragma omp for reduction(+                   \
                          : navg) reduction(+ \
                                            : davg)
            for (int i = 0; i < num_bin_1d; i++)
            {
                for (int j = 0; j < num_bin_1d; j++)
                {
                    BIN &bin = bins[i * num_bin_1d + j];
                    if (bin.size() != 0)
                    {
                        for (int k = 0; k < bin.size(); k++)
                        {
                            bin[k].ax = 0.0;
                            bin[k].ay = 0.0;
                        }
                        for (int i_neighbor = max(0, i - 1); i_neighbor < min(num_bin_1d, i + 2); i_neighbor++)
                        {
                            for (int j_neighbor = max(0, j - 1); j_neighbor < min(num_bin_1d, j + 2); j_neighbor++)
                            {
                                BIN &bin_neighbor = bins[i_neighbor * num_bin_1d + j_neighbor];
                                for (int k = 0; k < bin.size(); k++)
                                {
                                    for (int l = 0; l < bin_neighbor.size(); l++)
                                    {
                                        apply_force(bin[k], bin_neighbor[l], &dmin, &davg, &navg);
                                    }
                                }
                            }
                        }
                    }
                }
            }

#pragma omp for
            for (int i = 0; i < num_bin_1d; i++)
            {
                for (int j = 0; j < num_bin_1d; j++)
                {
                    BIN &bin = bins[i * num_bin_1d + j];
                    int current_index = 0;
                    int last = bin.size();
                    for (int k = 0; k < bin.size(); k++)
                    {
                        move(bin[current_index]);
                        int col = int(bin[current_index].x / bin_size);
                        int row = int(bin[current_index].y / bin_size);
                        // the particle has been moved, we need to exclude it from this bin
                        // and save it to a tmp place so that it will not be computed twice
                        if (col != i || row != j)
                        {
                            moved_particles[thread_num].push_back(bin[current_index]);
                            bin[current_index] = bin[last - 1];
                            last--;
                            continue;
                        }
                        current_index++;
                    }
                    bin.resize(last);
                }
            }
// #pragma omp master
// {
//     if (n == 10000){
//         int count = 0;
//         for (int j = 0; j<num_threads; j++){
//             count+=moved_particles[j].size();
//         }
//         printf("OMP out of bin particles size: %d\n", count);
//     }
// }
#pragma omp for
            for (int j = 0; j < num_threads; j++)
            {
                // printf("Accessing %d moved particle bin\n", j);
                // deal with the particles that has been moved out of the bin
                for (int i = 0; i < moved_particles[j].size(); i++)
                {
                    int col = moved_particles[j][i].x / bin_size;
                    int row = moved_particles[j][i].y / bin_size;
                    #pragma omp critical
                    bins[col * num_bin_1d + row].push_back(moved_particles[j][i]);
                }
                moved_particles[j].clear();
            }

            if (find_option(argc, argv, "-no") == -1)
            {
//
//  compute statistical data
//
#pragma omp master
                if (navg)
                {
                    absavg += davg / navg;
                    nabsavg++;
                }

#pragma omp critical
                if (dmin < absmin)
                    absmin = dmin;

//
//  save if necessary
//
#pragma omp master
                if (fsave && (step % SAVEFREQ) == 0)
                {
                    int index = 0;
                    for (int i = 0; i < num_bin_1d; i++)
                    {
                        for (int j = 0; j < num_bin_1d; j++)
                        {
                            BIN &bin = bins[i * num_bin_1d + j];
                            for (int k = 0; k < bin.size(); k++)
                            {
                                particles[index] = bin[k];
                                index++;
                            }
                        }
                    }
                    save(fsave, n, particles);
                }
            }
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("n = %d,threads = %d, simulation time = %g seconds", n, num_threads, simulation_time);

    if (find_option(argc, argv, "-no") == -1)
    {
        if (nabsavg)
            absavg /= nabsavg;
        //
        //  -the minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4)
            printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8)
            printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if (fsum)
        fprintf(fsum, "%d %d %g\n", n, num_threads, simulation_time);

    //
    // Clearing space
    //
    if (fsum)
        fclose(fsum);

    free(particles);
    if (fsave)
        fclose(fsave);

    return 0;
}
