#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include <algorithm> // std::sort

using namespace std;

#define cutoff 0.01
#define density 0.0005
#define bin_size_multiplier 2

double bin_size;
int num_bin_1d;

typedef std::vector<particle_t> BIN; // for not supporting c++11

void buildBins(vector<BIN> &bins, particle_t *particles, int n)
{
    double size = sqrt(n * density);
    bin_size = cutoff * bin_size_multiplier;
    num_bin_1d = size / bin_size + 1;

    bins.resize(num_bin_1d * num_bin_1d);
    for (int i = 0; i < num_bin_1d * num_bin_1d; i++)
    {
        bins[i].reserve(10);
    }

    for (int i = 0; i < n; i++)
    {
        int col = particles[i].x / bin_size;
        int row = particles[i].y / bin_size;
        bins[col * num_bin_1d + row].push_back(particles[i]);
    }
}

inline void binMovements(BIN &bin, int i, int j, BIN &moved_particles)
{
    int current_index = 0;
    int last = bin.size();
    for (int k = 0; k < bin.size(); k++)
    {
        move(bin[current_index]);
        int col = bin[current_index].x / bin_size;
        int row = bin[current_index].y / bin_size;
        // the particle has been moved, we need to exclude it from this bin
        // and save it to a tmp place so that it will not be computed twice
        if (col != i || row != j)
        {
            moved_particles.push_back(bin[current_index]);
            bin[current_index] = bin[last - 1];
            last--;
            continue;
        }
        current_index++;
    }
    bin.resize(last);
}

//
//  benchmarking program
//
int main(int argc, char **argv)
{
    int navg, nabsavg = 0;
    double davg, dmin, absmin = 1.0, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
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

    // init the bin
    vector<BIN> bins;
    buildBins(bins, particles, n);

    // whenever a particle is moved out of the bin
    // it will be placed here first
    BIN moved_particles;
    moved_particles.reserve(n); // reserve some space
    int i, j;

    double simulation_time = read_timer();

    for (int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;

        // first setting acceleration to 0
        for (i = 0; i < num_bin_1d; i++)
        {
            for (j = 0; j < num_bin_1d; j++)
            {
                BIN &bin = bins[i * num_bin_1d + j];
                for (int k = 0; k < bin.size(); k++)
                {
                    bin[k].ax = 0.0;
                    bin[k].ay = 0.0;
                }
            }
        }
        // printf("Max number of particle in bin is: %d\n", max_particles_in_bin);

        // now, for each bin, what we can do is to loop through bins at position
        // (i, j+1), (i + 1, j-1), (i+1, j), (i+1, j+1)
        // particles in the two neighbors will interact with each other
        // this way, we do not need to do 9 neighbors
        for (i = 0; i < num_bin_1d - 1; i++)
        {
            // deal with the first row
            j = 0;
            BIN &bin_first_row = bins[i * num_bin_1d + j];
            if (bin_first_row.size() != 0)
            {
                BIN &bin_neighbor_0 = bins[i * num_bin_1d + j + 1];
                BIN &bin_neighbor_2 = bins[(i + 1) * num_bin_1d + j];
                BIN &bin_neighbor_3 = bins[(i + 1) * num_bin_1d + j + 1];
                for (int k = 0; k < bin_first_row.size(); k++)
                {
                    for (int l = k + 1; l < bin_first_row.size(); l++)
                    {
                        apply_force(bin_first_row[k], bin_first_row[l], &dmin, &davg, &navg);
                        apply_force(bin_first_row[l], bin_first_row[k], &dmin, &davg, &navg);
                    }
                    for (int l = 0; l < bin_neighbor_0.size(); l++)
                    {
                        apply_force(bin_first_row[k], bin_neighbor_0[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_0[l], bin_first_row[k], &dmin, &davg, &navg);
                    }
                    for (int l = 0; l < bin_neighbor_2.size(); l++)
                    {
                        apply_force(bin_first_row[k], bin_neighbor_2[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_2[l], bin_first_row[k], &dmin, &davg, &navg);
                    }
                    for (int l = 0; l < bin_neighbor_3.size(); l++)
                    {
                        apply_force(bin_first_row[k], bin_neighbor_3[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_3[l], bin_first_row[k], &dmin, &davg, &navg);
                    }
                }
                binMovements(bin_first_row, i, j, moved_particles);
            }

            for (j = 1; j < num_bin_1d - 1; j++)
            {
                BIN &bin = bins[i * num_bin_1d + j];
                if (bin.size() != 0)
                {
                    BIN &bin_neighbor_0 = bins[i * num_bin_1d + j + 1];
                    BIN &bin_neighbor_1 = bins[(i + 1) * num_bin_1d + j - 1];
                    BIN &bin_neighbor_2 = bins[(i + 1) * num_bin_1d + j];
                    BIN &bin_neighbor_3 = bins[(i + 1) * num_bin_1d + j + 1];
                    for (int k = 0; k < bin.size(); k++)
                    {
                        for (int l = k + 1; l < bin.size(); l++)
                        {
                            apply_force(bin[k], bin[l], &dmin, &davg, &navg);
                            apply_force(bin[l], bin[k], &dmin, &davg, &navg);
                        }
                        for (int l = 0; l < bin_neighbor_0.size(); l++)
                        {
                            apply_force(bin[k], bin_neighbor_0[l], &dmin, &davg, &navg);
                            apply_force(bin_neighbor_0[l], bin[k], &dmin, &davg, &navg);
                        }
                        for (int l = 0; l < bin_neighbor_1.size(); l++)
                        {
                            apply_force(bin[k], bin_neighbor_1[l], &dmin, &davg, &navg);
                            apply_force(bin_neighbor_1[l], bin[k], &dmin, &davg, &navg);
                        }
                        for (int l = 0; l < bin_neighbor_2.size(); l++)
                        {
                            apply_force(bin[k], bin_neighbor_2[l], &dmin, &davg, &navg);
                            apply_force(bin_neighbor_2[l], bin[k], &dmin, &davg, &navg);
                        }
                        for (int l = 0; l < bin_neighbor_3.size(); l++)
                        {
                            apply_force(bin[k], bin_neighbor_3[l], &dmin, &davg, &navg);
                            apply_force(bin_neighbor_3[l], bin[k], &dmin, &davg, &navg);
                        }
                    }
                    binMovements(bin, i, j, moved_particles);
                }
            }

            // deal with the last row
            BIN &bin_last_row = bins[i * num_bin_1d + j];
            if (bin_last_row.size() != 0)
            {
                BIN &bin_neighbor_1 = bins[(i + 1) * num_bin_1d + j - 1];
                BIN &bin_neighbor_2 = bins[(i + 1) * num_bin_1d + j];
                for (int k = 0; k < bin_last_row.size(); k++)
                {
                    for (int l = k + 1; l < bin_last_row.size(); l++)
                    {
                        apply_force(bin_last_row[k], bin_last_row[l], &dmin, &davg, &navg);
                        apply_force(bin_last_row[l], bin_last_row[k], &dmin, &davg, &navg);
                    }
                    for (int l = 0; l < bin_neighbor_1.size(); l++)
                    {
                        apply_force(bin_last_row[k], bin_neighbor_1[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_1[l], bin_last_row[k], &dmin, &davg, &navg);
                    }
                    for (int l = 0; l < bin_neighbor_2.size(); l++)
                    {
                        apply_force(bin_last_row[k], bin_neighbor_2[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_2[l], bin_last_row[k], &dmin, &davg, &navg);
                    }
                }
                binMovements(bin_last_row, i, j, moved_particles);
            }
        }

        // deal with the last column
        for (j = 0; j < num_bin_1d - 1; j++)
        {
            BIN &bin = bins[i * num_bin_1d + j];
            if (bin.size() != 0)
            {
                BIN &bin_neighbor_0 = bins[i * num_bin_1d + j + 1];
                for (int k = 0; k < bin.size(); k++)
                {
                    for (int l = k + 1; l < bin.size(); l++)
                    {
                        apply_force(bin[k], bin[l], &dmin, &davg, &navg);
                        apply_force(bin[l], bin[k], &dmin, &davg, &navg);
                    }
                }
                for (int k = 0; k < bin.size(); k++)
                {
                    for (int l = 0; l < bin_neighbor_0.size(); l++)
                    {
                        apply_force(bin[k], bin_neighbor_0[l], &dmin, &davg, &navg);
                        apply_force(bin_neighbor_0[l], bin[k], &dmin, &davg, &navg);
                    }
                }
                binMovements(bin, i, j, moved_particles);
            }
        }

        // deal with the very last bin at the corner
        BIN &last_bin = bins[i * num_bin_1d + j];
        for (int k = 0; k < last_bin.size(); k++)
        {
            for (int l = k + 1; l < last_bin.size(); l++)
            {
                apply_force(last_bin[k], last_bin[l], &dmin, &davg, &navg);
                apply_force(last_bin[l], last_bin[k], &dmin, &davg, &navg);
            }
        }
        // move the particles in the last bin
        binMovements(last_bin, i, j, moved_particles);

        // deal with the particles that has been moved out of the bin
        for (i = 0; i < moved_particles.size(); i++)
        {
            int col = moved_particles[i].x / bin_size;
            int row = moved_particles[i].y / bin_size;
            bins[col * num_bin_1d + row].push_back(moved_particles[i]);
        }
        // printf("Out of bin particle size: %d\n", moved_particles.size());
        moved_particles.clear();

        if (find_option(argc, argv, "-no") == -1)
        {
            if (navg)
            {
                absavg += davg / navg;
                nabsavg++;
            }

            if (dmin < absmin)
                absmin = dmin;

            if (fsave && (step % SAVEFREQ) == 0)
            {
                int index = 0;
                for (i = 0; i < num_bin_1d; i++)
                {
                    for (j = 0; j < num_bin_1d; j++)
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
    simulation_time = read_timer() - simulation_time;

    printf("n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum, "%d %g\n", n, simulation_time);

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