#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <math.h>
#include <vector>

using namespace std;

#define cutoff 0.01
#define density 0.0005
#define bin_size_multiplier 2
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

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
    double rdavg, rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
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

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size(n);
    if (rank == 0)
        init_particles(n, particles);

    // broadcast the particles to let them do the binning
    // this part is not in timing so whatever
    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);
    vector<BIN> bins;
    buildBins(bins, particles, n); // now every process will have the bins initialized all the same

    // each processor will only do part of bins
    int cols_per_proc = (num_bin_1d + n_proc - 1) / n_proc;    // how many cols per processor on average
    int col_start = rank * cols_per_proc;                      // which col does my processor start at
    int col_end = min((rank + 1) * cols_per_proc, num_bin_1d); // which col does my processor end with
    int col_my_proc = col_end - col_start;                     // how many cols is my processor responsible for

    BIN moved_particles;                                 // the particles that has been moving around
    vector<BIN> out_of_proc_particles;                   // the particles that moved so far it is out of this processor's reach, we will record which proc it is going to
    BIN out_of_proc_particles_flattened;                 // the flattened list
    BIN left_border_particles;                           // the particles on the left most cols
    BIN right_border_particles;                          // the particles on the right most cols
    BIN incoming_right_border_particles;                 // for receiving the right border
    BIN incoming_left_border_particles;                  // for receiving the left border
    BIN incoming_particles_flattened;                    // the flattened list of particles that is being received
    moved_particles.reserve(num_bin_1d);                 // reserve some space
    out_of_proc_particles_flattened.reserve(num_bin_1d); // reserve some space
    incoming_particles_flattened.reserve(num_bin_1d);    // reserve some space
    out_of_proc_particles.resize(n_proc);                // resize it to the number of procs

    int num_particles_sending_out[n_proc];  // how many particles I will be sending to other processors
    int num_particles_receiving_in[n_proc]; // how many particles I will be receiving from other processors

    int send_displacement[n_proc];    // displacement for all to all send
    int receive_displacement[n_proc]; // displacement for all to all receive

    send_displacement[0] = 0;    // the displacement for the first element is always 0
    receive_displacement[0] = 0; // the displacement for the first element is always 0

    // MPI_Status from_left, from_right; // for the receiving events
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if (find_option(argc, argv, "-no") == -1)
            if (fsave && (step % SAVEFREQ) == 0)
                save(fsave, n, particles);

        // we only need to do part of the bins
        // the apply force part stays the same
        for (int i = col_start; i < col_end; i++)
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

        // dont care about this part
        // it will stay the same since its just reduce
        if (find_option(argc, argv, "-no") == -1)
        {

            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                //
                // Computing statistical data
                //
                if (rnavg)
                {
                    absavg += rdavg / rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin)
                    absmin = rdmin;
            }
        }

        // the move particles is tricky
        // there are particles that move out of the bins
        // but it still stays within the colums this processor is responsible for
        // but there are also particles moving out of this processor's region
        // there can be particles that moves so fast that it is also nit in last processor's region
        int count = 0;
        int count2 = 0;
        for (int i = col_start; i < col_end; i++)
        {
            for (int j = 0; j < num_bin_1d; j++)
            {
                BIN &bin = bins[i * num_bin_1d + j];
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
                        count++;
                        if (col >= col_start && col < col_end)
                        {
                            // if it is still within this processor
                            moved_particles.push_back(bin[current_index]);
                        }
                        else
                        {
                            count2++;
                            // it is out of this processor, put it in the corresponding place
                            out_of_proc_particles[col / cols_per_proc].push_back(bin[current_index]);
                        }
                        bin[current_index] = bin[last - 1];
                        last--;
                        continue;
                    }
                    current_index++;
                }
                bin.resize(last);
            }
        }
        // printf("Proc %d has %d out of bin particles, number of particles = %d, number of processors = %d\n", rank, count, n, n_proc);
        // if (count2 != 0)
        // {
        //     printf("Proc %d has %d out of proc particles, col start: %d, col end: %d, number of particles = %d, number of processors = %d\n", rank, count2, col_start, col_end, n, n_proc);
        // }
        // deal with the local particles
        for (int i = 0; i < moved_particles.size(); i++)
        {
            int col = moved_particles[i].x / bin_size;
            int row = moved_particles[i].y / bin_size;
            bins[col * num_bin_1d + row].push_back(moved_particles[i]);
        }
        moved_particles.clear();

        // how many particles I will be sending out to other processors
        num_particles_sending_out[0] = out_of_proc_particles[0].size();

        for (int i = 1; i < n_proc; i++)
        {
            send_displacement[i] = send_displacement[i - 1] + out_of_proc_particles[i - 1].size();
            num_particles_sending_out[i] = out_of_proc_particles[i].size();
        }

        // printf("Proc %d sending message out on number of particles, number of particles = %d, number of processors = %d\n", rank, n, n_proc);

        // for each processor how many elements I will be sending to you
        MPI_Alltoall(num_particles_sending_out, 1, MPI_INT, num_particles_receiving_in, 1, MPI_INT, MPI_COMM_WORLD);

        // calculate the displacement for receiving elements
        for (int i = 1; i < n_proc; i++)
        {
            receive_displacement[i] = receive_displacement[i - 1] + num_particles_receiving_in[i - 1];
        }

        // now we know how many elements we will be receiving from each processor
        // let us group the elements now
        for (int i = 0; i < n_proc; i++)
        {
            BIN &bin = out_of_proc_particles[i];
            for (int j = 0; j < bin.size(); j++)
            {
                out_of_proc_particles_flattened.push_back(bin[j]);
            }
            bin.clear(); // clear it for next iteration
        }

        // resize the array to receive elements from other processors
        incoming_particles_flattened.resize(receive_displacement[n_proc - 1] + num_particles_receiving_in[n_proc - 1]);

        // printf("Proc %d sending %d particles, number of particles = %d, number of processors = %d\n", rank, out_of_proc_particles_flattened.size(), n, n_proc);
        // now the particles are grouped, the number and displacement is already set, we begin the sending process
        MPI_Alltoallv(out_of_proc_particles_flattened.data(), num_particles_sending_out, send_displacement, PARTICLE, incoming_particles_flattened.data(), num_particles_receiving_in, receive_displacement, PARTICLE, MPI_COMM_WORLD);

        // now we have all the particles that is coming to this processor
        // start putting them in the correct bin
        for (int i = 0; i < incoming_particles_flattened.size(); i++)
        {
            int col = incoming_particles_flattened[i].x / bin_size;
            int row = incoming_particles_flattened[i].y / bin_size;
            bins[col * num_bin_1d + row].push_back(incoming_particles_flattened[i]);
        }

        // // this will not be used anymore, technically dont need to clear it since it will be written over
        // out_of_proc_particles_flattened.clear();

        int left_border_size = 0;
        int right_border_size = 0;
        for (int j = 0; j < num_bin_1d; j++)
        {
            int left_index = col_start * num_bin_1d + j;
            int right_index = (col_end - 1) * num_bin_1d + j;
            BIN &left_bin = bins[left_index];
            BIN &right_bin = bins[right_index];
            left_border_particles.insert(left_border_particles.begin() + left_border_size, left_bin.begin(), left_bin.end());
            right_border_particles.insert(right_border_particles.begin() + right_border_size, right_bin.begin(), right_bin.end());
            left_border_size += left_bin.size();
            right_border_size += right_bin.size();
            left_bin.clear();
            right_bin.clear();
        }

        // printf("Proc %d sending and receiving message about ghost zones, number of particles = %d, number of processors = %d\n", rank, n, n_proc);
        // how many particles from the left side border and right side border from other processors
        int border_from_left_size, border_from_right_size;
        if (rank != 0)
        {
            // receiving how large the border from left will be
            MPI_Recv(&border_from_left_size, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank != n_proc - 1)
        {
            // sending how large my right border (your left border) is
            MPI_Send(&right_border_size, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }

        if (rank != n_proc - 1)
        {
            // receiving how large the border from right will be
            MPI_Recv(&border_from_right_size, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank != 0)
        {
            // sending how large my left border (your right border) is
            MPI_Send(&left_border_size, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        }

        incoming_left_border_particles.resize(border_from_left_size);
        incoming_right_border_particles.resize(border_from_right_size);

        // printf("Proc %d sending and receiving ghost zones, left size = %d, right size = %d, number of particles = %d, number of processors = %d\n", rank, left_border_particles.size(), right_border_particles.size(), n, n_proc);
        if (rank != 0)
        {
            // receiving the left border
            MPI_Recv(incoming_left_border_particles.data(), border_from_left_size, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank != n_proc - 1)
        {
            // sending my right border (your left border)
            MPI_Send(right_border_particles.data(), right_border_size, PARTICLE, rank + 1, 0, MPI_COMM_WORLD);
        }

        if (rank != n_proc - 1)
        {
            // receiving my right border
            MPI_Recv(incoming_right_border_particles.data(), border_from_right_size, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank != 0)
        {
            // sending my left border (your right border)
            MPI_Send(left_border_particles.data(), left_border_size, PARTICLE, rank - 1, 0, MPI_COMM_WORLD);
        }

        // now put the new particles to the corresponding places
        for (int i = 0; i < incoming_right_border_particles.size(); i++)
        {
            int col = incoming_right_border_particles[i].x / bin_size;
            int row = incoming_right_border_particles[i].y / bin_size;
            bins[col * num_bin_1d + row].push_back(incoming_right_border_particles[i]);
        }

        for (int i = 0; i < incoming_left_border_particles.size(); i++)
        {
            int col = incoming_left_border_particles[i].x / bin_size;
            int row = incoming_left_border_particles[i].y / bin_size;
            bins[col * num_bin_1d + row].push_back(incoming_left_border_particles[i]);
        }

        // by this time everything is synced
    }

    simulation_time = read_timer() - simulation_time;

    if (rank == 0)
    {
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
            fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
    }

    //
    //  release resources
    //
    if (fsum)
        fclose(fsum);
    free(particles);
    if (fsave)
        fclose(fsave);

    MPI_Finalize();

    return 0;
}
