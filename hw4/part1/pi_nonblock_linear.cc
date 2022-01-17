#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

typedef long long int ll;

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // run with 4 hosts

    MPI_Status status;
    ll tosses_per_process = tosses / world_size;

    unsigned int seed = 123;
    seed *= world_rank;
    srand(seed);
    ll cnt = 0;

    for(ll i = 0; i < tosses_per_process; ++i){
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if(x * x + y * y <= 1){
            cnt++;
        }

    }

    ll total_cnt = 0;

    if (world_rank > 0)
    {
        // TODO: MPI workers
	
	MPI_Request r;
	MPI_Isend(&cnt, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &r);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size - 1];
	MPI_Status status[world_size - 1];
	ll buffer[world_size - 1];
	for (int i = 1; i < world_size; ++i){
	    MPI_Irecv(&buffer[i - 1], 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
	}

        MPI_Waitall(world_size - 1, requests, status);

	for (int i = 0; i < world_size - 1; i++){
	    cnt += buffer[i];
	}
    }

    if (world_rank == 0)
    {
        // TODO: PI result
	
	pi_result = 4 * cnt / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
