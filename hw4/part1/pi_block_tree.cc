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

    // TODO: binary tree redunction

    int tree_level = 1;
    int num_of_process = world_size;
    while(num_of_process != 1){
	if(tree_level == 1){
	    // send: 1 3 5 7 9 11 13 15
	    if(world_rank % 2 == 1){
		MPI_Send(&cnt, 1, MPI_UNSIGNED_LONG, world_rank - 1, 0, MPI_COMM_WORLD);
	    }else if(world_rank % 2 == 0){
	    // receive: 0 2 4 6 8 10 12 14
	    	ll buffer;
	    	MPI_Recv(&buffer, 1, MPI_UNSIGNED_LONG, world_rank + 1, 0, MPI_COMM_WORLD, &status);
	    	cnt += buffer;
	    }
	
	}else if(tree_level == 2){
	    // send: 2 6 10 14
            if(world_rank % 4 == 2){
                MPI_Send(&cnt, 1, MPI_UNSIGNED_LONG, world_rank - 2, 0, MPI_COMM_WORLD);
            }else if(world_rank % 4 == 0){
            // receive: 0 4 8 12
                ll buffer;
                MPI_Recv(&buffer, 1, MPI_UNSIGNED_LONG, world_rank + 2, 0, MPI_COMM_WORLD, &status);
                cnt += buffer;
            }

	}else if(tree_level == 3){
	    // send: 4 12
            if(world_rank % 8 == 4){
                MPI_Send(&cnt, 1, MPI_UNSIGNED_LONG, world_rank - 4, 0, MPI_COMM_WORLD);
            }else if(world_rank % 8 == 0){
            // receive: 0 8
                ll buffer;
                MPI_Recv(&buffer, 1, MPI_UNSIGNED_LONG, world_rank + 4, 0, MPI_COMM_WORLD, &status);
                cnt += buffer;
            }

	}else if(tree_level == 4){
	    // send: 8
            if(world_rank % 16 == 8){
                MPI_Send(&cnt, 1, MPI_UNSIGNED_LONG, world_rank - 8, 0, MPI_COMM_WORLD);
            }else if(world_rank % 16 == 0){
            // receive: 0
                ll buffer;
                MPI_Recv(&buffer, 1, MPI_UNSIGNED_LONG, world_rank + 8, 0, MPI_COMM_WORLD, &status);
                cnt += buffer;
            }
	}

	num_of_process /= 2;
	tree_level++;
	MPI_Barrier(MPI_COMM_WORLD);
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
