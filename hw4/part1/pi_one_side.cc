#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

typedef long long int ll;

// Compares schedule and oldschedule and prints schedule if different
// Also displays the time in seconds since the first invocation
int fnz (ll *schedule, ll *oldschedule, int size)
{
    static double starttime = -1.0;
    int diff = 0;

    for (int i = 0; i < size; i++)
       diff |= (schedule[i] != oldschedule[i]);

    if (diff)
    {
       int res = 0;

       if (starttime < 0.0) starttime = MPI_Wtime();

       printf("[%6.3f] Schedule:", MPI_Wtime() - starttime);
       for (int i = 0; i < size; i++)
       {
          printf("\t%lld", schedule[i]);
          res += schedule[i];
          oldschedule[i] = schedule[i];
       }
       printf("\n");

       return(res == size-1);
    }
    return 0;
}

int fnz2 (ll *schedule, ll *oldschedule, int size)
{
    int diff = 0;

    for (int i = 1; i < size; i++){
	if (schedule[i] != oldschedule[i]){
	    diff++;
	}
	oldschedule[i] = schedule[i];
    }

    return (diff == 0);
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

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

    if (world_rank == 0)
    {
       // Master
	
       ll *oldschedule = (ll*)malloc(world_size * sizeof(ll));
       // Use MPI to allocate memory for the target window
       ll *schedule;
       MPI_Alloc_mem(world_size * sizeof(ll), MPI_INFO_NULL, &schedule);

       for (int i = 0; i < world_size; i++)
       {
          schedule[i] = 0;
          oldschedule[i] = -1;
       }

       // Create a window. Set the displacement unit to sizeof(int) to simplify
       // the addressing at the originator processes
       MPI_Win_create(schedule, world_size * sizeof(ll), sizeof(ll), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       int ready = 0;
       while (!ready)
       {
          // Without the lock/unlock schedule stays forever filled with 0s
          MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
          ready = fnz2(schedule, oldschedule, world_size);
          MPI_Win_unlock(0, win);
       }
       printf("All workers checked in using RMA\n");

       total_cnt = cnt;
       for(int i = 1; i < world_size; i++){
	    total_cnt += schedule[i];
       }

       // Release the window
       MPI_Win_free(&win);
       // Free the allocated memory
       MPI_Free_mem(schedule);
       free(oldschedule);

       printf("Master done\n");
    }
    else
    {
        // Workers
	
	//int one = 1;

        // Worker processes do not expose memory in the window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Register with the master
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&cnt, 1, MPI_UNSIGNED_LONG, 0, world_rank, 1, MPI_UNSIGNED_LONG, win);
        MPI_Win_unlock(0, win);

        printf("Worker %d finished RMA\n", world_rank);

        // Release the window
        MPI_Win_free(&win);

        printf("Worker %d done\n", world_rank);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result

	pi_result = 4 * total_cnt / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
