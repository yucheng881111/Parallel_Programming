#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

// use code from L5_MPI2.pdf page 61 to 65

typedef long long ll;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    int numworkers = numtasks - 1;
    int *ptr;

    // read input from stdin in MASTER
    if(taskid == MASTER){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        int NRA = *n_ptr;
       	int NCA = *m_ptr;
       	int NCB = *l_ptr;
        int *a = (int*)malloc(sizeof(int) * NRA * NCA);
	int *b = (int*)malloc(sizeof(int) * NCA * NCB);
	int x;

        for(int i=0;i<NRA;++i){
            for(int j=0;j<NCA;++j){ 
                scanf("%d", &x);
		a[i * NCA + j] = x;
            }
        }
	for(int i=0;i<NCA;++i){
            for(int j=0;j<NCB;++j){
                scanf("%d", &x);
                b[i * NCB + j] = x;
            }
        }

	*a_mat_ptr = a;
	*b_mat_ptr = b;
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Status status;
    int numworkers = numtasks - 1;
    int averow, extra, offset;
    int rows;
    int NRA = n;
    int NCA = m;
    int NCB = l;
    int N, M, L;
    //int *c = (int*)malloc(sizeof(int) * NRA * NCB);
    int mtype;

    if(taskid == MASTER){
	// send matrix data to worker tasks
	int *c = (int*)malloc(sizeof(int) * NRA * NCB);
        averow = n / numworkers;
	extra = n % numworkers;
	offset = 0;
	mtype = FROM_MASTER;
	for(int dest=1;dest<=numworkers;++dest){

	    MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);

	    rows = (dest <= extra)? averow + 1: averow;
	    // send partial a
	    MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	    MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	    MPI_Send(&a_mat[offset * NCA], rows * NCA, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	    // send all b
	    MPI_Send(&b_mat[0], NCA * NCB, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	    offset += rows;
	}
	
	// receive results from worker tasks
	
	mtype = FROM_WORKER;
	for(int i=1;i<=numworkers;++i){
	    int source = i;
	    MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	    MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	    MPI_Recv(&c[offset * NCB], rows * NCB, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	}

	// print results
	
	for(int i=0;i<NRA;++i){
	    for(int j=0;j<NCB;++j){
	        printf("%d ", c[i * NCB + j]);
	    }
	    printf("\n");
	}
	free(c);

    }else{
	mtype = FROM_MASTER;

	MPI_Recv(&N, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

	int *a = (int*)malloc(sizeof(int) * N * M);
	int *b = (int*)malloc(sizeof(int) * M * L);
	int *c = (int*)malloc(sizeof(int) * N * L);
	MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	MPI_Recv(&a[0], rows * M, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], M * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

	for(int k=0;k<L;++k){
	    for(int i=0;i<rows;++i){
		c[i * L + k] = 0;
	        for(int j=0;j<M;++j){
		    c[i * L + k] += (a[i * M + j] * b[j * L + k]);
		}
	    }
	}
	
	// send result back to MASTER
	mtype = FROM_WORKER;
	MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

    //MPI_Finalize();
}


void destruct_matrices(int *a_mat, int *b_mat){
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    if(taskid == MASTER){
        free(a_mat);
        free(b_mat);
    }
}








