#include "page_rank.h"

#include <bits/stdc++.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

using namespace std;

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }
  vector<double> new_sol(numNodes, equal_prob);

  bool converged = 0;
  vector<double> sol(numNodes, 0.0);

  while(!converged){
    double global_diff = 0.0;
    double no_outgoing = 0.0;
    sol = new_sol;

    #pragma omp parallel for reduction(+:no_outgoing)
    for(int i = 0; i < numNodes; i++){
        if(outgoing_size(g, i) == 0){
            no_outgoing += damping * sol[i] / numNodes;
        }
    }

    #pragma omp parallel for reduction(+:global_diff)
    for(int i = 0; i < numNodes; i++){
      const Vertex* in_begin = incoming_begin(g, i);
      const Vertex* in_end = incoming_end(g, i);
      double sum = 0.0;
      for(const Vertex* v = in_begin; v != in_end; v++){
          sum += sol[*v] / (double)outgoing_size(g, *v);
      }
      sum = (damping * sum) + (1.0 - damping) / numNodes;
      sum += no_outgoing;
      new_sol[i] = sum;
      global_diff += fabs(sum - sol[i]);
    }
    converged = global_diff < convergence;
  }
  
  #pragma omp parallel for
  for(int i = 0; i < numNodes; i++){
    solution[i] = new_sol[i];
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
