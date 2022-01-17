#include "PPintrin.h"
#include <bits/stdc++.h>

using namespace std;

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH){
    __pp_mask Mask = _pp_init_ones();
    __pp_mask upperMask = _pp_init_ones(0);
    __pp_mask expMask = _pp_init_ones(0);
    __pp_mask initial_exp0 = _pp_init_ones(0);

    __pp_vec<float> vec_base = _pp_vset_float(0);
    _pp_vload_float(vec_base, values + i, Mask);
    __pp_vec<float> vec_ans = _pp_vset_float(1.0f);

    __pp_vec<int> vec_ones = _pp_vset_int(1);
    __pp_vec<int> vec_exp = _pp_vset_int(0);
    _pp_vload_int(vec_exp, exponents + i, Mask);

    __pp_vec<int> vec_zeros = _pp_vset_int(0);
    __pp_vec<float> vec_upperbound = _pp_vset_float(9.999999f);

    _pp_veq_int(initial_exp0, vec_exp, vec_zeros, Mask);
    __pp_mask initial_not_0 = _pp_mask_not(initial_exp0);
    Mask = _pp_mask_and(Mask, initial_not_0);

    while(1){
      _pp_vmult_float(vec_ans, vec_ans, vec_base, Mask);
      _pp_vsub_int(vec_exp, vec_exp, vec_ones, Mask);
      _pp_veq_int(expMask, vec_exp, vec_zeros, Mask); // reach exp[i] = 0;

      _pp_vgt_float(upperMask, vec_ans, vec_upperbound, Mask); // ans[i] reach upperbound
      _pp_vmove_float(vec_ans, vec_upperbound, upperMask);
      
      __pp_mask terminate = _pp_mask_or(expMask, upperMask);
      __pp_mask notTerminate = _pp_mask_not(terminate);
      Mask = _pp_mask_and(Mask, notTerminate);

      if(_pp_cntbits(Mask) == 0){
    	  break;
      }
    }
    __pp_mask m = _pp_init_ones(N-i);
    _pp_vstore_float(output + i, vec_ans, m);
  }
}

void print_vec(__pp_vec<float> &vec){
  float buffer[VECTOR_WIDTH];
  __pp_mask m = _pp_init_ones();
  _pp_vstore_float(buffer, vec, m);

  for(int i=0;i<VECTOR_WIDTH;++i){
    cout << buffer[i] << " ";
  }
  cout << endl << endl;
}
// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float ans = 0.0;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_mask Mask = _pp_init_ones();
    __pp_vec<float> vec = _pp_vset_float(0);
    _pp_vload_float(vec, values + i, Mask);
    //print_vec(vec);

    int tmp = VECTOR_WIDTH;
    while (tmp != 1){
      _pp_hadd_float(vec, vec);
      _pp_interleave_float(vec, vec);
      tmp /= 2;
      //print_vec(vec);
    }

    float buffer[VECTOR_WIDTH];
    __pp_mask m = _pp_init_ones();
    _pp_vstore_float(buffer, vec, m);
    ans += buffer[0];
  }

  return ans;
}
