#include "PPintrin.h"
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
    // 確認剩餘的資料量
    int width = (i + VECTOR_WIDTH > N) ? (N - i) : VECTOR_WIDTH;

    // 根據剩餘資料量初始化mask
    maskAll = _pp_init_ones(width);
    
    // All ones
    // maskAll = _pp_init_ones();

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
  __pp_vec_float base;
  __pp_vec_int exp;
  __pp_vec_float result;
  __pp_vec_int count = _pp_vset_int(0);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float max = _pp_vset_float(9.999999f);
  __pp_vec_float tempBase = _pp_vset_float(0.f);
  __pp_vec_int tempExp = _pp_vset_int(1);
  __pp_mask maskAll, maskIsEqual, maskIsNotEqual, maskIsPositive, maskIsLarge;

  int remainder = N % VECTOR_WIDTH;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsEqual = _pp_init_ones(0);

        // Handle remainder for non-multiple of VECTOR_WIDTH
    if ((i + VECTOR_WIDTH) > N)
    {
      for (int j = 0; j < remainder; j++)
      {
          tempBase.value[j] = values[i+j];
          tempExp.value[j] = exponents[i+j];
      }

      // Load the remainder elements into vectors
    	_pp_vmove_float(base, tempBase, maskAll); 
    	_pp_vmove_int(exp, tempExp, maskAll); 
    }
    else{
      // Load contiguous elements into vectors
    	_pp_vload_float(base, values + i, maskAll); 
      _pp_vload_int(exp, exponents + i, maskAll);
    }

    // Mask lanes where exponent == 0
    _pp_veq_int(maskIsEqual, exp, zero, maskAll); // if (y == 0) {

    // Set result to 1 for lanes where exponent == 0
    _pp_vset_float(result, 1.f, maskIsEqual);

    maskIsNotEqual = _pp_mask_not(maskIsEqual);

    // Execute instruction ("else" clause)
    _pp_vmove_float(result, base, maskIsNotEqual); //     result = x;
    _pp_vsub_int(count, exp, one, maskIsNotEqual);//     count = y - 1;}
    maskIsPositive = _pp_init_ones(0);
    _pp_vgt_int(maskIsPositive, count, zero, maskIsNotEqual);


    while (_pp_cntbits(maskIsPositive))
    {
      _pp_vmult_float(result, result, base, maskIsPositive);	
      _pp_vsub_int(count, count, one, maskIsPositive);
      _pp_vgt_int(maskIsPositive, count, zero, maskIsPositive);
    }

    // Set mask according to predicate
    maskIsLarge = _pp_init_ones(0);
    _pp_vgt_float(maskIsLarge, result, max, maskIsNotEqual); 
    // Clamp result to 9.999999 if it exceeds this value
    _pp_vset_float(result, 9.999999f, maskIsLarge);  
    // Store the final result
    _pp_vstore_float(output + i, result, maskAll);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  float totalSum = 0.f;
  __pp_vec_float currentVals, haddVals, interleavedVals;
  __pp_mask activeMask;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
      // Initialize mask with all lanes active
      activeMask = _pp_init_ones();

      // Load values from array into vector
      _pp_vload_float(currentVals, values + i, activeMask);

      // Perform horizontal addition
      _pp_hadd_float(haddVals, currentVals);

      // Interleave even and odd lanes
      _pp_interleave_float(interleavedVals, haddVals);

      // Sum the first half of the interleaved result
      for (int j = 0; j < VECTOR_WIDTH / 2; j++)
      {
          totalSum += interleavedVals.value[j];
      }
  }

  return totalSum;
}