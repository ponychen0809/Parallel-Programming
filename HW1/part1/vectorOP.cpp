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

// void clampedExpVector(float *values, int *exponents, float *output, int N)
// {
//   //
//   // PP STUDENTS TODO: Implement your vectorized version of
//   // clampedExpSerial() here.
//   //
//   // Your solution should work for any value of
//   // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
//   //
//   for (int i = 0; i < N; i += VECTOR_WIDTH)
//   {
//   }
// }


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


// void clampedExpVector(float* values, int* exponents, float* output, int N)
// {
//     // 先準備會重複用到的常數向量
//     __pp_vec_float vLimit = _pp_vset_float(9.999999f);
//     __pp_vec_float vOneF  = _pp_vset_float(1.0f);
//     __pp_vec_int   vZeroI = _pp_vset_int(0);
//     __pp_vec_int   vOneI  = _pp_vset_int(1);

//     for (int i = 0; i < N; i += VECTOR_WIDTH) {
//         // 1) 建立這一批有效 lane 的 active mask（處理尾端不足一組）
//         int remain = N - i;
//         __pp_mask mAll = (remain >= VECTOR_WIDTH)
//                        ? _pp_init_ones()        // 如果剩餘的元素數量 >= 向量寬度，mask 全部是 1
//                        : _pp_init_ones(remain); // 否則只開啟前 "remain" 個有效 lane

//         // 2) 向量載入 values 和 exponents
//         __pp_vec_float vX;
//         __pp_vec_int   vY;
//         _pp_vload_float(vX, values + i, mAll);    // 只載入有效 lane
//         _pp_vload_int  (vY, exponents + i, mAll); // 只載入有效 lane

//         // 3) 冪次初始化：res = 1；cnt = exponent
//         __pp_vec_float vRes;
//         __pp_vec_int   vCnt;
//         _pp_vset_float(vRes, 1.0f, mAll);     // 只在有效 lane 設為 1.0
//         _pp_vmove_int (vCnt, vY, mAll);       // 將 exponent 複製給每個 lane

//         // 4) while (cnt > 0) { res *= x; cnt--; } 以 mask 控制仍需運算的 lane
//         __pp_mask mCntPos;
//         _pp_vgt_int(mCntPos, vCnt, vZeroI, mAll);  // mCntPos = (cnt > 0) & mAll，檢查每個 lane 是否還需要運算
//         while (_pp_cntbits(mCntPos) > 0) {
//             // 只對那些需要運算的 lane 做乘法：res *= x
//             _pp_vmult_float(vRes, vRes, vX, mCntPos);
//             // 遞減計數：cnt--
//             _pp_vsub_int(vCnt, vCnt, vOneI, mCntPos);
//             // 更新：哪些 lane 還需要運算
//             _pp_vgt_int(mCntPos, vCnt, vZeroI, mAll); // 重新判斷哪些 lane 的 cnt > 0
//         }

//         // 5) clamp：如果 res > 9.999999，將其設為 9.999999（只在超過的 lane 進行）
//         __pp_mask mClamp;
//         _pp_vgt_float(mClamp, vRes, vLimit, mAll);     // 產生超過 9.999999 的 mask
//         _pp_vmove_float(vRes, vLimit, mClamp);          // 將超過的 lane 設為 9.999999

//         // 6) 存回結果（只將有效的 lane 寫回）
//         _pp_vstore_float(output + i, vRes, mAll);
//     }
// }



// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
// float arraySumVector(float *values, int N)
// {

//   //
//   // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
//   //

//   for (int i = 0; i < N; i += VECTOR_WIDTH)
//   {
//   }

//   return 0.0;
// }


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


