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


void clampedExpVector(float* values, int* exponents, float* output, int N)
{
    // 先準備會重複用到的常數向量
    __pp_vec_float vLimit = _pp_vset_float(9.999999f);
    __pp_vec_float vOneF  = _pp_vset_float(1.0f);
    __pp_vec_int   vZeroI = _pp_vset_int(0);
    __pp_vec_int   vOneI  = _pp_vset_int(1);

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // 1) 建立這一批有效 lane 的 active mask（處理尾端不足一組）
        int remain = N - i;
        __pp_mask mAll = (remain >= VECTOR_WIDTH)
                       ? _pp_init_ones()
                       : _pp_init_ones(remain);

        // 2) 向量載入 values/exponents
        __pp_vec_float vX;
        __pp_vec_int   vY;
        _pp_vload_float(vX, values + i, mAll);
        _pp_vload_int  (vY, exponents + i, mAll);

        // 3) 冪次初始化：res = 1；cnt = y
        __pp_vec_float vRes;
        __pp_vec_int   vCnt;
        _pp_vset_float(vRes, 1.0f, mAll);     // 只在有效 lane 設 1.0
        _pp_vmove_int (vCnt, vY, mAll);       // vCnt = vY（只寫有效 lane）

        // 4) while (cnt > 0) { res *= x; cnt--; } 以 mask 控制仍需運算的 lane
        __pp_mask mCntPos;
        _pp_vgt_int(mCntPos, vCnt, vZeroI, mAll);  // mCntPos = (cnt > 0) & mAll
        while (_pp_cntbits(mCntPos) > 0) {
            // res *= x   （僅在 mCntPos 的 lane 進行）
            _pp_vmult_float(vRes, vRes, vX, mCntPos);
            // cnt--
            _pp_vsub_int  (vCnt, vCnt, vOneI, mCntPos);
            // 更新：還有哪些 lane 的 cnt>0
            _pp_vgt_int(mCntPos, vCnt, vZeroI, mAll);
        }

        // 5) clamp：若 res > 9.999999，覆寫成 9.999999（只在超標的 lane）
        __pp_mask mClamp;
        _pp_vgt_float(mClamp, vRes, vLimit, mAll);     // mClamp = (vRes > vLimit) & mAll
        _pp_vmove_float(vRes, vLimit, mClamp);

        // 6) 存回輸出（僅寫有效 lane）
        _pp_vstore_float(output + i, vRes, mAll);
    }
}




// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}