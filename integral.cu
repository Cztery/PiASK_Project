#include <math.h>
#include <stdio.h>
/**
 * @struct FunctionData_t
 * @brief Data creating polynomial function - degree and array of coefficients
 */
typedef struct FunctionData {
  unsigned int polynomialDegree;  /**<Degree of polynomial*/
  float* coefficients;   /**<Polynomial's coefficients*/
} FunctionData_t;

__device__ double myPower(float number, int degree) {
  double result = 1.0;
  int fraction = 0;
  if(degree == 0) {
    return result;
  } else if( degree > 0) {
    degree = degree * (-1);
    fraction = 1;
  }

  for(int i = 1; i <= degree; degree++) {
    result *= number;
  }
  if(fraction == 0) {
    return result;
  } else  {
    return 1/result;
  }
}
/**
 * @brief Caclulate polynomial function from given data.
 * @param x Function input value
 * @param function Polynomial function data.
 */
__device__ double function(float x, FunctionData_t* functionInput) {
   unsigned int polynomialItertor = 0;
   double functionResult = 0;
   double tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= (functionInput->polynomialDegree); polynomialItertor++) {
     tmpCalc = functionInput->coefficients[polynomialItertor] * myPower(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__host__ double functionHost(float x, FunctionData_t* functionInput) {
   unsigned int polynomialItertor = 0;
   double functionResult = 0;
   double tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= (functionInput->polynomialDegree); polynomialItertor++) {
     tmpCalc = functionInput->coefficients[polynomialItertor] * pow(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__global__ void numericalIntegration(FunctionData_t* function_input, int N, float offs, float eps, float *result) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  float xl, xh, fl, fh, flh;
  if (i==0) {
    *result = 0.0;
  }
  if (i<N) {
    xl = offs + i*eps;
    xh = xl + eps;
    fl = function(xl,function_input);
    fh = function(xh,function_input);
    flh = 0.5*(fl+fh)*(xh-xl);
    atomicAdd( result, flh);
  }

}

int GPU_Integration(FunctionData_t* function_input, float lo, float hi, float prec, float *result, int *pn, int nBlk, int nThx) {
  float *result_d, del, flo, fhi;
  int n;
  cudaEvent_t start, stop;

  flo = functionHost(lo,function_input);
  fhi = functionHost(hi,function_input);

  n = 0.5*(hi-lo)*(flo-fhi)/prec;
  *pn = n;
  del = (hi-lo)/((double) n);
  cudaMalloc((void **) &result_d, sizeof(float));

  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  printf("    GPU integral with parameter : \n");
  printf("    Number of blocks: %d\n", nBlk);
  printf("    Number of thread per block: %d\n", nThx);
  printf("    Precision of integral calculation %f\n", prec);

  numericalIntegration<<<nBlk,nThx>>>(function_input,n,lo,del,result_d);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("  GPU time is %f ms\n", time);

  cudaMemcpy(result, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(result_d);
  return 0;
}

int main(void) {
  float polynomial1[3];

  polynomial1[0] = 1.0;
  polynomial1[1] = 2.5;
  polynomial1[2] = 1.25;

  FunctionData_t function1 = {
    .polynomialDegree = 2,
    .coefficients = polynomial1
  };

  float polynomial2[6];
  polynomial2[0] = 3.1;
  polynomial2[1] = 2.5;
  polynomial2[2] = 1.3;
  polynomial2[3] = 10.1;
  polynomial2[4] = 54.0;
  polynomial2[5] = 1.25;

  FunctionData_t function2 = {
    .polynomialDegree = 5,
    .coefficients = polynomial2
  };

  float lowData = 0.0;
  float highData = 10.0;
  float prec1 = 0.01;
  float prec2 = 0.001;

  float result1 = 0.0;
  float result2 = 0.0;
  float result3 = 0.0;
  float result4 = 0.0;

  float result11 = 0.0;
  float result21 = 0.0;
  float result31 = 0.0;
  float result41 = 0.0;

  int nBlk = 64;
  int nThx = 128;
  int nBlk1 = 128;
  int nThx1 = 256;

  int pn;

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(&function1, lowData, highData, prec1, &result1, &pn, nBlk, nThx);
  printf("  Result %f\n", result1);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(&function1, lowData, highData, prec1, &result2, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result2);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(&function1, lowData, highData, prec2, &result3, &pn, nBlk, nThx);
  printf("  Result %f\n", result3);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(&function1, lowData, highData, prec2, &result4, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result4);


  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(&function2, lowData, highData, prec1, &result11, &pn, nBlk, nThx);
  printf("  Result %f\n", result11);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(&function2, lowData, highData, prec1, &result21, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result21);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(&function2, lowData, highData, prec2, &result31, &pn, nBlk, nThx);
  printf("  Result %f\n", result31);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(&function2, lowData, highData, prec2, &result41, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result41);
}
