#include <math.h>
#include <stdio.h>

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
__device__ double function(float x, float* coefficients, unsigned int polynomialDegree) {
   unsigned int polynomialItertor = 0;
   double functionResult = 0;
   double tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= polynomialDegree; polynomialItertor++) {
     printf("   %f\n", coefficients[polynomialItertor]);
     tmpCalc = coefficients[polynomialItertor] * myPower(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__host__ double functionHost(float x, float* coefficients, unsigned int polynomialDegree) {
   unsigned int polynomialItertor = 0;
   double functionResult = 0;
   double tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= polynomialDegree; polynomialItertor++) {
     tmpCalc = coefficients[polynomialItertor] * pow(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__global__ void numericalIntegration(float* coefficients, unsigned int polynomialDegree, int N, float offs, float eps, float *result) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  float xl, xh, fl, fh, flh;
  if (i==0) {
    *result = 0.0;
  }
  if (i<N) {
    xl = offs + i*eps;
    xh = xl + eps;
    fl = function(xl,coefficients,polynomialDegree);
    fh = function(xh,coefficients,polynomialDegree);
    flh = 0.5*(fl+fh);
    atomicAdd( result, flh);
  }

}

int GPU_Integration(float* coefficients, unsigned int polynomialDegree, float lo, float hi, float prec, float *result, int *pn, int nBlk, int nThx) {
  float *result_d, del, flo, fhi;
  int n;
  cudaEvent_t start, stop;

  float* coefficients_d;
  cudaMalloc((void**) &coefficients_d, sizeof(float)*(polynomialDegree+1));
  cudaMemcpy(coefficients_d, coefficients, sizeof(float)*(polynomialDegree+1), cudaMemcpyHostToDevice);

  flo = functionHost(lo, coefficients, polynomialDegree);
  fhi = functionHost(hi, coefficients, polynomialDegree);

  n = abs(0.5*(hi-lo)*(flo-fhi)/prec);
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

  numericalIntegration<<<nBlk,nThx>>>(coefficients_d,polynomialDegree,n,lo,del,result_d);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("  GPU time is %f ms\n", time);

  cudaMemcpy(result, result_d, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(result_d);
  cudaFree(coefficients_d);


  *result = *result*del;
  return 0;
}

int main(void) {
  const unsigned int polynomial1Size = 3;
  const unsigned int polynomial2Size = 6;
  float polynomial1[3];

  polynomial1[0] = 1.0;
  polynomial1[1] = 2.5;
  polynomial1[2] = 1.25;

  float polynomial2[6];
  polynomial2[0] = 3.1;
  polynomial2[1] = 2.5;
  polynomial2[2] = 1.3;
  polynomial2[3] = 10.1;
  polynomial2[4] = 54.0;
  polynomial2[5] = 1.25;

  float lowData = 0.0;
  float highData = 2.0;
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

  int nBlk = 256;
  int nThx = 256;
  int nBlk1 = 512;
  int nThx1 = 512;

  int pn;

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec1, &result1, &pn, nBlk, nThx);
  printf("  Result %f\n", result1);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec1, &result2, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result2);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec2, &result3, &pn, nBlk, nThx);
  printf("  Result %f\n", result3);

  printf("Function 1 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec2, &result4, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result4);


  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec1, &result11, &pn, nBlk, nThx);
  printf("  Result %f\n", result11);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec1, &result21, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result21);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk, nThx);
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec2, &result31, &pn, nBlk, nThx);
  printf("  Result %f\n", result31);

  printf("Function 2 nBlk: %d, nThx: %d \n", nBlk1, nThx1);
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec2, &result41, &pn, nBlk1, nThx1);
  printf("  Result %f\n", result41);
}
