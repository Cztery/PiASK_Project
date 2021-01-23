#include <math.h>
#include <stdio.h>

__device__ float myPower(float* number, int degree) {
  float result = 1.0;
  int fraction = 0;


  if(degree == 0) {
    return result;
  } else if( degree < 0) {
    degree = degree * (-1);
    fraction = 1;
  }

  for(int i = 1; i <= degree; i++) {
    result *= (*number);
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
__device__ float function(float* x, float* coefficients, unsigned int polynomialDegree) {
   unsigned int polynomialItertor = 0;
   float functionResult = 0;
   float tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= polynomialDegree; polynomialItertor++) {
     tmpCalc = coefficients[polynomialItertor] * myPower(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__host__ float functionHost(float x, float* coefficients, unsigned int polynomialDegree) {
   unsigned int polynomialItertor = 0;
   float functionResult = 0;
   float tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= polynomialDegree; polynomialItertor++) {
     tmpCalc = coefficients[polynomialItertor] * pow(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

__global__ void numericalIntegrationArray(float* coefficients, unsigned int polynomialDegree, float* xArray_device, float* yArray_device, int N) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if( i < N) {
    yArray_device[i] = function(&xArray_device[i],coefficients,polynomialDegree);
  }
}

int GPU_Integration(float* coefficients, unsigned int polynomialDegree, float low, float high, float precision, float *result, int nThx) {
   int numberOfPoints = (int) (high-low) / precision;
   int sizeOfArray = sizeof(float)*numberOfPoints;
   float *array, *xArray_device, *yArray_device;
   cudaEvent_t start, stop;

   array = (float*) malloc(sizeOfArray);
   for(int i = 0; i < numberOfPoints; i++) {
     array[i] = low+i*precision;
   }
   cudaMalloc((void**)&xArray_device, sizeOfArray);
   cudaMemcpy(xArray_device, array, sizeOfArray,cudaMemcpyHostToDevice);

   cudaMalloc((void**)&yArray_device, sizeOfArray);

   float* coefficients_d;
   cudaMalloc((void**) &coefficients_d, sizeof(float)*(polynomialDegree+1));
   cudaMemcpy(coefficients_d, coefficients, sizeof(float)*(polynomialDegree+1), cudaMemcpyHostToDevice);

   int nBLK = (int)(numberOfPoints+nThx-1)/nThx;

   printf("    GPU integral with parameter : \n");
   printf("    Number of blocks: %d\n", nBLK);
   printf("    Number of thread per block: %d\n", nThx);
   printf("    Precision of integral calculation %f\n", precision);

   float time;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   numericalIntegrationArray<<<nBLK,nThx>>>(coefficients_d,polynomialDegree,xArray_device,yArray_device,sizeOfArray);
   cudaDeviceSynchronize();

   //for (int n = ceil((float)sizeOfArray/2.); n > 2; n = ceil((float)sizeOfArray/2.)) {
   //}

printf("sizeofArray = %d", sizeOfArray);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   printf("  GPU time is %f ms\n", time);

   cudaMemcpy(array, yArray_device, sizeOfArray, cudaMemcpyDeviceToHost);
   cudaFree(coefficients_d);
   cudaFree(xArray_device);
   cudaFree(yArray_device);


   for(int j = 0; j < numberOfPoints ; j++  ) {
     *result += array[j];
   }

   *result *= precision;

   free(array);

   return 0;
}

int main(void) {
  const unsigned int polynomial1Size = 2;
  const unsigned int polynomial2Size = 5;
  float polynomial1[3];

  polynomial1[0] = 1.25;
  polynomial1[1] = 2.5;
  polynomial1[2] = 1.0;

  float polynomial2[6];
  polynomial2[0] = 3.1;
  polynomial2[1] = 2.5;
  polynomial2[2] = 1.3;
  polynomial2[3] = 10.1;
  polynomial2[4] = 54.0;
  polynomial2[5] = 1.25;

  float lowData = 0.0;
  float highData = 4.0;
  float prec1 = 0.001;
  float prec2 = 0.0001;

  float result1 = 0.0;
  float result2 = 0.0;
  float result3 = 0.0;
  float result4 = 0.0;

  float result11 = 0.0;
  float result21 = 0.0;
  float result31 = 0.0;
  float result41 = 0.0;

  int nThx = 128;
  int nThx1 = 256;

  printf("Function 1:");
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec1, &result1, nThx);
  printf("  Result %f\n", result1);

  printf("Function 1:");
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec1, &result2, nThx1);
  printf("  Result %f\n", result2);

  printf("Function 1:");
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec2, &result3, nThx);
  printf("  Result %f\n", result3);

  printf("Function 1:");
  GPU_Integration(polynomial1, polynomial1Size, lowData, highData, prec2, &result4, nThx1);
  printf("  Result %f\n", result4);


  printf("Function 2:");
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec1, &result11, nThx);
  printf("  Result %f\n", result11);

  printf("Function 2:");
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec1, &result21, nThx1);
  printf("  Result %f\n", result21);

  printf("Function 2:");
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec2, &result31, nThx);
  printf("  Result %f\n", result31);

  printf("Function 2:");
  GPU_Integration(polynomial2, polynomial2Size, lowData, highData, prec2, &result41, nThx1);
  printf("  Result %f\n", result41);
}
