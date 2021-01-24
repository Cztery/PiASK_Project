#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include"my_timers.h"


#define LOW_VALUE 0.0
#define HIGH_VALUE  4.0

double function(float x, float* coefficients, unsigned int polynomialDegree) {
   unsigned int polynomialItertor = 0;
   float functionResult = 0;
   float tmpCalc;
   for(polynomialItertor = 0; polynomialItertor <= polynomialDegree; polynomialItertor++) {
     tmpCalc = coefficients[polynomialItertor] * pow(x,polynomialItertor);
     functionResult += tmpCalc;
   }
   return functionResult;
}

double calculateIntegration(float* coefficients, unsigned int polynomialDegree, double prec, int numberOfThreads) {
  int numberOfPoints = (HIGH_VALUE - LOW_VALUE) / prec;

  /* compute the estimate */
  double sum = 0;
  long i = 0;
  start_time();
  #pragma omp parallel for num_threads(numberOfThreads) \
  shared (numberOfThreads, prec) private(i) reduction(+: sum)
  for (i = 0; i < numberOfPoints; i++) {
       sum += function((LOW_VALUE + (i+0.5)*prec), coefficients, polynomialDegree) * prec;
  }
  stop_time();
  printf("With precision = %f  and %d threads, ", prec, numberOfThreads);
  printf("integrated value is %.20g\n", sum);
  print_time("Elapsed:");

  return sum;
}

int main(void) {
  double low = 0.0, high = 4.0;  /* lower and upper interval endpoints */
  double prec1 = 0.001;
  double prec2 = 0.0001;

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

  int threadct1 = 1;
  int threadct2 = 2;
  int threadct3 = 4;
  int threadct4 = 8;
  int threadct5 = 16;

  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec1, threadct2 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct2 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct1 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct2 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct3 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct4 );
  printf("Function 1: \n");
  calculateIntegration(polynomial1, polynomial1Size, prec2, threadct5 );

  printf("Function 2: \n");
  calculateIntegration(polynomial2, polynomial2Size, prec1, threadct1 );

  printf("Function 2: \n");
  calculateIntegration(polynomial2, polynomial2Size, prec1, threadct2 );

  printf("Function 2: \n");
  calculateIntegration(polynomial2, polynomial2Size, prec1, threadct3 );

  printf("Function 2: \n");
  calculateIntegration(polynomial2, polynomial2Size, prec1, threadct4 );

  printf("Function 2: \n");
  calculateIntegration(polynomial2, polynomial2Size, prec1, threadct5 );
}
