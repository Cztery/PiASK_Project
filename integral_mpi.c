#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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

double integral (float x, float* coefficients, unsigned int polynomialDegree, float prec, int numberOfPoints) {
  int j;
  double result = 0.0;
  double offs = prec/2;

  for(j=0; j< numberOfPoints; j++ ) {
    result += function((x+j*prec+offs),coefficients,polynomialDegree) * prec;
  }
}

int main(int argc, char** argv) {

  MPI_Status status;
  double result, tmp_result;
  int myid, p;
  double tmp_range;
  float x, tmpNoP;

  float prec = 0.001;

  int numberOfPoints = (HIGH_VALUE - LOW_VALUE) / prec;

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

  MPI_Init(&argc,&argv);              /* starts MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);  /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &p);     /* get number of processes */

  tmp_range = (HIGH_VALUE-LOW_VALUE)/p;
  x = LOW_VALUE + myid*tmp_range;
  tmp_result = integral();


  MPI_Finalize();
  return 0;
}
