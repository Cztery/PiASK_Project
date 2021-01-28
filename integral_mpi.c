#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


void get_full_domain(float **full_dom, int *full_dom_len, float DOM_MIN, float DOM_MAX, float DOM_STEP) {
  *full_dom_len = (int)((DOM_MAX - DOM_MIN) / DOM_STEP);
  *full_dom = (float*)calloc( *full_dom_len, sizeof(float) );
  int i;
  for (i = 0; i < *full_dom_len; ++i) {
    (*full_dom)[i] = DOM_MIN + i * DOM_STEP;
  }
}

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

double mpi_integrate (float* coefficients, unsigned int polynomialDegree, float dom_min, float dom_max, float precision, float* result) {
  int myid, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  float sub_sum, global_sum;
 
  float* domain;
  int domain_len;
  if (myid == 0) {
    get_full_domain(&domain, &domain_len, dom_min, dom_max, precision);
  }
  MPI_Bcast(&domain_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int subdomain_len = (domain_len + size - 1) / size;
  float* subdomain = (float*)calloc(subdomain_len, sizeof(float));
  MPI_Scatter(domain, subdomain_len, MPI_FLOAT, subdomain, subdomain_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // subdomain may contain fewer values than subdomain_len - :q
  int loop_limit = subdomain_len;
  if (myid == size - 1) {
    loop_limit = subdomain_len-(size*subdomain_len-domain_len);
  }
  for (int i = myid; i < loop_limit; ++i) {
    // sum function values for the whole subdomain
    sub_sum += function(subdomain[i], coefficients, polynomialDegree);
  }
  
  // sum function results for all subdomains
  MPI_Reduce(&sub_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  free(subdomain);
  if (myid == 0) {
    // calculate final integral using formula: I = Precision * (Y[min]/2 + Y[min+1] + ... Y[max-1] + Y[max]/2)
    global_sum -= function(domain[0], coefficients, polynomialDegree)/2.;
    global_sum -= function(domain[domain_len-1], coefficients, polynomialDegree)/2.;
    *result = global_sum*precision;
    free(domain);
  }
}

int main(int argc, char** argv) {
  MPI_Init( &argc, &argv);

  float result1, result2;
  float dom_min = 0.0;
  float dom_max = 4.0;

  const unsigned int polynomial1Size = 2;
  float polynomial1[3];
  polynomial1[0] = 1.25;
  polynomial1[1] = 2.5;
  polynomial1[2] = 1.0;

  const unsigned int polynomial2Size = 5;
  float polynomial2[6];
  polynomial2[0] = 3.1;
  polynomial2[1] = 2.5;
  polynomial2[2] = 1.3;
  polynomial2[3] = 10.1;
  polynomial2[4] = 54.0;
  polynomial2[5] = 1.25;
 
  int myid, size; 
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  printf("my id = %d of %d\n", myid, size);

  float prec = 0.00005;
  mpi_integrate(polynomial1, polynomial1Size, dom_min, dom_max, prec, &result1);
  mpi_integrate(polynomial2, polynomial2Size, dom_min, dom_max, prec, &result2);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
  printf("Precision = %f:\n", prec);
  printf(" Function 1 result: %f\n", result1);
  printf(" Function 2 result: %f\n", result2);
  }
  
  prec = 0.001;
  mpi_integrate(polynomial1, polynomial1Size, dom_min, dom_max, prec, &result1);
  mpi_integrate(polynomial2, polynomial2Size, dom_min, dom_max, prec, &result2);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
  printf("Precision = %f:\n", prec);
  printf(" Function 1 result: %f\n", result1);
  printf(" Function 2 result: %f\n", result2);
  }
  
  
  MPI_Finalize();
  return 0;
}
