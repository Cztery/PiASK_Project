double function(const double x) {
  return sin(x);
}

double trapezoidalMethod_OMP(const double x1, double x2, double dx, const int nThreads)
{
  const int N = (int)((x2 - x1) / dx);
  double now = omp_get_wtime();
  double s = 0;

  #pragma omp parallel for num_threads(nThreads) reduction(+: s)
  for (int i = 1; i < N; i++) {
    s += f(x1 + i * dx);
  }

  s = (s + (function(x1) + function(x2)) / 2) * dx;
  now = omp_get_wtime() - now;
  return s;
}

int main(void) {
  const uint8_t maxThreads = 10;
  short method;
  double x1, x2, dx;

  while (true) {

    list<pair<short, Result>> results;
    for (int i = 0; i < maxThreads; i++) {
        Result result = (method == 1) ?
        rectangleMethod(x1, x2, dx, i + 1) :
        trapezoidalMethod(x1, x2, dx, i + 1);

        pair<short, Result> s_result(i + 1, result);
        results.push_back(s_result);
      }

      for (auto & result : results) {
        cout << "Threads: " << result.first;
        cout << ", timestamp: " << result.second.timestamp;
        cout << ", area: " << result.second.area << endl;
      }
      cout << endl;
    }
  }
  return 0;
}
