#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#define THREAD_NUM 4
#define ERR 1e-3
#define DIFF(x,y) ((x-y)<0? y-x : x-y)
#define FPNEQ(x,y) (DIFF(x,y)>ERR ? 1 : 0)
int test(int N, float * sol, float * p, float * ax, float * ay) {
  int i;
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i],p[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+N],ax[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+2 * N], ay[i]))
      return 0;
  }
  return 1;
}

int main(int argc, char** argv) {
  // Initialize
  int pow = (argc > 1)? atoi(argv[1]) : 14;
  int N = 1 << pow;
  int i, j;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  float* x =  (float*)malloc(N * sizeof(float));
  float* y =  (float*)malloc(N * sizeof(float));
  float* m =  (float*)malloc(N * sizeof(float));
  float* p =  (float*)malloc(N * sizeof(float));
  float* ax = (float*)malloc(N * sizeof(float));
  float* ay = (float*)malloc(N * sizeof(float));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48() / N;
    p[i] = ax[i] = ay[i] =  0;
  }

  printf("Running for problem size N: %d\n", N);

  //Timers
  double ts, tf;

  //Serial version 
  printf("Running serial......................................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    for (j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }
  tf = omp_get_wtime();
  printf("Time: %.4lfs\n", tf - ts);

  //Copying solution for correctness check
  float* sol = (float*)malloc(3 * N * sizeof(float));
  memcpy(sol, p, N * sizeof(float));
  memcpy(sol + N, ax, N * sizeof(float));
  memcpy(sol+ 2 * N, ay, N * sizeof(float));


  //TODO: SPMD - Question 1 - Parallelize the outer loop 

  printf("Running parallel (outer loop).......................\n");
  ts = omp_get_wtime();
  #pragma omp parallel num_threads(THREAD_NUM)
  {
    int thread_count = omp_get_num_threads();
    int partition_size = N / thread_count;
    int thread_id = omp_get_thread_num();
    int start = thread_id * partition_size;
    int end = (thread_id + 1 ==thread_count)? N: start+partition_size;
    for (int i=start; i<end; i++) {            //FIXME: Parallelize
      float pi = 0;
      float axi = 0;
      float ayi = 0;
      float xi = x[i];
      float yi = y[i];
      for (j=0; j<N; j++) {
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float R2 = dx * dx + dy * dy + EPS2;
        float invR = 1.0f / sqrtf(R2);
        float invR3 = m[j] * invR * invR * invR;
        pi += m[j] * invR;
        axi += dx * invR3;
        ayi += dy * invR3;
      }
      p[i] = pi;
      ax[i] = axi;
      ay[i] = ayi;
    }
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  //TODO: SPMD - Question 2 - Parallelize the inner loop 

  printf("Running parallel (inner loop).......................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];

    float pii[THREAD_NUM];
    float axii[THREAD_NUM];
    float ayii[THREAD_NUM];

    #pragma omp parallel num_threads(THREAD_NUM)
    {
      int thread_count = omp_get_num_threads();
      int partition_size = N / thread_count;
      int thread_id = omp_get_thread_num();
      int start = thread_id * partition_size;
      int end = (thread_id + 1 ==thread_count)? N: start+partition_size;
      pii[thread_id] = 0;
      axii[thread_id] = 0;
      ayii[thread_id] = 0;
      for (int j=start; j<end; j++) {       //FIXME: Parallelize
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float R2 = dx * dx + dy * dy + EPS2;
        float invR = 1.0f / sqrtf(R2);
        float invR3 = m[j] * invR * invR * invR;
        pii[thread_id]  += m[j] * invR;
        axii[thread_id] += dx * invR3;
        ayii[thread_id] += dy * invR3;
      }
    }
    for (int m=0;m<THREAD_NUM;m++){
      pi+=pii[m];
      axi+=axii[m];
      ayi+=ayii[m];
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  //TODO: SPMD - Question 3 - Parallelize the inner loop and avoid false sharing

  printf("Running parallel (inner loop without false sharing).\n");
  ts = omp_get_wtime();
 
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];

    // to avoid false sharing, we should padding pii, axii, ayii to 64 Byte()
    float pii[THREAD_NUM  *(64/(sizeof(float)))];
    float axii[THREAD_NUM *(64/(sizeof(float)))];
    float ayii[THREAD_NUM *(64/(sizeof(float)))];

    #pragma omp parallel num_threads(THREAD_NUM)
    {
      int thread_count = omp_get_num_threads();
      int partition_size = N / thread_count;
      int thread_id = omp_get_thread_num();
      int start = thread_id * partition_size;
      int end = (thread_id + 1 ==thread_count)? N: start+partition_size;
      pii[thread_id*(64/(sizeof(float)))] = 0;
      axii[thread_id*(64/(sizeof(float)))] = 0;
      ayii[thread_id*(64/(sizeof(float)))] = 0;
      for (int j=start; j<end; j++) {       //FIXME: Parallelize
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float R2 = dx * dx + dy * dy + EPS2;
        float invR = 1.0f / sqrtf(R2);
        float invR3 = m[j] * invR * invR * invR;
        pii[thread_id*(64/(sizeof(float)))]  += m[j] * invR;
        axii[thread_id*(64/(sizeof(float)))] += dx * invR3;
        ayii[thread_id*(64/(sizeof(float)))] += dy * invR3;
      }
    }
    for (int m=0;m<THREAD_NUM;m++){
      pi+=pii[m*(64/(sizeof(float)))];
      axi+=axii[m*(64/(sizeof(float)))];
      ayi+=ayii[m*(64/(sizeof(float)))];
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  
  free(x);
  free(y);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(sol);
  return 0;
}
