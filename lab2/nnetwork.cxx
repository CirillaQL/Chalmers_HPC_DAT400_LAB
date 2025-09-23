//
//  nn_mpi.cpp
//  
//  To compile: mpicxx -std=c++11 -O3 -fopenmp -o train_mpi
//  To run: ./train_mpi



#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <sys/time.h>
#include "deep_core.h"
#include "vector_ops.h"



vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> tokens;
  while (getline(ss, item, delim)) {
    tokens.push_back(item);
  }
  return tokens;
}

int main(int argc, char * argv[]) {

  string line;
  vector<string> line_v;
  int len, mpirank = 0;
  cout << "Loading data ...\n";
  vector<float> X_train;
  vector<float> y_train;
  ifstream myfile ("train.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      line_v = split(line, '\t');
      int digit = strtof((line_v[0]).c_str(),0);
      for (unsigned i = 0; i < 10; ++i) {
        if (i == digit)
        {
          y_train.push_back(1.);
        }
        else y_train.push_back(0.);
      }
      
      int size = static_cast<int>(line_v.size());
      for (unsigned i = 1; i < size; ++i) {
        X_train.push_back(strtof((line_v[i]).c_str(),0));
      }
    }
    X_train = X_train/255.0;
    myfile.close();
  }
  
  else cout << "Unable to open file" << '\n';
  
  int xsize = static_cast<int>(X_train.size());
  int ysize = static_cast<int>(y_train.size());
  
  // Some hyperparameters for the NN
  int BATCH_SIZE = 256;
  float lr = .01/BATCH_SIZE;
  // Random initialization of the weights
  vector <float> W1 = random_vector(784*128);
  vector <float> W2 = random_vector(128*64);
  vector <float> W3 = random_vector(64*10);
  
  struct timeval program_start, t_before_dot, t_after_dot;
  double total_dot_time = 0.0;

  gettimeofday(&program_start, 0);

  std::chrono::time_point<std::chrono::system_clock> t1,t2;
  cout << "Training the model ...\n";
  for (unsigned i = 0; i < 10000; ++i) {
    t1 = std::chrono::system_clock::now();    
    // Building batches of input variables (X) and labels (y)
    int randindx = rand() % (42000-BATCH_SIZE);
    vector<float> b_X;
    vector<float> b_y;
    for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
      b_X.push_back(X_train[j]);
    }
    for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
      b_y.push_back(y_train[k]);
    }

    // Feed forward
    gettimeofday(&t_before_dot, 0);
    vector<float> a1 = relu(dot( b_X, W1, BATCH_SIZE, 784, 128 ));
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    gettimeofday(&t_before_dot, 0);
    vector<float> a2 = relu(dot( a1, W2, BATCH_SIZE, 128, 64 ));
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    gettimeofday(&t_before_dot, 0);
    vector<float> yhat = softmax(dot( a2, W3, BATCH_SIZE, 64, 10 ), 10);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);
    
    // Back propagation
    vector<float> dyhat = (yhat - b_y);
    // dW3 = a2.T * dyhat
    gettimeofday(&t_before_dot, 0);
    vector<float> dW3 = dot(transform( &a2[0], BATCH_SIZE, 64 ), dyhat, 64, BATCH_SIZE, 10);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    // dz2 = dyhat * W3.T * relu'(a2)
    gettimeofday(&t_before_dot, 0);
    vector<float> dz2 = dot(dyhat, transform( &W3[0], 64, 10 ), BATCH_SIZE, 10, 64) * reluPrime(a2);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    // dW2 = a1.T * dz2
    gettimeofday(&t_before_dot, 0);
    vector<float> dW2 = dot(transform( &a1[0], BATCH_SIZE, 128 ), dz2, 128, BATCH_SIZE, 64);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    // dz1 = dz2 * W2.T * relu'(a1)
    gettimeofday(&t_before_dot, 0);
    vector<float> dz1 = dot(dz2, transform( &W2[0], 128, 64 ), BATCH_SIZE, 64, 128) * reluPrime(a1);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);

    // dW1 = X.T * dz1
    gettimeofday(&t_before_dot, 0);
    vector<float> dW1 = dot(transform( &b_X[0], BATCH_SIZE, 784 ), dz1, 784, BATCH_SIZE, 128);
    gettimeofday(&t_after_dot, 0);
    total_dot_time += (t_after_dot.tv_sec - t_before_dot.tv_sec) * 1000000.0 + (t_after_dot.tv_usec - t_before_dot.tv_usec);
    

    // Updating the parameters
    W3 = W3 - lr * dW3;
    W2 = W2 - lr * dW2;
    W1 = W1 - lr * dW1;
             
    if ((mpirank == 0) && (i+1) % 100 == 0){
      struct timeval current_time;
      gettimeofday(&current_time, 0);
      double total_program_time = (current_time.tv_sec - program_start.tv_sec) * 1000000.0 + (current_time.tv_usec - program_start.tv_usec);
      double percentage = (total_dot_time / total_program_time) * 100.0;

      cout << "Predictions:" << "\n";
      print ( yhat, 10, 10 );
      cout << "Ground truth:" << "\n";
      print ( b_y, 10, 10 );
      vector<float> loss_m = yhat - b_y;
      float loss = 0.0;
      for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
        loss += loss_m[k]*loss_m[k];
      }
      t2 = std::chrono::system_clock::now();
      chrono::duration<double> elapsed_seconds = t2-t1;
      double ticks = elapsed_seconds.count();
      cout << "Iteration #: "  << i << endl;
      cout << "Iteration Time: "  << ticks << "s" << endl;
      cout << "Loss: " << loss/BATCH_SIZE << endl;
      cout << "Total time in dot(): " << total_dot_time / 1000000.0 << " seconds" << endl;
      cout << "Total program time: " << total_program_time / 1000000.0 << " seconds" << endl;
      cout << "Percentage of time spent in dot(): " << percentage << "%" << endl;

      // Amdahl's Law Analysis
      double f = percentage / 100.0;  // Parallel fraction
      int P = 4;  // M4 performance cores
      double max_speedup = 1.0 / ((1 - f) + f / P);
      cout << "Parallel fraction (f): " << f << endl;
      cout << "Theoretical max speedup (P=" << P << "): " << max_speedup << "x" << endl;
      cout << "*******************************************" << endl;
    };      
  };
  
  return 0;
}
