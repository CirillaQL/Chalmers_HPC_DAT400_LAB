//
//  nn_mpi.cpp
//
//  To compile: mpicxx -std=c++11 -O3 -o train_mpi nnetwork.cxx src/vector_ops.cpp src/deep_core.cpp
//  To run: mpirun -np 4 ./train_mpi



#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <mpi.h>
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
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  auto program_start = std::chrono::system_clock::now();

  string line;
  vector<string> line_v;
  int xsize, ysize;

  if (mpirank == 0) cout << "Loading data ...\n";
  vector<float> X_train;
  vector<float> y_train;

  // Only rank 0 loads the data
  if (mpirank == 0) {
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

    xsize = static_cast<int>(X_train.size());
    ysize = static_cast<int>(y_train.size());
  }

  // Broadcast data sizes to all processes
  MPI_Bcast(&xsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ysize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate memory on non-root processes
  if (mpirank != 0) {
    X_train.resize(xsize);
    y_train.resize(ysize);
  }

  // Broadcast training data to all processes
  MPI_Bcast(X_train.data(), xsize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(y_train.data(), ysize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // Some hyperparameters for the NN
  int BATCH_SIZE = 256;
  int LOCAL_BATCH = BATCH_SIZE / mpisize;  // Each process handles a portion
  float lr = .01/BATCH_SIZE;

  // Random initialization of the weights (same seed for all processes)
  srand(42);
  vector <float> W1 = random_vector(784*128);
  vector <float> W2 = random_vector(128*64);
  vector <float> W3 = random_vector(64*10);

  std::chrono::time_point<std::chrono::system_clock> t1,t2;
  if (mpirank == 0) cout << "Training the model with " << mpisize << " processes...\n";

  for (unsigned i = 0; i < 10000; ++i) {
    t1 = std::chrono::system_clock::now();

    // Rank 0 selects random index and broadcasts to all processes
    int randindx;
    if (mpirank == 0) {
      randindx = rand() % (42000-BATCH_SIZE);
    }
    MPI_Bcast(&randindx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process handles its own portion of the batch
    int local_start = randindx + mpirank * LOCAL_BATCH;
    vector<float> local_b_X;
    vector<float> local_b_y;

    for (unsigned j = local_start*784; j < (local_start+LOCAL_BATCH)*784; ++j){
      local_b_X.push_back(X_train[j]);
    }
    for (unsigned k = local_start*10; k < (local_start+LOCAL_BATCH)*10; ++k){
      local_b_y.push_back(y_train[k]);
    }

    // Feed forward (each process computes on its local batch)
    vector<float> local_a1 = relu(dot( local_b_X, W1, LOCAL_BATCH, 784, 128 ));
    vector<float> local_a2 = relu(dot( local_a1, W2, LOCAL_BATCH, 128, 64 ));
    vector<float> local_yhat = softmax(dot( local_a2, W3, LOCAL_BATCH, 64, 10 ), 10);

    // Back propagation (compute local gradients)
    vector<float> local_dyhat = (local_yhat - local_b_y);
    // dW3 = a2.T * dyhat
    vector<float> local_dW3 = dot(transform( &local_a2[0], LOCAL_BATCH, 64 ), local_dyhat, 64, LOCAL_BATCH, 10);
    // dz2 = dyhat * W3.T * relu'(a2)
    vector<float> local_dz2 = dot(local_dyhat, transform( &W3[0], 64, 10 ), LOCAL_BATCH, 10, 64) * reluPrime(local_a2);
    // dW2 = a1.T * dz2
    vector<float> local_dW2 = dot(transform( &local_a1[0], LOCAL_BATCH, 128 ), local_dz2, 128, LOCAL_BATCH, 64);
    // dz1 = dz2 * W2.T * relu'(a1)
    vector<float> local_dz1 = dot(local_dz2, transform( &W2[0], 128, 64 ), LOCAL_BATCH, 64, 128) * reluPrime(local_a1);
    // dW1 = X.T * dz1
    vector<float> local_dW1 = dot(transform( &local_b_X[0], LOCAL_BATCH, 784 ), local_dz1, 784, LOCAL_BATCH, 128);

    // Aggregate gradients across all processes using MPI_Allreduce
    vector<float> global_dW3(64*10);
    vector<float> global_dW2(128*64);
    vector<float> global_dW1(784*128);

    MPI_Allreduce(local_dW3.data(), global_dW3.data(), 64*10, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_dW2.data(), global_dW2.data(), 128*64, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_dW1.data(), global_dW1.data(), 784*128, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Updating the parameters (all processes update synchronously)
    W3 = W3 - lr * global_dW3;
    W2 = W2 - lr * global_dW2;
    W1 = W1 - lr * global_dW1;

    if ((mpirank == 0) && (i+1) % 100 == 0){
      // Gather predictions and ground truth from all processes
      vector<float> all_yhat(BATCH_SIZE*10);
      vector<float> all_b_y(BATCH_SIZE*10);

      MPI_Gather(local_yhat.data(), LOCAL_BATCH*10, MPI_FLOAT,
                 all_yhat.data(), LOCAL_BATCH*10, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(local_b_y.data(), LOCAL_BATCH*10, MPI_FLOAT,
                 all_b_y.data(), LOCAL_BATCH*10, MPI_FLOAT, 0, MPI_COMM_WORLD);

      cout << "Predictions:" << "\n";
      print ( all_yhat, 10, 10 );
      cout << "Ground truth:" << "\n";
      print ( all_b_y, 10, 10 );
      vector<float> loss_m = all_yhat - all_b_y;
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
      cout << "*******************************************" << endl;
    } else if ((i+1) % 100 == 0) {
      // Non-root processes also participate in gather
      MPI_Gather(local_yhat.data(), LOCAL_BATCH*10, MPI_FLOAT,
                 nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(local_b_y.data(), LOCAL_BATCH*10, MPI_FLOAT,
                 nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
  };

  auto program_end = std::chrono::system_clock::now();
  chrono::duration<double> total_elapsed = program_end - program_start;

  if (mpirank == 0) {
    cout << "\nTotal program runtime: " << total_elapsed.count() << "s" << endl;
  }

  MPI_Finalize();
  return 0;
}
