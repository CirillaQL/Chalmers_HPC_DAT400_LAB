#!/bin/bash

# Script to test different OpenMP scheduling policies and chunk sizes

# Array of scheduling policies
policies=("static" "dynamic" "guided")

# Array of chunk sizes
# For N=256 (BATCH_SIZE), P=4: N/P = 64
# For N=784 (transform operations), P=4: N/P = 196
# Testing: 1 (minimum), 16, 64 (N/P for BATCH_SIZE), 196 (N/P for largest)
chunk_sizes=(1 16 64 196)

# Output file
output_file="scheduling_results.txt"

# Backup original file
cp src/vector_ops.cpp src/vector_ops.cpp.backup

# Clear previous results
> $output_file

echo "Testing OpenMP Scheduling Policies" | tee -a $output_file
echo "===================================" | tee -a $output_file
echo "" | tee -a $output_file

# Loop through each policy
for policy in "${policies[@]}"; do
    echo "Testing policy: $policy" | tee -a $output_file
    echo "-----------------------------------" | tee -a $output_file

    # Loop through each chunk size
    for chunk in "${chunk_sizes[@]}"; do
        echo "  Chunk size: $chunk" | tee -a $output_file

        # Restore backup and modify the scheduling policy in vector_ops.cpp
        cp src/vector_ops.cpp.backup src/vector_ops.cpp

        # For macOS, use different sed syntax
        # Match with optional trailing whitespace
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/#pragma omp for[[:space:]]*$/#pragma omp for schedule($policy, $chunk)/" src/vector_ops.cpp
            sed -i '' "s/#pragma omp for schedule([^)]*)[[:space:]]*$/#pragma omp for schedule($policy, $chunk)/" src/vector_ops.cpp
        else
            sed -i "s/#pragma omp for[[:space:]]*$/#pragma omp for schedule($policy, $chunk)/" src/vector_ops.cpp
            sed -i "s/#pragma omp for schedule([^)]*)[[:space:]]*$/#pragma omp for schedule($policy, $chunk)/" src/vector_ops.cpp
        fi

        # Verify the modification
        current_pragma=$(grep "#pragma omp for" src/vector_ops.cpp)
        echo "    Modified: $current_pragma" | tee -a $output_file

        # Clean and rebuild
        make clean > /dev/null 2>&1
        make CXXFLAGS="-I./include -std=c++11 -fopenmp -O3 -DUSE_OPENMP" > /dev/null 2>&1

        if [ $? -ne 0 ]; then
            echo "    Compilation failed!" | tee -a $output_file
            continue
        fi

        # Run the program and capture output
        echo "    Running..." | tee -a $output_file
        ./nnetwork 2>&1 | grep "Total OpenMP time" | tee -a $output_file

        echo "" | tee -a $output_file
    done

    echo "" | tee -a $output_file
done

# Restore original file
cp src/vector_ops.cpp.backup src/vector_ops.cpp
rm src/vector_ops.cpp.backup

echo "Tests completed! Results saved in $output_file"
