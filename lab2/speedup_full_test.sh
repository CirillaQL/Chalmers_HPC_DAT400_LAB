#!/bin/bash

# Complete speedup test based on the working simple version
# Tests num_partitions from 1 to 20

echo "=== Complete MNIST Speedup Test (1-20 threads) ==="
echo "Testing all thread counts from 1 to 20..."
echo ""

# Create results directory
mkdir -p full_results

# Initialize summary
echo "=== MNIST Neural Network Complete Speedup Results ===" > full_results/summary.txt
echo "Date: $(date)" >> full_results/summary.txt
echo "" >> full_results/summary.txt

baseline_time=""

# Test all thread counts from 1 to 20
for threads in {1..20}; do
    echo "=========================================="
    echo "Testing $threads threads... ($(date))"
    echo "=========================================="

    # Backup original
    cp src/vector_ops.cpp src/vector_ops.cpp.backup

    # Modify num_partitions
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/const int num_partitions = [0-9]*;/const int num_partitions = $threads;/" src/vector_ops.cpp
    else
        sed -i "s/const int num_partitions = [0-9]*;/const int num_partitions = $threads;/" src/vector_ops.cpp
    fi

    echo "Modified to $threads threads"

    # Build
    echo "Building..."
    make clean > full_results/build_${threads}.log 2>&1
    make >> full_results/build_${threads}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "Build failed for $threads threads!"
        echo "THREADS=$threads: BUILD FAILED" >> full_results/summary.txt
        echo "" >> full_results/summary.txt
        cp src/vector_ops.cpp.backup src/vector_ops.cpp
        continue
    fi

    echo "Running neural network training..."
    start_time=$(date +%s)

    ./nnetwork > full_results/output_${threads}.txt 2>&1
    exit_code=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS (took ${duration}s) - extracting results..."

        # Extract final iteration (last 15 lines should contain the final iteration block)
        tail -15 full_results/output_${threads}.txt > full_results/final_${threads}.txt

        # Save to summary
        echo "THREADS=$threads (Duration: ${duration}s):" >> full_results/summary.txt
        cat full_results/final_${threads}.txt >> full_results/summary.txt
        echo "" >> full_results/summary.txt

        # Display results
        echo "Final results:"
        cat full_results/final_${threads}.txt

        # Calculate speedup
        current_time=$(grep "Total program time:" full_results/final_${threads}.txt | grep -o '[0-9]*\.[0-9]*' | head -1)

        if [ $threads -eq 1 ]; then
            baseline_time="$current_time"
            echo "Baseline time: ${baseline_time}s"
        else
            if [ -n "$baseline_time" ] && [ -n "$current_time" ] && command -v bc > /dev/null; then
                if [ $(echo "$current_time > 0" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
                    speedup=$(echo "scale=3; $baseline_time / $current_time" | bc -l 2>/dev/null)
                    efficiency=$(echo "scale=3; $speedup / $threads" | bc -l 2>/dev/null)
                    echo "Actual speedup vs 1 thread: ${speedup}x"
                    echo "Efficiency: $(echo "scale=1; $efficiency * 100" | bc -l)%"
                    echo "SPEEDUP: ${speedup}x, EFFICIENCY: $(echo "scale=1; $efficiency * 100" | bc -l)%" >> full_results/summary.txt
                fi
            fi
        fi

    else
        echo "FAILED with exit code $exit_code"
        echo "THREADS=$threads: PROGRAM FAILED (exit code: $exit_code)" >> full_results/summary.txt

        # Save error details
        echo "Error details:" >> full_results/summary.txt
        tail -10 full_results/output_${threads}.txt >> full_results/summary.txt
        echo "" >> full_results/summary.txt

        # Show error on screen
        echo "Last few lines of output:"
        tail -10 full_results/output_${threads}.txt
    fi

    # Always restore original
    cp src/vector_ops.cpp.backup src/vector_ops.cpp

    echo ""
    echo "Completed $threads threads test"
    echo ""
done

echo "=========================================="
echo "All tests completed! ($(date))"
echo "=========================================="
echo ""

echo "=== FINAL SUMMARY ==="
echo "All results saved in: full_results/summary.txt"
echo ""

# Show quick performance summary
echo "Performance Summary:"
echo "==================="
grep -E "THREADS=|SPEEDUP:" full_results/summary.txt | grep -v "BUILD FAILED\|PROGRAM FAILED"

echo ""
echo "Detailed files generated:"
echo "- full_results/summary.txt: Complete summary with all final iterations"
echo "- full_results/output_*.txt: Full program output for each thread count"
echo "- full_results/final_*.txt: Final iteration data for each thread count"
echo "- full_results/build_*.log: Build logs"

# Final cleanup
if [ -f src/vector_ops.cpp.backup ]; then
    rm src/vector_ops.cpp.backup
fi

echo ""
echo "Complete speedup analysis finished!"