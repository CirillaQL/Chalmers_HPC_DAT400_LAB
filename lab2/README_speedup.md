# MNIST Neural Network Speedup Analysis

This directory contains scripts to automatically test and analyze the speedup performance of the MNIST neural network implementation with different numbers of threads.

## Files

- `speedup_test.sh` - Main testing script that runs the analysis
- `plot_speedup.py` - Python script to visualize the results
- `README_speedup.md` - This documentation file

## Quick Start

### 1. Run the Speedup Analysis

```bash
./speedup_test.sh
```

This script will:
- Test `num_partitions` from 1 to 20 threads
- Automatically modify `src/vector_ops.cpp` for each test
- Compile and run the neural network for each configuration
- Extract timing data and calculate speedup ratios
- Generate a CSV file with all results

### 2. Visualize the Results

```bash
python3 plot_speedup.py
```

This will generate:
- Interactive plots showing theoretical vs actual speedup
- Parallel efficiency analysis
- Execution time breakdown
- Speedup vs efficiency correlation
- Saved plots as PNG and PDF files

## Expected Results

On M4 MacBook (4 performance cores):
- **Theoretical max speedup**: ~3.2x (based on Amdahl's law)
- **Actual speedup**: Likely 2.5-3.0x at 4 threads
- **Optimal thread count**: Probably 4 threads
- **Parallel portion**: ~92% of execution time

## Output Files

After running the analysis:

```
results/
├── speedup_results.csv      # Raw timing data
├── speedup_analysis.png     # Speedup plots (PNG)
├── speedup_analysis.pdf     # Speedup plots (PDF)
└── output_*.txt            # Individual run outputs
```

## Understanding the Results

### CSV Columns:
- `threads`: Number of pthread partitions used
- `total_time`: Total program execution time (seconds)
- `dot_time`: Time spent in matrix multiplication (seconds)
- `percentage`: Percentage of time in parallelizable code
- `theoretical_speedup`: Amdahl's law prediction
- `actual_speedup`: Measured speedup vs 1 thread
- `efficiency`: actual_speedup / thread_count

### Key Metrics:
- **Speedup**: How much faster than serial execution
- **Efficiency**: How well threads are utilized (100% = perfect)
- **Parallel fraction**: Portion of code that benefits from threading

## Troubleshooting

### Common Issues:

1. **Build failures**: Make sure you have `make` and `bc` installed
2. **Permission denied**: Run `chmod +x *.sh *.py`
3. **Missing data**: Check if `train.txt` dataset file exists
4. **Timeout errors**: Increase timeout in script if needed

### Requirements:

- Unix-like system (macOS/Linux)
- C++ compiler with pthread support
- Python 3 with matplotlib, pandas, numpy
- Basic Unix tools: sed, grep, bc

## Manual Testing

To test a specific thread count manually:

```bash
# Edit src/vector_ops.cpp
sed -i "s/const int num_partitions = [0-9]*;/const int num_partitions = 4;/" src/vector_ops.cpp

# Rebuild and run
make clean && make
./nnetwork
```

## Theoretical Background

The analysis uses **Amdahl's Law**:

```
Speedup = 1 / ((1-f) + f/P)
```

Where:
- `f` = fraction of code that can be parallelized
- `P` = number of processor cores
- Results show the fundamental limits of parallelization

## Advanced Usage

### Custom Thread Range:
Edit the `for threads in {1..20}` line in `speedup_test.sh`

### Different Iteration Counts:
Modify the timeout or add iteration control in the neural network code

### Additional Metrics:
The output files contain detailed timing information for further analysis

---

*This analysis was designed for the Chalmers HPC DAT400 Lab 2 assignment.*