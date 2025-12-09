#!/bin/bash
# Run sort_no_simd benchmarks for different thread counts and generate graph

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file>"
    echo "  Example: $0 datasets/uniform_1000000000.bin"
    exit 1
fi

INPUT_FILE=$1

# Remove old CSV file
rm -f multi_sort_results.csv

# Thread counts to test
THREADS=(1 2 4 8 16)

echo "=========================================="
echo "Running Sort No-SIMD Benchmarks"
echo "Input file: $INPUT_FILE"
echo "Thread counts: ${THREADS[@]}"
echo "=========================================="
echo ""

# Build if needed
make sort_no_simd

# Run benchmarks
for threads in "${THREADS[@]}"; do
    echo "----------------------------------------"
    echo "Testing with $threads thread(s)..."
    echo "----------------------------------------"
    ./sort_no_simd "$INPUT_FILE" "$threads"
    echo ""
done

echo "=========================================="
echo "All benchmarks complete!"
echo "CSV data saved to: multi_sort_results.csv"
echo ""
echo "Generating graph..."
echo "=========================================="

# Generate graph
python3 plot_results.py multi_sort_results.csv

echo ""
echo "Done! Check multi_sort_comparison.png"

