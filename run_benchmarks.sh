#!/bin/bash
# Run multi_sort benchmarks for different thread counts and generate graph

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
echo "Running Multi-Sort Benchmarks"
echo "Input file: $INPUT_FILE"
echo "Thread counts: ${THREADS[@]}"
echo "=========================================="
echo ""

# Build if needed
make multi_sort

# Run benchmarks
for threads in "${THREADS[@]}"; do
    echo "----------------------------------------"
    echo "Testing with $threads thread(s)..."
    echo "----------------------------------------"
    ./multi_sort "$INPUT_FILE" "$threads"
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

