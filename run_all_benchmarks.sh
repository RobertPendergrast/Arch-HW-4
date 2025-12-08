#!/bin/bash
# ============================================
# Unified Benchmark Script for Merge Sort Implementations
# ============================================
# Tests 4 sorting algorithms across multiple data distributions
# with 1 billion elements, cleaning up data files after each test.
#
# Algorithms:
#   1. sorting        - Base merge sort
#   2. improved_split - Cache-optimized merge sort
#   3. multi_sort     - Multithreaded merge sort
#   4. sort_simd      - SIMD-accelerated merge sort
#
# Distributions:
#   uniform, normal, pareto, sorted, reverse, nearly
# ============================================

set -e  # Exit on error

# Configuration
# SIZE=1000000000  # 1 billion elements (~4GB data file)
SIZE=1000000000  # 1 billion elements (~4GB data file)
DISTRIBUTIONS=("uniform" "normal" "pareto" "sorted" "reverse" "nearly")
ALGORITHMS=("sorting" "improved_split" "multi_sort" "sort_simd")
ALGORITHM_NAMES=("Base" "Cache-Optimized" "Multithreaded" "SIMD")
RESULTS_FILE="benchmark_results.csv"
OUTPUT_DIR="datasets"
TEMP_OUTPUT="/dev/null"  # Don't write output - saves ~4GB I/O per test

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Merge Sort Benchmark Suite${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Configuration:"
echo "  Array size: $(printf "%'d" $SIZE) elements (~$((SIZE * 4 / 1024 / 1024 / 1024)) GB)"
echo "  Algorithms: ${ALGORITHMS[*]}"
echo "  Distributions: ${DISTRIBUTIONS[*]}"
echo ""

# Build all executables
echo -e "${YELLOW}Building executables...${NC}"
make all
echo ""

# Initialize results CSV
echo "distribution,algorithm,algorithm_name,time_seconds,status" > "$RESULTS_FILE"

# Function to run a single benchmark
run_benchmark() {
    local dist=$1
    local algo=$2
    local algo_name=$3
    local data_file="$OUTPUT_DIR/${dist}_${SIZE}.bin"
    
    echo -e "  Running ${BLUE}${algo_name}${NC}..."
    
    # Run the algorithm and capture output (no timeout - let it run to completion)
    local start_time=$(date +%s.%N)
    
    if ./"$algo" "$data_file" "$TEMP_OUTPUT" > /tmp/sort_stdout.txt 2>&1; then
        local end_time=$(date +%s.%N)
        
        # Extract time from program output (looks for "Sorting took X.XXX seconds")
        local sort_time=$(grep -oP 'Sorting took \K[\d.]+' /tmp/sort_stdout.txt 2>/dev/null || echo "")
        
        if [ -z "$sort_time" ]; then
            # Fallback to wall-clock time
            sort_time=$(echo "$end_time - $start_time" | bc)
        fi
        
        # Check if sorting was successful
        if grep -q "sorted successfully" /tmp/sort_stdout.txt; then
            echo -e "    ${GREEN}✓${NC} ${sort_time}s"
            echo "${dist},${algo},${algo_name},${sort_time},OK" >> "$RESULTS_FILE"
        else
            echo -e "    ${RED}✗${NC} Sorting failed (verification error)"
            echo "${dist},${algo},${algo_name},,VERIFY_FAIL" >> "$RESULTS_FILE"
        fi
    else
        echo -e "    ${RED}✗${NC} Error (exit code: $?)"
        # Show error details if available
        if [ -s /tmp/sort_stdout.txt ]; then
            echo "    Output: $(head -3 /tmp/sort_stdout.txt)"
        fi
        echo "${dist},${algo},${algo_name},,ERROR" >> "$RESULTS_FILE"
    fi
    
    # Clean up temp output (skip if /dev/null)
    if [ "$TEMP_OUTPUT" != "/dev/null" ]; then
        rm -f "$TEMP_OUTPUT"
    fi
}

# Main benchmark loop
total_tests=$((${#DISTRIBUTIONS[@]} * ${#ALGORITHMS[@]}))
current_test=0

for dist in "${DISTRIBUTIONS[@]}"; do
    echo ""
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}Distribution: ${dist^^}${NC}"
    echo -e "${YELLOW}============================================${NC}"
    
    data_file="$OUTPUT_DIR/${dist}_${SIZE}.bin"
    
    # Generate data if it doesn't exist
    if [ ! -f "$data_file" ]; then
        echo -e "Generating ${dist} data (${SIZE} elements)..."
        python3 generate_data.py "$SIZE" "$dist"
        echo ""
    else
        file_size=$(stat -c%s "$data_file" 2>/dev/null || echo 0)
        expected_size=$((8 + SIZE * 4))  # 8-byte header + 4 bytes per element
        if [ "$file_size" -ne "$expected_size" ]; then
            echo -e "Data file size mismatch, regenerating..."
            rm -f "$data_file"
            python3 generate_data.py "$SIZE" "$dist"
            echo ""
        else
            echo "Using existing data file: $data_file"
        fi
    fi
    
    # Run each algorithm on this distribution
    for i in "${!ALGORITHMS[@]}"; do
        algo="${ALGORITHMS[$i]}"
        algo_name="${ALGORITHM_NAMES[$i]}"
        current_test=$((current_test + 1))
        
        echo -e "[$current_test/$total_tests] ${algo_name} (${algo})"
        run_benchmark "$dist" "$algo" "$algo_name"
    done
    
    # Delete the data file to save space
    echo ""
    echo -e "${YELLOW}Cleaning up: Removing $data_file to save storage...${NC}"
    rm -f "$data_file"
done

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Benchmark Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Display summary
echo "Summary:"
echo "--------"
cat "$RESULTS_FILE" | column -t -s ','
echo ""

# Generate plots
echo -e "${YELLOW}Generating plots...${NC}"
python3 plot_benchmark_results.py

echo ""
echo -e "${GREEN}Done! Check the generated PNG files for visualizations.${NC}"
