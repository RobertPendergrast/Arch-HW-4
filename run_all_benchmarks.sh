#!/bin/bash
# ============================================
# Unified Benchmark Script for Merge Sort Implementations
# ============================================
# Tests 4 sorting algorithms across multiple data distributions
# and array sizes, cleaning up data files after each test.
#
# Algorithms:
#   1. sorting        - Base merge sort (single-threaded, no optimizations)
#   2. sort_no_simd 1 - Cache-optimized merge sort (single-threaded with L3 blocking)
#   3. sort_no_simd 16- Multithreaded merge sort (16 threads, L3 blocking, parallel merge)
#   4. sort_simd      - SIMD-accelerated merge sort (16 threads, AVX-512, streaming stores)
#
# Distributions:
#   uniform, normal, pareto, sorted, reverse, nearly
#
# Sizes:
#   1k, 10k, 100k, 1M, 10M, 100M, 1B elements
# ============================================

set -e  # Exit on error

# Configuration
SIZES=(1000 10000 100000 1000000 10000000 100000000 1000000000)
DISTRIBUTIONS=("uniform" "normal" "pareto" "sorted" "reverse" "nearly")

# Algorithm commands (some with thread count argument)
ALGORITHM_CMDS=("./sorting" "./sort_no_simd" "./sort_no_simd" "./sort_simd")
ALGORITHM_ARGS=("" "1" "16" "")  # Thread count for sort_no_simd variants
ALGORITHM_NAMES=("Base" "Cache-Optimized" "Multithreaded" "SIMD")
RESULTS_FILE="benchmark_results.csv"
OUTPUT_DIR="."
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
echo "  Array sizes: ${SIZES[*]} elements"
echo "  Algorithms:"
echo "    - Base:           sorting (single-threaded)"
echo "    - Cache-Optimized: sort_no_simd with 1 thread"
echo "    - Multithreaded:  sort_no_simd with 16 threads"
echo "    - SIMD:           sort_simd (16 threads + AVX-512)"
echo "  Distributions: ${DISTRIBUTIONS[*]}"
echo "  Total tests: $((${#SIZES[@]} * ${#DISTRIBUTIONS[@]} * ${#ALGORITHM_NAMES[@]}))"
echo ""

# Build required executables
echo -e "${YELLOW}Building executables...${NC}"
make sorting sort_no_simd sort_simd
echo ""

# Initialize results CSV
echo "distribution,size,algorithm,algorithm_name,time_seconds,status" > "$RESULTS_FILE"

# Function to run a single benchmark
run_benchmark() {
    local dist=$1
    local size=$2
    local algo_cmd=$3
    local algo_args=$4
    local algo_name=$5
    local data_file="$OUTPUT_DIR/${dist}_${size}.bin"
    
    echo -e "  Running ${BLUE}${algo_name}${NC}..."
    
    # Build the full command (with optional thread argument)
    local full_cmd="$algo_cmd $data_file $TEMP_OUTPUT"
    if [ -n "$algo_args" ]; then
        full_cmd="$algo_cmd $data_file $TEMP_OUTPUT $algo_args"
    fi
    
    # Run the algorithm and capture output (no timeout - let it run to completion)
    local start_time=$(date +%s.%N)
    
    if $full_cmd > /tmp/sort_stdout.txt 2>&1; then
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
            echo "${dist},${size},${algo_name},${algo_name},${sort_time},OK" >> "$RESULTS_FILE"
        else
            echo -e "    ${RED}✗${NC} Sorting failed (verification error)"
            echo "${dist},${size},${algo_name},${algo_name},,VERIFY_FAIL" >> "$RESULTS_FILE"
        fi
    else
        echo -e "    ${RED}✗${NC} Error (exit code: $?)"
        # Show error details if available
        if [ -s /tmp/sort_stdout.txt ]; then
            echo "    Output: $(head -3 /tmp/sort_stdout.txt)"
        fi
        echo "${dist},${size},${algo_name},${algo_name},,ERROR" >> "$RESULTS_FILE"
    fi
    
    # Clean up temp output (skip if /dev/null)
    if [ "$TEMP_OUTPUT" != "/dev/null" ]; then
        rm -f "$TEMP_OUTPUT"
    fi
}

# Main benchmark loop
total_tests=$((${#DISTRIBUTIONS[@]} * ${#SIZES[@]} * ${#ALGORITHM_NAMES[@]}))
current_test=0

for dist in "${DISTRIBUTIONS[@]}"; do
    echo ""
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}Distribution: ${dist^^}${NC}"
    echo -e "${YELLOW}============================================${NC}"
    
    for size in "${SIZES[@]}"; do
        echo ""
        echo -e "${BLUE}--- Size: $(printf "%'d" $size) elements ---${NC}"
        
        data_file="$OUTPUT_DIR/${dist}_${size}.bin"
        # Also check datasets/ directory (where generate_data.py might output)
        datasets_file="datasets/${dist}_${size}.bin"
        
        # Generate data if it doesn't exist in either location
        if [ ! -f "$data_file" ] && [ ! -f "$datasets_file" ]; then
            echo -e "Generating ${dist} data (${size} elements)..."
            python3 generate_data.py "$size" "$dist"
            # If generate_data.py outputs to datasets/, copy to OUTPUT_DIR
            if [ -f "$datasets_file" ] && [ "$OUTPUT_DIR" != "datasets" ]; then
                mkdir -p "$OUTPUT_DIR"
                cp "$datasets_file" "$data_file"
            fi
        else
            # Use existing file (prefer OUTPUT_DIR, fallback to datasets/)
            if [ -f "$data_file" ]; then
                file_size=$(stat -c%s "$data_file" 2>/dev/null || echo 0)
                expected_size=$((8 + size * 4))  # 8-byte header + 4 bytes per element
                if [ "$file_size" -ne "$expected_size" ]; then
                    echo -e "Data file size mismatch, regenerating..."
                    rm -f "$data_file"
                    python3 generate_data.py "$size" "$dist"
                    if [ -f "$datasets_file" ] && [ "$OUTPUT_DIR" != "datasets" ]; then
                        cp "$datasets_file" "$data_file"
                    fi
                else
                    echo "Using existing data file: $data_file"
                fi
            elif [ -f "$datasets_file" ]; then
                # Copy from datasets/ to OUTPUT_DIR
                mkdir -p "$OUTPUT_DIR"
                cp "$datasets_file" "$data_file"
                echo "Copied data file from datasets/ to $data_file"
            fi
        fi
        
        # Run each algorithm on this distribution and size
        for i in "${!ALGORITHM_NAMES[@]}"; do
            algo_cmd="${ALGORITHM_CMDS[$i]}"
            algo_args="${ALGORITHM_ARGS[$i]}"
            algo_name="${ALGORITHM_NAMES[$i]}"
            current_test=$((current_test + 1))
            
            echo -e "[$current_test/$total_tests] ${algo_name}"
            run_benchmark "$dist" "$size" "$algo_cmd" "$algo_args" "$algo_name"
        done
        
        # Delete the data file to save space (after all algorithms tested)
        echo ""
        echo -e "${YELLOW}Cleaning up: Removing $data_file to save storage...${NC}"
        rm -f "$data_file"
    done
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

