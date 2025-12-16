#!/bin/bash

###############################################################################
# CUTLASS GEMM Benchmark Runner
#
# This script automatically detects GPU architecture and runs the appropriate
# benchmark binary (benchmark_hopper for Hopper SM90+ or benchmark_blackwell
# for Blackwell SM100+) with various parameter
# combinations to test different:
# - Problem sizes (M, N, K)
# - Decomposition modes (heuristic, streamk, splitk, dataparallel)
# - Rasterization orders (Along N, Along M, Heuristic)
# - Swizzle factors
# - Split-K counts (for splitk mode)
#
# Usage:
#   ./cuda/scripts/benchmarks.sh [OPTIONS]
#
# Options:
#   --binary PATH       Path to benchmark_hopper binary (default: ../../build/benchmark_hopper)
#   --csv               Output results in CSV format
#   --iterations N      Number of iterations per benchmark (default: 100)
#   --output FILE       Output file for results (default: stdout)
#   --quick             Run a quick subset of benchmarks
#   --full              Run comprehensive benchmarks (default)
#   --help              Display this help message
#
# Examples:
#   # Run all benchmarks with CSV output (from project root)
#   ./cuda/scripts/benchmarks.sh --csv --output results.csv
#
#   # Quick benchmark run
#   ./cuda/scripts/benchmarks.sh --quick
#
#   # Custom binary path with specific iterations
#   ./cuda/scripts/benchmarks.sh --binary ./build/benchmark_hopper --iterations 50
###############################################################################

set -e  # Exit on error

# Default configuration
BINARY=""  # Will be auto-detected based on GPU architecture
AUTO_DETECT=true
CSV_MODE=false
ITERATIONS=100
OUTPUT_FILE=""
BENCHMARK_MODE="full"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --binary)
            BINARY="$2"
            AUTO_DETECT=false
            shift 2
            ;;
        --csv)
            CSV_MODE=true
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --quick)
            BENCHMARK_MODE="quick"
            shift
            ;;
        --full)
            BENCHMARK_MODE="full"
            shift
            ;;
        --help)
            echo "CUTLASS GEMM Benchmark Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --binary PATH       Path to benchmark binary (default: auto-detected based on GPU)"
            echo "  --csv               Output results in CSV format"
            echo "  --iterations N      Number of iterations per benchmark (default: 100)"
            echo "  --output FILE       Output file for results (default: stdout)"
            echo "  --quick             Run a quick subset of benchmarks"
            echo "  --full              Run comprehensive benchmarks (default)"
            echo "  --help              Display this help message"
            echo ""
            echo "Examples:"
            echo "  # Run all benchmarks with CSV output"
            echo "  $0 --csv --output results.csv"
            echo ""
            echo "  # Quick benchmark run"
            echo "  $0 --quick"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect GPU architecture and select appropriate binary
if [ "$AUTO_DETECT" = true ]; then
    # Detect GPU compute capability
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
        # Remove dot: "9.0" -> "90", "10.0" -> "100"
        GPU_ARCH=$(echo "$COMPUTE_CAP" | tr -d '.')

        if [ "$CSV_MODE" = false ]; then
            echo -e "${BLUE}Detected GPU compute capability: $COMPUTE_CAP (SM$GPU_ARCH)${NC}"
        fi

        # Select binary based on architecture
        if [ "$GPU_ARCH" -ge 100 ]; then
            BINARY="../../build/benchmark_blackwell"
            ARCH_NAME="Blackwell"
        elif [ "$GPU_ARCH" -ge 90 ]; then
            BINARY="../../build/benchmark_hopper"
            ARCH_NAME="Hopper"
        else
            echo -e "${RED}Error: GPU compute capability SM$GPU_ARCH is not supported${NC}"
            echo "This benchmark requires Hopper (SM90+) or Blackwell (SM100+) architecture"
            exit 1
        fi

        if [ "$CSV_MODE" = false ]; then
            echo -e "${GREEN}Selected binary: $BINARY ($ARCH_NAME)${NC}"
            echo ""
        fi
    else
        echo -e "${RED}Error: nvidia-smi not found. Cannot auto-detect GPU architecture.${NC}"
        echo "Please specify the binary path manually with --binary"
        exit 1
    fi
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found at $BINARY${NC}"
    echo "Please build the project first or specify the correct path with --binary"
    exit 1
fi

# Setup output redirection
if [ -n "$OUTPUT_FILE" ]; then
    exec > "$OUTPUT_FILE"
fi

# Print header
if [ "$CSV_MODE" = true ]; then
    # Check if Blackwell benchmark to include additional columns
    if [[ "$BINARY" == *"blackwell"* ]]; then
        echo "M,N,K,Raster,Swizzle,Decomposition,Splits,Reduction,PreferredCluster,FallbackCluster,AvgRuntime_ms,GFLOPS,WorktileCount"
    else
        echo "M,N,K,Raster,Swizzle,Decomposition,Splits,AvgRuntime_ms,GFLOPS,WorktileCount"
    fi
else
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}CUTLASS GEMM Benchmark Suite${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    if [ "$AUTO_DETECT" = true ]; then
        echo "GPU Architecture: $ARCH_NAME (SM$GPU_ARCH)"
    fi
    echo "Binary: $BINARY"
    echo "Iterations: $ITERATIONS"
    echo "Mode: $BENCHMARK_MODE"
    echo ""
fi

# Define problem sizes based on mode
if [ "$BENCHMARK_MODE" = "quick" ]; then
    # Quick mode: smaller set of problem sizes
    PROBLEM_SIZES=(
        "1024 1024 1024"
        "2048 2048 2048"
        "4096 4096 4096"
    )
else
    # Full mode: comprehensive problem sizes
    PROBLEM_SIZES=(
        # Square matrices
        "512 512 512"
        "1024 1024 1024"
        "2048 2048 2048"
        "4096 4096 4096"
        "8192 8192 8192"
        # Rectangular matrices (M > N)
        "4096 1024 2048"
        "8192 2048 4096"
        # Rectangular matrices (N > M)
        "1024 4096 2048"
        "2048 8192 4096"
        # Different K values
        "2048 2048 512"
        "2048 2048 4096"
        "2048 2048 8192"
    )
fi

# Define decomposition modes
DECOMPOSITIONS=("heuristic" "streamk" "splitk" "dataparallel")

# Define rasterization options
RASTERS=("H" "N" "M")

# Define swizzle factors
if [ "$BENCHMARK_MODE" = "quick" ]; then
    SWIZZLES=(1)
else
    SWIZZLES=(1 2 4)
fi

# Define split-K values (only used for splitk decomposition)
if [ "$BENCHMARK_MODE" = "quick" ]; then
    SPLITS=(2 4)
else
    SPLITS=(2 4 8 16)
fi

# CSV mode arguments
CSV_ARG=""
if [ "$CSV_MODE" = true ]; then
    CSV_ARG="--csv"
fi

# Counter for total benchmarks
TOTAL_BENCHMARKS=0
CURRENT_BENCHMARK=0

# Calculate total number of benchmarks
for size in "${PROBLEM_SIZES[@]}"; do
    for decomp in "${DECOMPOSITIONS[@]}"; do
        for raster in "${RASTERS[@]}"; do
            for swizzle in "${SWIZZLES[@]}"; do
                if [ "$decomp" = "splitk" ]; then
                    for split in "${SPLITS[@]}"; do
                        ((TOTAL_BENCHMARKS++))
                    done
                else
                    ((TOTAL_BENCHMARKS++))
                fi
            done
        done
    done
done

if [ "$CSV_MODE" = false ]; then
    echo -e "${GREEN}Total benchmarks to run: $TOTAL_BENCHMARKS${NC}"
    echo ""
fi

# Function to run a single benchmark
run_benchmark() {
    local m=$1
    local n=$2
    local k=$3
    local decomp=$4
    local raster=$5
    local swizzle=$6
    local splits=$7

    ((CURRENT_BENCHMARK++))

    if [ "$CSV_MODE" = false ]; then
        echo -e "${YELLOW}[$CURRENT_BENCHMARK/$TOTAL_BENCHMARKS] Running: M=$m N=$n K=$k decomp=$decomp raster=$raster swizzle=$swizzle splits=$splits${NC}"
    fi

    # Build command
    CMD="$BINARY --m=$m --n=$n --k=$k --decomposition=$decomp --raster=$raster --swizzle=$swizzle --iterations=$ITERATIONS $CSV_ARG"

    # Add splits argument for splitk mode
    if [ "$decomp" = "splitk" ]; then
        CMD="$CMD --splits=$splits"
    fi

    # Run the benchmark
    if ! $CMD; then
        if [ "$CSV_MODE" = false ]; then
            echo -e "${RED}Error running benchmark${NC}"
        fi
        return 1
    fi

    if [ "$CSV_MODE" = false ]; then
        echo ""
    fi
}

# Main benchmark loop
for size in "${PROBLEM_SIZES[@]}"; do
    read -r M N K <<< "$size"

    for decomp in "${DECOMPOSITIONS[@]}"; do
        for raster in "${RASTERS[@]}"; do
            for swizzle in "${SWIZZLES[@]}"; do
                if [ "$decomp" = "splitk" ]; then
                    # For splitk, run with different split values
                    for split in "${SPLITS[@]}"; do
                        run_benchmark "$M" "$N" "$K" "$decomp" "$raster" "$swizzle" "$split"
                    done
                else
                    # For other decompositions, splits is not used (set to 1)
                    run_benchmark "$M" "$N" "$K" "$decomp" "$raster" "$swizzle" 1
                fi
            done
        done
    done
done

if [ "$CSV_MODE" = false ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All benchmarks completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
fi

exit 0
