#!/bin/bash

# Setup script for explore-gemm project
# This script downloads and sets up libtorch for CUDA development

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default versions
PYTORCH_VERSION="2.7.1"
CUDA_VERSION="128"

# Parse command-line arguments
usage() {
    echo -e "${BOLD}${CYAN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}-p, --pytorch VERSION${NC}    PyTorch version (default: 2.7.1)"
    echo -e "  ${GREEN}-c, --cuda VERSION${NC}       CUDA version: 118, 121, 124, 128 (default: 128)"
    echo -e "  ${GREEN}-h, --help${NC}              Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${YELLOW}$0${NC}                                    # Use defaults (PyTorch 2.7.1, CUDA 12.8)"
    echo -e "  ${YELLOW}$0 -p 2.5.1 -c 121${NC}                   # Use PyTorch 2.5.1 with CUDA 12.1"
    echo -e "  ${YELLOW}$0 --pytorch 2.4.0 --cuda 118${NC}        # Use PyTorch 2.4.0 with CUDA 11.8"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pytorch)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        -c|--cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

LIBTORCH_DIR="third-party/libtorch"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip"

# Catch2 configuration
CATCH2_DIR="third-party"
CATCH2_HEADER_URL="https://raw.githubusercontent.com/catchorg/Catch2/v2.13.10/single_include/catch2/catch.hpp"

echo -e "${BOLD}${MAGENTA}🚀 Setting up explore-gemm environment...${NC}"
echo -e "${CYAN}📦 PyTorch version:${NC} ${BOLD}${PYTORCH_VERSION}${NC}"
echo -e "${CYAN}⚡ CUDA version:${NC} ${BOLD}${CUDA_VERSION}${NC}"
echo ""

# Create third-party directory if it doesn't exist
mkdir -p third-party

# Check if libtorch already exists
if [ -d "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}⚠️  libtorch already exists in $LIBTORCH_DIR${NC}"
    read -p "$(echo -e ${CYAN}Do you want to re-download? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ Skipping libtorch download.${NC}"
    else
        echo -e "${RED}🗑️  Removing existing libtorch...${NC}"
        rm -rf "$LIBTORCH_DIR"

        # Download libtorch
        echo -e "${BLUE}⬇️  Downloading libtorch...${NC}"
        cd third-party
        wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip

        # Extract
        echo -e "${BLUE}📂 Extracting libtorch...${NC}"
        unzip -q libtorch.zip
        rm libtorch.zip
        cd ..
    fi
else
    # Download libtorch
    echo -e "${BLUE}⬇️  Downloading libtorch...${NC}"
    cd third-party
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip

    # Extract
    echo -e "${BLUE}📂 Extracting libtorch...${NC}"
    unzip -q libtorch.zip
    rm libtorch.zip
    cd ..
fi

# Download Catch2 header
echo -e "${BLUE}⬇️  Downloading Catch2 test framework...${NC}"
if [ ! -f "$CATCH2_DIR/catch.hpp" ]; then
    wget -q --show-progress "$CATCH2_HEADER_URL" -O "$CATCH2_DIR/catch.hpp"
    echo -e "${GREEN}✅ Catch2 header downloaded${NC}"
else
    echo -e "${GREEN}✅ Catch2 header already exists${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}✨ Setup complete!${NC}"
echo -e "${CYAN}📍 libtorch is installed in ${BOLD}$LIBTORCH_DIR${NC}"
echo -e "${CYAN}📍 Catch2 header is installed in ${BOLD}$CATCH2_DIR/catch.hpp${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 To use in your CMakeLists.txt, add:${NC}"
echo -e "${MAGENTA}  set(CMAKE_PREFIX_PATH \"\${CMAKE_PREFIX_PATH};./third-party/libtorch\")${NC}"
echo -e "${MAGENTA}  find_package(Torch REQUIRED)${NC}"
