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
PYTORCH_VERSION="2.9.1"
CUDA_VERSION="128"
CUTLASS_VERSION="4.3.0"
REMOTE_SETUP=false

# Parse command-line arguments
usage() {
    echo -e "${BOLD}${CYAN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}-p, --pytorch VERSION${NC}    PyTorch version (default: 2.7.1)"
    echo -e "  ${GREEN}-c, --cuda VERSION${NC}       CUDA version: 118, 121, 124, 128 (default: 128)"
    echo -e "  ${GREEN}-t, --cutlass VERSION${NC}    CUTLASS version (default: 4.3.0)"
    echo -e "  ${GREEN}-r, --remote${NC}             Remote setup: install pip and venv from apt"
    echo -e "  ${GREEN}-h, --help${NC}              Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${YELLOW}$0${NC}                                    # Use defaults (PyTorch 2.7.1, CUDA 12.8, CUTLASS 4.3.0)"
    echo -e "  ${YELLOW}$0 -p 2.5.1 -c 121${NC}                   # Use PyTorch 2.5.1 with CUDA 12.1"
    echo -e "  ${YELLOW}$0 --pytorch 2.4.0 --cuda 118${NC}        # Use PyTorch 2.4.0 with CUDA 11.8"
    echo -e "  ${YELLOW}$0 -t 4.2.0${NC}                          # Use CUTLASS 4.2.0"
    echo -e "  ${YELLOW}$0 --remote${NC}                          # Remote setup with pip/venv installation"
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
        -t|--cutlass)
            CUTLASS_VERSION="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE_SETUP=true
            shift 1
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
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip"

# Catch2 configuration
CATCH2_DIR="third-party"
CATCH2_HEADER_URL="https://raw.githubusercontent.com/catchorg/Catch2/v2.13.10/single_include/catch2/catch.hpp"

# CUTLASS configuration
CUTLASS_DIR="third-party/cutlass"
CUTLASS_URL="https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.zip"

echo -e "${BOLD}${MAGENTA}🚀 Setting up explore-gemm environment...${NC}"
echo -e "${CYAN}📦 PyTorch version:${NC} ${BOLD}${PYTORCH_VERSION}${NC}"
echo -e "${CYAN}⚡ CUDA version:${NC} ${BOLD}${CUDA_VERSION}${NC}"
echo -e "${CYAN}✂️  CUTLASS version:${NC} ${BOLD}${CUTLASS_VERSION}${NC}"
echo ""

# Create third-party directory if it doesn't exist
mkdir -p third-party

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo -e "${YELLOW}📦 unzip not found. Installing...${NC}"
    sudo apt install zip -y
    echo -e "${GREEN}✅ unzip installed${NC}"
fi

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

# Check if CUTLASS already exists
if [ -d "$CUTLASS_DIR" ]; then
    echo -e "${YELLOW}⚠️  CUTLASS already exists in $CUTLASS_DIR${NC}"
    read -p "$(echo -e ${CYAN}Do you want to re-download? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ Skipping CUTLASS download.${NC}"
    else
        echo -e "${RED}🗑️  Removing existing CUTLASS...${NC}"
        rm -rf "$CUTLASS_DIR"

        # Download CUTLASS
        echo -e "${BLUE}⬇️  Downloading CUTLASS...${NC}"
        cd third-party
        wget -q --show-progress "$CUTLASS_URL" -O cutlass.zip

        # Extract
        echo -e "${BLUE}📂 Extracting CUTLASS...${NC}"
        unzip -q cutlass.zip

        # Rename to cutlass
        mv cutlass-${CUTLASS_VERSION} cutlass
        rm cutlass.zip
        cd ..
        echo -e "${GREEN}✅ CUTLASS installed${NC}"
    fi
else
    # Download CUTLASS
    echo -e "${BLUE}⬇️  Downloading CUTLASS...${NC}"
    cd third-party
    wget -q --show-progress "$CUTLASS_URL" -O cutlass.zip

    # Extract
    echo -e "${BLUE}📂 Extracting CUTLASS...${NC}"
    unzip -q cutlass.zip

    # Rename to cutlass
    mv cutlass-${CUTLASS_VERSION} cutlass
    rm cutlass.zip
    cd ..
    echo -e "${GREEN}✅ CUTLASS installed${NC}"
fi

# Setup Python virtual environment
echo ""
echo -e "${BLUE}🐍 Setting up Python virtual environment...${NC}"

# Install pip and venv if remote setup is requested
if [ "$REMOTE_SETUP" = true ]; then
    echo -e "${YELLOW}📦 Remote setup: Installing pip and venv...${NC}"
    sudo apt update
    sudo apt install -y python3-pip python3-venv
    echo -e "${GREEN}✅ pip and venv installed${NC}"
fi

# Check if venv directory already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "$(echo -e ${CYAN}Do you want to recreate it? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}🗑️  Removing existing virtual environment...${NC}"
        rm -rf venv
        echo -e "${BLUE}📦 Creating virtual environment...${NC}"
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✅ Using existing virtual environment${NC}"
    fi
else
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment and install packages
echo -e "${BLUE}📦 Installing Python packages...${NC}"
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Construct PyTorch installation command based on CUDA version
CUDA_VERSION_FORMATTED="${CUDA_VERSION:0:2}.${CUDA_VERSION:2}"
TORCH_INSTALL_CMD="torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}"

# Install PyTorch and other packages
pip install $TORCH_INSTALL_CMD -q
pip install loguru pandas plotly pytest click ninja -q

echo -e "${GREEN}✅ Python packages installed${NC}"
deactivate

echo ""
echo -e "${BOLD}${GREEN}✨ Setup complete!${NC}"
echo -e "${CYAN}📍 libtorch is installed in ${BOLD}$LIBTORCH_DIR${NC}"
echo -e "${CYAN}📍 Catch2 header is installed in ${BOLD}$CATCH2_DIR/catch.hpp${NC}"
echo -e "${CYAN}📍 CUTLASS is installed in ${BOLD}$CUTLASS_DIR${NC}"
echo -e "${CYAN}📍 Python venv is created in ${BOLD}venv/${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 To use the Python environment:${NC}"
echo -e "${MAGENTA}  source venv/bin/activate${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 To use in your CMakeLists.txt, add:${NC}"
echo -e "${MAGENTA}  set(CMAKE_PREFIX_PATH \"\${CMAKE_PREFIX_PATH};./third-party/libtorch\")${NC}"
echo -e "${MAGENTA}  find_package(Torch REQUIRED)${NC}"
