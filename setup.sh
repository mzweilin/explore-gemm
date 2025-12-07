#!/bin/bash

# Setup script for explore-gemm project
# This script sets up dependencies using PyTorch from the current conda/virtualenv

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
CUTLASS_VERSION="4.3.0"

# Parse command-line arguments
usage() {
    echo -e "${BOLD}${CYAN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}-t, --cutlass VERSION${NC}    CUTLASS version (default: 4.3.0)"
    echo -e "  ${GREEN}-h, --help${NC}              Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${YELLOW}$0${NC}                                    # Use defaults (CUTLASS 4.3.0)"
    echo -e "  ${YELLOW}$0 -t 4.2.0${NC}                          # Use CUTLASS 4.2.0"
    echo ""
    echo -e "${BOLD}${YELLOW}Note:${NC}"
    echo -e "  This script uses PyTorch from your current conda/virtualenv environment."
    echo -e "  Make sure you have activated the environment with PyTorch installed before running."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--cutlass)
            CUTLASS_VERSION="$2"
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

# Detect PyTorch installation
echo -e "${BOLD}${MAGENTA}🚀 Setting up explore-gemm environment...${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python not found!${NC}"
    echo -e "${YELLOW}Please activate your conda/virtualenv environment first.${NC}"
    exit 1
fi

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    PYTHON_CMD=python3
fi

# Detect environment type and name
echo -e "${BLUE}🔍 Detecting Python environment...${NC}"
ENV_TYPE="unknown"
ENV_NAME="unknown"

# Check for conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    ENV_TYPE="conda"
    ENV_NAME="$CONDA_DEFAULT_ENV"
    echo -e "${GREEN}   Environment type: Conda${NC}"
    echo -e "${GREEN}   Environment name: ${BOLD}${ENV_NAME}${NC}"

    # Warn if using base environment
    if [ "$ENV_NAME" = "base" ]; then
        echo -e "${YELLOW}⚠️  Warning: You are using the 'base' conda environment!${NC}"
        echo -e "${YELLOW}   It's recommended to create a dedicated environment for this project.${NC}"
        echo ""
        read -p "$(echo -e ${CYAN}Do you want to continue with base environment? \(y/N\): ${NC})" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}💡 Create a new environment with:${NC}"
            echo -e "${CYAN}   conda create -n gemm python=3.12${NC}"
            echo -e "${CYAN}   conda activate gemm${NC}"
            echo -e "${CYAN}   conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia${NC}"
            exit 0
        fi
    fi
# Check for virtualenv
elif [ -n "$VIRTUAL_ENV" ]; then
    ENV_TYPE="virtualenv"
    ENV_NAME=$(basename "$VIRTUAL_ENV")
    echo -e "${GREEN}   Environment type: Virtualenv${NC}"
    echo -e "${GREEN}   Environment name: ${BOLD}${ENV_NAME}${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: No conda or virtualenv detected!${NC}"
    echo ""
    echo -e "${BLUE}💡 Would you like to automatically create a virtual environment?${NC}"
    echo -e "${CYAN}   This will:${NC}"
    echo -e "${CYAN}   1. Create a virtual environment in ./venv${NC}"
    echo -e "${CYAN}   2. Install PyTorch with CUDA 12.8 support${NC}"
    echo -e "${CYAN}   3. Continue with the setup${NC}"
    echo ""
    read -p "$(echo -e ${CYAN}Create virtual environment? \(Y/n\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo -e "${YELLOW}⚠️  Continuing with system Python installation.${NC}"
        echo ""
    else
        # Create virtual environment
        VENV_DIR="venv"

        # Detect Python version first
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

        # Check if python3-venv and python3-dev are available (required on Debian/Ubuntu)
        # On Debian/Ubuntu, the venv module and dev headers might be in separate packages
        if command -v apt &> /dev/null; then
            PACKAGES_TO_INSTALL=()

            # Check if the venv package is installed
            if ! dpkg -l | grep -q "python${PYTHON_VERSION}-venv"; then
                PACKAGES_TO_INSTALL+=("python${PYTHON_VERSION}-venv")
            fi

            # Check if the dev package is installed (needed for building extensions)
            if ! dpkg -l | grep -q "python${PYTHON_VERSION}-dev"; then
                PACKAGES_TO_INSTALL+=("python${PYTHON_VERSION}-dev")
            fi

            if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
                echo -e "${YELLOW}⚠️  Missing Python packages: ${PACKAGES_TO_INSTALL[*]}${NC}"
                echo -e "${BLUE}📦 Installing required packages...${NC}"

                if [ "$EUID" -eq 0 ]; then
                    apt update -qq && apt install -y "${PACKAGES_TO_INSTALL[@]}"
                else
                    echo -e "${YELLOW}   Need sudo privileges to install packages${NC}"
                    sudo apt update -qq && sudo apt install -y "${PACKAGES_TO_INSTALL[@]}"
                fi

                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✅ Python packages installed: ${PACKAGES_TO_INSTALL[*]}${NC}"
                else
                    echo -e "${RED}❌ Error: Failed to install Python packages${NC}"
                    echo -e "${YELLOW}Please install manually:${NC}"
                    echo -e "${CYAN}   apt install ${PACKAGES_TO_INSTALL[*]}${NC}"
                    exit 1
                fi
            fi
        fi

        echo -e "${BLUE}🔧 Creating virtual environment in ./${VENV_DIR}...${NC}"

        if [ -d "$VENV_DIR" ]; then
            echo -e "${YELLOW}⚠️  Virtual environment already exists at ./${VENV_DIR}${NC}"
            read -p "$(echo -e ${CYAN}Remove and recreate? \(y/N\): ${NC})" -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_DIR"
            else
                echo -e "${GREEN}✅ Using existing virtual environment${NC}"
                source "$VENV_DIR/bin/activate"
                ENV_TYPE="virtualenv"
                ENV_NAME=$(basename "$VIRTUAL_ENV")
                echo -e "${GREEN}   Environment activated: ${BOLD}${ENV_NAME}${NC}"
            fi
        fi

        if [ ! -d "$VENV_DIR" ]; then
            $PYTHON_CMD -m venv "$VENV_DIR"

            if [ $? -ne 0 ]; then
                echo -e "${RED}❌ Error: Failed to create virtual environment${NC}"
                exit 1
            fi

            echo -e "${GREEN}✅ Virtual environment created${NC}"

            # Activate the virtual environment
            echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
            source "$VENV_DIR/bin/activate"

            # Update Python command to use the one from venv
            PYTHON_CMD=python

            echo -e "${GREEN}✅ Virtual environment activated${NC}"

            # Install PyTorch
            echo -e "${BLUE}📦 Installing PyTorch with CUDA 12.8 support...${NC}"
            echo -e "${CYAN}   This may take a few minutes...${NC}"
            pip install --upgrade pip -q
            pip install torch --index-url https://download.pytorch.org/whl/cu128

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ PyTorch installed successfully${NC}"
                ENV_TYPE="virtualenv"
                ENV_NAME="venv"
            else
                echo -e "${RED}❌ Error: Failed to install PyTorch${NC}"
                echo -e "${YELLOW}Please try installing manually:${NC}"
                echo -e "${CYAN}   source venv/bin/activate${NC}"
                echo -e "${CYAN}   pip install torch --index-url https://download.pytorch.org/whl/cu128${NC}"
                exit 1
            fi
        fi

        echo ""
        echo -e "${BOLD}${YELLOW}📝 Note: To use this environment in the future, run:${NC}"
        echo -e "${CYAN}   source venv/bin/activate${NC}"
        echo ""
    fi
fi

echo ""

# Check if PyTorch is installed
echo -e "${BLUE}🔍 Checking for PyTorch installation...${NC}"
if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
    echo -e "${RED}❌ Error: PyTorch not found in current environment!${NC}"
    echo ""
    echo -e "${YELLOW}Please install PyTorch first:${NC}"
    if [ "$ENV_TYPE" = "conda" ]; then
        echo -e "${CYAN}   conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia${NC}"
    else
        echo -e "${CYAN}   pip install torch --index-url https://download.pytorch.org/whl/cu128${NC}"
    fi
    exit 1
fi

# Get PyTorch info from version metadata
PYTORCH_INFO=$($PYTHON_CMD -c "
import torch
import os
print(f'{torch.__version__}')
print(f'{os.path.dirname(torch.__file__)}')
print(f'{torch.version.cuda if torch.version.cuda else \"cpu\"}')
")

# Parse the output
PYTORCH_VERSION=$(echo "$PYTORCH_INFO" | sed -n '1p')
PYTORCH_PATH=$(echo "$PYTORCH_INFO" | sed -n '2p')
CUDA_VERSION=$(echo "$PYTORCH_INFO" | sed -n '3p')

echo -e "${GREEN}✅ Found PyTorch ${PYTORCH_VERSION}${NC}"
echo -e "${GREEN}   Location: ${PYTORCH_PATH}${NC}"
echo -e "${GREEN}   CUDA version: ${CUDA_VERSION}${NC}"

# Verify PyTorch C++ libraries are available
echo ""
echo -e "${BLUE}🔍 Verifying PyTorch C++ components...${NC}"
MISSING_LIBS=()

# Check for essential libtorch components
if [ ! -d "$PYTORCH_PATH/lib" ]; then
    MISSING_LIBS+=("lib directory")
fi
if [ ! -d "$PYTORCH_PATH/include" ]; then
    MISSING_LIBS+=("include directory")
fi
if [ ! -f "$PYTORCH_PATH/lib/libtorch.so" ] && [ ! -f "$PYTORCH_PATH/lib/libtorch.dylib" ]; then
    MISSING_LIBS+=("libtorch shared library")
fi
if [ ! -d "$PYTORCH_PATH/share/cmake" ]; then
    MISSING_LIBS+=("CMake configuration files")
fi

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Error: PyTorch C++ components not found!${NC}"
    echo -e "${YELLOW}   Missing components:${NC}"
    for lib in "${MISSING_LIBS[@]}"; do
        echo -e "${YELLOW}   - ${lib}${NC}"
    done
    echo ""
    echo -e "${YELLOW}Your PyTorch installation may be incomplete or Python-only.${NC}"
    echo -e "${YELLOW}For C++ development with PyTorch, you need the full distribution.${NC}"
    echo ""
    if [ "$ENV_TYPE" = "conda" ]; then
        echo -e "${BLUE}💡 Try reinstalling PyTorch with conda (includes C++ libraries):${NC}"
        echo -e "${CYAN}   conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia${NC}"
    else
        echo -e "${BLUE}💡 pip wheel packages include C++ libraries by default.${NC}"
        echo -e "${CYAN}   Verify your installation with:${NC}"
        echo -e "${CYAN}   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128${NC}"
    fi
    exit 1
fi

echo -e "${GREEN}✅ PyTorch C++ components verified${NC}"
echo -e "${GREEN}   Headers: ${PYTORCH_PATH}/include${NC}"
echo -e "${GREEN}   Libraries: ${PYTORCH_PATH}/lib${NC}"
echo -e "${GREEN}   CMake config: ${PYTORCH_PATH}/share/cmake${NC}"

# Create symlink to PyTorch installation for CMake
LIBTORCH_DIR="third-party/libtorch"
mkdir -p third-party

if [ -L "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}⚠️  Removing existing libtorch symlink...${NC}"
    rm "$LIBTORCH_DIR"
fi

if [ -d "$LIBTORCH_DIR" ] && [ ! -L "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}⚠️  Found existing libtorch directory (not a symlink)${NC}"
    read -p "$(echo -e ${CYAN}Do you want to remove it and create a symlink? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}🗑️  Removing existing libtorch...${NC}"
        rm -rf "$LIBTORCH_DIR"
    else
        echo -e "${YELLOW}⚠️  Skipping libtorch symlink creation.${NC}"
        LIBTORCH_DIR=""
    fi
fi

if [ -n "$LIBTORCH_DIR" ]; then
    echo -e "${BLUE}🔗 Creating symlink: ${LIBTORCH_DIR} -> ${PYTORCH_PATH}${NC}"
    ln -sf "$PYTORCH_PATH" "$LIBTORCH_DIR"
    echo -e "${GREEN}✅ libtorch symlink created${NC}"
fi

# Catch2 configuration
CATCH2_DIR="third-party"
CATCH2_HEADER_URL="https://raw.githubusercontent.com/catchorg/Catch2/v2.13.10/single_include/catch2/catch.hpp"

# CUTLASS configuration
CUTLASS_DIR="third-party/cutlass"
CUTLASS_URL="https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.zip"

echo ""
echo -e "${CYAN}✂️  CUTLASS version:${NC} ${BOLD}${CUTLASS_VERSION}${NC}"
echo ""

# Create third-party directory if it doesn't exist
mkdir -p third-party

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo -e "${YELLOW}📦 unzip not found. Installing...${NC}"
    sudo apt install unzip -y
    echo -e "${GREEN}✅ unzip installed${NC}"
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

# Install additional Python packages
echo ""
echo -e "${BLUE}📦 Installing additional Python packages...${NC}"
$PYTHON_CMD -m pip install --upgrade pip -q
$PYTHON_CMD -m pip install loguru pandas plotly pytest click ninja -q
echo -e "${GREEN}✅ Python packages installed${NC}"

echo ""
echo -e "${BOLD}${GREEN}✨ Setup complete!${NC}"
if [ -n "$LIBTORCH_DIR" ]; then
    echo -e "${CYAN}📍 libtorch symlink: ${BOLD}$LIBTORCH_DIR${NC} -> ${PYTORCH_PATH}"
fi
echo -e "${CYAN}📍 PyTorch version: ${BOLD}${PYTORCH_VERSION}${NC} (CUDA ${CUDA_VERSION})"
echo -e "${CYAN}📍 Catch2 header: ${BOLD}$CATCH2_DIR/catch.hpp${NC}"
echo -e "${CYAN}📍 CUTLASS: ${BOLD}$CUTLASS_DIR${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 Build the project:${NC}"
echo -e "${MAGENTA}  cmake -B build${NC}"
echo -e "${MAGENTA}  cmake --build build${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 Run tests:${NC}"
echo -e "${MAGENTA}  ctest --test-dir build${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 For Python scripts, use the same environment where PyTorch is installed.${NC}"
