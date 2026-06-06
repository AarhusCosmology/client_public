#!/bin/bash

# CLiENT Setup Script
# Automates the installation of CLiENT and its dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
CONFIG_DIR="${SCRIPT_DIR}/config"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Function to prompt user with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    read -p "$(echo -e ${BLUE}${prompt}${NC} [${default}]: )" user_input
    eval ${var_name}="${user_input:-${default}}"
}

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    
    if [ "$default" = "y" ]; then
        local options="([y]/n)"
    else
        local options="(y/[n])"
    fi
    
    while true; do
        read -p "$(echo -e ${BLUE}${prompt}${NC} ${options}: )" yn
        yn=${yn:-${default}}
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to detect if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect if Python package is installed
python_package_exists() {
    python -c "import $1" >/dev/null 2>&1
}

# Function to check if conda is available
check_conda() {
    if ! command_exists conda; then
        print_error "Conda is not installed or not in PATH."
        print_info "Please install Miniconda or Anaconda first:"
        print_info "https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    return 0
}

# Function to add line to .bashrc if not present
add_to_bashrc() {
    local line="$1"
    local comment="$2"
    local bashrc="${HOME}/.bashrc"
    
    if grep -Fq "$line" "$bashrc" 2>/dev/null; then
        print_info "Already in .bashrc"
    else
        echo "" >> "$bashrc"
        echo "# $comment" >> "$bashrc"
        echo "$line" >> "$bashrc"
        print_success "Added to .bashrc - will apply in new shells"
    fi
}

# Banner
clear
echo -e "${GREEN}"
cat << "EOF"
   _____ _      _ ______ _   _ _______ 
  / ____| |    (_)  ____| \ | |__   __|
 | |    | |     _| |__  |  \| |  | |   
 | |    | |    | |  __| | . ` |  | |   
 | |____| |____| | |____| |\  |  | |   
  \_____|______|_|______|_| \_|  |_|   
                                        
  Cosmological Likelihood Emulator
  using Neural networks with TensorFlow
EOF
echo -e "${NC}"
echo "Setup Script - Version 1.0"
echo ""

print_info "This script will guide you through the CLiENT installation process."
print_info "You can press Ctrl+C at any time to cancel."
echo ""

if ! prompt_yes_no "Do you want to proceed with the installation?" "y"; then
    print_info "Installation cancelled."
    exit 0
fi

# Create resources directory if it doesn't exist
mkdir -p "${RESOURCES_DIR}"

# ========================================
# 1. Conda Environment Setup
# ========================================
print_section "Step 1: Conda Environment (~6 GB)"

if ! check_conda; then
    exit 1
fi

prompt_with_default "Enter the name for the conda environment" "clienv" ENV_NAME

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    if prompt_yes_no "Do you want to remove and recreate it?" "n"; then
        print_info "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        print_info "Using existing environment '${ENV_NAME}'."
        # Still activate it for subsequent installations
        eval "$(conda shell.bash hook)"
        conda activate "${ENV_NAME}"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    print_info "Creating conda environment '${ENV_NAME}' from environment.yaml..."
    conda env create -f "${SCRIPT_DIR}/environment.yaml" -n "${ENV_NAME}"
    print_success "Conda environment '${ENV_NAME}' created successfully!"
fi

# Activate the environment
print_info "Activating environment '${ENV_NAME}'..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

print_success "Environment activated."
print_info "To activate this environment in the future, run: conda activate ${ENV_NAME}"

# Ask to add to .bashrc
if prompt_yes_no "Add 'conda activate ${ENV_NAME}' to your .bashrc?" "n"; then
    add_to_bashrc "conda activate ${ENV_NAME}" "Auto-activate CLiENT conda environment"
fi

# ========================================
# 2. best-inference Setup
# ========================================
print_section "Step 2: best-inference (~5 MB)"

BEST_INFERENCE_PATH="${RESOURCES_DIR}/best_inference"
BEST_INFERENCE_INSTALLED=false

if [ -d "$BEST_INFERENCE_PATH" ] && [ -f "$BEST_INFERENCE_PATH/pyproject.toml" ]; then
    print_info "best-inference already found at: ${BEST_INFERENCE_PATH}"
    BEST_INFERENCE_INSTALLED=true
    if prompt_yes_no "Do you want to reinstall best-inference?" "n"; then
        pip install -e "$BEST_INFERENCE_PATH" && print_success "best-inference reinstalled!"
    fi
elif prompt_yes_no "Do you want to install best-inference (surrogate sampler backend)?" "y"; then
    print_info "Cloning best-inference into ${BEST_INFERENCE_PATH}..."
    git clone https://github.com/LucaJanken/best-inference.git "$BEST_INFERENCE_PATH"
    pip install -e "$BEST_INFERENCE_PATH" && print_success "best-inference installed!" && BEST_INFERENCE_INSTALLED=true
else
    print_info "Skipping best-inference setup."
fi

# ========================================
# 3. CLASS Setup
# ========================================
print_section "Step 3: CLASS (~1 GB)"

CLASS_PATH=""
CLASS_INSTALLED=false

if prompt_yes_no "Do you want to use CLASS?" "y"; then
    DETECTED_CLASS_PATH=$(python -c "import classy, os; print(os.path.dirname(os.path.dirname(classy.__file__)))" 2>/dev/null || echo "")
    [ -n "$DETECTED_CLASS_PATH" ] && print_info "Detected CLASS at: ${DETECTED_CLASS_PATH}"
    
    prompt_with_default "Enter absolute path to CLASS installation" "${RESOURCES_DIR}/class_public" CLASS_PATH
    
    if [ -d "$CLASS_PATH" ] && [ -f "$CLASS_PATH/class" ]; then
        print_success "Using existing CLASS installation at: ${CLASS_PATH}"
        CLASS_INSTALLED=true
        if ! python_package_exists classy && prompt_yes_no "Install CLASS Python wrapper?" "y"; then
            print_info "Installing CLASS Python wrapper..."
            (cd "$CLASS_PATH/python" && python setup.py build && python setup.py install)
            print_success "CLASS Python wrapper installed!"
        fi
    else
        print_info "CLASS not found. Cloning and building..."
        mkdir -p "$(dirname "$CLASS_PATH")"
        git clone https://github.com/lesgourg/class_public.git "$CLASS_PATH"
        (cd "$CLASS_PATH" && make clean && make && cd python && python setup.py build && python setup.py install)
        print_success "CLASS installed successfully!"
        CLASS_INSTALLED=true
    fi
else
    print_info "Skipping CLASS setup."
fi

# ========================================
# 4. MontePython Setup
# ========================================
print_section "Step 4: MontePython (~1 GB, Optional)"

MONTEPYTHON_PATH=""

if prompt_yes_no "Do you want to use MontePython?" "y"; then
    prompt_with_default "Enter absolute path to MontePython installation" "${RESOURCES_DIR}/montepython_public" MONTEPYTHON_PATH
    
    if [ -d "$MONTEPYTHON_PATH" ] && [ -f "$MONTEPYTHON_PATH/montepython/MontePython.py" ]; then
        print_success "Using existing MontePython installation at: ${MONTEPYTHON_PATH}"
    else
        print_info "MontePython not found. Cloning..."
        mkdir -p "$(dirname "$MONTEPYTHON_PATH")"
        git clone https://github.com/brinckmann/montepython_public.git "$MONTEPYTHON_PATH"
        print_success "MontePython installed successfully!"
    fi
else
    print_info "Skipping MontePython setup."
fi

# ========================================
# 5. Planck Likelihood Setup (for MontePython)
# ========================================
print_section "Step 5: Planck Likelihood (~200 MB, for MontePython)"

CLIK_PATH=""

if [ -n "$MONTEPYTHON_PATH" ]; then
    print_info "Planck likelihood (clik) is required for using Planck data with MontePython."
    
    # Detect existing clik installation
    DETECTED_CLIK_PARENT="${RESOURCES_DIR}/planck"
    if [ -n "$CLIK" ] && [ -f "$CLIK/bin/clik_profile.sh" ]; then
        DETECTED_CLIK_PARENT=$(dirname "${CLIK%/}")
        print_info "Detected Planck likelihood at: ${CLIK}"
    elif command_exists clik_print_version; then
        _clik_dir=$(which clik_print_version 2>/dev/null | sed 's|/bin/clik_print_version$||')
        [ -n "$_clik_dir" ] && DETECTED_CLIK_PARENT=$(dirname "$_clik_dir") && \
            print_info "Detected Planck likelihood in PATH"
    fi
    
    prompt_with_default "Enter directory where Planck likelihood is/will be installed" "$DETECTED_CLIK_PARENT" CLIK_INSTALL_DIR
    CLIK_PATH="${CLIK_INSTALL_DIR}/clik"
    
    if [ -d "$CLIK_PATH" ] && [ -f "$CLIK_PATH/bin/clik_profile.sh" ]; then
        print_success "Using existing Planck likelihood at: ${CLIK_PATH}"
    elif prompt_yes_no "Download and install Planck likelihood?" "y"; then
        print_info "Cloning clik and downloading Planck likelihood data..."
        mkdir -p "$CLIK_INSTALL_DIR" && cd "$CLIK_INSTALL_DIR"
        
        # Clone clik from GitHub
        if [ ! -d "clik" ]; then
            git clone https://github.com/benabed/clik.git clik
        else
            print_info "clik directory already exists, skipping clone."
        fi
        
        # Download Planck likelihood data from the Planck Legacy Archive
        [ ! -f "COM_Likelihood_Data-baseline_R3.00.tar.gz" ] && \
            wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"
        tar -xzf COM_Likelihood_Data-baseline_R3.00.tar.gz
        rm COM_Likelihood_Data-baseline_R3.00.tar.gz
        
        cd clik

        # Detect cfitsio and optionally install via conda
        CFITSIO_PREFIX=""
        if [ -f "${CONDA_PREFIX}/lib/libcfitsio.so" ] || [ -f "${CONDA_PREFIX}/lib/libcfitsio.a" ]; then
            print_success "cfitsio found in conda environment."
            CFITSIO_PREFIX="${CONDA_PREFIX}"
        elif pkg-config --exists cfitsio 2>/dev/null; then
            print_success "cfitsio found via pkg-config."
            CFITSIO_PREFIX="$(pkg-config --variable=prefix cfitsio)"
        else
            print_warning "cfitsio not found. clik requires it and will try to build it from source if not provided (this can fail on some systems)."
            if prompt_yes_no "Install cfitsio via conda (recommended)?" "y"; then
                conda install -y -c conda-forge cfitsio
                if [ -f "${CONDA_PREFIX}/lib/libcfitsio.so" ] || [ -f "${CONDA_PREFIX}/lib/libcfitsio.a" ]; then
                    print_success "cfitsio installed successfully."
                    CFITSIO_PREFIX="${CONDA_PREFIX}"
                else
                    print_warning "cfitsio conda install may have failed. clik will attempt to build it from source."
                fi
            else
                print_info "Skipping cfitsio installation. clik will attempt to build it from source."
            fi
        fi

        # Build waf configure flags
        WAF_FLAGS="--install_all_deps"
        [ -n "$CFITSIO_PREFIX" ] && WAF_FLAGS="${WAF_FLAGS} --cfitsio_prefix=${CFITSIO_PREFIX}"

        if ./waf configure $WAF_FLAGS && ./waf install; then
            CLIK_PATH="$(pwd)"
            cd "$SCRIPT_DIR"
            print_success "Planck likelihood (clik) installed successfully!"
        else
            cd "$SCRIPT_DIR"
            print_error "Planck likelihood (clik) installation failed!"
            CLIK_PATH=""
        fi
    else
        CLIK_PATH=""
    fi
    
    if [ -n "$CLIK_PATH" ] && [ -f "${CLIK_PATH}/bin/clik_profile.sh" ] && \
       prompt_yes_no "Add clik profile sourcing to your .bashrc?" "n"; then
        add_to_bashrc "source ${CLIK_PATH}/bin/clik_profile.sh" "Source Planck likelihood (clik) profile"
    fi
else
    print_info "Skipping Planck likelihood (MontePython not configured)."
fi

# ========================================
# 6. Cobaya Setup
# ========================================
print_section "Step 6: Cobaya (~5 MB, Optional)"

COBAYA_INSTALLED=false

if python_package_exists cobaya; then
    print_info "Cobaya is already installed."
    COBAYA_INSTALLED=true
    if prompt_yes_no "Do you want to reinstall Cobaya?" "n"; then
        pip install --upgrade cobaya && print_success "Cobaya reinstalled!"
    fi
elif prompt_yes_no "Do you want to install Cobaya?" "y"; then
    pip install cobaya && print_success "Cobaya installed!" && COBAYA_INSTALLED=true
fi

[ "$COBAYA_INSTALLED" = true ] && print_info "Install likelihoods with: cobaya-install <likelihood_name>"

# ========================================
# 7. MPI Support (mpi4py)
# ========================================
print_section "Step 7: MPI Support (Optional)"

MPI_INSTALLED=false

if python_package_exists mpi4py; then
    print_info "mpi4py is already installed."
    MPI_INSTALLED=true
    if prompt_yes_no "Do you want to reinstall mpi4py?" "n"; then
        pip install --upgrade mpi4py && print_success "mpi4py reinstalled!"
    fi
else
    print_info "MPI support enables parallel likelihood evaluations."
    if command_exists mpirun || command_exists mpiexec; then
        print_info "MPI runtime detected."
        if prompt_yes_no "Install mpi4py?" "y"; then
            pip install mpi4py && print_success "mpi4py installed!" && MPI_INSTALLED=true
        fi
    else
        print_warning "No MPI runtime detected."
        if prompt_yes_no "Install OpenMPI?" "y"; then
            conda install -y -c conda-forge openmpi && print_success "OpenMPI installed!"
            if prompt_yes_no "Install mpi4py?" "y"; then
                pip install mpi4py && print_success "mpi4py installed!" && MPI_INSTALLED=true
            fi
        fi
    fi
fi

# ========================================
# 8. Configuration File Generation
# ========================================
print_section "Step 8: Configuration"


if [ -n "$MONTEPYTHON_PATH" ]; then
    print_info "Generating MontePython configuration file..."
    mkdir -p "$CONFIG_DIR"
    DEFAULT_CONF="${CONFIG_DIR}/default.conf"
    
    {
        [ -n "$CLASS_PATH" ] && echo "path['cosmo'] = '${CLASS_PATH}'"
        [ -n "$CLIK_PATH" ]  && echo "path['clik'] = '${CLIK_PATH}/share/clik/'"
    } > "$DEFAULT_CONF"
    
    print_success "Configuration file created: ${DEFAULT_CONF}"

    # Generate montepython.yaml
    MONTEPYTHON_YAML="${CONFIG_DIR}/montepython.yaml"
    cat > "$MONTEPYTHON_YAML" << EOF
conf: ${DEFAULT_CONF}
path: ${MONTEPYTHON_PATH}/montepython
EOF
    print_success "Configuration file created: ${MONTEPYTHON_YAML}"
fi

# ========================================
# Installation Summary
# ========================================
print_section "Installation Complete!"

print_success "CLiENT setup has finished successfully!"
echo ""
echo "Summary of installed components:"
echo ""

echo -e "  ${GREEN}✓${NC} Conda Environment: ${ENV_NAME}"
[ "$BEST_INFERENCE_INSTALLED" = true ] && echo -e "  ${GREEN}✓${NC} best-inference: ${BEST_INFERENCE_PATH}" || echo -e "  ${YELLOW}○${NC} best-inference: Not installed"
[ -n "$CLASS_PATH" ] && echo -e "  ${GREEN}✓${NC} CLASS: ${CLASS_PATH}" || echo -e "  ${YELLOW}○${NC} CLASS: Not configured"
[ -n "$MONTEPYTHON_PATH" ] && echo -e "  ${GREEN}✓${NC} MontePython: ${MONTEPYTHON_PATH}" || echo -e "  ${YELLOW}○${NC} MontePython: Not configured"
[ -n "$CLIK_PATH" ] && echo -e "  ${GREEN}✓${NC} Planck Likelihood: ${CLIK_PATH}" || echo -e "  ${YELLOW}○${NC} Planck Likelihood: Not configured"
[ "$COBAYA_INSTALLED" = true ] && echo -e "  ${GREEN}✓${NC} Cobaya: Installed" || echo -e "  ${YELLOW}○${NC} Cobaya: Not installed"
[ "$MPI_INSTALLED" = true ] && echo -e "  ${GREEN}✓${NC} MPI Support: Installed" || echo -e "  ${YELLOW}○${NC} MPI Support: Not installed"

echo ""
echo "Next steps:"
echo "  1. Activate: conda activate ${ENV_NAME}"
[ -n "$CLIK_PATH" ] && [ -f "${CLIK_PATH}/bin/clik_profile.sh" ] && echo "  2. Source clik: source ${CLIK_PATH}/bin/clik_profile.sh"
echo "  3. Run example: python client.py input/example_cobaya.yaml -n test_run"
[ -n "$MONTEPYTHON_PATH" ] && echo "     Or MontePython: python client.py input/example_montepython.yaml -n test_run"
[ "$MPI_INSTALLED" = true ] && echo "  4. Parallel: mpirun -n 4 python client.py input/example_cobaya.yaml -n test_run"

echo ""
print_info "For more info, see README.md. Issues? Open on GitHub."
echo ""
print_success "Happy cosmology! 🌌"
echo ""
