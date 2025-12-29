# CLiENT

**CLiENT** (Cosmological Likelihood Emulator using Neural networks with TensorFlow) is a framework for emulating cosmological likelihood functions, bypassing the need for Einstein-Boltzmann solver codes like [CLASS](https://github.com/lesgourg/class_public) and [CAMB](https://github.com/cmbant/CAMB) for evaluation of the likelihood. CLiENT compares to observable emulators like [CONNECT](https://github.com/AarhusCosmology/connect_public), but has the advantage of producing a surrogate likelihood which is completely auto-differentiable.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Prerequisites](#prerequisites)
    - [CLASS Setup](#class-setup)
    - [Planck Likelihood Setup (for MontePython)](#planck-likelihood-setup-for-montepython)
    - [MontePython Setup](#montepython-setup)
    - [Cobaya Setup](#cobaya-setup)
- [Usage](#usage)
  - [Running CLiENT](#running-client)
  - [Command Line Options](#command-line-options)
  - [Example Configurations](#example-configurations)
- [Benchmarking](#benchmarking)
- [Algorithm Overview](#algorithm-overview)
  - [Key Features](#key-features)
    - [Loss Function](#loss-function)
    - [Activation Function](#activation-function)
    - [Temperature Scheme](#temperature-scheme)
- [Configuration](#configuration)
  - [Likelihood Configuration](#likelihood-configuration)
  - [Data Configuration](#data-configuration)
  - [Model Architecture](#model-architecture)
  - [Training Configuration](#training-configuration)
  - [Sampling Configuration](#sampling-configuration)
  - [Convergence Configuration](#convergence-configuration)
- [Output Structure](#output-structure)
- [Performance](#performance)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Getting Started

### Installation

To use CLiENT, start by cloning the repository:

```bash
git clone https://github.com/AarhusCosmology/client_public.git
```

You can create a conda environment containing all necessary dependencies using:

```bash
conda env create -f environment.yaml -n clienv
```

Alternatively, manually install the dependencies listed in `environment.yaml`. To activate the environment, run:

```bash
conda activate clienv
```
or add this command to your `.bashrc` file to activate it automatically.
### Prerequisites

CLiENT requires working installations of [CLASS](https://github.com/lesgourg/class_public) and either [MontePython](https://github.com/brinckmann/montepython_public) or [Cobaya](https://github.com/CobayaSampler/cobaya). If using MontePython, you will also need the C-based `clik` code for Planck likelihoods (see setup below). Cobaya can use either the newer Python-native Planck likelihoods or the original `clik`-based Planck 2018 likelihoods, both installed via `cobaya-install`. For performance, the neural network training and MCMC sampling can utilize GPU resources via TensorFlow when available. The environment includes `tensorflow[and-cuda]` for automatic GPU detection and utilization.

#### CLASS Setup

Install the CLASS Boltzmann code in your chosen directory:

```bash
git clone https://github.com/lesgourg/class_public.git
```

Build CLASS from the `class_public` directory:

```bash
make clean
make
```

Install the Python wrapper from the `class_public/python` directory:

```bash
python setup.py build
python setup.py install --user  # Use --user unless in a virtual/conda environment
```

#### Planck Likelihood Setup (for MontePython)

Install the Planck likelihood package in your chosen directory:

```bash
wget -O COM_Likelihood_Code-v3.0_R3.01.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.01.tar.gz"
wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"
tar -xvzf COM_Likelihood_Code-v3.0_R3.01.tar.gz
tar -xvzf COM_Likelihood_Data-baseline_R3.00.tar.gz
rm COM_Likelihood_*tar.gz
```

From the `code/plc_3.0/plc-3.01` directory, configure and install:

```bash
./waf configure --install_all_deps
./waf install
```

If `./waf configure --install_all_deps` fails, ensure you have working C and Fortran compilers as well as the BLAS/LAPACK and CFITSIO libraries installed. For more details, including building with Intel MKL, see the [clik documentation](https://github.com/benabed/clik). To source the clik profile, run (replace with your actual installation path):

```bash
source /absolute/path/to/plc-3.01/bin/clik_profile.sh
```
or add this command to your `.bashrc` file to source it automatically.

#### MontePython Setup

Install MontePython in your chosen directory:

```bash
git clone https://github.com/brinckmann/montepython_public.git
```

Copy the default configuration template to the CLiENT config directory (or another location of your choice):

```bash
cp /path/to/montepython_public/default.conf.template /path/to/client_public/config/default.conf
```

Edit `config/default.conf` to point to your actual installation paths:

```python
path['cosmo']    = '/absolute/path/to/class_public'
path['clik']     = '/absolute/path/to/plc-3.01/'
```

Replace with the absolute paths to your installations. The `clik` path is only needed if using Planck likelihoods.

#### Cobaya Setup

Install Cobaya via pip:

```bash
pip install cobaya
```

To install likelihoods, you can either install individual likelihoods directly by name:

```bash
cobaya-install planck_2018_highl_plik.TTTEEE
```

or install all likelihoods referenced in your Cobaya input file:

```bash
cobaya-install your_cobaya_input.yaml
```

Use `--packages-path` to specify where likelihood codes and data will be installed, and add `--skip-global` to skip reinstalling packages already available globally (like CLASS). For more details, see the [Cobaya installation documentation](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html).

## Usage

### Running CLiENT

Start a new run:

```bash
python client.py input/example_cobaya.yaml -n my_run
```

or with MontePython:

```bash
python client.py input/example_montepython.yaml -n my_run
```

Continue from an existing run (skips retraining by default):

```bash
python client.py results/my_run_directory
```

For parallel likelihood evaluation with MPI (requires OpenMPI and mpi4py):

```bash
mpirun -n <N_processes> python client.py <input_yaml|run_directory>
```

MPI parallelizes initial sampling and resampling likelihood evaluations. Training and MCMC remain serial (leveraging TensorFlow/emcee internal parallelism).

### Command Line Options

**New Run Mode:**
- `-n`, `--name`: Optional run name/tag for output organization
- `-o`, `--output`: Base results directory (default: `results`)
- `-i`, `--n-it N`: Number of iterations (overrides convergence criterion)

**Continue Mode:**
- `-r`, `--retrain`: Retrain model at the starting iteration (default: skip retraining)
- `-s`, `--start-it N`: Starting iteration (auto-detected from latest if not specified)
- `-i`, `--n-it N`: Number of (additional) iterations (overrides convergence criterion)

CLiENT automatically detects the mode based on whether the path is a directory (continue) or file (new run).

### Example Configurations

**Cosmological Likelihoods:**
- `input/base2018TTTEEE_lensing_bao.yaml` - Base ΛCDM with Planck 2018 TT,TE,EE+lowE+lensing+BAO
- `input/sterileLCDM_TTTEEE_lensing_bao.yaml` - Sterile neutrino extension (N<sub>s</sub>m<sub>s</sub>ΛCDM)

**Test/Example Likelihoods:**
- `input/example_cobaya.yaml` / `input/example_montepython.yaml` - Simple 2D Gaussian examples
- `input/gaussian.yaml` - 27D Gaussian with Planck-like covariance
- `input/banana.yaml` - 29D Banana-shaped likelihood

## Benchmarking

Compare surrogate likelihood against reference chains:

```bash
python benchmarking/benchmark.py results/my_run_directory
```

Benchmark options:
- `-it`, `--iteration N`: Iteration to benchmark (auto-detects latest if not specified)
- `-n`, `--n-steps N`: Number of MCMC steps (defaults to `max_steps` from config)
- `-t`, `--thin N`: Thinning factor for chains (default: 1)
- `-p`, `--params N1 N2 ...`: Parameter indices to include in analysis
- `-c`, `--chains DIR`: Path to MontePython or Cobaya chains directory for comparison
- `--no-training-data`: Skip loading training data visualization
- `--no-training-history`: Skip loading training history

The benchmark script generates:
- Triangle plots comparing posteriors
- KL divergence metrics between distributions
- MAP point comparisons
- Convergence diagnostics

Additional benchmarking and analysis scripts are available in the `benchmarking/` directory for reproducing figures from the accompanying paper.

## Algorithm Overview

CLiENT implements a temperature-based iterative training scheme:

1. **Initial Sampling**: Latin Hypercube sampling within restricted prior bounds (±n<sub>std</sub>σ around fiducial values)

2. **Neural Network Training**: 
   - Deep feedforward network with the Alsing activation function
   - Mean Square Relative Error (MSRE) loss function emphasizing high-likelihood regions
   - Early stopping with validation monitoring

3. **Tempered MCMC Sampling**: 
   - Affine-invariant ensemble sampler (emcee) using surrogate likelihood
   - Temperature T<sub>MCMC</sub> controls exploration
   - Adaptive convergence based on autocorrelation time and ESS

4. **Adaptive Resampling**:
    - Candidates weighted by L<sup>1/T<sub>T</sub> - 1/T<sub>MCMC</sub></sup>
    - k-NN density-aware acceptance criterion targeting a distribution proportional to L<sup>1/T<sub>T</sub>
    - Training temperature T<sub>T</sub> interpolates between evidence-based (T<sub>T</sub> → 1) and uniform (T<sub>T</sub> → ∞) sampling strategies

5. **Convergence Monitoring**:
   - Gelman-Rubin R-1 statistic between successive iterations
   - Iterates until R-1 < threshold or maximum iterations reached

### Key Features

#### Loss Function

The MSRE loss transitions from absolute to relative error based on distance from best-fit:

```
loss ∝ [(χ²_surrogate - χ²_exact) / (χ²_exact + ε)]²
```

where ε ~ n(1 - 2/9n + k√(2/9n))³ (Wilson-Hilferty transformation) with k controlling the transition scale.

#### Activation Function

The Alsing activation function (which is also utilized in CONNECT) replaces ReLU (which lacks expressivity at negative values) with a linear function at negative values:

```
f(x) = [γ + (1 + e^(-βx))^(-1) (1 - γ)] x
```

where β controls the broadness of the transition region and γ controls the asymptotic slope at negative values. Both hyperparameters are trainable for each node in each layer.

#### Temperature Scheme

- T<sub>T</sub> → 1: samples proportional to evidence integral
- T<sub>T</sub> → ∞: uniform sampling across parameter space

## Configuration

All hyperparameters are specified in YAML format. See `input/example_cobaya.yaml` or `input/example_montepython.yaml` for documented configuration examples.

### Key Configuration Sections

### Likelihood Configuration

**MontePython:**
```yaml
likelihood:
  wrapper: montepython
  param: input/montepython/example.param  # MontePython .param file
  conf: config/default.conf               # MontePython .conf file
  path: resources/montepython_public/montepython
```

**Cobaya:**
```yaml
likelihood:
  wrapper: cobaya
  param: input/cobaya/example.yaml  # Cobaya .yaml file
  conf:                             # Leave empty for Cobaya
  path:                             # Leave empty for Cobaya
```

### Data Configuration
```yaml
data:
  scalers:
    parameters: standard    # 'standard' or 'minmax' scaling for input parameters
    targets: standard       # 'standard' or 'minmax' scaling for targets (loglkl)
  initial:
    n_samples: 5000         # Number of initial samples
    strategy: lhs           # 'lhs' (Latin Hypercube) or 'random'
    n_std: 10.0             # Prior restriction (±10σ from fiducial)
  iterative:
    n_candidates: 1000      # MCMC candidate pool size per iteration
    strategy: adaptive      # 'adaptive' or 'random'
    k_NN: 20                # Neighbors for density estimation
    temperature: 7.0        # Training temperature T_T
    update_freq: 50         # KDTree rebuild frequency
```

### Model Architecture
```yaml
model:
  n_layers: 5               # Number of hidden layers
  n_neurons: 512            # Neurons per hidden layer
  activation: alsing        # 'alsing', 'custom_tanh', or TensorFlow activations
```

### Training Configuration
```yaml
training:
  n_epochs: 5000            # Maximum training epochs
  batch_size: 128           # Training batch size
  loss: msre                # 'msre' or TensorFlow loss functions
  kappa_sigma: 3            # MSRE transition scale (σ)
  learning_rate: 0.0001     # Adam optimizer learning rate
  val_split: 0.1            # Validation split fraction
  patience: 250             # Early stopping patience (epochs)
```

### Sampling Configuration
```yaml
sampling:
  save_chains: false        # Save training chains to disk
  temperature: 7.0          # MCMC temperature T_MCMC
  method: emcee             # MCMC sampler (currently only 'emcee')
  emcee:
    n_walkers: 216          # Number of walkers (typically 8 × n_dim)
    max_steps: 100000       # Maximum MCMC steps
    burn_in: 5000           # Burn-in steps to discard
    ess_target: 50          # Target effective sample size per walker
    delta_tau_tol: 0.05     # Autocorrelation stability tolerance
    chunk_size: 5000        # Processing chunk size
    ac_thin: 10             # Thinning for autocorrelation calculation
```

### Convergence Configuration
```yaml
convergence:
  r_minus_one_threshold: 0.01  # Gelman-Rubin R-1 convergence threshold
  max_iterations: 20           # Maximum iterations (prevents infinite loops)
```

## Output Structure

```
results/YYYYMMDD_HHMMSS_run_name/
├── scalers/                  # Standard/MinMax scalers per iteration
├── training_data/            # HDF5 datasets (x, y) per iteration
├── trained_models/           # Keras models (.keras format)
├── training_history/         # Training histories as .pkl files (loss, val_loss)
├── training_chains/          # emcee chains (if save_chains: true)
├── convergence_stats/        # R-1 statistics and chain statistics
├── benchmark_chains/         # Benchmark MCMC chains (from benchmark.py)
├── benchmark_figures/        # Triangle plots and visualizations (from benchmark.py)
├── benchmark_results/        # Diagnostics logs (from benchmark.py)
├── metrics.log               # Various metrics for the CLiENT pipeline
├── run.log                   # Complete run configuration and timing
└── example.yaml              # Copy of the input configuration file
```

When continuing from an existing run, the original YAML configuration is preserved.

## Performance

For cosmological likelihoods with ~30 varying parameters, CLiENT typically requires fewer than 2×10⁴ function evaluations to produce credible intervals within better than 0.1σ of those obtained using the true likelihood, while maintaining single-point emulator precision better than Δχ² ~ 0.5 across relevant regions in parameter space.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CLiENT in your publications, please cite [arXiv:2512.17509](https://arxiv.org/abs/2512.17509)

## Contact

For questions, issues, or contributions, please open an issue or contact me at luca.janken@post.au.dk.
