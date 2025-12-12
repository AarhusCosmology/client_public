# CLiENT

**CLiENT** (Cosmological Likelihood Emulator using Neural networks with TensorFlow) is a framework for emulating cosmological likelihood functions, bypassing the need for Einstein-Boltzmann solver codes like CLASS and CAMB for evaluation of the likelihood. CLiENT compares to observable emulators like CONNECT, but has the advantage of producing a surrogate likelihood which is completely auto-differentiable. 

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/AarhusCosmology/client.git
```

You can either create and activate the provided conda environment:

```bash
conda env create -f environment.yaml
conda activate clienv
```

or install the dependencies listed in `environment.yaml` in your preferred Python environment.

To automatically activate the conda environment, add to your `.bashrc`:

```bash
echo "conda activate clienv" >> ~/.bashrc
```

### Prerequisites

CLiENT requires working installations of CLASS and either MontePython or Cobaya (not yet implemented). For Planck likelihood analyses, ensure the Planck likelihood package is properly installed and configured.

**Performance Note**: CLiENT benefits significantly from GPU acceleration with CUDA support. The neural network training (TensorFlow) and MCMC sampling (emcee) both leverage GPU resources when available, substantially reducing computation time. The environment includes `tensorflow[and-cuda]` for automatic GPU detection and utilization.

## Usage

Basic usage:

```bash
python client.py input/example.yaml --name my_run
```

### Command Line Options

- `input` (positional, required): Input configuration YAML file
- `-n`, `--name`: Run name/tag for output organization
- `-o`, `--output`: Base results directory (default: `results`)
- `-c`, `--continue DIR`: Resume from DIR with retraining
- `-m`, `--mcmc-continue DIR`: Resume from DIR without retraining
- `-s`, `--start-it N`: Starting iteration (required for continue modes)
- `-i`, `--n-it N`: Number of iterations (overrides convergence criterion)

### Example Configurations

- `input/base2018TTTEEE_lensing_bao.yaml` - Base ΛCDM with Planck 2018 TT,TE,EE+lowE+lensing+BAO
- `input/sterileLCDM_TTTEEE_lensing_bao.yaml` - Sterile neutrino extension (N<sub>s</sub>m<sub>s</sub>ΛCDM)
- `input/example.yaml` - Quick test configuration

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

**Loss Function**: The MSRE loss transitions from absolute to relative error based on distance from best-fit:

```
loss ∝ [(χ²_surrogate - χ²_exact) / (χ²_exact + ε)]²
```

where ε ~ n(1 - 2/9n + k√(2/9n))³ (Wilson-Hilferty transformation) with k controlling the transition scale.

**Activation Function**: The Alsing activation function (which is also utilized in CONNECT) replaces ReLu (which lacks expressivity at negative values) by with a linear function at negative values and is characterised by two hyperparameters: The slope of the linear function at negative values and the broadness of the transition region between the asymptotic functions at negative and positive values.

**Temperature Scheme**: 
- T<sub>T</sub> → 1: samples proportional to evidence integral
- T<sub>T</sub> → ∞: uniform sampling across parameter space

## Configuration

All hyperparameters are specified in YAML format. See `input/example.yaml` for a fully documented configuration file. Key sections:

### Likelihood Configuration
```yaml
likelihood:
  wrapper: montepython                  # 'montepython' or 'cobaya'
  param: path/to/parameter_file.param   # MontePython .param or Cobaya .yaml
  conf: path/to/config_file.conf        # MontePython .conf (omit for Cobaya)
  path: path/to/montepython             # MontePython installation (omit for Cobaya)
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
  k_sigma: 3                # MSRE transition scale (σ)
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
├── scalers/              # Standard/MinMax scalers per iteration
├── training_data/        # HDF5 datasets (x, y) per iteration
├── trained_models/       # Keras models (.keras format)
├── training_history/     # Training histories as .pkl files (loss, val_loss)
├── training_chains/      # emcee chains (if save_chains: true)
├── convergence_stats/    # R-1 statistics and chain statistics
├── metrics.log           # Various metrics for the CLiENT pipeline
├── run.log               # Complete run configuration and timing
└── example.yaml          # Copy of the input file
```

## MPI Support

Parallel likelihood evaluation with MPI:

```bash
mpirun -n <N_processes> python client.py input/example.yaml
```

MPI parallelizes initial sampling and resampling likelihood evaluations. Training and MCMC remain serial (leveraging TensorFlow/emcee internal parallelism).

## Performance

[Add performance information]

## License

[Add license information]

## Citation

If you use CLiENT, please cite:

```
[Add citation when paper is published]
```

## Contact

[Add contact information]
