from pathlib import Path

import numpy as np


def _chain_files(chain_dir):
    chain_files = sorted(Path(chain_dir).glob("*.txt"))
    if not chain_files:
        raise ValueError(f"No chain files found in {chain_dir}")
    return chain_files


def detect_chain_format(chain_dir):
    """Detect whether chains are MontePython or Cobaya format by checking for a header."""
    with open(_chain_files(chain_dir)[0], encoding="utf-8") as f:
        first_line = f.readline().strip()

    return "cobaya" if first_line.startswith("#") else "montepython"


def load_montepython_chains(chain_dir, param_names, thin=1, scales=None):
    chain_files = _chain_files(chain_dir)
    print(f"Loading {len(chain_files)} MontePython chain files from {chain_dir}")

    samples_list = []
    loglikes_list = []
    n_params = len(param_names)
    scales_arr = np.ones(n_params) if scales is None else np.array(scales)

    for chain_file in chain_files:
        data = np.atleast_2d(np.loadtxt(chain_file))
        mult = data[:, 0].astype(int)
        neg_loglkl = data[:, 1]
        params = data[:, 2:2 + n_params]

        # MontePython stores values in internal units; convert to physical units.
        params = params * scales_arr
        samples_list.append(np.repeat(params, mult, axis=0)[::thin])
        loglikes_list.append(-np.repeat(neg_loglkl, mult)[::thin])

    print(f"Loaded {sum(len(s) for s in samples_list)} total samples (thinned by {thin})")
    return samples_list, loglikes_list


def load_cobaya_chains(chain_dir, param_names, thin=1):
    chain_files = _chain_files(chain_dir)
    print(f"Loading {len(chain_files)} Cobaya chain files from {chain_dir}")

    samples_list = []
    loglikes_list = []

    for chain_file in chain_files:
        data = np.atleast_2d(np.loadtxt(chain_file))

        with open(chain_file, encoding="utf-8") as f:
            header = f.readline().strip("#").split()

        mult = data[:, header.index("weight")].astype(int)
        loglkl = -data[:, header.index("minuslogpost")]
        param_cols = [_header_index(header, param_name) for param_name in param_names]
        params = data[:, param_cols]

        samples_list.append(np.repeat(params, mult, axis=0)[::thin])
        loglikes_list.append(np.repeat(loglkl, mult)[::thin])

    print(f"Loaded {sum(len(s) for s in samples_list)} total samples (thinned by {thin})")
    return samples_list, loglikes_list


def _header_index(header, param_name):
    if param_name not in header:
        raise ValueError(f"Parameter {param_name} not found in chain header: {header}")
    return header.index(param_name)
