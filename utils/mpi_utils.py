# utils/mpi_utils.py

import numpy as np
import os

_MPI_AVAILABLE = None
_MPI_MODULE = None

def _detect_mpi_environment():
    mpi_env_vars = [
        'OMPI_COMM_WORLD_SIZE',
        'PMI_SIZE',
        'SLURM_NTASKS',
        'MPI_LOCALNRANKS',
    ]
    return any(os.environ.get(var) for var in mpi_env_vars)

def _check_mpi():
    global _MPI_AVAILABLE, _MPI_MODULE
    if _MPI_AVAILABLE is None:
        try:
            from mpi4py import MPI
            _MPI_MODULE = MPI
            _MPI_AVAILABLE = MPI.COMM_WORLD.Get_size() > 1
        except (ImportError, Exception) as e:
            if _detect_mpi_environment():
                raise RuntimeError(
                    "MPI environment detected (running under mpirun/mpiexec) but mpi4py is not available.\n"
                    "Install it with: pip install mpi4py\n"
                    f"Original error: {e}"
                )
            _MPI_AVAILABLE = False
            _MPI_MODULE = None
    return _MPI_AVAILABLE

def is_mpi_available():
    return _check_mpi()

def get_communicator():
    if not _check_mpi():
        return None
    return _MPI_MODULE.COMM_WORLD

def get_rank():
    comm = get_communicator()
    return comm.Get_rank() if comm else 0

def get_size():
    comm = get_communicator()
    return comm.Get_size() if comm else 1

def is_master():
    return get_rank() == 0

def print_master(msg, end='\n'):
    if is_master():
        print(msg, end=end, flush=True)

def barrier():
    comm = get_communicator()
    if comm:
        comm.Barrier()


def bcast_array(arr, root=0):
    comm = get_communicator()
    if not comm:
        return arr
    
    shape = arr.shape if is_master() else None
    dtype = arr.dtype if is_master() else None
    
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if not is_master():
        arr = np.empty(shape, dtype=dtype)
    
    comm.Bcast(arr, root=root)
    return arr


def _distribute_tasks(tasks, size):
    n_tasks = len(tasks)
    tasks_per_rank = [n_tasks // size + (1 if i < n_tasks % size else 0) for i in range(size)]
    
    start_idx = 0
    scattered_tasks = []
    scattered_indices = []
    
    for rank_id in range(size):
        end_idx = start_idx + tasks_per_rank[rank_id]
        scattered_tasks.append(tasks[start_idx:end_idx])
        scattered_indices.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    return scattered_tasks, scattered_indices

def scatter_tasks(tasks, root=0):
    comm = get_communicator()
    
    if not comm:
        return tasks, list(range(len(tasks)))
    
    size = get_size()
    
    if is_master():
        scattered_tasks, scattered_indices = _distribute_tasks(tasks, size)
    else:
        scattered_tasks = None
        scattered_indices = None
    
    local_tasks = comm.scatter(scattered_tasks, root=root)
    local_indices = comm.scatter(scattered_indices, root=root)
    
    return local_tasks, local_indices


def _evaluate_local_samples(local_samples, likelihood_func):
    local_results = []
    for i, sample in enumerate(local_samples):
        local_results.append(likelihood_func(sample))
        
        if is_master() and (i + 1) % max(1, len(local_samples) // 10) == 0:
            print(f"  Rank 0 progress: {i+1}/{len(local_samples)} local evaluations complete", flush=True)
    
    return local_results

def _gather_results(all_indices, all_results, n_samples):
    results = np.zeros(n_samples)
    for rank_results, rank_indices in zip(all_results, all_indices):
        for idx, result in zip(rank_indices, rank_results):
            results[idx] = result
    return results

def parallel_evaluate_likelihood(samples, likelihood_func, description="samples"):
    comm = get_communicator()
    
    if not comm or get_size() == 1:
        return np.array([likelihood_func(x) for x in samples])
    
    local_samples, local_indices = scatter_tasks(samples)
    print_master(f"Evaluating {len(samples)} {description} with {get_size()} MPI processes...")
    
    local_results = _evaluate_local_samples(local_samples, likelihood_func)
    
    all_indices = comm.gather(local_indices, root=0)
    all_results = comm.gather(local_results, root=0)
    
    if is_master():
        return _gather_results(all_indices, all_results, len(samples))
    return np.array([])
