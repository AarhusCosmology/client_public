import os

import numpy as np


_MPI_ENV_VARS = (
    'OMPI_COMM_WORLD_SIZE',
    'OMPI_COMM_WORLD_RANK',
    'PMI_SIZE',
    'PMI_RANK',
    'PMIX_RANK',
    'MV2_COMM_WORLD_SIZE',
    'SLURM_NTASKS',
    'SLURM_PROCID',
    'MPI_LOCALNRANKS',
    'WORLD_SIZE',
)

_MPI_CHECKED = False
_COMM = None


def _detect_mpi_environment():
    return any(os.environ.get(var) for var in _MPI_ENV_VARS)


def _get_communicator():
    global _MPI_CHECKED, _COMM
    if not _MPI_CHECKED:
        if _detect_mpi_environment():
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise RuntimeError(
                    "MPI environment detected, but mpi4py is not available.\n"
                    "Install it with: pip install mpi4py\n"
                    f"Original error: {exc}"
                ) from exc
            comm = MPI.COMM_WORLD
            _COMM = comm if comm.Get_size() > 1 else None
        else:
            _COMM = None
        _MPI_CHECKED = True
    return _COMM


def is_mpi_available():
    return _get_communicator() is not None


def _get_rank():
    comm = _get_communicator()
    return comm.Get_rank() if comm else 0


def get_size():
    comm = _get_communicator()
    return comm.Get_size() if comm else 1

def is_master():
    return _get_rank() == 0


def print_master(message, end='\n'):
    if is_master():
        print(message, end=end, flush=True)


def bcast(obj, root=0):
    comm = _get_communicator()
    return comm.bcast(obj, root=root) if comm else obj


def _broadcast_array(array, root=0):
    comm = _get_communicator()
    if comm is None:
        return np.asarray(array)

    is_root = _get_rank() == root
    if is_root:
        array = np.ascontiguousarray(array)
        shape = array.shape
        dtype = array.dtype
    else:
        array = None
        shape = None
        dtype = None

    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if not is_root:
        array = np.empty(shape, dtype=dtype)

    comm.Bcast(array, root=root)
    return array


def _evaluate_local(samples, evaluator):
    return np.asarray([evaluator(sample) for sample in samples])


def _evaluate_distributed(samples, evaluator):
    comm = _get_communicator()
    if comm is None:
        return _evaluate_local(samples, evaluator)

    if is_master():
        print_master(
            f"Evaluating {len(samples)} samples with {get_size()} MPI processes..."
        )
        chunks = np.array_split(samples, get_size(), axis=0)
    else:
        chunks = None

    local_samples = comm.scatter(chunks, root=0)
    local_values = _evaluate_local(local_samples, evaluator)
    gathered_values = comm.gather(local_values, root=0)

    if is_master():
        return np.concatenate(gathered_values)
    return np.array([])


def broadcast_and_evaluate(samples, evaluator):
    """Broadcast master-owned samples and evaluate them across MPI ranks."""
    points = _broadcast_array(samples)
    values = _evaluate_distributed(points, evaluator)
    return points, values
