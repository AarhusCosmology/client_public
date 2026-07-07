import numpy as np


_MPI_CHECKED = False
_COMM = None


def _get_communicator():
    global _MPI_CHECKED, _COMM
    if not _MPI_CHECKED:
        try:
            from mpi4py import MPI
        except ImportError:
            _COMM = None
        else:
            comm = MPI.COMM_WORLD
            _COMM = comm if comm.Get_size() > 1 else None
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


def _evaluate_local(samples, evaluator):
    return np.asarray([evaluator(sample) for sample in samples])


def broadcast_and_evaluate(samples, evaluator):
    """Evaluate master-owned samples across MPI ranks."""
    comm = _get_communicator()
    if comm is None:
        points = np.asarray(samples)
        return points, _evaluate_local(points, evaluator)

    if is_master():
        points = np.asarray(samples)
        print_master(
            f"Evaluating {len(points)} samples with {get_size()} MPI processes..."
        )
        chunks = np.array_split(points, get_size(), axis=0)
    else:
        points = np.array([])
        chunks = None

    local_samples = comm.scatter(chunks, root=0)
    local_values = _evaluate_local(local_samples, evaluator)
    gathered_values = comm.gather(local_values, root=0)

    if is_master():
        return points, np.concatenate(gathered_values)
    return points, np.array([])
