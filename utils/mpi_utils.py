import time

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


def print_master(message, end="\n"):
    if is_master():
        print(message, end=end, flush=True)


def bcast(obj, root=0):
    comm = _get_communicator()
    return comm.bcast(obj, root=root) if comm else obj


def _format_progress(done, total, elapsed):
    percent = 100.0 * done / total if total else 100.0
    return f"Evaluated {done}/{total} samples ({percent:.1f}%) in {elapsed:.1f}s"


def _progress_batch_size(n_processes):
    return max(1, n_processes)


def _evaluate_local(samples, evaluator):
    return np.asarray([evaluator(sample) for sample in samples])


def broadcast_and_evaluate(samples, evaluator):
    """Evaluate master-owned samples across MPI ranks."""
    comm = _get_communicator()
    if comm is None:
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples...")
        start_time = time.monotonic()
        values = []
        batch_size = _progress_batch_size(get_size())
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            values.append(_evaluate_local(points[start:end], evaluator))
            print_master(_format_progress(end, total, time.monotonic() - start_time))
        return points, np.concatenate(values) if values else np.array([])

    if is_master():
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples with {get_size()} MPI processes...")
    else:
        points = np.array([])
        total = None

    total = comm.bcast(total, root=0)
    start_time = time.monotonic()
    gathered_values = []
    batch_size = _progress_batch_size(get_size())

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if is_master():
            batch = points[start:end]
            chunks = np.array_split(batch, get_size(), axis=0)
        else:
            chunks = None

        local_samples = comm.scatter(chunks, root=0)
        local_values = _evaluate_local(local_samples, evaluator)
        batch_values = comm.gather(local_values, root=0)

        if is_master():
            gathered_values.append(np.concatenate(batch_values))
            print_master(_format_progress(end, total, time.monotonic() - start_time))

    if is_master():
        if gathered_values:
            return points, np.concatenate(gathered_values)
        return points, np.array([])
    return points, np.array([])
