import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def calculate_qs(double[:] metrics, int[:] labels, bint higher_better = True):
    cdef double[:] x = metrics
    cdef int[:] y = labels
    cdef double[:] qs = np.empty_like(x, dtype=np.double)
    cdef Py_ssize_t i
    cdef double n_targets, n_decoys
    cdef Py_ssize_t j
    cdef Py_ssize_t max_i = len(metrics)
    cdef bint c

    assert int(max_i) == len(x) == len(y) == len(qs)

    with nogil:
        for i in prange(max_i):
            n_targets = 0
            n_decoys = 0
            j = 0
            for j in range(max_i):
                if higher_better:
                    c = x[j] >= x[i]
                    if c:
                        if y[j] == 1:
                            n_targets = n_targets + 1
                        else:
                            n_decoys = n_decoys + 1
                else:
                    if x[j] <= x[i]:
                        if y[j] == 1:
                            n_targets = n_targets + 1
                        else:
                            n_decoys = n_decoys + 1

            qs[i] = n_decoys / n_targets

    return np.asarray(qs)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate_peptide_level_qs(double[:] metrics,
                               int[:] labels,
                               peptides,
                               bint higher_better = True):
    cdef double[:] x
    cdef int[:] y

    cdef int n_peps = len(np.unique(peptides))
    cdef double[:] best_x = np.empty(n_peps, dtype=np.double)
    cdef int[:] best_y = np.empty(n_peps, dtype=np.intc)
    best_peps = np.array(['A'*30]*n_peps, dtype=str)

    cdef double current_metric
    #cdef str current_peptide
    cdef Py_ssize_t i

    ordered_idx = np.argsort(peptides)
    x = np.asarray([metrics[i] for i in ordered_idx], dtype=np.double)
    y = np.asarray([labels[i] for i in ordered_idx], dtype=np.int)
    peps = np.asarray([peptides[i] for i in ordered_idx], dtype=str)

    current_peptide = peps[0]
    current_metric = x[0]
    cdef Py_ssize_t pep_idx = 0
    cdef Py_ssize_t max_i = len(x)

    assert int(max_i) == len(x) == len(y)

    for i in range(1, max_i):
        if peps[i] == current_peptide:
            if higher_better:
                if x[i] > current_metric:
                    current_metric = x[i]
            else:
                if x[i] < current_metric:
                    current_metric = x[i]
        else:
            best_x[pep_idx] = current_metric
            pre_i = i - 1
            best_y[pep_idx] = y[pre_i]
            best_peps[pep_idx] = current_peptide

            pep_idx += 1
            current_peptide = peps[i]
            current_metric = x[i]

    qs = calculate_qs(metrics=best_x, labels=best_y, higher_better=higher_better)

    return np.asarray(qs), np.asarray(best_x), np.asarray(best_y), np.asarray(best_peps)
