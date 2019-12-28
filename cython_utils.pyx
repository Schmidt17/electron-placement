#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport sqrt

DTYPE = np.float64
DTYPE_int = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_int_t

cdef double pair_potential_from_pos(double x1, double y1, double x2, double y2, double qQ) nogil:
	cdef double distance
	distance = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))
	return - qQ / distance / 0.69508

def total_energy(np.ndarray[DTYPE_int_t, ndim=1] electron_indices, np.ndarray[DTYPE_int_t, ndim=1] dopant_indices, np.ndarray[DTYPE_t, ndim=2] site_positions):
	cdef int pointer = 0
	cdef int N = len(electron_indices)
	cdef double Etot = 0.

	cdef double [:, :] pos = site_positions
	cdef long [:] ind = electron_indices
	cdef long [:] dind = dopant_indices

	cdef int i, j
	cdef double [:] diff, dist
	for i in prange(N - 1, schedule="guided", nogil=True, num_threads=11):
		for j in range(i + 1, N):
			Etot += pair_potential_from_pos(pos[ind[i], 0], pos[ind[i], 1], pos[ind[j], 0], pos[ind[j], 1], -1.)

	# add attractive potential of dopants
	for i in prange(N, schedule="guided", nogil=True, num_threads=11):
		for j in range(N):
			if ind[i] != dind[j]:
				Etot += pair_potential_from_pos(pos[ind[i], 0], pos[ind[i], 1], pos[dind[j], 0], pos[dind[j], 1], 1.)

	return Etot