# Electrostatic optimization

Find the ground state configuration of _N_ negatively charged electrons on a discrete configuration space in the presence of _N_ positively charged fixed charges.

## Method
Simulated annealing. `place.py` runs it on the CPU with the core routine implemented in Cython, while `place_cuda.py` runs on a GPU that supports CUDA, using `pycuda` as an interface.
