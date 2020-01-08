"""
Optimize electron configuration on discrete sites by Simulated Annealing.
"""

import numpy as np
import matplotlib.pyplot as plt
# import cupy as cp
from numpy.linalg import norm
# import cython_utils
import pycuda.gpuarray as gpuarray

def make_square_grid_positions(N_per_dim, d):
	template = d * np.arange(N_per_dim)

	positions = np.array([[x, y] for x in template for y in template])

	return positions

def pairwise_energy(pos1, pos2, eps_r=1., qQ=-1.):
	diff = pos2 - pos1
	dist = norm(diff * diff, axis=-1)
	return - qQ / dist / eps_r / 0.69508

def pairwise_energy_from_dists(dists, eps_r=1., qQ=-1.):
	return - qQ / dists / eps_r / 0.69508

def write_shit_to_GPU(electron_indices, Etot_gpu):
	cuda.memcpy_htod(electron_indices_gpu, electron_indices.astype(np.int32))
	cuda.memcpy_htod(Etot_gpu, np.float32(0.))

def total_energy(electron_indices, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu):
	cuda.memcpy_htod(electron_indices_gpu, electron_indices.astype(np.int32))
	cuda.memcpy_htod(Etot_gpu, np.float32(0.))

	blockSize = 1024
	nBlocks = N_electrons * int((N_el + blockSize - 1) / blockSize)
	block = (blockSize, 1, 1)
	grid = (nBlocks, 1)


	total_energy_gpu.prepared_call(grid, block, N_el, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, Etot_gpu)
	# total_energy_gpu(N_el, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, E_gpuarray.gpudata, block=block, grid=grid)
	cuda.Context.synchronize()
	# print(E_gpuarray.get())
	Etot_local = np.empty_like(Etot_cpu)
	cuda.memcpy_dtoh(Etot_local, Etot_gpu)
 	
	return Etot_local


def total_energy_cupy(electron_indices, site_positions):
	site_positions_GPU = cp.array(site_positions)
	electron_indices_GPU = cp.array(electron_indices)
	energies = cp.zeros(int(cp.sum(cp.arange(len(electron_indices_GPU)))))
	# E_tot = 0.
	pointer = 0
	for i in range(len(electron_indices_GPU) - 1):
		# partner_positions = site_positions[electron_indices_GPU[i+1:]]
		# diffs = partner_positions - site_positions[electron_indices_GPU[i]]
		# dists = norm(diffs, axis=-1)
		# slightly faster version of the three lines above:
		dists = norm(site_positions_GPU[electron_indices_GPU[i+1:]] - site_positions_GPU[electron_indices_GPU[i]], axis=-1)
		new_pointer = pointer + dists.shape[0]
		energies[pointer:new_pointer] = pairwise_energy_from_dists(dists)
		pointer = new_pointer
		# E_tot += np.sum(pairwise_energy_from_dists(dists))

	return float(cp.sum(energies))

def total_energy_cython(electron_indices, dopant_indices, site_positions):
	return cython_utils.total_energy(electron_indices, dopant_indices, site_positions)

def total_energy_numpy(electron_indices, site_positions):
	energies = np.zeros(int(np.sum(np.arange(len(electron_indices)))))
	# E_tot = 0.
	pointer = 0
	for i in range(len(electron_indices) - 1):
		# partner_positions = site_positions[electron_indices[i+1:]]
		# diffs = partner_positions - site_positions[electron_indices[i]]
		# dists = norm(diffs, axis=-1)
		# slightly faster version of the three lines above:
		# dists = norm(site_positions[electron_indices[i+1:]] - site_positions[electron_indices[i]], axis=-1)
		diffs = site_positions[electron_indices[i+1:]] - site_positions[electron_indices[i]]
		dists = np.sqrt(np.diag(np.dot(diffs, diffs.T)))
		new_pointer = pointer + dists.shape[0]
		energies[pointer:new_pointer] = pairwise_energy_from_dists(dists)
		pointer = new_pointer
		# E_tot += np.sum(pairwise_energy_from_dists(dists))

	return np.sum(energies)

def fp(P, direction='right'):
	dP = np.zeros_like(P)

	if direction == 'right':
		dP[1:] = (P[:-1] > 1) * (P[:-1] - 1) + (P[1:] > 1) * (-P[1:] + 1) 
		dP[0] = (P[-1] > 1) * (P[-1] - 1) + (P[0] > 1) * (-P[0] + 1) 
	elif direction == 'left':
		dP[:-1] = (P[1:] > 1) * (P[1:] - 1) + (P[:-1] > 1) * (-P[:-1] + 1) 
		dP[-1] = (P[0] > 1) * (P[0] - 1) + (P[-1] > 1) * (-P[-1] + 1) 
	else:
		raise ValueError("Invalid direction specifier")

	return dP

def even_out(arr):
	direction = np.random.choice(['left', 'right'])
	while np.any(arr > 1):
		arr = arr + fp(arr, direction=direction)
	return arr

def random_rearrangement(config, N_sites):
	nudge_vector_xy = np.random.randint(-nudge_size, nudge_size+1, size=config.size * 2)

	local_config = (config + nudge_vector_xy[:N_electrons] + N_sites_per_dim * nudge_vector_xy[N_electrons:]) % N_sites

	occ = np.zeros(N_sites)
	for ind in local_config:
		occ[ind] += 1.

	occ = even_out(occ)
	local_config = np.where(occ == 1.)[0]


	# success = False
	# while not success:
	# 	nudge_vector_x = np.random.randint(-nudge_size, nudge_size+1, size=config.shape)
	# 	nudge_vector_y = np.random.randint(-nudge_size, nudge_size+1, size=config.shape)
	# 	# nudge_vector_x = (nudge_size * np.random.normal(size=config.shape)).astype(np.int32)
	# 	# nudge_vector_y = (nudge_size * np.random.normal(size=config.shape)).astype(np.int32)
	# 	local_config = (config + nudge_vector_x + N_sites_per_dim * nudge_vector_y) % N_sites
	# 	if len(np.unique(local_config)) == len(local_config):
	# 		success = True
	return local_config

# Structure-specific parameters
N_sites_per_dim = 100
lattice_spacing = 1.

# Make the structure
site_positions = make_square_grid_positions(N_sites_per_dim, lattice_spacing)
N_sites = len(site_positions)

# General parameters
N_electrons = 5
hold_T_for = 100# * N_electrons
hold_T_until_succ = 10 * N_electrons
nudge_size = 2
N_options = 100

# Plot options
plot_every = None  # None means only plot on new lowest state
plot_all_iterations = False

# Generate a random configuration
config = np.random.choice(N_sites, N_electrons, replace=False)

# Init memory and computation for CUDA-supported energy calculation
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

Etot_cpu = np.float32(0.)

site_positions_gpu = cuda.mem_alloc(site_positions.astype(np.float32).nbytes)
electron_indices_gpu = cuda.mem_alloc(config.astype(np.int32).nbytes)
Etot_gpu = cuda.mem_alloc(Etot_cpu.nbytes)
# Etotdop_gpu = cuda.mem_alloc(Etot_cpu.nbytes)

N_el = np.int32(N_electrons)
zero_arr = np.zeros(N_el * N_el, dtype=np.float32)

# E_gpuarray = gpuarray.GPUArray(shape=Etot_cpu.shape, dtype=np.float32, gpudata=Etot_gpu)
# Edop_gpuarray = gpuarray.GPUArray(shape=Etot_cpu.shape, dtype=np.float32, gpudata=Etotdop_gpu)

cuda.memcpy_htod(site_positions_gpu, site_positions.flatten().astype(np.float32))

# Load the kernel source
with open("kernel_acc.cpp", "r") as f:
	kernel_source = f.read()
cuda_mod = SourceModule(kernel_source)

# extract the kernel function
total_energy_gpu = cuda_mod.get_function("total_energy")
total_energy_gpu.prepare([np.int32, np.intp, np.intp, np.intp, np.intp])

# Add dopants
dopant_indices = np.random.choice(N_sites, N_electrons, replace=False).astype(np.int32)
dopant_indices_gpu = cuda.mem_alloc(dopant_indices.nbytes)
cuda.memcpy_htod(dopant_indices_gpu, dopant_indices.astype(np.int32))

# This is just a placeholder for temporarily saving the pairwise energies
energy_placeholder = np.zeros(int(np.sum(np.arange(N_electrons))))

# Before starting the optimization, explore the variability of
# energies by sampling some random rearrangements of a random configuration.
energies = []
for i in range(100):
	config = random_rearrangement(config, N_sites)
	energies.append(total_energy(config, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu))
delta_energies = np.diff(np.array(energies))
# Plot a histogram of energy changes
# plt.hist(delta_energies)

# Choose a value above the largest change as a starting temperature
# for the simulated annealing
T = 1.5 * np.max(delta_energies)

def metropolis_step(config, site_positions, T):
	current_energy = total_energy(config, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu)
	# Generate some random options and save the corresponding
	# changes in energy
	optional_energies = []
	optional_configs = []
	for i in range(N_options):
		optional_configs.append(random_rearrangement(config, site_positions.shape[0]))
		optional_energies.append(total_energy(optional_configs[-1], electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu))
	optional_energies = np.array(optional_energies)
	deltaEs = optional_energies - current_energy

	probabilities = np.exp( (deltaEs > 0) * (- deltaEs / T))
	probabilities /= np.sum(probabilities)  # norm to sum 1
	probabilities = np.nan_to_num(probabilities)

	if np.sum(probabilities) == 1.:
		chosen_config_idx = np.random.choice(N_options, p=probabilities)

		return optional_configs[chosen_config_idx]
	else:
		return config

energies = [total_energy(config, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu)]
best_config_so_far = config
lowest_energy_so_far = energies[0]

# Initialize a figure for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot the sites
ax.scatter(site_positions[:, 0], site_positions[:, 1], s=4, color='grey')
# Plot the dopants
ax.scatter(site_positions[dopant_indices, 0], site_positions[dopant_indices, 1], color='red')
# Plot the electrons
el_plot = ax.scatter(site_positions[config, 0], site_positions[config, 1], color='blue')

def update_plot(conf, pos):
	el_plot.set_offsets(pos[conf])
	fig.canvas.draw()
	plt.pause(0.05)  # give matplotlib the chance to handle events

plt.show(block=False)
plt.draw()
plt.pause(0.05)

success_counter = 0
T_step_counter = 0
for step_id in range(1, 100000):
	# generate a new config
	config = metropolis_step(config, site_positions, T)
	# save its energy
	energies.append(total_energy(config, electron_indices_gpu, dopant_indices_gpu, site_positions_gpu, total_energy_gpu, Etot_gpu))

	# if the energy got lower, increase the success counter
	if energies[-1] < energies[-2]:
		success_counter += 1
	# count the steps since the last change of T
	T_step_counter += 1
	# check if we found a new best config and if yes, save it
	if energies[-1] < lowest_energy_so_far:
		lowest_energy_so_far = energies[-1]
		best_config_so_far = config
		if plot_every is None:
			update_plot(best_config_so_far, site_positions)

	# condition for picking a new T
	if (T_step_counter % hold_T_for == 0) or (success_counter > hold_T_until_succ):
		if (T > 0.0005):
			T *= 0.9
		T_step_counter = 0
		success_counter = 0

		print(f"Step {step_id}: T = {T}\tLowest: {lowest_energy_so_far}")

	# condition for updating the plot of the current best config
	if not plot_every is None: 
		if step_id % plot_every == 0:
			update_plot(best_config_so_far, site_positions)

	# condition for plotting all the encountered configs, not only shortest
	if plot_all_iterations:
		update_plot(config, site_positions)
