"""
Optimize electron configuration on discrete sites by Simulated Annealing.
"""

import numpy as np
import matplotlib.pyplot as plt
# import cupy as cp
from numpy.linalg import norm
import cython_utils

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

def total_energy_GPU(electron_indices, site_positions):
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

def total_energy(electron_indices, dopant_indices, site_positions):
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

def random_rearrangement(config, N_sites):
	success = False
	while not success:
		nudge_vector_x = np.random.randint(-nudge_size, nudge_size+1, size=config.shape)
		nudge_vector_y = np.random.randint(-nudge_size, nudge_size+1, size=config.shape)
		local_config = (np.copy(config) + nudge_vector_x + N_sites_per_dim * nudge_vector_y) % N_sites
		if len(np.unique(local_config)) == len(local_config):
			success = True
	return local_config

# Structure-specific parameters
N_sites_per_dim = 100
lattice_spacing = 1.

# Make the structure
site_positions = make_square_grid_positions(N_sites_per_dim, lattice_spacing)
N_sites = len(site_positions)

# General parameters
N_electrons = 3
hold_T_for = 10 * N_electrons
hold_T_until_succ = 10 * N_electrons
nudge_size = 1
N_options = 40

# Plot options
plot_every = None  # None means only plot on new lowest state
plot_all_iterations = False

# Add dopants
dopant_indices = np.random.choice(N_sites, N_electrons, replace=False)

# This is just a placeholder for temporarily saving the pairwise energies
energy_placeholder = np.zeros(int(np.sum(np.arange(N_electrons))))

# Before starting the optimization, explore the variability of
# energies by sampling some random rearrangements of a random configuration.
# Generate a random configuration
config = np.random.choice(N_sites, N_electrons, replace=False)
energies = []
for i in range(100):
	config = random_rearrangement(config, N_sites)
	energies.append(total_energy(config, dopant_indices, site_positions))
delta_energies = np.diff(np.array(energies))
# Plot a histogram of energy changes
# plt.hist(delta_energies)

# Choose a value above the largest change as a starting temperature
# for the simulated annealing
T = 1.5 * np.max(delta_energies)

def metropolis_step(config, site_positions, T):
	current_energy = total_energy(config, dopant_indices, site_positions)
	# Generate some random options and save the corresponding
	# changes in energy
	optional_energies = []
	optional_configs = []
	for i in range(N_options):
		optional_configs.append(random_rearrangement(config, site_positions.shape[0]))
		optional_energies.append(total_energy(optional_configs[-1], dopant_indices,
											  site_positions))
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

energies = [total_energy(config, dopant_indices, site_positions)]
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
	energies.append(total_energy(config, dopant_indices, site_positions))

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
