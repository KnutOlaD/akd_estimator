'''
Script for testing the performance of different kernel density estimation methods

This script is designed to compare the performance of various kernel density estimation (KDE) methods
using a set of trajectories. It includes functionality to visualize the results and save them as images.

author: Knut Ola Dølven

'''

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import time
import os
import KDEpy as KDEpy

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pdm_data_generator as pdg
import akd_estimator as akd

# Define grids
frac_diff = 1000 #pick every 1000th particle for the test data
grid_size = 120
grid_size_physical = 120
grid_size_plot = int(grid_size_physical/grid_size)
grid_x = np.linspace(0, grid_size_physical, grid_size)
grid_y = np.linspace(0, grid_size_physical, grid_size)

# Create a dataset without illegal positions
illegal_positions = np.zeros((grid_size, grid_size))
#create bool from the illegal positions
illegal_positions = illegal_positions.astype(bool)
# Generate test data
#trajectories, bws = pdg.create_test_data(stdev=1.4, num_particles_per_timestep=5000, time_steps=380, dt=0.1, grid_size=grid_size, illegal_positions=illegal_positions)

#Let the trajectories be a simple 2d gaussian distribution and create a dataset of 10**7 particles
# ...existing code...
trajectories_gaussian = 50+np.random.normal(loc=0, scale=1, size=(10**7, 2))* np.array([grid_size_physical/10, grid_size_physical/10])
bws = np.ones(trajectories_gaussian.shape[0]) * 1.4  # Set a constant bandwidth for the gaussian distribution

# Define vectors to store the results
comp_times = dict()
comp_times['sci_stats_gauss'] = []
comp_times['KDEpy_tree'] = []
comp_times['KDEpy_naive'] = []
comp_times['KDEpy_FFT'] = []
comp_times['akd'] = []

comp_times_variance = dict()
comp_times_variance['sci_stats_gauss'] = []
comp_times_variance['KDEpy_tree'] = []
comp_times_variance['KDEpy_naive'] = []
comp_times_variance['KDEpy_FFT'] = []
comp_times_variance['akd'] = []

# Generate a gaussian kernel
num_kernels = 2
ratio = 1
gaussian_kernels, bandwidths_h = akd.generate_gaussian_kernels(num_kernels, ratio)
bandwidths_h = bandwidths_h* grid_size_physical / grid_size #Scale the bandwidths to the grid size

# Define sample sizes to test
sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
print(f"Testing KDE performance with sample sizes: {sample_sizes}")

# Loop through different sample sizes
for num_particles in sample_sizes:
    print(f"\nTesting with {num_particles} particles:")
    
    # Extract sample of specified size
    indices = (np.random.choice(trajectories_gaussian.shape[0], size=num_particles, replace=False))

    trajectories_tmp = trajectories_gaussian[indices]
    bws_tmp = bws[indices]
    
    # KDE WITH SCIPY.STATS.GAUSSIAN_KDE # 
    tmp_vec = np.zeros(30)
    print("Running scipy.stats.gaussian_kde for sample size:", num_particles)
    for i in range(30):
        start_time = time.time()
        kde_scipy = sps.gaussian_kde(trajectories_tmp.T, bw_method=bws_tmp[0])
        kde_scipy.set_bandwidth(bw_method=1.4)  # Set a constant bandwidth
        elapsed = time.time() - start_time
        tmp_vec[i] = elapsed
    elapsed = np.median(tmp_vec)
    print(f"scipy took: {elapsed:.4f} seconds")
    comp_times_variance['sci_stats_gauss'].append(np.var(tmp_vec))
    comp_times['sci_stats_gauss'].append(elapsed)
    
    # KDE WITH KDEPY KDEFFT #
    tmpvec = np.zeros(30)
    print("Running KDEpy FFTKDE for sample size:", num_particles)
    for i in range(30):
        # Create a grid for evaluation
        start_time = time.time()
        kde = KDEpy.FFTKDE(kernel='gaussian', norm=2, bw=1)
        grid, points = kde.fit(trajectories_tmp).evaluate(grid_size)
        x,y = np.unique(grid[:,0]), np.unique(grid[:,1])
        z = points.reshape(len(x), len(y))
        elapsed = time.time() - start_time
        tmpvec[i] = elapsed
    elapsed = np.median(tmpvec)
    print(f"KDEpy FFTKDE took: {elapsed:.4f} seconds")
    comp_times_variance['KDEpy_FFT'].append(np.var(tmpvec))
    comp_times['KDEpy_FFT'].append(elapsed)
    
    # KDE WITH KDEPY KDETREE #
    #do this only if the number of particles is less than 10**5
    
    if num_particles < 10**4:
        print("Running KDEpy TreeKDE for sample size:", num_particles)
        tmp_vec = np.zeros(30)
        for i in range(30):
            start_time = time.time()
            kde = KDEpy.TreeKDE(kernel='gaussian', norm=2, bw=1)
            grid, points = kde.fit(trajectories_tmp).evaluate(grid_size)
            x,y = np.unique(grid[:,0]), np.unique(grid[:,1])
            z = points.reshape(len(x), len(y))
            elapsed = time.time() - start_time
            tmp_vec[i] = elapsed
        elapsed = np.median(tmp_vec)
        print(f"KDEpy TreeKDE took: {elapsed:.4f} seconds")
        comp_times_variance['KDEpy_tree'].append(np.var(tmp_vec))
        comp_times['KDEpy_tree'].append(elapsed)
    else:
        print("Skipping KDEpy TreeKDE for sample size:", num_particles, "as it is too large.")
        comp_times_variance['KDEpy_tree'].append(np.nan)
        # add a nan value to the list for this sample size
        comp_times['KDEpy_tree'].append(np.nan)
    
    # KDE WITH KDEPY NAIVE #
    if num_particles < 10**4:
        # run 10 times and take the average time
        tmp_vec = np.zeros(30)
        print("Running KDEpy NaiveKDE for sample size:", num_particles)
        for i in range(30):
            start_time = time.time()
            kde = KDEpy.NaiveKDE(kernel='gaussian', norm=2, bw=1)
            grid, points = kde.fit(trajectories_tmp).evaluate(grid_size)
            x,y = np.unique(grid[:,0]), np.unique(grid[:,1])
            z = points.reshape(len(x), len(y))
            elapsed = time.time() - start_time
            tmp_vec[i] = elapsed
        elapsed = np.median(tmp_vec)
        print(f"KDEpy naive took: {elapsed:.4f} seconds")
        comp_times_variance['KDEpy_naive'].append(np.var(tmp_vec))
        comp_times['KDEpy_naive'].append(elapsed)
    else:
        print("Skipping KDEpy NaiveKDE for sample size:", num_particles, "as it is too large.")
        # add a nan value to the list for this sample size
        comp_times_variance['KDEpy_naive'].append(np.nan)
        comp_times['KDEpy_naive'].append(np.nan)
        
    # KDE WITH AKD #
    tmp_vec = np.zeros(30)
    print("Running AKD for sample size:", num_particles)
    for i in range(30):
        start_time = time.time()
        p_kde,p_kde_N,p_kde_bw = akd.histogram_estimator(trajectories_tmp[:,0],
                                                        trajectories_tmp[:,1],
                                                        grid_x, grid_y)
        akd_estimate = akd.grid_proj_kde(grid_x,
                                        grid_y,
                                        p_kde,
                                        gaussian_kernels,
                                        bandwidths_h,
                                        p_kde_bw)
        elapsed = time.time() - start_time
        tmp_vec[i] = elapsed
    elapsed = np.median(tmp_vec)
    print(f"akd_kde took: {elapsed:.4f} seconds")
    comp_times_variance['akd'].append(np.var(tmp_vec))
    comp_times['akd'].append(elapsed)

# Create a linear scale plot
plt.figure(figsize=(9, 6))
# Compare only KDEpy_FFT, scipy_gaussian_kde, and akd
for method, times in comp_times.items():
    if method in ['KDEpy_FFT', 'sci_stats_gauss', 'akd']:
        plt.plot(sample_sizes, times, marker='o', label=method)
#make the x axis logarithmic
plt.xscale('log')
plt.ylim([0, 0.1])
plt.title('KDE Computation Time vs Sample Size (log scale) in a 100x100 sized grid',fontsize=16)
# get custom legends
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['scipy.stats.gaussian_kde','KDEpy.KDEFFT','akd_estimator'], 
           loc='upper left', 
           fontsize=14)
plt.grid(True, alpha=0.3)
#Increase font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#Increase fontsize of the labels
plt.xlabel('Number of Particles', fontsize=16)
plt.ylabel('Computation Time (seconds)', fontsize=16)
plt.tight_layout()
plt.savefig('kde_performance_linear.png', dpi=300)
plt.tight_layout()
plt.savefig('kde_performance_bars.png', dpi=300)
plt.show()


# Plot the raw data in a scatter plot with transparent points for N=100000

plt.figure(figsize=(6, 6))
plt.scatter(trajectories_gaussian[::frac_diff, 0], trajectories_gaussian[::frac_diff, 1], 
            s=10, alpha=0.5, color='blue', label='Raw Data (N=100000)', edgecolor='none')
plt.title('Input data (N=100000)', fontsize=16)
plt.xlabel('X Coordinate', fontsize=14)
plt.ylabel('Y Coordinate', fontsize=14)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('raw_data_scatter.png', dpi=300)
plt.show()

plt.imshow(akd_estimate, extent=(0, 100, 0, 100),
              origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Density Estimate')
plt.title('AKD Density Estimate', fontsize=16)
plt.xlabel('X Coordinate', fontsize=14)
plt.ylabel('Y Coordinate', fontsize=14)
plt.tight_layout()

plt.savefig('akd_density_estimate.png', dpi=300)
plt.show()
# Save the computation times to a CSV file




