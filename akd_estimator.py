'''

Adaptive kernel density estimator with boundary control.

Also generates test data and compares. 

Author: Knut Ola Dølven

'''

import numpy as np
from scipy.stats import gaussian_kde
from numba import jit, prange
import time as time
import pdm_data_generator as pdg

# ------------------------------------------------------- #
###########################################################
##################### FUNCTIONS ###########################
###########################################################
# ------------------------------------------------------- #

@jit(nopython=True, parallel=True)
def _process_kernels(non_zero_indices, kde_pilot, cell_bandwidths, kernel_bandwidths, 
                    gaussian_kernels, illegal_cells, gridsize_x, gridsize_y):
    
    """
    Project kernel density estimation onto a 2D grid with optimized memory layout and Numba acceleration.
    Handles illegal/blocked cells by redistributing their density contribution to surrounding legal cells.
    Uses pre-computed Gaussian kernels for efficiency and variable bandwidth selection based on pilot estimate.
    
    Parameters
    ----------
    grid_x : array-like
        X-coordinates of the grid points
    grid_y : array-like
        Y-coordinates of the grid points
    kde_pilot : ndarray
        Pilot kernel density estimate on the grid
    gaussian_kernels : list of ndarrays
        Pre-computed Gaussian kernels for different bandwidths
    kernel_bandwidths : ndarray
        Bandwidths corresponding to pre-computed kernels
    cell_bandwidths : ndarray
        Bandwidth values for each cell in the grid
    illegal_cells : ndarray, optional
        Boolean mask of illegal/blocked cells (True = blocked)
        
    Returns
    -------
    ndarray
        Updated kernel density estimate with illegal cell handling
        
    Notes
    -----
    - Uses contiguous memory layout for performance
    - Handles illegal cells by redistributing their weights
    - Optimized with Numba for parallel processing
    - Only processes non-zero pilot KDE values
    - Kernel selection based on closest available bandwidth
    - Memory efficient by processing only required grid cells
    """
    
    n_u = np.zeros((gridsize_x, gridsize_y))
    
    for idx in prange(len(non_zero_indices)):
        i, j = non_zero_indices[idx]
        
        # Get kernel index and kernel
        kernel_index = np.argmin(np.abs(kernel_bandwidths - cell_bandwidths[i, j]))
        kernel = gaussian_kernels[kernel_index].copy()  # Need copy for numba
        kernel_size = len(kernel) // 2
        
        # Window boundaries
        i_min = max(i - kernel_size, 0)
        i_max = min(i + kernel_size + 1, gridsize_x)
        j_min = max(j - kernel_size, 0)
        j_max = min(j + kernel_size + 1, gridsize_y)
        
        # Handle illegal cells
        illegal_window = illegal_cells.copy()[i_min:i_max, j_min:j_max]

        # ## Find blocked cells... ## #

        # Define adaptation grid
        x0, y0 = i, j
        xi = np.arange(i_min, i_max)
        yj = np.arange(j_min, j_max)
        
        legal_cells = ~illegal_cells.copy()
        illegal_sum = 0

        if np.any(illegal_window):# and illegal_sum > 0:
            shadowed_cells = identify_shadowed_cells(x0, y0, xi, yj, legal_cells)
            #print(shadowed_cells)
            
            for cell_idx in range(len(shadowed_cells)):
                shadow_i = shadowed_cells[cell_idx][0] - i_min #convert to adaptation grid
                shadow_j = shadowed_cells[cell_idx][1] - j_min
                #print(shadow_i, shadow_j)
                if (0 <= shadow_i < illegal_window.shape[0] and 
                    0 <= shadow_j < illegal_window.shape[1]):
                    illegal_window[shadow_i, shadow_j] = True
                    illegal_sum += kde_pilot[i,j]*kernel[shadow_i, shadow_j]                       
                    kernel[shadow_i, shadow_j] = 0 #setting the kernel to zero in the shadowed cells

        weighted_kernel = kernel * (kde_pilot[i,j] + illegal_sum) #adding the shadowed cell weight to the non-zero cells
        #else:
        #    weighted_kernel = kernel * kde_pilot[i,j]

        # Add contribution
        n_u[i_min:i_max, j_min:j_max] += weighted_kernel[
            max(0, kernel_size - i):kernel_size + min(gridsize_x - i, kernel_size + 1),
            max(0, kernel_size - j):kernel_size + min(gridsize_y - j, kernel_size + 1)
        ]
    
    return n_u

def grid_proj_kde(grid_x, 
                  grid_y, 
                  kde_pilot, 
                  gaussian_kernels, 
                  kernel_bandwidths, 
                  cell_bandwidths, 
                  illegal_cells=None):
    """
    Project kernel density estimation onto a 2D grid with optimized memory layout and Numba acceleration.
    
    Parameters
    ----------
    grid_x : array-like
        X-coordinates of the grid points
    grid_y : array-like
        Y-coordinates of the grid points
    kde_pilot : ndarray
        Pilot kernel density estimate on the grid
    gaussian_kernels : list of ndarrays
        Pre-computed Gaussian kernels for different bandwidths
    kernel_bandwidths : ndarray
        Bandwidths corresponding to pre-computed kernels
    cell_bandwidths : ndarray
        Bandwidth values for each cell in the grid
    illegal_cells : ndarray, optional
        Boolean mask of illegal/blocked cells (True = blocked)
        
    Returns
    -------
    ndarray
        Updated kernel density estimate with illegal cell handling
        
    Notes
    -----
    - Uses contiguous memory layout for performance
    - Handles illegal cells by redistributing their weights
    - Optimized with Numba for parallel processing
    - Only processes non-zero pilot KDE values
    """
    
    # Initialize illegal cells if None
    if illegal_cells is None:
        illegal_cells = np.zeros((len(grid_x), len(grid_y)), dtype=np.bool_)
    
    # Get grid sizes
    gridsize_x, gridsize_y = len(grid_x), len(grid_y)
    
    # Convert gaussian_kernels to homogeneous float64 arrays
    gaussian_kernels = [np.ascontiguousarray(kernel, dtype=np.float64) for kernel in gaussian_kernels]
    
    # Pre-compute non-zero indices
    non_zero_indices = np.array(np.where(kde_pilot > 0)).T
    
    # Convert inputs to contiguous arrays with consistent dtypes
    kde_pilot = np.ascontiguousarray(kde_pilot, dtype=np.float64)
    cell_bandwidths = np.ascontiguousarray(cell_bandwidths, dtype=np.float64)
    kernel_bandwidths = np.ascontiguousarray(kernel_bandwidths, dtype=np.float64)
    illegal_cells = np.ascontiguousarray(illegal_cells, dtype=np.bool_)
    
    # Process kernels using numba-optimized function
    n_u = _process_kernels(non_zero_indices, kde_pilot, cell_bandwidths,
                          kernel_bandwidths, gaussian_kernels, illegal_cells,
                          gridsize_x, gridsize_y)
    
    return n_u

def generate_gaussian_kernels(num_kernels, ratio, stretch=1):
    """
    Generates Gaussian kernels and their bandwidths. The function generates a kernel with support
    equal to the bandwidth multiplied by the ratio and the ratio sets the "resolution" of the 
    gaussian bandwidth family, i.e. ratio = 1/3 means that one kernel will be created for 0.33, 0.66, 1.0 etc.
    The kernels are stretched in the x-direction by the stretch factor.


    Parameters:
    num_kernels (int): The number of kernels to generate.
    ratio (float): The ratio between the kernel bandwidth and integration support.
    stretch (float): The stretch factor of the kernels. Defined as the ratio between the bandwidth in the x and y directions.

    Returns:
    gaussian_kernels (list): List of Gaussian kernels.
    bandwidths_h (np.array): Array of bandwidths associated with each kernel.
    kernel_origin (list): List of kernel origins.
    """

    gaussian_kernels = [np.array([[1]])]
    bandwidths_h = np.zeros(num_kernels)
    #kernel_origin = [np.array([0, 0])]

    for i in range(1, num_kernels):
        a = np.arange(-i, i + 1, 1).reshape(-1, 1)
        b = np.arange(-i, i + 1, 1).reshape(1, -1)
        h = (i * ratio) #+ ratio * len(a) #multiply with 2 here, since it goes in all directions (i.e. the 11 kernel is 22 wide etc.). 
        #impose stretch and calculate the kernel
        h_a = h*stretch
        h_b = h
        kernel_matrix = ((1 / (2 * np.pi * h_a * h_b)) * np.exp(-0.5 * ((a / h_a) ** 2 + (b / h_b) ** 2)))
        #append the kernel matrix and normalize (to make sure the sum of the kernel is 1)
        gaussian_kernels.append(kernel_matrix / np.sum(kernel_matrix))
        bandwidths_h[i] = h
        #kernel_origin.append(np.array([0, 0]))

    return gaussian_kernels, bandwidths_h#, kernel_origin

@jit(nopython=True, parallel=True)
def compute_adaptive_bandwidths(preGRID_active_padded, 
                                preGRID_active_counts_padded,
                            window_size, 
                            stats_threshold, 
                            grid_cell_size=1):
    """
    Compute adaptive bandwidths for all non-zero grid cell adaptation windows in the grid.
    
    Input:
    -----------
    preGRID_active_padded : np.ndarray
        Padded grid of active particles
    preGRID_active_counts_padded : np.ndarray
        Padded grid of particle counts
    window_size : int
        Size of the processing window
    stats_threshold : float
        Threshold for statistical calculations
    grid_cell_size : float (default=1) 

    Output:
    -----------
    std_estimate : np.ndarray
        Standard deviation estimate
    N_eff : np.ndarray
        Effective sample size estimate
    integral_length_scale_matrix : np.ndarray
        Integral length scale estimate
    h_matrix_adaptive : np.ndarray
        Adaptive bandwidth estimate

    """

    pad_size = window_size // 2
        
    # Type and shape checking
    shape = preGRID_active_padded.shape
    std_estimate = np.zeros((shape[0]-2*pad_size, shape[1]-2*pad_size), dtype=np.float64)
    N_eff = np.zeros_like(std_estimate)
    h_matrix_adaptive = np.zeros_like(std_estimate)
    integral_length_scale_matrix = np.zeros_like(std_estimate)
    
    # Main processing loop
    for row in prange(pad_size, shape[0]-pad_size):
        for col in range(pad_size, shape[1]-pad_size):
            if preGRID_active_counts_padded[row, col] > 0:
                # Extract data subset
                data_subset = preGRID_active_padded[
                    row-pad_size:row+pad_size+1,
                    col-pad_size:col+pad_size+1
                ]
                subset_counts = preGRID_active_counts_padded[
                    row-pad_size:row+pad_size+1,
                    col-pad_size:col+pad_size+1
                ]
                
                # Skip if center cell is empty
                if data_subset[pad_size,pad_size] == 0:
                    continue
                
                # Protect against zero division in normalization (this is normalizastion to 1)

                ### SOME NORMALIZATION STUFF NEEDS TO GO HERE SOMEWHERE ###
                total_weighted_sum = np.sum(data_subset)
                if total_weighted_sum > 0:
                    data_subset = (data_subset/total_weighted_sum)*subset_counts
                else:
                    continue
                
                row_idx = row - pad_size
                col_idx = col - pad_size
                
                # Process statistics with zero protection
                total_counts = np.sum(subset_counts) #Alternative if the number of particles is very low. 
                if total_counts < stats_threshold:
                    std = window_size/4 #np.sqrt(total_weighted_sum) represents the P
                    integral_length_scale = window_size/2 #One dimensional integral length scale assuming L is np.sqrt(total_weighted_sum)
                    n_eff = np.sum(data_subset)/window_size #One dimensional effective sample size assuming L is np.sqrt(total_weighted_sum)
                else:
                    std = max(histogram_std(data_subset, None, 1), 1e-10)
                    autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
                    autocorr = (autocorr_rows + autocorr_cols) / 2
                    
                    if autocorr.any():
                        non_zero_idx = np.where(autocorr != 0)[0]

                        if len(non_zero_idx) > 0:
                            denominator = autocorr[non_zero_idx[0]]
                            if denominator < 1e-10:
                                denominator = 1e-10

                            integral_length_scale = (np.sum(autocorr)*grid_cell_size) / denominator
                            
                        else:
                            integral_length_scale = 1e-10
                    else:
                        integral_length_scale = 1e-10
                    
                    #Scaled integral length scale
                    #integral_length_scale_scaled = integral_length_scale / grid_cell_size

                   
                    if denominator < 1e-10:
                        denominator = 1e-10
                    n_eff = np.sum(data_subset) / integral_length_scale_scaled
                
                # Calculate bandwidth with protection
                #get dimensionality of the data
                dim = len(data_subset.shape)
                h = std*(4/(dim+2))**(1/(dim+4))*n_eff**(-1/(dim+4)) #square root?
                
                # Store results
                std_estimate[row_idx, col_idx] = std
                N_eff[row_idx, col_idx] = n_eff
                integral_length_scale_matrix[row_idx, col_idx] = integral_length_scale
                h_matrix_adaptive[row_idx, col_idx] = h
    
    return std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive

def histogram_estimator(x_pos, y_pos, grid_x, grid_y, bandwidths=None, weights=None):
    '''
    Input:
    x_pos (np.array): x-coordinates of the particles
    y_pos (np.array): y-coordinates of the particles
    grid_x (np.array): grid cell boundaries in the x-direction
    grid_y (np.array): grid cell boundaries in the y-direction
    bandwidths (np.array): bandwidths of the particles
    weights (np.array): weights of the particles

    Output:
    total_weight: np.array of shape (grid_size, grid_size)
    particle_count: np.array of shape (grid_size, grid_size)
    average_bandwidth: np.array of shape (grid_size, grid_size)
    '''

    # Get size of grid in x and y direction
    grid_size_x = len(grid_x)
    grid_size_y = len(grid_y)

    # Initialize the histograms
    particle_count = np.zeros((grid_size_x, grid_size_y), dtype=np.int32)
    total_weight = np.zeros((grid_size_x, grid_size_y), dtype=np.float64)
    cell_bandwidth = np.zeros((grid_size_x, grid_size_y), dtype=np.float64)
    
    # Normalize the particle positions to the grid
    grid_x0 = grid_x[0]
    grid_y0 = grid_y[0]
    grid_x1 = grid_x[1]
    grid_y1 = grid_y[1]
    x_pos = (x_pos - grid_x0) / (grid_x1 - grid_x0)
    y_pos = (y_pos - grid_y0) / (grid_y1 - grid_y0)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(x_pos) & ~np.isnan(y_pos)
    x_pos = x_pos[valid_mask]
    y_pos = y_pos[valid_mask]
    weights = weights[valid_mask]
    bandwidths = bandwidths[valid_mask]
    
    # Convert positions to integer grid indices
    x_indices = x_pos.astype(np.int32)
    y_indices = y_pos.astype(np.int32)
    
    # Boundary check
    valid_mask = (x_indices >= 0) & (x_indices < grid_size_x) & (y_indices >= 0) & (y_indices < grid_size_y)
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    weights = weights[valid_mask]
    bandwidths = bandwidths[valid_mask]
    
    # Accumulate weights and counts
    np.add.at(total_weight, (x_indices, y_indices), weights) #This is just the mass in each cell
    np.add.at(particle_count, (x_indices, y_indices), 1)
    np.add.at(cell_bandwidth, (x_indices, y_indices), bandwidths * weights)

    cell_bandwidth = np.divide(cell_bandwidth, total_weight, out=np.zeros_like(cell_bandwidth), where=total_weight!=0)

    return total_weight, particle_count, cell_bandwidth

#Numba variant - faster but gives some errors in the results.. It is approx 4 times faster than the numpy version
#I keep it here in case I want to fix it later... 
@jit(parallel=True,nopython=True)
def histogram_estimator_numba(x_pos,y_pos,grid_x,grid_y,bandwidths = None,weights=None):
    '''
    Input:
    x_pos (np.array): x-coordinates of the particles
    y_pos (np.array): y-coordinates of the particles
    grid_x (np.array): grid cell boundaries in the x-direction
    grid_y (np.array): grid cell boundaries in the y-direction

    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    total_weight: np.array of shape (grid_size, grid_size)
    average_bandwidth: np.array of shape (grid_size, grid_size)
    '''

    #get size of grid in x and y direction
    grid_size_x = len(grid_x)
    grid_size_y = len(grid_y)

    # Initialize the histograms
    particle_count = np.zeros((grid_size_x, grid_size_y), dtype=np.int32)
    total_weight = np.zeros((grid_size_x,grid_size_y), dtype=np.float64)
    sum_bandwidth = np.zeros((grid_size_x,grid_size_y), dtype=np.float64)
    
    #Normalize the particle positions to the grid
    x_pos = (x_pos - grid_x[0])/(grid_x[1]-grid_x[0])
    y_pos = (y_pos - grid_y[0])/(grid_y[1]-grid_y[0])
    
    # Create a 2D histogram of particle positions
    for i in prange(len(x_pos)):
        if np.isnan(x_pos[i]) or np.isnan(y_pos[i]):
            continue
        x = int(x_pos[i])
        y = int(y_pos[i])
        if x >= grid_size_x or y >= grid_size_y or x < 0 or y < 0: #check if the particle is outside the grid
            continue
        total_weight[y, x] += weights[i] #This is just the mass in each cell
        particle_count[y, x] += 1
        sum_bandwidth[y, x] += bandwidths[i]*weights[i] #weighted sum of bandwidths
    
    #print(np.shape(particle_count))

    return particle_count, total_weight, sum_bandwidth 

@jit(nopython=True)
def histogram_std(binned_data, effective_samples=None, bin_size=1):
    '''Calculate the simple variance of the binned data
    using normal Bessel correction for unweighted samples adding Sheppards correction for weighted samples
    calculates standard deviation for weigthed data assuming reliability weights and 
    applying Bessels correction accordingly. Checks this automatically by comparing the sum of the binned data
    with the effective samples.

    Input:
    binned_data: np.array
        The binned data
    effective_samples: float
        The number of effective samples
    bin_size: float
        The size of the bins

    Output:
    float: The standard deviation of the binned data
    '''
    #Calculate the effective number of particles using Kish's formula. (for weighted data)
    if effective_samples == None:
        effective_samples = np.sum(binned_data)**2/np.sum(binned_data**2) #This is Kish's formula

    # Check that there's data in the binned data
    if np.sum(binned_data) == 0:
        return 0

    # Ensure effective_samples is larger than 1

    grid_size = len(binned_data)
    X = np.arange(0, grid_size * bin_size, bin_size)
    Y = np.arange(0, grid_size * bin_size, bin_size)
    
    sum_data = np.sum(binned_data)
    mu_x = np.sum(binned_data * X) / sum_data
    mu_y = np.sum(binned_data * Y) / sum_data
    
    #Sheppards correction term
    sheppard = (1/12)*bin_size*bin_size #weighted data

    #variance = (np.sum(binned_data*((X-mu_x)**2+(Y-mu_y)**2))/(sum_data-1))-2/12*bin_size*bin_size

    #Do Bessel correction for weighted binned data using Kish's formula and add Sheppards correction
    variance = (np.sum(binned_data * ((X - mu_x)**2 + (Y - mu_y)**2)) / sum_data) * \
            (1/(1 - 1/max(effective_samples,1.0000001))) - sheppard #Sheppards correction
    #sheppards: https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
    
    return np.sqrt(variance)

@jit(nopython=True)
def calculate_autocorrelation(data):
    """
    Calculate spatial autocorrelation along rows and columns of 2D data.

    Computes the autocorrelation function separately for rows and columns of a 2D array,
    using a vectorized implementation optimized with Numba. The autocorrelation is normalized
    by the number of points and includes protection against zero division.

    Parameters
    ----------
    data : ndarray
        2D input array for which to calculate autocorrelation

    Returns
    -------
    autocorr_rows : ndarray
        1D array containing autocorrelation values for row-wise shifts
    autocorr_cols : ndarray
        1D array containing autocorrelation values for column-wise shifts

    Notes
    -----
    - Uses Numba JIT compilation
    - Handles edge cases (small arrays, zero values)
    - Maximum lag is determined by smallest dimension
    - Includes epsilon protection against zero division
    - Returns single zero value arrays if input is too small
    """

    num_rows, num_cols = data.shape
    max_lag = min(num_rows, num_cols) - 1

    # Ensure we have enough data points
    if num_rows < 2 or num_cols < 2:
        return np.array([0.0]), np.array([0.0])
    
    autocorr_rows = np.zeros(max_lag)
    autocorr_cols = np.zeros(max_lag)
    
    # Precompute denominators
    #row_denominators = 1.0 / np.arange(num_cols - 1, num_cols - max_lag - 1, -1)
    #col_denominators = 1.0 / np.arange(num_rows - 1, num_rows - max_lag - 1, -1)

    # Protect against zero division in denominators
    row_range = np.arange(num_cols - 1, num_cols - max_lag - 1, -1)
    col_range = np.arange(num_rows - 1, num_rows - max_lag - 1, -1)
    
    # Add small epsilon to prevent division by zero
    row_denominators = 1.0 / np.maximum(row_range, 1e-10)
    col_denominators = 1.0 / np.maximum(col_range, 1e-10)
    
    # Vectorized autocorrelation calculation
    for k in range(0, max_lag + 1):
        row_sum = 0.0
        col_sum = 0.0
        
        for i in range(num_rows):
            row_sum += np.sum(data[i, :num_cols-k] * data[i, k:])
        for j in range(num_cols):
            col_sum += np.sum(data[:num_rows-k, j] * data[k:, j])
            
        autocorr_rows[k] = row_sum * row_denominators[k] / max(num_rows, 1e-10)
        autocorr_cols[k] = col_sum * col_denominators[k] / max(num_cols, 1e-10)
    
    return autocorr_rows, autocorr_cols


@jit(nopython=True)
def identify_shadowed_cells(x0, y0, xi, yj, legal_grid):
    """
    Identify shadowed cells by tracing from edges inward.
    Cells start as potentially shadowed and are marked free 
    if they have line of sight to kernel center.
    """
    grid_size = legal_grid.shape[0]
    # Start with all cells potentially shadowed
    shadowed = np.ones((grid_size, grid_size), dtype=np.bool_)
    
    # Trace from edges
    for edge_x in [0, grid_size-1]:
        for y in range(grid_size):
            los_cells = bresenham(x0,y0,edge_x, y)
            # Mark cells as free until hitting illegal cell
            for cell in los_cells:
                if not legal_grid[cell[0], cell[1]]:
                    break
                shadowed[cell[0], cell[1]] = False
                
    for edge_y in [0, grid_size-1]:
        for x in range(grid_size):
            los_cells = bresenham(x0, y0, x, edge_y)
            for cell in los_cells:
                if not legal_grid[cell[0], cell[1]]:
                    break
                shadowed[cell[0], cell[1]] = False
    
    # Convert to list format
    shadowed_cells = []
    for i in range(len(xi)):
        for j in range(len(yj)):
            if shadowed[xi[i], yj[j]]:
                shadowed_cells.append((xi[i], yj[j]))
                
    return shadowed_cells

#Make a bresenham line
@jit(nopython=True)
def bresenham(x0, y0, x1, y1): 
    """
    Bresenham's Line Algorithm to generate points between (x0, y0) and (x1, y1)

    Intput:
    x0: x-coordinate of the starting point
    y0: y-coordinate of the starting point
    x1: x-coordinate of the ending point
    y1: y-coordinate of the ending point

    Output:
    points: List of points between (x0, y0) and (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1 # Step direction for x
    sy = 1 if y0 < y1 else -1 # Step direction for y
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

#This function is also not used, but keep it here in case I want to make this work someday...
def reflect_with_shadow(x, y, xi, yj, legal_grid):
    """
    Helper function to reflect (xi, yj) back to a legal position
    across the barrier while respecting the shadow.
    """
    x_reflect, y_reflect = xi, yj

    # Reflect along x-axis if needed
    while not legal_grid[x_reflect, yj] and x_reflect != x:
        x_reflect += np.sign(x - xi)  # Step towards the particle

    # Reflect along y-axis if needed
    while not legal_grid[xi, y_reflect] and y_reflect != y:
        y_reflect += np.sign(y - yj)  # Step towards the particle
    
    # Check final reflection position legality
    if legal_grid[x_reflect, y_reflect]:
        return x_reflect, y_reflect
    else:
        return None, None  # No valid reflection found



# -------------------------------------------------- #





# -------------------------------------------------- #
######################################################
##################### INITIATION #####################
######################################################
# -------------------------------------------------- #

if __name__ == "__main__":

    time_start = time.time()

    create_data = True
    do_plotting = True

    frac_diff = 1000 #pick every 1000th particle for the test data
    grid_size = 120
    grid_size_plot = 100
    grid_x = np.linspace(0, grid_size, grid_size)
    grid_y = np.linspace(0, grid_size, grid_size)
    
    if create_data == True:

        #Define illegal grid cells
        illegal_cells = np.zeros((grid_size,grid_size))
        #illegal_cells[55:65,70:85] = 1

        a = 40
        b = 25
        x0 = 55
        y0 = 95
        for i in range(grid_size):
            for j in range(grid_size):
                if ((i-x0)/a)**2 + ((j-y0)/b)**2 <= 1:
                    illegal_cells[i,j] = 1


        illegal_positions = illegal_cells.astype(bool)

        #define illegal hollow ellipse for the boundary testing

        illegal_hollow_ellipse = illegal_cells.copy()
        a = 40-5
        b = 25-5
        x0 = 55
        y0 = 95
        for i in range(grid_size):
            for j in range(grid_size):
                if ((i-x0)/a)**2 + ((j-y0)/b)**2 <= 1:
                    illegal_hollow_ellipse[i,j] = 0

        illegal_positions_hollow_ellipse = illegal_hollow_ellipse.astype(bool)

        # Generate test data with illegal positions
        trajectories, bw = pdg.create_test_data(stdev=1.4, num_particles_per_timestep=5000, time_steps=380, dt=0.1, grid_size=100, illegal_positions=illegal_positions)

        trajectories_test = trajectories[::frac_diff]
        #Normalize the weights
        weights_test = np.ones(len(trajectories_test))*len(trajectories)/len(trajectories_test)



    #Create histogram estimate
    ground_truth, particle_count, cell_bandwidth = histogram_estimator(trajectories[:,0],trajectories[:,1],grid_x,grid_y,bandwidths=bw,weights=np.ones(len(trajectories)))
    histogram_estimate,particle_count, cell_bandwidth = histogram_estimator(trajectories_test[:,0],trajectories_test[:,1],grid_x,grid_y,bandwidths=np.ones(len(trajectories_test)),weights=weights_test)

    #######################################
    ############ SILVERMAN KDE ############
    #######################################

    '''
    Silverman's KDE using the gaussian_kde from scipy.stats
    '''

    kde = gaussian_kde(trajectories_test[~np.isnan(trajectories_test).any(axis=1)].T, 
                       bw_method='silverman')
    x = np.linspace(0, 120, 120)
    y = np.linspace(0, 120, 120)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.flatten(), Y.flatten()])).reshape(X.shape)
    kde_silverman_naive = Z* np.mean(weights_test) * len(trajectories_test)
    kde_silverman_naive = kde_silverman_naive.T



    #######################################
    ############ AKDE ESTIMATE ############
    #######################################

    '''
    Adaptive Kernel Density Estimate

    This involves the following steps:
    1. Determine the pilot KDE (histogram estimate)
    2. Create the Gaussian kernels
    3. Estimate the bandwidths h
    4. Do the density estimate
    '''


    # ###- Pilot KDE -### #


    particle_initial_bandwidths = np.ones(len(trajectories_test)) #not relevant unless using time dependent bandwidths
    pilot_kde,pilot_kde_counts,pilot_kde_bandwidths = histogram_estimator(trajectories_test[:,0],
                                                                          trajectories_test[:,1],
                                                                          grid_x,
                                                                          grid_y,
                                                                          bandwidths=particle_initial_bandwidths,
                                                                          weights = weights_test)

    # ###- Gaussian kernels -### #

    num_kernels = 20 #This is the number of kernels to use
    ratio = 1/3 #This is the ratio between the kernel bandwidth and the support

    # Generate 20 kernels with bandwidths from 1/3 to 20/3 and support from 1 to 20
    gaussian_kernels, bandwidths_h = generate_gaussian_kernels(num_kernels, 
                                                               ratio) 



    # ###- Bandwidth h estimation -### #


    # Calculate integral length scale of the whole field to get rough size of adaptation window 
    autocorr_rows,autocorr_cols = calculate_autocorrelation(pilot_kde) #Calculate the autocorrelation along rows and columns
    autocorr = (autocorr_rows + autocorr_cols) / 2 #Average the autocorrelation
    integral_length_scale = np.sum(autocorr) / autocorr[np.argwhere(autocorr != 0)[0]] #Calculate the integral length scale
    
    # Set limits for the adaptation window and make sure it's odd (for it to have a center)
    adapt_window_size = int(np.clip(integral_length_scale.item(), 5, grid_size/4))
    if adapt_window_size % 2 == 0:
        adapt_window_size += 1 #Make sure it's odd.
    adapt_window_size = np.array([adapt_window_size])

    # Pad the pilot KDE to avoid edge issues ###
    pad_size = adapt_window_size // 2
    pilot_kde_padded = np.pad(pilot_kde, pad_size, mode='reflect')
    pilot_kde_counts_padded = np.pad(pilot_kde_counts, pad_size, mode='reflect')

    # Threshold for statistical calculations
    stats_threshold = adapt_window_size[0]

    # Compute statistics and bandwidths
    std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive = compute_adaptive_bandwidths(
                        pilot_kde_padded, pilot_kde_counts_padded,
                        adapt_window_size[0], stats_threshold
                    )

    # ###- Do the KDE estimate -### #
    akde_estimate = grid_proj_kde(grid_x,
                                    grid_y,
                                    pilot_kde,
                                    gaussian_kernels,
                                    bandwidths_h,
                                    h_matrix_adaptive,
                                    illegal_cells=illegal_positions_hollow_ellipse)

    


    # ------------------------------------------------ #
    ####################################################
    ##################### PLOTTING #####################
    ####################################################
    # ------------------------------------------------ #

    if do_plotting == True:

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        #set plotting style
        plotting_style = 'light'
        if plotting_style == 'light':
            plt.style.use('default')
            cmap1 = 'plasma'
            cmap2 = 'Spectral'
        else:
            plt.style.use('dark_background')
            cmap1 = 'magma'
            # Create custom colormap: red -> black -> blue
            colors_list = ['red', 'black', 'blue']
            n_bins = 100  # Number of color gradations
            cmap2 = colors.LinearSegmentedColormap.from_list("custom", colors_list, N=n_bins)
            

        ##### PLOT TRAJECTORIES #####

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(trajectories[:,1],trajectories[:,0],'.')
        plt.title('Full data')
        plt.subplot(1,2,2)
        plt.plot(trajectories_test[:,1],trajectories_test[:,0],'.')
        plt.title('Test data')
        plt.show()

        ##### PLOT RESULTS AND RESIDUALS #####

        # Create figure with 4x2 layout
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3)

        # Find common color range for density plots
        vmin = 0
        vmax = np.max(ground_truth)
        levels = np.linspace(vmin, vmax, 100)

        # Top row - density plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # AKDE plot
        pcm1 = ax1.pcolor(grid_x, grid_y, akde_estimate, vmin=vmin, vmax=vmax,cmap=cmap1)
        ax1.contour(grid_x, grid_y, akde_estimate, levels[::2], colors='white',linewidths=0.2,alpha=0.5)
        #plot illegal cells using patch
        for i in range(grid_size):
            for j in range(grid_size):
                if illegal_positions[i,j]:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='grey', alpha=0.2))
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 100])
        ax1.set_title('Adaptive KDE')

        # HE plot
        ax2.pcolor(grid_x, grid_y, pilot_kde, vmin=vmin, vmax=vmax,cmap=cmap1)
        ax2.contour(grid_x, grid_y, pilot_kde, levels[::2], colors='white',linewidths=0.2,alpha=0.5)   
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 100])
        ax2.set_title('Histogram Estimate')

        # Silverman KDE plot
        ax3.pcolor(grid_x, grid_y, kde_silverman_naive, vmin=vmin, vmax=vmax,cmap=cmap1)
        ax3.contour(grid_x, grid_y, kde_silverman_naive, levels[::2], colors='white',linewidths=0.2,alpha=0.5) 
        ax3.set_xlim([0, 100])
        ax3.set_ylim([0, 100])
        ax3.set_title('Silverman KDE')

        # GT plot
        ax4.pcolor(grid_x, grid_y, ground_truth, vmin=vmin, vmax=vmax,cmap=cmap1)
        ax4.contour(grid_x, grid_y, ground_truth, levels[::2], colors='white',linewidths=0.2,alpha=0.5)    
        ax4.set_xlim([0, 100])
        ax4.set_ylim([0, 100])
        ax4.set_title('Ground Truth')

        # Add density colorbar
        cbar1 = fig.colorbar(pcm1, ax=[ax1, ax2, ax3, ax4], label='Density')

        # Bottom row - residual plots and statistics
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])

        # Compute residuals
        akde_residuals = akde_estimate - ground_truth
        he_residuals = pilot_kde - ground_truth
        silverman_residuals = kde_silverman_naive - ground_truth

        # Find common residual color range
        res_max = max(abs(akde_residuals).max(), abs(he_residuals).max(), abs(silverman_residuals).max())/1.5
        res_min = -res_max

        # Use in plotting
        pcm2 = ax5.pcolor(grid_x, grid_y, akde_residuals, vmin=res_min, vmax=res_max, cmap=cmap2)

        # Plot residuals
        pcm2 = ax5.pcolor(grid_x, grid_y, akde_residuals, vmin=res_min, vmax=res_max, cmap=cmap2)
        ax5.set_title('AKDE Residuals')

        ax6.pcolor(grid_x, grid_y, he_residuals, vmin=res_min, vmax=res_max, cmap=cmap2)
        ax6.set_title('HE Residuals')

        ax7.pcolor(grid_x, grid_y, silverman_residuals, vmin=res_min, vmax=res_max, cmap=cmap2)
        ax7.set_title('Silverman Residuals')

        # Add residuals colorbar
        cbar2 = fig.colorbar(pcm2, ax=[ax5, ax6, ax7], label='Residuals',extend='both')

        # Calculate R² scores
        r2_akde = np.corrcoef(ground_truth.flatten(), akde_estimate.flatten())[0, 1]**2 #R² score for AKDE
        r2_he = np.corrcoef(ground_truth.flatten(), pilot_kde.flatten())[0, 1]**2 #R² score for HE
        r2_silverman = np.corrcoef(ground_truth.flatten(), kde_silverman_naive.flatten())[0, 1]**2 #R² score for Silverman  

        # Calculate max values
        max_akde = np.max(akde_estimate)
        max_he = np.max(pilot_kde)
        max_silverman = np.max(kde_silverman_naive)
        max_gt = np.max(ground_truth)

        # Calculate total sums (field integral)
        sum_akde = np.sum(akde_estimate)
        sum_he = np.sum(pilot_kde)
        sum_silverman = np.sum(kde_silverman_naive)
        sum_gt = np.sum(ground_truth)

        # Create textbox with all metrics
        r2_text = (f'R² Scores:\n'
                f'AKDE: {r2_akde:.4f}\n'
                f'HE: {r2_he:.4f}\n'
                f'Naive silverman: {r2_silverman:.4f}\n\n'
                f'Maximum Values:\n'
                f'AKDE: {max_akde:.4f}\n'
                f'HE: {max_he:.4f}\n'
                f'Silverman: {max_silverman:.4f}\n'
                f'Ground Truth: {max_gt:.4f}\n\n'
                f'Field Integrals:\n'
                f'AKDE: {sum_akde:.4f}\n'
                f'HE: {sum_he:.4f}\n'
                f'Silverman: {sum_silverman:.4f}\n'
                f'Ground Truth: {sum_gt:.4f}')

        ax8.text(-0.2, 0.2, r2_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.0))
        ax8.axis('off')

        plt.show()



        ########## PLOTTING HISTOGRAMS ##########

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #HISTOGRAM PLOTS!!
        axs[0, 0].hist(std_estimate.flatten()[std_estimate.flatten()>0], bins=50, color='b', alpha=0.7)
        axs[0, 0].set_title('Standard deviation estimates')

        axs[0, 1].hist(N_eff.flatten()[N_eff.flatten()>0], bins=50, color='r', alpha=0.7)
        axs[0, 1].set_title('Effective sample size estimates')

        axs[1, 0].hist(integral_length_scale_matrix.flatten()[integral_length_scale_matrix.flatten()>0], bins=50, color='g', alpha=0.7)
        axs[1, 0].set_title('Integral length scale estimates')

        axs[1, 1].hist(h_matrix_adaptive.flatten()[h_matrix_adaptive.flatten()>0], bins=50, color='y', alpha=0.7)
        axs[1, 1].set_title('Adaptive bandwidth estimates')

        plt.show()

    time_end = time.time()

    print('Time elapsed: ', time_end-time_start)


    ########### TESTING ###########
    testing = False
    if testing == True:
        #RUN SIMPLE TEST WITH 1 KERNEL
        #craete a dataset with only 1 particle located at 60,60 and a bandwidth of 1.0
        test_data = np.array([[28,73]])
        test_bw = np.array([7])
        test_weights = np.array([1.0])
        #use the same grid as the pilot KDE
        grid_x = np.arange(0, 100, 1)
        grid_y = np.arange(0, 100, 1)


        #create the pilot KDE
        pilot_kde_weight,pilot_kde_count,pilot_kde_average_bandwidth = histogram_estimator(test_data[:,0], test_data[:,1], grid_x, grid_y, test_bw, test_weights)
        kde = grid_proj_kde(grid_x, grid_y, pilot_kde_weight, gaussian_kernels, bandwidths_h  ,pilot_kde_average_bandwidth , illegal_cells=illegal_cells[:100,:100])  


        plt.figure()
        plt.imshow(kde)#, cmap=cmap1)
        #draw patch for the illegal cells
        #plt.imshow(illegal_positions_hollow_ellipse[:100,:100], cmap='gray', alpha=0.2)
        #add patch of the hollow ellipse
        for i in range(100):
            for j in range(100):
                if illegal_positions_hollow_ellipse[i,j]:
                    plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='grey', alpha=0.2))

