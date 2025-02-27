"""
2D Particle Dispersion Test Data Generator

This script generates synthetic particle dispersion data for testing adaptive kernel 
density estimation (AKDE) algorithms. It simulates particle trajectories in a 2D domain.
The generated data serves as test data for the akd_estimator, which performs
adaptive kernel density estimation on particle dispersion data. 

The simulator has the following features:

- Particles follow advection-diffusion dynamics
- Handles illegal positions (obstacles) via nearest-neighbor mapping
- Generates both trajectories and bandwidth values for time dependent/diffusion based bandwidth kernel density estimation
- Stochastic diffusion
- Configurable simulation parameters
- Periodic velocity field with conservation of mass

Dependencies:
- numpy
- scipy (KDTree for obstacle handling)
- matplotlib (for test plotting)

Author: Knut Ola Dølven
Date: 2024
"""

import numpy as np
from scipy.spatial import KDTree

# ########################################## #
# #### FUNCTIONS FOR CREATING TEST DATA #### #
# ########################################## #

def update_positions(particles, U_a, stdev, dt):
    '''
    Input:
    particles: np.array of shape (num_particles, 2)
    U_a: np.array of shape (2,)

    '''
    # Advective term
    advective_displacement = U_a * dt
    # Stochastic term
    stochastic_displacement = np.random.normal(0, stdev, particles.shape) * np.sqrt(dt)
    # Update positions
    particles += advective_displacement + stochastic_displacement
    #particles = np.mod(particles, grid_size)
    return particles

def update_positions(particles, U_field, stdev, dt):
    """Update particle positions using interpolated velocity field."""
    new_positions = particles.copy()
    
    for i in range(len(particles)):
        if not np.any(np.isnan(particles[i])):
            velocity = U_field[int(particles[i][1]), int(particles[i][0])]
            advective_displacement = velocity * dt
            stochastic_displacement = np.random.normal(0, stdev, 2) * np.sqrt(dt)
            new_positions[i] += advective_displacement + stochastic_displacement
    
    return new_positions

def create_2d_velocity_field(grid_size):
    """Create a 2D velocity field that varies in space and time.
    
    Returns
    -------
    U_field : ndarray
        Array of shape (time_steps, grid_size, grid_size, 2) containing
        velocity vectors [u, v] for each grid point and time
    """
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    U_field = np.zeros((grid_size, grid_size, 2))
    
    # Create spatially varying velocity field
    # U component (x direction)
    U_field[:, :, 0] = 2 + np.sin(X/2 + 50) * np.cos(Y/2)
    # V component (y direction)
    U_field[:, :, 1] = 1 + np.cos(X/2) * np.sin(Y/2 + 50)
    
    # Normalize to maintain constant magnitude
    magnitude = np.sqrt(U_field[:, :, 0]**2 + U_field[:, :, 1]**2)
    U_field[:, :, 0] /= magnitude
    U_field[:, :, 1] /= magnitude
    
    # Scale to desired magnitude
    U_field *= 5  # Same magnitude as original
    
    return U_field

def get_velocity_at_position(position, U_field, grid_size):
    """Get interpolated velocity at particle position."""
    x, y = position
    
    # Convert position to grid indices
    x_idx = np.clip(int(x), 0, grid_size-2)
    y_idx = np.clip(int(y), 0, grid_size-2)
    
    # Get fractional position within grid cell
    dx = x - x_idx
    dy = y - y_idx
    
    # Bilinear interpolation
    v00 = U_field[y_idx, x_idx]
    v10 = U_field[y_idx+1, x_idx]
    v01 = U_field[y_idx, x_idx+1]
    v11 = U_field[y_idx+1, x_idx+1]
    
    velocity = (v00 * (1-dx)*(1-dy) + 
               v10 * (1-dx)*dy + 
               v01 * dx*(1-dy) + 
               v11 * dx*dy)
    
    return velocity

def create_test_data(stdev=1.4, 
                     num_particles_per_timestep=5000, 
                     time_steps=380, 
                     dt=0.1, 
                     grid_size=100,
                     illegal_positions=None):
    """Generate test data for particle dispersion simulation with obstacles.

    This function simulates particle trajectories in a 2D domain with optional illegal regions
    (obstacles). Particles are released from a fixed position and advected by a time-varying
    velocity field while experiencing stochastic diffusion. Particles that enter illegal regions
    are mapped to the nearest legal position.

    Parameters
    ----------
    stdev : float, default=1.4
        Standard deviation of the stochastic diffusion term
    num_particles_per_timestep : int, default=5000
        Number of particles released at each timestep
    time_steps : int, default=380
        Total number of simulation timesteps
    dt : float, default=0.1
        Timestep size in simulation units
    grid_size : int, default=100
        Size of the square simulation domain
    illegal_positions : ndarray, optional
        Boolean array of shape (grid_size, grid_size) marking illegal positions.
        True indicates position is illegal/obstacle.

    Returns
    -------
    trajectories : ndarray
        Array of shape (num_particles_per_timestep * time_steps, 2) containing
        particle trajectories
    bw : ndarray
        Array of shape (num_particles_per_timestep * time_steps,) containing
        bandwidth values for each particle

    Notes
    -----
    - Uses KDTree for efficient nearest-neighbor queries when mapping illegal particles
    - Velocity field varies sinusoidally in time while conserving magnitude
    - Bandwidth grows as sqrt(time) for each particle
    """
    # Create a true/false mask of illegal cells
    if illegal_positions is None:
        legal_cells = np.ones((grid_size, grid_size), dtype=bool)
        illegal_positions = np.zeros((grid_size, grid_size), dtype=bool)
    else:
        legal_cells = ~illegal_positions
    # Indices of legal cells
    legal_indices = np.argwhere(legal_cells)
    # Coordinates
    x_grid = np.arange(illegal_positions.shape[0])
    y_grid = np.arange(illegal_positions.shape[1])
    legal_coordinates = np.array([x_grid[legal_indices[:, 0]], y_grid[legal_indices[:, 1]]]).T
    tree = KDTree(legal_coordinates)

    # Release position
    release_position = np.array([10, 10])
    # Make U_a a periodic function with size time_steps
    U_a = [0, 5]  # Initial value
    # Initial magnitude
    magU = np.sqrt(U_a[0]**2 + U_a[1]**2)
    U_a = np.tile(U_a, (time_steps, 1))
    for i in range(1, time_steps):
        U_a[i][:][0] = 2 * magU + np.sin(i / 50) * 2 * magU
        # make it a bit more complex by adding another sine function with different frequency
        # U_a[i][:][1] = 2*magU+ np.sin(i/50)*2*magU + np.sin(i/10)*2*magU
        #print(np.sin(i / 10))
        # L2 normalize the velocity
        U_a[i] = (U_a[i] / (np.sqrt(U_a[i][0]**2 + U_a[i][1]**2))) * magU  # Conservation of mass

    # Simulate particle trajectories
    trajectories = np.zeros((num_particles_per_timestep * time_steps, 2)) * np.nan
    # Create the bandwidth vector for each particle
    bw = np.ones(num_particles_per_timestep * time_steps) * 0

    for t in range(time_steps - 1):
        if t == 0:
            # Initialize particle matrix at first timestep
            particles = np.ones([num_particles_per_timestep, 2]) * release_position
        else:
            particles_old = particles

            # Add particles to the particle array
            particles = np.ones([num_particles_per_timestep * (t + 1), 2]) * release_position
            # Add in the old particle positions to the new array
            particles[:num_particles_per_timestep * t] = particles_old
            # Set particles that have left the domain to nan
            # Update the bw vector

        print(np.shape(particles))
        particles = update_positions(particles, U_a[t], stdev, dt)

        # Reposition illegal particles
        p_x, p_y = particles[:, 0], particles[:, 1]
        valid_indices = ~np.isnan(p_x) & ~np.isnan(p_y) & (p_x >= 0) & (p_x < grid_size) & (p_y >= 0) & (p_y < grid_size)
        is_illegal = np.zeros(p_x.shape, dtype=bool)
        is_illegal[valid_indices] = ~legal_cells[p_x[valid_indices].astype(int), p_y[valid_indices].astype(int)]
        illegal_positions = particles[is_illegal]
        _, nearest_indices = tree.query(illegal_positions)
        mapped_positions = legal_coordinates[nearest_indices]
        particles[is_illegal, 0] = mapped_positions[:, 0]
        particles[is_illegal, 1] = mapped_positions[:, 1]

        trajectories[:len(particles)] = particles
        bw[:len(particles)] = bw[:len(particles)] + np.sqrt(stdev * 0.001)
        # Limit bw to a maximum value
        # bw[bw > 20] = 20

    return trajectories, bw

def plot_velocity_field(U_field, trajectories=None, skip=5):
    """Plot velocity field with optional particle positions."""
    plt.figure(figsize=(10, 10))
    
    # Create grid for quiver plot
    x = np.arange(0, U_field.shape[1])
    y = np.arange(0, U_field.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Plot velocity field
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              U_field[::skip, ::skip, 0], U_field[::skip, ::skip, 1],
              scale=50)
    
    # Plot particles if provided
    if trajectories is not None:
        valid_particles = ~np.isnan(trajectories[:, 0])
        plt.scatter(trajectories[valid_particles, 0], 
                   trajectories[valid_particles, 1],
                   c='r', s=1, alpha=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Velocity Field')



    plt.axis('equal')
    plt.show()





#-------------------------------------------------------#

###################
##### TESTING #####
###################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
   
    # Generate test data
    trajectories, bw = create_test_data()

    # Plot particle trajectories
    plt.figure(figsize=(6, 6))
    plt.scatter(trajectories[:, 0], trajectories[:, 1], s=1, c=bw, cmap='viridis')
    plt.colorbar(label='Bandwidth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Particle Trajectories')
    plt.show()
