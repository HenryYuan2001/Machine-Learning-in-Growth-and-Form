import jax
import jax.numpy as jnp
from jax import random, jit, grad, lax
import optax
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np

# Settings
N = 21  # Number of points in each dimension of the 2D mesh
I = 50  # Number of U variables for each point (U_1, U_2, ..., U_I)
num_iterations = 30000
M = 1200  # +1 to include the initial condition

# Check device
device = jax.devices()[0]
device_kind = device.device_kind

print(f"JAX is using: {device_kind}")

# Initialize Us with zeros
Us = jnp.zeros((M, N, N, I, 2))

# Set initial conditions
center = N // 2
Us = Us.at[0, center, center-1, :, 0].set(1)  # Set left peak for Ua
Us = Us.at[0, center, center+1, :, 1].set(1)  # Set right peak for Ub

# Function to add boundary points
def add_boundary_points(U):
    # Add zeros to all edges
    U_padded = jnp.pad(U, ((1, 1), (1, 1), (0, 0), (0, 0)), mode='constant')
    return U_padded

# Initialize the parameters (unchanged)
def init_params():
    key = random.PRNGKey(0)
    return {
        'w_a_unconstrained': random.normal(key, (I, I)),
        'w_b_unconstrained': random.normal(key, (I, I)),
        'w_c_unconstrained': random.normal(key, (I, I)),
        'w_d_unconstrained': random.normal(key, (I, I)),
        'Da_unconstrained': random.normal(key, ()),
        'Db_unconstrained': random.normal(key, ()),
    }

params = init_params()

# Constrain parameters (unchanged)
def constrain_params(params):
    return {
        'w_a': jax.nn.softplus(params['w_a_unconstrained']),
        'w_b': jax.nn.softplus(params['w_b_unconstrained']),
        'w_c': jax.nn.softplus(params['w_c_unconstrained']),
        'w_d': jax.nn.softplus(params['w_d_unconstrained']),
        'Da': jax.nn.softplus(params['Da_unconstrained']),
        'Db': jax.nn.softplus(params['Db_unconstrained']),
    }

@jit
def run_simulation(Us, params, dt):
    constrained_params = constrain_params(params)
    _, Us = lax.scan(diffusion_step, (Us[0], constrained_params), jnp.arange(1, M))
    return jnp.concatenate([Us[0][None, ...], Us], axis=0)

# New functions for target shapes
def create_ellipse(N, a, b):
    y, x = jnp.ogrid[-N//2:N//2, -N//2:N//2]
    return ((x/a)**2 + (y/b)**2 <= 1).astype(float)

def create_target_shapes(N):
    target_a = create_ellipse(N, N//5, N//2)  # Ellipse polarized in y direction
    target_b = create_ellipse(N, N//2, N//5)  # Ellipse polarized in x direction

    target_a_full = jnp.zeros((N, N, I, 2))
    target_b_full = jnp.zeros((N, N, I, 2))

    target_a_full = target_a_full.at[:, :, 0, 0].set(target_a)  # Us1, a part
    target_b_full = target_b_full.at[:, :, 0, 1].set(target_b)  # Us1, b part

    # Add boundary points
    target_a_with_boundary = add_boundary_points(target_a_full)
    target_b_with_boundary = add_boundary_points(target_b_full)

    return target_a_full, target_b_full, target_a_with_boundary, target_b_with_boundary

# Create target shapes
target_a, target_b, target_a_with_boundary, target_b_with_boundary = create_target_shapes(N)

# Modified loss function
@jit
def diffusion_step(carry, t):
    U_curr, params = carry
    w_a, w_b, w_c, w_d, Da, Db = params['w_a'], params['w_b'], params['w_c'], params['w_d'], params['Da'], params['Db']
    dt = 0.01
    # Compute w terms
    w_term_a = jnp.einsum('ij,nmj->nmi', w_a, U_curr[:, :, :, 0])
    w_term_b = jnp.einsum('ij,nmj->nmi', w_b, U_curr[:, :, :, 1])
    w_term_c = jnp.einsum('ij,nmj->nmi', w_c, U_curr[:, :, :, 0])
    w_term_d = jnp.einsum('ij,nmj->nmi', w_d, U_curr[:, :, :, 1])

    # Compute cubic terms
    U_cube_a = U_curr[:, :, :, 0] ** 3
    U_cube_b = U_curr[:, :, :, 1] ** 3

    # Initialize Laplacian arrays
    laplacian_a = jnp.zeros_like(U_curr[:, :, :, 0])
    laplacian_b = jnp.zeros_like(U_curr[:, :, :, 1])

    # Compute 2D Laplacian for interior points
    laplacian_a = laplacian_a.at[1:-1, 1:-1].set(
        U_curr[2:, 1:-1, :, 0] + U_curr[:-2, 1:-1, :, 0] +
        U_curr[1:-1, 2:, :, 0] + U_curr[1:-1, :-2, :, 0] -
        4 * U_curr[1:-1, 1:-1, :, 0]
    )
    laplacian_b = laplacian_b.at[1:-1, 1:-1].set(
        U_curr[2:, 1:-1, :, 1] + U_curr[:-2, 1:-1, :, 1] +
        U_curr[1:-1, 2:, :, 1] + U_curr[1:-1, :-2, :, 1] -
        4 * U_curr[1:-1, 1:-1, :, 1]
    )

    # Compute Laplacian for edge points (excluding corners)
    # Top edge
    laplacian_a = laplacian_a.at[0, 1:-1].set(
        U_curr[1, 1:-1, :, 0] + U_curr[0, 2:, :, 0] + U_curr[0, :-2, :, 0] - 4 * U_curr[0, 1:-1, :, 0]
    )
    laplacian_b = laplacian_b.at[0, 1:-1].set(
        U_curr[1, 1:-1, :, 1] + U_curr[0, 2:, :, 1] + U_curr[0, :-2, :, 1] - 4 * U_curr[0, 1:-1, :, 1]
    )

    # Bottom edge
    laplacian_a = laplacian_a.at[-1, 1:-1].set(
        U_curr[-2, 1:-1, :, 0] + U_curr[-1, 2:, :, 0] + U_curr[-1, :-2, :, 0] - 4 * U_curr[-1, 1:-1, :, 0]
    )
    laplacian_b = laplacian_b.at[-1, 1:-1].set(
        U_curr[-2, 1:-1, :, 1] + U_curr[-1, 2:, :, 1] + U_curr[-1, :-2, :, 1] - 4 * U_curr[-1, 1:-1, :, 1]
    )

    # Left edge
    laplacian_a = laplacian_a.at[1:-1, 0].set(
        U_curr[2:, 0, :, 0] + U_curr[:-2, 0, :, 0] + U_curr[1:-1, 1, :, 0] - 4 * U_curr[1:-1, 0, :, 0]
    )
    laplacian_b = laplacian_b.at[1:-1, 0].set(
        U_curr[2:, 0, :, 1] + U_curr[:-2, 0, :, 1] + U_curr[1:-1, 1, :, 1] - 4 * U_curr[1:-1, 0, :, 1]
    )

    # Right edge
    laplacian_a = laplacian_a.at[1:-1, -1].set(
        U_curr[2:, -1, :, 0] + U_curr[:-2, -1, :, 0] + U_curr[1:-1, -2, :, 0] - 4 * U_curr[1:-1, -1, :, 0]
    )
    laplacian_b = laplacian_b.at[1:-1, -1].set(
        U_curr[2:, -1, :, 1] + U_curr[:-2, -1, :, 1] + U_curr[1:-1, -2, :, 1] - 4 * U_curr[1:-1, -1, :, 1]
    )

    # Compute Laplacian for corner points
    # Top-left corner
    laplacian_a = laplacian_a.at[0, 0].set(U_curr[1, 0, :, 0] + U_curr[0, 1, :, 0] - 4 * U_curr[0, 0, :, 0])
    laplacian_b = laplacian_b.at[0, 0].set(U_curr[1, 0, :, 1] + U_curr[0, 1, :, 1] - 4 * U_curr[0, 0, :, 1])

    # Top-right corner
    laplacian_a = laplacian_a.at[0, -1].set(U_curr[1, -1, :, 0] + U_curr[0, -2, :, 0] - 4 * U_curr[0, -1, :, 0])
    laplacian_b = laplacian_b.at[0, -1].set(U_curr[1, -1, :, 1] + U_curr[0, -2, :, 1] - 4 * U_curr[0, -1, :, 1])

    # Bottom-left corner
    laplacian_a = laplacian_a.at[-1, 0].set(U_curr[-2, 0, :, 0] + U_curr[-1, 1, :, 0] - 4 * U_curr[-1, 0, :, 0])
    laplacian_b = laplacian_b.at[-1, 0].set(U_curr[-2, 0, :, 1] + U_curr[-1, 1, :, 1] - 4 * U_curr[-1, 0, :, 1])

    # Bottom-right corner
    laplacian_a = laplacian_a.at[-1, -1].set(U_curr[-2, -1, :, 0] + U_curr[-1, -2, :, 0] - 4 * U_curr[-1, -1, :, 0])
    laplacian_b = laplacian_b.at[-1, -1].set(U_curr[-2, -1, :, 1] + U_curr[-1, -2, :, 1] - 4 * U_curr[-1, -1, :, 1])

    # Compute dU
    dUa = w_term_a / (1 + w_term_b ** 2) - U_cube_a + Da * laplacian_a
    dUb = w_term_c / (1 + w_term_d ** 2)- U_cube_b + Db * laplacian_b

    # Update U
    U_next_a = U_curr[:, :, :, 0] + dt * dUa
    U_next_b = U_curr[:, :, :, 1] + dt * dUb

    U_next = jnp.stack([U_next_a, U_next_b], axis=-1)

    return (U_next, params), U_next

# Optimization setup (unchanged)
learning_rate = 1e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Modified update step
@jit
def update_step(params, opt_state, Us, dt, target_a, target_b):
    loss, grads = jax.value_and_grad(loss_function)(params, Us, dt, target_a, target_b)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
def loss_function(params, Us, dt, target_a, target_b):
    final_Us = run_simulation(Us, params, dt)

    # Calculate the difference between final state and target shapes for Us1
    diff_a = final_Us[-1, :, :, 0, 0] - target_a[:, :, 0, 0]  # Us1, a part
    diff_b = final_Us[-1, :, :, 0, 1] - target_b[:, :, 0, 1]  # Us1, b part

    # Sum squares of points outside the ellipse
    shape_loss_a = jnp.sum((diff_a ) ** 2)
    shape_loss_b = jnp.sum((diff_b ) ** 2)

    shape_loss = shape_loss_a + shape_loss_b

    # Calculate the buffer loss for the last 50 steps
    buffer_loss = jnp.sum((final_Us[-50:] - final_Us[-51:-1]) ** 2)

    # You can adjust the weights of the shape loss and buffer loss
    shape_weight = 1
    buffer_weight = 0.5

    return shape_weight * shape_loss + buffer_weight * buffer_loss
# Optimization loop
dt = 0.01
losses = []

# Create a tqdm progress bar
progress_bar = tqdm.tqdm(total=num_iterations, desc="Optimization Progress", ncols=100)

print("Starting optimization:")
for iteration in range(num_iterations):
    params, opt_state, loss = update_step(params, opt_state, Us, dt, target_a, target_b)
    losses.append(loss)

    # Update the progress bar
    progress_bar.update(1)

    if iteration % 100 == 0:
        # Update the progress bar description with the current loss
        progress_bar.set_description(f"Iteration {iteration}, Loss: {loss:.6f}")
        progress_bar.refresh()  # Force refresh the display

    if loss < 1e-6:
        print(f"\nConverged at iteration {iteration}")
        break

# Close the progress bar
progress_bar.close()

# Final parameters and simulation
final_params = constrain_params(params)
final_Us = run_simulation(Us, params, dt)

print("\nFinal Parameters:")
for key, value in final_params.items():
    print(f"{key}:\n{value}")

print("\nFinal State:")
print(final_Us[-1])

# Add boundary points to the final state
final_Us_with_boundary = add_boundary_points(final_Us[-1])

# Calculate final shape difference
diff_a = final_Us_with_boundary[1:-1, 1:-1, 0, 0] - target_a[:, :, 0, 0]
diff_b = final_Us_with_boundary[1:-1, 1:-1, 0, 1] - target_b[:, :, 0, 1]
shape_diff = jnp.sum((diff_a * (1 - target_a[:, :, 0, 0])) ** 2) + jnp.sum((diff_b * (1 - target_b[:, :, 0, 1])) ** 2)

# Calculate shape error percentage
target_norm = jnp.sum(target_a[:, :, 0, 0] ** 2) + jnp.sum(target_b[:, :, 0, 1] ** 2)
shape_error_percentage = (shape_diff / target_norm) * 100

buffer_loss = jnp.sum(jnp.sqrt((final_Us[-1] - final_Us[-2]) ** 2))

# Calculate buffer error percentage
previous_step_norm = jnp.sum(final_Us[-2])
buffer_error_percentage = (buffer_loss / previous_step_norm) * 100

print("\nError Percentages:")
print(f"Shape Error: {shape_error_percentage:.4f}%")
print(f"Buffer Error: {buffer_error_percentage:.4f}%")

# Create a 'data' folder with parameters in the name
data_folder = f'data_I{I}_N{N}_iter{num_iterations}_M{M}'
os.makedirs(data_folder, exist_ok=True)

# Save all optimized final data for all Ui, both a and b
for i in range(I):
    np.save(f'{data_folder}/U{i+1}_a_final.npy', final_Us[-1, :, :, i, 0])
    np.save(f'{data_folder}/U{i+1}_b_final.npy', final_Us[-1, :, :, i, 1])

# Save optimized parameters
np.savez(f'{data_folder}/optimized_parameters.npz',
         w_a=final_params['w_a'],
         w_b=final_params['w_b'],
         w_c=final_params['w_c'],
         w_d=final_params['w_d'],
         Da=final_params['Da'],
         Db=final_params['Db'])

# Save loss history
np.save(f'{data_folder}/loss_history.npy', np.array(losses))

print(f"Optimized data, parameters, and loss history have been saved in the '{data_folder}' folder.")

# Plotting
fig = plt.figure(figsize=(16, 12))

# Row 1: Dynamics, Combined Initial U1a and U1b
# Row 2: Final U1a, Final U1b
# Row 3: Target U1a, Target U1b

# Plot dynamics of U1a and U1b at the center point
center = N // 2
ax1 = fig.add_subplot(231)
ax1.plot(range(M), final_Us[:, center, center, 0, 0], label='U1a')
ax1.plot(range(M), final_Us[:, center, center, 0, 1], label='U1b')
ax1.set_title(f'Dynamics of U1 at Center Point\nBuffer Error: {buffer_error_percentage:.2f}%')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Value')
ax1.legend()

# Plot combined initial shape of U1a and U1b with boundary
initial_Us_with_boundary = add_boundary_points(Us[0])
ax2 = fig.add_subplot(232)
im2 = ax2.imshow(initial_Us_with_boundary[:, :, 0, 0] + initial_Us_with_boundary[:, :, 0, 1], cmap='viridis')
fig.colorbar(im2, ax=ax2, label='U1a + U1b')
ax2.set_title('Initial Shape of U1a and U1b')

# Plot final shape of U1a with boundary
ax3 = fig.add_subplot(233)
im3 = ax3.imshow(final_Us_with_boundary[:, :, 0, 0], cmap='viridis')
fig.colorbar(im3, ax=ax3, label='U1a')
ax3.set_title(f'Final Shape of U1a\nShape Error: {shape_error_percentage:.2f}%')

# Plot final shape of U1b with boundary
ax4 = fig.add_subplot(234)
im4 = ax4.imshow(final_Us_with_boundary[:, :, 0, 1], cmap='viridis')
fig.colorbar(im4, ax=ax4, label='U1b')
ax4.set_title(f'Final Shape of U1b\nShape Error: {shape_error_percentage:.2f}%')

# Plot target shape for U1a with boundary
ax5 = fig.add_subplot(235)
im5 = ax5.imshow(target_a_with_boundary[:, :, 0, 0], cmap='viridis')
fig.colorbar(im5, ax=ax5, label='Target U1a')
ax5.set_title('Target Shape for U1a')

# Plot target shape for U1b with boundary
ax6 = fig.add_subplot(236)
im6 = ax6.imshow(target_b_with_boundary[:, :, 0, 1], cmap='viridis')
fig.colorbar(im6, ax=ax6, label='Target U1b')
ax6.set_title('Target Shape for U1b')

plt.tight_layout()

# Save the main figure
plt.savefig('reaction_diffusion_2d_main_results.png', dpi=300, bbox_inches='tight')

# Create a separate figure for the loss function
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')

# Save the loss function figure
plt.savefig('reaction_diffusion_2d_loss.png', dpi=300, bbox_inches='tight')

# Show all plots (optional, you can comment this out if you don't need to display them)
plt.show()

print("Main results plot has been saved as 'reaction_diffusion_2d_main_results.png'")
print("Loss function plot has been saved as 'reaction_diffusion_2d_loss.png'")