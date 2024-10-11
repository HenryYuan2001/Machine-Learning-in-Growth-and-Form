import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, lax, jit
import matplotlib.pyplot as plt
import optax
import tqdm
import igl

# Check device
device = jax.devices()[0]
device_kind = device.device_kind
print(f"JAX is using: {device_kind}")

# Settings
refinement_levels = 3
I = 25  # Number of U variables for each vertex (U_1, U_2, ..., U_I)
num_iterations = 1070
M = 200  # Number of time steps

# Generate spherical mesh using the original method
def generate_sphere_mesh(radius=1.0, refinement_levels=3):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]) / np.sqrt(1 + t ** 2)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])

    def refine_mesh(v, f):
        edges = {}
        new_faces = []
        for face in f:
            new_vertices = []
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edges:
                    new_vertex = (v[edge[0]] + v[edge[1]]) / 2
                    new_vertex /= np.linalg.norm(new_vertex)
                    edges[edge] = len(v)
                    v = np.vstack((v, new_vertex))
                new_vertices.append(edges[edge])
            new_faces.extend([
                [face[0], new_vertices[0], new_vertices[2]],
                [new_vertices[0], face[1], new_vertices[1]],
                [new_vertices[2], new_vertices[1], face[2]],
                [new_vertices[0], new_vertices[1], new_vertices[2]]
            ])
        return v, np.array(new_faces)

    for _ in range(refinement_levels):
        vertices, faces = refine_mesh(vertices, faces)
    vertices *= radius
    return vertices, faces

v, f = generate_sphere_mesh(radius=1.0, refinement_levels=refinement_levels)
num_vertices = len(v)

# Function to compute adjacency list
def compute_adjacency_list(f):
    adj_list = [set() for _ in range(len(v))]
    for face in f:
        for i in range(3):
            adj_list[face[i]].update([face[(i+1)%3], face[(i+2)%3]])
    return [list(neighbors) for neighbors in adj_list]

# Compute adjacency list
adj_list = compute_adjacency_list(f)

# Create padded adjacency list
max_neighbors = max(len(neighbors) for neighbors in adj_list)
padded_adj_list = jnp.array([neighbors + [-1] * (max_neighbors - len(neighbors)) for neighbors in adj_list])

# Initialize Us with zeros: (num_vertices, I, 2)
Us = jnp.zeros((num_vertices, I, 2))

# Find the index of the vertex closest to the north pole (maximum z-coordinate)
north_pole_index = jnp.argmax(v[:, 2])

# Get the neighbors of the north pole vertex
north_pole_neighbors = adj_list[north_pole_index]

# Positions of the neighbors
neighbor_positions = v[north_pole_neighbors]

# Compute azimuthal angles of the neighbors
phi_neighbors = jnp.arctan2(neighbor_positions[:, 1], neighbor_positions[:, 0])  # Azimuthal angle

# Find the left and right neighbors based on phi
left_neighbor_index = north_pole_neighbors[jnp.argmax(phi_neighbors)]
right_neighbor_index = north_pole_neighbors[jnp.argmin(phi_neighbors)]

# Set initial conditions
Us = Us.at[left_neighbor_index, 0, 0].set(3)  # U1a at left neighbor
Us = Us.at[right_neighbor_index, 0, 1].set(3)  # U1b at right neighbor

# Initialize the parameters
def init_params():
    key = random.PRNGKey(0)
    return {
        'w_a': random.normal(key, (I, I)),
        'w_b': random.normal(key, (I, I)),
        'w_c': random.normal(key, (I, I)),
        'w_d': random.normal(key, (I, I)),
        'Da': random.normal(key, (I,)),
        'Db': random.normal(key, (I,)),
    }

params = init_params()
# Compute cotangent Laplacian (this should be done once, outside the function)
L = igl.cotmatrix(v, f)
L_csr = L.tocsr()

# Convert sparse matrix to dense for JAX compatibility
L_dense = L_csr.todense()
L_jax = jnp.array(L_dense)


@jit
def diffusion_step(carry, t):
    U_curr, params, dt = carry
    w_a, w_b, w_c, w_d, Da, Db = params['w_a'], params['w_b'], params['w_c'], params['w_d'], params['Da'], params['Db']

    # Compute w terms
    w_term_a = jnp.einsum('ij,vj->vi', w_a, U_curr[:, :, 0])
    w_term_b = jnp.einsum('ij,vj->vi', w_b, U_curr[:, :, 1])
    w_term_c = jnp.einsum('ij,vj->vi', w_c, U_curr[:, :, 0])
    w_term_d = jnp.einsum('ij,vj->vi', w_d, U_curr[:, :, 1])

    # Compute cubic terms
    U_cube_a = U_curr[:, :, 0] ** 3
    U_cube_b = U_curr[:, :, 1] ** 3

    # Apply Laplacian directly
    laplacian_a = L_jax @ U_curr[:, :, 0]
    laplacian_b = L_jax @ U_curr[:, :, 1]

    # Compute dU
    dUa = w_term_a / (1 + w_term_b ** 2) - U_cube_a + jnp.einsum('i,vi->vi', Da, laplacian_a)
    dUb = w_term_c / (1 + w_term_d ** 2) - U_cube_b + jnp.einsum('i,vi->vi', Db, laplacian_b)

    # Update U
    U_next_a = U_curr[:, :, 0] + dt * dUa
    U_next_b = U_curr[:, :, 1] + dt * dUb

    U_next = jnp.stack([U_next_a, U_next_b], axis=-1)

    return (U_next, params, dt), U_next

@jit
def run_simulation(U_init, params, dt):
    _, Us = lax.scan(diffusion_step, (U_init, params, dt), jnp.arange(M))
    return Us[-1]  # Return only the final state


def create_target_shapes(v):
    import numpy as np
    from scipy.special import sph_harm

    def spherical_harmonic(l, m, theta, phi):
        """
        Compute the real spherical harmonic Y_l^m(theta, phi).

        :param l: Degree of the harmonic
        :param m: Order of the harmonic
        :param theta: Polar angle (colatitude) in radians
        :param phi: Azimuthal angle (longitude) in radians
        """
        return np.real(sph_harm(m, l, phi, theta))

    # Convert Cartesian coordinates to spherical coordinates
    r = np.sqrt(np.sum(v ** 2, axis=1))
    theta = np.arccos(v[:, 2] / r)  # Polar angle
    phi = np.arctan2(v[:, 1], v[:, 0])  # Azimuthal angle

    # Create a mask for the upper hemisphere (north pole)
    upper_hemisphere_mask = v[:, 2] > 0

    # Create target shapes using different spherical harmonics
    target_a = np.zeros((len(v), I))
    target_b = np.zeros((len(v), I))

    # For U1a: Use Y_3^2
    target_a[upper_hemisphere_mask, 0] = spherical_harmonic(3, 2, theta[upper_hemisphere_mask], phi[upper_hemisphere_mask])

    # For U1b: Use Y_4^3
    target_b[upper_hemisphere_mask, 0] = spherical_harmonic(4, 3, theta[upper_hemisphere_mask], phi[upper_hemisphere_mask])

    # Normalize the values to be between 0 and 1 (only for the upper hemisphere)
    target_a[upper_hemisphere_mask, 0] = (target_a[upper_hemisphere_mask, 0] - np.min(target_a[upper_hemisphere_mask, 0])) / (np.max(target_a[upper_hemisphere_mask, 0]) - np.min(target_a[upper_hemisphere_mask, 0]))
    target_b[upper_hemisphere_mask, 0] = (target_b[upper_hemisphere_mask, 0] - np.min(target_b[upper_hemisphere_mask, 0])) / (np.max(target_b[upper_hemisphere_mask, 0]) - np.min(target_b[upper_hemisphere_mask, 0]))

    return jnp.array(target_a), jnp.array(target_b)

target_a, target_b = create_target_shapes(v)


# Loss function
@jit
def loss_function(params, U_init, dt, target_a, target_b):
    final_U = run_simulation(U_init, params, dt)

    # Calculate the difference between final state and target shapes for U1
    diff_a = final_U[:, 0, 0] - target_a[:, 0]  # U1a
    diff_b = final_U[:, 0, 1] - target_b[:, 0]  # U1b

    # Sum squares of differences
    shape_loss_a = jnp.sum(diff_a ** 2)
    shape_loss_b = jnp.sum(diff_b ** 2)

    shape_loss = shape_loss_a + shape_loss_b

    # Calculate the buffer loss (difference between final and initial states)
    buffer_loss = jnp.sum((final_U[-1:] - final_U[-2:-1]) ** 2)

    # Weights for losses
    shape_weight = 1.0
    buffer_weight = 0.5

    total_loss = shape_weight * shape_loss + buffer_weight * buffer_loss
    return total_loss

# Optimization setup
learning_rate = 1e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Update step
@jit
def update_step(params, opt_state, U_init, dt, target_a, target_b):
    loss, grads = jax.value_and_grad(loss_function)(params, U_init, dt, target_a, target_b)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

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

    if iteration % 10 == 0:
        # Update the progress bar description with the current loss
        progress_bar.set_description(f"Iteration {iteration}, Loss: {loss:.6f}")
        progress_bar.refresh()  # Force refresh the display

# Close the progress bar
progress_bar.close()

print("Final loss:", losses[-1])

# Run the final simulation
final_U = run_simulation(Us, params, dt)

# After running the simulation and before plotting
print("\nFinal Shapes:")
print("U1a final shape:", final_U[:, 0, 0].shape)
print("U1b final shape:", final_U[:, 0, 1].shape)
print("\nTarget Shapes:")
print("U1a target shape:", target_a[:, 0].shape)
print("U1b target shape:", target_b[:, 0].shape)

# Calculate final shape difference
diff_a = final_U[:, 0, 0] - target_a[:, 0]
diff_b = final_U[:, 0, 1] - target_b[:, 0]
shape_diff = jnp.sum(diff_a ** 2) + jnp.sum(diff_b ** 2)

# Calculate shape error percentage
target_norm = jnp.sum(target_a[:, 0] ** 2) + jnp.sum(target_b[:, 0] ** 2)
shape_error_percentage = (shape_diff / target_norm) * 100

# Calculate buffer loss
buffer_loss = jnp.sum((final_U - Us) ** 2)

# Calculate buffer error percentage
initial_norm = jnp.sum(Us ** 2)
buffer_error_percentage = (buffer_loss / initial_norm) * 100 if initial_norm != 0 else 0

print("\nError Percentages:")
print(f"Shape Error: {shape_error_percentage:.4f}%")
print(f"Buffer Error: {buffer_error_percentage:.4f}%")

def compute_face_values(vertices, faces, vertex_values):
    face_values = np.mean(vertex_values[faces], axis=1)
    return face_values

def plot_sphere(ax, vertices, faces, face_values, title):
    # Plot the faces with a heatmap
    triang = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             triangles=faces, cmap='viridis',
                             linewidth=0.2, edgecolors='k', alpha=0.8)
    triang.set_array(face_values)

    # Set the viewing angle so that the north pole faces the viewer
    ax.view_init(elev=60, azim=0)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 1))
    return triang

# Plot initial, target, and final states
def create_plots(Us, target_a, target_b, final_U):
    # U1a plots
    fig_a = plt.figure(figsize=(20, 6))

    # Initial state U1a
    ax1 = fig_a.add_subplot(131, projection='3d')
    face_values_a_initial = compute_face_values(v, f, np.array(Us[:, 0, 0]))
    triang_a_initial = plot_sphere(ax1, v, f, face_values_a_initial, 'Initial U1a')

    # Target state U1a
    ax2 = fig_a.add_subplot(132, projection='3d')
    face_values_a_target = compute_face_values(v, f, np.array(target_a[:, 0]))
    triang_a_target = plot_sphere(ax2, v, f, face_values_a_target, 'Target U1a (Filled Ellipse)')

    # Final state U1a
    ax3 = fig_a.add_subplot(133, projection='3d')
    face_values_a_final = compute_face_values(v, f, np.array(final_U[:, 0, 0]))
    triang_a_final = plot_sphere(ax3, v, f, face_values_a_final, 'Final U1a')

    # Add colorbars
    fig_a.colorbar(triang_a_initial, ax=ax1, label='Face Values')
    fig_a.colorbar(triang_a_target, ax=ax2, label='Face Values')
    fig_a.colorbar(triang_a_final, ax=ax3, label='Face Values')

    plt.tight_layout()
    plt.show()

    # U1b plots
    fig_b = plt.figure(figsize=(20, 6))

    # Initial state U1b
    ax4 = fig_b.add_subplot(131, projection='3d')
    face_values_b_initial = compute_face_values(v, f, np.array(Us[:, 0, 1]))
    triang_b_initial = plot_sphere(ax4, v, f, face_values_b_initial, 'Initial U1b')

    # Target state U1b
    ax5 = fig_b.add_subplot(132, projection='3d')
    face_values_b_target = compute_face_values(v, f, np.array(target_b[:, 0]))
    triang_b_target = plot_sphere(ax5, v, f, face_values_b_target, 'Target U1b (Filled Ellipse)')

    # Final state U1b
    ax6 = fig_b.add_subplot(133, projection='3d')
    face_values_b_final = compute_face_values(v, f, np.array(final_U[:, 0, 1]))
    triang_b_final = plot_sphere(ax6, v, f, face_values_b_final, 'Final U1b')

    # Add colorbars
    fig_b.colorbar(triang_b_initial, ax=ax4, label='Face Values')
    fig_b.colorbar(triang_b_target, ax=ax5, label='Face Values')
    fig_b.colorbar(triang_b_final, ax=ax6, label='Face Values')

    plt.tight_layout()
    plt.show()

# Run the simulation and create plots
final_U = run_simulation(Us, params, dt)
create_plots(Us, target_a, target_b, final_U)

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Loss over iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.show()
