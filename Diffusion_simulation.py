import numpy as np
import igl
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # Import for colormaps

def generate_sphere_mesh(radius=1.0, refinement_levels=3):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ]) / np.sqrt(1 + t ** 2)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
        [1, 5, 9], [5,11, 4], [11,10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4,11], [6, 2,10], [8, 6, 7], [9, 8, 1]
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

def compute_laplacian(v, f):
    # Compute cotangent Laplacian using libigl
    L = igl.cotmatrix(v, f)
    return L

def diffusion_step(u, L, dt, D):
    # Explicit time stepping without source term
    du = D * L.dot(u)
    return u + dt * du


def initialize_point_source_indices(v, cluster_size=5, epsilon=0.25):
    # Find the north pole
    north_pole = np.array([0, 0, 1])

    # Create two base offset points
    offset_left = np.array([-epsilon, 0, 1])
    offset_right = np.array([epsilon, 0, 1])

    # Normalize the offset points
    offset_left /= np.linalg.norm(offset_left)
    offset_right /= np.linalg.norm(offset_right)

    # Function to create a cluster around a point
    def create_cluster(center, size):
        cluster_indices = []
        distances = np.linalg.norm(v - center, axis=1)
        sorted_indices = np.argsort(distances)
        return sorted_indices[:size]

    # Create clusters
    left_cluster = create_cluster(offset_left, cluster_size)
    right_cluster = create_cluster(offset_right, cluster_size)

    return left_cluster, right_cluster

def simulate_diffusion(v, f, num_steps=100, dt=0.01, D=1.0, num_fields=2):
    num_vertices = v.shape[0]
    # Initialize diffusion fields with zeros
    u = np.zeros((num_vertices, num_fields))

    # Initialize point sources at t=0
    source_indices = initialize_point_source_indices(v)
    u[source_indices[0], 0] = 3.0  # Set U1 = 3 at first source (left of north pole)
    u[source_indices[1], 1] = 3.0  # Set U2 = 3 at second source (right of north pole)

    # Compute Laplacian
    L = compute_laplacian(v, f)

    # Store u values at different time steps
    u_history = [u.copy()]

    # Simulation loop
    for step in range(num_steps):
        u = diffusion_step(u, L, dt, D)
        u_history.append(u.copy())

    return np.array(u_history)

def plot_diffusion_steps(v, f, u_history):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    num_fields = u_history.shape[2]

    # Determine the time steps to plot: initial, middle, final
    steps = [0, len(u_history)//2, len(u_history)-1]
    step_names = ['Initial', 'Mid', 'Final']

    # Set the desired colorbar range
    colorbar_min = 0
    colorbar_max = 1  # Set colorbar max to 1

    # Create a colormap and normalization based on the desired range
    norm = plt.Normalize(vmin=colorbar_min, vmax=colorbar_max)
    cmap = plt.get_cmap('viridis')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(18, 12))

    plot_number = 1

    for field_index in range(num_fields):
        for i, step in enumerate(steps):
            ax = fig.add_subplot(2, 3, plot_number, projection='3d')
            ax.set_title(f"U{field_index+1}, {step_names[i]} Step (t = {step})")
            # Display the axes
            #ax.set_axis_off()
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1,1)

            # Remove tick labels but keep the axes
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Get the field at this step
            u = u_history[step][:, field_index]

            # Compute face colors by averaging values at vertices
            face_values = np.mean(u[f], axis=1)

            # Map face values to colors
            face_colors = mappable.to_rgba(face_values)

            # Plot trisurf
            tri = ax.plot_trisurf(
                v[:, 0], v[:, 1], v[:, 2],
                triangles=f,
                shade=False,
                linewidth=0.2,
                antialiased=True
            )

            tri.set_facecolors(face_colors)

            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Diffusion Value')

            plot_number += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    refinement_levels = 3
    v, f = generate_sphere_mesh(radius=1.0, refinement_levels=refinement_levels)
    u_history = simulate_diffusion(v, f, num_steps=200, dt=0.01, D=1, num_fields=2)

    print("Simulation complete. Plotting diffusion steps...")
    plot_diffusion_steps(v, f, u_history)
    print("Plots displayed.")