# Machine-Learning-in-Growth-and-Form

This project explores the application of machine learning techniques to reaction-diffusion equations, with a focus on simulating growth and form in biological systems. Currently, it includes a 2D reaction-diffusion simulation and a 3D diffusion test using libigl.

## Project Overview

### 1. 2D Reaction-Diffusion Optimization

We have implemented a 2D reaction-diffusion simulation using Python and JAX. The simulation is based on the following equations:

$$
\Huge
\begin{align*}
\partial_t U_a &= \frac{\sum w_{ab}U_b}{(1 + \sum w_{a\beta}\mu_\beta)^2} - U_a^3 + D_a \partial^2 U_a \\
\partial_t \mu_\alpha &= \frac{\sum w_{\alpha b}U_b}{(1 + \sum w_{\alpha\beta}\mu_\beta)^2} - \mu_\alpha^3 + D_b \partial^2 \mu_\alpha
\end{align*}
$$

Key features of the 2D simulation:
- Initialization of a 2D grid with specific initial conditions 
- Implementation of the reaction-diffusion equations using Python and JAX and write the diffusion simulation based on discrete laplacian operator in uniform 2D grid
- Parameter optimization of w matrix and diffusion coefficients using Adam optimizer from Jax
- Custom loss function for shape targeting
- Visualization of results, including initial state, final state, and target shapes

The image below presents an output from 2D optimization. It depicts the behavior of U1 and U2, initially concentrated in two nearby point sources. As the simulation progresses, we observe the diffusion of these components across the grid. Our optimization algorithm attempts to adjust the w matrix and diffusion coefficients, aiming to approximate target shapes resembling two ellipses polarized in different directions. The visualization includes the time evolution at the center point, initial and final states, and the intended target shapes.
<p align="center">
  <img src="https://github.com/user-attachments/assets/ee344a5e-8e08-464e-9579-1ea82b86a632" alt="2D Optimization Results" width="800"/>
</p>


### 2. 3D Diffusion Simulation Test

As a preliminary step towards a full 3D reaction-diffusion system, we have implemented a basic 3D diffusion simulation using libigl. This demonstrates our ability to work with 3D geometries and sets the stage for more complex 3D simulations in the future. While in 2D we can safely compute discrete Laplacian, in 3D we need to use comatrix since the mesh might not necessarily be uniform, which we use from comatrix from libigl. 

The code in diffusion_simulation.py creates a simple u vector containing two fields, u1 and u2, initializes them with two cluster sources near the North Pole, and simulates their diffusion using the cotangent Laplacian operator computed by igl.cotmatrix.
<p align="center">
  <img width="800" alt="Screenshot 2024-10-10 at 8 31 02 PM" src="https://github.com/user-attachments/assets/506d8aa1-792c-4048-a74d-b8b27a0967f1">
</p>

### 2. 3D Reaction-Diffusion Optimization

Combining the 2D optimization into the 3D setting and using libigl for a more generalized Laplacian operator, the comatrix function is compatible with just-in-time compilation in JAX with a few twists. One of the most non-trivial cases is still trying to get two differently polarized ellipses projected into 3D. Here are the results.
<p align="center">
  <img width="800" alt="Screenshot 2024-10-10 at 8 42 46 PM" src="https://github.com/user-attachments/assets/7b3e3f53-b27c-43c6-96f0-3c311504965c">
  <img width="800" alt="Screenshot 2024-10-10 at 8 43 04 PM" src="https://github.com/user-attachments/assets/30b13767-4fde-477e-a0e4-cd92e85e0b97">
</p>

## Contact

Henry Yuan - hengyuan@ucsb.edu

