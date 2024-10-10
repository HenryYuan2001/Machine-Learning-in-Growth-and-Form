# Machine-Learning-in-Growth-and-Form

This project explores the application of machine learning techniques to reaction-diffusion equations, with a focus on simulating growth and form in biological systems. Currently, it includes a 2D reaction-diffusion simulation and a 3D diffusion test using libigl.

## Project Overview

### 1. 2D Reaction-Diffusion Simulation

We have implemented a 2D reaction-diffusion simulation using Python and JAX. The simulation is based on the following equations:

$$
\begin{align*}
\partial_t U_a &= \frac{\sum w_{ab}U_b}{(1 + \sum w_{a\beta}\mu_\beta)^2} - U_a^3 + D_a \partial^2 U_a \\
\partial_t \mu_\alpha &= \frac{\sum w_{\alpha b}U_b}{(1 + \sum w_{\alpha\beta}\mu_\beta)^2} - \mu_\alpha^3 + D_b \partial^2 \mu_\alpha
\end{align*}
$$

Key features of the 2D simulation:
- Initialization of a 2D grid with specific initial conditions
- Implementation of the above reaction-diffusion equations
- Parameter optimization using machine learning techniques such as the Adam optimizer from JAX
- Visualization of results, including initial state, final state, and target shapes

One example output of the 2D optimization is the image below, showing the dynamics of U1 at the center point, initial and final shapes, and target shapes:

### 2. 3D Diffusion Simulation Test

As a preliminary step towards a full 3D reaction-diffusion system, we have implemented a basic 3D diffusion simulation using libigl. This demonstrates our ability to work with 3D geometries and sets the stage for more complex 3D simulations in the future.

## Results

After running the 2D simulation, you should see output similar to the image below, showing the dynamics of U1 at the center point, initial and final shapes, and target shapes:

[Insert results image here]

## Future Work

- Extend the 2D reaction-diffusion system to 3D using libigl
- Implement more complex reaction-diffusion equations in 3D
- Develop advanced machine learning techniques for parameter optimization in 3D
- Create a user-friendly interface for setting up and running simulations

## Contributing

We welcome contributions from the community! If you're interested in contributing, please open an issue or submit a pull request.

## License

[Choose an appropriate license and add it here]

## Contact

Henry Yuan - hengyuan@ucsb.edu

Project Link: [https://github.com/YourUsername/Machine-Learning-in-Growth-and-Form](https://github.com/YourUsername/Machine-Learning-in-Growth-and-Form)
