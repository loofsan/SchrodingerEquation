# Exploring Solutions to the Schrödinger Equation in a Finite Potential Box

## Introduction

The Schrödinger equation is one of the most significant equations of quantum mechanics, providing a mathematical framework to describe the quantum state of physical systems. It allows physicists to predict the behavior of particles at atomic and subatomic scales, revolutionizing our understanding of the universe.

This project will explore solutions to the three-dimensional time-independent Schrödinger equation for a particle confined in a finite potential box. The goal is to visualize quantized energy levels using heatmaps and 3-dimensional models. By focusing on the hydrogen atom, we aim to enhance our understanding of the interplay between quantum mechanics and physical phenomena.

## Finite vs Infinite Potential Box

Unlike the infinite potential box, where the potential walls are infinitely high, a finite potential box allows for a more realistic model. In this case, the particle can penetrate and even tunnel through the potential barriers, though with diminishing probability. This difference leads to significant changes in the energy levels and wavefunctions compared to the infinite potential box.

## Methodology

### Separation of Variables

We use the method of separation of variables to solve the Schrödinger equation. The basic assumption is that the wavefunction can be expressed as a product of three independent functions, each dependent on one spatial coordinate: \(x\), \(y\), or \(z\). This results in three coupled differential equations, which are solved under boundary conditions that the wavefunction approaches zero outside the potential walls, but does not necessarily vanish sharply at the walls.

## Energy Quantization

The quantized energy levels for a finite potential box are influenced by the depth and width of the potential well. This leads to non-uniform spacing between energy states, which becomes more prominent as we move to higher energies. The ground state energy is higher compared to the infinite potential box due to the finite height of the potential walls.

## Visualizing Wavefunction Probability Densities

### Ground State

To visualize the wavefunction probability densities, I look at each selected energy state. For example, the ground state, where \((n_x=1, n_y=0, n_z=0)\), shows the highest probability density near the center of the box, gradually decreasing towards the edges.

### Higher Energy States

Higher energy states, such as \((n_x=2, n_y=1, n_z=1)\), exhibit more complex structures with nodes, indicating regions where the probability density is zero. This visualization is intuitive; if you were to guess the location of the particle inside the box, it would most likely be near the center, reflecting the higher probability density in that region.

## Graphical Representation

To provide a graphical representation, I generated heatmaps for the probability densities of these states, illustrating where the particle is most likely to be found within the box. These visualizations, combined with the numerical solutions, offer insights into the behavior of particles in finite potential wells.

## Conclusion

This study highlights the quantized nature of energy in confined systems and the significance of potential barrier characteristics in determining energy states. It demonstrates the fundamental principles of quantum mechanics through the analysis and visualization of solutions to the Schrödinger equation.
