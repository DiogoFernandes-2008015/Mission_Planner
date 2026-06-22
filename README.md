# Mission Planner: 3-DoF Exterior Ballistics & PSO Trajectory Optimization

An advanced, high-performance mission planner and trajectory optimization suite for ballistic projectiles. This repository combines accurate 3-Degrees-of-Freedom (3-DoF) exterior ballistics simulation, including International Standard Atmosphere models, Mach-dependent drag coefficient lookup, and wind profiles, with a custom Particle Swarm Optimization (PSO) algorithm to solve inverse ballistic targeting problems.

Implementations are provided in both highly vectorized/differentiable **Python (JAX & Diffrax)** and low-latency **C++ (native Win32 GUI and CLI versions)**.

---

## 📌 Features

- **High-Fidelity Physical Modeling:**
  - **3-DoF Ballistic Solver:** Simulates projectile flight accounting for gravity, aerodynamic drag, and variable wind velocity vectors.
  - **ISA Atmosphere Model:** Dynamic air density ($\rho$) computations based on the International Standard Atmosphere equations for the troposphere.
  - **Mach-Dependent Drag ($C_d$):** Look-up and interpolation of drag coefficients based on real-time Mach number transitions.
  - **Wind Profiles:** Simulates power-law wind shear distributions acting on the projectile during flight.
- **Advanced Optimization Engine:**
  - Custom **Particle Swarm Optimization (PSO)** tuned to minimize target impact errors.
  - Solves for the optimal **4D launch parameters**: Initial position $(x_0, y_0)$, initial velocity $(v_0)$, and azimuth angle $(\theta)$ for a given target coordinate $(x_d, y_d)$ and initial height $z_0$.
- **Multi-Language Architecture:**
  - **Python:** Leverages **JAX** for accelerated linear algebra and **Diffrax** for highly efficient, adaptive Ordinary Differential Equation (ODE) solving.
  - **C++:** Native, zero-dependency implementations utilizing Runge-Kutta 4th Order (RK4) integration for real-time calculation speeds. Includes a fully native Windows Graphical User Interface (Win32 API).

---

## 📂 Repository Structure

```text
├── Python/
│   ├── PSO.py                          # Generic Vectorized PSO algorithm (
