# Mission Planner: 3-DoF Exterior Ballistics & PSO Trajectory Optimization

An advanced, high-performance mission planner and trajectory optimization suite for ballistic projectiles. This repository combines accurate **3-Degrees-of-Freedom (3-DoF) exterior ballistics simulation** (including International Standard Atmosphere models, Mach-dependent drag coefficient lookup, and wind profiles) with a custom **Particle Swarm Optimization (PSO)** algorithm to solve inverse ballistic targeting problems.

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
│   ├── PSO.py                          # Generic Vectorized PSO algorithm (tested on Rosenbrock)
│   ├── ext_bal_3dof.py                 # Pure 3-DoF simulation with adaptive time steps (Diffrax)
│   ├── ext_bal_3dof_fun.py             # Functional version of the ballistic solver for optimization
│   ├── calc_imp.py                     # PSO + Ballistic solver integration without wind effects
│   └── calc_imp_wind.py                # Complete PSO + Ballistic solver integrating atmospheric wind
│
├── C++/
│   ├── calc_imp_wind.cpp               # High-performance C++ CLI implementation of PSO + Ballistics
│   ├── planejador_missao_inter_prompt.cpp # Interactive Command-Line Interface for user target input
│   └── planejador_missao_gui.cpp       # Native Windows (Win32) GUI desktop application for mission planning
└── README.md

Mathematical & Physical Models
1. Equations of Motion (3-DoF)

The state vector is defined as y=[x,y,z,vx​,vy​,vz​]T. The system of ODEs solved during flight is:
dtd​​xyz​​=​vx​vy​vz​​​
dtd​​vx​vy​vz​​​=​00−g​​−2⋅mρ⋅S⋅Cd​(M)⋅∥vrel​∥​vrel​

Where vrel​ is the velocity relative to the wind vector: vrel​=v−vwind​.
2. Optimization Objective

The PSO minimizes the squared Euclidean distance between the projectile's terminal impact coordinate on the ground (z=cota) and the designated mission target location (xd​,yd​):
Minimize f(x0​,y0​,v0​,θ)=(ximpact​−xd​)2+(yimpact​−yd​)2
🚀 Getting Started
Python Environment Setup

    Install Dependencies:
    Ensure you have a modern Python environment (>= 3.9) and install jax, diffrax, numpy, and matplotlib:
    Bash

    pip install jax jaxlib diffrax numpy matplotlib

    Run Trajectory Optimization:
    To find the ideal launch parameters for a target with wind disturbances, run:
    Bash

    python Python/calc_imp_wind.py

C++ Compilation & Execution
Command-Line Version (Cross-platform)

Compile using any standard C++11 compliant compiler (e.g., g++ or clang):
Bash

g++ -O3 C++/calc_imp_wind.cpp -o ballistic_pso
./ballistic_pso

Windows GUI Application

The GUI is built using pure Win32 API and requires no external framework dependencies (like Qt or wxWidgets).

    Using MSVC (Developer Command Prompt):
    Bash

    cl.exe /O2 /EHsc C++/planejador_missao_gui.cpp user32.lib gdi32.lib

    Using MinGW:
    Bash

    g++ -O3 C++/planejador_missao_gui.cpp -o MissionPlannerGUI -mwindows

🖥️ Graphical User Interface (GUI) Overview

The Windows desktop version (planejador_missao_gui.cpp) provides an interactive interface to control mission setup parameters in real-time:

    Target Coordinates Input: Set targeted X and Y locations.

    Initial Height (Z0​): Adjust the altitude of the launcher.

    Wind Parameters: Input wind speed vectors (Vwx​,Vwy​) and reference altitude (Zref​) to build the atmospheric wind shear profile.

    Optimization Output: Instantly prints out the optimal calculated initial positions, required muzzle velocity, and azimuth launch angle.

🤝 Contributing

Contributions to enhance physical fidelity (e.g., adding 6-DoF rigid body dynamics, Coriollis effect, or alternative optimization solvers like GA or CMA-ES) are welcome.

    Fork the Project

    Create your Feature Branch (git checkout -b feature/BallisticsEnhancement)

    Commit your Changes (git commit -m 'Add 6-DoF model structure')

    Push to the Branch (git push origin feature/BallisticsEnhancement)

    Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author

Developed by Diogo Fernandes

    GitHub: @DiogoFernandes-2008015
