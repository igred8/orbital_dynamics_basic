# Orbital Dynamics Exploration and State Estimation
---

Ivan Gadjev, 2026
---

This project is an exploration into the simulation of orbital dynamics with some state estimation techniques. 

The particle simulation functionality is in `src/satellites.py` where the particle simulation object and methods are defined. Working examples are presented inside `/notebooks`
The state estimation methods are in `src/stateest.py`. 

Development todo's:

2026.02
- done Implement Kalman filter
- done Set up a basic example for KF
- Add ellipse of covariance to state estimate in the x-v space
- Use orbit simulation data as input to KF and make a state estimation flow
- 


2026.01
1. done Minimal example including Earth's gravity
2. done Use "Velocity Verlet" update step for the orbit integrator. 
   - done The Verlet integrator conserves the energy of the particle system, which is something the forward Euler method does not guarantee.
   - done The Velocity Verlet integrator is widely used in orbital dynamics for satellites, becuase it explicitly calculates and tracks the velocity of the orbiting object i.e. velocity is not a derived quantity.
3. done Bring in `pyvista` visualization suite for particles
4. Incorporate second order effects:
   - done J2 oblateness of Earth
     - done add comparison of orbits
   - Drag for LEO
   - Moon's gravity 
5. Incorporate thrusts for maneuvers 
6. Convert between ECEF and LLA coordinates
 