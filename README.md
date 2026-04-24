# Orbital Dynamics Exploration and State Estimation
---

Ivan Gadjev, 2026
---

This project is an exploration into the simulation of orbital dynamics with some state estimation techniques. 

## References
Orbital dynamics literature references:
1. Montenbruck and Gill, "Satellite Orbits", 2000

The implementation of a Kalman Filter (KF) is done according to standard literature. For references see:
1. Labbe, "Kalman and Bayesian Filters in Python", 2020
2. Murphy, "Machine Learning: A Probabilistic Perspective", 2012
3. Welch Bishop, "Kalman Intro", 2006


## Structure 
The particle simulation functionality is in `src/satellites.py` where the particle simulation object and methods are defined. Working examples are presented inside `/notebooks`
The state estimation methods are in `src/stateest.py`. 


## Development to-do's:
2026.04
- Continue Kalman filter updates and refinement
  - Constant velocity example is showing strange behavior
    - `fixed` (the state transition matrix for newtonian motion had a bug) estimates are worse for shorter time steps. systematically short of true position and measurements
    - `fixed` (get_covarriances was not striding over intermediate steps) covariances oscillate in amplitude?
    - `fixed` (it was a bug)these seem like bugs in the code, but could they be due to KF itself for given params?
  - `done` Constant velocity in 2D position
  - `done` Constant velocity in 2D polar coordinates
  - `done` Constant acceleration in 2D position
  - 
- Use orbit simulation data as input to KF and make a state estimation flow
- 

2026.03
- `ongoing` Read KF literature

2026.02
- `done` Implement Kalman filter
- `done` Set up a basic example for KF
- `done` Add ellipse of covariance to state estimate in the x-v space
- Use orbit simulation data as input to KF and make a state estimation flow
- 


2026.01
1. `done` Minimal example including Earth's gravity
2. `done` Use "Velocity Verlet" update step for the orbit integrator. 
   - `done` The Verlet integrator conserves the energy of the particle system, which is something the forward Euler method does not guarantee.
   - `done` The Velocity Verlet integrator is widely used in orbital dynamics for satellites, becuase it explicitly calculates and tracks the velocity of the orbiting object i.e. velocity is not a derived quantity.
3. `done` Bring in `pyvista` visualization suite for particles
4. Incorporate second order effects:
   - `done` J2 oblateness of Earth
     - `done` add comparison of orbits
   - Drag for LEO
   - Moon's gravity 
5. Incorporate thrusts for maneuvers 
6. Convert between ECEF and LLA coordinates
 