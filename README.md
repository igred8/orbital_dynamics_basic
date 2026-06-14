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

## Description of the numerical simulation

The simulation of orbital trajectories is based on a step-by-step integration of the particle-like object's acceleration. The Earth's gravitational potential gives the object's acceleration at a given point in space. From the acceleration, the velocity and then the position are calculated by a "Velocty Verlet" intergration step. The Verlet integrator provides 
- numerical stability for the particle trajectories 
- time reversibility
- preservation of the system's phase-space volume 

The **velocity Verlet** recursion equations for Newton's equations are:

1. Compute the new position, $r_{k+1}$ from the previous state $x_{k} = (r,v,a)_{k}$
  $$r_{k+1} = r_k + \Delta t v_k + \frac{a_k^2}{2}\Delta t$$

2. Compute the new acceleration from the potential based on the new position
  $$a_{k+1} = A(r_{k+1})$$

3. Compute the new veloctiy from the mean of the new and old accelerations
  $$v_{k+1} = v_k + \frac{(a_k + a_{k+1})}{2}\Delta t$$

In general, the time-step, $\Delta t$ can also depend on the iteration number i.e. $\Delta t_k$. Also note that the acceleration update assumes that the acceleration at a time-step $k$ only depends on the position of the particle at that time-step. 


Note that since the velocity Verlet requires knowledge of the kinematic state in the previous two steps, an initialization procedure is needed. Fortunately, it is sufficient to use a basic forward Euler integration scheme to generate the first two states. 

Each orbit is given an initial position and veloctiy, with the initial acceleration being computed from the initial position i.e.:

**Initial conditions**
$$r_{0}$$
$$v_{0}$$
$$a_{0} = A(r_{0})$$

The next step is computed as:

**Forward Euler**:
$$r_{1} = r_{0} + v_{0}\Delta t + \frac{a_{0}}{2}\Delta t^{2} $$
$$v_{1} = v_{0} + a_{0}\Delta t$$
$$a_{1} = A(r_{1})$$

As mentioned above, it is possible to carry the Forwar Euler integration to simulate the orbit of the object, but this technique does not preserve the system's phase-space volume, is not time-reversible, and can be numerically unstable. Hence, after initialization, the simulation is carried through with the velocity Verlet scheme.


### J2 correction
The J2 correction is a second order term in the spherical harmonics expansion of Earth's gravitational potential. It accounts for the slight, axially symmetric flattening at poles due to Earth's rotation. The estimated amplitude of the J2 term is $J_2 = 0.00162$, which is about a thousand times smaller than the perfect sphere potential term. 

## State estimation

### Kalman Filter

### Measurement Simulation


## Development to-do's:

2026.05
- `done` KF state estimation on simulated orbital data initial implementation 
- `TODO` stress test KF on sim data with different scenarios
- `look into` When using the J2 correction and populating orbits with different radial initial distances, there appear discrete jumps in the orbital paths. Smells like a floating point round off, but I am using float64 in the integration scheme.
  - not a big concern for showing how KF works on this data, but something to look into
- `TODO` write up KF equations on README
- `fixed` bug in orbit propagation step, where the new position, velocity, and accel get added to the snapshots. I had switched the order of arguments.
- `todo` When J2 correction is ON and the starting position is on the equator with varying radius there appear "bands" of orbits. It smells like a floating point issue to me. Should make sure all arrays are float64. 

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
 