# Orbital Dynamics Exploration
---

Ivan Gadjev, 2026
---

This project is an exploration into the simulation of orbital dynamics. 

The main functionality is in `src/satellites.py` where the particle simulation object and methods are defined. Working examples are presented inside `/notebooks`

Roadmap of development:

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
 