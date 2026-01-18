# Orbital Dynamics Exploration
---

Ivan Gadjev, 2026
---

This project is an exploration into the simulation of orbital dynamics. 

Roadmap of development:

1. Minimal example including Earth's gravity
2. Use "Velocity Verlet" update step for the orbit integrator. 
   - The Verlet integrator conserves the energy of the particle system, which is something the forward Euler method does not guarantee.
   - The Velocity Verlet integrator is widely used in orbital dynamics for satellites, becuase it explicitly calculates and tracks the velocity of the orbiting object i.e. velocity is not a derived quantity.
3. Bring in `pyvista` visualization suite for particles
4. Incorporate second order effects:
   - J2 oblateness of Earth
   - Drag for LEO
   - Moon's gravity 
5. Incorporate thrusts for maneuvers 
6. Convert between ECEF and LLA coordinates
