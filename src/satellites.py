import numpy as np
import numpy.linalg as LA
import scipy.constants as const

class GravityObject():
    def __init__(self, xyz_m:list=[0,0,0], vxyz_mps:list=[0,0,0], mass_kg:float=1 ):
        self.position = np.array( xyz_m ).reshape([1,3])
        self.velocity = np.array( vxyz_mps ).reshape([1,3])
        self.mass = mass_kg


# use GM_factor factor in gravity calcs as it is much more accurately known than either G or M_Earth
GM_factor = 3.98600441889e14 # m^3 s^-2

EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_M = 6378137.0
MOON_MASS_KG = 7.346e22
EARTH_MOON_DIST_M = 384784000

class EarthSystem():
    """Particle simulation container for orbital dynamics of objects in the Earth's gravity. 
    
    Attributes
    ---
    self.objects — a list that will hold GravityObject instances (satellites, etc.) added via add_object()
    self.positions, self.velocities, self.accels, self.masses — all None until init_snapshots() is called, at which point they become lists of NumPy arrays tracking state over time
    """
    def __init__(self, j2_correction:bool=True, delta_time_s:float=1.0):
        """Constructor for EarthSystem

        Parameters
        ----------
        j2_correction : bool, optional
            Toggles the J2 oblateness correction in gravity calculations (accounts for Earth being slightly flattened at the poles), by default True
        delta_time_s : float, optional
            The simulation timestep in seconds, used by the Euler and Verlet integrators, by default 1.0
        """
        self.objects = []
        self.positions = None
        self.velocities = None
        self.accels = None
        self.masses = None

        self.j2_correction = j2_correction
        self.delta_time_s = delta_time_s

    def add_object(self, gravobj:GravityObject ):
        self.objects.append(gravobj)
    
    def init_snapshots(self, n_time_steps:int=2):
        """Calculate the state of the system in the first two time steps. 
        Needed to get the Verlet integration going.

        Raises
        ------
        ValueError
            Needs there to have been objects defined for the class instance.

        Parameters
        ----------
        n_time_steps : int, optional
            Number of time steps to initialize. Need 2 steps if using Verlet, by default 2

        Returns
        -------
        int
            0 - completed, 1 - failed
        """
        if not self.objects:
            return 1
        
        nobj = len(self.objects)
        self.positions = [np.zeros( [nobj, 3] )]
        self.velocities = [np.zeros( [nobj, 3] )]
        self.accels = [np.zeros( [nobj, 3] )]
        self.masses = np.zeros( [nobj, 1] )

        # init the snapshot arrays from all objects
        for i,o in enumerate(self.objects):
            self.positions[0][i,:] = np.array( o.position )
            self.velocities[0][i,:] = np.array( o.velocity )
            self.masses[i,:] = np.array( o.mass )
        self.accels[0] = self.calc_gravity_accel( self.positions[0] )
        
        # init the following steps with forward Euler
        for i in range(n_time_steps-1):
            ri, vi, ai = self._forward_euler( self.positions[-1], self.velocities[-1], self.accels[-1] )
            self.add_snapshot( ri, vi, ai )
        return 0
    
    def add_snapshot( self, pos_new, vel_new, accel_new ):
        """Append the given arrays as new elements in the lists.

        Parameters
        ----------
        pos_new : ndarray
            XYZ positions 
        vel_new : ndarray
            Velocities 
        accel_new : ndarray
            Accelerations

        Raises
        ------
        ValueError
            Needs the position, velocity, and acceleration arrays to have been initialized, typically using the `calc_init_states` method.
        """
        if (self.positions is None) or (self.velocities is None):
            raise ValueError("Please initialize the position and velocity arrays for the objects. Typically done with `calc_init_states`.")
        self.positions.append( pos_new )
        self.velocities.append( vel_new )
        self.accels.append( accel_new )

    def calc_gravity_accel(self, xyz ):
        """With Earth as the center of the gravitational well, calculate the acceleration at the given points.
        
        No corrections due to oblateness of Earth.

        Parameters
        ----------
        xyz : ndarray
            [N x 3] array of the XYZ coordinates of the points at which to calculate the acceleration due to gravity.

        Returns
        -------
        ndarray
            [N x 3] array of the vectors of the acceleration at the given points.
        """
        distances = LA.norm( xyz, ord=2, axis=1 ).reshape([-1,1])
        r_unit = xyz / distances
        grav_accel_00 = (-GM_factor / distances**2) * r_unit

        if self.j2_correction:
            J2 = 0.00162
            x_ = xyz[:,0].reshape([-1,1])
            y_ = xyz[:,1].reshape([-1,1])
            z_ = xyz[:,2].reshape([-1,1])
            grav_accel_20x = -GM_factor*EARTH_RADIUS_M*J2 * (x_ / distances**7) * (5*z_ - distances**2)
            grav_accel_20y = -GM_factor*EARTH_RADIUS_M*J2 * (y_ / distances**7) * (5*z_ - distances**2)
            grav_accel_20z = -GM_factor*EARTH_RADIUS_M*J2 * (z_ / distances**7) * (5*z_ - 3*distances**2)

            grav_accel_20 = np.array( [grav_accel_20x, grav_accel_20y, grav_accel_20z] ).reshape([-1,3])

            return grav_accel_00 + grav_accel_20
        else:
            return grav_accel_00
        
    def _forward_euler( self, r0,v0,a0 ):
        """Make a step in time using the forward Euler integrator. Should just be used to init snapshots.
        
        """
            
        r1 = r0 + v0*self.delta_time_s + 0.5*a0*self.delta_time_s**2
        v1 = v0 + a0*self.delta_time_s
        a1 = self.calc_gravity_accel( r1 )

        return r1, v1, a1

    def propagate_step( self ):
        """Calculate the position, velocity, and acceleration of the objects for the next time-step, using the velocity Verlet integration.
        Append the arrays by calling `add_snapshot`
        """
        # Velocity Verlet step
        r_new = self.positions[-1] + self.velocities[-1]*self.delta_time_s + 0.5*self.accels[-1]*self.delta_time_s**2
        a_new = self.calc_gravity_accel( r_new )
        v_new = self.velocities[-1] + 0.5*( self.accels[-1] + a_new )*self.delta_time_s

        self.add_snapshot( r_new, a_new, v_new )
        
__namespace__ = 'satellites'
__authoer__ = 'Ivan Gadjev'
__year__ = '2026'
if __name__ == '__main__':
    print(f"This is the {__namespace__}. Written by {__author__}, {__year__}.")

    