import numpy as np
import numpy.linalg as LA
import scipy.constants as const

class GravityObject():
    def __init__(self, xyz_m:list=[0,0,0], vxyz_mps:list=[0,0,0], mass_kg:float=1 ):
        self.position = np.array( xyz_m ).reshape([1,3])
        self.velocity = np.array( vxyz_mps ).reshape([1,3])
        self.mass = mass_kg


EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_M = 6378137.0
MOON_MASS_KG = 7.346e22
EARTH_MOON_DIST_M = 384784000

class EarthSystem():
    def __init__(self, delta_time_s:float=1.0):
        self.objects = []
        self.positions = None
        self.velocities = None
        self.accels = None
        self.masses = None

        self.delta_time_s = delta_time_s

    def add_object(self, gravobj:GravityObject ):
        self.objects.append(gravobj)
    
    def init_snapshot(self):
        if not self.objects:
            return 1
        
        nobj = len(self.objects)
        self.positions = [np.zeros( [nobj, 3] )]
        self.velocities = [np.zeros( [nobj, 3] )]
        self.accels = [np.zeros( [nobj, 3] )]
        self.masses = np.zeros( [nobj, 1] )

        for i,o in enumerate(self.objects):
            self.positions[0][i,:] = np.array( o.position )
            self.velocities[0][i,:] = np.array( o.velocity )
            self.accels[0][i,:] = self.calc_gravity_accel( o.position )
            self.masses[i,:] = np.array( o.mass )

        return 0
    
    def add_snapshot( self, pos_new, vel_new, accel_new ):
        if (self.positions is None) or (self.velocities is None):
            raise ValueError("Please initialize the position and velocity arrays for the objects.")
        self.positions.append( pos_new )
        self.velocities.append( vel_new )
        self.accels.append( accel_new )

    def calc_gravity_accel(self, xyz ):
        distances = LA.norm( xyz, ord=2, axis=1 ).reshape([-1,1])
        grav_accel = -const.G * EARTH_MASS_KG * xyz / distances**3
        return grav_accel
        
    def calc_init_states( self ):
        no_objects_present = self.init_snapshot()
        if no_objects_present: 
            raise ValueError("No objects present. Exiting time-step propagation.")
            
        a0 = self.accels[0]
        v0 = self.velocities[0]
        r0 = self.positions[0]

        r1 = r0 + v0*self.delta_time_s + 0.5*a0*self.delta_time_s**2
        v1 = v0 + a0*self.delta_time_s
        a1 = self.calc_gravity_accel( r1 )

        self.add_snapshot( r1, v1, a1 )

    def propagate_step( self ):
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

    