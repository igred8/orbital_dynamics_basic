import numpy as np
import numpy.linalg as LA
import scipy.constants as const
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import pyvista as pv

import sys
sys.path.append('E:/google-drive/py_projects/satellites/src')
import satellites as sat


XHAT  = np.reshape( [1,0,0], [1,3])
YHAT  = np.reshape( [0,1,0], [1,3])
ZHAT  = np.reshape( [0,0,1], [1,3])


if __name__ == "__main__":

    dist1 = sat.EARTH_RADIUS_M + 1000
    vel1 = 7904.76692
    mass1 = 100

    npts = 7
    rng = np.random.default_rng( seed = 123 )

    rdilate = rng.uniform(0.9, 1.3, [npts,1])
    phi = const.pi * rng.uniform(0, 2, [npts,1])
    theta = const.pi * rng.uniform(0.25, 0.55, [npts,1])
    velang = np.deg2rad( rng.uniform( 0, 3, [npts,1] ))
    
    rvec = rdilate * (
          np.cos(phi)*np.sin(theta)*XHAT
        + np.sin(phi)*np.sin(theta)*YHAT
        + np.cos(theta)*ZHAT
        )
    thvec = (
          -np.sin(phi)*np.sin(theta)*XHAT
        + np.cos(phi)*np.sin(theta)*YHAT
        + np.cos(theta)*ZHAT
        )

    sats = []
    for r,th,va in zip(rvec, thvec, velang):
        rot = Rotation.from_rotvec( va * ZHAT )
        sats.append(
            sat.GravityObject( 
                xyz_m = dist1 * r,
                vxyz_mps = rot.as_matrix() @ (vel1*th),
                mass_kg=mass1
            )
        )

    esys = sat.EarthSystem( delta_time_s=2 )
    for s in sats:
        esys.add_object( s )

    esys.init_snapshots( n_time_steps=2 )
    
    for i in range(20000):
        esys.propagate_step()
        
    r = np.stack(esys.positions)
    v = np.stack(esys.velocities)
    a = np.stack(esys.accels)

    plotter = pv.Plotter()
    cloud = pv.PolyData(r[0] / sat.EARTH_RADIUS_M)
    cloud['rdilate'] = rdilate
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=6, cmap='viridis')
    # plotter.enable_eye_dome_lighting()
    plotter.open_movie("orbits.mp4", framerate=30)
    # plotter.show()

    for k in range(0, len(r), 11):   # downsample frames
        cloud.points = r[k] / sat.EARTH_RADIUS_M 
        # plotter.render()
        plotter.write_frame(  )
    plotter.close()

    rnorm = LA.norm(r, ord=2, axis=2)
    vnorm = LA.norm(v, ord=2, axis=2)

    energy_gravity = np.sum( const.G * sat.EARTH_MASS_KG * esys.masses.T / rnorm, axis=1 )
    energy_kinetic = 0.5 * np.sum( esys.masses.T * vnorm**2, axis=1 )

    print(energy_gravity.shape)
    print(energy_kinetic.shape)

    fig, ax = plt.subplots(1,1)

    ax.plot( energy_kinetic[2:]+energy_gravity[2:])
    plt.show()  