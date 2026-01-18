import numpy as np
import numpy.linalg as LA
import scipy.constants as const
from scipy.spatial.transform import Rotation

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

    rng = np.random.default_rng( seed = 123 )
    npts = 3000
    dilate = rng.uniform(0.9, 1.3, npts)
    thetas = rng.uniform(0, 2*const.pi, npts)
    velang = np.deg2rad( rng.uniform( 0, 30, npts ))
    
    sats = []
    for dil,th,va in zip(dilate,thetas, velang):
        rhat_ = np.cos(th)*XHAT + np.sin(th)*YHAT
        rot = Rotation.from_rotvec( va * ZHAT )
        sats.append(
            sat.GravityObject( 
                xyz_m = dist1 * dil* rhat_,
                vxyz_mps = rot.as_matrix() @ np.transpose(vel1 * LA.cross( rhat_, -ZHAT )),
                mass_kg=mass1
            )
        )

    esys = sat.EarthSystem( delta_time_s=2 )
    for s in sats:
        esys.add_object( s )

    esys.init_snapshot()
    esys.calc_init_states()
    for i in range(1000):
        esys.propagate_step()
        
    r = np.stack(esys.positions)
    v = np.stack(esys.velocities)
    a = np.stack(esys.accels)

    plotter = pv.Plotter()
    cloud = pv.PolyData(r[0] / sat.EARTH_RADIUS_M)
    cloud['dilate'] = dilate
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=6, cmap='berlin')
    # plotter.enable_eye_dome_lighting()
    plotter.open_movie("orbits.mp4", framerate=30)
    # plotter.show()

    for k in range(0, len(r), 11):   # downsample frames
        cloud.points = r[k] / sat.EARTH_RADIUS_M 
        # plotter.render()
        plotter.write_frame(  )
    plotter.close()
