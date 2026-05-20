from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from utils import rng_initiator

class MeasurementDevice(ABC):
    """Abstract class for devices.

    Parameters
    ----------
    device_name : str, int
        A name for the device. Not intended as a unique identifier. 
    rng : int, np.randomg.Generator
        Sets the RNG for this instance.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('device_name', '0000')
        self.rng = rng_initiator(kwargs.get('rng',None))

    @abstractmethod
    def measure(self, state_vector:NDArray):
        pass


class Position(MeasurementDevice):
    """Rudimentary measurement device that adds Gaussian noise to the position coordinates of the state vector.

    Parameters
    ----------
    n_dims : int
        Number of position dimensions in the measurement space of this device.
    mean : NDArray
        shape (n_position_coordinates,) must be 1 dim vector for multivariate normal
    covariance : NDArray
        shape (n_position_coordinates, n_position_coordinates)
    kwargs : key word arguments passed to `MeasurementDevice`. See MeasurementDevice docs.

    
    """
    n_dims : int
    mean : NDArray
    covariance : NDArray

    def __init__(self, noise_mean, noise_cov, **kwargs):
        super().__init__(**kwargs)
        self.n_dims = noise_mean.shape[0]
        self.mean = noise_mean
        self.covariance = noise_cov

    def _gen_state2measurement_mat(self, state_order:int=1):
        """Generate a measurement matrix that 
        Assume: If the state is of order n, then it is in this order [x, vx, ax, ... x_n, y, vy, ay, ...y_n]
        The position coordinates in this 2D case are then [x,y]
        Parameters
        ----------
        state_order : int, optional
            What order is the state, by default 1 corresponds to constant velocity model.
        """
        stride = state_order + 1
        meas_mat = np.zeros([self.n_dims, self.n_dims*stride])
        for i in range(self.n_dims):
            meas_mat[i, i*stride] = 1
        return meas_mat


    def measure(self, state_vector:NDArray, state_order:int=1):
        meas_mat = self._gen_state2measurement_mat(state_order=state_order)
        state_pos = meas_mat @ state_vector # only get the position coordinates
        noise = self.rng.multivariate_normal(self.mean, self.covariance, size=state_pos.shape[1])
        return state_pos + noise.T
        




__namespace__ = 'measurement'
__author__ = 'Ivan Gadjev'
__year__ = '2026'
if __name__ == '__main__':
    print(f"This is the {__namespace__}. Written by {__author__}, {__year__}.")
