import numpy as np
# import numpy.linalg as LA
# import scipy.constants as const


def gen_newtonian_motion_update_matrix(time_step:float, 
state_vec_order:int=2, approx_order:int=None ):
    """Generates the 'fundamental matrix' that transitions the state vector to the next time-step.
    
    x_vec(t + dt) = F . x_vec(t)

    F = I + A dt + A^2 dt^2 / 2! + ... A^n dt^n / n!
    where A is the matrix that defines the differential equation: x' = A . x.
    The state vector is x_vec = [x, x', x'', ... x^(k)] = [x_0, x_1, ... x_k], where x^(k) is the kth time derivative of the position x. 
    d(x_k)/dt = 0
    d(x_i)/dt = x_(i+1)

    This results in a simple form of A. Its elements are 1 for the first above diagonal and zeros otherwise. 
    Example: constant acceleration
    A = [ 0 1 0
          0 0 1
          0 0 0 ]
    F then becomes: F = I + A dt + A^2 dt^2 / 2! with higher order terms vanishing.
    F = [ 0 1 0.5
          0 0 1
          0 0 0 ]
    
    Parameters
    ----------
    time_step : float
        The time step, dt, over which the state vector is updated.
    state_vec_order : int, optional
        The order of the state vector, which is the order of the position derivative that is zero, by default 2, which corresponds to constant acceleration, x'' = 0.
    approx_order : int, optional
        A^n vanishes for n>state_vec_order, but the user may want to approx with fewer terms in the expansion of F, by default None which keeps all non-vanishing powers of A.

    Returns
    -------
    np.ndarray
        Return the F matrix that transitions the state vector to the next time step.
    """
    ndim_state = state_vec_order + 1
    # make the A matrix
    A = np.zeros([ndim_state, ndim_state])
    idx = np.arange(state_vec_order)
    A[idx, idx+1] = 1

    if approx_order is None:
        approx_order = state_vec_order
    largest_nonzero_power = min(state_vec_order, approx_order)
    F = np.eye(ndim_state)
    AA = np.eye(ndim_state)
    factorial = 1
    for i in range(1,largest_nonzero_power+1):
        # print("---")
        # print(i)
        AA = AA @ A
        # print(AA)
        factorial *= i
        # print(factorial)
        F += AA * time_step**i / factorial

    return F



class KalmanFilter():
    x: list[np.ndarray]
    P: list[np.ndarray]
    F: np.ndarray
    B: np.ndarray
    u: list[np.ndarray]
    Q: np.ndarray
    z: list[np.ndarray]
    H: np.ndarray
    R: np.ndarray
    K: list[np.ndarray]

    def __init__(self, x0, P0, F, B, u, Q, H, R):
        """
        System model prediction step
        x(t) = F(t).x(t-1) + B(t).u(t)
        P(t) = F(t).P(t-1).F(t)^T + Q(t)

        x - state vector
        F - transition matrix based on model
        B - control matrix that applies the controls state u(t) to the state
        u - state of the controls

        P - convariance matrix of the state
        Q - convariance matrix of system noise (e.g. windy weather randomly changes position and velocity)

        measurement update step
        x(t) = x(t-1) + K(t) . (z(t) - H(t).x(t-1))
        P(t) = P(t) - K(t).H(t).P(t)

        K = P(t).H(t)^T . (H(t).P(t).H(t)^T + R(t))^-1

        z - measurement vector
        H - transformation matrix that maps state to measurements
        R - covariance of measurement noise
        K - "Kalman filter gain"

        
        """
        self.x = [x0]
        self.P = [P0]
        
        self.F = F
        self.B = B
        self.u = [u]

        self.Q = Q

        self.z = []
        self.H = H
        self.R = R
        self.K = []
    def get_states(self, which='all'):
        """Return a np.ndarray of the state estimates. 
        note: self.x is a list of np.ndarrays, but method's output converts it to np.ndarray.

        Parameters
        ----------
        which : str, optional
            "all": return all x states including intervediates
            "estimates": return only x after each epoch (prediction+measurement updates)
            By default 'all'

        Returns
        -------
        np.ndarray
        """
        if which == "all":
            outarr = np.squeeze(self.x)
        elif which == "estimates":
            outarr = np.squeeze(self.x[2::2])
        else:
            raise ValueError("`which` must be one of {'all', 'estimates'}")
        return outarr
    
    def get_covariances(self, which='all'):
        """Return a np.ndarray of the covariances of state estimates. 
        note: self.P is a list of np.ndarrays, but method's output converts it to np.ndarray.

        Parameters
        ----------
        which : str, optional
            "all": return all x states including intervediates
            "estimates": return only x after each epoch (prediction+measurement updates)
            By default 'all'

        Returns
        -------
        np.ndarray
        """
        if which == 'all':
            outarr = np.array(self.P)
        elif which == 'estimates':
            outarr = np.array(self.P[2::2])
        else:
            raise ValueError("`which` must be one of {'all', 'estimates'}")
        return outarr
    
    def update_xPuK(self, xnew, Pnew, unew, Knew):
        """Append the new matricies for x, P, u, K.
        """
        xnew = self._shape_x(xnew)
        self.x.append(xnew)
        self.P.append(Pnew)
        unew = self._shape_u(unew)
        self.u.append(unew)
        self.K.append(Knew)

    def _shape_x(self, x):
        x = np.reshape(x, [-1,1])
        # dimensions check
        if (self.P) and ((self.P[-1].shape[0] != x.shape[0]) or (self.P[-1].shape[1] != x.shape[0])):
            raise ValueError("Dimenesions of state space are inconsistent. x must be [N x 1] and P must be [N x N]. ")
        elif (self.x) and (self.x[-1].shape != x.shape):
            raise ValueError(f"Dimensions of the state vector are inconsistent. x must have shape = {self.x[-1].shape}.")
        
        return x
    
    def _shape_u(self, u):
        if u is not None:
            u = np.reshape(u, [-1,1])
            if (self.B.shape[0] != u.shape[0]) or (self.B.shape[1] != u.shape[0]):
                raise ValueError("Dimensions of the controls u and B are inconsistent. B must be [M x M] and u must be [M x 1].")
            elif (self.u) and (self.u[-1].shape != u.shape):
                raise ValueError(f"Dimensions of the controls u are inconsistent. u must have shape = {self.u[-1].shape}.")
        
        return u    
        
    def model_predition_step(self, u=None):
        """Given a state vector and how it transitions in a single time-step based on a model, that may include the controls, predict the next state.

        x(t) = F(t).x(t-1) + B(t).u(t)
        P(t) = F(t).P(t-1).F(t)^T + Q(t)

        x - state vector
        F - transition matrix based on model
        B - control matrix that applies the controls state u(t) to the state
        u - state of the controls

        P - convariance matrix of the state
        Q - convariance matrix of system noise (e.g. windy weather randomly changes position and velocity)

        Parameters
        ----------
        u : ndarray, optional
            [n_controls x 1] state of the controls, by default None

        Returns
        -------
        xnew, Pnew, u : ndarray, ndarray, ndarray
            New predicted state, covariance, and control (passed from input)
        """
        xcurrent = self.x[-1]
        
        if u is not None:
            xnew = self.F @ xcurrent + self.B @ u
        else:
            xnew = self.F @ xcurrent
        
        Pcurrent = self.P[-1]
        Pnew = self.F @ Pcurrent @ self.F.T + self.Q
        
        return xnew, Pnew, u

    def measurement_update_step(self, z=None, postcovkwargs:dict={}):
        """Updates the state estimate based on a measurement.

        x(t) = x(t-1) + K(t) . (z(t) - H(t).x(t-1))
        P(t) = P(t) - K(t).H(t).P(t)

        K = P(t).H(t)^T . (H(t).P(t).H(t)^T + R(t))^-1

        z - measurement vector
        H - transformation matrix that maps state to measurements
        R - covariance of measurement noise
        K - "Kalman filter gain"

        Parameters
        ----------
        z : ndarray
            [n_meas x n_meas_space] measurements vector
        postcovkwargs : dict
            pass kwargs to the `calc_posterior_covariance()` method
        
        Returns
        -------
        xnew, Pnew : ndarray, ndarray
        """
        xcurrent = self.x[-1]
        Pcurrent = self.P[-1]
        
        K = self.calc_kalman_gain(Pcurrent)
        # no measurement, only K might have updated
        if z is None:
            return xcurrent, Pcurrent, K
        
        xnew = xcurrent + K @ (z - self.H @ xcurrent)
        Pnew = self.calc_posterior_covariance(Pcurrent, K, **postcovkwargs)

        return xnew, Pnew, K

    def calc_posterior_covariance(self, P, K, stable:bool=True):
        """The posterior covariance in the form:
        P(t) = ( I - K(t).H(t) ) . P(t)
        can become unstable over many iterations due to accumulation of floating point math errors. This stems from the subtraction in the parentheses, where K and/or H may have very small values for some elements. This could lead to non-symmetric P, but P is a covariance matrix and must be symmetric.

        Instead use the symmetrized version: P = (P + P^T) / 2
        see Labbe's chapter on "Kalman Filter Math"
        
        P = (I - K.H).P.(I - K.H)^T + K.R.K^T

        """
        # P = self.P[-1]
        Imat = np.eye(*P.shape) # eye needs shape tuple unpacked
        # K = self.K[-1]
        A = Imat - K @ self.H

        if stable:
            Pnew = A @ P @ A.T + K @ self.R @ K.T
        else:
            Pnew = A @ P
        
        return Pnew


    def calc_kalman_gain(self, P):
        """Calculate the Kalman gain, K, which specifies the optimal ratio of process vs. measurement estimate.

            K = P(t).H(t)^T . (H(t).P(t).H(t)^T + R(t))^-1
        """
        
        S_ = np.linalg.inv(self.H @ P @ self.H.T + self.R)
        K = P @ self.H.T @ S_
        # self.K.append(K)
        
        return K

    def perform_epoch(self, u=None, z=None, postcovkwargs=None):
        """_summary_

        Parameters
        ----------
        u : ndarray, optional
            _description_, by default None
        z : ndarray, optional
            _description_, by default None
        postcovkwargs : dict, optional
            _description_, by default None

        Returns
        -------
        ndarray, ndarray
            state and state covariance estimates
        """
        xnew, Pnew, u = self.model_predition_step(u=u)
        self.update_xPuK(xnew, Pnew, u, None)
        xnew, Pnew, K = self.measurement_update_step(z=z, postcovkwargs=postcovkwargs)
        self.update_xPuK(xnew, Pnew, None, K)
    
        return self.x[-1], self.P[-1]
    


__namespace__ = 'stateest'
__author__ = 'Ivan Gadjev'
__year__ = '2026'
if __name__ == '__main__':
    print(f"This is the {__namespace__}. Written by {__author__}, {__year__}.")

    