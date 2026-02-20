import numpy as np
import numpy.linalg as LA
import scipy.constants as const


class KalmanFilter():

    def __init__(self, x0, P0, F, B, u, Q, H, R):
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

    def update_xPuK(self, xnew, Pnew, unew, Knew):
        """Append the new matricies for x, P, u, K.
        """
        self.x.append(xnew)
        self.P.append(Pnew)
        self.u.append(unew)
        self.K.append(Knew)

    def model_predition_step(self, u=None):
        """Given a state vector and how it transitions in a single time-step based on a model, that may include the controls, predict the next state.

        x(t) = F(t).x(t-1) + B(t).u(t)
        P(t) = F(t).P(t-1).F(t)^T + Q(t)

        x - state vector
        F - transition matrix based on model
        B - control matrix that applies the controls state u(t) to the state
        u - state of the controls

        P - convariance matrix of the state
        Q - convariance matrix of noisy controls

        Parameters
        ----------
        u : ndarray, optional
            [n_controls x 1] state of the controls, by default None

        Returns
        -------
        xnew, Pnew, u : ndarray, ndarray, ndarray
            _description_
        """
        xcurrent = self.x[-1]
        self.u.append(u)
        if u is not None:
            xnew = self.F @ xcurrent + self.B @ u
        else:
            xnew = self.F @ xcurrent
        self.x.append(xnew)

        Pcurrent = self.P[-1]
        Pnew = self.F @ Pcurrent @ self.F.T + self.Q
        self.P.append(Pnew)

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

        # self.x.append(xnew)
        # self.P.append(Pnew)

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
        I = np.eye(*P.shape) # eye needs shape tuple unpacked
        # K = self.K[-1]
        A = I - K @ self.H

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

    