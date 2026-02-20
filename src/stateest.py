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

        return xnew, Pnew

    def measurement_update_step(self, z):
        """Updates the state estimate based on a measurement.

        x(t) = x(t-1) + K(t) . (z(t) - H(t).x(t-1))
        P(t) = P(t) - K(t).H(t).P(t)

        K = P(t).H(t)^T . (H(t).P(t).H(t)^T + R(t))^-1

        z - measurement vector
        H - transformation matrix that maps state to measurements
        R - covariance of measurement noise
        K - "Kalman filter gain"
        """
        xcurrent = self.x[-1]
        Pcurrent = self.P[-1]
        
        K = self.calc_kalman_gain() # updates the Kalman gain
        
        xnew = xcurrent + K @ (z - self.H @ xcurrent)
        Pnew = Pcurrent - K @ self.H @ Pcurrent
        
        self.x.append(xnew)
        self.P.append(Pnew)

        return xnew, Pnew


    def calc_kalman_gain(self):
        """Calculate the Kalman gain, K, which specifies the optimal ratio of process vs. measurement estimate.

            K = P(t).H(t)^T . (H(t).P(t).H(t)^T + R(t))^-1
        """
        Pcurrent = self.P[-1]
        S_ = np.linalg.inv(self.H @ Pcurrent @ self.H.T + self.R)
        K = Pcurrent @ self.H.T @ S_
        self.K.append(K)
        
        return K


__namespace__ = 'stateest'
__author__ = 'Ivan Gadjev'
__year__ = '2026'
if __name__ == '__main__':
    print(f"This is the {__namespace__}. Written by {__author__}, {__year__}.")

    