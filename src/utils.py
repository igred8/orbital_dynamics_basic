

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

def generate_covariance_ellipse(mu, cov, enclosed_frac=0.95, npts=100):
    """Create the representative ellipse outline of the covariance matrix centered at the mean.

    Parameters
    ----------
    mu : NDArray
        Mean of the Gaussian shape (2x1)
    cov : NDArray
        Covariance matrix shape (2x2)
    enclosed_frac : float, optional
        What fraction of the Guassian's volume is enclosed by the ellipse, by default 0.95. One sigma ellipse encloses ~0.39 of Guassian volume.
    npts : int, optional
        Number of points to plot. More = smoother. By default 100

    Returns
    -------
    NDArray
        shape (2, npts) x-y coordinates of the ellipse
    """
    evals, evecs = np.linalg.eigh(cov)
    scale_factor = -2 * np.log(1 - enclosed_frac) 

    a_major = np.sqrt(scale_factor * evals[1])
    v_major = evecs[:,1].reshape([-1,1])
    a_minor = np.sqrt(scale_factor * evals[0])
    v_minor = evecs[:,0].reshape([-1,1])
    
    mu = np.reshape(mu, v_major.shape)
    thvec = np.linspace(0, 2*np.pi, npts)
    xy = (
        a_major*np.cos(thvec)*v_major + 
        a_minor*np.sin(thvec)*v_minor + 
        mu
    )
    return xy

def get_position_covariances(cov:NDArray, state_order=1):
    """Extract the covariance matrix of the position coordinates for a state.
    Assumptions:
    If the state is of order n, then it is in this order [x, vx, ax, ... x_n, y, vy, ay, ...y_n]
    The covariance matrix of positions will then be composed from the full covariance matrix S as:
    [ S[0,0], S[0,n+1],
      S[n+1,0], S[n+1, n+1] ]
    
    Parameters
    ----------
    cov : NDArray
        Full covariance matrix of state
    state_order : int, optional
        What order is the state, by default 1 corresponds to constant velocity model.

    Returns
    -------
    NDArray
        shape (n_position_dim, n_position_dim) The output covariance of the position coordinates.
        In 3D space, the shape will be (3,3).
    """
    ndims = cov.shape[0]
    cc, rr = np.meshgrid(np.arange(ndims), np.arange(ndims))
    stride = state_order + 1
    rr = (rr % stride) == 0
    nr = np.sum(rr, axis=0)[0]
    # print(rr)
    cc = (cc % stride) == 0
    nc = np.sum(cc, axis=1)[0]
    # print(cc)
    mask = rr & cc
    # print(mask)
    return cov[mask].reshape([nr, nc])

def polar_to_cartesian(rvec:NDArray, thvec:NDArray, th_degrees:bool=False):
    """Transform from r-theta to x-y coordinates. 

    Parameters
    ----------
    rvec : NDArray
        Values for the r coordinate. rvec and thvec should have shapes that are broadcastable to eachother.
    thvec : NDArray
        Values for the theta coordinate in radians by default.
    th_degrees : bool, optional
        Flag to switch between readian and degrees for the theta vector. By default False, i.e. use radians.
    
    Returns
    -------
    NDArray, NDArray
        
    """
    if th_degrees:
        thvec = np.deg2rad(thvec)

    xvec = rvec * np.cos(thvec)
    yvec = rvec * np.sin(thvec)
    return xvec, yvec

def transform_polar_Gaussian_to_Cartesian(r_phi_mean:NDArray, r_phi_covariances:NDArray, _n_samples:int=10000):
    """Calculate a mean and covariance for a Gaussian in Cartesian coordinates, 
    based on samples from a Gaussian in polar coordinates.
    In general, a Gaussian distribution in polar coordinates G(r,th; mu, sig) does not transform to a Gaussian in Cartesian.
    Steps:
    1. Sample from the specified polar Gaussian
    2. Transform r-th samples to x-y
    3. Calculate mean and covariance of the x-y points


    Parameters
    ----------
    r_phi_mean : NDArray
        A 2D vector specifying the means in r and theta
    r_phi_covariances : NDArray
        shape (4, 4) Covariance matrix in r-theta space
    _n_samples : int, optional
        The number of samples to draw for the calculation. Larger number of samples yields more stable results
        The default value, 10000, is usually sufficient.

    Returns
    -------
    NDArray, NDArray
        mean, covariance of the x-y distribution of samples
    """
    # sample from r-phi Gaussian distribution
    rphi_dist = multivariate_normal(r_phi_mean, r_phi_covariances)
    rphi_samples = rphi_dist.rvs(_n_samples)
    # transform r-phi samples to x-y
    x_samples, y_samples = polar_to_cartesian(rphi_samples[:,0], rphi_samples[:,1])
    xy_samples = np.vstack((x_samples, y_samples)).T
    # estimate mean and covariances
    xy_mean = np.mean(xy_samples, axis=0)
    xy_covariances = np.cov(xy_samples.T)
    return xy_mean, xy_covariances
