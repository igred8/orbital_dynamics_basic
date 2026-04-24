import numpy as np
from scipy.stats import multivariate_normal

def generate_covariance_ellipse(mu, cov, enclosed_frac=0.95, npts=100):
    """Create the representative ellipse outline of the covariance matrix centered at the mean.

    Parameters
    ----------
    mu : np.ndarray
        Mean of the Gaussian shape (2x1)
    cov : np.ndarray
        Covariance matrix shape (2x2)
    enclosed_frac : float, optional
        What fraction of the Guassian's volume is enclosed by the ellipse, by default 0.95. One sigma ellipse encloses ~0.39 of Guassian volume.
    npts : int, optional
        Number of points to plot. More = smoother. By default 100

    Returns
    -------
    np.ndarray
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

def get_position_covariances(cov, state_order=1):
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

def polar_to_cartesian(rvec, thvec):
    xvec = rvec * np.cos(thvec)
    yvec = rvec * np.sin(thvec)
    return xvec, yvec

def transform_polar_Gaussian_to_Cartesian(r_phi_mean, r_phi_covariances):
    
    # sample from r-phi Gaussian distribution
    rphi_dist = multivariate_normal(r_phi_mean, r_phi_covariances)
    rphi_samples = rphi_dist.rvs(10000)
    # transform r-phi samples to x-y
    x_samples, y_samples = polar_to_cartesian(rphi_samples[:,0], rphi_samples[:,1])
    xy_samples = np.vstack((x_samples, y_samples)).T
    # estimate mean and covariances
    xy_mean = np.mean(xy_samples, axis=0)
    xy_covariances = np.cov(xy_samples.T)
    return xy_mean, xy_covariances
