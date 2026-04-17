import numpy as np

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
