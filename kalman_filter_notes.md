# Kalman Filter State Estimation
---

## References

- Labbe - Kalman and Bayesian Filters in Python 2020
- Murphy - Machine Learning: A Probabalistic Perspective 2012
- Welch Bishop - Kalman Intro 2006

---

### notation note
The measurement step in Murphy looks a bit different from that in Labbe, because Murphy defines the measurement from the state and the controls. Labbe, on the other hand, defines the measurement as a transformation of the state (which was predicted using last state and controls) into measurement space. As far as I can tell this is just a matter of convention and difference in matrix equations. In matrix form, the measurement update step:
$$ x_t = x_{t-1} + K (\bar{z_t} - z_{t-1}) $$

Where, the measured vector is $\bar{z_t}$ and the predicted measured vector is $z_{t-1}$. The predicted measured vector is:
$$ z_(t-1) = H( F x_{t-1} + B u_{t-1} ) $$ 

In Murphy this is essentially rewritten with $C = HF$ and $D = HB$. 

### notation note
Notice that in Murphy, the state estimate *is* the mean of the posterior Gaussian, $\mu_{t|t-1}$, while Labbe calls it $x_k$ for the kth value of t. Which is a bit clunky because then the covariance matrix, which is $\Sigma_{t|t-1}$ in Murphy, is called $P_k$. I think the Murphy notation is clearer because it makes explicit that our estimate is just the mean of the Bayesian updated Gaussian and our uncertainty is the covariance matrix of that Gaussian. 

## Newtonian Dynamics Models

### SHO

$$ -kx = ma $$
$$ \frac{d^2x}{dt^2} = -\omega^2 x $$
The general solution is the complex exponential:
$$ x(t) = Ae^{i\omega t} + Be^{-i\omega t} $$
The initial conditions will define the constants.
The position initial condition:
$$ x(t=0) = x_0 $$
$$ A + B = x_0 $$ 
The velocity initial condition:
$$ x'(t=0) = v_0 $$
$$ A - B = -iv_0/\omega $$
From these linear algebraic equations we solve for A and B:
$$ A = \frac{1}{2}x_0 - \frac{iv_0}{2\omega} $$
$$ B = \frac{1}{2}x_0 + \frac{iv_0}{2\omega} $$

