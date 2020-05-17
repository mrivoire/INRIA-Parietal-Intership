import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz 
from numpy.random import randn

#####################################################################################
#                      Iterative Solver With Gap Safe Rules
#####################################################################################

"""
One chooses Coordinate Descent as iterative solver. 
The idea of coordinate descent is to decompose a large optimisation problem into
a sequence of one-dimensional optimisation problems. 
Coordinate descent methods have become unavoidable in machine learning because
they are very efficient for key problems, namely Lasso, Logistic Regression and
Support Vector Machines. 
Moreover, the decomposition into small subproblems means that only a small part 
of the data is processed at each iteration and this makes coordinate descent
easily scalable to high dimensions. 
The idea of coordinate gradient descent is to perform one iteration of gradient
in the 1-dimensional problem 
min_{z \in X_{i}} f(x_{k}^{(1)}, \ldots, x_{k}^{(l-1)}, z, x_{k}^{(l+1)}, \ldots, x_{k}^{(n)})
instead of solving it completely. In general it reduces drastically the cost 
of each iteration while keeping the same convergence behaviour.
"""

def simu(coefs, n_samples=1000, corr=0.5, for_logreg=False):
    n_features=len(coefs)
    cov = toeplitz(corr ** np.array2string(0, n_features))

    # Features Matrix
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Target labels vector with noise
    y = X.dot(coefs) + randn(n_samples)

    if for_logreg:
        y = np.sign(y)

    return X, y

def cyclic_coordinate_descent(X, y, n_iter=10):
    """Solver : cyclic coordinate descent 

    Parameters
    ----------

    X: numpy.ndarray, shape (n_samples, n_features)
       features matrix

    y: numpy.array, shape (n_samples, )
       labels vector

    n_iter: int, default = 10
            number of iterations  

    Returns
    -------

    w: numpy.array, shape(n_features,)
       weights vector

    all_objs: numpy.array, shape(n_features)
            residuals vector

    """

    # Initialisation of the parameters

    n_samples, n_features = X.shape
    all_objs = []

    w = np.zeros(n_features)
    residuals = y - X.dot(w)

    # Computation of the lipschitz constants vector 

    lips_const = np.linalg.norm(X, axis=0)**2

    # Iterations of the algorithm
    for k in range(n_iter):

        # One cyclicly updates the i^{th} coordinate corresponding to the rest 
        # in the Euclidean division by the number of features 
        # This allows to always selecting an index between 1 and n_features
        i = k % n_features + 1

        old_w_i = w[i].copy()
        step = 1/lips_const[i]
        grad = (X[:,i].T).dot(residuals)

        # Update of the parameters
        w[i] += step*grad 

        # Update of the residuals
        residuals += np.dot(X[:,i], old_w_i - w[i])

        if k % n_features == 0:
            # If k % n_features == 0 then we have updated all the coordinates
            # One computes the objective function 
            residuals += np.dot((residuals**2).sum()/2)

    return w, np.array(all_objs)      


# Equation 11

def compute_theta_k(X, y, beta, lmbda):
  """Iterative computation of the dual optimal solution theta

  Parameters
  ----------

  X: numpy.ndarray, shape = (n_samples, n_features)
     features matrix

  y: numpy.array, shape = (n_features, )
     target labels vector

  lmbda: numpy.array, shape = (n_iter, )
         regularization parameters vector

  beta_k: numpy.array, shape = (n_features, )
          primal optimal parameters vector

  Returns
  -------

  theta_k: float 
           dual optimal parameters vector

  """

  # Initialization of the parameters 
  rho_k = y - X @ beta_k
  alpha_k = np.min(np.max((y @ rho_k)/(lmbda*np.linalg.norm(rho_k)**2), -1/np.linalg.nomr(X.T @ rho_k, ord='inf')), 1/np.linalg.norm(X.T @ rho_k, ord='inf'))
  theta_k = alpha_k * rho_k

  return theta_k
   
def radius(beta, theta, lmbda_t, lmbda_t_1, r_lmbda_t_1):
  """
  Parameters
  ----------

  beta: numpy.array, shape = (n_features, )
        primal optimal parameters vector

  theta: numpy.array, shape = (n_features, )
         dual optimal parameters vector

  lmbda_t: float
           regularization parameter at iteration t

  lmbda_t_1: float
             regularization parameter at iteration t-1
  
  r_lmbda_t_1: float
               radius related to the regularization parameter lmbda_t_1

  Returns
  -------
  r_lmbda_t: float
             radius related to the regularization parameter lmbda_t 
  """

  r_square_lmbda_t = (lmbda_t_1/lmbda_t)*r_lmbda_t_1**2 + (1 - lmbda_t/lmbda_t_1)*np.linalg.norm((X @ beta - y)/lmbda_t)**2 - (lmbda_t_1/lmbda_t - 1)*np.linalg.norm(theta)**2
  r_lmbda_t = np.sqrt(r_square_lmbda_t)

  return r_lmbda_t

# Equation 18 : Gap Safe Sphere

def gap_safe_sphere(theta_k, r_lmbda):
  """
  Parameters
  ----------

  theta_k: numpy.array, shape = (n_features,)
           dual optimal parameters vector
  
  r_lmbda: float
           radius related to the regularization parameter lmbda

  Returns
  -------

  C_k: interval
       sphere of center theta_k and of radius r_lmbda
  """

  inf_bound = theta_k - r_lmbda
  sup_bound = theta_k + r_lmbda
  C_k = interval[inf_bound, sup_bound]
  
  return C_k

def R_primal(lmbda, y, X, beta):
  """
  Parameters
  ----------

  lmbda: float
         regularization parameter
  
  y: numpy.array, shape=(n_samples, )

  X: numpy.ndarray, shape=(n_samples, n_features)
     features matrix 

  beta: numpy.array, shape=(n_features, )
        primal optimal parameters

  Returns
  -------

  R_hat_lmbda: float
               primal radius of the dome
  """

  R_hat_lmbda = (1/lmbda)*np.max(np.linalg.norm(y)**2 - np.linalg.norm(X @ beta - y)**2 - 2*lmbda*np.linalg.norm(beta, ord=1), 0)**(1/2)

  return R_hat_lmbda

def R_dual(y, theta, lmbda):
  """
  Parameters
  ----------

  lmbda: float
         regularization parameter
  
  y: numpy.array, shape=(n_samples, ) 

  theta: numpy.array, shape=(n_features, )
        dual optimal parameters

  Returns
  -------

  R_inv_hat_lmbda: float
                   dual radius of the dome
  """

  R_inv_hat_lmbda = np.linalg.norm(theta - y/lmbda)

  return R_inv_hat_lmbda

# Equation 19 : Gap Safe Dome
def gap_safe_dome(y, lmbda, theta_k, beta_k, R_hat_lmbda, R_inv_hat_lmbda):
  """
  Parameters
  ---------

  y: np.array, shape = (n_samples, )

  lmbda: float 
         regularization parameter
  
  theta_k: np.array, shape = (n_features, )
           dual optimal parameters vector

  beta_k: np.array, shape = (n_features, )
          primal optimal parameters vector
  
  R_hat_lmbda: float
               primal radius of the dome

  R_inv_hat_lmbda: float
                   dual radius of the dome
  

  Returns
  -------

  C_k: interval
       gape safe dome

  """

  c_k = ((y/lmbda) + theta_k)/2
  r_k = R_inv_hat_lmbda/2
  sphere_k = interval[c_k - r_k, c_k + r_k]

  alpha_k = 2*(R_hat_lmbda/R_inv_hat_lmbda)**2 - 1
  w_k = (theta_k - y/lmbda)/np.linalg.norm(theta_k - y/lmbda)

  margin_k = - alpha_k*r_k*w_k

  # c_k - alpah_k*r_k*w_k is the projection of the ball center c on the hyperplane
  # c = ball center
  # r = ball radius
  # oriented hyperplane with unit normal vector w and parameter alpha 
  # such that c - alpha*r*w is the projection of c on the hyperplane

  C_k = sphere_k.intersection(margin_k) # problème d'intersection avec l'hyperplane

  return C_k


def sigma_C(x, x_0):
  """Support Function

  Parameters
  ----------

  x: np.array, shape = (n_samples, )
     vector of samples for a given feature 

  x_0: np.array, shape = (n_samples, )
       initial values 

  Returns
  -------

  sigma: float
         maximum of the scalar product <x, theta> w.r.t theta
  """

  sigma = x @ theta 
  res = -minimize(-sigma, x_0, method=cd_with_gap_safe_rules, options={'xatol': 1e-8, 'disp': True})

  return res

def mu_C(x_j):
  """
  Parameters
  ----------
  x_j: np.array, shape = (n_samples, )
       feature x_j 

  Returns
  -------
  mu: float 
      maximum between two sigma_C(x_j) and sigma_C(-x_j)
  """

  mu = np.max(sigma_C(x_j), sigma_C(-x_j))

  return mu

# Equation 7 : Active Set
def active_set_vs_zero_set(X):
  """
  Parameters
  ----------
  X: numpy.ndarray, shape = (n_samples, n_features)

  Returns
  -------
  A_C: numpy.array, shape = (n_idx_active_features, )
       active set : contains the indices of the relevant features
  
  Z_C: numpy.array, shape = (n_idx_zero_features, )
       zero set : contains the indices of the irrelevant features
  """

  A_C = []
  Z_C = []
  p = X.shape[1]
  for j in range(p):
    mu = mu_C(X[:,j])
    if mu >= 1:
      A_C.append(j)
    else:
      Z_C.append(j)

  return A_C, Z_C

def sign(x):
  """
  Parameters
  ----------
  x: float
     
  Returns
  -------
  s: sign int
     (-1) if x < 0,, (+1) if x > 0, 0 if x = 0
  """

  if x > 0:
    s = 1
  elif x < 0:
    s = -1
  else:
    s = 0

  return s

def soft_thresholding(u,x):
  """

  Parameters
  ----------
  u: float
     threshold

  x: float

  Returns
  -------
  ST: float
      0 between -u and +u or slope of the straight line x - u otherwise 
      
  """

  ST = sign(x)*np.max(np.abs(x) - u, 0)


  return ST

# Equation 1 : Primal Problem 
def primal_pb(X, y, beta, lmbda):
  """
  Parameters
  ----------
  X: numpy.ndarray, shape = (n_samples, n_features)
     features matrix

  y: numpy.array, shape = (n_samples, )
     target labels vector

  beta: numpy.array, shape = (n_features, )
        initial vector of primal parameters

  lmbda: float
         regularization parameter

  Returns
  -------

  beta_hat_lmbda: numpy.array shape = (n_features, )
                  primal optimal parameters vector
  """

  P_lmbda = (1/2)*np.linalg.norm(X @ beta - y, ord=2)**2 + lmbda*np.linalg.norm(beta, ord=1)
  beta_hat_lmbda = minimize(P_lmbda, beta, method=cd_with_gap_safe_rules)

  return beta_hat_lmbda

def dual_pb(y, theta, lmbda):
  """
  Parameters
  ----------
  y: numpy.array, shape = (n_features, )

  theta: numpy.array, shape = (n_features, )
         initial vector of dual parameters

  lmbda: float
         regularization parameter

  Returns
  -------

  theta_hat_lmbda: numpy.array, shape = (n_features, )
                   dual optimal parameters vector

  """

  D_lmbda = (1/2)*np.linalg.norm(y, ord=2)**2 - ((lmbda**2)/2)*np.linalg.norm(theta - y/lmbda, ord=2)**2
  theta_hat_lmbda = -minimize(-D_lmbda, theta, method=cd_with_gap_safe_rules) # Probleme du feasible set
  return theta_hat_lmbda

def duality_gap(beta_hat_lmbda, theta_hat_lmbda):
  """
  Parameters
  ----------

  beta_hat_lmbda: numpy.array shape = (n_features, )
                  primal optimal parameters vector

  theta_hat_lmbda: numpy.array, shape = (n_features, )
                   dual optimal parameters vector

  Returns
  -------
  G_lmbda: float
           duality gap between the primal optimal and the dual optimal

  """

  G_lmbda = beta_hat_lmbda - theta_hat_lmbda

  return G_lmbda

def cd_with_gap_safe_rules(X, y, epsilon, K, f, lmbda, T, lmbda_max):
  """Coordinate descent with gap safe rules 

  Parameters
  ----------
  X: numpy.ndarray, shape = (n_samples, n_features)
     features matrix

  y: numpy.array, shape = (n_features, )
     target labels vector

  epsilon: float 
           accuracy

  T: int
     number of epochs

  K: int
     number of iterations

  f: int
     frequency

  lmbda: numpy.array, shape = (T-1, )
          regularization parameters vector

  Returns
  -------

  beta_lmbda_t: numpy.array, shape = (n_features, )
                primal optimal parameters vector 
                for the regularization parameter lmbda at iteration t

  """

  # Initialization 
  lmbda = []
  lmbda[0] = lmbda_max
  beta = dict()
  beta["lmbda0"] = 0

  for t in range(T-1):
    beta = beta["lambda" & (t-1)]

    for k in range(K):
      if k % f == 1:

        # Computation of theta 
        # Equation 11
        # Equation 18
        # Equation 19
        if G_lmbda(beta, theta) <= epsilon:
          beta_lmbda = beta
          break

        for j in A_C:
          beta_j = soft_thresholding(lmbda_t/np.linalg.norm(X[:,j])**2, beta_j - ((X[:,j].T @ (X @ beta - y))/np.linalg.norm(X[:,j])**2))

  return beta_lmbda_t


def main():
    n_features = 100
    np.random.seed(1970)
    coefs = np.random.randn(n_features)

    X, y = simu(coefs, n_samples=1000, for_logreg=False)

    w_min_cyclic_cd, obj_w_cyclic_cd = cyclic_coordinate_descent(X, y, 10000)

    


if __name__ == "__main__":
    main()        


