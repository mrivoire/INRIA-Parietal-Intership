#######################################################################
#                   Safe Pattern Pruning Algorithm
#######################################################################


"""
1. Implement a function allowing to give the required patterns under the
proper format to the SPP Algorithm. In this step, perform a binning process
over the continuous variables so as to obtain discrete one-hot encoded features
from which we can create interactions variables.

2. Implement a function which from a given subtree is able to compute the
value of lambda max.

3. Implement the SPP algorithm which scans the tree built over the whole set of
patterns from the root to the leaf and for each node t that tests the SPPC(t).
If the SPPC(t) is satisfied it prunes the whole subtree, otherwise it leaves
it. For the scan of the tree from the root to the leaves, look at depth at
first search algorithm.
At the end of the pruning process, we form the active set with all the patterns
which remain in the tree i.e. which have not been pruned out.

4. Implement a solver allowing to run an optimization process over the active
set A hat. (sparse_cd)

5. Implement a grid search over the lambdas which runs SPP algorithms for
each value of lambda of the grid. The values of lambda in the grid search start
from lambda max and evenly decrease until a very small lambda to obtain a
solution always more dense.
For each value of lambda in the grid, run a SPP algorithm in order
to find the active set A hat. Once A hat have been determined, run the
optimization process thanks to the implemented Lasso solver.

6. At each iteration of the grid search, we obtain a couple of primal dual
solutions, store for each lambda these solutions and return the whole set of
primal / dual solutions.
"""


def features_preprocess(X):
    """Build the tree based on the patterns of the database

    Parameters
    ----------
    X: numpy.ndarray shape = (n_samples, n_features)
        original features matrix

    Returns
    -------
    X_binned: np.ndarray shape = (n_samples, n_features x n_bins)
        binned features matrix

    tau: tree built over the patterns of the database
    """

    return X_binned, tau


def SPPC(t):
    """Safe Patterns Pruning Criterion at node t

    Parameters
    ----------
    t: node t i.e. pattern at node t

    Returns
    -------
    sppc_t: float
        safe patterns pruning criterion at node t
    """
    return sppc_t


def SPP(tau):
    """Safe Patterns Pruning Algorithm
       Scan the tree from the root to the leaves and prunes out the subtree
       which statisfie the SPPC(t) criterion

    Parameters
    ----------
    tau: tree built over the whole set of patterns of the database

    Returns
    -------

    A_hat: numpy.array, shape = (nb active features, )
        contains the active features taking part to the optimal
        predictive model
    """

    return A_hat


def lambda_max(X, y, tau):
    """Compute lambda_max which is first lambda in the grid search process

    Parameters
    ----------
    X: numpy.ndarray shape = (n_samples, n_features)
        original features matrix

    y: numpy.array shape = (n_samples, )
        target variable

    tau: tree over which we want to run the SPP process

    Returns
    -------
    lmbda_max: float
        highest regularization value in the grid search
    """

    

    return lmbda_max


def grid_search(lmbdas_set, step, K, X, y, tau):
    """Run for each value of lambda a SPP process and an optimization step

    Parameters
    ----------
    lmbdas_set: numpy.array shape = (nb lmbdas, )
        set of lambda values in the grid search
        the grid search starts from lmbda_0 = lmbda_max and evenly decreases
        until a certain small value of lambda to have solutions always more 
        dense

    step: float 
        space between two consecutive lambdas

    K: int
        number of lambdas in the grid search
        (make a for loop over the number of lambdas values in the grid search)

    X: numpy.ndarray shape = (n_samples, n_features)
        original features matrix

    y: numpy.array shape = (n_samples, )
        target variable

    tau: tree built over the whole set of discrete patterns in the database

    Returns
    -------
    primal_dual_optimal_set: numpy.array shape = (nb lmbdas, )
        set of primal dual solutions for each value of lambda in the grid 
        search
    """

    return primal_dual_optimal_set
