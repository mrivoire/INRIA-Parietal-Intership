import numpy as np

from scipy.linalg import toeplitz
from numba import njit
from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state
from cd_solver_lasso_numba import sparse_cd


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

#############################################################################
#                           Data Simulation
#############################################################################


def simu(beta, n_samples=1000, corr=0.5, for_logreg=False,
         random_state=None):
    n_features = len(beta)
    cov = toeplitz(corr ** np.arange(0, n_features))

    rng = check_random_state(random_state)

    # Features Matrix
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X = np.asfortranarray(X)

    # Target labels vector with noise
    y = np.dot(X, beta) + rng.randn(n_samples)

    if for_logreg:
        y = np.sign(y)

    return X, y


#############################################################################
#                         Maximum Inner Product
#############################################################################

@njit
def max_val(X_binned_data, X_binned_indices, X_binned_indptr, residuals,
            max_depth):
    """Compute the maximum inner product between X_binned.T and the residuals
       Return the feature and its id which allows to maximize the inner product

    Parameters
    ----------
    X_binned: numpy.ndarray shape = (n_samples, n_features * n_bins)
        discrete features matrix, sparse matrix

    ids_list: list of length n_features * n_bins
        contains the ids of the discrete features

    max_depth: int
        maximal depth of the tree
        maximal degree of interactions between the original features

    Returns
    -------
    max_feature: feature allowing to maximize the inner product between
        the sparse matrix of features X_binned.T and the residuals vector

    max_id: int
        id of the feature allowing to maximize the inner product
    """

    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    max_val = 0
    max_key = None
    depth = 1

    parent_data = np.ones(n_samples)
    parent_indices = np.arange(n_samples)

    for i in range(n_features):
        current_key = List([int(x) for x in range(0)])
        max_val, max_key = max_val_rec(X_binned_data=X_binned_data,
                                       X_binned_indices=X_binned_indices,
                                       X_binned_indptr=X_binned_indptr,
                                       parent_data=parent_data,
                                       parent_indices=parent_indices,
                                       current_max_val=max_val,
                                       current_max_key=max_key,
                                       j=i,
                                       current_key=current_key,
                                       residuals=residuals,
                                       max_depth=max_depth,
                                       depth=depth)

    # the recursion is used to scan the tree from the root to the leaves ?
    # stopping criterion = when the pruned tree contains no more node for
    # which the sppc criterion is satisfied ?

    # DFS
    # 1. Create a recursive function that takes the index of node and a visited
    # array
    # 2. Mark the current node as visited and print the node
    # 3. Traverse all the adjacent and unmarked nodes and call the recursive
    # function with index of adjacent node.

    return max_val, max_key


@njit
def compute_inner_prod(data1, ind1, residuals):
    """
    Parameters
    ----------
    data1: numpy.array(), shape = (n_non_zero_coeffs, )
        contains all the non-zero elements of the sparse verctor

    ind1: numpy.array(), shape = (n_non_zero_coeffs, )
        contains the indices of the rows in the dense matrix of the
        non-zero elements

    residuals: numpy.array(), shape = (n_samples, )
        contains the residuals between the predictions and the ground-truth
        values y

    Returns
    -------
    inner_prod: float
        inner product between the sparse vector data1 and the vector of
        residuals

    inner_prod_neg: float
        negative part of the inner product

    inner_prod_pos: float
        positive part of the inner product
    """

    inner_prod = 0
    inner_prod_pos = 0
    inner_prod_neg = 0

    # only for loop
    # not to scan residuals
    # only access to the elements corresponding to the indices stored in ind1
    prod = 0
    for count1 in range(len(ind1)):
        prod = data1[count1] * residuals[ind1[count1]]
        inner_prod += prod

        if residuals[ind1[count1]] >= 0:
            inner_prod_pos += prod
        else:
            inner_prod_neg += prod

    return inner_prod, inner_prod_neg, inner_prod_pos


@njit
def compute_interactions(data1, ind1, data2, ind2):
    """Compute interaction between 2 features

    Parameters
    ----------
    data1: numpy.array(), shape = (n_non_zero_coeffs, )
        contains all the non-zero elements of the 1st sparse verctor

    ind1: numpy.array(), shape = (n_non_zero_coeffs, )
        contains the indices of the rows in the dense matrix of the
        non-zero elements of the 1st sparse vector

    data2: numpy.array(), shape = (n_non_zero_coeffs, )
        contains all the non-zero elements of the 2nd sparse vector

    ind2: numpy.array(), shape = (n_non_zero_coeffs, )
        contains the indices of the rows in the dense matrix of the non-zero
        elements of the 2nd sparse vector

    Returns
    -------
    inter_feat_data: numpy.array(), shape = (n_non_zero_elements, )
        contains the non-zero elements of the sparse vector of interactions
        resulting in the product of the non-zero coefficients of same indices
        of the two parent vectors data1 and data2

    inter_feat_indices: numpy.array(), shape = (n_non_zero_elements, )
        contains the indices of the rows of the non-zero elements in the dense
        matrix
    """

    count1 = 0
    count2 = 0

    inter_feat_data = List([float(x) for x in range(0)])
    inter_feat_ind = List([int(x) for x in range(0)])

    # inter_feat_ind = tableau de booléens avec des 1 là où on veut
    # prendre les indices

    while count1 < len(ind1) and count2 < len(ind2):
        if ind1[count1] == ind2[count2]:
            prod = data1[count1] * data2[count2]
            inter_feat_ind.append(ind1[count1])
            inter_feat_data.append(prod)
            count1 += 1
            count2 += 1
        elif ind1[count1] < ind2[count2]:
            count1 += 1
        else:
            count2 += 1

    return inter_feat_data, inter_feat_ind


@njit
def max_val_rec(X_binned_data, X_binned_indices, X_binned_indptr,
                parent_data, parent_indices, current_max_val, current_max_key,
                j, current_key, residuals, max_depth, depth):
    """Compute the maximal inner product value between the given feature and
        the vector of residuals provided thanks to the pre-solve processing

    Parameters
    ----------
    X_binned: numpy.ndarray(), shape = (n_samples, n_features)
        binned features matrix

    parent: numpy.array(), shape = (n_samples, )
        vector of 0 and 1 corresponding to the discrete parent feature

    current_max_val: float
        current maximal inner product between the visited features
        and the residuals

    j (line 254): int
        index of the child node

    residuals: numpy.array(), shape = (n_samples, )
        vector of residuals

    max_depth: int
        maximum degree of interactions between the original features
        maximum depth in the tree

    depth: int
        current depth
        goes from 0 to max_depth if we start from the root to the leaves
        or from max_dpeth to 0 if we start from the leaves to the root

    Returns
    -------

    current_key: int
        index of the feature which provides the max_val

    current_max_val: float
        maximal inner product between the features and the vector of residuals

    """
    # res obtained by pre-solve
    # compare current_max_inner_prod to inner_product
    # if equation above algo in spp paper is satisfied then prune the subtree
    # otherwise visit the subtree

    # for a given node which is the root of a subtree, returns the max_val
    # if the max_val of the subtree is greater it returns the maxval of the
    # subtree otherwise it returns the previous maxval
    # for a given node t, compute the criterion with the
    # negative and positive parts

    # Compute the current feature
    # element wise product between parent and feature j to obtain the
    # interaction result feature x

    current_key.append(j)

    n_features = len(X_binned_indptr) - 1

    start, end = X_binned_indptr[j:j + 2]
    X_j_indices = X_binned_indices[start:end]
    X_j_data = X_binned_data[start:end]

    (inter_feat_data,
     inter_feat_ind) = compute_interactions(X_j_data, X_j_indices,
                                            parent_data, parent_indices)

    # print("ok inter feat")

    (inner_prod,
     inner_prod_neg,
     inner_prod_pos) = compute_inner_prod(inter_feat_data,
                                          inter_feat_ind,
                                          residuals)

    # print("ok inner_prod")

    upper_bound = max(inner_prod_pos, -inner_prod_neg)

    # print("ok upper bound")

    if upper_bound <= current_max_val:
        current_key.pop()
        # print("criterion non satisfied")
        return current_max_val, current_max_key

    else:
        # If the criterion is not satisfied:
        # Check the current node :
        # compute max_val
        # update max_val if required

        if abs(inner_prod) > current_max_val:
            # print("criterion satisfied")
            current_max_val = abs(inner_prod)
            current_max_key = current_key.copy()

        # If depth < max_depth:
            # for loop over the child nodes (number of children = n_features)
            # for ind = j+1 ... p if the parent node is j
            # maxval = max_val_rec(depth = depth + 1)

        if depth < max_depth:
            # start from 0 to max_depth
            # recursive call of the function on the following stage
            # print("criterion satisfied and depth < max depth")
            for k in range(j + 1, n_features):
                current_max_val, current_max_key = max_val_rec(
                    X_binned_data, X_binned_indices, X_binned_indptr,
                    inter_feat_data, inter_feat_ind, current_max_val,
                    current_max_key, k, current_key, residuals, max_depth,
                    depth + 1)

        # We keep the same parent node and we change the child nodes ?
        # How to find the key of the feature providing the maxval ?
        current_key.pop()
        return current_max_val, current_max_key


# #############################################################################
# #                           Safe Pattern Pruning
# #############################################################################

@njit
def safe_prune(X_binned_data, X_binned_indices, X_binned_indptr,
               safe_sphere_center, safe_sphere_radius, max_depth):
    """Update the safe_set with the active features

    Parameters
    ----------
    X_binned: numpy.ndarray(), shape = (n_samples, n_features)
        binned features matrix

    X_binned_data: numpy.array()
        contains the non zero elements of X_binned

    X_binned_indices: numpy.array()
        contains the indices of the rows of the non zero elemnts of X_binned

    X_binned_indptr: numpy.array()
        contains the indices of the first non zero elemnts of each column
        in X_binned

    safe_sphere_center: numpy.array, shape = (n_samples, )
        feasible theta, solution of the dual problem
        it is given by the maximum inner product between the features matrix
        X_binned and the vector of residuals

    safe_sphere_radius: float
        corresponds to the dual gap between the primal and the dual problems

    max_depth: int
        maximum degree of the interaction features

    Returns
    -------
    None
    """
    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    # safe_set_data (initialize with something for numpa)
    safe_set_data = List([List([0.])])
    safe_set_ind = List([List([0])])
    safe_set_keys = List([List([0])])

    depth = 1

    parent_data = np.ones(n_samples)
    parent_indices = np.arange(n_samples)
    current_key = List([int(x) for x in range(0)])
    # current_key = list()

    for i in range(n_features):
        safe_prune_rec(X_binned_data=X_binned_data,
                       X_binned_indices=X_binned_indices,
                       X_binned_indptr=X_binned_indptr,
                       parent_data=parent_data,
                       parent_indices=parent_indices,
                       current_key=current_key,
                       safe_set_data=safe_set_data,
                       safe_set_ind=safe_set_ind,
                       safe_set_key=safe_set_keys,
                       j=i,
                       safe_sphere_center=safe_sphere_center,
                       safe_sphere_radius=safe_sphere_radius,
                       max_depth=max_depth,
                       depth=depth)

    # drop with was inserted above for numba
    safe_set_data = safe_set_data[1:]
    safe_set_ind = safe_set_ind[1:]
    safe_set_keys = safe_set_keys[1:]

    return safe_set_data, safe_set_ind, safe_set_keys


@njit
def safe_prune_rec(X_binned_data, X_binned_indices, X_binned_indptr,
                   parent_data, parent_indices, current_key,
                   safe_set_data, safe_set_ind, safe_set_key, j,
                   safe_sphere_center, safe_sphere_radius,
                   max_depth, depth):
    """Recursively update the active set with the active features for each
    node in the tree.

    Parameters
    ----------
    X_binned: numpy.ndarray(), shape = (n_samples, n_features)
        binned features matrix

    X_binned_data: numpy.array()
        contains the non zero elements of X_binned

    X_binned_indices: numpy.array()
        contains the indices of the rows of the non zero elemnts of X_binned

    X_binned_indptr: numpy.array()
        contains the indices of the first non zero elemnts of each column
        in X_binned

    parent_data: numpy.array()
        feature in sparse format corresponding to the root node of the
        considered subtree
        contains the non zero elements of the feature in X_binned

    parent_indices: numpy.array()
        contains the indices of the rows of the non-zero elements of the
        root feature in X_binned

    parent_key: list
        contains the indices of the original features taking part into the
        interaction giving the root feature

    current_safe_set_data: list of list of data
        contains the numpy array of data of each active feature that
        have not been pruned out

    current_safe_set_ind: list of list of indices
        contains the numpy array of indices of each active feature that have
        not been pruned out

    curret_safe_set_key: list of list of keys
        contains the key of each active feature (each key is a list of
        indices of the original feature taking part into the given interaction)

    j: int
        index of the child feature (it is necessarily an original feature
        since we do not perform interactions between interaction features)

    safe_sphere_center: numpy.array, shape = (n_samples, )
        feasible theta, solution of the dual problem
        it is given by the maximum inner product between the features matrix
        X_binned and the vector of residuals

    safe_sphere_radius: float
        corresponds to the dual gap between the primal and the dual problems

    max_depth: int
        maximum degree of the interaction features

    depth: int
        current degree of the interaction features

    Returns
    -------
    None
    """
    # recursive function depth first search type
    # call safe prune rec on a feature i and
    # compute the interaction ij
    # apply the sppc criterion on the interaction feature ij

    # Test the SPPC criterion on the current feature
    # If the SPPC criterion is satisfied prune the whole subtree for which
    # the feature is the root otherwise test on the children

    n_features = len(X_binned_indptr) - 1

    # Compute the SPPC criterion
    # Compute u_t and v_t with feasible theta instead of residuals
    # Compute interactions between parent feature and j
    # parent = root
    # same structure than max_val
    # then call the rec function in safe prune in a for loop scanning all j
    # for j in range(n_features)
    # no return
    # max_val compute feasible theta since it computes the maximum inner
    # product between X and the residuals
    # compute the feasible theta in SPP function
    # make tests with nested for loops and compare the safe sets by making
    # an interaction function

    start, end = X_binned_indptr[j:j + 2]
    X_j_indices = X_binned_indices[start:end]
    X_j_data = X_binned_data[start:end]

    (inter_feat_data,
     inter_feat_ind) = compute_interactions(X_j_data, X_j_indices,
                                            parent_data, parent_indices)
    current_key.append(j)

    (inner_prod,
     inner_prod_neg,
     inner_prod_pos) = compute_inner_prod(inter_feat_data,
                                          inter_feat_ind,
                                          safe_sphere_center)

    u_t = max(inner_prod_pos, -inner_prod_neg)
    v_t = 0

    # for element in inter_feat_data
    for i in range(len(inter_feat_data)):
        v_t += inter_feat_data[i]**2

    sppc_t = u_t + safe_sphere_radius * np.sqrt(v_t)

    # create a safe_set_ind (list of lists of int),
    # safe_set_data (list of lists of values),
    # safe_set_key (list of lists of int)
    # append each list when we test the SPPC
    # test the feature itself before the recursive call

    # If SPPC is satisfied we can prune out the whole subtree from the root
    # node In this case the active set remains unchanged since all the
    # features of the subtree are pruned out

    # If SPPC is not satisfied then we recursively call the function
    # we do not prune the whole subtree, we save the root key in the
    # active set if the latter satisfied the sppc
    # and we recursively call the function on the child nodes

    if sppc_t >= 1:

        if abs(inner_prod) + safe_sphere_radius * np.sqrt(v_t) >= 1:
            safe_set_data.append(inter_feat_data)
            safe_set_ind.append(inter_feat_ind)
            key = current_key.copy()
            safe_set_key.append(key)

        if depth < max_depth:
            for k in range(j + 1, n_features):
                safe_prune_rec(X_binned_data, X_binned_indices,
                               X_binned_indptr,
                               inter_feat_data, inter_feat_ind, current_key,
                               safe_set_data, safe_set_ind,
                               safe_set_key, k, safe_sphere_center,
                               safe_sphere_radius, max_depth, depth + 1)

    current_key.pop()


@njit
def from_numbalists_tocsc(numbalist_data, numbalist_ind):
    """
    Parameters
    ----------
    numbalist_data: List([int])
        numba list of lists of integers containing the non-zero elements
        of each feature the sparse matrix

    numbalist_ind: list([int])
        nubma list of lists of integers containing the row indices of the
        non-zero elements of each feature of the sparse matrix

    Returns
    -------
    csc_data: list(int)
        list of integers containing the non-zero elements of the sparse matrix
        without separation between the features

    csc_ind: list(int)
        list of integers containing the row indices of the non-zero elements
        of the sparse matrix without separation between the features

    csc_indptr: list(int)
        list of integers containing the indices of each first non-zero element
        of each feature in the csc_data vector

    """
    csc_data = []
    csc_ind = []
    csc_indptr = [0]

    indptr = 0
    for feat in numbalist_data:
        indptr += len(feat)
        csc_indptr.append(indptr)
        for data in feat:
            csc_data.append(data)

    for ind_list in numbalist_ind:
        for ind in ind_list:
            csc_ind.append(ind)

    return csc_data, csc_ind, csc_indptr


@njit
def from_key_to_interactions_feature(csc_data, csc_ind, csc_indptr,
                                     key):
    """
    Parameters
    ----------
    csc_data: list(int)
        list of integers containing the non-zero elements of the sparse matrix

    csc_ind: list(int)
        list of integers containing the row indices of the non-zero elements
        of the sparse matrix

    csc_indptr: list(int)
        list of integers containing the indices of each first non-zero elements
        of each feature in the csc_data vector

    key: list(int)
        list of integers containing the indices of the binned features taking
        part in the interaction

    Returns
    -------
    interfeat_data: list(int)
        list of integers containing the non-zero elements of the
        feature of interactions

    interfeat_ind: list(int)
        list of integers containing the row indices of the non-zero elements
        of the feature of interactions

    """
    start, end = csc_indptr[key[0]: key[0] + 2]
    interfeat_data = List(csc_data[start: end])
    interfeat_ind = List(csc_ind[start: end].astype(np.int64))

    for idx in key[1:]:
        start, end = csc_indptr[idx: idx + 2]
        data2 = csc_data[start: end]
        ind2 = csc_ind[start: end]
        interfeat_data, interfeat_ind = \
            compute_interactions(data1=interfeat_data,
                                 ind1=interfeat_ind,
                                 data2=data2,
                                 ind2=ind2)

    return list(interfeat_data), list(interfeat_ind)


# @njit
def spp_solver(X_binned, y,
               n_lambda, max_depth, epsilon, f, n_epochs, tol, 
               lambda_max_ratio, n_active_max, screening=True, 
               store_history=True):
               # n_lmbda = n_lambda, lmbda_max_ratio, n_active_max
    """Safe Patterns Pruning Algorithm
       Scan the tree from the root to the leaves and prunes out the subtrees
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
    # compute lambda_max with max_val with beta = 0 and decrease it with a
    # logarithmic step
    # on part de lmbda_max et on décroit d'un pas logarithmic
    # en entrée remplacer la lmbdas_grid par le nombre de lmbdas à tester

    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    n_features = len(X_binned_indptr) - 1

    lambda_max, max_key = max_val(X_binned_data=X_binned_data,
                                  X_binned_indices=X_binned_indices,
                                  X_binned_indptr=X_binned_indptr,
                                  residuals=y, max_depth=max_depth)
    # If we compute once again the feature of interactions corresponding to
    # max_key

    max_feat_data, max_feat_ind = \
        from_key_to_interactions_feature(csc_data=X_binned_data,
                                         csc_ind=X_binned_indices,
                                         csc_indptr=X_binned_indptr,
                                         key=max_key)

    beta_hat_t = np.zeros(n_features)

    # lmbdas_grid = np.logspace(start=0, stop=lambda_max, num=n_lambda,
    #                           endpoint=True, base=10.0, dtype=None, axis=0)

    # test on only one value of lambda which is lower than lambda_max
    # if lambda is greater than lambda max then all the nodes of the features
    # tree are pruned out and the active set is empty.
    lmbdas_grid = ((lambda_max * lambda_max_ratio) 
                   * np.logspace(start=0, stop=-2, num=n_lambda))
    # add an argument lmbda_0_ratio (= 1/2 for istance) to always keep lmbda_max
    # x np.logspace()

    # Create a grid starting from lmbda_max to lmbda_max / 100
    # lmbda max has not to be too small otherwise we obtain too many features
    # find a bettter initialization
    # active_set = List([List([0])])

    # Initialization of the active set
    # (with the feature corresponding to max_key)
    # Two possibilities : either modify the max_val function so that it
    # returns the feature of interactions corresponding to max_key
    # or compute again the feature of interactions corresponding to max_key
    # For this purpose : slice the X_binned_data vector to extract the binned
    # features corresponding to each ind in the key, then compute the
    # feature of interactions by making the inner product between the binned
    # features or by calling the function which aims at computing the
    # features of interactions
    active_set_data_csc = []
    active_set_data_csc.extend(max_feat_data)
    active_set_ind_csc = []
    active_set_ind_csc.extend(max_feat_ind)
    active_set_indptr_csc = []
    active_set_indptr_csc.append(0)
    active_set_indptr_csc.append(len(active_set_ind_csc))  # NNZ element
    active_set_keys = []
    active_set_keys.append(max_key)

    solutions = []

    for lmbda_t in lmbdas_grid:

        print('lmbda_t = ', lmbda_t)
        # Pre-solve : solve the optimization problem with the new lambda on
        # the previous optimal set of features. (epsilon not too small ~ 10^-8)
        # use the implemented lasso with as input only the previous
        # optimal subset on features
        # we obtain a beta_hat which is not the optimal beta but which is a
        # better optimization of beta since it is closer to the optimum then
        # the screening is better from the beginning

        # instead of using the class Lasso, use the sparse_cd function to solve
        # the Lasso optimization problem with an input under
        # the csc attribut format
        # sparse_lasso = Lasso(lmbda=lmbda_t, epsilon=epsilon, f=f,
        #                      n_epochs=n_epochs,
        #                      screening=False,
        #                      store_history=False).fit(X_active_set, y)

        (beta_hat_t, residuals, primal_hist_sparse, dual_hist_sparse,
         gap_hist_sparse, r_list_sparse, n_active_features_true_sparse,
         theta_hat_cyclic_cd_sparse, P_lmbda_sparse, D_lmbda_sparse,
         G_lmbda_sparse, safe_set_sparse) = \
            sparse_cd(X_data=active_set_data_csc,
                      X_indices=active_set_ind_csc,
                      X_indptr=active_set_indptr_csc, y=y, lmbda=lmbda_t,
                      epsilon=epsilon, f=f, n_epochs=n_epochs,
                      screening=screening, store_history=store_history)

        # ajouter un point fit sur les features actives données par
        # safe_set_membership
        # Pas besoin d'avoir un epsilon très grand ni de screening
        # initialiser avant la boucle for l'ensemble des features actives à nul
        # le passer en paramètre du lasso
        # ensuite après l'appel de safe prune updater l'ensemble des features
        # actives grâce au safeset membership

        # between pre-solve and safe prune
        # compute the dual gap (compute the primal and the dual)
        # compute of a feasible dual solution
        # rescaling of the solution to make it feasible
        # compute feasible theta with the max_val
        # compute the radius of the safe sphere

        max_inner_prod, max_key = \
            max_val(X_binned_data=X_binned_data,
                    X_binned_indices=X_binned_indices,
                    X_binned_indptr=X_binned_indptr,
                    residuals=residuals,
                    max_depth=max_depth)

        theta = residuals / max(max_inner_prod, lmbda_t)
        P_lmbda = 0.5 * residuals.dot(residuals)
        P_lmbda += lmbda_t * np.linalg.norm(beta_hat_t, 1)

        D_lmbda = 0.5 * np.linalg.norm(y, ord=2) ** 2
        D_lmbda -= (((lmbda_t ** 2) / 2)
                    * np.linalg.norm(theta - y / lmbda_t, ord=2) ** 2)

        G_lmbda = P_lmbda - D_lmbda
        safe_sphere_radius = np.sqrt(2 * G_lmbda) / lmbda_t
        safe_sphere_center = theta

        solutions_dict = {}
        solutions_dict['lambda'] = lmbda_t

        if abs(G_lmbda) < tol:
            print('The current active set has already reached the support' +
                  ' of the optimal model')

            solutions_dict['data'] = active_set_data_csc
            solutions_dict['ind'] = active_set_ind_csc
            solutions_dict['indptr'] = active_set_indptr_csc
            solutions_dict['keys'] = active_set_keys
            solutions_dict['spp_lasso_slopes'] = beta_hat_t

        else:
            # Safe Prune after pre-solve:
            # epoch == 0 => first perform spp then launch the solver with
            # screening
            # then SPP: we obtain a safe set
            # launch the solver taking as input the safe set
            # the solver will screen even more the features in the safe set
            # safe set vector which contains all the nodes which have not
            # been
            # screened
            # safe set contains both the sparse features and an id
            # (corresponding to the ancestors of the features)
            # representation of the id of the features
            # 2 vectors of same size : 1 with the sparse features
            # (vector of vectors = sparse matrix) and
            # 1 with the id (vector of tuples)
            # can be directly given as input to the solver
            # then remove the ids outside from the solver with regards to the
            # 0 coeffs of beta
            # or implement a class "node" with attribute key (id = tuple)
            # and the sparse
            # vector which represents the feature
            # then make a vector of node

            # launch the already implemented solver on the safe set

            (safe_set_data, safe_set_ind, safe_set_key) = safe_prune(
                X_binned_data=X_binned_data, X_binned_indices=X_binned_indices,
                X_binned_indptr=X_binned_indptr,
                safe_sphere_center=safe_sphere_center,
                safe_sphere_radius=safe_sphere_radius, max_depth=max_depth)

            # Convert safe_set_data, safe_set_ind and safe_set_key which are
            # list  of lists numba into csc attributs

            # To convert safe_set_data which is a numba list of lists we just
            # have to flatten the numba list as follows

            safe_set_data_csc, safe_set_ind_csc, safe_set_indptr_csc = \
                from_numbalists_tocsc(safe_set_data, safe_set_ind)

            # Les safe sets retournent des listes de listes numba
            # convertir les listes de listes numba en une matrice sparse
            # pour pouvoir les donner en entrée au sparse_cd
            # la matrice sparse correspond à une matrice csc
            # ensuite construire les 3 attributs data, ind et indptr
            # ou déterminer directement data, ind et indptr
            # essayer de retourner directement data, ind et indptr dans
            # safe_prune

            # ou réécrire sparse_cd avec des listes de listes numba
            # list_data, list_ind, list_indptr

            (beta_hat_t, residuals, primal_hist, dual_hist, gap_hist, r_list,
             n_active_features, theta, P_lmbda, D_lmbda, G_lmbda,
             safeset_membership) = sparse_cd(
                X_data=safe_set_data_csc, X_indices=safe_set_ind_csc,
                X_indptr=safe_set_indptr_csc, y=y, lmbda=lmbda_t,
                epsilon=epsilon, f=f, n_epochs=n_epochs, screening=screening,
                store_history=store_history)

            print('beta_hat_t: ', beta_hat_t)
            print('sum of safeset_membership: ', sum(safeset_membership))

            active_set_data = List([List([0.])])
            active_set_ind = List([List([0])])
            active_set_keys = List([List([0])])
            beta_hat_t_sparse = []

            for idx, beta in enumerate(beta_hat_t):
                if beta != 0:
                    active_set_data.append(safe_set_data[idx])
                    active_set_ind.append(safe_set_ind[idx])
                    active_set_keys.append(safe_set_key[idx])
                    beta_hat_t_sparse.append(beta_hat_t[idx])

            active_set_data = active_set_data[1:]
            active_set_ind = active_set_ind[1:]
            active_set_keys = active_set_keys[1:]

            active_set_data_csc, active_set_ind_csc, active_set_indptr_csc = \
                from_numbalists_tocsc(numbalist_data=active_set_data,
                                      numbalist_ind=active_set_ind)

            solutions_dict['data'] = active_set_data_csc
            solutions_dict['ind'] = active_set_ind_csc
            solutions_dict['indptr'] = active_set_indptr_csc
            solutions_dict['keys'] = active_set_keys
            solutions_dict['spp_lasso_slopes'] = beta_hat_t_sparse

            if len(active_set_indptr_csc) + 1 >= n_active_max:
                break

        solutions.append(solutions_dict)

    return solutions

# Faire un objet "solution" avec en attribut lambda, beta, active_set_data,
# active_set_ind, active_set_indptr, active_set_keys
# retourner une liste d'objets solution (un pour chaque lambda)
# retourner un dictionnaire de dictionnaires avec comme clés : beta, data, ind,
# indptr, keys
# Exemple tirer en 2 dimensions aléatoirement des points entre 0 et 1 et
# ensuite classifier entre noir (y = 1) et blanc (y = 0)
# écrire un estimateur spp à la scikit learn avec une fonction predict
# (sous forme de classe comme pour le Lasso)
# store dans les attributs de la classe les keys, data, ind, indptr et qui est
# capable de les retourner écrire les tests associés
# puis faire tourner les tests sur les datasets
# commencer par housing_prices (car plus petit), stopper max_depth à 3, 4
# lire le papier de la review d'Alex Safe rule fit
# (pour la littérature de nos travaux)
# feedback sur le paper
# (qu'est-ce qui est similaire et qu'est ce qui est différent par rapport à ce
# que je fais)


class SPPRegressor():
    def __init__(self, n_lambda, max_depth, epsilon, f, n_epochs, tol, 
                 lambda_max_ratio, n_active_max, screening, store_history):

        # self.lmbda = lmbda
        self.n_lambda = n_lambda
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.f = f
        self.n_epochs = n_epochs
        self.tol = tol
        self.lambda_max_ratio = lambda_max_ratio
        self.n_active_max = n_active_max
        self.screening = screening
        self.store_history = store_history

        assert epsilon > 0
        assert tol > 0 and tol < 1e-01

    def fit(self, X_binned, y):
        """Fit the data (X, y) based on the solver of the SPP class
        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        Returns
        -------
        y: numpy.array, shape = (n_samples, )
            target vector
        """
        solutions = spp_solver(X_binned.tocsc(), y=y,
                               n_lambda=self.n_lambda,
                               max_depth=self.max_depth,
                               epsilon=self.epsilon,
                               f=self.f, n_epochs=self.n_epochs,
                               tol=self.tol,
                               lambda_max_ratio=self.lambda_max_ratio,
                               n_active_max=self.n_active_max,
                               screening=self.screening,
                               store_history=self.store_history)

        self.spp_solutions = solutions
        # remove this part to keep for all lmbdas
        # self.spp_lasso_slopes = solutions[0]['spp_lasso_slopes']
        # self.activeset_keys = solutions[0]['keys']
        # self.activeset_data = solutions[0]['data']
        # self.activeset_ind = solutions[0]['ind']
        # self.activeset_indptr = solutions[0]['indptr']

        return self

    def predict(self, X_binned):
        """Predict the target from the observations matrix

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        Returns
        -------
        y_hat: numpy.array, shape = (n_samples, )
            predicted target vector
        """

        # itérer sur les solutions et calculer le y_hat
        # renvoyer une liste de y_hat pour chaque lmbda
        X_binned = X_binned.tocsc()
        X_binned_data = X_binned.data
        X_binned_ind = X_binned.indices
        X_binned_indptr = X_binned.indptr

        n_samples = X_binned.shape[0]
        print('dim solutions = ', len(self.spp_solutions))
        # y_hats = np.zeros((n_samples, len(self.spp_solutions)))
        y_hats = [0] * len(self.spp_solutions)

        for i in range(len(self.spp_solutions)):
            y_hat = np.zeros(n_samples)
            interfeats = []
            for key, slope in zip(self.spp_solutions[i]['keys'],
                                  self.spp_solutions[i]['spp_lasso_slopes']):

                interfeat_data, interfeat_ind = \
                    from_key_to_interactions_feature(csc_data=X_binned_data,
                                                     csc_ind=X_binned_ind,
                                                     csc_indptr=X_binned_indptr,
                                                     key=key)

                interfeats.append(interfeat_data)

                y_hat[interfeat_ind] += slope * np.array(interfeat_data)

            y_hats[i] = y_hat

        y_hats = np.array(y_hats)
        return y_hats

    def score(self, X, y):
        """Compute the cross-validation score to assess the performance of the
           model (use R-square)

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        y: numpy.array, shape = (n_samples, )
            target vector

        Returns
        -------
        score: float
            chosen metric : R-sqare
        """

        y_hats = self.predict(X)
        cv_scores = []
        for y_hat in y_hats:
            u = ((y - y_hat) ** 2).sum()
            v = ((y - np.mean(y)) ** 2).sum()
            score = 1 - u / v
            cv_scores.append(score)

        return cv_scores


def main():
    rng = check_random_state(2)
    n_samples, n_features = 100, 10
    beta = rng.randn(n_features)
    lmbda = 1.
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True
    encode = 'onehot'
    strategy = 'quantile'
    n_bins = 3
    max_depth = 2
    n_lambda = 100
    tol = 1e-08
    lambda_max_ratio = 0.5
    n_active_max = 100

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    print("X_binned = ", X_binned)
    X_binned_data = X_binned.data
    print("X_binned_data = ", X_binned_data)
    X_binned_indices = X_binned.indices
    print("X_binned_indices = ", X_binned_indices)
    X_binned_indptr = X_binned.indptr
    print("X_binned_indptr = ", X_binned_indptr)

    ################################################################
    #                           Lasso
    ################################################################

    lasso_sklearn = Lasso(alpha=(lmbda / X_binned.shape[0]),
                          fit_intercept=False,
                          normalize=False,
                          max_iter=n_epochs,
                          tol=1e-14).fit(X_binned, y)

    beta_star = lasso_sklearn.coef_
    residuals = y - X_binned.dot(beta_star)
    XTR_absmax = 0
    for j in range(n_features - 1):
        start, end = X_binned_indptr[j:j + 2]
        X_j_indices = X_binned_indices[start:end]
        X_j_data = X_binned_data[start:end]

        (inner_prod,
         inner_prod_pos,
         inner_prod_neg) = compute_inner_prod(X_j_data,
                                              X_j_indices,
                                              residuals)

        XTR_absmax = max(abs(inner_prod), XTR_absmax)

    XTR_absmax = 0

    for j in range(n_features):
        XTR_absmax = max(abs(X_binned[:, j].T.dot(residuals)), XTR_absmax)

    theta = residuals / max(XTR_absmax, lmbda)

    P_lmbda = 0.5 * residuals.dot(residuals)
    P_lmbda += lmbda * np.linalg.norm(beta_star, 1)

    D_lmbda = 0.5 * np.linalg.norm(y, ord=2) ** 2
    D_lmbda -= (((lmbda ** 2) / 2) * np.linalg.norm(theta - y / lmbda, ord=2)
                ** 2)

    # Computation of the dual gap
    G_lmbda = P_lmbda - D_lmbda
    safe_sphere_radius = np.sqrt(2 * G_lmbda) / lmbda
    safe_sphere_radius = 1
    safe_sphere_center = theta

    ##########################################################
    #       Test for compute interactions function
    ##########################################################

    # ind1 = List([0, 1, 2])
    # ind2 = List([0, 1, 2, 3])
    # data1 = List([1, 5, 3])
    # data2 = List([4, 7, 9, 8])

    # inter_feat_data, inter_feat_ind = compute_interactions(data1,
    #                                                        ind1,
    #                                                        data2,
    #                                                        ind2)

    # print("inter feat data = ", inter_feat_data)
    # print("inter feat ind = ", inter_feat_ind)

    ##############################################################
    #             Test for compute inner product
    ##############################################################

    # (inner_prod,
    #  inner_prod_neg,
    #  inner_prod_pos) = compute_inner_prod(data1, ind1, residuals)

    #################################################################
    #                   Test for SPP function
    #################################################################
    # solutions = \
    #     spp_solver(X_binned, y=y, n_lambda=n_lambda,
    #                max_depth=max_depth, epsilon=epsilon, f=f,
    #                n_epochs=n_epochs, tol=tol, 
    #                lambda_max_ratio=lambda_max_ratio,
    #                n_active_max=n_active_max, screening=screening,
    #                store_history=store_history)

    # print('solutions = ', solutions)
    # print('length solutions = ', len(solutions))
    # print('lmbda = ', solutions[0]['lambda'])
    # print('data = ', solutions[0]['data'])
    # print('ind = ', solutions[0]['ind'])
    # print('indptr = ', solutions[0]['indptr'])
    # print('keys = ', solutions[0]['keys'])
    # print('slopes = ', solutions[0]['spp_lasso_slopes'])

    #################################################################
    #                     Test Class SPP Solver
    #################################################################

    # lmbda_max, max_key = max_val(X_binned_data=X_binned_data,
    #                              X_binned_indices=X_binned_indices,
    #                              X_binned_indptr=X_binned_indptr,
    #                              residuals=residuals, max_depth=max_depth)

    # Binning process
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode,
                           strategy=strategy)
    X_binned = enc.fit_transform(X)

    lmbda = 0.2481874128375465
    spp_reg = SPPRegressor(n_lambda=n_lambda,
                           max_depth=max_depth,
                           epsilon=epsilon, f=f, n_epochs=n_epochs, tol=tol, 
                           lambda_max_ratio=lambda_max_ratio, 
                           n_active_max=n_active_max,
                           screening=screening, store_history=store_history)

    solver = spp_reg.fit(X_binned, y)
    y_hats = spp_reg.predict(X_binned)
    print('y_hats = ', y_hats)
    cv_scores = spp_reg.score(X_binned, y)
    print('cv_scores = ', cv_scores)

if __name__ == "__main__":
    main()
