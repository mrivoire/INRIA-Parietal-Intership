import numpy as np

from scipy import sparse
from SPP import from_key_to_interactions_feature


def test_from_key_to_interactions_feature():

    rng = np.random.RandomState(42)
    X = sparse.random(10, 7, density=0.7, random_state=rng)
    X = X.tocsc()

    key = (1, 4, 5)

    inter_feat = X[:, key[0]]
    for k in key[1:]:
        inter_feat = inter_feat.multiply(X[:, k])

    n_samples, n_features = X.shape
    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=X.data, csc_ind=X.indices,
                                         csc_indptr=X.indptr, key=key,
                                         n_samples=n_samples,
                                         n_features=n_features)

    np.testing.assert_array_equal(interfeat_data, inter_feat.data)
    np.testing.assert_array_equal(interfeat_ind, inter_feat.indices)
