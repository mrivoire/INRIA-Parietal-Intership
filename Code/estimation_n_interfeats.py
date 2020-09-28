from scipy.special import binom
import numpy as np


def estimate_n_interfeats(n_features, n_bins, max_depth):
    n_binned_feats = n_features * n_bins

    n_interfeats = 0
    for k in range(max_depth):
        n_interfeats = binom(n_binned_feats, k)

    return n_interfeats


def main():
    n_features = 80
    n_bins = 3
    max_depth = 3

    n_interfeats = estimate_n_interfeats(
        n_features=n_features, n_bins=n_bins, max_depth=max_depth)

    print('n_interfeats = ', n_interfeats)


if __name__ == "__main__":
    main()
