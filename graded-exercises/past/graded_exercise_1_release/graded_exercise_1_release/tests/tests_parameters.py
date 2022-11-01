""" Parameters for generating unit tests data.
"""

test_params = {
    'normalization': dict(N=100, dims=3, mn=-3., mx=5.),
    'init_centers': dict(),
    'compute_distance': dict(),
    'find_closest_cluster': dict(),
    'knnweighted': dict(),
    'influential_features': dict(dims=30, mn=-20., mx=20.),
    'correlated_features': dict(dims=30, mn=-20., mx=20.),
    'num_folds_LOOCV': dict()
}
