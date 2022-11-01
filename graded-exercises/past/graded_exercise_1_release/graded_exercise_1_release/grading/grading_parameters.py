""" Parameters for generating grading data. This file must be hidden from
the students.
"""

grading_params = {
    'normalization': dict(N=150, dims=5, mn=-5., mx=4.),
    'init_centers': dict(),
    'compute_distance': dict(),
    'find_closest_cluster': dict(),
    'knnweighted': dict(q_te=558, q_grad=259),
    'influential_features': dict(dims=20, mn=-10., mx=10.),
    'correlated_features': dict(dims=20, mn=-10., mx=10.)
}
