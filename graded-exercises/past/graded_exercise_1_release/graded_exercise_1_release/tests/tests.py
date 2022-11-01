""" Unit tests for Graded Exercise #1.
"""

# 3rd party.
import numpy as np

# Project files.
import tests.helpers as helpers
from helpers.helper import KNNHelper
from matplotlib import pyplot as plt

def test(test_function):
    """ Decorator used for each test case. Pretty-prints assert exceptions
    (i.e. those asserts which are used within separate tests) and raises
    unhandeled exceptions (i.e. those which might appear due to the bugs
    on students' side).
    """
    def process_exceptions(*args, **kwargs):
        exception_caught = None
        function_result = None

        try:
            function_result = test_function(*args, **kwargs)
        except AssertionError as e:
            exception_caught = e
        except Exception as other_exception:
            raise other_exception

        if exception_caught is None:
            print("[{}] No problems detected. Your code is correct! "
                  "\U0001f60e".format(test_function.__name__))
        else:
            print("[{}] - {} \U0001f635".format(test_function.__name__,
                                                exception_caught))
        return function_result
    return process_exceptions


################################################################################
# TESTS
################################################################################

@test
def test_normalization(scope):
    exercise_id = 'normalization'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)

    X_te = test_data['X_te']
    Xn_te = test_data['Xn_te']
    min_te = test_data['min_te']
    max_te = test_data['max_te']
    X_gr = test_data['X_gr']

    ### Test `find_min_max_values`.
    find_min_max_values = helpers.resolve('find_min_max_values', scope)

    # Apply student's `find_min_max_values` and register the results.
    stud_minmax_te = find_min_max_values(X_te)
    stud_minmax_gr = find_min_max_values(X_gr)
    helpers.register_answer('find_min_max_values', stud_minmax_gr, scope)

    # [TEST] `find_min_max_values` returns a tuple/list.
    fail_msg = 'Function "find_min_max_values" should return a tuple of 2 ' \
               'scalars, whereas your function returns type {}'.\
        format(type(stud_minmax_te))
    assert isinstance(stud_minmax_te, (tuple, list)), fail_msg

    # [TEST] `find_min_max_values` returns a tuple/list of two values.
    fail_msg = 'Function "find_min_max_values" should return a tuple of 2 ' \
               'values, whereas your function returns a tuple of {} values'.\
        format(len(stud_minmax_te))
    assert len(stud_minmax_te) == 2, fail_msg

    # [TEST] `find_min_max_values` returns correct min and max.
    fail_msg = '"find_min_max_values" returns incorrect values. Expected {}, ' \
               'received {}'.format((min_te, max_te), tuple(stud_minmax_te))
    assert np.allclose(np.array(stud_minmax_te),
                       np.array((min_te, max_te))), fail_msg

    ### Test `min_max_normalization`.
    min_max_normalization = helpers.resolve('min_max_normalization', scope)

    # Apply student's `min_max_normalization` and register the results.
    stud_norm_te = min_max_normalization(X_te, *stud_minmax_te)
    stud_norm_gr = min_max_normalization(X_gr, *stud_minmax_gr)
    helpers.register_answer('min_max_normalization', stud_norm_gr, scope)

    # [TEST] `min_max_normalization` returns correct data type, shape, values.
    helpers.compare_np_arrays(stud_norm_te, Xn_te, varname='normalized_data')


@test
def test_influential_features(scope):
    exercise_id = 'influential_features'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)

    w_test = test_data['w_test']
    least_inf_te = test_data['least_inf_te']
    most_inf_te = test_data['most_inf_te']
    w_grade = test_data['w_grade']

    ### Test `find_min_max_values`.
    func_find_least_and_most_influential_features = helpers.resolve('find_least_and_most_influential_features', scope)

    # Apply student's `find_least_and_most_influential_features` and register the results.
    stud_influential_feat_te = func_find_least_and_most_influential_features(w_test)
    stud_influential_feat_gr = func_find_least_and_most_influential_features(w_grade)
    helpers.register_answer('find_least_and_most_influential_features', stud_influential_feat_gr, scope)

    # [TEST] `find_least_and_most_influential_features` returns a tuple/list.
    fail_msg = 'Function "find_least_and_most_influential_features" should return a tuple of 2 ' \
               'scalars, whereas your function returns type {}'.\
        format(type(stud_influential_feat_te))
    assert isinstance(stud_influential_feat_te, (tuple, list)), fail_msg

    # [TEST] `find_least_and_most_influential_features` returns a tuple/list of two values.
    fail_msg = 'Function "find_least_and_most_influential_features" should return a tuple of 2 ' \
               'values, whereas your function returns a tuple of {} values'.\
        format(len(stud_influential_feat_te))
    assert len(stud_influential_feat_te) == 2, fail_msg

    # [TEST] `find_least_and_most_influential_features` 
    fail_msg = 'Function "find_least_and_most_influential_features" returns incorrect values.' \
                'Your function returns the bias term index as either the least or most influential feature.'
    assert (stud_influential_feat_te[0]!=0) and (stud_influential_feat_te[1]!=0), fail_msg


    # [TEST] `find_least_and_most_influential_features` 
    fail_msg = 'Function "find_least_and_most_influential_features" returns incorrect values.' \
                'You might not be taking into account that bias term is index 0.'
    assert (stud_influential_feat_te[0]+1!=least_inf_te) and (stud_influential_feat_te[1]+1!=most_inf_te), fail_msg


    # [TEST] `find_least_and_most_influential_features` returns incorrect values.
    fail_msg = 'Function "find_least_and_most_influential_features" returns incorrect values.'
    assert np.allclose(np.array(stud_influential_feat_te), np.array((least_inf_te, most_inf_te))), fail_msg

@test
def test_correlated_features(scope):
    exercise_id = 'correlated_features'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)
    w_test = test_data['w_test']
    neg_corr_te = test_data['neg_corr_te']
    pos_corr_te = test_data['pos_corr_te']
    w_grade = test_data['w_grade']

    ### Test `find_min_max_values`.
    func_find_positively_correlated_negatively_correlated = helpers.resolve('find_positively_correlated_negatively_correlated', scope)

    # Apply student's `find_positively_correlated_negatively_correlated` and register the results.
    stud_corr_feat_te = func_find_positively_correlated_negatively_correlated(w_test)
    stud_corr_feat_gr = func_find_positively_correlated_negatively_correlated(w_grade)
    helpers.register_answer('find_positively_correlated_negatively_correlated', stud_corr_feat_gr, scope)

    # [TEST] `find_positively_correlated_negatively_correlated` returns a tuple/list.
    fail_msg = 'Function "find_positively_correlated_negatively_correlated" should return a tuple of 2 ' \
               'np.arrays, whereas your function returns type {}'.\
        format(type(stud_corr_feat_te))
    assert isinstance(stud_corr_feat_te, (tuple, list)), fail_msg

    # [TEST] `find_positively_correlated_negatively_correlated` returns a tuple/list of two values.
    fail_msg = 'Function "find_positively_correlated_negatively_correlated" should return a tuple of 2 ' \
               'values, whereas your function returns a tuple of {} values'.\
        format(len(stud_corr_feat_te))
    assert len(stud_corr_feat_te) == 2, fail_msg

    # [TEST] `find_positively_correlated_negatively_correlated` returns a tuple/list of two arrays.
    fail_msg = 'Function "find_positively_correlated_negatively_correlated" should return a tuple of 2 ' \
                'np.arrays, whereas your function returns a tuple of {},{}'.\
        format(type(stud_corr_feat_te[0]), type(stud_corr_feat_te[1]))
    assert isinstance(stud_corr_feat_te[0], (np.ndarray, list, tuple)) and isinstance(stud_corr_feat_te[1], (np.ndarray, list, tuple)), fail_msg

    # [TEST] `find_positively_correlated_negatively_correlated` returns incorrect values.
    fail_msg = 'Function "find_positively_correlated_negatively_correlated" returns incorrect values.'
    assert np.allclose(stud_corr_feat_te[0], np.array(neg_corr_te)), fail_msg
    assert np.allclose(stud_corr_feat_te[1], np.array(pos_corr_te)), fail_msg

@test
def test_kmeans_init_centers(scope):
    exercise_id = 'init_centers'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)
    X_te = test_data['X_te']
    Xn_te = test_data['Xn_te']
    X_gr = test_data['X_gr']

    # np.random.RandomState(123)
    ### Test `find_min_max_values`.
    init_centers = helpers.resolve('init_centers', scope)
    # Apply student's `init_centers` and register the results.
    stud_initcenter_te = init_centers(X_te, 10)
    stud_initcenter_gr = init_centers(X_gr, 10)
    helpers.register_answer('init_centers', stud_initcenter_gr, scope)

    # # [TEST]
    # fail_msg = 'Function "init_centers" should return a ndarray of shape (10, 3) ' \
    #            ' whereas your function returns type {}'.\
    #     format(type(stud_initcenter_te))
    # assert isinstance(stud_initcenter_te, (np.ndarray)), fail_msg
    #
    # # [TEST]
    # fail_msg = 'Function "init_centers" should return a ndarray of shape (10, 3) ' \
    #            'values, whereas your function returns a ndarray of {} values'.\
    #     format(stud_initcenter_te.shape)
    # assert stud_initcenter_te.shape == (10,3), fail_msg
    #
    # # [TEST]
    # fail_msg = '"init_centers" returns incorrect values. Expected {}, ' \
    #            'received {}'.format(Xn_te, tuple(stud_initcenter_te))
    # assert np.allclose(stud_initcenter_te,
    #                    np.array(Xn_te)), fail_msg

    helpers.compare_np_arrays(stud_initcenter_te, Xn_te, varname='init_centers')

@test
def test_kmeans_compute_distance(scope):
    exercise_id = 'compute_distance'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)
    X_te = test_data['X_te']
    X_te_center = test_data['X_te_center']
    Xn_te = test_data['Xn_te']
    X_gr = test_data['X_gr']
    X_gr_center = test_data['X_gr_center']

    ### Test `compute_distance`.
    compute_distance = helpers.resolve('compute_distance', scope)
    # Apply student's `compute_distance` and register the results.
    stud_dist_te = compute_distance(X_te, X_te_center, 10)
    stud_dist_gr = compute_distance(X_gr, X_gr_center, 10)
    helpers.register_answer('compute_distance', stud_dist_gr, scope)

    # # [TEST]
    # fail_msg = 'Function "compute_distance" should return a ndarray of shape (100, 10) ' \
    #            ' whereas your function returns type {}'.\
    #     format(type(stud_dist_te))
    # assert isinstance(stud_dist_te, (np.ndarray)), fail_msg
    #
    # # [TEST]
    # fail_msg = 'Function "compute_distance" should return a ndarray of shape (100, 10) ' \
    #            'values, whereas your function returns a ndarray of {} values'.\
    #     format(stud_dist_te.shape)
    # assert stud_dist_te.shape == (100, 10), fail_msg
    #
    # # [TEST]
    # fail_msg = '"compute_distance" returns incorrect values. Expected {}, ' \
    #            'received {}'.format(Xn_te, tuple(stud_dist_te))
    # assert np.allclose(stud_dist_te,
    #                    np.array(Xn_te)), fail_msg
    helpers.compare_np_arrays(stud_dist_te, Xn_te, varname='compute_distance')


@test
def test_kmeans_find_closest_cluster(scope):
    exercise_id = 'find_closest_cluster'

    # Load test data
    test_data = helpers.get_test_data(exercise_id)
    X_te = test_data['X_te']
    Xn_te = test_data['Xn_te']
    X_gr = test_data['X_gr']

    # np.random.RandomState(123)
    ### Test `find_closest_cluster`.
    find_closest_cluster = helpers.resolve('find_closest_cluster', scope)
    # Apply student's `find_closest_cluster` and register the results.
    stud_dist_te = find_closest_cluster(X_te)
    stud_dist_gr = find_closest_cluster(X_gr)
    helpers.register_answer('find_closest_cluster', stud_dist_gr, scope)

    # [TEST]
    # fail_msg = 'Function "find_closest_cluster" should return a ndarray of shape (100, 10) ' \
    #            ' whereas your function returns type {}'.\
    #     format(type(stud_dist_te))
    # assert isinstance(stud_dist_te, (np.ndarray)), fail_msg
    #
    # # [TEST]
    # fail_msg = 'Function "find_closest_cluster" should return a ndarray of shape (100, 10) ' \
    #            'values, whereas your function returns a ndarray of {} values'.\
    #     format(stud_dist_te.shape)
    # assert stud_dist_te.shape == (100, ), fail_msg
    #
    # # [TEST]
    # fail_msg = '"find_closest_cluster" returns incorrect values. Expected {}, ' \
    #            'received {}'.format(Xn_te, tuple(stud_dist_te))
    # assert np.allclose(stud_dist_te,
    #                    np.array(Xn_te)), fail_msg
    helpers.compare_np_arrays(stud_dist_te, Xn_te, varname='find_closest_cluster')


@test
def test_knn_weighted(scope):
    exercise_id = 'knnweighted'

    # Intantiate KNN
    KNN = KNNHelper()
    
    # Load test data
    test_data = helpers.get_test_data(exercise_id)
    te_best_label = test_data['te_best_label']
    te_nn_indices = test_data['te_nn_indices']
    te_w = test_data['te_w'] 
    te_query = test_data['te_query']
    query_grad = test_data['query_grad']

    
    ### Test `predict_label_with_weighted_distance`.
    predict_label_func = helpers.resolve('predict_label_with_weighted_distance', scope)
    # Apply student's `predict_label_with_weighted_distance` and register the results.
    KNN.find_label_with_weighted_KNN(predict_label_func,te_query, verbose=True)
    stud_w_te = KNN.w
    stud_bestlabel_te = KNN.best_label
    stud_nnindices_te = KNN.nn_indices
    
    KNN.find_label_with_weighted_KNN(predict_label_func,query_grad)
    stud_grad = dict(grad_best_label=KNN.best_label, grad_nn_indices=KNN.nn_indices,
        grad_w = KNN.w)
    helpers.register_answer('knnweighted', stud_grad, scope)

    # [TEST]
    fail_msg = 'Function "predict_label_with_weighted_distance" should return {} as ndarray of shape {} ' \
               ' whereas your function returns type {}'.\
        format('w',te_w.shape,type(stud_w_te))
    assert isinstance(stud_w_te, (np.ndarray)), fail_msg
    
    # [TEST]
    fail_msg = 'Function "predict_label_with_weighted_distance" should return {} as ndarray of shape {} ' \
               ' whereas your function returns shape {}'.\
        format('w',te_w.shape,stud_w_te.shape)
    assert stud_w_te.shape == te_w.shape, fail_msg
    
     # [TEST]
    fail_msg = '"predict_label_with_weighted_distance" should return {} as a scalar having one of the types in {} , ' \
               'received {} instead'.format('predicted_label',['numpy.int64','numpy.int32', 'int'],type(stud_bestlabel_te))
    assert isinstance(stud_bestlabel_te,(np.int64,np.int32, int)), fail_msg
    

    # [TEST]
    fail_msg = '"predict_label_with_weighted_distance" returns incorrect predicted label. Expected {}, ' \
               'received {}'.format(te_best_label,stud_bestlabel_te)
    assert te_best_label == stud_bestlabel_te, fail_msg
    
    
    # [TEST]
    helpers.compare_np_arrays(stud_w_te, te_w, varname='w')
    
    # in order to the plot of the neighbors
    KNN.find_label_with_weighted_KNN(predict_label_func,te_query)
    KNN.plot_k_nearest_neighbors()

@test
def test_num_folds_LOOCV(scope):
    
    exercise_id = 'num_folds_LOOCV'
    test_data = helpers.get_test_data(exercise_id)

    # get implementation
    numfolds_LOOCV_func = helpers.resolve('num_folds_LOOCV', scope)


    # get data from stored file
    X_te = test_data['X_te']
    X_gr = test_data['X_gr']
    num_folds_te = test_data['num_folds_te']


    # run the function on test data
    stud_num_folds_loocv_te  =  numfolds_LOOCV_func(X_te)
   

    # [TEST]
    fail_msg = 'Error! "num_folds_LOOCV" should return number of folds in LOOCV as a scalar having one of the types in {} , ' \
               'received {} instead'.format(['numpy.int64','numpy.int32', 'int'],type(stud_num_folds_loocv_te))
    assert isinstance(stud_num_folds_loocv_te,(np.int64,np.int32, int)), fail_msg
    

    # [TEST]
    fail_msg = 'Error! "num_folds_LOOCV" returns incorrect output for number of folds in LOOCV.'
  
    assert stud_num_folds_loocv_te ==  num_folds_te, fail_msg


    
    # run the function on grading data
    stud_num_folds_loocv_gr  =  numfolds_LOOCV_func(X_gr)

    # register answer 
    helpers.register_answer('num_folds_LOOCV', stud_num_folds_loocv_gr, scope)


