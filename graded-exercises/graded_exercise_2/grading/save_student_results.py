import numpy as np
from matplotlib import pyplot as plt
import grading.helpers as helpers
import torch
from helpers.helper import *


################################################################################
# Saving student results
################################################################################
def initialize_res(scope):
    exercise_id = "sciper"
    sciper_number = helpers.resolve('sciper_number', scope)
    stud_grad = dict(sciper_number=sciper_number)
    helpers.register_answer(exercise_id, stud_grad, scope)

def kernel_function(scope):
    exercise_id = 'kernel_function'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    stud_res = func(test_data['gr_X1'],test_data['gr_X2'])

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)

def save_tanh(scope):
    exercise_id = 'Tanh'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    res1 = func.forward(test_data['gr_z'])
    res2 = func.gradient(test_data['gr_z'])

    #save the students' results
    stud_grad = dict(res1=res1, res2=res2)
    helpers.register_answer(exercise_id, stud_grad, scope)


def save_forward_pass(scope):
    exercise_id = 'forward_pass'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    gr_x = test_data["gr_x"]
    gr_weights1_2 = test_data["gr_weights1_2"]
    gr_bias2 = test_data["gr_bias2"]
    gr_weights2_3 = test_data["gr_weights2_3"]
    gr_bias3 = test_data["gr_bias3"]
    gr_weights3_4 = test_data["gr_weights3_4"]
    gr_bias4 = test_data["gr_bias4"]

    res1 = func(gr_x, gr_weights1_2, gr_bias2, gr_weights2_3, gr_bias3, gr_weights3_4, gr_bias4, Sigmoid)

    #save the students' results
    stud_grad = dict(res1=res1)
    helpers.register_answer(exercise_id, stud_grad, scope)


### PYTORCH PART



def f(scope):
    exercise_id = 'f'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    stud_res = func(torch.tensor(test_data['gr_x'])).numpy()

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)


def init_x_and_optim(scope):
    exercise_id = 'init_x_and_optim'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    x_0, opt = func(test_data['gr_x'][0], test_data['gr_x'][1])
    stud_res = np.array([x_0.item(), opt.param_groups[0]['lr']])
    
    try:
        res_bool = x_0.requires_grad
    except:
        res_bool = False

    #save the students' results
    stud_grad = dict(stud_res=stud_res, stud_res_bool=res_bool)
    helpers.register_answer(exercise_id, stud_grad, scope)


def minimize(scope):
    exercise_id = 'minimize'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data_PICKLED(exercise_id)

    # run the students' functions
    # Create a fake test function
    def f_test(x):
        return (x-1)**2

    # Load initial tensor and optimizer
    x_0_test = test_data['gr_x'][0]
    opt_test = test_data['gr_x'][1]

    stud_res = func(f_test, x_0_test, opt_test, 5)
    stud_res = np.array(stud_res)

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)


def count_steps(scope):
    exercise_id = 'count_steps'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data(exercise_id)

    # run the students' functions
    stud_res = func(test_data['gr_x'].tolist(), test_data['gr_thresh'][0])
    stud_res = np.array([stud_res])

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)


def maximize(scope):
    exercise_id = 'maximize'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data_PICKLED(exercise_id)

    # run the students' functions
    # Create a fake test function
    def f_test(x):
        return -(x-1)**2

    # Load initial tensor and optimizer
    x_0_test = test_data['gr_x'][0]
    opt_test = test_data['gr_x'][1]

    stud_res = func(f_test, x_0_test, opt_test, 5)
    stud_res = np.array(stud_res)

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)






def construct_dictionary(scope):
    exercise_id = 'construct_dictionary'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data_PICKLED(exercise_id)

    # Load initial tensor and optimizer
    input_bow_list = test_data['gr_x'][0]
    stopwords = test_data['gr_x'][1]

    stud_res = func(input_bow_list, stopwords)

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)





def get_bow_array(scope):
    exercise_id = 'get_bow_array'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get the grading data
    test_data = helpers.get_data_PICKLED(exercise_id)

    # Load initial tensor and optimizer
    input_bow_list = test_data['gr_x'][0]
    dictionary = test_data['gr_x'][1]

    stud_res = func(input_bow_list, dictionary)

    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)



