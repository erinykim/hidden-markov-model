import numpy as np
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """
    # TODO: complete this function.

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0.000, 'Bend': 0.000},
        'B2': {'B1': 0.000, 'B2': 0.625, 'B3': 0.375, 'Bend': 0.000},
        'B3': {'B1': 0.000, 'B2': 0.000, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0.000, 'B2': 0.000, 'B3': 0.000, 'Bend': 1.000},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.750, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0.000, 'Cend': 0.000},
        'C2': {'C1': 0.000, 'C2': 0.000, 'C3': 1.000, 'Cend': 0.000},
        'C3': {'C1': 0.000, 'C2': 0.000, 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0.000, 'C2': 0.000, 'C3': 0.000, 'Cend': 1.000},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0.000, 'Hend': 0.000},
        'H2': {'H1': 0.000, 'H2': 0.857, 'H3': 0.143, 'Hend': 0.000},
        'H3': {'H1': 0.000, 'H2': 0.000, 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0.000, 'H2': 0.000, 'H3': 0.000, 'Hend': 1.000},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.

    """
    
    sequence = []
    probability = 0.0
    likely_states = []
    length_ev = len(evidence_vector)

    if length_ev == 0:
        return (sequence, probability)
    else:
        b1 = (['B1'],gaussian_prob(evidence_vector[0],emission_paras['B1'])*prior_probs['B1'])
        c1 = (['C1'],gaussian_prob(evidence_vector[0],emission_paras['C1'])*prior_probs['C1'])
        h1 = (['H1'],gaussian_prob(evidence_vector[0],emission_paras['H1'])*prior_probs['H1'])
        likely_states = [b1, c1, h1]
        for i in range(1,length_ev):
            counter = 0
            length = len(likely_states)
            for j in likely_states:
                if counter > length:
                    break
                elif j[0][0] == 'B1':
                    trans = ['B1','B2','B3']
                elif j[0][0] == 'C1':
                    trans = ['C1','C2','C3']
                elif j[0][0] == 'H1':
                    trans = ['H1','H2','H3']
                if len(j[0]) == i:
                    end = j[0][-1]
                    for curr in trans:
                        if transition_probs[end][curr] != 0:
                            curr_t_probs = transition_probs[end][curr]
                            curr_p_probs = j[1]
                            curr_guassian_probs = gaussian_prob(evidence_vector[i],emission_paras[curr])*curr_t_probs*curr_p_probs
                            curr_likely_states = j[0][:]
                            curr_likely_states.append(curr)
                            likely_states.append((curr_likely_states, curr_guassian_probs))
                counter += 1
    
    for i in likely_states:
        if len(i[0]) == length_ev:
            if i[1] > probability:
                sequence = i[0]
                probability = i[1]

    return (sequence, probability)


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """

    # TODO: complete this function.

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0.000, 0.000), 'Bend': (0.000, 0.000)},
        'B2': {'B1': (0.000, 0.000), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0.000, 0.000)},
        'B3': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'H1': (0.125, 0.091), 'C1': (0.125, 0.091)},
        'Bend': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.000, 0.000), 'Bend': (1.000, 1.000)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0.000, 0.000), 'Cend': (0.000, 0.000)},
        'C2': {'C1': (0.000, 0.000), 'C2': (0.000, 0.625), 'C3': (1.000, 0.375), 'Cend': (0.000, 0.000)},
        'C3': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125), 'H1': (0.067, 0.125)},
        'Cend': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.000, 0.000), 'Cend': (1.000, 1.000)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.300, 10.659)],
        'C2': [(43.667, 1.700), (37.110, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0.000, 0.000), 'Hend': (0.000, 0.000)},
        'H2': {'H1': (0.000, 0.000), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0.000, 0.000)},
        'H3': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'B1': (0.063, 0.059), 'C1': (0.063, 0.059)},
        'Hend': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.000, 0.000), 'Hend': (1.000, 1.000)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    # TODO: complete this function.

    sequence = []
    probability = 0.0
    likely_states = []
    length_ev = len(evidence_vector)

    if length_ev == 0:
        return (sequence, probability)
    else:
        b1 = (['B1'], gaussian_prob(evidence_vector[0][0],emission_paras['B1'][0])*gaussian_prob(evidence_vector[0][1],emission_paras['B1'][1])*prior_probs['B1'])
        c1 = (['C1'], gaussian_prob(evidence_vector[0][0],emission_paras['C1'][0])*gaussian_prob(evidence_vector[0][1],emission_paras['C1'][1])*prior_probs['C1'])
        h1 = (['H1'], gaussian_prob(evidence_vector[0][0],emission_paras['H1'][0])*gaussian_prob(evidence_vector[0][1],emission_paras['H1'][1])*prior_probs['H1'])
        likely_states = [b1, c1, h1]
        prob = []
        percentile_75 = 0
        for i in range(1,length_ev):
            counter = 0
            length = len(likely_states)
            if i > 15:
                percentile_75 = np.percentile(prob, 75) 
            for j in likely_states:
                if counter > length:
                    break
                if len(j[0]) == i and j[1] > percentile_75:
                    end = j[0][-1]
                    if j[0][-1] in ('B1','B2','B3'):
                        trans = ['B1','B2','B3','C1','H1']
                    elif j[0][-1] in ('C1','C2','C3'):
                        trans = ['C1','C2','C3','B1','H1']
                    elif j[0][-1] in ('H1','H2','H3'):
                        trans = ['H1','H2','H3','B1','C1']
                    for k in trans:
                        if transition_probs[end].get(k) is None:
                            break
                        else:
                            if transition_probs[end][k][0] != 0 and transition_probs[end][k][1] != 0:
                                curr_t_prob = transition_probs[end][k][0]
                                curr_p_prob = j[1]
                                curr_left_t_prob = transition_probs[end][k][1]
                                curr_gaussian_left_prob = gaussian_prob(evidence_vector[i][1], emission_paras[k][1])
                                curr_gaussian_right_prob = gaussian_prob(evidence_vector[i][0], emission_paras[k][0])
                                curr_prob = curr_t_prob*curr_p_prob*curr_left_t_prob*curr_gaussian_left_prob*curr_gaussian_right_prob
                                curr_likely_states = j[0][:]
                                curr_likely_states.append(k)
                                likely_states.append((curr_likely_states, curr_prob))
                                prob.append(curr_prob)
                counter += 1

    for j in likely_states:
        if len(j[0]) == length_ev:
            if j[1] > probability:
                sequence = j[0]
                probability = j[1]

    return (sequence, probability)
