import json
import sys
import itertools
import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal
from copy import deepcopy
from matplotlib import pyplot as plt
from collections import OrderedDict
from hmmlearn.hmm import GaussianHMM

""" ----------------------------- Function ----------------------------- """

# Function to sort parameters from its lowest state
def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping
	
def sort_startprob(mapping, startprob):
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob

def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars

def sort_transition_matrix(mapping, A):
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new

def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]
	
# Function to combined each parameter off all appliances
def compute_A_fhmm(list_A):
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result

def compute_pi_fhmm(list_pi):
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result
	
def compute_means_fhmm(list_means):
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    
    return [means, cov]

def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]
				  
    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

    combined_model = GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    
    return combined_model


# Disagregate energy, finding state and power of each appliance
def decode_hmm(length_sequence, centroids, appliance_list, states):
    hmm_states = OrderedDict()
    hmm_power = OrderedDict()
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):
        factor = total_num_combinations
        
        for appliance in appliance_list:
            factor = factor // len(centroids[appliance])
            temp = int(states[i])/factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])	
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]
			
    return [hmm_states, hmm_power]

def disaggregate(combined_model, model, test_power):
	
    length = len(test_power)
    temp = test_power.reshape(length, 1)
    states = (combined_model.predict(temp))
	
    means = OrderedDict()
    for appliance in model:
        means[appliance] = (
            model[appliance].means_.flatten().tolist())
        means[appliance].sort()

    [decoded_states, decoded_power] = decode_hmm(
        len(states), means, means.keys(), states)
        
    return [decoded_power, decoded_states]

data = sys.argv[1]
data = json.loads(data)

model_appliance = data['model']

state_appliances = OrderedDict()
state_appliances['B1E'] = 3
state_appliances['B2E'] = 2
state_appliances['BME'] = 2
state_appliances['CDE'] = 2
state_appliances['CWE'] = 3
state_appliances['DNE'] = 3	#bye
state_appliances['DWE'] = 3
state_appliances['EBE'] = 1	#bye
state_appliances['EQE'] = 1
state_appliances['FGE'] = 2
state_appliances['FRE'] = 1
state_appliances['GRE'] = 1 #bye
state_appliances['HPE'] = 2 
state_appliances['HTE'] = 3
state_appliances['OFE'] = 2
state_appliances['OUE'] = 1 #bye
state_appliances['TVE'] = 2
state_appliances['UTE'] = 1
state_appliances['WOE'] = 3
state_appliances['RSE'] = 3 #bye
state_appliances['UNE'] = 4


startprob = OrderedDict()
transmat = OrderedDict()
means = OrderedDict()
covars = OrderedDict()
model = OrderedDict()

for appliance in model_appliance:	
	startprob[appliance] = np.array(model_appliance[appliance]['startprob'])
	transmat[appliance] = np.array(model_appliance[appliance]['transmat'])
	means[appliance] = np.array(model_appliance[appliance]['means'])
	covars[appliance] = np.array(model_appliance[appliance]['covars'])
	
for appliance in model_appliance:
	model[appliance] = GaussianHMM(n_components=state_appliances[appliance], covariance_type="full")
	model[appliance].startprob_ = startprob[appliance]
	model[appliance].transmat_ = transmat[appliance]
	model[appliance].means_ = means[appliance]
	model[appliance].covars_ = covars[appliance]


new_model = OrderedDict()
for appliance in model:
    startprob_new, means_new, covars_new, transmat_new = sort_learnt_parameters(			
		startprob[appliance], means[appliance],
        covars[appliance], transmat[appliance])
                
    new_model[appliance] = GaussianHMM(n_components=startprob_new.size, covariance_type="full")
    new_model[appliance].startprob_ = startprob_new
    new_model[appliance].transmat_ = transmat_new
    new_model[appliance].means_ = means_new
    new_model[appliance].covars_ = covars_new
	
combined_model = create_combined_hmm(new_model)

# Testing data and predicting power of each appliance
test_power = data['powerTest']
test_power = test_power.split(',')

for i in range(len(test_power)):
	test_power[i] = float(test_power[i])

# test_power = sp.signal.medfilt(np.array(test_power), kernel_size=3)
test_power = sp.signal.medfilt(np.array(test_power))
length = len(test_power)
[predicted_power,predicted_states] = disaggregate(combined_model, new_model, test_power)
	
predicted_total = np.zeros(length)
for appliance in predicted_power:
	predicted_power[appliance] = predicted_power[appliance]
	predicted_total += predicted_power[appliance]

buff = "\"power\" : {"
for appliance in predicted_power:
	buff += "\"power" + appliance + "\" : " + str(predicted_power[appliance].tolist()) + ","
buff += "\"total\" : " + str(predicted_total.tolist()) + "}"

buff += ", \"state\" : {"
for appliance in predicted_power:
	buff += "\"state" + appliance + "\" : " + str(predicted_states[appliance].tolist()) + ","

buff = buff[0:len(buff)-1]
buff += "} }"

print(buff)
