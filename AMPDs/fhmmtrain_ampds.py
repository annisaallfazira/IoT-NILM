import sys
import json
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from hmmlearn.hmm import GaussianHMM
from collections import OrderedDict

train_power = OrderedDict()
data = pd.read_csv('D:/15. DERARA/4. Data/ampds.csv')
# train_power['B1E'] = data['B1E'].values[0:10080].reshape(-1,1)
# train_power['B2E'] = data['B2E'].values[0:10080].reshape(-1,1)
train_power['BME'] = sp.signal.medfilt(data['BME'].values[0:20160]).reshape(-1,1)
train_power['CDE'] = sp.signal.medfilt(data['CDE'].values[0:20160]).reshape(-1,1)
# train_power['CWE'] = sp.signal.medfilt(data['CWE'].values[0:10080]).reshape(-1,1)
# train_power['DWE'] = sp.signal.medfilt(data['DWE'].values[0:10080]).reshape(-1,1)
# train_power['EQE'] = sp.signal.medfilt(data['EQE'].values[0:10080]).reshape(-1,1)
train_power['FGE'] = sp.signal.medfilt(data['FGE'].values[0:20160]).reshape(-1,1)
# train_power['FRE'] = sp.signal.medfilt(data['FRE'].values[0:10080]).reshape(-1,1)
train_power['HPE'] = sp.signal.medfilt(data['HPE'].values[0:20160]).reshape(-1,1)
# train_power['HTE'] = sp.signal.medfilt(data['HTE'].values[0:10080]).reshape(-1,1)
# train_power['OFE'] = data['OFE'].values[0:10080].reshape(-1,1)
train_power['TVE'] = sp.signal.medfilt(data['TVE'].values[0:20160]).reshape(-1,1)
# train_power['UNE'] = sp.signal.medfilt(data['UNE'].values[0:10080]).reshape(-1,1)
# train_power['UTE'] = data['UTE'].values[0:10080].reshape(-1,1)
# train_power['WOE'] = sp.signal.medfilt(data['WOE'].values[0:10080]).reshape(-1,1)

"""train_power['BME'] = sp.signal.medfilt(data['BME'].values[0:10080]).reshape(-1,1)
train_power['CDE'] = sp.signal.medfilt(data['CDE'].values[0:10080]).reshape(-1,1)
train_power['DWE'] = sp.signal.medfilt(data['DWE'].values[0:10080]).reshape(-1,1)
train_power['FGE'] = sp.signal.medfilt(data['FGE'].values[0:10080]).reshape(-1,1)
train_power['FRE'] = sp.signal.medfilt(data['FRE'].values[0:10080]).reshape(-1,1)
train_power['HPE'] = sp.signal.medfilt(data['HPE'].values[0:10080]).reshape(-1,1)
train_power['OFE'] = sp.signal.medfilt(data['OFE'].values[0:10080]).reshape(-1,1)
train_power['TVE'] = sp.signal.medfilt(data['TVE'].values[0:10080]).reshape(-1,1)
train_power['WOE'] = sp.signal.medfilt(data['WOE'].values[0:10080]).reshape(-1,1)"""

# train_power['CWE'] = sp.signal.medfilt(data['CWE'].values[0:10080]).reshape(-1,1)

num_appliances = len(train_power)

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
state_appliances['HTE'] = 1  #3
state_appliances['OFE'] = 2
state_appliances['OUE'] = 1 #bye
state_appliances['TVE'] = 2
state_appliances['UTE'] = 1
state_appliances['WOE'] = 3
state_appliances['RSE'] = 3 #bye
state_appliances['UNE'] = 4

"""for appliance in state_appliances:
	state_appliances[appliance] = 2"""

# Modeling each appliance with GaussianHMM
model = OrderedDict()
for appliance in train_power:
	model[appliance] = GaussianHMM(n_components=state_appliances[appliance], covariance_type="full").fit(train_power[appliance])

startprob = OrderedDict()
for appliance in train_power:
	if state_appliances[appliance] == 2:
		startprob[appliance] = [0.5, 0.5]
	elif state_appliances[appliance] == 3:
		startprob[appliance] = [1/3, 1/3, 1/3]
	else:
		startprob[appliance] = model[appliance].startprob_.tolist()
	# startprob[appliance] = model[appliance].startprob_.tolist()

buff = "{"

for appliance in model:
	buff += "\"" + appliance + "\" : {" + \
			"\"startprob\" : " + str(startprob[appliance]) + \
			", \"transmat\" : " + str(model[appliance].transmat_.tolist()) + \
			",\"means\" : " + str(model[appliance].means_.tolist()) + \
			", \"covars\" : " + str(model[appliance].covars_.tolist()) + "},"

buff = buff[0:len(buff)-1]	
buff += "}"	
print(buff)