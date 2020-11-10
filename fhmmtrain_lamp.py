import sys
import json
import numpy as np
from hmmlearn.hmm import GaussianHMM

power = sys.argv[1]
power = power.split(',')

for i in range(len(power)):
	power[i] = float(power[i])

train = np.array(power).reshape(-1,1)
model = GaussianHMM(n_components=2, covariance_type="full").fit(train)
startprob = np.array([0.5, 0.5])

buff = "{\"startprob\" : " + str(startprob.tolist()) + ", \"transmat\" : " + str(model.transmat_.tolist()) + ",\"means\" : " + str(model.means_.tolist()) + ", \"covars\" : " + str(model.covars_.tolist()) + "}"

print(buff)