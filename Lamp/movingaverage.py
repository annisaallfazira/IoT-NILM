import sys
import numpy as np
import scipy as sp
from scipy import signal

power = sys.argv[1]

power = power[1:len(power)-1]
power = power.split(',')


"""------------------- Median Filter ----------------------"""
for i in range(len(power)):
	power[i] = float(power[i])

power = sp.signal.medfilt(power) # add noise to the signal

"""------------------- Moving Average ----------------------"""
window_size = 5

count = 0
power_MA = np.zeros(len(power))
batas = 6.8

for i in range(0,len(power)):
	this_window = power[i-count:i+1]
	power_MA[i] = sum(this_window) / len(this_window)
	
	if count < 4:
		count+=1
	
	if i < len(power)-1:
		if abs(power[i]-power[i+1]) > batas:
			count=0


buff = "{\"power\" : " + str(power_MA.tolist()) + "}"
print(buff)