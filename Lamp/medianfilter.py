import sys
import scipy as sp
from scipy import signal

power = sys.argv[1]

power = power[1:len(power)-1]
power = power.split(',')

for i in range(len(power)):
	power[i] = float(power[i])

power = sp.signal.medfilt(power) # add noise to the signal
buff = "{\"power\" : " + str(power.tolist()) + "}"

print(buff)