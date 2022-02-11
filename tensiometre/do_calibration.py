from datetime import datetime
import numpy as np
# from contextlib import closing
# from scipy.optimize import curve_fit
# from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
# from tensiometre.mpc385 import MPC385
# from tensiometre.pid import PID
# from tensiometre import moverPID
import calibration
#from tensiometre import mechtest3sensors
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('tkagg')

#print(plt.get_backend())
fig, axs = plt.subplots(2,2, sharex='col', sharey='row')
ab2xy= calibration.interactive(samples = 16*10*np.arange(21), axs=axs);
name = datetime.now().strftime('%Y%m%d_%H%M_calib')
np.save(name+'.npy', ab2xy)
plt.savefig(name+'.pdf')
