import time
import numpy as np
from contextlib import closing
from matplotlib import pyplot as plt
from IPython import display
from tensiometre.dt3100 import DT3100, ureg
from threading import Thread

#physical units
#from pint import UnitRegistry
#ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.default_format = '~P'



class Updater(Thread):
    """A thread to update continuously a ring buffer for sensor's data"""
    def __init__(self, ringbuff, capteur, chunk=32, outputFile=False, subsample=1):
        Thread.__init__(self)
        self.go = True
        self.capteur = capteur
        self.ringbuff = ringbuff
        self.chunk = chunk
        self.file = outputFile
        self.subsample = subsample
                
    def run(self):
        shape = (self.chunk // self.subsample, self.subsample)
        leng = np.prod(shape)
        while self.go:
            buff = self.capteur.readN(self.chunk)
            self.ringbuff[:] = np.roll(self.ringbuff, -self.chunk)
            self.ringbuff[-self.chunk:] = buff
            if self.file:
                buff[-leng:].reshape(shape).mean(-1).tofile(self.file)
        self.capteur.end_acquisition()
            
class Shower:
    """A class to show a figure for a sensor"""
    def __init__(self, ringbuff, dt, ymin, ymax, name='DT3100', sleeptime=0.1):
        self.ringbuff = ringbuff
        ndisplayed = len(self.ringbuff)
        #create empty plot
        self.fig = plt.figure(name)
        self.fig.clf()
        plt.show()
        self.hl, = plt.plot(dt * np.arange(ndisplayed), self.ringbuff)
        #set x and y range
        plt.xlim(0, dt.m * ndisplayed)
        plt.xlabel('time (%s)' % dt.u)
        plt.ylim(ymin, ymax)
        plt.ylabel('distance (um)')
        self.st = sleeptime
        #self.go = True
        
    def __call__(self):
        #if not plt.fignum_exists(self.fig.number):
        #    self.go = False
        #    return
        self.hl.set_ydata(self.ringbuff)
        #If the graph is displayed inline in the notebook, 
        #updating procedure is different
        if plt.get_backend()[-6:] == 'inline':
            display.clear_output(wait=True)
            display.display(self.fig)
            time.sleep(self.st)
        else:
            plt.draw()
            plt.pause(self.st)
            
def show_measurement(capteur, ndisplayed = 2**13, chunk = None, ymin=None, ymax=None):
    """Display measurements in a rolling graph"""
    if chunk is None:
        chunk = 2**int(np.log2(ndisplayed/32))
    #unit time per sample
    dt = capteur.unit_time()
    ringbuff = np.zeros(ndisplayed, float)
    ringbuff[:] = capteur.readN(ndisplayed)
    #set x and y range
    if ymin is None:
        ymin = capteur.sensor.smr.m
    if ymax is None:
        ymax = capteur.sensor.emr.m
    shw = Shower(
        ringbuff, dt, ymin, ymax, 
        #name='DT3100-%d'%capteur.sensor.sn, 
        #sleeptime=0.1
    )
    updt = Updater(ringbuff, capteur, chunk)
    updt.start()
    try:
        while updt.is_alive():
            shw()
            
    except KeyboardInterrupt:
        pass
    finally:
        updt.go = False
        updt.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Display measurements from a DT3100 sensor in a rolling graph (In Notebook only).')
    parser.add_argument('ip', nargs='?', default='169.254.3.100', help='IP address of the DT3100.')
    parser.add_argument(
        '--avt', default=0, type=int, 
        help=DT3100.set_averaging_type.__doc__
    )
    parser.add_argument(
        '--avn', default=0, type=int, 
        help=DT3100.set_averaging_number.__doc__
    )
    parser.add_argument('--ymin', default=None, type=float, help='minimum distance')
    parser.add_argument('--ymax', default=None, type=float, help='maximum distance')
    parser.add_argument('--tw', default=1.0, type=float, help='Width of the time window, in seconds.')
    args = parser.parse_args()
    with closing(DT3100(args.ip)) as capteur:
        capteur.set_averaging_type(args.avt)
        capteur.set_averaging_number(args.avn)
        show_measurement(capteur, ndisplayed = int(Q_(args.tw, 's')/capteur.unit_time()), ymin=args.ymin, ymax=args.ymax)
