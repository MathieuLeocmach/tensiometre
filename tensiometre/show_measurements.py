import time
import numpy as np
from contextlib import closing, ExitStack
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython import display
from tensiometre.dt3100 import DT3100, ureg, recover
from threading import Thread, Event

#physical units
#from pint import UnitRegistry
#ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.default_format = '~P'



class Updater(Thread):
    """A thread to update continuously a ring buffer for sensor's data"""
    def __init__(self, ringbuff, sensor, chunk=32, outputFile=False, subsample=1):
        Thread.__init__(self)
        self.stop = Event()
        self.sensor = sensor
        self.ringbuff = ringbuff
        self.chunk = chunk
        self.file = outputFile
        self.subsample = subsample

    def run(self):
        shape = (self.chunk // self.subsample, self.subsample)
        leng = np.prod(shape)
        while not self.stop.is_set():
            buff = self.sensor.readN(self.chunk).m
            self.ringbuff[:] = np.roll(self.ringbuff, -self.chunk)
            self.ringbuff[-self.chunk:] = buff
            if self.file:
                buff[-leng:].reshape(shape).mean(-1).tofile(self.file)
        self.sensor.end_acquisition()

class DiscontinuousUpdater(Thread):
    """A thread to fetch one by one data from a sensor and update a ring buffer"""
    def __init__(self, ringbuff, sensor, outputFile=False):
        Thread.__init__(self)
        self.stop = Event()
        self.sensor = sensor
        self.ringbuff = ringbuff
        self.file = outputFile

    def run(self):
        while not self.stop.is_set():
            v = self.sensor.readOne().m
            self.ringbuff[:] = np.roll(self.ringbuff, -1, axis=0)
            self.ringbuff[-1] = [time.time(), v]
            if self.file:
                buff[-1:].tofile(self.file)

class Shower:
    """A class to show a figure for a sensor"""
    def __init__(self, ringbuffs, dts, ymin, ymax, name='DT3100', sleeptime=0.1, tw=1, names=None):
        if names is None:
            names = [f'sensor {i:d}' for i in range(len(ringbuffs))]
        self.ringbuffs = ringbuffs
        self.dts = dts
        self.st = sleeptime
        #plt.ion()
        #create empty plot
        self.fig, self.ax = fig, ax = plt.subplots(num=name)
        self.fig.clf()
        #set x and y range
        plt.xlim(0, tw)
        plt.xlabel('time (s)')
        plt.ylim(ymin, ymax)
        plt.ylabel('distance (µm)')
        self.hl = []
        for ringbuff, dt, label in zip(self.ringbuffs, self.dts, names):
            self.hl.append(plt.plot(dt.m * np.arange(len(ringbuff)), ringbuff, label=label)[0])
        plt.legend()

        def animate(i):
            for line, ringbuff in zip(self.hl, self.ringbuffs):
                line.set_ydata(ringbuff)  # update the data.
            return self.hl
        self.ani = animation.FuncAnimation(
                self.fig, animate, interval=self.st*1000, blit=True, save_count=1, repeat=True)
        plt.show(block=False)

class DiscontinuousShower:
    """A class to show a figure from buffers that contain time and value information"""
    def __init__(self, ringbuffs, ymin, ymax, name='DT3100', sleeptime=0.1, tw=1, names=None):
        if names is None:
            names = [f'sensor {i:d}' for i in range(len(ringbuffs))]
        self.ringbuffs = ringbuffs
        self.st = sleeptime
        #create empty plot
        self.fig, self.ax = fig, ax = plt.subplots(num=name)
        self.fig.clf()
        #set x and y range
        plt.xlim(0, tw)
        plt.xlabel('time (s)')
        plt.ylim(ymin, ymax)
        plt.ylabel('distance (µm)')
        self.hl = []
        for ringbuff, label in zip(self.ringbuffs, names):
            self.hl.append(plt.plot(ringbuff[:,0], ringbuff[:,1], label=label)[0])
        plt.legend()

        def animate(i):
            for line, ringbuff in zip(self.hl, self.ringbuffs):
                line.set_xdata(ringbuff[:,0]-ringbuff[-1,0]+tw)
                line.set_ydata(ringbuff[:,1])  # update the data.
            return self.hl
        self.ani = animation.FuncAnimation(
                self.fig, animate, interval=self.st*1000, blit=True, save_count=1, repeat=True)
        plt.show(block=False)

def show_measurement(sensors, tw=1, ymin=None, ymax=None, names=None, fast=False):
    """Display measurements in a rolling graph"""
    ringbuffs = []
    updts = []
    if fast:
        for sensor in sensors:
            ndisplayed = int(Q_(tw, 's')/sensor.unit_time())
            ringbuff = np.zeros(ndisplayed, float)
            ringbuffs.append(ringbuff)
            chunk = 2**int(np.log2(ndisplayed/32))
            updts.append(Updater(ringbuff, sensor, chunk))
    else:
        for sensor in sensors:
            ringbuff = np.zeros((int(100*tw),2), float)
            ringbuffs.append(ringbuff)
            updts.append(DiscontinuousUpdater(ringbuff, sensor))

    for updt in updts:
        updt.start()




    dts = [sensor.unit_time() for sensor in sensors]
    #set x and y range
    if ymin is None:
        ymin = min(sensor.sensor.smr.m for sensor in sensors)
    if ymax is None:
        ymax = max(sensor.sensor.emr.m for sensor in sensors)

    if fast:
        shw = Shower(
            ringbuffs, dts, ymin, ymax, tw=tw, names=names,
        )
    else:
        shw = DiscontinuousShower(
            ringbuffs, ymin, ymax, tw=tw, names=names,
        )

    fig = shw.fig
    #fig.axes[0].has_been_closed = False
    #fig.canvas.mpl_connect('close_event', on_close)
    plt.show()#(block=True)
    #plt.pause(0.01)
    try:
        while min(updt.is_alive() and not updt.stop.is_set() for upt in updts):
            #pass
            active_fig_managers = plt._pylab_helpers.Gcf.figs.values()
            if fig not in active_fig_managers:
                break

    except KeyboardInterrupt:
        pass
    finally:
        for updt in updts:
            updt.stop.set()
        for updt in updts:
            updt.join()
        for sensor in sensors:
            sensor.end_acquisition()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Display measurements from a DT3100 sensor in a rolling graph.')
    parser.add_argument('ip', nargs='*', help='IP addresses of the DT3100 (default 169.254.3.100).')
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
    parser.add_argument('--fast', default=False, type=bool, help='Whether to use continuous reading of the sensors.')
    args = parser.parse_args()
    if args.ip is None or len(args.ip)==0:
        args.ip = ['169.254.3.100']
    for ip in args.ip:
        recover(ip)
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(ip))) for ip in args.ip]
        for sensor in sensors:
            sensor.set_averaging_type(args.avt)
            sensor.set_averaging_number(args.avn)
        show_measurement(sensors, args.tw, ymin=args.ymin, ymax=args.ymax, names=args.ip, fast=args.fast)
