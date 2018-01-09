import numpy as np
from contextlib import closing
from matplotlib import pyplot as plt
from IPython import display
from dt3100 import DT3100, ureg

#physical units
#from pint import UnitRegistry
#ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.default_format = '~P'

def show_measurement(capteur, ndisplayed = 2**13, chunk = None, ymin=None, ymax=None):
    """Display measurements in a rolling graph"""
    if chunk is None:
        chunk = 2**int(np.log2(ndisplayed/32))
    #unit time per sample
    dt = capteur.unit_time()
    #create empty plot
    hl, = plt.plot([], [])
    #set x and y range
    plt.xlim(0, dt.m * ndisplayed)
    plt.xlabel('time (%s)'%dt.u)
    if ymin is None:
        ymin = capteur.sensor.smr.m
    if ymax is None:
        ymax = capteur.sensor.emr.m
    plt.ylim(ymin, ymax)
    plt.ylabel('distance (%s)'%capteur.sensor.smr.u)
    
    while True:
        try:
            data = capteur.readN(chunk)
            hl.set_ydata(np.append(hl.get_ydata(), data)[-ndisplayed:])
            hl.set_xdata(dt * np.arange(len(hl.get_ydata())))
            display.clear_output(wait=True)
            display.display(plt.gcf())
        except KeyboardInterrupt:
            return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Display measurements from a DT3100 controller in a rolling graph (In Notebook only).')
    parser.add_argument('ip', nargs='?', default='169.254.3.100', help='IP address of the controller.')
    parser.add_argument(
        '--avt', default=0, type=int, 
        help=DT3100.set_averaging_type.__doc__
    )
    parser.add_argument(
        '--avn', default=0, type=int, 
        help=DT3100.set_averaging_number.__doc__
    )
    parser.add_argument('--ymin', default=None, help='minimum distance')
    parser.add_argument('--ymax', default=None, help='maximum distance')
    parser.add_argument('--tw', default=1.0, type=float, help='Width of the time window, in seconds.')
    args = parser.parse_args()
    with closing(DT3100(args.ip)) as capteur:
        capteur.set_averaging_type(args.avt)
        capteur.set_averaging_number(args.avn)
        show_measurement(capteur, ndisplayed = int(Q_(args.tw, 's')/capteur.unit_time()), ymin=args.ymin, ymax=args.ymax)
