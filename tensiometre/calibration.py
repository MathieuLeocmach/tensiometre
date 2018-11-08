from contextlib import closing
import numpy as np
from scipy.optimize import curve_fit
from tensiometre.dt3100 import DT3100, ReadOne, read_both
from tensiometre.mpc385 import MPC385
from matplotlib import pyplot as plt

def two_points(dx=100, dz=100, nsamples=1):
    """To calibrate, mechanically block the head of the cantilever at least from the bottom and the positive x direction (left). The resulting matrix allows to convert sensor measurements into micromanipulator coordinates."""
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as mpc:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        ab0 = np.mean([
            [sensor.readOne().m for sensor in sensors] 
            for i in range(nsamples)], 0)
        assert ab0.min()>0 and ab0.max()<800
        x0,y0,z0 = mpc.update_current_position()[1:]
        #move along x
        mpc.move_to(x0+mpc.um2integer_step(dx),y0,z0)
        abx = np.mean([
            [sensor.readOne().m for sensor in sensors] 
            for i in range(nsamples)], 0)
        assert abx.min()>0 and abx.max()<800
        #move along z
        mpc.move_to(x0,y0,z0+mpc.um2integer_step(dz))
        abz = np.mean([
            [sensor.readOne().m for sensor in sensors] 
            for i in range(nsamples)], 0)
        assert abz.min()>0 and abz.max()<800
        #move back to original position
        mpc.move_to(x0,y0,z0)
    #the transfer matrix from actuator to sensor coordinates is the dispacements 
    #we just measured as column vectors
    xy2ab = ((np.array([abx, abz])-ab0).T/[dx,dz])
    return np.linalg.inv(xy2ab)

def sampled_single_direction(direction='x', samples=None, repeat = 10):
    """To calibrate along x (resp z), mechanically block the head of the cantilever from the left (resp bottom). By default, test 11 dispacements sampled in log scale. 
    The resulting coefficients allows to convert micromanipulator coordinates to sensor measurements coordinates. Returns coefficients and figure testing linearity."""
    if samples is None:
        samples = 2**np.arange(11)
    measures = np.zeros((len(samples), 2))
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        x0, y0, z0 = actuator.update_current_position()[1:]
        try:
            initial = np.zeros(2)
            for j in range(repeat):
                initial += read_both(sensorA, sensorB)
            initial /= repeat
            #move along z
            for i, sample in enumerate(samples):
                if direction == "x":
                    actuator.move_to(x0 + sample, y0, z0)
                elif direction == "z":
                    #manipulator going down by sample
                    #head going up by sample since blocked
                    actuator.move_to(x0, y0, z0 + sample) 
                else:
                    raise ValueError('direction should be x or z')
                for j in range(repeat):
                    m = np.array(read_both(sensorA, sensorB))
                    if np.any(m==800) or np.any(m==0):
                        raise ValueError('Sensor out of range')
                    measures[i] += m
        finally:
            actuator.move_to(x0, y0, z0)
    measures /= repeat
    measures -= initial
    #fit the results to get coefficients
    func = lambda x,p: p * x
    dxs = -actuator.step2um(samples)
    fig, axs = plt.subplots(1,2, sharex=True)
    x2ab = []
    for ax,m in zip(axs.ravel(), (measures).T):
        ax.plot(dxs, m, 'o')
        param = curve_fit(func, dxs, m)[0][0]
        print(param)
        x2ab.append(param)
        ax.plot(dxs, func(dxs, param), ':')
    axs[0].set_title(direction+'a')
    axs[1].set_title(direction+'b')
    return np.array(x2ab), fig

def sampled(samples=None, repeat = 10):
    """To calibrate, mechanically block the head of the cantilever from the right (looking to the micromanipulator), then from the bottom. By default, test 11 dispacements in each direction, sampled in lag scale.
    The resulting matrix allows to convert sensor measurements into micromanipulator coordinates."""
    input("Please block X direction")
    x2ab = sampled_single_direction('x', samples, repeat)[0]
    input("Please block Z direction")
    z2ab = sampled_single_direction('z', samples, repeat)[0]
    return np.linalg.inv(np.column_stack((x2ab, z2ab)))
    
    
def have_moved(initial, sensors, repeat=10, precision=1):
    """Have the sensor readings changed with respect to initial"""
    #measure head position
    ab = np.zeros(2)
    for j in range(repeat):
        ab += read_both(*sensors)
    ab /= repeat
    #have we moved ?
    return np.abs(ab - initial).max() > precision
    
def find_wall(direction='z', repeat = 10, precision=1, verbose=False):
    """Find the position at which the head encounters a wall"""
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        x0, y0, z0 = actuator.update_current_position()[1:]
        try:
            initial = np.zeros(2)
            for j in range(repeat):
                initial += read_both(*sensors)
            initial /= repeat
            #estimate the maximum range
            lbound = 0
            ubound = actuator.um2integer_step(initial.min())//2
            #if sensors in the middle of their range, should be 200um
            #try to expand upper bound
            moved = False
            while not moved:
                if verbose: 
                    print("Looking between %d and %d"%(z0+lbound, z0+ubound))
                actuator.move_to(x0, y0, z0+ubound)
                moved = have_moved(initial, sensors, repeat, precision)
                if not moved:
                    lbound = ubound
                    ubound += actuator.um2integer_step(initial.min())//2
            #move back to lower bound
            actuator.move_to(x0, y0, z0+lbound)
            while ubound - lbound > actuator.um2integer_step(precision):
                if verbose:
                    print("Looking between %d and %d"%(z0+lbound, z0+ubound))
                midrange = (ubound+lbound)//2
                actuator.move_to(x0, y0, z0+midrange)
                moved = have_moved(initial, sensors, repeat, precision)
                if moved:
                    ubound = midrange
                else:
                    lbound = midrange
            touching = z0 + (ubound+lbound)//2
            print("touching at %d"%touching)
        finally:
            print("Backing up to original position")
            actuator.move_to(x0, y0, z0)
    return touching
   



