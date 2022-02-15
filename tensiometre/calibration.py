from contextlib import closing
import time
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from contextlib import closing, ExitStack
from tensiometre.dt3100 import DT3100, ReadOne, read_both, recover
from tensiometre.mpc385 import MPC385
from tensiometre.show_measurements import show_measurement
from matplotlib import pyplot as plt

def two_points(dx=100, dy=100, nsamples=1):
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
        #move along x (width)
        mpc.move_to(x0+mpc.um2integer_step(dx),y0,z0)
        abx = np.mean([
            [sensor.readOne().m for sensor in sensors]
            for i in range(nsamples)], 0)
        assert abx.min()>0 and abx.max()<800
        #move along y (depth)
        mpc.move_to(x0,y0+mpc.um2integer_step(dy),z0)
        abz = np.mean([
            [sensor.readOne().m for sensor in sensors]
            for i in range(nsamples)], 0)
        assert abz.min()>0 and abz.max()<800
        #move back to original position
        mpc.move_to(x0,y0,z0)
    #the transfer matrix from actuator to sensor coordinates is the dispacements
    #we just measured as column vectors
    xy2ab = ((np.array([abx, abz])-ab0).T/[dx,dy])
    return np.linalg.inv(xy2ab)

def sampled_single_direction(direction='x', samples=None, repeat = 10, axs=None):
    """To calibrate along x (resp y), mechanically block the head of the cantilever from the left (resp bottom). By default, test 11 dispacements sampled in log scale.
    The resulting coefficients allows to convert micromanipulator coordinates to sensor measurements coordinates. Returns coefficients."""
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
                elif direction == "y":
                    #manipulator going down by sample
                    #head going up by sample since blocked
                    actuator.move_to(x0, y0 - sample, z0)
                else:
                    raise ValueError('direction should be x or y')
                for j in range(repeat):
                    m = np.array(read_both(sensorA, sensorB))
                    if np.any(m==800) or np.any(m==0):
                        raise ValueError('Sensor out of range')
                    measures[i] += m
        finally:
            actuator.move_to(x0, y0, z0)
    measures /= repeat
    measures -= initial
    np.save('calib_' + direction + '_' + datetime.now().strftime('%Y%m%d_%H%M_calib.npy'), measures)
    #fit the results to get coefficients
    func = lambda x,p: p * x
    if direction == "x":
        dxs = -actuator.step2um(samples)
    elif direction == "y":
        dxs = actuator.step2um(samples)
    else:
        raise ValueError('direction should be x or y')
    if axs is None:
        fig, axs = plt.subplots(1,2, sharex=True)
    x2ab = []
    for ax,m in zip(axs.ravel(), (measures).T):
        ax.plot(dxs, m, 'o')
        param = curve_fit(func, dxs, m)[0][0]
        print(param)
        x2ab.append(param)
        ax.plot(dxs, func(dxs, param), ':')
    #axs[0].set_title(direction+'a')
    #axs[1].set_title(direction+'b')
    axs[0].set_xlabel(direction)
    axs[0].set_ylabel('a')
    axs[1].set_xlabel(direction)
    axs[1].set_ylabel('b')
    return np.array(x2ab)

def sampled_direction(direction='y',samples=None, repeat = 10):
    """To calibrate along z (resp y) the 3rd sensor, By default, test 11 dispacements sampled in log scale.
    Added the option for direction to check how movement in other direction affects the sensor reading
    The resulting coefficients allows to convert micromanipulator coordinates to sensor measurements coordinates.
    Returns coefficients and figure testing linearity."""
    if samples is None:
        samples = 2**np.arange(11)
    measures = np.zeros((len(samples)))
    with closing(DT3100('169.254.5.100')) as sensorC, closing(MPC385()) as actuator:
        sensors = [sensorC]
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        x0, y0, z0 = actuator.update_current_position()[1:]
        try:
            initial = np.zeros(1)
            for j in range(repeat):
                initial += sensorC.readOne().m #why .m
            initial /= repeat
            #move along z
            for i, sample in enumerate(samples):
                    #manipulator going up
                if direction == "y":
                    actuator.move_to(x0, y0 + sample, z0)
                elif direction == "x":
                    actuator.move_to(x0+ sample, y0, z0)
                else:
                    actuator.move_to(x0, y0, z0 + sample)
                for j in range(repeat):
                    m = sensorC.readOne().m
                    if np.any(m==800) or np.any(m==0):
                        raise ValueError('Sensor out of range')
                    measures[i] += m
        finally:
            actuator.move_to(x0, y0, z0)
    measures /= repeat
    measures -= initial
    np.save('calib_sensorC_' + '_' + datetime.now().strftime('%Y%m%d_%H%M_calib.npy'), measures)
    #fit the results to get coefficients
    func = lambda x,p: p * x
    dxs = actuator.step2um(samples)
    fig, axs = plt.subplots(1,1)
    x2c = []
    #for ax,m in zip(axs, (measures).T):
    axs.plot(dxs, measures, 'o')
    param = curve_fit(func, dxs, measures)[0][0]
    print(param)
    x2c.append(param)
    axs.plot(dxs, func(dxs, param), ':')
    axs.set_xlabel(direction + ' manip (µm)')
    axs.set_ylabel('Sensor c (µm)')
    axs.set_title(direction + '_C')
    return np.array(x2c), fig

def sampled(samples=None, repeat = 10, axs=None):
    """To calibrate, mechanically block the head of the cantilever from the right (looking to the micromanipulator), then from the bottom. By default, test 11 dispacements in each direction, sampled in log scale.
    The resulting matrix allows to convert sensor measurements into micromanipulator coordinates."""
    if axs is None:
        fig, axs = plt.subplots(2,2, sharex='col', sharey='row')
    input("Please block lateral direction")
    x2ab = sampled_single_direction('x', samples, repeat, axs=axs[:,0])
    input("Please block depth direction")
    y2ab = sampled_single_direction('y', samples, repeat, axs=axs[:,1])
    return np.linalg.inv(np.column_stack((x2ab, y2ab)))

def sampled_auto(samples=None, repeat = 10, settle_time=1):
    """To calibrate, mechanically block the head of the cantilever from the right (looking to the micromanipulator), then from the bottom. By default, test 11 dispacements in each direction, sampled in lag scale.
    note: For the borrowed micromanipulator, block from left when looking at head from front
    The resulting matrix allows to convert sensor measurements into micromanipulator coordinates."""
    with closing(MPC385()) as actuator:
        x0, y0, z0 = actuator.update_current_position()[1:]
    x = find_wall('x', repeat, precision=1, verbose=True, settle_time=settle_time, backup=False)
    time.sleep(1)
    x2ab = sampled_single_direction('x', samples, repeat)[0]
    time.sleep(1)
    with closing(MPC385()) as actuator:
        actuator.move_to(x0, y0, z0)
    y = find_wall('y', repeat, precision=1, verbose=True, settle_time=settle_time, backup=False)
    #with closing(MPC385()) as actuator:
        #actuator.move_to(x0, y0+y+100*16, z0)
    time.sleep(1)
    y2ab = sampled_single_direction('y', samples, repeat)[0]
    time.sleep(1)
    with closing(MPC385()) as actuator:
        actuator.move_to(x0, y0, z0)
    return np.linalg.inv(np.column_stack((x2ab, y2ab)))

def interactive(samples=None, repeat = 10, axs=None):
    """To calibrate, mechanically block the head of the cantilever from the right (looking to the micromanipulator), then from the bottom. By default, test 11 dispacements in each direction, sampled in log scale.
    The resulting matrix allows to convert sensor measurements into micromanipulator coordinates."""
    if axs is None:
        fig, axs = plt.subplots(2,2, sharex='col', sharey='row')
    else:
        fig = axs[0,0].figure
    plt.close(fig)
    xy2ab = []
    for direction, axss, dirname in zip('xy', axs.T, ['lateral', 'depth']):
        while True:
            for i in range(3):
                recover(f'169.254.{3+i}.100')
            print(f"Please block {dirname} direction. Close figure window when OK.")
            with ExitStack() as stack:
                sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
                for sensor in sensors:
                    sensor.set_averaging_type(3)
                    sensor.set_averaging_number(3)
                show_measurement(sensors, 1, ymin=-80)
            print(f"Performing measurments in {dirname} direction.")
            for ax in axss:
                ax.clear()
            dir2ab = sampled_single_direction(direction, samples, repeat, axs=axss)
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            plt.show(block=False)
            if(input('Are you happy with the results? (N)') in ['Y', 'y', 'yes', 'Yes', 'oui']):
                xy2ab.append(dir2ab)
                break
            plt.close(dummy)
    return np.linalg.inv(np.column_stack((xy2ab)))


def have_moved(initial, sensors, repeat=10, precision=1, settle_time=None):
    """Have the sensor readings changed with respect to initial"""
    if settle_time is not None:
        time.sleep(settle_time)
    #measure head position
    ab = np.zeros(2)
    for j in range(repeat):
        ab += read_both(*sensors)
    ab /= repeat
    #have we moved ?
    return np.abs(ab - initial).max() > precision

def find_wall(direction='y', repeat = 10, precision=1, verbose=False, backup=True, settle_time=None, k=100*16):
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
                if direction == "x":
                    if verbose:
                        print("Looking between %d and %d in width"%(x0+lbound, x0+ubound))
                    actuator.move_to(x0+ubound, y0, z0)
                elif direction == "y":
                    if verbose:
                        print("Looking between %d and %d in depth"%(y0-ubound, y0-lbound))
                    actuator.move_to(x0, y0-ubound, z0)
                moved = have_moved(initial, sensors, repeat, precision, settle_time)
                if not moved:
                    lbound = ubound
                    ubound += actuator.um2integer_step(initial.min())//2
            #move back to lower bound
            if direction == "x":
                print("excessive touching")
                actuator.move_to(x0+lbound, y0, z0)
            elif direction == "y":
                print("excessive touching")
                actuator.move_to(x0, y0-lbound, z0)
            #actuator.move_to(x0, y0+lbound, z0)
            while ubound - lbound > actuator.um2integer_step(precision):
                midrange = (ubound+lbound)//2
                if direction == "x":
                    if verbose:
                        print("Looking between %d and %d in width"%(x0+lbound, x0+ubound))
                    actuator.move_to(x0+midrange, y0, z0)
                elif direction == "y":
                    if verbose:
                        print("Looking between %d and %d in depth"%(y0-ubound, y0-lbound))
                    actuator.movadjustmente_to(x0, y0-midrange, z0)
                moved = have_moved(initial, sensors, repeat, precision, settle_time)
                if moved:
                    ubound = midrange
                else:
                    lbound = midrange
            touching = (ubound+lbound)//2
            if direction == "x":
                touching += x0
                actuator.move_to(touching, y0, z0)
            elif direction == "y":
                touching = y0 - touching
                actuator.move_to(x0, touching, z0)
                #k += touching
                #actuator.move_to(x0, k, z0)
            print("touching at %d"%touching)
            #print("moved actuator to %d"%k)
        finally:
            if backup:
                print("Backing up to original position")
                actuator.move_to(x0, y0, z0)
    return touching
