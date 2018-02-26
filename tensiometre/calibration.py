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
        mpc.move_to(x0+mpc.um2step(dx),y0,z0)
        abx = np.mean([
            [sensor.readOne().m for sensor in sensors] 
            for i in range(nsamples)], 0)
        assert abx.min()>0 and abx.max()<800
        #move along z
        mpc.move_to(x0,y0,z0+mpc.um2step(dz))
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


def log_sampled(repeat = 10):
    """To calibrate, mechanically block the head of the cantilever at least from the bottom and the positive x direction (left). Test 11 dispacements in each direction, sampled in lag scaleThe resulting matrix allows to convert sensor measurements into micromanipulator coordinates. Returns both calibration matrix and figure testing linearity."""
    measures = np.zeros((11,4))
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
            #move along x
            for i in range(len(measures)):
                actuator.move_to(x0+2**i, y0, z0)
                for j in range(repeat):
                    measures[i,:2] += np.array(read_both(sensorA, sensorB))
            actuator.move_to(x0, y0, z0)
            #move along z
            for i in range(len(measures)):
                actuator.move_to(x0, y0, z0+2**i)
                for j in range(repeat):
                    measures[i,2:] += np.array(read_both(sensorA, sensorB))
        finally:
            actuator.move_to(x0, y0, z0)
    measures /= repeat
    measures[:,:2] -= initial
    measures[:,2:] -= initial
    #fit the results to get coefficients
    dxs = actuator.step2um(2**np.arange(len(measures)))
    fig, axs = plt.subplots(2,2, sharex=True)
    func = lambda x,p: p * x
    xy2ab = []
    for ax,m in zip(axs.ravel(), (measures).T):
        ax.plot(dxs, m, 'o')
        param = curve_fit(func, dxs, m)[0][0]
        print(param)
        xy2ab.append(param)
        ax.plot(dxs, func(dxs, param), ':')
    xy2ab = np.reshape(xy2ab, (2,2)).T
    ab2xy = np.linalg.inv(xy2ab)
    return ab2xy, fig



def log_sampled_z(repeat = 10):
    """To calibrate, mechanically block the head of the cantilever at least from the bottom and the positive x direction (left). Test 11 dispacements in each direction, sampled in lag scaleThe resulting matrix allows to convert sensor measurements into micromanipulator coordinates. Returns both calibration matrix and figure testing linearity."""
    measures = np.zeros((11))
    with closing(DT3100('169.254.4.100')) as sensorA, closing(MPC385()) as actuator:
        sensors = [sensorA]
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        x0, y0, z0 = actuator.update_current_position()[1:]
        initial = 0
        for j in range(repeat):
            initial += sensorA.readOne().m
        initial /= repeat
        #move along z
        for i in range(len(measures)):
            actuator.move_to(x0, y0, z0+2**i)
            for j in range(repeat):
                measures[i] += np.array(sensorA.readOne().m)
        actuator.move_to(x0, y0, z0)
    measures /= repeat
    measures -= initial
    #fit the results to get coefficients
    dzs = actuator.step2um(2**np.arange(len(measures)))
    plt.plot(dzs, measures, 'o')
    func = lambda x,p: p * x
    param = curve_fit(func, dzs, measures)
    plt.plot(dzs, func(dzs, param)[0], ':')
    return param[0][0]
    #return b2z       



