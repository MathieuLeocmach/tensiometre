import numpy as np
from contextlib import closing
from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
from tensiometre.mpc385 import MPC385


def traction(dz = -3000, velocity=2, duration = 1., avt=3, avn=3, outputname=None):
    """Perform a traction test, moving in z by dz (in microns, pointing down) at a velocity imposed by actuator. Aquires in avt avn mode for a fixed duration."""
    if outputname is None:
        outputname = 'traction_v%d_fast'%velocity
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB,closing(MPC385()) as actuator:
        for sensor in (sensorA, sensorB):
            sensor.set_averaging_type(avt)
            sensor.set_averaging_number(avn)
        x0, y0, z0 = actuator.update_current_position()[1:]
        try:
            with open(outputname+'A.raw', 'wb') as outputFileA, open(outputname+'B.raw', 'wb') as outputFileB:
                #get ready to measure
                sa = ReadDuration(sensorA, outputFileA, duration)
                sb = ReadDuration(sensorB, outputFileB, duration)
                #start measurement
                sa.start()
                sb.start()
                #move
                actuator.move_straight(x0, y0, z0 + dz*16, velocity)
                sa.join()
                sb.join()
            nsamples = int(duration / sensorA.unit_time().m)
            valuesA = np.fromfile(outputname+'A.raw')[:nsamples]
            valuesB = np.fromfile(outputname+'B.raw')[:nsamples]
            t = np.arange(nsamples) * sensorA.unit_time().m
            np.save(outputname+'.npy', np.column_stack((t,valuesA, valuesB)))
        finally:
            actuator.move_to(x0, y0, z0)


def shear(dx = 3000, velocity=2, duration = 1., avt=3, avn=3, outputname=None):
    """Perform a shear test, moving in x by dx (in microns, pointing down) at a velocity imposed by actuator. Aquires in avt avn mode for a fixed duration."""
    if outputname is None:
        outputname = 'traction_v%d_fast'%velocity
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB,closing(MPC385()) as actuator:
        for sensor in (sensorA, sensorB):
            sensor.set_averaging_type(avt)
            sensor.set_averaging_number(avn)
        x0, y0, z0 = actuator.update_current_position()[1:]
        try:
            with open(outputname+'A.raw', 'wb') as outputFileA, open(outputname+'B.raw', 'wb') as outputFileB:
                #get ready to measure
                sa = ReadDuration(sensorA, outputFileA, duration)
                sb = ReadDuration(sensorB, outputFileB, duration)
                #start measurement
                sa.start()
                sb.start()
                #move
                actuator.move_straight(x0 + dx*16, y0, z0 , velocity)
                sa.join()
                sb.join()
            nsamples = int(duration / sensorA.unit_time().m)
            valuesA = np.fromfile(outputname+'A.raw')[:nsamples]
            valuesB = np.fromfile(outputname+'B.raw')[:nsamples]
            t = np.arange(nsamples) * sensorA.unit_time().m
            np.save(outputname+'.npy', np.column_stack((t,valuesA, valuesB)))
        finally:
            actuator.move_to(x0, y0, z0)
            
