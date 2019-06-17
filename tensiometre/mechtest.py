import time
import numpy as np
from contextlib import closing
from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID
from tensiometre import moverPID

def move_to_constant_positions(ab2xy, outnames, dxs, dys, durations, kp=0.9, moveback=False):
    """Moving the absolute position of the head by dy (depth) and dx (width) and stay at that position using PID feedback. Need ab2xy calibration matrix. Duration is in seconds. If None specified, continues until stopped."""
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        X0, Y0 = np.matmul(ab2xy, read_both(sensorA, sensorB))
        x0,y0,z0 = actuator.update_current_position()[1:]
        #setting up PID
        pids = []
        initialposition = [x0, y0] + actuator.um2step(np.array([X0, Y0]))
        for s in initialposition:
            pid = PID(kp, 0, 0)
            pid.setPoint = s
            pids.append(pid)
        try:
            for outname, dx, dy, duration in zip(outnames, dxs, dys, durations):
                for s, pid in zip(initialposition + actuator.um2step(np.array([dx,dy])), pids):
                    pid.setPoint = s
                with open(outname, "wb") as fout:
                    m = moverPID.constant_position_XY(sensors, actuator, ab2xy, pids, outputFile=fout)
                    t0 = time.time()
                    m.start()
                    try:
                        while (duration is None) or (time.time() < t0 + duration):
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                    finally:
                        #stop PID
                        m.go = False
                        m.join()
        finally:
            if moveback:
                #move the actuator back to its original position
                actuator.move_to(x0, y0, z0)
                
def add_constant_deflection(ab2xy, outnames, dXs, dYs, durations, kp=0.9, moveback=False):
    """Considering initial deflection is (0,0), successively add dX and dY to the deflection during duration and stay at this deflection using using PID feedback (constant stress). Iterate on next dX, dY, duration. Need ab2xy calibration matrix. Duration is in seconds. If None specified, continues until stopped."""

    #remember original positions of the sensors and actuator
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember initial state
        x0,y0,z0 = actuator.update_current_position()[1:]
        X0, Y0 = np.matmul(ab2xy, read_both(sensorA, sensorB))
        #setting up PID
        pids = []
        initialposition = (np.array([X0, Y0]))
        for s in initialposition:
            pid = PID(kp, 0, 0)
            pid.setPoint = s
            pids.append(pid)
        try:
            for dX, dY, duration, outname in zip(dXs, dYs, durations, outnames):
                for s, pid in zip(
                    np.array([X0,Y0]) + np.array([dX, dY]), 
                    pids
                ):
                    pid.setPoint = s
                with open(outname, "wb") as fout:
                    m = moverPID.constant_deflection_XY(
                        sensors, actuator, 
                        ab2xy, pids, outputFile=fout
                        )
                    t0 = time.time()
                    m.start()
                    try:
                        while (duration is None) or (time.time() < t0 + duration):
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                    finally:
                        #stop PID
                        m.go = False
                        m.join()
        finally:
            if moveback:
                #move the actuator back to its original position
                actuator.move_to(x0, y0, z0)

def stay_constant_deflection(outname, ab2xy, kp=0.9, duration=None, moveback=False):
    """keep the head at constant position from the sensor"""             
    move_to_constant_deflection(ab2xy, outnames = [outname], dxs = [0], dys = [0],durations=[duration], kp=kp, moveback=moveback)
    
def move_to_constant_position(outname, ab2xy, kp=0.9, dy =-100, dx=0, duration=None, moveback=False):
    """Moving the absolute position of the head by dy (depth) and dx (width) and stay at that position using PID feedback. Need ab2xy calibration matrix. Duration is in seconds. If None specified, continues until stopped."""
    move_to_constant_positions(ab2xy, outnames=[outname], dxs=[dx], dys=[dy], durations=[duration], kp=kp, moveback=moveback)
                
def stay_constant_position(outname, ab2xy, kp=0.9, duration=None, moveback=False):
    """Keep the absolute position of the head constant using PID feedback. Need ab2xy calibration matrix."""
    move_to_constant_position(outname, ab2xy, kp, dy=0, dx=0, duration=duration, moveback=False)


def traction(dy = -3000, velocity=2, duration = 1., avt=3, avn=3, outputname=None):
    """Perform a traction test, moving in y by dy (in microns, pointing down) at a velocity imposed by actuator. Aquires in avt avn mode for a fixed duration."""
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
                actuator.move_straight(x0, y0 + dy*16, z0, velocity)
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
    """Perform a shear test, moving in x by dx (in microns, pointing left) at a velocity imposed by actuator. Aquires in avt avn mode for a fixed duration."""
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
            
