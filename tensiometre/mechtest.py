import numpy as np
from contextlib import closing
from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
from tensiometre.mpc385 import MPC385

def move_to_constant_position(outname, ab2xz, kp=0.9, dz =-100, dx=0, moveback=False):
    """Moving the absolute position of the head by dz and dx and stay at that position using PID feedback. Need ab2xz calibration matrix."""
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        X0, Z0 = np.matmul(ab2xz, read_both(sensorA, sensorB))
        x0,y0,z0 = actuator.update_current_position()[1:]
        #setting up PID
        pids = []
        for s in [x0, z0] - actuator.um2step(np.array([X0+dx, Z0+dz])):
            pid = PID(kp, 0, 0)
            pid.setPoint = s
            pids.append(pid)
        try:
            with open(outname, "wb") as fout:
                m = moverPID.constant_position_XZ(sensors, actuator, ab2xz, pids, outputFile=fout)
                m.start()
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                #stop PID
                m.go = False
                m.join()
        finally:
            if moveback:
                #move the actuator back to its original position
                actuator.move_to(x0, y0, z0)
                
def stay_constant_position(outname, ab2xz, kp=0.9, moveback=False):
    """Keep the absolute position of the head constant using PID feedback. Need ab2xz calibration matrix."""
    move_to_constant_position(outname, ab2xz, kp, dz=0, dx=0, moveback=False)


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
            
