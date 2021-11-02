import time
import numpy as np
from contextlib import closing
from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID
from tensiometre import moverPID

class State:
    """A state of the tensiometer, containing both actuator position and deflection. All in microns."""
    def __init__(self, sensors, actuator, ab2xy):
        self.actuator_pos, self.deflection, self.y_ag = moverPID.current_positions(ab2xy, sensors, actuator)
        self.actuator_pos = actuator.step2um(np.array(list(self.actuator_pos)))

    @property
    def deflection_real(self):
        return np.array([self.deflection[0], self.deflection[1] + self.y_ag -self.actuator_pos[1]])

    @property
    def arm_to_ground(self):
        return np.array([self.actuator_pos[0], self.y_ag])

    @property
    def head_to_ground(self):
        return self.arm_to_ground + self.deflection


def move_to_constant_positions(ab2xy, outnames, dxs, dys, durations, kp=0.9,ki = 0.0, kd =0.0,  moveback=False, state0=None):
    """Moving the absolute position of the head by dy (depth) and dx (width)
    and stay at that position using PID feedback.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    if not hasattr(kp, "__len__"):
        kps = [kp, kp]
        kis = [ki,ki]
        kds = [kd,kd]
    else:
        kps = kp
        kis = ki
        kds = kd
    assert len(kps) == 2
    assert len(kis) == 2
    assert len(kds) == 2
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(DT3100('169.254.5.100')) as sensorC, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB, sensorC]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State(sensors, actuator, ab2xy)
        #setting up PID, in microns
        pids = []
        initialposition = state0.head_to_ground
        for s, kp,ki,kd in zip(initialposition, kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            for outname, dx, dy, duration in zip(outnames, dxs, dys, durations):
                for s, pid in zip(initialposition + np.array([dx,dy]), pids):
                    pid.setPoint = s
                with open(outname, "wb") as fout:
                    m = moverPID.constant_position_XY_3sensors(sensors, actuator, ab2xy, pids, outputFile=fout)
                    t0 = time.time()
                    m.start()
                    try:
                        while (duration is None) or (time.time() < t0 + duration):
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                    finally:
                        #stop PID thread
                        m.go = False
                        m.join()
        finally:
            if moveback:
                #move the actuator back to its original position
                actuator.move_to(*actuator.um2integer_step(state0.actuator_pos))

def add_constant_deflection(ab2xy, outnames, dXs, dYs, durations, kp=0.9,ki = 0.0, kd =0.0, moveback=False, delay=0, state0=None):
    """Considering initial deflection is 0 in x
    and successively add dX to the deflection during duration
    and stay at this deflection using PID feedback (constant stress).
    Iterate on next dX, dY, duration.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""

    #remember original positions of the sensors and actuator
    if not hasattr(kp, "__len__"):
        kps = [kp, kp]
        kis = [ki,ki]
        kds = [kd,kd]
    else:
        kps = kp
        kis = ki
        kds = kd
    assert len(kps)==2
    assert len(kis)==2
    assert len(kds)==2
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(DT3100('169.254.5.100')) as sensorC, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB, sensorC]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State(sensors, actuator, ab2xy)
        #setting up PID, in microns
        pids = []
        initialposition = state0.deflection_real
        for s, kp,ki,kd in zip(initialposition, kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            for dX, dY, duration, outname in zip(dXs, dYs, durations, outnames):
                for s, pid in zip(
                    state0.deflection_real + np.array([dX, dY]),
                    pids
                ):
                    pid.setPoint = s
                with open(outname, "wb") as fout:
                    m = moverPID.constant_deflection_XY_3sensors(
                        sensors, actuator,
                        ab2xy, pids, outputFile=fout, delay=delay,
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
                actuator.move_to(*actuator.um2integer_step(state0.actuator_pos))

def add_constant_deflectionX_move_to_constant_positiony(ab2xy, outnames, dXs, dys, durations, kp=0.9,ki = 0.0, kd =0.0, moveback=False, maxYdispl=None, state0=None):
    """Considering initial deflection is (0,0),
    successively add dX to the deflection and dy to absolute position
    and stay at this constant X deflection and constant y position during duration.
    Stay at this deflection and positon using using PID feedback (constant stress and constant strain respectively).
    Iterate on next dX, duration.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    if not hasattr(kp, "__len__"):
        kps = [kp, kp]
        kis = [ki,ki]
        kds = [kd,kd]
    else:
        kps = kp
        kis = ki
        kds = kd

    #remember original positions of the sensors and actuator
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(DT3100('169.254.5.100')) as sensorC, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB, sensorC]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State(sensors, actuator, ab2xy)
        #setting up PID
        pids = []
        for s,kp,ki,kd in zip([state0.deflection[0], state0.head_to_ground[1]],kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            for outname, dX, dy, duration in zip(outnames, dXs, dys, durations):

                ip = [state0.deflection[0] + dX, state0.head_to_ground[1] + dy]
                for s, pid in zip(ip, pids):
                    pid.setPoint = s
                with open(outname, "wb") as fout:
                    m = moverPID.constant_deflectionX_positionY_3sensors(sensors, actuator, ab2xy, pids, outputFile=fout)
                    t0 = time.time()
                    m.start()
                    try:
                        while (duration is None) or (time.time() < t0 + duration):
                            time.sleep(1) #is it too long ?
                            x,y,z = m.xyz
                            y0 = actuator.um2integer_step(state0.actuator_pos[1])
                            if maxYdispl is not None and abs(y - y0) > actuator.um2step(maxYdispl):
                                break
                    except KeyboardInterrupt:
                        pass
                    finally:
                        #stop PID thread
                        m.go = False
                        m.join()
        finally:
            if moveback:
                #move the actuator back to its original position
                actuator.move_to(*actuator.um2integer_step(state0.actuator_pos))

def add_constant_deflectionX_stay_constant_positiony(outname, ab2xy,kp=0.9,ki = 0.0, kd = 0.0, dX=30, dy=0, duration=None, moveback=False, maxYdispl=None, state0=None):
    """Add a constant deflection of dx while staying at the same absolute position in y"""
    add_constant_deflectionX_move_to_constant_positiony(
        ab2xy,outnames = [outname], dXs = [dX], dys=[dy], durations=[duration],
        kp=kp,ki=ki, kd=kd, moveback=moveback, maxYdispl= maxYdispl, state0=state0
        )

def stay_constant_deflection(outname, ab2xy, kp=0.9,ki = 0.0, kd = 0.0, duration=None, moveback=False, delay = 0, state0=None):
    """keep the head at constant position from the sensor"""
    add_constant_deflection(
        ab2xy, outnames = [outname], dXs = [0], dYs = [0],durations=[duration],
        kp=kp,ki=ki, kd=kd, moveback=moveback, delay= delay, state0=state0
        )

def move_to_constant_position(outname, ab2xy, kp=0.9,ki = 0.0, kd = 0.0, dy =-100, dx=0, duration=None, moveback=False, state0=None):
    """Moving the absolute position of the head by dy (depth) and dx (width) and stay at that position using PID feedback. Need ab2xy calibration matrix. Duration is in seconds. If None specified, continues until stopped."""
    move_to_constant_positions(
        ab2xy, outnames=[outname], dxs=[dx], dys=[dy], durations=[duration],
        kp=kp,ki=ki, kd=kd, moveback=moveback, state0=state0
        )

def stay_constant_position(outname, ab2xy, kp=0.9, ki = 0.0, kd = 0.0, duration=None, moveback=False, state0=None):
    """Keep the absolute position of the head constant using PID feedback. Need ab2xy calibration matrix."""
    move_to_constant_position(
        outname, ab2xy, kp,ki,kd, dy=0, dx=0, duration=duration,
        moveback=False, state0=state0
        )
