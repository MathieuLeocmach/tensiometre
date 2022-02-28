import time
import numpy as np
from contextlib import closing, ExitStack
from tensiometre.dt3100 import DT3100, ReadOne, recover, read_both, ReadDuration
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID
from tensiometre import moverPID

class State:
    """A state of the tensiometer, containing both actuator position and deflection. All in microns."""
    def __init__(self):
        self.actuator_pos = np.zeros(3)
        self.deflection = np.zeros(2)
        self.y_ag = 0

    def read(self, sensors, actuator, ab2xy):
        """Read the current state from sensors and actuator"""
        self.actuator_pos[:], self.deflection[:], self.y_ag = moverPID.current_positions(ab2xy, sensors, actuator)
        self.actuator_pos[:] = actuator.step2um(self.actuator_pos)
        return self

    def load(self, filename):
        """Load state from a file"""
        a = np.load(filename)
        self.actuator_pos[:] = a[:3]
        self.deflection[:] = a[3:5]
        self.y_ag = a[5]
        return self

    def save(self, filename):
        """Save state to a file"""
        np.save(filename, np.concatenate((self.actuator_pos, self.deflection, [self.y_ag])))

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
    else:
        kps = kp
    if not hasattr(ki, "__len__"):
        kis = [ki,ki]
    else:
        kis = ki
    if not hasattr(kd, "__len__"):
        kds = [kd,kd]
    else:
        kds = kd
    assert len(kps) == 2
    assert len(kis) == 2
    assert len(kds) == 2
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
        actuator = stack.enter_context(closing(MPC385()))
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State().read(sensors, actuator, ab2xy)
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
    else:
        kps = kp
    if not hasattr(ki, "__len__"):
        kis = [ki,ki]
    else:
        kis = ki
    if not hasattr(kd, "__len__"):
        kds = [kd,kd]
    else:
        kds = kd
    assert len(kps)==2
    assert len(kis)==2
    assert len(kds)==2
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
        actuator = stack.enter_context(closing(MPC385()))
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State().read(sensors, actuator, ab2xy)
        #setting up PID, in microns
        pids = []
        initialposition = state0.deflection
        for s, kp,ki,kd in zip(initialposition, kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            for dX, dY, duration, outname in zip(dXs, dYs, durations, outnames):
                for s, pid in zip(
                    state0.deflection + np.array([dX, dY]),
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
    else:
        kps = kp
    if not hasattr(ki, "__len__"):
        kis = [ki,ki]
    else:
        kis = ki
    if not hasattr(kd, "__len__"):
        kds = [kd,kd]
    else:
        kds = kd

    #remember original positions of the sensors and actuator
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
        actuator = stack.enter_context(closing(MPC385()))
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State().read(sensors, actuator, ab2xy)
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


def timedep_position(ab2xy, outname, functions, duration=None, kp=0.9,ki = 0.0, kd =0.0,  moveback=False, state0=None):
    """Moving the absolute position of the head in depth and width following two
    time-dependent functions around state0.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    if not hasattr(kp, "__len__"):
        kps = [kp, kp]
    else:
        kps = kp
    if not hasattr(ki, "__len__"):
        kis = [ki,ki]
    else:
        kis = ki
    if not hasattr(kd, "__len__"):
        kds = [kd,kd]
    else:
        kds = kd
    assert len(kps) == 2
    assert len(kis) == 2
    assert len(kds) == 2
    assert len(functions) == 2
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
        actuator = stack.enter_context(closing(MPC385()))
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State().read(sensors, actuator, ab2xy)
        #setting up PID, in microns
        pids = []
        initialposition = state0.head_to_ground
        for s, kp,ki,kd in zip(initialposition, kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            with open(outname, "wb") as fout:
                m = moverPID.constant_position_XY_3sensors(sensors, actuator, ab2xy, pids, outputFile=fout)
                t0 = time.time()
                m.start()
                try:
                    t = time.time() - t0
                    while (duration is None) or (t < duration):
                        t = time.time() - t0
                        for pid, p0, function in zip(pids, initialposition, functions):
                            pid.setPoint = p0 + function(t)
                        time.sleep(0.001) #no use to update above 1kHz
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

def oscillating_position(ab2xy, outname, amplitudex, amplitudey=0, freqx=10, freqy=0, duration=None, kp=0.9,ki = 0.0, kd =0.0,  moveback=False, state0=None):
    """Moving the absolute position of the head in depth and width following two
    sine functions around state0.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    functionx = lambda t: amplitudex * np.sin(2*np.pi*freqx*t)
    functiony = lambda t: amplitudey * np.sin(2*np.pi*freqy*t)
    timedep_position(
        ab2xy, outname, functions=[functionx, functiony], duration=duration,
        kp=kp,ki = ki, kd =kd, moveback=moveback, state0=state0
        )

def timedep_armX_positionY(ab2xy, outname, functions, duration=None, kp=0.9,ki = 0.0, kd =0.0,  moveback=False, state0=None):
    """Moving the absolute position of the head in depth (Y) and the arm position in Y following two
    time-dependent functions around state0.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    if not hasattr(kp, "__len__"):
        kps = [kp, kp]
    else:
        kps = kp
    if not hasattr(ki, "__len__"):
        kis = [ki,ki]
    else:
        kis = ki
    if not hasattr(kd, "__len__"):
        kds = [kd,kd]
    else:
        kds = kd
    assert len(kps) == 2
    assert len(kis) == 2
    assert len(kds) == 2
    assert len(functions) == 2
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{3+i}.100'))) for i in range(3)]
        actuator = stack.enter_context(closing(MPC385()))
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        if state0 is None:
            #remember original positions of the sensors and actuator
            state0 = State().read(sensors, actuator, ab2xy)
        #setting up PID, in microns
        pids = []
        initialposition = [state0.arm_to_ground[0], state0.head_to_ground[1]]
        for s, kp,ki,kd in zip(initialposition, kps,kis,kds):
            pid = PID(kp, ki, kd)
            pid.setPoint = s
            pids.append(pid)
        try:
            with open(outname, "wb") as fout:
                m = moverPID.armX_position_Y_3sensors(sensors, actuator, ab2xy, pids, outputFile=fout)
                t0 = time.time()
                m.start()
                try:
                    t = time.time() - t0
                    while (duration is None) or (t < duration):
                        t = time.time() - t0
                        for pid, p0, function in zip(pids, initialposition, functions):
                            pid.setPoint = p0 + function(t)
                        time.sleep(0.001) #no use to update above 1kHz

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

def oscillating_armX_positionY(ab2xy, outname, amplitudex, amplitudey=0, freqx=10, freqy=0, duration=None, kp=0.9,ki = 0.0, kd =0.0,  moveback=False, state0=None):
    """Moving the absolute position of the head in depth and width following two
    sine functions around state0.
    Need ab2xy calibration matrix.
    Duration is in seconds. If None specified, continues until stopped."""
    functionx = lambda t: amplitudex * np.sin(2*np.pi*freqx*t)
    functiony = lambda t: amplitudey * np.sin(2*np.pi*freqy*t)
    timedep_armX_positionY(
        ab2xy, outname, functions=[functionx, functiony], duration=duration,
        kp=kp,ki = ki, kd =kd, moveback=moveback, state0=state0
        )
