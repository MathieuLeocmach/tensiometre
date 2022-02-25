import time
from contextlib import closing
from threading import Thread
import numpy as np
from tensiometre.dt3100 import DT3100, ReadOne
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID

def current_positions(ab2xy, sensors, actuator):
    """Ask all sensor readings and actuator current position. The two first sensors are converted according to ab2xy."""
    #ask asynchronously for a position measurement for each sensor
    readers = [ReadOne(sensor) for sensor in sensors]
    for reader in readers:
        reader.start()
    #in parallel, ask for the current position of the micromanipulator
    x,y,z = actuator.update_current_position()[-3:]
    #wait for the position measurements
    for reader in readers:
        reader.join()
    #translate sensor coordinates into micromanipulator coordinates
    measureXY = np.matmul(ab2xy, [reader.value.m for reader in readers[:2]])
    return ((x,y,z), measureXY,) + tuple(reader.value.m for reader in readers[2:])


class MoverPID_Y(Thread):
    """Thread in charge of controlling the Y position of the tensiometer (depth) so that the target stays at a given distance from the sensor (constant force)."""
    def __init__(self, sensor, actuator, pid=PID(), outputFile=False):
        Thread.__init__(self)
        self.sensor = sensor
        self.actuator = actuator
        self.pid = pid
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            #ask asynchronously for a position measurement
            reader = ReadOne(self.sensor)
            reader.start()
            #in parallel, ask for the current position of the micromanipulator
            x,y,z = self.actuator.update_current_position()[-3:]
            #wait for the position measurement
            reader.join()
            measure = reader.value.m
            #feed to PID
            self.pid.update(measure)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pid.current_time-t0, measure, self.pid.output])
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newy = y - self.actuator.um2step(self.pid.output)
            newpos = self.actuator.truncate_steps(np.array(x, newy, z))
            #update micromanipulator position
            if newpos[1] != y:
                self.actuator.move_to(*newpos)
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()


class constant_deflection_XY(Thread):
    """Thread in charge of controlling the x (width) and y (depth) position of the micromanipulator so that the target stays at a given distances from the sensors (constant force)."""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False, delay=0):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        assert len(pids) == 2, "Two PIDs are needed"
        self.pids = pids
        self.delay = delay
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY = current_positions(self.AB2XY, self.sensors, self.actuator)
            #feed to PIDs
            outputs = []
            for measure, pid in zip(measureXY, self.pids):
                #PID works in microns
                pid.update(measure)
                outputs.append(pid.output)
            outputs = np.array(outputs)
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newxy = (np.array([x,y]) - self.actuator.um2integer_step(outputs)).astype(int)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([pid.current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY])
            #newxy = np.minimum(self.actuator._NSTEP, np.maximum(0, newxy))
            newxy = self.actuator.truncate_steps(newxy)
            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
            time.sleep(self.delay)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()

class constant_position_XY(Thread):
    """Thread in charge of controlling the X (width) and Y (depth) position of the tensiometer so that the head stays at the same absolute position."""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        self.pids = pids
        assert len(self.pids) == 2
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY = current_positions(self.AB2XY, self.sensors, self.actuator)
            #feed state to PIDs, in microns
            outputs = []
            for measure, pid in zip(self.actuator.step2um(np.array([x,y])) + measureXY, self.pids):
                #PID works in microns
                pid.update(measure)
                outputs.append(pid.output)
            outputs = np.array(outputs)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([pid.current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY])
            #the new position of the micromanipulator in steps
            newxy = self.actuator.um2integer_step(outputs) + [x,y]
            newxy = self.actuator.truncate_steps(newxy)
            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()


class constant_deflectionX_positionY(Thread):
    """Thread in charge of controlling the x (width) and y (depth) position of the micromanipulator so that deflection stays constant in x (constant force) and the y position of the head stays constant with respect to the lab (constant position)."""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        self.pids = pids
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY = current_positions(self.AB2XY, self.sensors, self.actuator)
            #save current position of the micromanipulator for external access
            self.xyz = (x,y,z)
            #feed to PIDs
            outputs = []
            #PID on X works in microns
            self.pids[0].update(measureXY[0])
            outputX = self.pids[0].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newx = x - self.actuator.um2integer_step(outputX)
            #PID on y works in microns
            self.pids[1].update(self.actuator.step2um(y) + measureXY[1])
            outputy = self.pids[1].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newy = y + self.actuator.um2integer_step(outputy)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pids[0].current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY])
            newxy = np.array((newx,newy))
            newxy = self.actuator.truncate_steps(newxy)

            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()

class constant_position_XY_3sensors(Thread):
    """Thread in charge of controlling the X (width) and Y (depth) position of
the tensiometer so that the head stays at the same absolute position.
The y position is obtained from a third sensor between the arm and the tank (ground).
"""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        assert len(sensors)==3, "Three sensors are needed"
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        self.pids = pids
        assert len(self.pids) == 2
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY, y_ag = current_positions(self.AB2XY, self.sensors, self.actuator)
            #feed state to PIDs, in microns
            #in x, we beleive the micromanipulator reading
            self.pids[0].update(self.actuator.step2um(x) + measureXY[0])
            #in y we beleive the third sensor reading
            self.pids[1].update(y_ag + measureXY[1])
            outputs = np.array([pid.output for pid in self.pids])
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pids[0].current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY, y_ag])
            #the new position of the micromanipulator in steps
            newxy = self.actuator.um2integer_step(outputs) + [x,y]
            newxy = self.actuator.truncate_steps(newxy)
            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()

class constant_deflectionX_positionY_3sensors(Thread):
    """Thread in charge of controlling the x (width) and y (depth) position of
the micromanipulator so that deflection stays constant in x (constant force) and
the y position of the head stays constant with respect to the lab (constant
position).
The y position is obtained from a third sensor between the arm and the tank (ground).
"""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        assert len(sensors)==3, "Three sensors are needed"
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        self.pids = pids
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY, y_ag = current_positions(self.AB2XY, self.sensors, self.actuator)
            #save current position of the micromanipulator for external access
            self.xyz = (x,y,z)
            #feed to PIDs
            outputs = []
            #PID on X works in microns
            self.pids[0].update(measureXY[0])
            outputX = self.pids[0].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newx = x - self.actuator.um2integer_step(outputX)
            #PID on y works in microns
            self.pids[1].update(measureXY[1] + y_ag)
            outputy = self.pids[1].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newy = y + self.actuator.um2integer_step(outputy)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pids[0].current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY, y_ag])
            newxy = np.array((newx,newy))
            newxy = self.actuator.truncate_steps(newxy)

            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()

class constant_deflection_XY_3sensors(Thread):
    """Thread in charge of controlling the x (width) and y (depth) position of the micromanipulator so that the target stays at a given distances from the sensors (constant force)."""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False, delay=0):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        assert len(sensors)==3, "Three sensors are needed"
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        assert len(pids) == 2, "Two PIDs are needed"
        self.pids = pids
        self.delay = delay
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY, y_ag = current_positions(self.AB2XY, self.sensors, self.actuator)
            #feed to PIDs
            self.pids[0].update(measureXY[0])
            outputX = self.pids[0].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newx = x - self.actuator.um2integer_step(outputX)
            #PID on y works in microns
            self.pids[1].update(measureXY[1])
            outputy = self.pids[1].output
            #translate PID output in microns into
            #the new position of the micromanipulator in steps
            newy = y - self.actuator.um2integer_step(outputy)
            newxy = np.array((newx,newy))

            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pids[0].current_time-t0, x, y, *self.actuator.um2step(measureXY), y_ag])
            #newxy = np.minimum(self.actuator._NSTEP, np.maximum(0, newxy))
            newxy = self.actuator.truncate_steps(newxy)
            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
            time.sleep(self.delay)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()

class armX_position_Y_3sensors(Thread):
    """Thread in charge of controlling the controller position of so that :
 - the actuator position in X follows a (time dependent) setpoint
 - the absolute position of the head in Y follows (time dependent) setpoint.
The y position is obtained from a third sensor between the arm and the tank (ground).
"""
    def __init__(self, sensors, actuator, AB2XY, pids=[PID(), PID()], outputFile=False):
        """AB2XY is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        assert len(sensors)==3, "Three sensors are needed"
        self.sensors = sensors
        self.actuator = actuator
        self.AB2XY = AB2XY
        self.pids = pids
        assert len(self.pids) == 2
        if outputFile:
            self.recorder = Recorder(outputFile)
        else:
            self.recorder = False
        self.go = True

    def run(self):
        self.recorder.start()
        t0 = time.time()
        while self.go:
            (x,y,z), measureXY, y_ag = current_positions(self.AB2XY, self.sensors, self.actuator)
            #feed state to PIDs, in microns
            #in x, we beleive the micromanipulator reading
            self.pids[0].update(self.actuator.step2um(x))
            #in y we beleive the third sensor reading
            self.pids[1].update(y_ag + measureXY[1])
            outputs = np.array([pid.output for pid in self.pids])
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([self.pids[0].current_time-t0, *self.actuator.step2um(np.array([x,y])), *measureXY, y_ag])
            #the new position of the micromanipulator in steps
            newxy = self.actuator.um2integer_step(outputs) + [x,y]
            newxy = self.actuator.truncate_steps(newxy)
            #update micromanipulator position
            if newxy[1] != y or newxy[0] != x:
                self.actuator.move_to(newxy[0], newxy[1], z)
        #wait for recorder completion
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()



class Recorder(Thread):
    """Thread in charge of recording the data to a file."""
    def __init__(self, outputFile):
        Thread.__init__(self)
        self.queue = []
        self.file = outputFile
        self.go = True

    def run(self):
        while self.go:
            while len(self.queue)>0:
                np.array(self.queue.pop(0)).tofile(self.file)
            time.sleep(0.1)

def step_response_Y(outname, kp, ki=0, kd=0, dy=10., originalsetpoint=None):
    """Perform a single depthwise step of dy microns and record the PID response to a file named outname."""
    with closing(DT3100('169.254.3.100')) as sensor, closing(MPC385()) as actuator:
        #setting up sensor
        sensor.set_averaging_type(3)
        sensor.set_averaging_number(3)
        #remember original positions of the sensor and actuator
        if originalsetpoint is None:
            originalsetpoint = sensor.readOne().m
        x0,y0,z0 = actuator.update_current_position()[1:]
        #setting up PID
        pid = PID(kp, ki, kd)
        pid.setPoint = originalsetpoint
        with open(outname, "wb") as fout:
            m = MoverPID_Y(sensor, actuator, pid, outputFile=fout)
            m.start()
            time.sleep(1)
            pid.setPoint += dy
            time.sleep(20)
            #stop PID
            m.go = False
            m.join()
        #move the actuator back to its original position
        actuator.move_to(x0, y0, z0)
    #read back the data and print some metrics
    meas = np.fromfile(outname)
    ts, measured, out = meas.reshape((meas.shape[0]//3, 3)).T
    if not np.any(np.abs(measured-originalsetpoint-dy)<actuator.step2um(1)):
        print("does not converge")
    else:
        print("cverge\teRMS\t%overshoot")
        print('%g\t%g\t%g'%(
            ts[np.where(np.abs(measured-originalsetpoint-dy)<actuator.step2um(1))[0][0]-1],
            np.sqrt(np.mean((measured[np.where(ts>2)[0][0]:]-originalsetpoint-dy)**2)),
            100*(measured.max()-originalsetpoint-dy)/dy
        ))

def step_response(outname, AB2XY, kp, ki=0, kd=0, dx=10., dy=10., originalsetpoint=None):
    """Perform a single step of (dx,dy) microns and record the PID response to a file named outname."""
    if np.isscalar(kp):
        kp = [kp,kp]
    if np.isscalar(ki):
        ki = [ki,ki]
    if np.isscalar(kd):
        kd = [kd,kd]
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as actuator:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        if originalsetpoint is None:
            originalsetpoint = np.array([sensor.readOne().m for sensor in sensors])
            originalsetpoint = np.matmul(AB2XY, originalsetpoint)
        x0,y0,z0 = actuator.update_current_position()[1:]
        #setting up PID
        pids = []
        for p,i,d, s in zip(kp, ki, kd, originalsetpoint):
            pid = PID(p, i, d)
            pid.setPoint = s
            pids.append(pid)
        with open(outname, "wb") as fout:
            m = MoverPID_XY(sensors, actuator, AB2XY, pids, outputFile=fout)
            m.start()
            time.sleep(1)
            pids[0].setPoint += dx
            pids[1].setPoint += dy
            time.sleep(20)
            #stop PID
            m.go = False
            m.join()
        #move the actuator back to its original position
        actuator.move_to(x0, y0, z0)
    #read back the data and print some metrics
    meas = np.fromfile(outname)
    ts, mx, my, ox, oy = meas.reshape((meas.shape[0]//5, 5)).T
    measures = np.column_stack([mx,my])
    rms = np.sqrt(np.sum((originalsetpoint+[dx,dy]-measures)**2, -1))
    if not np.any(rms<actuator.step2um(1)):
        print("does not converge")
    else:
        print("cverge\teRMS\t%overshoot")
        print('%g\t%g\t%g'%(
            ts[np.where(rms<actuator.step2um(1))[0][0]-1],
            np.sqrt(np.mean(rms[np.where(ts>2)[0][0]:]**2)),
            (100*(measures-originalsetpoint-[dx,dy])/[dx,dy]).max()
        ))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perform a single step of dy microns and record the PID response.')
    parser.add_argument('outName', help='Output file name')
    parser.add_argument('kp', type=float, help='Gain of the proportional term of the PID')
    parser.add_argument('--ki', default=0., type=float, help='Gain of the integral term of the PID')
    parser.add_argument('--kd', default=0., type=float, help='Gain of the derivative term of the PID')
    parser.add_argument('--dy', default=10., type=float, help='Amplitude of the step.')
    parser.add_argument('--setPoint', default=None, type=float, help='Original set point of the PID.')
    parser.add_argument('--ip', default='169.254.3.100', help='IP address of the DT3100.')
    args = parser.parse_args()
    step_response_Y(args.outName, args.kp, args.ki, args.kd, args.dy, args.setPoint)
