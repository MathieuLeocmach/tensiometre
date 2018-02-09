import time
from contextlib import closing
from threading import Thread
import numpy as np
from tensiometre.dt3100 import DT3100, ReadOne
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID

def calibrate_transfer_matrix(dx=100, dz=100, nsamples=1):
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
    

class MoverPID_Z(Thread):
    """Thread in charge of controlling the Z position of the tensiometer so that the target stays at a given distance from the sensor."""
    def __init__(self, capteur, mpc, pid=PID(), outputFile=False):
        Thread.__init__(self)
        self.capteur = capteur
        self.mpc = mpc
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
            reader = ReadOne(self.capteur)
            reader.start()
            #in parallel, ask for the current position of the micromanipulator
            x,y,z = self.mpc.update_current_position()[-3:]
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
            newz = z - self.mpc.um2step(self.pid.output)
            newz = int(min(200000, max(0, newz)))
            #update micromanipulator position
            if newz != z:
                self.mpc.move_to(x, y, newz)
        while len(self.recorder.queue)>0:
            time.sleep(0.01)
        self.recorder.go = False
        self.recorder.join()


class MoverPID_XZ(Thread):
    """Thread in charge of controlling the X and Z position of the tensiometer so that the target stays at a given distances from the sensors."""
    def __init__(self, sensors, mpc, AB2XZ, pids=[PID(), PID()], outputFile=False):
        """AB2XZ is the transfer matrix between sensor coordinates and actuator coordinates."""
        Thread.__init__(self)
        self.sensors = sensors
        self.mpc = mpcDT3100
        self.AB2XZ = AB2XZ
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
            #ask asynchronously for a position measurement for each sensor
            readers = [ReadOne(sensor) for sensor in self.sensors]
            for reader in readers:
                reader.start()
            #in parallel, ask for the current position of the micromanipulator
            x,y,z = self.mpc.update_current_position()[-3:]
            #wait for the position measurements
            for reader in readers:
                reader.join()
            #translate sensor coordinates into micromanipulator coordinates
            measureXZ = np.matmul(self.AB2XZ, [reader.value.m for reader in readers])
            #feed to PIDs
            outputs = []
            for measure, pid in zip(measureXZ, self.pids):
                pid.update(measure)
                outputs.append(pid.output)
            outputs = np.array(outputs)
            #save state to file asynchronously
            if self.recorder:
                self.recorder.queue.append([pid.current_time-t0, *measureXZ,  *outputs])
            #translate PID output in microns into 
            #the new position of the micromanipulator in steps
            newxz = (np.array([x,z]) + self.mpc.um2step(outputs)).astype(int)
            newxz = np.minimum(200000, np.maximum(0, newxz))
            #update micromanipulator position
            if newxz[1] != z or newxz[0] != x:
                self.mpc.move_to(newxz[0], y, newxz[1])
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

def step_response_Z(outname, kp, ki=0, kd=0, dz=10., originalsetpoint=None):
    """Perform a single step of dz microns and record the PID response to a file named outname."""
    with closing(DT3100('169.254.3.100')) as capteur, closing(MPC385()) as mpc:
        #setting up sensor
        capteur.set_averaging_type(3)
        capteur.set_averaging_number(3)
        #remember original positions of the sensor and actuator
        if originalsetpoint is None:
            originalsetpoint = capteur.readOne().m
        x0,y0,z0 = mpc.update_current_position()[1:]
        #setting up PID
        pid = PID(kp, ki, kd)
        pid.setPoint = originalsetpoint
        with open(outname, "wb") as fout:
            m = MoverPID_Z(capteur, mpc, pid, outputFile=fout)
            m.start()
            time.sleep(1)
            pid.setPoint += dz
            time.sleep(20)
            #stop PID
            m.go = False
            m.join()
        #move the actuator back to its original position
        mpc.move_to(x0, y0, z0)
    #read back the data and print some metrics
    meas = np.fromfile(outname)
    ts, measured, out = meas.reshape((meas.shape[0]//3, 3)).T
    if not np.any(np.abs(measured-originalsetpoint-dz)<mpc.step2um(1)):
        print("does not converge")
    else:
        print("cverge\teRMS\t%overshoot")
        print('%g\t%g\t%g'%(
            ts[np.where(np.abs(measured-originalsetpoint-dz)<mpc.step2um(1))[0][0]-1],
            np.sqrt(np.mean((measured[np.where(ts>2)[0][0]:]-originalsetpoint-dz)**2)),
            100*(measured.max()-originalsetpoint-dz)/dz
        ))
        
def step_response(outname, AB2XZ, kp, ki=0, kd=0, dx=10., dz=10., originalsetpoint=None):
    """Perform a single step of dz microns and record the PID response to a file named outname."""
    if np.isscalar(kp):
        kp = [kp,kp]
    if np.isscalar(ki):
        ki = [ki,ki]
    if np.isscalar(kd):
        kd = [kd,kd]
    with closing(DT3100('169.254.3.100')) as sensorA, closing(DT3100('169.254.4.100')) as sensorB, closing(MPC385()) as mpc:
        sensors = [sensorA, sensorB]
        #setting up sensors
        for sensor in sensors:
            sensor.set_averaging_type(3)
            sensor.set_averaging_number(3)
        #remember original positions of the sensors and actuator
        if originalsetpoint is None:
            originalsetpoint = np.array([sensor.readOne().m for sensor in sensors])
            originalsetpoint = np.matmul(AB2XZ, originalsetpoint)
        x0,y0,z0 = mpc.update_current_position()[1:]
        #setting up PID
        pids = []
        for p,i,d, s in zip(kp, ki, kd, originalsetpoint):
            pid = PID(p, i, d)
            pid.setPoint = s
            pids.append(pid)
        with open(outname, "wb") as fout:
            m = MoverPID_XZ(sensors, mpc, AB2XZ, pids, outputFile=fout)
            m.start()
            time.sleep(1)
            pids[0].setPoint += dx
            pids[1].setPoint += dz
            time.sleep(20)
            #stop PID
            m.go = False
            m.join()
        #move the actuator back to its original position
        mpc.move_to(x0, y0, z0)
    #read back the data and print some metrics
    meas = np.fromfile(outname)
    ts, mx, mz, ox, oz = meas.reshape((meas.shape[0]//5, 5)).T
    measures = np.column_stack([mx,mz])
    rms = np.sqrt(np.sum((originalsetpoint+[dx,dz]-measures)**2, -1))
    if not np.any(rms<mpc.step2um(1)):
        print("does not converge")
    else:
        print("cverge\teRMS\t%overshoot")
        print('%g\t%g\t%g'%(
            ts[np.where(rms<mpc.step2um(1))[0][0]-1],
            np.sqrt(np.mean(rms[np.where(ts>2)[0][0]:]**2)),
            (100*(measures-originalsetpoint-[dx,dz])/[dx,dz]).max()
        ))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perform a single step of dz microns and record the PID response.')
    parser.add_argument('outName', help='Output file name')
    parser.add_argument('kp', type=float, help='Gain of the proportional term of the PID')
    parser.add_argument('--ki', default=0., type=float, help='Gain of the integral term of the PID')
    parser.add_argument('--kd', default=0., type=float, help='Gain of the derivative term of the PID')
    parser.add_argument('--dz', default=10., type=float, help='Amplitude of the step.')
    parser.add_argument('--setPoint', default=None, type=float, help='Original set point of the PID.')
    parser.add_argument('--ip', default='169.254.3.100', help='IP address of the DT3100.')
    args = parser.parse_args()
    step_response_Z(args.outName, args.kp, args.ki, args.kd, args.dz, args.setPoint)
    
