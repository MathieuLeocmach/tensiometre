import time
from contextlib import closing
from threading import Thread
import numpy as np
from tensiometre.dt3100 import DT3100
from tensiometre.mpc385 import MPC385
from tensiometre.pid import PID

class MoverPID_Z(Thread):
    """Thread in charge of controlling the Z position of the tensiometer so that the target stays at a given distance from the sensor."""
    def __init__(self, inputfunction, mpc, pid=PID(), outputFile=False):
        Thread.__init__(self)
        self.inputfunction = inputfunction
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
            measure = self.inputfunction()
            self.pid.update(measure)
            if self.recorder:
                self.recorder.queue.append([self.pid.current_time-t0, measure, self.pid.output])
            x,y,z = self.mpc.update_current_position()[-3:]
            newz = z - self.mpc.um2step(self.pid.output)
            newz = int(min(200000, max(0, newz)))
            if newz != z:
                self.mpc.move_to(x, y, newz)
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

def step_response(outname, kp, ki=0, kd=0, dz=10.):
    """Perform a single step of dz microns and record the PID response to a file named outname."""
    with closing(DT3100('169.254.3.100')) as capteur, closing(MPC385()) as mpc:
        #setting up sensor
        capteur.set_averaging_type(3)
        capteur.set_averaging_number(3)
        dt = capteur.unit_time()
        #remember original positions of the sensor and actuator
        originalsetpoint = capteur.readOne().m
        x0,y0,z0 = mpc.update_current_position()[1:]
        #setting up PID
        pid = PID(kp, ki, kd)
        pid.setPoint = originalsetpoint
        with open(outname, "wb") as fout:
            m = MoverPID_Z(lambda : capteur.readOne().m, mpc, pid, outputFile=fout)
            m.start()
            time.sleep(1)
            pid.setPoint += dz
            time.sleep(20)
            m.go = False
            m.join()
        #move the actuator back to its original position
        mpc.move_to(x0, y0, z0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perform a single step of dz microns and record the PID response.')
    parser.add_argument('outName', help='Output file name')
    parser.add_argument('kp', type=float, help='Gain of the proportional term of the PID')
    parser.add_argument('--ki', default=0., type=float, help='Gain of the integral term of the PID')
    parser.add_argument('--kd', default=0., type=float, help='Gain of the derivative term of the PID')
    parser.add_argument('--dz', default=10., type=float, help='Amplitude of the step.')
    parser.add_argument('--ip', default='169.254.3.100', help='IP address of the DT3100.')
    args = parser.parse_args()
    step_response(args.outName, args.kp, args.ki, args.kd, args.dz)
