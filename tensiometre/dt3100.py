import re, struct, select, socket, time
from threading import Thread
import numpy as np
#import visa
#Let us have a context manager, not to forget to close instruments
from contextlib import closing

#physical units
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.default_format = '~P'

#pyVISA ressource manager (global variable)
_resource_manager = None

def get_resource_manager():
    """Return the PyVISA Resource Manager, creating an instance if necessary.
    :rtype: visa.ResourceManager
    """
    global _resource_manager
    if _resource_manager is None:
        try:
            _resource_manager = visa.ResourceManager()
        except OSError:
            #fall back on pure python implementation of VISA
            _resource_manager = visa.ResourceManager('@py')
    return _resource_manager

def recover(IPAddress='169.254.3.100'):
    """Recover the basic state of the instrument with low-level commands"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((IPAddress, 10001))
    except ConnectionRefusedError:
        time.sleep(1)
        sock.connect((IPAddress, 10001))
    sock.send(b'$MMD0\r')
    time.sleep(0.04)
    answer = sock.recv(1024*4)
    sock.close()
    if len(answer)<9 or answer[-9:] != b'$MMD0OK\r\n':
        print("%s not ready"%IPAddress)
    else:
        print("%s ready"%IPAddress)

class ControllerInfo:
    """Controller information"""
    def __init__(self, answer):
        m = re.match(
            '\$INDSN([0-9]+);PC([0-9]+);RI([A-Z]+);SW([0-9][0-9][a-z]);OP([0-9]{1,2});NM(.{,32})OK',
            answer
        )
        if m is None:
            raise ValueError("Unable to parse controller info: %s"%answer)
        for i, s in zip([1,2,5], ['sn', 'pc', 'op']):
            setattr(self, s, int(m.group(i)))
        self.ri = m.group(3)
        sw = m.group(4)
        self.sw = (int(sw[0]), int(sw[1]), sw[2])
        self.nm = m.group(6)

    def __str__(self):
        return "\n".join([
            "- serial number {sn:d}",
            "- product code {pc:d}",
            "- revision index {ri:s}",
            "- software version {sw[0]:d}.{sw[1]:d}{sw[2]:s}",
            "- option {op:d}",
            "- name {nm:s}"
        ]).format(**self.__dict__)

class SensorInfo:
    """Sensor informations"""
    def __init__(self, answer):
        m = re.match(
            '\$SENSN([0-9]+);PC([0-9]+);RI([A-Z]+);OP([0-9]+);NM([SU][0-9 ][0-9 ]);L([0-9]+);SMR([0-9]+);MMR([0-9]+);EMR([0-9]+)OK',
            answer
        )
        if m is None:
            raise ValueError("Unable to parse sensor info: %s"%answer)
        for i, s in zip([1,2,4], ['sn', 'pc', 'op']):
            setattr(self, s, int(m.group(i)))
        #cable length in 10 cm
        self.l = Q_(int(m.group(i)), 'dm')
        for i, s in zip([7,8,9], ['smr', 'mmr', 'emr']):
            setattr(self, s, Q_(int(m.group(i)), 'um'))
        self.ri = m.group(3)
        self.nm = m.group(5)

    @property
    def range(self):
        return self.emr - self.smr

    def __str__(self):
        return "\n".join([
            "- serial number {sn:d}",
            "- product code {pc:d}",
            "- revision index {ri:s}",
            "- option {op:d}",
            "- designation of sensor {nm:s}",
            "- length of integrated cable {l:}",
            "- start of measuring range {smr:}",
            "- midrange {mmr:}",
            "- end of measuring range {emr:}",
            "- range {rng:}",
        ]).format(rng=self.range, **self.__dict__)

    def scale(self, values):
        """Scale sensor values to micrometers"""
        return values / 65535 * self.range

class DT3100:
    """Class to communicate with MicroEpsilon DT3100 controller"""
    def __init__(self, IPAddress = '169.254.3.100'):
        self.IPAddress = IPAddress
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((IPAddress, 10001))
        except ConnectionRefusedError:
            time.sleep(1)
            self.sock.connect((IPAddress, 10001))
        #self.inst = get_resource_manager().open_resource('TCPIP::%s::10001::SOCKET'%IPAddress)
        #self.inst.write_termination = '\r'
        #self.inst.read_termination = '\r\n'
        #self.inst.timeout = 2000
        self.end_acquisition()
        self._status()
        self._settings()
        self._controller_info()
        self._sensor_info()
        self._read_potentiometer()
        self._buffer = b''

    def close(self):
        """End acquisition, clear internal buffers and close the connection to the instrument."""
        self.end_acquisition()
        self.sock.close()

    def wait_readable(self, timeout=1):
        """wait for the interface to be readable"""
        r, w, x = select.select([self.sock], [], [], timeout)
        return self.sock in r

    def query(self, command, ascii=True):
        """Write a command and read the answer, supposing no aquisition streaming"""
        assert self.mmd == 0, "Cannot send commands in measuring mode"
        self.sock.send(b'$'+command.encode('ascii')+b'\r')
        answer = b''
        while len(answer)<2 or answer[-2:] != b'\r\n':
            self.wait_readable()
            answer += self.sock.recv(1024)
        #if len(answer)<2 or answer[-2:] != b'\r\n':
         #   raise ValueError("Ill formed answer to %s: %s"%(command, answer))
        if ascii:
            return answer[:-2].decode('ascii')
        else:
            return answer

    def _status(self):
        """Fetch the status and parse it."""
        answer = self.query('STS')
        m = re.match('\$STSCBL([0-9]+);ATR([0-9]+)OK', answer)
        if m is None:
            raise ValueError("Unable to parse status: %s"%answer)
        self.cbl = int(m.group(1))
        self.atr = int(m.group(2))
        self.customer_specific = self.atr & 0x80 != 0

    @property
    def status(self):
        """Status in human readable form"""
        return "\n".join([
            "- Cable at factory setting: %s" % (self.cbl==0),
            "- Available targets:",
            "\t- ferromagnetic: %s" % (self.atr&1 != 0),
            "\t- non ferro: %s" % (self.atr&2 != 0),
            "\t- customized traget: %s" % (self.atr&4 != 0),
            "\t- customised sensor: %s" % self.customer_specific,
        ])

    def _settings(self):
        """Fetch the current settings of measuring mode, data rate,
        Values To Take (without leading zeros), target selection and
        the content of the text field."""
        answer = self.query('SET')
        m = re.match(
            '\$SETMMD([0-9]+);SRA([0-9]+);AVT([0-9]+);AVN([0-9]+);VTT([0-9]{1,4});TAR([0-9]+);ETF(.*)OK',
            answer
        )
        if m is None:
            raise ValueError("Unable to parse settings: %s"%answer)
        for i, s in enumerate(['mmd', 'sra', 'avt', 'avn', 'vtt', 'tar']):
            setattr(self, s, int(m.group(i+1)))
        self.etf = m.group(7)

    @property
    def settings(self):
        """Settings in human readable form"""
        return "\n".join([
            "- Measuring mode: %s"%self.h_mmd(),
            "- Data rate: %d samples/s" % self.h_sra(),
            "- Averaging type: %s"  % self.h_avt(),
            "- Averaging over %d values" % self.h_avn(),
            "- Values to take: %d" % self.vtt,
            "- Target: %s" % self.h_tar()
        ])

    def h_mmd(self):
        """Measuring mode in human readable format"""
        modes = [
            "not measuring",
            "continuous transmission",
            "trigger on rising edge",
            "trigger on falling edge",
            "gate function at high level",
            "gate function at low level",
        ]
        return modes[self.mmd]

    def h_sra(self):
        """Data rate in samples per second"""
        rates = [3600, 7200, 14400]
        return rates[self.sra]

    def h_avt(self):
        """Averaging type in human readable format"""
        types = ["none", "moving", "recursive", "median"]
        return types[self.avt]

    def h_avn(self):
        """Averaging number, depends both on averaging type and on AVN setting"""
        ns = [
            [0]*4,
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [3, 5, 7, 9]
        ]
        return ns[self.avt][self.avn]

    def unit_time(self):
        """Unit time per sample"""
        dt = 1./self.h_sra()
        #in case of median averaging, sampling rate is reduced
        if self.avt == 3:
            dt *= self.h_avn()
        return Q_(dt, 's')

    def h_tar(self):
        """Target material in human readable format"""
        targets = {
            1:"ferromagnetic (iron)",
            2:"not ferromagnetic (aluminum)",
            4:"customized 1 (iron 2)",
            8:"customized 2 (aluminum 2)"
        }
        return targets[self.tar]

    def _controller_info(self):
        """Reading the index of controller."""
        answer = self.query('IND')
        self.controller = ControllerInfo(answer)

    def _sensor_info(self):
        """Reading the index of sensor."""
        answer = self.query('SEN')
        self.sensor = SensorInfo(answer)

    def _read_potentiometer(self):
        """Readout the potentiometer positions in the order: DA_Null, DA_Gain and DA_Lin"""
        answer = self.query('RPT')
        m = re.match(
            '\$RPT([0-9]+);([0-9]+);([0-9]+)OK',
            answer
        )
        if m is None:
            raise ValueError("Unable to parse potentiometer: %s"%answer)
        self.calibSettings = [int(g) for g in m.groups()]

    @property
    def sensor_type(self):
        """Type of the sensor"""
        sensorname = "EP%s" % self.sensor.nm
        if self.customer_specific:
            sensorname += "-LC"
        return sensorname


    def __str__(self):
        sep = "\n============================================\n"
        return sep.join([
            "DT3100 information",
            "Status:\n%s"%self.status,
            "Controller:\n%s" % self.controller,
            "Sensor:\n%s" % self.sensor,
            "Settings:\n%s" % self.settings
        ])

    def decode(self,buffer):
        """From raw output to distance values"""
        assert len(buffer)%3 == 0
        return self.sensor.scale(np.fromiter(
            (
                (val[0] & 0x3F) + ((val[1] & 0x3F) << 6) + ((val[2] & 0xF) << 12)
                for val in struct.iter_unpack('BBB', buffer)
            ),
            dtype=np.float64,
            count=len(buffer)//3
        ))

    def read_duration(self, stream, duration=Q_(1., 's'), chunk=4096):
        """Read measurements at least for a given time duration to a stream."""
        try:
            T = duration.to('s').m
            nacq = duration / self.unit_time()
        except AttributeError:
            #Input duration is not a pint object. Supposed to be in seconds
            T = duration
            nacq = duration / self.unit_time().m
        t0 = time.time()
        self.sock.send(b'$MMD1\r')
        ret = b''
        while len(ret)<9 and self.wait_readable():
            ret += self.sock.recv(chunk)
        assert ret[:9] == b'$MMD1OK\r\n', "%s instead of $MMD1OK\r\n" % ret[:9]
        ret = ret[9:]
        while time.time() < t0 + T and self.wait_readable():
            ret += self.sock.recv(chunk)
            iM = 3*(len(ret)//3)
            values = self.decode(ret[:iM])
            values.m.tofile(stream)
            ret = ret[iM:]
        ret += self.end_acquisition(chunk)
        iM = 3*(len(ret)//3)
        values = self.decode(ret[:iM])
        values.m.tofile(stream)


    def start_aquisition(self):
        """Set the instrument to continuous acquisition mode."""
        self.sock.send(b'$MMD1\r')
        ret = b''
        while len(ret)<9 and self.wait_readable():
            ret += self.sock.recv(9)
        assert ret[:9] == b'$MMD1OK\r\n', "%s instead of $MMD1OK\r\n" % ret[:9]
        self.mmd = 1

    def readN(self, N):
        """Read N distances"""
        if self.mmd == 0:
            self.start_aquisition()
        buffer = b''
        while len(self._buffer)<3*N:
            self._buffer += self.sock.recv(3*N-len(buffer))
        buffer = self._buffer[:3*N]
        self._buffer = self._buffer[3*N:]
        return self.decode(buffer)

    def readOne(self, timeout=1):
        """Read the instantaneous distance."""
        #ask for a single distance
        #The command `$GMD<CR>` returns both ascii and binary
        #so we cannot use self.query
        assert self.mmd == 0, "Cannot send commands in measuring mode"
        self.sock.send(b'$GMD\r')
        answer = b''
        #expected answer length is 11 bits
        while len(answer)<11:
            self.wait_readable()
            answer += self.sock.recv(11)
        if answer[:6].decode('ascii') != '$GMDOK':
            raise ValueError("Unable to parse instrument answer: %s"%answer)
        #read the 3 bytes of the measuring value. Maybe they are already at the end of the answer
        ret = answer[8:11]
        #convert to a distance
        return self.decode(ret)[0]


    def end_acquisition(self, chunk=4096, maxsize=None):
        """Stop acquisition and returns all bytes read until the stop command is taken into account by the instrument. To avoid memory overload, only the last `maxsize` bytes are returned."""
        #set measuring mode to 0 (no acquisition)
        self.sock.send(b'$MMD0\r')
        self.mmd = 0
        ret = b''
        ok = True
        while ok:
            ret = (ret + self.sock.recv(chunk))
            if maxsize is not None:
                ret = ret[-maxsize:]
            ok = self.wait_readable()
        return ret[:-9]

    def set_averaging_type(self, avt):
        """Averaging type:
        0 none
        1 moving
        2 recursive
        3 median"""
        if avt not in (0,1,2,3):
            raise ValueError("Averaging type %s is not recognised"%avt)
        answer = self.query('AVT%d'%avt)
        if answer != '$AVT%dOK'%avt:
            raise ValueError("Unable to parse instrument answer: %s"%answer)
        self.avt = avt

    def set_averaging_number(self, avn):
        """Averaging number:
        0 for moving and recursive average =  4; for Median = 3
        1 for moving and recursive average =  8; for Median = 5
        2 for moving and recursive average = 16; for Median = 7
        3 for moving and recursive average = 32; for Median = 9"""
        if avn not in (0,1,2,3):
            raise ValueError("Averaging number %s is not recognised"%avn)
        answer = self.query('AVN%d'%avn)
        if answer != '$AVN%dOK'%avn:
            raise ValueError("Unable to parse instrument answer: %s"%answer)
        self.avn = avn


class ReadOne(Thread):
    """Thread to read asynchronously a single value from a DT3100.

    Usage:
    with closing(DT3100()) as sensor:
        r = ReadOne(sensor)
        r.start()
        #do things
        r.join()
        print(r.value)"""
    def __init__(self, sensor):
        Thread.__init__(self)
        self.sensor = sensor
        self.value = None

    def run(self):
        self.value = self.sensor.readOne()
        if self.value is None:
            raise ValueError("Sensor on %s returned None value"%self.sensor.IPAddress)

def read_both(sensorA, sensorB):
    """Read both sensors asynchronously"""
    readerA = ReadOne(sensorA)
    readerB = ReadOne(sensorB)
    readerA.start()
    readerB.start()
    #do other things
    readerA.join()
    readerB.join()
    return readerA.value.m, readerB.value.m

def read_all(sensors):
    """Read several sensors asynchronously"""
    readers = [ReadOne(sensor) for sensor in sensors]
    for reader in readers:
        reader.start()
    #do other things
    for reader in readers:
        reader.join()
    return np.array([reader.value.m for reader in readers])

class ReadDuration(Thread):
    """Read asynchronously a sensor to a stream during a fixed amount of time."""
    def __init__(self, sensor, stream, duration):
        Thread.__init__(self)
        self.sensor = sensor
        self.stream = stream
        self.duration = duration
    def run(self):
        self.sensor.read_duration(self.stream, self.duration)
