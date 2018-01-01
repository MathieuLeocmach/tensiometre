import visa
import re, struct
import numpy as np
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
    
def recover(ip='169.254.3.100'):
    """Recover the basic state of the instrument with low-level commands"""
    with closing(get_resource_manager().open_resource(
        'TCPIP::%s::10001::SOCKET'%ip
    )) as inst:
        inst.write_termination = '\r'
        inst.read_termination = '\r\n'
        inst.timeout = 2000
        lib = inst.visalib
        session = inst.session
        s = lib.sessions[session].interface
        print(inst.query('$MMD0'))

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
    def __init__(self, iIPAddress = '169.254.3.100'):
        self.iIPAddress = '169.254.3.100'
        self.inst = get_resource_manager().open_resource('TCPIP::%s::10001::SOCKET'%self.iIPAddress)
        self.inst.write_termination = '\r'
        self.inst.read_termination = '\r\n'
        self.inst.timeout = 2000
        self._status()
        self._settings()
        self._controller_info()
        self._sensor_info()
        self._read_potentiometer()
        self._buffer = b''
    
    def close(self):
        """End acquisition, clear internal buffers and close the connection to the instrument."""
        self.end_acquisition()
        self.inst.close()
    
    def _status(self):
        """Fetch the status in human readable form."""
        answer = self.inst.query('$STS')
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
        answer = self.inst.query('$SET')
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
        answer = self.inst.query('$IND')
        self.controller = ControllerInfo(answer)
        
    def _sensor_info(self):
        """Reading the index of sensor."""
        answer = self.inst.query('$SEN')
        self.sensor = SensorInfo(answer)
    
    def _read_potentiometer(self):
        """Readout the potentiometer positions in the order: DA_Null, DA_Gain and DA_Lin"""
        answer = self.inst.query('$RPT')
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
    
    def start_aquisition(self):
        """Set the instrument to continuous acquisition mode."""
        answer = self.inst.query('$MMD1')
        if answer == '$MMD1OK':
            self.mmd = 1
            self._buffer = b''
        else:
            raise ValueError("Unable to parse instrument answer: %s"%answer)
    
    def readN(self, N):
        """Read N distances"""
        if self.mmd == 0:
            self.start_aquisition()
        buffer = b''
        while len(self._buffer)<3*N:
            self._buffer += self.inst.read_raw(3*N-len(buffer))
        buffer = self._buffer[:3*N]
        self._buffer = self._buffer[3*N:]
        return self.decode(buffer)
    
    
    def end_acquisition(self, maxsize=1024):
        """Stop acquisition and returns all bytes read until the stop command is taken into account by the instrument. To avoid memory overload, only the last `maxsize` bytes are returned."""
        #set measuring mode to 0 (no acquisition)
        self.inst.write('$MMD0')
        self.mmd = 0
        #flush the buffer from binary data
        ret = self._buffer
        while ret[-9:] != b'$MMD0OK\r\n':
            try:
                ret += self.inst.read_raw()
            except visa.VisaIOError:
                return ret
            ret = ret[:-maxsize]
        return ret[:-9]
    
    def set_averaging_type(self, avt):
        """Averaging type: 
        0 none
        1 moving
        2 recursive
        3 median"""
        if avt not in (0,1,2,3):
            raise ValueError("Averaging type %s is not recognised"%avt)
        answer = self.inst.query('$AVT%d'%avt)
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
        answer = self.inst.query('$AVN%d'%avn)
        if answer != '$AVN%dOK'%avn:
            raise ValueError("Unable to parse instrument answer: %s"%answer)
        self.avn = avn
