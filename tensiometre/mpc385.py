from contextlib import closing
import struct, time, select
import numpy as np
import serial
import warnings

#from pyftdi.ftdi import Ftdi
#from pyftdi.serialext import serial_for_url
#allow the library to look for custom vendor and product ID
#Ftdi.add_custom_vendor(0x1342, 'Sutter Instrument')
#Ftdi.add_custom_product(0x1342, 0x0001, 'ROE-200')

#16 microsteps per microns
_ISTEP = 16
_STEP = 1./_ISTEP

class MPC385:
    """Class to interact with a Sutter MPC-385 (=ROE-200 + MPC-200) micromanipulator controller using only the pySerial on Linux. Requires a FTDI driver configured for the custom vendor and product IDs of the ROE-200.

    In short, one has to add the following line in /etc/udev/rules.d/99-ftdi.rules

    ACTION=="add", ATTRS{idVendor}=="1342", ATTRS{idProduct}=="0001", RUN+="/sbin/modprobe ftdi_sio" RUN+="/bin/sh -c 'echo 1342 0001 > /sys/bus/usb-serial/drivers/ftdi_sio/new_id'"
    """

    _NSTEP = 400000
    """Maximum number of steps"""

    def __init__(self, serial_number='SI9L8W3A', timeout=10.0):
        #self.port = serial.Serial('/dev/ttyUSB0', baudrate=128000)
        self.port = serial.Serial(
            '/dev/serial/by-id/usb-Sutter_Sutter_Instrument_ROE-200_%s-if00-port0'%serial_number,
            baudrate=128000,
            timeout=timeout,
        )
        #self.port = serial_for_url('ftdi://0x1342:0x0001/1', baudrate=128000)
        """internal position array for the 4 possible drives, in microsteps"""
        self.positions = np.zeros((4,3), np.uint32)
        try:
            self.update_positions()
        except AssertionError as e:
            self.port.close()
            raise ConnectionError("Unable to initialize micromanipulator state. Please unplug, turn off, wait, turn on, plug back.")

    def close(self):
        self.port.close()

    def wait_readable(self, timeout=1):
        """wait for the interface to be readable during timeout seconds"""
        r, w, x = select.select([self.port], [], [], timeout)
        if self.port not in r:
            raise ConnectionError('Port not readable')

    def wait_writeable(self, timeout=1):
        """wait for the interface to be writeable during timeout seconds"""
        r, w, x = select.select([], [self.port], [], timeout)
        if self.port not in w:
            raise ConnectionError('Port not writeable')

    def query(self, command, pattern='', timeout=1):
        """Write a command and parse the answer using struct syntax, stripped of final CR byte. Timeout in seconds."""
        self.wait_writeable(timeout)
        #ensure a delay between commands of 2ms, as wtritten in the manual
        time.sleep(3e-3)
        self.port.write(command)
        #ensure a delay between commands of 2ms, as wtritten in the manual
        time.sleep(3e-3)
        s = struct.Struct(pattern)
        self.wait_readable(timeout)
        answer = self.port.read(s.size+1)
        assert len(answer) == s.size + 1, 'Answer too short (%d instead of %d), probably a timeout'%(len(answer), s.size + 1)
        assert answer[-1:] == b'\r', '%s does not end by \\r (command was %s)'%(answer, command)
        try:
            return s.unpack(answer[:-1])
        except:
            e = IOError("%s did not match pattern %s"%(answer[:-1], pattern))
            e.answer = answer
            raise e

    def currently_active_drive(self):
        """The currently active drive and firmware version"""
        K, Vl, Vh = self.query(b'K', '=BBB')
        self.version = (Vl, Vh)
        return K

    def connected_manipulators(self):
        """Get number N of connected manipulators and drive status for all 4 drives"""
        return self.query(b'U', '=B????')

    def change_drive(self, manipulator=1):
        """Change the currently selected micromanupulator"""
        m = int(manipulator)
        if m not in [1,2,3,4]:
            raise ValueError('manipulator should be 1, 2, 3 or 4')
        answer, = self.query(struct.pack('=cB', b'I', m), '=c')
        if answer == b'E':
            raise ValueError('manipulator %s is not connected'%manipulator)
        mo = struct.unpack('=B', answer)[0]
        if mo != m:
            raise ValueError('manipulator %s was selected instead of %s'%(mo, manipulator))

    def move_to_center(self):
        """Move to position (0,0,0). This operation is blocking."""
        self.query(b'N')

    def update_current_position(self):
        """Get the currently selected micromanipulator and its current position in term of motor microsteps"""
        t = self.query(b'C', '=Biii', timeout=10.)
        self.positions[t[0]-1] = t[1:]
        return t

    def update_positions(self):
        """Update the internal value of current positions of all connected manipulators."""
        #remember the current drive
        activedrive = self.currently_active_drive()
        for m, ok in enumerate(self.connected_manipulators()):
            if m>0 and ok:
                self.change_drive(m)
                self.update_current_position()
        #change back to the original drive
        self.change_drive(activedrive)

    def step2um(self, pos):
        """Convert a position or an array of positions in microsteps to microns"""
        return pos * _STEP

    def um2step(self, pos):
        """Convert a position or an array of positions in microns to microsteps (float values)"""
        return pos * _ISTEP

    def um2integer_step(self, pos):
        """Convert a position or an array of positions in microns to integer microsteps"""
        if np.isscalar(pos):
            return int(np.rint(self.um2step(pos)))
        return np.rint(self.um2step(pos)).astype(int)

    def truncate_steps(self, pos):
        """Take an arbitrary position in microsteps and saturates it between 0 and the maximum number of steps"""
        return np.minimum(self._NSTEP, np.maximum(0, pos))

    def get_position(self, m=None):
        """Get the current position in microns. Either for all manipulators (default) or one"""
        if m is None:
            return self.step2um(self.positions)
        assert m in range(1,5), "Manipulator must be either None or in [1,2,3,4]"
        return self.step2um(self.positions[m-1])

    def check_in_range(self, pos):
        if pos<0 or pos >self._NSTEP:
            raise ValueError('position %s is not in the range [0,400000)'%pos)

    def move_straight(self, x, y, z, speed=16):
        """Move in a straight line to specified coordinates (in microsteps).
        speed is the velocity of the longest moving axis, from 1 to 16"""
        #raise NotImplementedError("move_straight command as defined by ROE-200 manual makes the instrument hang.")
        assert speed in range(1,17)
        for pos in [x,y,z]:
            self.check_in_range(pos)
        #save current timeout
        timeout = self.port.timeout
        #compute how long it should take
        vmin = 360
        dist = self.step2um(np.max(np.abs(self.positions[0] - [x,y,z])))
        tmax = dist / vmin + 1
        #set longer timeout
        self.port.timeout = tmax
        #disable output
        self.query(b'F', '', timeout=tmax)
        #send command
        self.port.write(struct.pack('=cB', b'S', speed-1))
        #wait 24ms
        time.sleep(0.024)
        #send coordinates to move to
        self.query(struct.pack('=iii', int(x),int(y),int(z)), '')
        #restore timout
        self.port.timeout = timeout

    def move_to(self, x,y,z):
        """Fast, stereotypic movement with firmware controlled velocity. Final position in microsteps."""
        for pos in [x,y,z]:
            self.check_in_range(pos)
        try:
            self.query(struct.pack('=ciii', b'M', int(x),int(y),int(z)), timeout=10.)
        except ConnectionError as e:
            self.interrupt_move()
            warnings.warn(f'Due to "{e}" error, the move was interrupted.')

    def interrupt_move(self):
        """Interrupt move in progress"""
        self.query(b'\x03', '')
