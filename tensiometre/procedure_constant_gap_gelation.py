import argparse, os.path
from datetime import datetime
import numpy as np
from contextlib import closing, ExitStack
from scipy.optimize import curve_fit
from tensiometre.dt3100 import DT3100, recover
from tensiometre.mpc385 import MPC385
from tensiometre import mechtest3sensors
from tensiometre.chirp import optimal_chirp, load_chirp_moduli, fit_powerlaw_modulus
from tensiometre.show_measurements import show_measurement


def measure_state():
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{i+3:d}.100'))) for i in range(3)]
        mpc = stack.enter_context(closing(MPC385()))
        return mechtest3sensors.State().read(sensors, mpc, ab2xy)

from scipy.signal import chirp
def optimal_chirp(t, amplitude=10, f1=1e-2, f2=1, T=66, r=0.1, delay=4):
    td = np.mod(t,T+delay)
    tt = td-delay
    chrp = chirp(tt, f1, T, f2, phi=90, method='logarithmic')
    chrp = np.where(
        tt<0, 0, np.where(
            2*tt < r*T,
            np.cos(np.pi/r*(tt/T-r/2))**2 * chrp,
            np.where(
                tt/T > 1 - r/2,
                np.cos(np.pi/r*(tt/T-1+r/2))**2 * chrp,
                chrp
            )
        )
    )
    return amplitude*chrp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procedure to gelify while chirping and maintaining a gap.')
    parser.add_argument('calibrationfilename', type=str, help='path and name of the ab2xy calibration file. Expects a .npy.')
    parser.add_argument('--gap', type=float, default=100, help='Gap to open from the initial (touching) position. Default 100µm.')
    parser.add_argument('--fmin', type=float, default=0.01, help='Lowest frequency of the chirp in X during gelation (Hz). Default 0.01Hz.')
    parser.add_argument('--fmax', type=float, default=1, help='Highest frequency of the chirp in X during gelation (Hz). Default 1Hz.')
    parser.add_argument('--ampl', type=float, default=3, help='Amplitude of the chirp in X during gelation (µm).')
    parser.add_argument('--T', type=float, default=198, help='Duration of the chirps in X during gelation (s)')
    parser.add_argument('--delay', type=float, help='Delay between the chirps in X during gelation (s). By default T/16.')

    args = parser.parse_args()
    if args.delay is None:
        args.delay = args.T / 16
    ab2xy = np.load(args.calibrationfilename)

    recover(), recover('169.254.4.100'), recover('169.254.5.100')
    with closing(MPC385()) as actuator:
        print(actuator.update_current_position())

    input(f"Please move to just touching the bottom. Enter when OK.")
    touching_state = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    touching_state.save(f'touching_{now}.npy')

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Lifting up by {args.gap} µm to setup the initial gap size. Maintain 60s to ensure steady state.")
    mechtest3sensors.move_to_constant_positions(
        ab2xy,
        outnames = [f'positon_from_bottom_100um_{now}.raw'],
        dxs=[0], dys=[100],
        durations=[60],
        kp=0.1,
        state0=touching_state
        )
    force_free = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    force_free.save(f'force_free_{now}.npy')
    print(f"position wrt touching: {force_free.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt touching: {force_free.deflection - touching_state.deflection}")

    functions = [
        lambda t: optimal_chirp(
            t, amplitude=args.ampl, T=args.T, delay=args.delay,
            f1=args.fmin, f2=args.fmax,
            ),
        lambda t: 0
    ]
    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Maintain 'forever' the gap size while chirping in shear with amplitude {args.ampl} microns, duration {args.T}s, delay {args.delay} s.")
    chirpname = f'maintain_gap_100um_armchirpX_{now}.raw'
    mechtest3sensors.timedep_armX_positionY(
        ab2xy,
        chirpname,
        functions,
        duration=None, kp=[1,0.01], ki=[0,1e-5],
        moveback=False, state0=force_free
    )

    recover(), recover('169.254.4.100'), recover('169.254.5.100')
    with closing(MPC385()) as actuator:
        print(actuator.update_current_position())
    after_gelation = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_gelation.save(f'after_gelation_{now}.npy')
    print(f"position: {after_gelation.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_gelation.deflection - force_free.deflection}")
