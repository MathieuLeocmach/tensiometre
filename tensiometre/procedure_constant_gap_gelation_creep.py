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
    parser = argparse.ArgumentParser(description='Procedure to apply a constant stress while maintaining a gap of 100µm.')
    parser.add_argument('stress', type=float, help = """Stress to apply, in units of the measured modulus.""")
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
    print(f"{now}: Maintain the gap size while chirping in shear with amplitude {args.ampl} microns, duration {args.T}s, delay {args.delay} s.")
    chirpname = f'maintain_gap_100um_armchirpX_{now}.raw'
    mechtest3sensors.timedep_armX_positionY(
        ab2xy,
        chirpname,
        functions,
        duration=3600, kp=[1,0.01], ki=[0,1e-5],
        moveback=False, state0=force_free
    )
    # now = datetime.now().strftime('%Y%m%d_%H%M')
    # print(f"{now}: Maintain the gap size for 1h while oscillating in shear 2µm 0.1Hz.")
    # mechtest3sensors.oscillating_position(
    #     ab2xy,
    #     f'maintain_gap_100um_oscillateX_{now}.raw',
    #     amplitudex = args.ampl, amplitudey=0, freqx=args.freq, freqy=0,
    #     duration=3600, kp=[0.1,0.01], ki=[0,1e-5],
    #     moveback=False, state0=force_free
    # )
    # print(f"{now}: Maintain the gap size for 1h.")
    # mechtest3sensors.move_to_constant_positions(
    #     ab2xy,
    #     outnames = [f'maintain_gap_100um_{now}.raw'],
    #     dxs=[0], dys=[100],
    #     durations=[3600],
    #     kp=0.01, ki=1e-5,
    #     state0=touching_state
    #     )
    recover(), recover('169.254.4.100'), recover('169.254.5.100')
    with closing(MPC385()) as actuator:
        print(actuator.update_current_position())
    after_gelation = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_gelation.save(f'after_gelation_{now}.npy')
    print(f"position: {after_gelation.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_gelation.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Estimate shear modulus at 1Hz from the last chirp, extrapolating the power law.")
    freqs, Gs = load_chirp_moduli(chirpname, T=args.T, delay=args.delay)
    alpha, modulus = fit_powerlaw_modulus(freqs, Gs[-1], M=190)
    print(f'Module deflection vs displacement @1Hz: {modulus}. Power alpha={alpha}')


    # now = datetime.now().strftime('%Y%m%d_%H%M')
    # print(f"{now}: Estimate shear modulus of the gel by step strains<3µm. Repeated 3 times.")
    # for k in range(1,4):
    #     dxs = np.arange(0,10,3).astype(float)
    #     dxs = np.append(dxs, 0.0)
    #     dys = np.zeros_like(dxs)
    #     outnames = ['shear%d_%d.raw'%(k,dx) for dx in dxs]
    #     outnames[-1] = 'moveback_original_%d.raw'%k
    #     durations = [60]*len(dxs)
    #     durations[-1] = 6*60
    #     mechtest3sensors.move_to_constant_positions(
    #         ab2xy,
    #         outnames,
    #         dxs, dys,
    #         durations,
    #         kp=0.1, moveback=False,
    #         state0=after_gelation)
    #     print(f'Linear rheology number {k} at time {180+(k-1)*10}')
    #     dXs = np.zeros(len(dxs))
    #     dYs = np.zeros(len(dxs))
    #     stdX = np.zeros(len(dxs))
    #     stdY = np.zeros(len(dxs))
    #     for i, outname in enumerate(outnames):
    #         data = np.fromfile(outname)
    #         t, x, y, X, Y, y_ag = data.reshape((len(data)//6,6)).T
    #         y_hg = Y + y_ag
    #         avgX = X[-100:].mean()
    #         avgY = Y[-100:].mean() #(Y[-100:] + 16*y_ag[-100:] - y[-100:]).mean()
    #         stdX[i] = np.std(X[-100:])
    #         stdY[i] = np.std(Y[-100:])#np.std(Y[-100:]+ 16*y_ag[-100:] - y[-100:])
    #         dXs[i] = avgX - after_gelation.deflection[0]
    #         dYs[i] = avgY - after_gelation.deflection[1]
    #
    #     np.savetxt((datetime.now().strftime('%Y%m%d_%H%M_stressvsshear_method2.txt')),(dxs,dXs,dYs,stdX,stdY))
    #     #Stress strain linear rheology for linear regime
    #     a,b = curve_fit(lambda u,a: u*a, dxs, dXs)
    #     module = a[0] # unit µm/µm
    #     print(f'Module deflection vs displacement: {module}')

    # Again check the change w.r.t the inital set position(Xinit,Yinit) and initial zero stress deflection
    # after_linear_rheology = measure_state()
    # now = datetime.now().strftime('%Y%m%d_%H%M')
    # after_linear_rheology.save(f'after_linear_rheology_{now}.npy')
    # print(f"position: {after_linear_rheology.head_to_ground - touching_state.head_to_ground}")
    # print(f"deflection wrt force free: {after_linear_rheology.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Apply the constant stress")
    h = (after_gelation.head_to_ground - touching_state.head_to_ground)[1]
    defl = -args.stress*modulus*h # calculated from the linear rheology
    print(f'We plan to apply a deflection of {defl:0.3f} um')
    #Present sensor readings
    xy2ab = np.linalg.inv(ab2xy)
    a0, b0 = xy2ab @ after_gelation.deflection
    #what is the condition on Delta X to keep sensor a in range?
    #0 < a0 + Delta a < 800
    #0 < a0 + xy2ab[0,0] * Delta X +  xy2ab[0,1] * Delta Y < 800
    #here we suppose that there is no Delta Y
    #0 < a0 + xy2ab[0,0] * Delta X < 800
    #minimum and maximum deflexion depend on the sign of the xy2ab matrix elements
    if xy2ab[0,0]>0:
        mindefl = -a0/xy2ab[0,0]
        maxdefl = (800-a0)/xy2ab[0,0]
    else:
        mindefl = (800-a0)/xy2ab[0,0]
        maxdefl = -a0/xy2ab[0,0]
    #same for sensor b and combine the two conditions
    if xy2ab[1,0]>0:
        mindefl = max(mindefl, -b0/xy2ab[1,0])
        maxdefl = min(maxdefl, (800-b0)/xy2ab[1,0])
    else:
        mindefl = max(mindefl, (800-b0)/xy2ab[1,0])
        maxdefl = min(maxdefl, -b0/xy2ab[1,0])
    #security coefficient of 0.9
    mindefl *=0.9
    maxdefl *=0.9
    print(f'Deflection must be between {mindefl:0.3f} um and {maxdefl:0.3f} um')
    #saturates deflection
    defl = np.maximum(mindefl, np.minimum(maxdefl, defl))
    print(f'The applied deflection will be {defl:0.3f} um')
    now = datetime.now().strftime('%Y%m%d_%H%M')
    outname = f'add_constant_deflectionX{defl:0.3f}_stay_constant_positiony_{now}.raw'
    mechtest3sensors.add_constant_deflectionX_stay_constant_positiony(
        outname,
        ab2xy,
        kp=[0.2,0.1], ki=[0.001,0.001], kd =[0.0,0.0],
        dX=defl,
        moveback= True, state0 = force_free, maxYdispl = 300
    )
    after_creep = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_creep.save(f'after_creep_{now}.npy')
    print(f"position: {after_creep.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_creep.deflection - force_free.deflection}")
