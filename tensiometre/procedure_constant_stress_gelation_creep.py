import argparse, os.path
from datetime import datetime
import numpy as np
from contextlib import closing, ExitStack
from scipy.optimize import curve_fit
from tensiometre.dt3100 import DT3100, recover
from tensiometre.mpc385 import MPC385
from tensiometre import mechtest3sensors
from tensiometre.show_measurements import show_measurement


def measure_state():
    with ExitStack() as stack:
        sensors = [stack.enter_context(closing(DT3100(f'169.254.{i+3:d}.100'))) for i in range(3)]
        mpc = stack.enter_context(closing(MPC385()))
        return mechtest3sensors.State().read(sensors, mpc, ab2xy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Procedure to apply a constant stress while maintaining a gap of 100µm.')
    parser.add_argument('stress', type=float, help = """Stress to apply, in units of the measured modulus.""")
    parser.add_argument('calibrationfilename', type=str, help='path and name of the ab2xy calibration file. Expects a .npy.')

    args = parser.parse_args()
    ab2xy = np.load(args.calibrationfilename)

    recover(), recover('169.254.4.100'), recover('169.254.5.100')
    with closing(MPC385()) as actuator:
        print(actuator.update_current_position())

    input(f"Please move to just touching the bottom. Enter when OK.")
    touching_state = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    touching_state.save(f'touching_{now}.npy')

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Lifting up by 100 µm to setup the initial gap size. Maintain 60s to ensure steady state.")
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

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Maintain the gap size for 1h.")
    mechtest3sensors.move_to_constant_positions(
        ab2xy,
        outnames = [f'maintain_gap_100um_{now}.raw'],
        dxs=[0], dys=[100],
        durations=[3600],
        kp=0.01, ki=1e-5,
        state0=touching_state
        )
    after_gelation = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_gelation.save(f'after_gelation_{now}.npy')
    print(f"position: {after_gelation.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_gelation.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Relax to zero stress for 5 min.")
    mechtest3sensors.stay_constant_deflection(
        'relax_to_zero_force_after_gelation.raw',
        ab2xy,
        kp=0.01, ki = 0.0005,
        duration=300, moveback = False, delay = 1,
        state0 = force_free
        )
    after_relax = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_relax.save(f'after_relaxation_{now}.npy')
    print(f"position: {after_relax.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_relax.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Estimate shear modulus of the gel by step strains<3µm. Repated 3 times.")
    for k in range(1,4):
        dxs = np.arange(0,10,3).astype(float)
        dxs = np.append(dxs, 0.0)
        dys = np.zeros_like(dxs)
        outnames = ['shear%d_%d.raw'%(k,dx) for dx in dxs]
        outnames[-1] = 'moveback_original_%d.raw'%k
        durations = [60]*len(dxs)
        durations[-1] = 6*60
        mechtest3sensors.move_to_constant_positions(
            ab2xy,
            outnames,
            dxs, dys,
            durations,
            kp=0.1, moveback=False,
            state0=after_relax)
        print(f'Linear rheology number {k} at time {180+(k-1)*10}')
        dXs = np.zeros(len(dxs))
        dYs = np.zeros(len(dxs))
        stdX = np.zeros(len(dxs))
        stdY = np.zeros(len(dxs))
        for i, outname in enumerate(outnames):
            data = np.fromfile(outname)
            t, x, y, X, Y, y_ag = data.reshape((len(data)//6,6)).T
            y_hg = Y + y_ag
            avgX = X[-100:].mean()
            avgY = Y[-100:].mean() #(Y[-100:] + 16*y_ag[-100:] - y[-100:]).mean()
            stdX[i] = np.std(X[-100:])
            stdY[i] = np.std(Y[-100:])#np.std(Y[-100:]+ 16*y_ag[-100:] - y[-100:])
            dXs[i] = avgX - after_relax.deflection[0]
            dYs[i] = avgY - after_relax.deflection[1]

        np.savetxt((datetime.now().strftime('%Y%m%d_%H%M_stressvsshear_method2.txt')),(dxs,dXs,dYs,stdX,stdY))
        #Stress strain linear rheology for linear regime
        a,b = curve_fit(lambda u,a: u*a, dxs, dXs)
        module = a[0] # unit µm/µm
        print(f'Module deflection vs displacement: {module}')

    # Again check the change w.r.t the inital set position(Xinit,Yinit) and initial zero stress deflection
    after_linear_rheology = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_linear_rheology.save(f'after_linear_rheology_{now}.npy')
    print(f"position: {after_linear_rheology.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_linear_rheology.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Relax to zero stress for 5 min.")
    mechtest3sensors.stay_constant_deflection(
        f'relax_to_zero_force_before_creep_{now}.raw',
        ab2xy,
        kp=0.01, ki = 0.0005,
        duration=300, moveback = False, delay = 1,
        state0 = force_free
        )
    before_creep = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    before_creep.save(f'before_creep_{now}.npy')
    print(f"position: {before_creep.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {before_creep.deflection - force_free.deflection}")

    now = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{now}: Apply the constant stress"'")
    h = (before_creep.head_to_ground - touching_state.head_to_ground)[1]
    defl = args.stress*module*h # calculated from the linear rheology
    print(f'The applied deflection will be {defl:0.3f} um')
    outname = f'add_constant_deflectionX{defl:0.3f}_stay_constant_positiony_{now}.raw'
    mechtest3sensors.add_constant_deflectionX_stay_constant_positiony(
        outname,
        ab2xy,
        kp=[0.2,0.1], ki=[0.001,0.001], kd =[0.0,0.0],
        dX=defl,
        moveback= True, state0 = before_creep, maxYdispl = 300
    )
    after_creep = measure_state()
    now = datetime.now().strftime('%Y%m%d_%H%M')
    after_creep.save(f'after_creep_{now}.npy')
    print(f"position: {after_creep.head_to_ground - touching_state.head_to_ground}")
    print(f"deflection wrt force free: {after_creep.deflection - force_free.deflection}")
