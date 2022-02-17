import argparse, os.path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def read_all(outnames):
    data = []
    for outname in outnames:
        #try:
        d = np.fromfile(outname)
        data.append(d.reshape((len(d)//6,6)))
        #except
    for i, d in enumerate(data[:-1]):
        t0 = 2*d[-1,0] - d[-2,0]
        data[i+1][:,0] += t0
    return np.vstack((data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display results in real time')
    parser.add_argument('calibrationfilename', type=str, help='path and name of the ab2xy calibration file. Expects a .npy.')
    parser.add_argument('--path', type= str, default='.', help = """path of the measurement files. Expects binary files containing:
    #time, x_arm/ground, y_arm/ground, X_head/arm, Y_head/arm, y_arm/ground""")
    parser.add_argument('--k', type=int, default=1, help="kth measurement")
    parser.add_argument('--freq', type=float, default=20, help='refresh frequency in frames per second. If zero of negative, do not animate.')

    args = parser.parse_args()
    ab2xy = np.load(args.calibrationfilename)
    xy2ab = np.linalg.inv(ab2xy)

    dxs = np.arange(0,10,3).astype(float)
    dxs = np.append(dxs, 0.0)
    outnames = [os.path.join(args.path, 'shear%d_%d.raw'%(args.k,dx)) for dx in dxs]
    outnames[-1] = os.path.join(args.path, 'moveback_original_%d.raw'%args.k)



    t, x,y,X,Y,y_ag = read_all(outnames).T
    #prepare figure
    mosaic = """
    ABE
    CDE
    """
    fig, axs = plt.subplot_mosaic(mosaic, sharex=True, figsize=(12,6))
    xhg, = axs['A'].plot(t, (x-x[0])+(X-X[0]))
    axs['A'].set_ylabel('x_head/ground (µm)')
    yhg, = axs['C'].plot(t, y_ag-y_ag[0] + (Y-Y[0]))
    axs['C'].set_ylabel('y_head/ground (µm)')
    xha, = axs['B'].plot(t, X-X[0])
    axs['B'].set_ylabel('x_head/arm (µm)')
    yha, = axs['D'].plot(t, Y-Y[0])
    axs['D'].set_ylabel('y_head/arm (µm)')
    a,b = np.matmul(xy2ab, [X, Y])
    lines = [
        axs['E'].plot(t, m, label=f'sensor {i}')[0]
        for i,m in enumerate([a,b,y_ag])
        ]
    for id in 'CDE':
        axs[id].set_xlabel('t (s)')
    axs['A'].set_title('position')
    axs['B'].set_title('deflection')
    axs['E'].set_title('sensors')
    axs['E'].set_ylim(-80,880)
    plt.tight_layout()
    axs['E'].legend()

    def animate(i):
        t, x,y,X,Y,y_ag = read_all(outnames).T
        a,b = np.matmul(xy2ab, [X, Y])
        xhg.set_ydata((x-x[0])+(X-X[0]))
        yhg.set_ydata(y_ag-y_ag[0] + (Y-Y[0]))
        xha.set_ydata(X-X[0])
        yha.set_ydata(Y-Y[0])
        for line,m in zip(lines, [a,b,y_ag]):
            line.set_ydata(m)
        for line in [xhg, yhg, xha, yha]+lines:
            line.set_xdata(t)
        axs['A'].set_xlim(axs['A'].get_xlim()[0], t[-1])
        return [xhg, yhg, xha, yha]+lines
    if args.freq>0:
        ani = animation.FuncAnimation(
            fig, animate, interval=1000/args.freq, blit=False, save_count=1, repeat=True
        )

    fig.savefig(os.path.join(args.path, f'shear{args.k:d}.pdf'))
    plt.show()
