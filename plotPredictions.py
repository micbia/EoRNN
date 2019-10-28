import numpy as np, matplotlib.pyplot as plt, os 
from sys import argv

script, path = argv
os.chdir(path)

pred = np.loadtxt('predictions.txt')
true = np.loadtxt('expected_outputs.txt')


def getSmartBin(arrData, nr_BINS):
    num_part = int(len(arrData)/nr_BINS)
    bins = np.array([arrData[0]])
    if nr_BINS != 1:
        for i in range(1, nr_BINS):
            bins = np.append(bins, arrData[i*num_part-1])
    else:
        pass
    bins = np.append(bins, arrData[-1])
    print('number of density bins: %d' %nr_BINS)
    print('Approximate number of particle per density bin: %d\n' %num_part)
    return bins


def EllipseContours(x, y, nrc=1, col='green', lw=3):
    from matplotlib.patches import Ellipse
    style=[':', '--', '-']
    cov = np.cov(x,y)
    evals, evects = np.linalg.eig(cov)
    theta = np.rad2deg(np.dot([1,0], evects[:,0]))
    print(theta)
    center = (np.mean(x), np.mean(y))
    if(nrc==1):
        ellipse = Ellipse(xy=center, width=2*np.sqrt(1*evals[0]), height=2*np.sqrt(1*evals[1]), angle=theta, linestyle=style[0], color=col, linewidth=lw)
        ellipse.set_facecolor('none') 
        ax.add_patch(ellipse) 
    else:
        for k in range(nrc):
            ellipse = Ellipse(xy=center, width=2*np.sqrt((k+1)*evals[0]), height=2*np.sqrt((k+1)*evals[1]), angle=theta, linestyle=style[k], color=col, linewidth=lw) 
            ellipse.set_facecolor('none') 
            ax.add_patch(ellipse) 


print('size of data:\t%i' %true.size)
plt.figure(figsize=(10,8))
ax = plt.gca()
#ax.plot(true, pred, 'x', zorder=1)
ax.loglog(true, pred, 'x', zorder=1)
EllipseContours(true, pred, nrc=2, col='red')
ax.plot(np.linspace(true.min(),true.max(),3), np.linspace(true.min(),true.max(),3), 'k--')
ax.grid()
plt.savefig('predictions.png', bbox_inches='tight') 
plt.show()
