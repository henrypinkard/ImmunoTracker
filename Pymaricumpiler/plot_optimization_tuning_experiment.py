import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path ='/Users/henrypinkard/Desktop/optimize_logs/'

frame0_data = {}
frame7_data = {}

for filename in os.listdir(path):
    if filename.startswith('frame_0'):
        frame0_data[filename] = np.genfromtxt(path + filename, delimiter=',')
    else:
        frame7_data[filename] = np.genfromtxt(path + filename, delimiter=',')

def plotit(data, title, index, maxiter=300):
    #0 stitch loss    1 stack loss    2 stich rms    3 stack rms
    for d in data.values():
        plt.plot(np.arange(maxiter), d[:maxiter, index])
    plt.title(title)
    plt.legend(data.keys())

plt.figure()
plt.subplot(221)
plotit(frame0_data, 'stitch loss', 0)
plt.subplot(222)
plotit(frame0_data, 'stack loss', 1)
plt.subplot(223)
plotit(frame0_data, 'stitch rms', 2)
plt.subplot(224)
plotit(frame0_data, 'stack rms', 3)
plt.show()


plt.figure()
plt.subplot(221)
plotit(frame7_data, 'stitch loss', 0)
plt.subplot(222)
plotit(frame7_data, 'stack loss', 1)
plt.subplot(223)
plotit(frame7_data, 'stitch rms' , 2)
plt.subplot(224)
plotit(frame7_data, 'stack rms', 3)
plt.show()