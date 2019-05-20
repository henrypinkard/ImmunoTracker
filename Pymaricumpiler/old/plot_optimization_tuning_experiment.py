import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path ='/Users/henrypinkard/Desktop/optimize_logs/'

frame0_data = {}
frame7_data = {}

for filename in os.listdir(path):
    if filename.startswith('.'):
        continue
    if filename.startswith('frame_0'):
        frame0_data[filename] = np.genfromtxt(path + filename, delimiter=',')
    else:
        frame7_data[filename] = np.genfromtxt(path + filename, delimiter=',')

def plotit(data, title, index, prefilter):
    #0 stitch loss    1 stack loss    2 stich rms    3 stack rms
    legend = []
    for key in data.keys():
        if 'prefilter_{}'.format(prefilter) not in key:
            continue
        if ('0.1' not in key) and ('0.01' not in key):
            continue
        plt.plot(np.arange(300), data[key][:300, index])
        legend.append(key.split('_')[4])
    plt.title(title + ' Prefilter {}'.format(prefilter))
    plt.legend(legend)



plt.figure()
plt.subplot(221)
plotit(frame0_data, 'stitch loss', 0, prefilter=True)
plt.subplot(222)
plotit(frame0_data, 'stack loss', 1, prefilter=True)
plt.subplot(223)
plotit(frame0_data, 'stitch rms', 2, prefilter=True)
plt.subplot(224)
plotit(frame0_data, 'stack rms', 3, prefilter=True)

plt.figure()
plt.subplot(221)
plotit(frame0_data, 'stitch loss', 0, prefilter=False)
plt.subplot(222)
plotit(frame0_data, 'stack loss', 1, prefilter=False)
plt.subplot(223)
plotit(frame0_data, 'stitch rms', 2, prefilter=False)
plt.subplot(224)
plotit(frame0_data, 'stack rms', 3, prefilter=False)

plt.figure()
plt.subplot(221)
plotit(frame7_data, 'stitch loss', 0, prefilter=True)
plt.subplot(222)
plotit(frame7_data, 'stack loss', 1, prefilter=True)
plt.subplot(223)
plotit(frame7_data, 'stitch rms', 2, prefilter=True)
plt.subplot(224)
plotit(frame7_data, 'stack rms', 3, prefilter=True)

plt.figure()
plt.subplot(221)
plotit(frame7_data, 'stitch loss', 0, prefilter=False)
plt.subplot(222)
plotit(frame7_data, 'stack loss', 1, prefilter=False)
plt.subplot(223)
plotit(frame7_data, 'stitch rms', 2, prefilter=False)
plt.subplot(224)
plotit(frame7_data, 'stack rms', 3, prefilter=False)
plt.show()
pass